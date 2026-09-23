package agentrunner

import (
	"encoding/json/v2"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

// partialQueueSize bounds the partial deltas buffered between the model stream and stdout.
const partialQueueSize = 1024

// partialShutdownGrace bounds how long shutdown waits for a partial write to a consumer that stopped
// reading stdout. A variable so tests can shorten it.
var partialShutdownGrace = 2 * time.Second

type partialEvent struct {
	Type   string `json:"type"`
	Kind   string `json:"kind"`
	ItemID string `json:"item_id,omitempty"`
	Delta  string `json:"delta,omitempty"`
}

// partialStream forwards ephemeral partial events to stdout for include_partial_messages.
//
// Contract:
//   - Send (the llm.PartialSink) never blocks and never performs I/O.
//   - A reset discards the preview, so enqueueing a reset also discards every partial still queued: the
//     queue always starts with the newest reset and holds deltas of that attempt only. This keeps resets
//     ordered before the deltas they precede.
//   - Only the latest request attempt is shown: partials older than the latest reset's Attempt are dropped.
//   - If deltas overflow the queue, the preview is invalidated: the queue becomes a single reset and later
//     deltas of that attempt are dropped. The model_response session item stays authoritative.
//   - Queued partials are taken in one step and written under the session observer's mutex, by the flush
//     goroutine or by the observer before a session item, so partials reported before a response
//     completes precede its model_response item. Each line is a single Write.
//   - Partials share stdout backpressure with session items and add output volume, so a consumer that
//     enables them must keep reading stdout: an unread pipe fills sooner than without partials, and a
//     session item can then wait behind a blocked partial write. If only a partial write is blocked when
//     the run is canceled, Run returns after at most partialShutdownGrace; that one in-flight write may
//     still complete afterwards, and no other partial write starts.
type partialStream struct {
	output io.Writer
	notify chan struct{}
	done   chan struct{}

	// writeMu serializes partial writes; once abandoned is set, later writes are skipped.
	writeMu   sync.Mutex
	abandoned atomic.Bool

	mu       sync.Mutex
	queue    []llm.Partial
	deltas   int
	closed   bool
	joined   bool
	attempt  uint64
	dropping bool
}

func newPartialStream(output io.Writer) *partialStream {
	return &partialStream{output: output, notify: make(chan struct{}, 1), done: make(chan struct{})}
}

// start launches the flush goroutine. flush must drain under the lock that also guards session items.
func (stream *partialStream) start(flush func()) {
	go func() {
		defer close(stream.done)
		for range stream.notify {
			flush()
		}
	}()
}

// Send is the llm.PartialSink.
func (stream *partialStream) Send(partial llm.Partial) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.closed || partial.Attempt < stream.attempt {
		return
	}
	switch {
	case partial.Kind == llm.PartialReset:
		stream.attempt, stream.dropping = partial.Attempt, false
		stream.resetLocked()
	case partial.Attempt != stream.attempt || stream.dropping:
		return
	case stream.deltas >= partialQueueSize:
		// Overflow: invalidate this attempt's preview.
		stream.dropping = true
		stream.resetLocked()
	default:
		stream.queue = append(stream.queue, partial)
		stream.deltas++
	}
	select {
	case stream.notify <- struct{}{}:
	default:
	}
}

func (stream *partialStream) resetLocked() {
	stream.queue = append(stream.queue[:0:0], llm.Partial{Kind: llm.PartialReset})
	stream.deltas = 0
}

// drain writes every queued partial. The caller holds the session observer's mutex.
func (stream *partialStream) drain() error {
	stream.mu.Lock()
	batch := stream.queue
	stream.queue, stream.deltas = nil, 0
	stream.mu.Unlock()
	for _, partial := range batch {
		if err := stream.write(partial); err != nil {
			return err
		}
	}
	return nil
}

func (stream *partialStream) write(partial llm.Partial) error {
	encoded, err := json.Marshal(partialEvent{
		Type: "partial", Kind: string(partial.Kind), ItemID: partial.ItemID, Delta: partial.Delta,
	})
	if err != nil {
		return fmt.Errorf("encode partial event: %w", err)
	}
	stream.writeMu.Lock()
	defer stream.writeMu.Unlock()
	if stream.abandoned.Load() {
		return nil
	}
	if _, err := stream.output.Write(append(encoded, '\n')); err != nil {
		return fmt.Errorf("write partial event: %w", err)
	}
	return nil
}

// Close stops accepting partials and waits for the flush goroutine. It reports false when the goroutine is
// still blocked on stdout after partialShutdownGrace; later writes are then skipped. Idempotent.
func (stream *partialStream) Close() bool {
	stream.mu.Lock()
	if !stream.closed {
		stream.closed = true
		close(stream.notify)
	}
	if stream.joined || stream.abandoned.Load() {
		joined := stream.joined
		stream.mu.Unlock()
		return joined
	}
	stream.mu.Unlock()
	select {
	case <-stream.done:
		stream.mu.Lock()
		stream.joined = true
		stream.mu.Unlock()
		return true
	case <-time.After(partialShutdownGrace):
		stream.abandoned.Store(true)
		return false
	}
}

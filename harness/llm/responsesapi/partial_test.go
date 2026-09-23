package responsesapi

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

type partialRecorder struct {
	mu       sync.Mutex
	partials []llm.Partial
	seen     chan struct{}
}

func newPartialRecorder() *partialRecorder {
	return &partialRecorder{seen: make(chan struct{}, 64)}
}

func (recorder *partialRecorder) sink(partial llm.Partial) {
	recorder.mu.Lock()
	recorder.partials = append(recorder.partials, partial)
	recorder.mu.Unlock()
	recorder.seen <- struct{}{}
}

func (recorder *partialRecorder) snapshot() []llm.Partial {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return append([]llm.Partial(nil), recorder.partials...)
}

// withoutAttempts returns partials with Attempt cleared, after checking that each reset starts a newer
// attempt and every other partial belongs to the attempt of the reset before it.
func withoutAttempts(t *testing.T, partials []llm.Partial) []llm.Partial {
	t.Helper()
	var current uint64
	out := make([]llm.Partial, len(partials))
	for i, partial := range partials {
		if partial.Kind == llm.PartialReset {
			if partial.Attempt <= current {
				t.Fatalf("reset %d has attempt %d, not newer than %d", i, partial.Attempt, current)
			}
			current = partial.Attempt
		} else if partial.Attempt != current {
			t.Fatalf("partial %d has attempt %d, want %d", i, partial.Attempt, current)
		}
		partial.Attempt = 0
		out[i] = partial
	}
	return out
}

// waitSeen waits for n sink calls, failing (not hanging) after a bound.
func waitSeen(t *testing.T, recorder *partialRecorder, n int) {
	t.Helper()
	for range n {
		select {
		case <-recorder.seen:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for partials; got %#v", recorder.snapshot())
		}
	}
}

func writeEvent(w http.ResponseWriter, data string) {
	_, _ = io.WriteString(w, "data: "+data+"\n\n")
	w.(http.Flusher).Flush()
}

func TestResponsesStreamReportsPartialsBeforeCompletion(t *testing.T) {
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseHandler := func() { releaseOnce.Do(func() { close(release) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeEvent(w, `{"type":"response.reasoning_summary_text.delta","item_id":"rs-1","delta":"Check"}`)
		writeEvent(w, `{"type":"response.output_text.delta","item_id":"msg-1","delta":"Hel"}`)
		writeEvent(w, `{"type":"response.output_text.delta","item_id":"msg-1","delta":"lo"}`)
		// Hold the terminal event until the test has observed every partial.
		<-release
		writeEvent(w, `{"type":"response.completed","response":`+completedResponse+`}`)
	}))
	defer server.Close()
	// Runs before server.Close, so a failing assertion cannot leave the handler blocked.
	defer releaseHandler()
	adapter := newTestAdapterWithConfig(t, Config{Endpoint: server.URL})
	recorder := newPartialRecorder()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	type result struct {
		response llm.Response
		err      error
	}
	done := make(chan result, 1)
	go func() {
		response, err := adapter.Respond(llm.WithPartialSink(ctx, recorder.sink), validRequest(), llm.RequestOptions{})
		done <- result{response, err}
	}()
	waitSeen(t, recorder, 4)
	select {
	case <-done:
		t.Fatal("response completed before the terminal event was sent")
	default:
	}
	releaseHandler()
	var got result
	select {
	case got = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("response did not complete after the terminal event")
	}
	if got.err != nil {
		t.Fatal(got.err)
	}
	want := []llm.Partial{
		{Kind: llm.PartialReset},
		{Kind: llm.PartialReasoning, ItemID: "rs-1", Delta: "Check"},
		{Kind: llm.PartialText, ItemID: "msg-1", Delta: "Hel"},
		{Kind: llm.PartialText, ItemID: "msg-1", Delta: "lo"},
	}
	if partials := withoutAttempts(t, recorder.snapshot()); !reflect.DeepEqual(partials, want) {
		t.Fatalf("partials = %#v, want %#v", partials, want)
	}
	terminal, err := decodeResponse([]byte(completedResponse))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.response, terminal) {
		t.Fatal("partials changed the terminal response")
	}
}

func TestResponsesStreamRetryResetsPartials(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		if attempts.Add(1) == 1 {
			// Truncated stream: the adapter retries after an unexpected EOF.
			w.Header().Set("Content-Length", "100000")
			writeEvent(w, `{"type":"response.output_text.delta","item_id":"old","delta":"stale"}`)
			return
		}
		writeEvent(w, `{"type":"response.output_text.delta","item_id":"new","delta":"fresh"}`)
		writeEvent(w, `{"type":"response.completed","response":{"id":"new","status":"completed","output":[]}}`)
	}))
	defer server.Close()
	adapter := newTestAdapterWithConfig(t, Config{Endpoint: server.URL, MaxAttempts: new(2)})
	recorder := newPartialRecorder()
	if _, err := adapter.Respond(llm.WithPartialSink(t.Context(), recorder.sink), validRequest(), llm.RequestOptions{}); err != nil {
		t.Fatal(err)
	}
	want := []llm.Partial{
		{Kind: llm.PartialReset},
		{Kind: llm.PartialText, ItemID: "old", Delta: "stale"},
		{Kind: llm.PartialReset},
		{Kind: llm.PartialText, ItemID: "new", Delta: "fresh"},
	}
	if got := withoutAttempts(t, recorder.snapshot()); !reflect.DeepEqual(got, want) {
		t.Fatalf("partials = %#v, want %#v", got, want)
	}
}

func TestResponsesStreamStopsPartialsAfterCancellation(t *testing.T) {
	canceled := make(chan struct{})
	var cancelOnce sync.Once
	releaseHandler := func() { cancelOnce.Do(func() { close(canceled) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeEvent(w, `{"type":"response.output_text.delta","item_id":"msg-1","delta":"before"}`)
		<-canceled
		for range 20 {
			writeEvent(w, `{"type":"response.output_text.delta","item_id":"msg-1","delta":"after"}`)
		}
	}))
	defer server.Close()
	defer releaseHandler()
	adapter := newTestAdapterWithConfig(t, Config{Endpoint: server.URL, MaxAttempts: new(1)})
	recorder := newPartialRecorder()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := adapter.Respond(llm.WithPartialSink(ctx, recorder.sink), validRequest(), llm.RequestOptions{})
		done <- err
	}()
	waitSeen(t, recorder, 2) // reset, "before"
	cancel()
	releaseHandler()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled request did not return")
	}
	for _, partial := range recorder.snapshot() {
		if partial.Delta == "after" {
			t.Fatal("partial reported after the request was canceled")
		}
	}
}

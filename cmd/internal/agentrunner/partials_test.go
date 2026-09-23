package agentrunner

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

// runWithPartials runs the runner with a model that streams two text partials before responding.
// includePartials is the raw JSON value for include_partial_messages ("" omits the field).
func runWithPartials(t *testing.T, includePartials string) (stdout string, logs string, sawSink bool) {
	t.Helper()
	var mu sync.Mutex
	client := &fakeClient{
		respond: func(ctx context.Context, _ llm.Request) (llm.Response, error) {
			if sink := llm.PartialSinkFrom(ctx); sink != nil {
				mu.Lock()
				sawSink = true
				mu.Unlock()
				sink(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
				sink(llm.Partial{Kind: llm.PartialText, ItemID: "msg-1", Delta: "do", Attempt: 1})
				sink(llm.Partial{Kind: llm.PartialText, ItemID: "msg-1", Delta: "ne", Attempt: 1})
			}
			return llm.Response{
				ID: "response-1", Stop: llm.StopComplete,
				Output: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleAssistant, Text: "done"}}},
			}, nil
		},
	}
	request := `{"prompt":"hi"`
	if includePartials != "" {
		request += `,"include_partial_messages":` + includePartials
	}
	request += `}`
	logDirectory := t.TempDir()
	sessionDirectory := t.TempDir()
	var out, stderr bytes.Buffer
	code := RunMain(
		t.Context(),
		[]string{"-workspace", t.TempDir(), "-session-directory", sessionDirectory, "-log-directory", logDirectory},
		func(name string) string {
			switch name {
			case "OPENAI_API_KEY":
				return "secret"
			case "SHELL":
				return "/bin/sh"
			default:
				return ""
			}
		},
		func() []string { return []string{"PATH=/usr/bin:/bin"} },
		strings.NewReader(request),
		&out,
		&stderr,
		testConfig(client),
	)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q, stdout = %q", code, stderr.String(), out.String())
	}
	// Everything persisted: the datetime JSONL log and the session store.
	var persisted strings.Builder
	for _, root := range []string{logDirectory, sessionDirectory} {
		err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return err
			}
			data, err := os.ReadFile(path)
			persisted.Write(data)
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return out.String(), persisted.String(), sawSink
}

func TestIncludePartialMessagesWritesPartialsToStdoutOnly(t *testing.T) {
	stdout, logs, sawSink := runWithPartials(t, "true")
	if !sawSink {
		t.Fatal("model request had no partial sink")
	}
	partials := strings.Index(stdout, `{"type":"partial","kind":"text","item_id":"msg-1","delta":"do"}`)
	second := strings.Index(stdout, `{"type":"partial","kind":"text","item_id":"msg-1","delta":"ne"}`)
	response := strings.Index(stdout, `"Kind":"model_response"`)
	if partials < 0 || second < 0 || response < 0 {
		t.Fatalf("stdout missing partials or model_response:\n%s", stdout)
	}
	if !(partials < second && second < response) {
		t.Fatalf("partials must precede their model_response in order:\n%s", stdout)
	}
	if strings.Contains(logs, "partial") {
		t.Fatalf("partials leaked into the session log or session store:\n%s", logs)
	}
	if !strings.Contains(logs, `"Kind":"model_response"`) {
		t.Fatalf("session log missing model_response:\n%s", logs)
	}
}

func TestPartialMessagesAreOptIn(t *testing.T) {
	for _, value := range []string{"", "false"} {
		stdout, _, sawSink := runWithPartials(t, value)
		if sawSink || strings.Contains(stdout, `"type":"partial"`) {
			t.Fatalf("include_partial_messages=%q produced partials:\n%s", value, stdout)
		}
	}
}

const resetLine = `{"type":"partial","kind":"reset"}`

func lines(out *bytes.Buffer) []string {
	return strings.Split(strings.TrimSpace(out.String()), "\n")
}

func TestPartialStreamInvalidatesPreviewWhenQueueOverflows(t *testing.T) {
	var out bytes.Buffer
	stream := newPartialStream(&out)
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
	for range partialQueueSize + 1 {
		stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "m", Delta: "x", Attempt: 1})
	}
	// Stdout recovers while the provider is silent: the consumer gets exactly an invalidating reset.
	if err := stream.drain(); err != nil {
		t.Fatal(err)
	}
	if got := lines(&out); len(got) != 1 || got[0] != resetLine {
		t.Fatalf("after overflow got %d lines, want only a reset: %q", len(got), got)
	}
	out.Reset()
	stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "m", Delta: "gap", Attempt: 1})
	if err := stream.drain(); err != nil {
		t.Fatal(err)
	}
	if out.Len() != 0 {
		t.Fatalf("delta of an invalidated attempt was written: %s", out.String())
	}
}

func TestPartialStreamReplacementAfterOverflowIsOrderedAfterItsReset(t *testing.T) {
	var out bytes.Buffer
	stream := newPartialStream(&out)
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
	for range partialQueueSize + 1 {
		stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "m", Delta: "x", Attempt: 1})
	}
	// A new attempt starts before anything was drained.
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 2})
	stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "n", Delta: "fresh", Attempt: 2})
	if err := stream.drain(); err != nil {
		t.Fatal(err)
	}
	want := []string{resetLine, `{"type":"partial","kind":"text","item_id":"n","delta":"fresh"}`}
	if got := lines(&out); !slices.Equal(got, want) {
		t.Fatalf("output = %q, want %q", got, want)
	}
}

func TestPartialStreamDropsSupersededAttempts(t *testing.T) {
	var out bytes.Buffer
	stream := newPartialStream(&out)
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
	stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "old", Delta: "valid", Attempt: 1})
	if err := stream.drain(); err != nil {
		t.Fatal(err)
	}
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 2})
	// Late arrivals from the canceled attempt 1: neither its delta nor a stale reset may touch attempt 2.
	stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "old", Delta: "stale", Attempt: 1})
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
	stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "new", Delta: "fresh", Attempt: 2})
	if err := stream.drain(); err != nil {
		t.Fatal(err)
	}
	want := []string{
		resetLine,
		`{"type":"partial","kind":"text","item_id":"old","delta":"valid"}`,
		resetLine,
		`{"type":"partial","kind":"text","item_id":"new","delta":"fresh"}`,
	}
	if got := lines(&out); !slices.Equal(got, want) {
		t.Fatalf("output = %q, want %q", got, want)
	}
}

func TestObserverShutdownFlushesQueuedPartialsThenIgnoresSends(t *testing.T) {
	var out bytes.Buffer
	observer := &sessionObserver{output: &out, cancel: func() {}, partials: newPartialStream(&out)}
	stream := observer.partials
	// The flush goroutine does nothing here: shutdown itself must deliver what is queued (an overflow reset).
	stream.start(func() {})
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
	for range partialQueueSize + 1 {
		stream.Send(llm.Partial{Kind: llm.PartialText, ItemID: "m", Delta: "x", Attempt: 1})
	}
	observer.closePartials()
	if got := lines(&out); len(got) != 1 || got[0] != resetLine {
		t.Fatalf("shutdown output = %q, want a single reset", got)
	}
	stream.Send(llm.Partial{Kind: llm.PartialReset, Attempt: 2})
	observer.closePartials()
	if got := lines(&out); len(got) != 1 {
		t.Fatalf("wrote after shutdown: %q", got)
	}
}

// stallDetector wraps the pipe writer and reports when a Write has been blocked for a while, which proves
// the pipe is full rather than assuming it.
type stallDetector struct {
	file    *os.File
	writing atomic.Int64 // UnixNano when the in-progress Write began, 0 when idle
}

func (detector *stallDetector) Write(p []byte) (int, error) {
	detector.writing.Store(time.Now().UnixNano())
	defer detector.writing.Store(0)
	return detector.file.Write(p)
}

func (detector *stallDetector) blocked(threshold time.Duration) bool {
	began := detector.writing.Load()
	return began != 0 && time.Since(time.Unix(0, began)) > threshold
}

// runStalled runs the runner with stdout on a real OS pipe that nobody reads while the model keeps
// streaming. Without partials the runner is canceled once the model request has started; with partials,
// once a partial write is provably blocked on the full pipe. It reports how long RunMain took to return.
func runStalled(t *testing.T, partials bool) time.Duration {
	t.Helper()
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	defer writer.Close()
	output := &stallDetector{file: writer}
	started := make(chan struct{})
	client := &fakeClient{
		respond: func(requestContext context.Context, _ llm.Request) (llm.Response, error) {
			close(started)
			sink := llm.PartialSinkFrom(requestContext)
			for i := 0; requestContext.Err() == nil; i++ {
				if sink != nil {
					if i == 0 {
						sink(llm.Partial{Kind: llm.PartialReset, Attempt: 1})
					}
					sink(llm.Partial{Kind: llm.PartialText, ItemID: "m", Delta: strings.Repeat("x", 512), Attempt: 1})
				}
				time.Sleep(time.Millisecond)
			}
			return llm.Response{}, requestContext.Err()
		},
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	request := `{"prompt":"hi"}`
	if partials {
		request = `{"prompt":"hi","include_partial_messages":true}`
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		var stderr bytes.Buffer
		RunMain(ctx, []string{"-workspace", t.TempDir(), "-session-directory", t.TempDir(), "-log-directory", t.TempDir()},
			func(name string) string {
				if name == "OPENAI_API_KEY" {
					return "secret"
				}
				return ""
			},
			func() []string { return nil }, strings.NewReader(request), output, &stderr, testConfig(client))
	}()
	select {
	case <-started:
	case <-done:
		t.Fatal("runner exited before the model request started")
	case <-time.After(10 * time.Second):
		t.Fatal("model request never started")
	}
	if partials {
		deadline := time.Now().Add(10 * time.Second)
		for !output.blocked(100 * time.Millisecond) {
			if time.Now().After(deadline) {
				t.Fatal("stdout never filled up; the test precondition was not reached")
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	begin := time.Now()
	cancel()
	select {
	case <-done:
		return time.Since(begin)
	case <-time.After(10 * time.Second):
		t.Fatalf("partials=%v: runner did not return after cancellation with stdout unread", partials)
		return 0
	}
}

func TestCancellationWithUnreadStdoutReturnsLikeUpstream(t *testing.T) {
	previous := partialShutdownGrace
	partialShutdownGrace = 200 * time.Millisecond
	defer func() { partialShutdownGrace = previous }()
	if elapsed := runStalled(t, false); elapsed > 2*time.Second {
		t.Fatalf("baseline without partials took %v", elapsed)
	}
	if elapsed := runStalled(t, true); elapsed > 2*time.Second {
		t.Fatalf("with partials, shutdown took %v; want within the grace period", elapsed)
	}
}

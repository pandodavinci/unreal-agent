package agentrunner

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/contextbuilder"
	"github.com/unreallabsai/unreal-agent/harness/inbox"
	"github.com/unreallabsai/unreal-agent/harness/llm"
)

const (
	streamFirstID = "a1b2c3d4-0000-4000-8000-000000000001"
	streamSteerID = "a1b2c3d4-0000-4000-8000-000000000002"
	streamLastID  = "a1b2c3d4-0000-4000-8000-000000000003"
)

// gatedCommand starts, marks itself started, and runs until the test creates the release file.
const gatedCommand = `touch started; while [ ! -f release ]; do sleep 0.02; done; printf released`

type steeredRun struct {
	stdout   string
	stderr   string
	code     int
	requests []llm.Request
	// steeredWhileRunning is true when the model saw the release message while the Bash call was running.
	steeredWhileRunning bool
}

// runSteered starts a run whose first model turn launches gatedCommand. Once the command is running,
// it writes lines to stdin. The command is released only when a model request carries releaseText
// while the command is still running, so without live input the run times out instead of finishing.
func runSteered(t *testing.T, positional bool, lines []string, releaseText string, closeStdin bool) steeredRun {
	t.Helper()
	workspace := t.TempDir()
	release := func() {
		if err := os.WriteFile(filepath.Join(workspace, "release"), nil, 0o600); err != nil {
			t.Error(err)
		}
	}
	var result steeredRun
	client := &fakeClient{}
	client.respond = func(_ context.Context, request llm.Request) (llm.Response, error) {
		client.mu.Lock()
		defer client.mu.Unlock()
		client.calls++
		result.requests = append(result.requests, request)
		if client.calls == 1 {
			return llm.Response{ID: "response-1", Stop: llm.StopComplete, Output: []llm.Item{{
				Type: llm.ItemToolCall,
				Data: llm.ToolCall{CallID: "call-1", Name: "Bash", Arguments: fmt.Sprintf(`{"command":%q}`, gatedCommand)},
			}}}, nil
		}
		status := bashStatus(request)
		if status == "released" {
			return textResponse("finished"), nil
		}
		if status == "running" && slices.Contains(userTexts(request), releaseText) {
			result.steeredWhileRunning = true
			release()
			return textResponse("noted"), nil
		}
		return llm.Response{}, fmt.Errorf("unexpected request: bash %q, user messages %q", status, userTexts(request))
	}

	request := fmt.Sprintf(`{"messages":[{"content":"start","message_id":%q}],"stream_input":true}`, streamFirstID)
	stdinReader, stdinWriter := io.Pipe()
	defer stdinWriter.Close()
	args := []string{"-workspace", workspace, "-session-directory", t.TempDir()}
	if positional {
		args = append(args, request)
	}
	go func() {
		if !positional {
			if _, err := io.WriteString(stdinWriter, request+"\n"); err != nil {
				return
			}
		}
		for {
			if _, err := os.Stat(filepath.Join(workspace, "started")); err == nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		for _, line := range lines {
			if _, err := io.WriteString(stdinWriter, line+"\n"); err != nil {
				return
			}
		}
		if closeStdin {
			stdinWriter.Close()
			release()
		}
	}()

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	defer release()
	var stdout, stderr bytes.Buffer
	result.code = RunMain(ctx, args, testEnvironment, func() []string { return []string{"PATH=/usr/bin:/bin"} },
		stdinReader, &stdout, &stderr, testConfig(client))
	result.stdout, result.stderr = stdout.String(), stderr.String()
	return result
}

func testEnvironment(name string) string {
	switch name {
	case "OPENAI_API_KEY":
		return "secret"
	case "SHELL":
		return "/bin/sh"
	}
	return ""
}

func textResponse(text string) llm.Response {
	return llm.Response{ID: "response-" + text, Stop: llm.StopComplete, Output: []llm.Item{{
		Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleAssistant, Text: text},
	}}}
}

func streamLine(content, id string) string {
	return fmt.Sprintf(`{"role":"user","content":%q,"message_id":%q}`, content, id)
}

// bashStatus reports "running", the final output of call-1, or "" when the model has not seen it.
func bashStatus(request llm.Request) string {
	status := ""
	for _, item := range request.Input {
		if item.Type != llm.ItemToolResult {
			continue
		}
		result := item.Data.(llm.ToolResult)
		if result.CallID != "call-1" {
			continue
		}
		status = result.Output[0].Value
		if status == contextbuilder.ToolCallRunningPayload {
			status = "running"
		}
	}
	return status
}

func userTexts(request llm.Request) []string {
	var texts []string
	for _, item := range request.Input {
		if item.Type == llm.ItemMessage && item.Data.(llm.Message).Role == llm.RoleUser {
			texts = append(texts, item.Data.(llm.Message).Text)
		}
	}
	return texts
}

func inputErrorLines(t *testing.T, output string) []int {
	t.Helper()
	decoder := jsontext.NewDecoder(strings.NewReader(output))
	var lines []int
	for {
		var event inputErrorEvent
		if err := json.UnmarshalDecode(decoder, &event); err != nil {
			if errors.Is(err, io.EOF) {
				return lines
			}
			t.Fatal(err)
		}
		if event.Type == "input_error" {
			if event.Message == "" {
				t.Fatalf("input_error on line %d has no message", event.Line)
			}
			lines = append(lines, event.Line)
		}
	}
}

func assertFinished(t *testing.T, run steeredRun) {
	t.Helper()
	if run.code != 0 {
		t.Fatalf("exit = %d, stderr = %q, stdout = %q", run.code, run.stderr, run.stdout)
	}
	if !run.steeredWhileRunning {
		t.Fatal("the streamed message did not reach the model while the command was running")
	}
	final := run.requests[len(run.requests)-1]
	if bashStatus(final) != "released" {
		t.Fatalf("last model request saw bash %q, want the finished command output", bashStatus(final))
	}
}

func TestStreamInputReachesModelWhileCommandRuns(t *testing.T) {
	for _, test := range []struct {
		name       string
		positional bool
	}{{"request on stdin", false}, {"positional request", true}} {
		t.Run(test.name, func(t *testing.T) {
			run := runSteered(t, test.positional, []string{streamLine("steer", streamSteerID)}, "steer", false)
			assertFinished(t, run)
			if ids := inputIDs(t, run.stdout); !slices.Equal(ids, []inbox.ID{streamFirstID, streamSteerID}) {
				t.Fatalf("external input IDs = %q", ids)
			}
		})
	}
}

func TestStreamInputIgnoresDuplicateMessageIDs(t *testing.T) {
	run := runSteered(t, false, []string{
		streamLine("resent start", streamFirstID),
		streamLine("steer", streamSteerID),
		streamLine("resent steer", streamSteerID),
		streamLine("last", streamLastID),
	}, "last", false)
	assertFinished(t, run)
	if ids := inputIDs(t, run.stdout); !slices.Equal(ids, []inbox.ID{streamFirstID, streamSteerID, streamLastID}) {
		t.Fatalf("external input IDs = %q", ids)
	}
	for _, request := range run.requests {
		for _, text := range userTexts(request) {
			if strings.HasPrefix(text, "resent") {
				t.Fatalf("model saw duplicate message %q", text)
			}
		}
	}
}

func TestStreamInputReportsBadLinesAndKeepsRunning(t *testing.T) {
	run := runSteered(t, false, []string{
		`not json`,
		`{"role":"assistant","content":"x"}`,
		`{"content":"x","message_id":"not-a-uuid"}`,
		`{"content":"x","unknown":true}`,
		``,
		streamLine("after bad lines", streamSteerID),
	}, "after bad lines", false)
	assertFinished(t, run)
	// The request is stdin line 1, so the bad lines are 2 to 5 and the blank line 6 is skipped.
	if lines := inputErrorLines(t, run.stdout); !slices.Equal(lines, []int{2, 3, 4, 5}) {
		t.Fatalf("input_error lines = %v, stdout = %s", lines, run.stdout)
	}
	var items strings.Builder
	for line := range strings.Lines(run.stdout) {
		if !strings.Contains(line, `"type":"input_error"`) {
			items.WriteString(line)
		}
	}
	if ids := inputIDs(t, items.String()); !slices.Equal(ids, []inbox.ID{streamFirstID, streamSteerID}) {
		t.Fatalf("external input IDs = %q", ids)
	}
}

func TestStreamInputClosedDuringCommandFinishesWhenIdle(t *testing.T) {
	// The command is released right after stdin closes, so the model never needs a streamed message.
	run := runSteered(t, false, nil, "", true)
	if run.code != 0 {
		t.Fatalf("exit = %d, stderr = %q, stdout = %q", run.code, run.stderr, run.stdout)
	}
	if final := run.requests[len(run.requests)-1]; bashStatus(final) != "released" {
		t.Fatalf("last model request saw bash %q, want the finished command output", bashStatus(final))
	}
	if !strings.Contains(run.stdout, `"Text":"finished"`) {
		t.Fatalf("final answer missing from stdout: %s", run.stdout)
	}
}

func TestStreamInputWithStdinClosedAfterRequest(t *testing.T) {
	for _, request := range []string{
		`{"prompt":"hello","stream_input":true}`,
		`{"prompt":"hello","stream_input":true}` + "\n",
	} {
		client := &fakeClient{respond: func(_ context.Context, request llm.Request) (llm.Response, error) {
			if !slices.Equal(userTexts(request), []string{"hello"}) {
				return llm.Response{}, fmt.Errorf("user messages = %q", userTexts(request))
			}
			return textResponse("hi"), nil
		}}
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		var stdout, stderr bytes.Buffer
		code := RunMain(ctx, []string{"-workspace", t.TempDir(), "-session-directory", t.TempDir()}, testEnvironment,
			func() []string { return nil }, strings.NewReader(request), &stdout, &stderr, testConfig(client))
		cancel()
		if code != 0 || !strings.Contains(stdout.String(), `"Text":"hi"`) {
			t.Fatalf("request %q: exit = %d, stderr = %q, stdout = %q", request, code, stderr.String(), stdout.String())
		}
	}
}

func TestStreamInputRequiresSingleLineRequest(t *testing.T) {
	for _, test := range []struct{ name, input, want string }{
		{"multi-line request", "{\n\"prompt\":\"hello\",\n\"stream_input\":true\n}\n", "stream_input requires the JSON request on the first stdin line"},
		{"trailing lines without stream_input", `{"prompt":"hello"}` + "\n" + streamLine("more", streamSteerID) + "\n", "invalid JSON"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &fakeClient{respond: func(context.Context, llm.Request) (llm.Response, error) {
				return llm.Response{}, errors.New("model must not be called")
			}}
			var stdout, stderr bytes.Buffer
			code := RunMain(t.Context(), []string{"-workspace", t.TempDir(), "-session-directory", t.TempDir()}, testEnvironment,
				func() []string { return nil }, strings.NewReader(test.input), &stdout, &stderr, testConfig(client))
			if code != 1 || !strings.Contains(stderr.String(), test.want) {
				t.Fatalf("exit = %d, stderr = %q, want %q", code, stderr.String(), test.want)
			}
		})
	}
}

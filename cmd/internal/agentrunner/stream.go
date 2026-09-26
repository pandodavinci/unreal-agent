package agentrunner

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"

	"github.com/unreallabsai/unreal-agent/harness/inbox"
)

type inputErrorEvent struct {
	Type    string `json:"type"`
	Line    int    `json:"line"`
	Message string `json:"message"`
}

// splitRequestLine separates a one-line request that enables stream_input from the stdin lines after
// it. Any other input is returned whole as the request, so multi-line requests parse as before.
func splitRequestLine(input io.Reader) (io.Reader, *bufio.Reader) {
	reader := bufio.NewReader(input)
	line, err := reader.ReadBytes('\n')
	if (err == nil || errors.Is(err, io.EOF)) && requestsStreamInput(line) {
		return bytes.NewReader(line), reader
	}
	return io.MultiReader(bytes.NewReader(line), reader), nil
}

func requestsStreamInput(line []byte) bool {
	var probe struct {
		StreamInput *bool `json:"stream_input"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(line), &probe); err != nil {
		return false
	}
	return probe.StreamInput != nil && *probe.StreamInput
}

// readStreamedInput submits each JSONL user message from input to the inbox until input ends or the
// run stops. The inbox drops IDs it has already seen, so hosts may resend a message safely. Lines that
// fail validation are reported as input_error events and skipped.
func (observer *sessionObserver) readStreamedInput(ctx context.Context, input *bufio.Reader, lineNumber int, inputs inbox.Writer) {
	for ; ; lineNumber++ {
		line, readErr := input.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) != 0 {
			if err := submitStreamedMessage(ctx, inputs, line); err != nil {
				if ctx.Err() != nil {
					return
				}
				observer.reportInputError(lineNumber, err)
			}
		}
		if readErr != nil {
			if !errors.Is(readErr, io.EOF) {
				observer.reportInputError(lineNumber, fmt.Errorf("read stdin: %w", readErr))
			}
			return
		}
	}
}

func submitStreamedMessage(ctx context.Context, inputs inbox.Writer, line []byte) error {
	var message RequestMessage
	if err := json.Unmarshal(bytes.TrimSpace(line), &message, json.RejectUnknownMembers(true)); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	if err := validateMessage("message", message); err != nil {
		return err
	}
	input, err := messageInput(message)
	if err != nil {
		return fmt.Errorf("encode message: %w", err)
	}
	return inputs.Submit(ctx, input)
}

func (observer *sessionObserver) reportInputError(line int, cause error) {
	if observer.finished.Load() {
		return
	}
	observer.mu.Lock()
	defer observer.mu.Unlock()
	if observer.err != nil || observer.finished.Load() {
		return
	}
	encoded, err := json.Marshal(inputErrorEvent{Type: "input_error", Line: line, Message: cause.Error()})
	if err != nil {
		observer.fail(fmt.Errorf("encode input error: %w", err))
		return
	}
	if _, err := fmt.Fprintf(observer.stdout, "%s\n", encoded); err != nil {
		observer.fail(fmt.Errorf("write input error: %w", err))
	}
}

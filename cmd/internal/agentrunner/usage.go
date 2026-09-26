package agentrunner

import (
	"flag"
	"fmt"
)

const requestHelp = `
Request schema (JSON object; unknown fields are rejected):
  messages: array of {role: "user", content: string, message_id?: UUID string}
    Non-empty array of user messages delivered in order. role defaults to "user";
    message_id defaults to a generated UUID.
  prompt: string
    Shorthand for one user message; used when messages is absent.
    Supply messages or prompt. messages takes precedence when both are present.
  model: string (optional)
    Provider model ID; defaults to UNREAL_HARNESS_LLM_MODEL or the provider default.
  max_attempts: positive integer (optional)
    Overrides UNREAL_HARNESS_LLM_MAX_ATTEMPTS (default 5); 1 disables retries.
  system_prompt: string (optional)
    Replaces the default system prompt.
  thinking_level: "low" | "medium" | "high" | "xhigh" | "max" (optional; default "high")
  session_id: non-empty string (optional)
    Creates or resumes a persisted session.
  disallowed_tools: array of non-empty strings (optional)
    Static tool names excluded from model context and execution.
  extra_allowed_tools: array of non-empty strings (optional; accepted but ignored)
  include_partial_messages: boolean (optional; default false)
    Also write ephemeral preview lines to stdout while a model response streams:
      {"type":"partial","kind":"reset"}
      {"type":"partial","kind":"text","item_id":"msg_...","delta":"Hel"}
      {"type":"partial","kind":"reasoning","item_id":"rs_...","delta":"Checking"}
    item_id is the provider's output item ID; concatenate deltas per item_id. "reasoning" carries
    reasoning summary text only. A reset precedes every request attempt (each model turn and each
    retry) and discards the preview so far; it is also sent if previews had to be dropped because
    stdout could not keep up. The preview is retired by the model_response session item, which is
    written after the partials it supersedes and is authoritative, or by the run ending on error or
    cancellation. Partials are never written to the session store or the JSONL log. They add output
    volume, so keep reading stdout: an unread pipe fills sooner than without partials.
  stream_input: boolean (optional; default false)
    Keep reading user messages from stdin while the agent works, one JSON object per line:
      {"role":"user","content":"...","message_id":"<UUID>"}
    role and message_id are optional as in messages; set message_id so resending is safe.
    Each message reaches the model at its next turn, even while tools are still running.
    With the request on stdin, it must be the first line. With a positional request, all of
    stdin is messages. The run still ends when the agent is idle; closing stdin changes
    nothing. A message counts as delivered once its input session item is written to stdout.
    Messages still unacknowledged when the run ends are dropped, so resend them in the next
    run with the same message_id: seen IDs are ignored, also across resumed sessions. An
    invalid line is skipped and reported without stopping the run:
      {"type":"input_error","line":3,"message":"..."}
    line counts stdin lines from 1. input_error events go to stdout only, not the JSONL log.
`

func writeUsage(flags *flag.FlagSet) error {
	if _, err := fmt.Fprintf(flags.Output(), `Usage:
  %[1]s [options] < request.json
  %[1]s [options] 'JSON request'
  %[1]s [options] -p 'prompt'

Reads one JSON request from stdin unless a positional request or -p is supplied.
With stream_input, stdin then keeps supplying user messages (see below).
Place options before the positional request. -p and a positional request are mutually exclusive.

Options:
`, flags.Name()); err != nil {
		return fmt.Errorf("write usage: %w", err)
	}
	flags.PrintDefaults()
	if _, err := fmt.Fprint(flags.Output(), requestHelp); err != nil {
		return fmt.Errorf("write request schema: %w", err)
	}
	return nil
}

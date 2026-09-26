# unreal-agent-runner

Run an AI agent from a prompt or JSON request. It writes events to stdout as
JSONL and exits when the task finishes.

Install with Go 1.27+:

```sh
go install github.com/unreallabsai/unreal-agent/cmd/unreal-agent-runner@latest
```

Set an OpenAI API key and run a prompt in the current directory:

```sh
export OPENAI_API_KEY="..."
unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

Or run from source at the repository root:

```sh
go run ./cmd/unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

Choose a workspace and save the output:

```sh
unreal-agent-runner -workspace ./my-project -p 'Summarize this project.' > run.jsonl
```

Sessions: `${XDG_STATE_HOME:-$HOME/.local/state}/unreal-agent/sessions`
(override with `-session-directory`).

You can also pass a JSON request as an argument or through stdin:

```sh
unreal-agent-runner '{"prompt":"Summarize this project."}'
unreal-agent-runner < request.json
```

To message the agent while it works, set `stream_input` and keep writing user
messages to stdin, one JSON object per line. The request must be the first line:

```sh
{
  echo '{"prompt":"Run the tests and fix what fails.","stream_input":true}'
  sleep 30
  echo '{"role":"user","content":"Skip the slow e2e suite.","message_id":"7f9c1f1e-6a1b-4a53-9a4c-2b6f5d0e8c11"}'
} | unreal-agent-runner
```

Each message reaches the model at its next turn, even while commands are still
running. The run still ends when the agent is idle. A message is delivered once
its input item appears on stdout; if the run ends first, send it again in the
next run with the same `message_id` (repeated IDs are ignored). Invalid lines
are reported as `{"type":"input_error",...}` events and skipped.

OpenAI is the default provider. Set `UNREAL_HARNESS_LLM_PROVIDER` to `openai`,
`openai-codex`, `openrouter`, `fireworks`, or `ollama`, and
`UNREAL_HARNESS_LLM_MODEL` to choose a model.

Run `unreal-agent-runner -h` for options and the JSON request fields.

## Docker

The `unrea1labs/unreal-agent` image supports Linux on AMD64 and ARM64. Run it
with a project mounted as the workspace:

```sh
docker run --rm -i --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD:/workspace" \
  -v unreal-agent-state:/state \
  unrea1labs/unreal-agent:latest -p 'Summarize this project.'
```

Each release also publishes its Git tag (for example, `v0.1.0`) for version pinning.

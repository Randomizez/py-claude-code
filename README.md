# pyccode

`pyccode` is a minimal Python extraction of repository A.

It follows the engineering strategy proven in repository b:

- do not rebuild the full product shell first
- extract the smallest useful agent and tool loop first
- keep the implementation compact, readable, and testable

Semantically, `pyccode` is guided by repository A rather than by OpenAI Codex:

- the main turn loop follows A's query pipeline
- tool registration and orchestration follow A's tool abstractions
- the model request layer follows A's Anthropic Messages path
- prompt and context injection follow A's prompt and message assembly flow

The current implementation provides:

- a minimal multi-turn agent loop
- a complete `tool_use` / `tool_result` loop
- an Anthropic Messages adapter shaped after A's non-bare request path
- a minimal CLI
- the 22 tool names and input schemas observed in local non-bare A captures, declared directly in code
- a first-turn request that is now very close to the local non-bare A capture, aside from dynamic session values and a few remaining wire-level details
- unit tests covering the runtime loop, auth resolution, request shaping, tool schemas, and doctor checks

The current implementation intentionally does not include:

- the full TUI / React / Ink UI layer
- telemetry, feature flags, remote control, or internal-only modules
- full real implementations for all 22 default A-shaped tools
- the wider product surfaces such as full memory, hooks, MCP, or advanced background orchestration

## Quick Start

Install dependencies:

```bash
cd <repo-root>
uv sync
```

Set a model and API credentials:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=...
```

Run a single prompt:

```bash
uv run pyccode "Summarize this repository in one sentence."
```

Run the interactive CLI:

```bash
uv run pyccode
```

Run environment diagnostics:

```bash
uv run pyccode doctor --skip-live
uv run pyccode doctor
```

Exit the interactive CLI:

```text
/exit
```

## Alignment Checklist

### Request / Context

- [x] `/v1/messages?beta=true`
- [x] streaming SSE request/response skeleton
- [x] Bearer / OAuth-first auth fallback chain
- [x] `system: list[block]`
- [x] the 22 tool schemas observed in the local non-bare A capture are exposed by default
- [x] skills reminder injection
- [x] `max_tokens = 32000` default
- [x] persistent `device_id` get-or-create logic
- [x] first-turn request is largely aligned with the local non-bare A capture except for dynamic `session_id` values and a few remaining details

### Tool Schemas

- [x] `Agent`
- [x] `AskUserQuestion`
- [x] `Bash`
- [x] `CronCreate`
- [x] `CronDelete`
- [x] `CronList`
- [x] `Edit`
- [x] `EnterPlanMode`
- [x] `EnterWorktree`
- [x] `ExitPlanMode`
- [x] `ExitWorktree`
- [x] `Glob`
- [x] `Grep`
- [x] `NotebookEdit`
- [x] `Read`
- [x] `Skill`
- [x] `TaskOutput`
- [x] `TaskStop`
- [x] `TodoWrite`
- [x] `WebFetch`
- [x] `WebSearch`
- [x] `Write`

### Tool Implementations

- [x] `Read`
- [x] `Write`
- [x] `Edit`
- [x] `Glob`
- [x] `Grep`
- [x] `Bash`
- [ ] `Agent`
- [ ] `AskUserQuestion`
- [ ] `CronCreate`
- [ ] `CronDelete`
- [ ] `CronList`
- [ ] `EnterPlanMode`
- [ ] `EnterWorktree`
- [ ] `ExitPlanMode`
- [ ] `ExitWorktree`
- [ ] `NotebookEdit`
- [ ] `Skill`
- [ ] `TaskOutput`
- [ ] `TaskStop`
- [ ] `TodoWrite`
- [ ] `WebFetch`
- [ ] `WebSearch`
- [x] the older local execution subset is still available via `build_local_execution_tool_registry()`

See `docs/ARCHITECTURE.md` for the current architectural mapping.

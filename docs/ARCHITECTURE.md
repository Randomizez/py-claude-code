# Architecture

The starting point of this repository is simple: A and B differ at the product layer, but their underlying agent frameworks are structurally similar. Both can be reduced to the same core loop:

1. maintain turn history
2. assemble prompt and context
3. sample assistant content and tool uses from the model
4. execute tools
5. feed tool results back into the model
6. repeat until no new tool uses remain

## Implementation Strategy

`pyccode` takes A as the semantic target, but defaults to `pycodex` for implementation strategy:

- If `pycodex` already has a validated module shape, testing pattern, or repository layout that is compatible with the relevant A behavior, `pyccode` should reuse or adapt it rather than inventing a third pattern.
- `pyccode` should diverge only where A and Codex differ materially in protocol, prompt structure, or tool semantics.
- As a result, the top-level organization of `agent.py`, `runtime.py`, `context.py`, `model.py`, `tools/`, and `tests/` is intentionally close to `pycodex`.

## Core Pipeline in A

The current minimal extraction from A maps to these source areas:

- CLI entrypoint
- main query loop
- query dependency and config boundary
- tool abstraction
- tool registry
- tool orchestration
- streaming tool execution
- model API adapter for the Anthropic Messages path
- prompt and context assembly

## Mapping to `pyccode`

- `pyccode/cli.py`
  - corresponds to the CLI entry area in A
  - keeps only the minimal runnable command surface

- `pyccode/agent.py`
  - corresponds to the central query loop in A
  - owns the assistant -> tool_use -> tool_result -> next-iteration loop

- `pyccode/tools/base_tool.py`
  - corresponds to A's tool abstraction layer
  - defines `BaseTool`, `ToolRegistry`, and `ToolContext`

- `pyccode/tools/*.py`
  - correspond to local tool implementations and tool schema exposure
  - the default registry currently exposes the 22 schemas observed in local non-bare A captures
  - real execution is still partial; the older high-value local execution registry remains available as a non-default path

- `pyccode/model.py`
  - corresponds to A's Anthropic Messages adapter
  - currently implements the A-shaped streaming `/v1/messages?beta=true` skeleton and the minimal SSE event lifecycle

- `pyccode/context.py`
  - corresponds to A's prompt and request-context assembly
  - owns base system blocks, user-facing `<system-reminder>` blocks, environment context, and `AGENTS.md` injection

- `pyccode/auth.py`
  - corresponds to A's auth resolution and local secure-storage path
  - currently implements the minimal Bearer / OAuth-first resolution chain using explicit env tokens, config-dir OAuth state, and API-key fallback

- `pyccode/runtime.py`
  - borrows the outer submission-loop pattern from B
  - wraps the inner turn loop in a small async submission queue

## Intentional Simplifications

Compared with A, the current version deliberately leaves out:

- feature-gated internal modules
- prompt compaction, context collapse, and memory prefetch
- streaming tool execution
- multi-agent, MCP, remote, and skill-discovery layers
- the full permission system

So `pyccode` is not "A in Python". It is a minimal Python kernel for the core execution framework behind A.

## Recommended Next Steps

The most sensible order for future work is:

1. expand the A-native tool surface and naming coverage
2. move prompt fragments, reminders, and mode switches closer to A
3. extend the streaming Messages path toward richer `tool_use` and `thinking` behavior
4. consider permissions, plan mode, and subagent structure only after the core request path is stable

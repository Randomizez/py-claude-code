# Alignment

This document records the current alignment work between `pyccode` and repository A.

Repository b is the engineering reference, not the semantic target. When A and `pycodex` are structurally compatible, `pyccode` should prefer the validated `pycodex` code shape, test style, and repository layout instead of inventing a third pattern.

## Scope

The current alignment pass focuses on the smallest reusable core from A:

- the outer submission and runtime loop
- the inner turn loop (`model -> tool_use -> tool_result -> follow-up`)
- Anthropic Messages request and context shape
- the default tool registry and a minimal high-value local execution subset
- repository structure and test layout

This document does not claim full product parity with A.

Out of scope for now:

- TUI / React / Ink UI
- telemetry, remote-managed settings, and GrowthBook
- compaction, context collapse, and memory prefetch
- permissions, plan mode, subagents, and MCP
- feature-gated internal modules that are not relevant to the minimal public core

## Comparison Method

The current comparison method combines source mapping with behavior checks.

1. Use A as the semantic baseline.
   - query loop behavior
   - tool registration and orchestration
   - Anthropic request shaping
   - prompt and context assembly
2. Reuse `pycodex` whenever the underlying framework pattern is already validated.
   - file and module layout
   - runtime queue shape
   - small-tool organization
   - test style and fixture style
3. Lock in behavior with local tests under `tests/`.
4. Distinguish clearly between:
   - `implemented`: behavior exists in `pyccode`
   - `derived`: behavior is inferred from A source but not yet live-captured
   - `captured`: behavior is verified by a real local request capture

## Current Result

At the current snapshot, `pyccode` is aligned with A at the minimal framework level, not at the full product level.

Current wins:

- repository structure intentionally mirrors `pycodex`
- the outer runtime loop exists and is tested
- the inner agent loop exists and is tested
- tool batching distinguishes serial and parallel-safe execution at a minimal level
- the Anthropic Messages client exists for the A-shaped request family
- the default visible tool surface now exposes the 22 tool names and schemas seen in local non-bare A captures
- a smaller local execution registry still exists for real file and shell work

## Local Capture Strategy

The current local comparison artifacts come from three request paths:

- `pyccode`
- A bare mode (`--bare -p`)
- A non-bare mode with sandboxed fake OAuth state

These captures do not contact real upstream services. They only show what each client tries to send to a local fake endpoint.

## Latest Capture Summary

```python
LatestCapturedFacts = {
    "target": "a_nonbare",
    "pyccode": {
        "path": "/v1/messages?beta=true",
        "stream": True,
        "system_type": "list[block]",
        "tool_count": 22,
        "auth": "bearer_or_api_key_fallback",
    },
    "a_bare": {
        "path": "/v1/messages?beta=true",
        "stream": True,
        "system_type": "list[block]",
        "tool_count": 3,
        "auth": "api_key",
    },
    "a_nonbare": {
        "path": "/v1/messages?beta=true",
        "stream": True,
        "system_type": "list[block]",
        "tool_count": 22,
        "auth": "bearer",
        "max_tokens": 32000,
    },
}
```

The key takeaway is not just that `pyccode` is closer to A than before. The important conclusion is that the non-bare A capture is the real target, while the bare capture is only a smaller validation slice.

## Why the Bare Capture Still Matters

The bare-mode request is still useful because it gives a smaller, easier-to-debug slice of the same request family:

- same endpoint family
- same streaming transport shape
- smaller tool surface
- less product-side context shaping

That makes it a good intermediate checkpoint, but not the final target.

## Current Gaps Against A

The main remaining gaps are:

- some dynamic session fields still differ by construction
- A's request builder still has richer context layering than `pyccode`
- A performs more normalization on outbound `messages`
- the default tool set in `pyccode` is schema-aligned first, execution-aligned second
- several top-level request features are still partial or simplified in `pyccode`

## Repository-Shape Policy

The guiding policy remains:

```python
AlignmentPolicy = {
    "semantic_target": "A",
    "engineering_baseline": "prefer pycodex layout unless A-specific semantics require divergence",
    "request_target": "local non-bare A capture",
    "tool_surface_target": "22 tool names and schemas seen in local non-bare A captures",
    "avoid": "schema snapshot files when code-local definitions are sufficient",
}
```

## Practical Reading Order

If you need to understand the current state quickly, read the docs in this order:

1. `README.md` for scope and current status
2. `docs/ARCHITECTURE.md` for module mapping
3. `docs/CONTEXT.md` for request-shape details
4. `docs/MESSAGES_SERVER.md` for server-side behavior expectations


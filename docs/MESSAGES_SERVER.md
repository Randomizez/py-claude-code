# Messages Server

This document records current server-side behavior observations for the Anthropic Messages path relevant to `pyccode`.

Goals:

1. record what the clients currently expect from a Messages-compatible server
2. record what the local fake server currently emulates and what it does not

This document is intentionally narrower than `docs/ALIGNMENT.md` and `docs/CONTEXT.md`:

- `ALIGNMENT.md` focuses on A vs `pyccode`
- `CONTEXT.md` focuses on request and context shape
- `MESSAGES_SERVER.md` focuses on the server side of that exchange

## Scope

Current observations come from three sources:

- source reading in repository A
- local fake-server captures produced by the debug workflow
- the current fake implementation in `tests/fake_anthropic_messages_server.py`

These notes do not claim full Anthropic API correctness. They describe the minimum behavior needed for the current alignment and debugging work.

## Current Local Artifacts

The latest local captures include:

- basic request and response captures for `pyccode`
- matching captures for A bare mode
- a non-bare capture backed by sandboxed fake OAuth state
- a joined three-way diff summary

These are all local-only captures. They do not contact real upstream services.

## Endpoint Observations

### `pyccode`

Current `pyccode` uses the same streaming endpoint family as A:

```python
PyccodeObservedRequest = {
    "path": "/v1/messages?beta=true",
    "method": "POST",
    "stream": True,
}
```

From `pyccode/model.py`, the minimum server contract is:

```python
PyccodeMinimumServerContract = {
    "request": {
        "headers": [
            "authorization | x-api-key",
            "anthropic-version",
            "anthropic-beta",
            "content-type: application/json",
        ],
        "body": {
            "model": "...",
            "system": [*system_blocks],
            "messages": [...],
            "max_tokens": 32000,
            "metadata": {"user_id": "..."},
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "context_management": {"edits": [...]},
            "stream": True,
            "tools": "optional",
            "temperature": "optional",
        },
    },
    "response": {
        "status": 200,
        "content-type": "text/event-stream",
        "events": [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
    },
}
```

### A Bare Mode

Current local A bare-mode captures also use the streaming request path:

```python
ABareObservedRequest = {
    "path": "/v1/messages?beta=true",
    "method": "POST",
    "stream": True,
}
```

Its minimum server contract is close to:

```python
ABareMinimumServerContract = {
    "request": {
        "headers": [
            "x-api-key",
            "anthropic-version",
            "anthropic-beta",
            "content-type: application/json",
        ],
        "body": {
            "model": "...",
            "system": [*system_blocks],
            "messages": [*message_items],
            "tools": [*tool_schemas],
            "max_tokens": 8000,
            "thinking": {"type": "enabled", "budget_tokens": 7999},
            "context_management": {"edits": [...]},
            "stream": True,
        },
    },
    "response": {
        "status": 200,
        "content-type": "text/event-stream",
        "events": [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
    },
}
```

### A Non-Bare Mode

The local non-bare A capture is the more important target because it uses the richer Bearer-token path and the wider tool surface.

```python
ANonBareObservedRequest = {
    "path": "/v1/messages?beta=true",
    "method": "POST",
    "stream": True,
    "auth": "bearer",
    "tool_count": 22,
    "max_tokens": 32000,
}
```

## Streaming Event Observations

The broad streaming lifecycle that matters most is:

```python
AnthropicStreamingLifecycle = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
]
```

Approximate semantics:

```python
StreamingEventSemantics = {
    "message_start": "seed the assistant message shell and initial usage",
    "content_block_start": "start one block such as text, thinking, or tool_use",
    "content_block_delta": "append text, partial JSON, or thinking deltas",
    "content_block_stop": "finalize one content block",
    "message_delta": "carry final usage updates and stop_reason",
    "message_stop": "mark stream completion",
}
```

A fake streaming server should preserve the broad ordering above if we want client behavior to remain stable.

## Fake Server Behavior

The current fake implementation lives in `tests/fake_anthropic_messages_server.py`.

### Supported Capability Groups

The fake server currently supports:

```python
FakeServerCapabilities = {
    "model_listing": True,
    "non_streaming_messages": True,
    "streaming_messages": True,
    "token_counting": True,
    "oauth_token_exchange": True,
    "oauth_profile_lookup": True,
    "role_lookup": True,
    "first_token_date_lookup": True,
    "bootstrap_lookup": True,
    "profile_lookup": True,
    "oauth_api_key_creation": True,
}
```

In other words, it already emulates the minimum endpoint family needed to capture both bare and non-bare local request paths.

### Non-Streaming Response Shape

```python
FakeNonStreamingResponse = {
    "id": "msg_fake",
    "type": "message",
    "role": "assistant",
    "model": "{requested model}",
    "content": [{"type": "text", "text": "{configured response text}"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
```

### Streaming Response Shape

```python
FakeStreamingResponse = [
    ("message_start", {"message": {"content": [], "usage": {...}}}),
    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "..."}}),
    ("content_block_stop", {"index": 0}),
    ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
]
```

## Authentication Behavior in Fake Mode

The fake server does not validate credentials.

That means local captures can be driven with:

- a fake API key for bare-mode request capture
- sandboxed fake OAuth state for non-bare Bearer-token capture
- a separate fake OAuth home for `pyccode` so it never reads real user credentials by accident

This is intentional. The current purpose of fake mode is request capture, not auth correctness.

## What the Fake Server Still Does Not Model

The fake server still does not try to model:

- real account authorization rules
- subscription checks
- model access rules
- real token validation or refresh semantics
- nuanced server-side error surfaces
- the full variety of thinking, tool, and attachment block types

That boundary is acceptable for the current alignment work because the main need is stable local capture of outbound request shape.

## Practical Takeaway

For current work, the server-side contract to protect is:

- preserve the streaming event order
- preserve the request endpoint family
- preserve the minimum OAuth-related fake endpoints needed for non-bare capture
- keep the fake auth behavior permissive so the client request shape stays observable

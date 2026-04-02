from __future__ import annotations

from tests.compare_three_way_messages_requests import (
    build_comparison_payload,
    build_markdown,
    summarize,
)


def make_capture(
    *,
    path: str,
    headers: dict[str, str],
    system: str | list[dict[str, str]],
    tools: list[str],
    stream: bool,
    max_tokens: int,
    messages: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "path": path,
        "headers": headers,
        "body": {
            "system": system,
            "messages": messages,
            "tools": [{"name": name} for name in tools],
            "stream": stream,
            "max_tokens": max_tokens,
        },
    }


def test_summarize_detects_bearer_auth_and_system_blocks() -> None:
    capture = make_capture(
        path="/v1/messages?beta=true",
        headers={
            "Authorization": "Bearer fake-access-token",
            "Anthropic-Beta": "oauth-2025-04-20",
            "User-Agent": "claude-cli/test",
        },
        system=[
            {"type": "text", "text": "billing"},
            {"type": "text", "text": "prompt"},
        ],
        tools=["Read", "Write"],
        stream=True,
        max_tokens=32000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder> skills"},
                    {"type": "text", "text": "say hi"},
                ],
            }
        ],
    )

    summary = summarize(capture)

    assert summary["path"] == "/v1/messages?beta=true"
    assert summary["auth_mode"] == "bearer"
    assert summary["system_type"] == "list"
    assert summary["system_block_count"] == 2
    assert summary["tool_names"] == ["Read", "Write"]
    assert summary["first_message_text_prefixes"] == [
        "<system-reminder> skills",
        "say hi",
    ]


def test_build_comparison_payload_uses_nonbare_as_target() -> None:
    pyccode = make_capture(
        path="/v1/messages?beta=true",
        headers={"authorization": "Bearer fake-access-token"},
        system=[{"type": "text", "text": "prompt"}],
        tools=["Agent", "AskUserQuestion", "Bash", "Write"],
        stream=True,
        max_tokens=32000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder> skills"},
                    {"type": "text", "text": "<system-reminder> date"},
                    {
                        "type": "text",
                        "text": "say hi",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
    )
    bare = make_capture(
        path="/v1/messages?beta=true",
        headers={"x-api-key": "fake-key"},
        system=[{"type": "text", "text": "prompt"}],
        tools=["Bash", "Read", "Edit"],
        stream=True,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder> date"},
                    {"type": "text", "text": "say hi"},
                ],
            }
        ],
    )
    nonbare = make_capture(
        path="/v1/messages?beta=true",
        headers={"authorization": "Bearer fake-access-token"},
        system=[
            {"type": "text", "text": "billing"},
            {"type": "text", "text": "prompt"},
        ],
        tools=["Agent", "Bash", "Read", "Write"],
        stream=True,
        max_tokens=32000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder> skills"},
                    {"type": "text", "text": "<system-reminder> date"},
                    {"type": "text", "text": "say hi"},
                ],
            }
        ],
    )

    payload = build_comparison_payload(pyccode, bare, nonbare)
    markdown = build_markdown(payload)

    assert payload["target"] == "claude_nonbare"
    assert payload["diff_vs_target"]["pyccode"]["path_matches"] is True
    assert payload["diff_vs_target"]["pyccode"]["stream_matches"] is True
    assert payload["diff_vs_target"]["pyccode"]["auth_mode_matches"] is True
    assert payload["diff_vs_target"]["pyccode"]["max_tokens_matches"] is True
    assert payload["diff_vs_target"]["pyccode"]["first_message_content_count_matches"] is True
    assert payload["diff_vs_target"]["pyccode"]["tool_overlap_with_target"] == [
        "Agent",
        "Bash",
        "Write",
    ]
    assert payload["diff_vs_target"]["claude_bare"]["path_matches"] is True
    assert payload["diff_vs_target"]["claude_bare"]["stream_matches"] is True
    assert payload["diff_vs_target"]["claude_bare"]["tool_overlap_with_target"] == [
        "Bash",
        "Read",
    ]
    assert "Target alignment: `claude_nonbare`" in markdown
    assert "| pyccode | `/v1/messages?beta=true` | `True` | `bearer` | `list` | 4 | 32000 |" in markdown

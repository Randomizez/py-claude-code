from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer

from pyccode.context import ContextManager
from pyccode.model import AnthropicMessagesConfig, AnthropicMessagesModelClient
from pyccode.protocol import ConversationMessage
from pyccode.tools import build_default_tool_registry
from tests.fake_anthropic_messages_server import CaptureStore, build_handler
from tests.prepare_fake_oauth_home import prepare_fake_oauth_home


async def test_model_client_uses_streaming_v1_messages_shape(tmp_path, monkeypatch) -> None:
    claude_home = tmp_path / "claude-home"
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "STREAM HELLO"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    try:
        manager = ContextManager(cwd=tmp_path)
        tools = build_default_tool_registry().model_visible_specs()
        prompt = manager.build_prompt(
            [ConversationMessage.user_text("say hi")],
            tools,
            turn_id="turn-123",
        )
        client = AnthropicMessagesModelClient(
            AnthropicMessagesConfig(
                model="claude-fake",
                base_url=f"http://127.0.0.1:{httpd.server_port}",
            )
        )

        response = await client.complete(prompt)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert response.message.text_content() == "STREAM HELLO"
    assert response.stop_reason == "end_turn"

    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(capture_files) == 1
    capture = json.loads(capture_files[0].read_text())
    headers = {key.lower(): value for key, value in capture["headers"].items()}
    body = capture["body"]

    assert capture["path"] == "/v1/messages?beta=true"
    assert headers["x-api-key"] == "fake-key"
    assert headers["x-app"] == "cli"
    assert "x-claude-code-session-id" in headers
    assert "claude-code-20250219" in headers["anthropic-beta"]
    assert body["stream"] is True
    assert isinstance(body["system"], list)
    assert len(body["system"]) == 4
    assert body["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert body["system"][2]["cache_control"] == {"type": "ephemeral", "scope": "global"}
    assert len(body["tools"]) == 22
    assert body["tools"][0]["name"] == "Agent"
    assert body["tools"][-1]["name"] == "Write"
    assert "The following skills are available for use with the Skill tool" in body["messages"][0]["content"][0]["text"]
    assert "Today's date is" in body["messages"][0]["content"][1]["text"]
    assert body["messages"][0]["content"][2]["text"] == "say hi"
    assert body["messages"][0]["content"][2]["cache_control"] == {"type": "ephemeral"}
    assert body["max_tokens"] == 32000
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 31999}
    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }
    user_id = json.loads(body["metadata"]["user_id"])
    assert len(user_id["device_id"]) == 64
    assert user_id["session_id"] == headers["x-claude-code-session-id"]
    global_config = json.loads((claude_home / ".claude.json").read_text())
    assert global_config["userID"] == user_id["device_id"]


def test_model_client_prefers_bearer_auth_when_auth_token_exists(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fake-access-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    client = AnthropicMessagesModelClient(AnthropicMessagesConfig(model="claude-fake"))
    headers = client._build_headers()

    assert headers["authorization"] == "Bearer fake-access-token"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" in headers["anthropic-beta"]


async def test_model_client_reads_bearer_token_from_claude_config_dir(
    tmp_path,
    monkeypatch,
) -> None:
    oauth_home = tmp_path / "oauth-home"
    prepare_fake_oauth_home(oauth_home)
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "STREAM HELLO"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(oauth_home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    try:
        manager = ContextManager(cwd=tmp_path)
        tools = build_default_tool_registry().model_visible_specs()
        prompt = manager.build_prompt(
            [ConversationMessage.user_text("say hi")],
            tools,
            turn_id="turn-456",
        )
        client = AnthropicMessagesModelClient(
            AnthropicMessagesConfig(
                model="claude-fake",
                base_url=f"http://127.0.0.1:{httpd.server_port}",
            )
        )
        response = await client.complete(prompt)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert response.message.text_content() == "STREAM HELLO"
    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    capture = json.loads(capture_files[0].read_text())
    headers = {key.lower(): value for key, value in capture["headers"].items()}
    assert headers["authorization"] == "Bearer fake-access-token"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" in headers["anthropic-beta"]
    assert len(capture["body"]["tools"]) == 22
    user_id = json.loads(capture["body"]["metadata"]["user_id"])
    assert user_id["account_uuid"] == "00000000-0000-4000-8000-000000000001"
    assert user_id["session_id"] == headers["x-claude-code-session-id"]

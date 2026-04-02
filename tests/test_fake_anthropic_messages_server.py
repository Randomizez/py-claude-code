from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
import urllib.request

from tests.fake_anthropic_messages_server import CaptureStore, build_handler


def test_fake_handler_captures_non_streaming_messages_request(tmp_path) -> None:
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "FAKE HELLO"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/messages",
            data=json.dumps(
                {
                    "model": "claude-fake",
                    "system": "demo-system",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "say hi"}]}],
                    "max_tokens": 128,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "fake-key",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body = json.loads(response.read().decode("utf-8"))
            content_type = response.headers.get("Content-Type")
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert content_type == "application/json"
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "FAKE HELLO"

    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(capture_files) == 1
    capture = json.loads(capture_files[0].read_text())
    assert capture["path"] == "/messages"
    assert capture["body"]["model"] == "claude-fake"
    assert capture["response"]["status"] == 200
    assert capture["response"]["body"]["content"][0]["text"] == "FAKE HELLO"


def test_fake_handler_captures_streaming_messages_request(tmp_path) -> None:
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "STREAM HELLO"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/messages",
            data=json.dumps(
                {
                    "model": "claude-fake",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "say hi"}]}],
                    "max_tokens": 128,
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-api-key": "fake-key",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body_text = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type")
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert content_type == "text/event-stream"
    assert "event: message_start" in body_text
    assert "STREAM HELLO" in body_text

    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(capture_files) == 1
    capture = json.loads(capture_files[0].read_text())
    assert capture["path"] == "/v1/messages"
    assert capture["body"]["stream"] is True
    assert capture["response"]["status"] == 200
    assert "STREAM HELLO" in capture["response"]["body"]


def test_fake_handler_serves_oauth_token_exchange_and_profile(tmp_path) -> None:
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "unused"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        token_request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/oauth/token",
            data=json.dumps(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": "demo-refresh",
                    "client_id": "demo-client",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(token_request, timeout=5.0) as response:
            token_payload = json.loads(response.read().decode("utf-8"))

        profile_request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/api/oauth/profile",
            headers={"Authorization": "Bearer fake-access-token"},
            method="GET",
        )
        with urllib.request.urlopen(profile_request, timeout=5.0) as response:
            profile_payload = json.loads(response.read().decode("utf-8"))

        roles_request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/api/oauth/claude_cli/roles",
            headers={"Authorization": "Bearer fake-access-token"},
            method="GET",
        )
        with urllib.request.urlopen(roles_request, timeout=5.0) as response:
            roles_payload = json.loads(response.read().decode("utf-8"))
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert token_payload["access_token"] == "fake-access-token"
    assert token_payload["refresh_token"] == "fake-refresh-token"
    assert "user:profile" in token_payload["scope"]
    assert profile_payload["account"]["email"] == "fake-user@example.com"
    assert profile_payload["organization"]["organization_type"] == "claude_max"
    assert roles_payload["organization_role"] == "owner"

    capture_files = sorted(capture_root.glob("*_*.json"))
    assert len(capture_files) == 3

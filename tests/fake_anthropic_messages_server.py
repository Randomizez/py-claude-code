"""Local Anthropic Messages capture server for `pyccode` alignment work.

This helper mirrors the role of `tests/fake_responses_server.py` in
`pycodex`, but targets Anthropic's Messages API surface instead of OpenAI's
Responses API.

It supports two modes:

- fake mode: return a fixed local Anthropic-compatible response while recording
  requests
- proxy mode: forward requests to a real upstream base URL while recording both
  the request and the upstream response
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request

DEFAULT_PORT = 8766
DEFAULT_MODEL_ID = "claude-sonnet-4-5"
DEFAULT_OUTPUT_ROOT = Path(".debug") / "anthropic_prompt_capture"
DEFAULT_RESPONSE_TEXT = "Hi from fake Anthropic."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_FAKE_ACCESS_TOKEN = "fake-access-token"
DEFAULT_FAKE_REFRESH_TOKEN = "fake-refresh-token"
DEFAULT_FAKE_RAW_API_KEY = "sk-ant-fake-created-api-key"
DEFAULT_FAKE_ACCOUNT_UUID = "00000000-0000-4000-8000-000000000001"
DEFAULT_FAKE_ORG_UUID = "00000000-0000-4000-8000-000000000002"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.fake_anthropic_messages_server",
        description=(
            "Capture local Anthropic Messages API traffic. By default returns a "
            "fixed fake response; with --proxy-base-url it forwards to a real "
            "upstream."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory used to store captured request JSON files.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model id returned from /models in fake mode.",
    )
    parser.add_argument(
        "--response-text",
        default=DEFAULT_RESPONSE_TEXT,
        help="Assistant text returned from the fake /messages response.",
    )
    parser.add_argument(
        "--proxy-base-url",
        default=None,
        help="When set, proxy requests to this upstream base URL instead of using fake responses.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Timeout used when proxying upstream requests.",
    )
    return parser


class CaptureStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._root / "counter.txt"

    @property
    def root(self) -> Path:
        return self._root

    def next_request_id(self) -> int:
        if self._counter_path.exists():
            value = int(self._counter_path.read_text()) + 1
        else:
            value = 1
        self._counter_path.write_text(str(value))
        return value

    def write_capture(
        self,
        request_id: int,
        method: str,
        path: str,
        headers: dict[str, str],
        body: object,
        response_status: int,
        response_headers: dict[str, str],
        response_body: object,
    ) -> None:
        parsed = urlparse(path)
        safe_name = parsed.path.strip("/").replace("/", "_") or "root"
        filename = self._root / f"{request_id:03d}_{method}_{safe_name}.json"
        filename.write_text(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "headers": headers,
                    "body": body,
                    "response": {
                        "status": response_status,
                        "headers": response_headers,
                        "body": response_body,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _decode_body(body_bytes: bytes, content_type: str | None = None) -> object:
    text = body_bytes.decode("utf-8", errors="replace")
    if content_type and "application/json" in content_type.lower():
        try:
            return json.loads(text)
        except Exception:
            return text
    try:
        return json.loads(text)
    except Exception:
        return text


def _write_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    headers: dict[str, str],
    body_bytes: bytes,
) -> None:
    handler.send_response(status)
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"content-length", "connection", "transfer-encoding"}:
            continue
        handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    handler.wfile.write(body_bytes)


def _request_headers_for_proxy(headers) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"host", "content-length", "connection"}:
            continue
        forwarded[key] = value
    return forwarded


def _build_upstream_url(upstream_base_url: str, request_path: str) -> str:
    parsed_base = urlparse(upstream_base_url)
    base_origin = f"{parsed_base.scheme}://{parsed_base.netloc}"
    base_path = parsed_base.path.rstrip("/")
    parsed_request = urlparse(request_path)
    request_only_path = parsed_request.path or "/"

    if base_path and request_only_path.startswith(f"{base_path}/"):
        path = request_only_path
    elif base_path and request_only_path == base_path:
        path = request_only_path
    else:
        path = urljoin(f"{base_path}/", request_only_path.lstrip("/"))

    url = f"{base_origin}{path}"
    if parsed_request.query:
        return f"{url}?{parsed_request.query}"
    return url


def _fake_models_payload(model_id: str) -> dict[str, object]:
    return {
        "data": [
            {
                "type": "model",
                "id": model_id,
                "display_name": model_id,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "has_more": False,
        "first_id": model_id,
        "last_id": model_id,
    }


def _fake_count_tokens_payload() -> dict[str, int]:
    return {"input_tokens": 1}


def _fake_oauth_token_payload() -> dict[str, object]:
    return {
        "access_token": DEFAULT_FAKE_ACCESS_TOKEN,
        "refresh_token": DEFAULT_FAKE_REFRESH_TOKEN,
        "token_type": "bearer",
        "expires_in": 3600,
        "scope": (
            "user:profile user:inference user:sessions:claude_code "
            "user:mcp_servers user:file_upload"
        ),
        "account": {
            "uuid": DEFAULT_FAKE_ACCOUNT_UUID,
            "email_address": "fake-user@example.com",
        },
        "organization": {
            "uuid": DEFAULT_FAKE_ORG_UUID,
        },
    }


def _fake_oauth_profile_payload() -> dict[str, object]:
    return {
        "account": {
            "uuid": DEFAULT_FAKE_ACCOUNT_UUID,
            "email": "fake-user@example.com",
            "display_name": "Fake Claude User",
            "created_at": "2026-01-01T00:00:00Z",
        },
        "organization": {
            "uuid": DEFAULT_FAKE_ORG_UUID,
            "name": "Fake Claude Org",
            "organization_type": "claude_max",
            "rate_limit_tier": "default_claude_max_20x",
            "has_extra_usage_enabled": True,
            "billing_type": "subscription",
            "subscription_created_at": "2026-01-02T00:00:00Z",
        },
    }


def _fake_roles_payload() -> dict[str, object]:
    return {
        "organization_role": "owner",
        "workspace_role": "owner",
        "organization_name": "Fake Claude Org",
    }


def _fake_bootstrap_payload(model_id: str) -> dict[str, object]:
    return {
        "client_data": {
            "fake": True,
            "source": "fake_anthropic_messages_server",
        },
        "additional_model_options": [
            {
                "model": model_id,
                "name": model_id,
                "description": "Fake model option returned by the local capture server.",
            }
        ],
    }


def _fake_first_token_date_payload() -> dict[str, object]:
    return {
        "first_token_date": "2026-01-03T00:00:00Z",
    }


def _fake_created_api_key_payload() -> dict[str, object]:
    return {
        "raw_key": DEFAULT_FAKE_RAW_API_KEY,
    }


def _fake_claude_cli_profile_payload() -> dict[str, object]:
    profile = _fake_oauth_profile_payload()
    return {
        "account": profile["account"],
        "organization": profile["organization"],
    }


def _fake_message_payload(model_id: str, response_text: str) -> dict[str, object]:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": response_text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _fake_stream_payload(model_id: str, response_text: str) -> str:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": response_text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    parts: list[str] = []
    for event_name, payload in events:
        parts.append(f"event: {event_name}\n")
        parts.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
    return "".join(parts)


def build_fake_handler(
    capture_store: CaptureStore,
    model_id: str,
    response_text: str,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AnthropicPromptCapture/0.1"

        def log_message(self, format: str, *args) -> None:
            del format, args
            return

        def do_GET(self) -> None:
            request_id = capture_store.next_request_id()
            parsed = urlparse(self.path)
            if parsed.path.endswith("/models") or parsed.path == "/models":
                payload = _fake_models_payload(model_id)
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            if parsed.path.endswith("/api/oauth/profile"):
                payload = _fake_oauth_profile_payload()
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            if parsed.path.endswith("/api/oauth/claude_cli/roles"):
                payload = _fake_roles_payload()
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            if parsed.path.endswith("/api/organization/claude_code_first_token_date"):
                payload = _fake_first_token_date_payload()
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            if parsed.path.endswith("/api/claude_cli/bootstrap"):
                payload = _fake_bootstrap_payload(model_id)
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            if parsed.path.endswith("/api/claude_cli_profile"):
                payload = _fake_claude_cli_profile_payload()
                body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    None,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, body_bytes)
                return

            payload = {"ok": True}
            body_bytes = json.dumps(payload).encode("utf-8")
            response_headers = {"Content-Type": "application/json"}
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                None,
                200,
                response_headers,
                payload,
            )
            _write_response(self, 200, response_headers, body_bytes)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request_body_bytes = self.rfile.read(length)
            decoded_request_body = _decode_body(
                request_body_bytes,
                self.headers.get("Content-Type"),
            )
            request_id = capture_store.next_request_id()
            parsed = urlparse(self.path)

            if parsed.path.endswith("/messages/count_tokens"):
                payload = _fake_count_tokens_payload()
                response_body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    decoded_request_body,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, response_body_bytes)
                return

            if parsed.path.endswith("/v1/oauth/token"):
                payload = _fake_oauth_token_payload()
                response_body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    decoded_request_body,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, response_body_bytes)
                return

            if parsed.path.endswith("/api/oauth/claude_cli/create_api_key"):
                payload = _fake_created_api_key_payload()
                response_body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    decoded_request_body,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, response_body_bytes)
                return

            if parsed.path.endswith("/messages") or parsed.path == "/messages":
                is_stream = (
                    isinstance(decoded_request_body, dict)
                    and bool(decoded_request_body.get("stream"))
                )
                if is_stream:
                    response_payload = _fake_stream_payload(model_id, response_text)
                    response_body_bytes = response_payload.encode("utf-8")
                    response_headers = {
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    }
                    capture_store.write_capture(
                        request_id,
                        self.command,
                        self.path,
                        dict(self.headers),
                        decoded_request_body,
                        200,
                        response_headers,
                        response_payload,
                    )
                    _write_response(self, 200, response_headers, response_body_bytes)
                    return

                payload = _fake_message_payload(model_id, response_text)
                response_body_bytes = json.dumps(payload).encode("utf-8")
                response_headers = {"Content-Type": "application/json"}
                capture_store.write_capture(
                    request_id,
                    self.command,
                    self.path,
                    dict(self.headers),
                    decoded_request_body,
                    200,
                    response_headers,
                    payload,
                )
                _write_response(self, 200, response_headers, response_body_bytes)
                return

            payload = {"ok": True}
            response_body_bytes = json.dumps(payload).encode("utf-8")
            response_headers = {"Content-Type": "application/json"}
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                decoded_request_body,
                200,
                response_headers,
                payload,
            )
            _write_response(self, 200, response_headers, response_body_bytes)

    return Handler


def build_proxy_handler(
    capture_store: CaptureStore,
    upstream_base_url: str,
    timeout_seconds: float,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AnthropicPromptProxy/0.1"

        def log_message(self, format: str, *args) -> None:
            del format, args
            return

        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            self._forward()

        def _forward(self) -> None:
            request_id = capture_store.next_request_id()
            length = int(self.headers.get("Content-Length", "0"))
            request_body_bytes = self.rfile.read(length) if length else b""
            decoded_request_body = (
                _decode_body(request_body_bytes, self.headers.get("Content-Type"))
                if request_body_bytes
                else None
            )

            target_url = _build_upstream_url(upstream_base_url, self.path)
            forwarded_headers = _request_headers_for_proxy(self.headers)
            request = urllib.request.Request(
                target_url,
                data=request_body_bytes if self.command != "GET" else None,
                headers=forwarded_headers,
                method=self.command,
            )

            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    response_status = getattr(response, "status", 200)
                    response_headers = dict(response.headers.items())
                    response_body_bytes = response.read()
            except urllib.error.HTTPError as exc:
                response_status = exc.code
                response_headers = dict(exc.headers.items())
                response_body_bytes = exc.read()

            decoded_response_body = _decode_body(
                response_body_bytes,
                response_headers.get("Content-Type"),
            )
            capture_store.write_capture(
                request_id,
                self.command,
                self.path,
                dict(self.headers),
                decoded_request_body,
                response_status,
                response_headers,
                decoded_response_body,
            )
            _write_response(self, response_status, response_headers, response_body_bytes)

    return Handler


def build_handler(
    capture_store: CaptureStore,
    model_id: str,
    response_text: str,
):
    """Backward-compatible alias used by tests."""

    return build_fake_handler(capture_store, model_id, response_text)


def main() -> None:
    args = build_parser().parse_args()
    capture_store = CaptureStore(Path(args.root))
    if args.proxy_base_url:
        handler = build_proxy_handler(
            capture_store,
            args.proxy_base_url,
            args.request_timeout_seconds,
        )
        mode = f"proxy -> {args.proxy_base_url}"
    else:
        handler = build_fake_handler(
            capture_store,
            args.model_id,
            args.response_text,
        )
        mode = "fake"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"listening on http://127.0.0.1:{args.port}", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"capturing into {capture_store.root}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

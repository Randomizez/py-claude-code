from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pyccode.agent import AgentLoop
from pyccode.codex_model import ResponsesModelClient
from pyccode.context import ContextManager
from pyccode.protocol import ConversationMessage
from pyccode.tools import BaseTool, ToolRegistry


def _write_codex_config(path: Path, base_url: str) -> None:
    path.write_text(
        "\n".join(
            [
                'model = "gpt-test"',
                'model_provider = "local"',
                '',
                '[model_providers.local]',
                f'base_url = "{base_url}"',
                'env_key = "OPENAI_API_KEY"',
                'wire_api = "responses"',
            ]
        )
    )


class EchoTool(BaseTool):
    name = "Echo"
    description = "Echo text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def run(self, context, args):
        return {"echo": args["text"]}


def _build_sse_payload(events: list[str]) -> bytes:
    return "".join(events).encode("utf-8")


def build_handler(captures: list[dict[str, object]]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FakeResponses/0.1"

        def log_message(self, format: str, *args) -> None:
            del format, args
            return

        def do_GET(self) -> None:
            payload = {
                "object": "list",
                "data": [{"id": "gpt-test", "object": "model"}],
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request_bytes = self.rfile.read(length)
            request_body = json.loads(request_bytes.decode("utf-8"))
            captures.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": request_body,
                }
            )

            input_items = request_body.get("input", [])
            saw_tool_output = any(
                isinstance(item, dict) and item.get("type") == "function_call_output"
                for item in input_items
            )
            if saw_tool_output:
                body = _build_sse_payload(
                    [
                        'event: response.output_item.done\n',
                        'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}\n\n',
                        'event: response.completed\n',
                        'data: {"type":"response.completed","response":{"id":"resp_2","output":[]}}\n\n',
                    ]
                )
            else:
                body = _build_sse_payload(
                    [
                        'event: response.output_item.done\n',
                        'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"Echo","arguments":"{\\"text\\":\\"hello\\"}"}}\n\n',
                        'event: response.completed\n',
                        'data: {"type":"response.completed","response":{"id":"resp_1","output":[]}}\n\n',
                    ]
                )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_responses_model_client_builds_payload_from_pyccode_prompt(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_codex_config(config_path, "http://127.0.0.1:9999")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    registry = ToolRegistry()
    registry.register(EchoTool())
    manager = ContextManager(cwd=tmp_path)
    prompt = manager.build_prompt(
        [ConversationMessage.user_text("say hi")],
        registry.model_visible_specs(),
        turn_id="turn-1",
    )
    client = ResponsesModelClient.from_codex_config(config_path=config_path)

    payload = client._build_payload(prompt)

    assert payload["model"] == "gpt-test"
    assert payload["stream"] is True
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert isinstance(payload["instructions"], str)
    assert "You are a Claude agent" in payload["instructions"]
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "Echo",
            "description": "Echo text.",
            "parameters": EchoTool.input_schema,
            "strict": False,
        }
    ]
    first_input = payload["input"][0]
    assert first_input["type"] == "message"
    assert first_input["role"] == "user"
    assert len(first_input["content"]) == 3
    assert "The following skills are available for use with the Skill tool" in first_input["content"][0]["text"]
    assert "Today's date is" in first_input["content"][1]["text"]
    assert first_input["content"][2]["text"] == "say hi"


async def test_responses_model_client_lists_models(tmp_path, monkeypatch) -> None:
    captures: list[dict[str, object]] = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(captures))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    config_path = tmp_path / "config.toml"
    _write_codex_config(config_path, f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = ResponsesModelClient.from_codex_config(config_path=config_path)
    try:
        models = await client.list_models()
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert models == ["gpt-test"]


async def test_agent_loop_runs_on_codex_responses_backend(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    captures: list[dict[str, object]] = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(captures))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    _write_codex_config(config_path, f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    registry = ToolRegistry()
    registry.register(EchoTool())
    manager = ContextManager(cwd=tmp_path)
    client = ResponsesModelClient.from_codex_config(config_path=config_path)
    loop = AgentLoop(client, registry, context_manager=manager)

    try:
        result = await loop.run_turn(["use the echo tool"])
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert result.output_text == "done"
    assert len(captures) == 2
    assert captures[0]["path"] == "/responses"
    first_headers = {key.lower(): value for key, value in captures[0]["headers"].items()}
    assert first_headers["authorization"] == "Bearer test-key"
    assert first_headers["originator"] == "pyccode"
    first_body = captures[0]["body"]
    assert first_body["model"] == "gpt-test"
    assert first_body["tools"][0]["name"] == "Echo"
    second_input = captures[1]["body"]["input"]
    assert any(
        isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == "call_1"
        for item in second_input
    )

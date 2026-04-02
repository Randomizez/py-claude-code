from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib.parse import urlencode, urlparse

import requests

from .auth import ResolvedAuth, resolve_auth
from .protocol import (
    ConversationMessage,
    JSONDict,
    ModelResponse,
    ModelStreamEvent,
    Prompt,
    SystemTextBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from .utils import uuid7_string

ModelStreamEventHandler = Callable[[ModelStreamEvent], None]
NOOP_MODEL_STREAM_EVENT_HANDLER: ModelStreamEventHandler = lambda _event: None


class ModelClient(Protocol):
    async def complete(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
        ...


@dataclass(frozen=True, slots=True)
class AnthropicMessagesConfig:
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    oauth_token_env: str = "CLAUDE_CODE_OAUTH_TOKEN"
    base_url: str = "https://api.anthropic.com/v1"
    anthropic_version: str = "2023-06-01"
    beta_header: str | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> "AnthropicMessagesConfig":
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL")
        if not resolved_model:
            raise RuntimeError("missing model; pass --model or set ANTHROPIC_MODEL")
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", cls.base_url)
        return cls(
            model=resolved_model,
            base_url=resolved_base_url,
            timeout_seconds=timeout_seconds,
            beta_header=os.environ.get("ANTHROPIC_BETA"),
        )

    def resolve_auth(self) -> ResolvedAuth:
        return resolve_auth(
            api_key_env=self.api_key_env,
            auth_token_env=self.auth_token_env,
            oauth_token_env=self.oauth_token_env,
        )


class AnthropicApiError(RuntimeError):
    pass


class AnthropicMessagesModelClient:
    _user_agent = "claude-cli/2.1.89 (external, sdk-cli)"

    def __init__(self, config: AnthropicMessagesConfig) -> None:
        self._config = config
        self.model = config.model
        self._session_id = uuid7_string()

    async def complete(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, prompt, event_handler)

    def _complete_sync(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
        auth = self._config.resolve_auth()
        payload = self._build_payload(prompt)
        headers = self._build_headers(auth)
        url = self._messages_url()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._config.timeout_seconds,
                stream=bool(payload.get("stream")),
            )
        except requests.RequestException as exc:
            raise AnthropicApiError(f"messages request failed: {exc}") from exc

        with response:
            if response.status_code >= 400:
                raise AnthropicApiError(
                    f"messages request failed with status {response.status_code}: "
                    f"{response.text[:500]}"
                )
            if payload.get("stream"):
                return self._parse_stream(
                    response.iter_lines(chunk_size=1, decode_unicode=True),
                    event_handler,
                )
            data = response.json()
            return self._parse_response(data, event_handler)

    def _messages_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.path.rstrip("/").endswith("/v1"):
            messages_path = f"{base_url}/messages"
        else:
            messages_path = f"{base_url}/v1/messages"
        return f"{messages_path}?{urlencode({'beta': 'true'})}"

    def _build_headers(self, auth: ResolvedAuth | None = None) -> dict[str, str]:
        auth = auth or self._config.resolve_auth()
        headers = {
            "accept": "application/json",
            "accept-language": "*",
            "content-type": "application/json",
            "anthropic-version": self._config.anthropic_version,
            "anthropic-beta": self._resolve_beta_header(auth),
            "anthropic-dangerous-direct-browser-access": "true",
            "sec-fetch-mode": "cors",
            "x-app": "cli",
            "x-claude-code-session-id": self._session_id,
            "user-agent": self._user_agent,
            "x-stainless-arch": "x64",
            "x-stainless-lang": "js",
            "x-stainless-os": "Linux",
            "x-stainless-package-version": "0.74.0",
            "x-stainless-retry-count": "0",
            "x-stainless-runtime": "node",
            "x-stainless-runtime-version": "v18.20.8",
            "x-stainless-timeout": "600",
        }
        if auth.mode == "bearer":
            headers["authorization"] = f"Bearer {auth.value}"
        else:
            headers["x-api-key"] = auth.value
        return headers

    def _build_payload(self, prompt: Prompt) -> dict[str, object]:
        auth = self._config.resolve_auth()
        payload: dict[str, object] = {
            "model": self.model,
            "system": self._serialize_system(prompt.system),
            "messages": self._serialize_messages(prompt),
            "max_tokens": prompt.max_tokens,
            "stream": prompt.stream,
        }
        if prompt.tools:
            payload["tools"] = [tool.serialize() for tool in prompt.tools]
        if prompt.temperature is not None:
            payload["temperature"] = prompt.temperature
        payload["metadata"] = self._resolve_metadata(prompt, auth)
        if prompt.thinking is not None:
            payload["thinking"] = prompt.thinking
        if prompt.context_management is not None:
            payload["context_management"] = prompt.context_management
        return payload

    def _serialize_system(
        self,
        system: str | tuple[SystemTextBlock, ...],
    ) -> str | list[dict[str, object]]:
        if isinstance(system, str):
            return system
        return [block.serialize() for block in system]

    def _serialize_messages(self, prompt: Prompt) -> list[dict[str, object]]:
        serialized = [
            message.serialize(include_thinking=False)
            for message in prompt.messages
        ]
        if not prompt.user_reminders:
            return serialized

        reminder_blocks = [
            {"type": "text", "text": reminder}
            for reminder in prompt.user_reminders
        ]
        if serialized and serialized[0].get("role") == "user":
            first_message = dict(serialized[0])
            first_content = list(first_message.get("content", []))
            first_message["content"] = reminder_blocks + first_content
            serialized[0] = first_message
        else:
            serialized.insert(0, {"role": "user", "content": reminder_blocks})

        if serialized:
            last_message = dict(serialized[-1])
            if last_message.get("role") == "user":
                last_content = list(last_message.get("content", []))
                if last_content and last_content[-1].get("type") == "text":
                    last_item = dict(last_content[-1])
                    last_item["cache_control"] = {"type": "ephemeral"}
                    last_content[-1] = last_item
                    last_message["content"] = last_content
                    serialized[-1] = last_message
        return serialized

    def _resolve_metadata(
        self,
        prompt: Prompt,
        auth: ResolvedAuth,
    ) -> dict[str, str]:
        user_id_payload = {
            "device_id": auth.device_id or "pyccode-local",
            "account_uuid": auth.account_uuid or "",
            "session_id": self._session_id,
        }
        if prompt.metadata is not None and isinstance(prompt.metadata.get("user_id"), str):
            try:
                parsed = json.loads(prompt.metadata["user_id"])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                if not auth.device_id:
                    user_id_payload["device_id"] = str(
                        parsed.get("device_id", user_id_payload["device_id"])
                    )
        return {"user_id": json.dumps(user_id_payload, ensure_ascii=False, separators=(",", ":"))}

    def _resolve_beta_header(self, auth: ResolvedAuth | None = None) -> str:
        auth = auth or self._config.resolve_auth()
        if self._config.beta_header:
            return self._config.beta_header

        beta_headers = [
            "claude-code-20250219",
            "interleaved-thinking-2025-05-14",
            "context-management-2025-06-27",
            "prompt-caching-scope-2026-01-05",
        ]
        if auth.mode == "bearer":
            beta_headers.insert(1, "oauth-2025-04-20")
        return ",".join(beta_headers)

    def _parse_response(
        self,
        data: JSONDict,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
        blocks = []
        for block in data.get("content", []):
            kind = block.get("type")
            if kind == "text":
                text = str(block.get("text", ""))
                event_handler(
                    ModelStreamEvent(kind="assistant_text", payload={"text": text})
                )
                blocks.append(TextBlock(text=text))
            elif kind == "tool_use":
                tool_use = ToolUseBlock(
                    id=str(block["id"]),
                    name=str(block["name"]),
                    input=dict(block.get("input", {})),
                )
                event_handler(
                    ModelStreamEvent(
                        kind="tool_use",
                        payload={
                            "id": tool_use.id,
                            "name": tool_use.name,
                            "input": tool_use.input,
                        },
                    )
                )
                blocks.append(tool_use)
            elif kind == "thinking":
                thinking = str(block.get("thinking") or block.get("text") or "")
                blocks.append(
                    ThinkingBlock(
                        text=thinking,
                        signature=(
                            str(block.get("signature"))
                            if block.get("signature") is not None
                            else None
                        ),
                    )
                )
        message = ConversationMessage(role="assistant", content=tuple(blocks))
        return ModelResponse(
            message=message,
            stop_reason=(str(data["stop_reason"]) if data.get("stop_reason") else None),
            raw=data,
        )

    def _parse_stream(
        self,
        lines,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
        block_states: dict[int, dict[str, object]] = {}
        output_blocks: list[TextBlock | ToolUseBlock | ThinkingBlock] = []
        stop_reason: str | None = None
        raw_events: list[dict[str, object]] = []
        current_event: str | None = None
        data_lines: list[str] = []

        def dispatch_event(event_name: str, payload: JSONDict) -> None:
            nonlocal stop_reason
            raw_events.append({"event": event_name, "data": payload})

            if event_name == "content_block_start":
                index = int(payload.get("index", 0))
                content_block = dict(payload.get("content_block", {}))
                block_type = str(content_block.get("type", "text"))
                state: dict[str, object] = {"type": block_type}
                if block_type == "text":
                    state["text"] = str(content_block.get("text", ""))
                elif block_type == "thinking":
                    state["thinking"] = str(
                        content_block.get("thinking") or content_block.get("text") or ""
                    )
                    if content_block.get("signature") is not None:
                        state["signature"] = str(content_block["signature"])
                elif block_type == "tool_use":
                    state["id"] = str(content_block.get("id", ""))
                    state["name"] = str(content_block.get("name", ""))
                    state["input"] = dict(content_block.get("input", {}))
                    state["input_json"] = ""
                block_states[index] = state
                return

            if event_name == "content_block_delta":
                index = int(payload.get("index", 0))
                state = block_states.setdefault(index, {"type": "text", "text": ""})
                delta = dict(payload.get("delta", {}))
                delta_type = str(delta.get("type", ""))
                if delta_type == "text_delta":
                    text = str(delta.get("text", ""))
                    state["text"] = str(state.get("text", "")) + text
                    event_handler(
                        ModelStreamEvent(
                            kind="assistant_text_delta",
                            payload={"text": text, "index": index},
                        )
                    )
                elif delta_type == "thinking_delta":
                    state["thinking"] = str(state.get("thinking", "")) + str(
                        delta.get("thinking") or delta.get("text") or ""
                    )
                elif delta_type == "input_json_delta":
                    state["input_json"] = str(state.get("input_json", "")) + str(
                        delta.get("partial_json", "")
                    )
                elif delta_type == "signature_delta":
                    state["signature"] = str(delta.get("signature", ""))
                return

            if event_name == "content_block_stop":
                index = int(payload.get("index", 0))
                state = block_states.pop(index, None)
                if state is None:
                    return
                block_type = str(state.get("type", "text"))
                if block_type == "text":
                    output_blocks.append(TextBlock(text=str(state.get("text", ""))))
                    return
                if block_type == "thinking":
                    output_blocks.append(
                        ThinkingBlock(
                            text=str(state.get("thinking", "")),
                            signature=(
                                str(state.get("signature"))
                                if state.get("signature") is not None
                                else None
                            ),
                        )
                    )
                    return
                if block_type == "tool_use":
                    tool_input = dict(state.get("input", {}))
                    input_json = str(state.get("input_json", "")).strip()
                    if input_json:
                        try:
                            parsed_input = json.loads(input_json)
                        except json.JSONDecodeError:
                            parsed_input = {"raw_input_json": input_json}
                        if isinstance(parsed_input, dict):
                            tool_input = parsed_input
                        else:
                            tool_input = {"value": parsed_input}
                    tool_use = ToolUseBlock(
                        id=str(state.get("id", "")),
                        name=str(state.get("name", "")),
                        input=tool_input,
                    )
                    event_handler(
                        ModelStreamEvent(
                            kind="tool_use",
                            payload={
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": tool_use.input,
                            },
                        )
                    )
                    output_blocks.append(tool_use)
                return

            if event_name == "message_delta":
                delta = payload.get("delta") or {}
                if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                    stop_reason = str(delta["stop_reason"])

        for raw_line in lines:
            if raw_line is None:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
            if not line:
                if current_event is None:
                    continue
                payload = json.loads("\n".join(data_lines)) if data_lines else {}
                dispatch_event(current_event, payload)
                current_event = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                current_event = line.partition(":")[2].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.partition(":")[2].lstrip())

        if current_event is not None:
            payload = json.loads("\n".join(data_lines)) if data_lines else {}
            dispatch_event(current_event, payload)

        return ModelResponse(
            message=ConversationMessage(role="assistant", content=tuple(output_blocks)),
            stop_reason=stop_reason,
            raw={"events": raw_events},
        )

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Protocol
import urllib.parse

import requests

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib

from .model import ModelStreamEventHandler, NOOP_MODEL_STREAM_EVENT_HANDLER
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

DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_ORIGINATOR = "pyccode"


class ModelClient(Protocol):
    async def complete(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
        ...


@dataclass(frozen=True, slots=True)
class ResponsesProviderConfig:
    model: str
    provider_name: str
    base_url: str
    api_key_env: str
    wire_api: str = "responses"
    query_params: dict[str, str] = field(default_factory=dict)
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    beta_features_header: str | None = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: str | Path = DEFAULT_CODEX_CONFIG_PATH,
        profile: str | None = None,
        model: str | None = None,
    ) -> "ResponsesProviderConfig":
        data = tomllib.loads(Path(config_path).read_text())
        selected = dict(data)
        if profile is not None:
            overrides = data.get("profiles", {}).get(profile)
            if overrides is None:
                raise ValueError(f"unknown Codex profile: {profile}")
            selected.update(overrides)

        provider_name = selected["model_provider"]
        provider = data["model_providers"][provider_name]
        wire_api = provider.get("wire_api", "responses")
        if wire_api != "responses":
            raise ValueError(f"unsupported wire_api for Python client: {wire_api}")

        api_key_env = provider.get("env_key")
        if not api_key_env:
            raise ValueError(
                f"provider {provider_name} does not define env_key in Codex config"
            )

        query_params = {
            str(key): str(value)
            for key, value in provider.get("query_params", {}).items()
        }
        features = selected.get("features", {})
        beta_features: list[str] = []
        if isinstance(features, dict) and features.get("guardian_approval") is True:
            beta_features.append("guardian_approval")
        return cls(
            model=model or selected["model"],
            provider_name=provider_name,
            base_url=provider["base_url"],
            api_key_env=api_key_env,
            wire_api=wire_api,
            query_params=query_params,
            reasoning_effort=selected.get("model_reasoning_effort"),
            reasoning_summary=selected.get("model_reasoning_summary"),
            verbosity=selected.get("model_verbosity"),
            beta_features_header=",".join(beta_features) or None,
        )

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise RuntimeError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        return value

    def with_overrides(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> "ResponsesProviderConfig":
        return replace(
            self,
            model=self.model if model is None else model,
            reasoning_effort=(
                self.reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
        )


class ResponsesApiError(RuntimeError):
    pass


class ResponsesModelClient:
    def __init__(
        self,
        config: ResponsesProviderConfig,
        timeout_seconds: float = 120.0,
        session_id: str | None = None,
        originator: str = DEFAULT_ORIGINATOR,
        user_agent: str | None = None,
    ) -> None:
        self._config = config
        self.model = config.model
        self._timeout_seconds = timeout_seconds
        self._session_id = session_id or uuid7_string()
        self._originator = originator
        self._user_agent = user_agent or f"{originator}/0.1.0"

    @classmethod
    def from_codex_config(
        cls,
        config_path: str | Path = DEFAULT_CODEX_CONFIG_PATH,
        profile: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        originator: str = DEFAULT_ORIGINATOR,
        user_agent: str | None = None,
    ) -> "ResponsesModelClient":
        config = ResponsesProviderConfig.from_codex_config(
            config_path=config_path,
            profile=profile,
            model=model,
        )
        return cls(
            config,
            timeout_seconds=timeout_seconds,
            originator=originator,
            user_agent=user_agent,
        )

    def responses_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/responses"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    def models_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/models"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    async def list_models(self) -> list[str]:
        return await asyncio.to_thread(self._list_models_sync)

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
        payload = self._build_payload(prompt)
        body = json.dumps(payload).encode("utf-8")
        prepared = requests.PreparedRequest()
        prepared.prepare(
            method="POST",
            url=self.responses_url(),
            headers=self._build_headers(),
            data=body,
        )
        try:
            with requests.Session() as session:
                settings = session.merge_environment_settings(
                    prepared.url,
                    proxies={},
                    stream=True,
                    verify=None,
                    cert=None,
                )
                verify = _requests_verify_setting()
                if verify is not None:
                    settings["verify"] = verify
                response = session.send(
                    prepared,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                    **settings,
                )
                with response:
                    if response.status_code >= 400:
                        raise ResponsesApiError(
                            f"responses request failed with status {response.status_code}: "
                            f"{response.text[:500]}"
                        )
                    return self._parse_stream(
                        response.iter_lines(chunk_size=1, decode_unicode=False),
                        event_handler,
                    )
        except requests.RequestException as exc:
            raise ResponsesApiError(f"responses request failed: {exc}") from exc

    def _list_models_sync(self) -> list[str]:
        prepared = requests.PreparedRequest()
        prepared.prepare(
            method="GET",
            url=self.models_url(),
            headers=self._build_model_list_headers(),
        )
        try:
            with requests.Session() as session:
                settings = session.merge_environment_settings(
                    prepared.url,
                    proxies={},
                    stream=False,
                    verify=None,
                    cert=None,
                )
                verify = _requests_verify_setting()
                if verify is not None:
                    settings["verify"] = verify
                response = session.send(
                    prepared,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                    **settings,
                )
                with response:
                    if response.status_code >= 400:
                        raise ResponsesApiError(
                            f"models request failed with status {response.status_code}: "
                            f"{response.text[:500]}"
                        )
                    payload = response.json()
        except requests.RequestException as exc:
            raise ResponsesApiError(f"models request failed: {exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise ResponsesApiError("models response is missing `data` list")
        models: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
        return models

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "authorization": f"Bearer {self._config.api_key()}",
            "x-client-request-id": self._session_id,
            "session_id": self._session_id,
            "originator": self._originator,
            "user-agent": self._user_agent,
        }
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        return headers

    def _build_model_list_headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self._config.api_key()}",
            "originator": self._originator,
            "user-agent": self._user_agent,
        }
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        return headers

    def _build_payload(self, prompt: Prompt) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "instructions": self._serialize_system(prompt.system),
            "input": self._serialize_messages(prompt),
            "tools": [self._serialize_tool(tool) for tool in prompt.tools],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self._session_id,
        }

        reasoning: dict[str, str] = {}
        if self._config.reasoning_effort is not None:
            reasoning["effort"] = self._config.reasoning_effort
        if self._config.reasoning_summary is not None:
            reasoning["summary"] = self._config.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning

        if self._config.verbosity is not None:
            payload["text"] = {"verbosity": self._config.verbosity}

        return payload

    def _serialize_system(
        self,
        system: str | tuple[SystemTextBlock, ...],
    ) -> str:
        if isinstance(system, str):
            return system
        return "\n\n".join(block.text for block in system if block.text.strip())

    def _serialize_messages(self, prompt: Prompt) -> list[JSONDict]:
        merged_messages = self._merge_user_reminders(prompt.messages, prompt.user_reminders)
        serialized: list[JSONDict] = []
        for message in merged_messages:
            serialized.extend(self._serialize_message(message))
        return serialized

    def _merge_user_reminders(
        self,
        messages: tuple[ConversationMessage, ...],
        reminders: tuple[str, ...],
    ) -> tuple[ConversationMessage, ...]:
        if not reminders:
            return messages

        reminder_blocks = tuple(TextBlock(text=reminder) for reminder in reminders)
        merged = list(messages)
        if merged and merged[0].role == "user":
            first = merged[0]
            merged[0] = ConversationMessage(
                role="user",
                content=reminder_blocks + first.content,
            )
            return tuple(merged)

        merged.insert(0, ConversationMessage(role="user", content=reminder_blocks))
        return tuple(merged)

    def _serialize_message(self, message: ConversationMessage) -> list[JSONDict]:
        if message.role == "user":
            return self._serialize_user_message(message)
        return self._serialize_assistant_message(message)

    def _serialize_user_message(self, message: ConversationMessage) -> list[JSONDict]:
        items: list[JSONDict] = []
        content_items: list[JSONDict] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                content_items.append({"type": "input_text", "text": block.text})
                continue
            if content_items:
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": content_items,
                    }
                )
                content_items = []
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": block.content,
                }
            )
        if content_items:
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": content_items,
                }
            )
        return items

    def _serialize_assistant_message(
        self,
        message: ConversationMessage,
    ) -> list[JSONDict]:
        items: list[JSONDict] = []
        content_items: list[JSONDict] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                content_items.append({"type": "output_text", "text": block.text})
                continue
            if content_items:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content_items,
                    }
                )
                content_items = []
            if isinstance(block, ThinkingBlock):
                if block.raw_payload is not None:
                    items.append(deepcopy(block.raw_payload))
                continue
            items.append(
                {
                    "type": "function_call",
                    "name": block.name,
                    "arguments": json.dumps(
                        block.input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "call_id": block.id,
                }
            )
        if content_items:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": content_items,
                }
            )
        return items

    @staticmethod
    def _serialize_tool(tool) -> JSONDict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": False,
        }

    def _parse_stream(
        self,
        response,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
        blocks: list[TextBlock | ToolUseBlock | ThinkingBlock] = []
        saw_completed = False

        for event_name, data in self._iter_sse_events(response):
            if not data:
                continue
            payload = json.loads(data)
            event_type = payload.get("type", event_name)

            if event_type == "response.output_text.delta":
                delta = str(payload.get("delta", ""))
                event_handler(
                    ModelStreamEvent(
                        kind="assistant_text_delta",
                        payload={"text": delta},
                    )
                )
                continue

            if event_type == "response.output_item.done":
                item_payload = payload.get("item", {})
                if not isinstance(item_payload, dict):
                    continue
                item_type = item_payload.get("type")
                if item_type == "reasoning":
                    blocks.append(
                        ThinkingBlock(
                            text=_reasoning_summary_text(item_payload),
                            raw_payload=dict(item_payload),
                        )
                    )
                    continue
                if item_type == "message" and item_payload.get("role") == "assistant":
                    for part in item_payload.get("content", []):
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "output_text"
                        ):
                            blocks.append(TextBlock(text=str(part.get("text", ""))))
                    continue
                if item_type == "function_call":
                    raw_arguments = str(item_payload.get("arguments", "") or "{}")
                    try:
                        parsed_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError as exc:
                        raise ResponsesApiError(
                            f"function call arguments failed to decode: {exc}"
                        ) from exc
                    if not isinstance(parsed_arguments, dict):
                        raise ResponsesApiError(
                            "function call arguments must decode to an object"
                        )
                    tool_use = ToolUseBlock(
                        id=str(item_payload["call_id"]),
                        name=str(item_payload["name"]),
                        input=parsed_arguments,
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
                    continue
                continue

            if event_type == "response.completed":
                saw_completed = True
                break

            if event_type == "response.failed":
                error = payload.get("response", {}).get("error") or {}
                message = error.get("message") or "responses stream failed"
                raise ResponsesApiError(str(message))

        if not saw_completed:
            raise ResponsesApiError("responses stream ended before response.completed")

        return ModelResponse(
            message=ConversationMessage(role="assistant", content=tuple(blocks)),
            stop_reason=None,
            raw={"content": [block.serialize() for block in blocks]},
        )

    @staticmethod
    def _iter_sse_events(response):
        event_name: str | None = None
        data_lines: list[str] = []

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    yield event_name or "message", "\n".join(data_lines)
                event_name = None
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].lstrip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        if data_lines:
            yield event_name or "message", "\n".join(data_lines)


def _reasoning_summary_text(payload: JSONDict) -> str:
    summary = payload.get("summary")
    if isinstance(summary, list):
        texts = [
            str(item.get("text", ""))
            for item in summary
            if isinstance(item, dict) and item.get("type") == "summary_text"
        ]
        return "\n".join(text for text in texts if text).strip()
    if isinstance(summary, str):
        return summary
    return ""


def _requests_verify_setting() -> str | bool | None:
    for env_name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None

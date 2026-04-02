from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


JSONValue: TypeAlias = Any
JSONDict: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"

    def serialize(self) -> JSONDict:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class SystemTextBlock:
    text: str
    cache_control: JSONDict | None = None
    type: Literal["text"] = "text"

    def serialize(self) -> JSONDict:
        payload: JSONDict = {"type": self.type, "text": self.text}
        if self.cache_control is not None:
            payload["cache_control"] = self.cache_control
        return payload


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    text: str
    signature: str | None = None
    type: Literal["thinking"] = "thinking"

    def serialize(self) -> JSONDict:
        payload: JSONDict = {"type": "thinking", "thinking": self.text}
        if self.signature:
            payload["signature"] = self.signature
        return payload


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: JSONDict
    type: Literal["tool_use"] = "tool_use"

    def serialize(self) -> JSONDict:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"

    def serialize(self) -> JSONDict:
        payload: JSONDict = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            payload["is_error"] = True
        return payload


ContentBlock: TypeAlias = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]

    @classmethod
    def user_text(cls, text: str) -> "ConversationMessage":
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def assistant_text(cls, text: str) -> "ConversationMessage":
        return cls(role="assistant", content=(TextBlock(text=text),))

    def serialize(self, include_thinking: bool = False) -> JSONDict:
        content: list[JSONDict] = []
        for block in self.content:
            if isinstance(block, ThinkingBlock) and not include_thinking:
                continue
            content.append(block.serialize())
        if not content:
            raise ValueError("message has no serializable content")
        return {"role": self.role, "content": content}

    def has_serializable_content(self, include_thinking: bool = False) -> bool:
        return any(
            include_thinking or not isinstance(block, ThinkingBlock)
            for block in self.content
        )

    def text_content(self) -> str:
        return "\n".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        ).strip()

    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(
            block for block in self.content if isinstance(block, ToolUseBlock)
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JSONDict
    supports_parallel: bool = True

    def serialize(self) -> JSONDict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str | tuple[SystemTextBlock, ...]
    messages: tuple[ConversationMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int = 32000
    temperature: float | None = None
    user_reminders: tuple[str, ...] = ()
    metadata: JSONDict | None = None
    thinking: JSONDict | None = None
    context_management: JSONDict | None = None
    stream: bool = True


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    kind: str
    payload: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    message: ConversationMessage
    stop_reason: str | None = None
    raw: JSONDict | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: str
    turn_id: str
    payload: JSONDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: str
    output_text: str | None
    iterations: int
    response: ModelResponse
    history: tuple[ConversationMessage, ...]


@dataclass(frozen=True, slots=True)
class UserTurnOp:
    texts: list[str]


@dataclass(frozen=True, slots=True)
class ShutdownOp:
    pass


Operation: TypeAlias = UserTurnOp | ShutdownOp


@dataclass(frozen=True, slots=True)
class Submission:
    id: str
    op: Operation

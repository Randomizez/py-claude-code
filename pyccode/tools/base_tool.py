from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..protocol import ConversationMessage, JSONDict, ToolResultBlock, ToolSpec, ToolUseBlock


@dataclass(frozen=True, slots=True)
class ToolContext:
    turn_id: str
    history: tuple[ConversationMessage, ...]
    prior_results: tuple[ToolResultBlock, ...] = ()
    cwd: Path = field(default_factory=Path.cwd)


class BaseTool:
    name: str = ""
    description: str = ""
    input_schema: JSONDict = {"type": "object", "properties": {}}
    supports_parallel: bool = True

    async def run(self, context: ToolContext, args: JSONDict):
        raise NotImplementedError

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            supports_parallel=self.supports_parallel,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def model_visible_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec() for tool in self._tools.values())

    def supports_parallel(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.supports_parallel)

    async def execute(
        self,
        call: ToolUseBlock,
        context: ToolContext,
    ) -> ToolResultBlock:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"No such tool: {call.name}",
                is_error=True,
            )
        try:
            output = await tool.run(context, call.input)
        except Exception as exc:
            return ToolResultBlock(
                tool_use_id=call.id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResultBlock(
            tool_use_id=call.id,
            content=_stringify_output(output),
            is_error=False,
        )


def _stringify_output(output) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)

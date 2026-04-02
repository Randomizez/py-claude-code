from __future__ import annotations

from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import (
    DEFAULT_EXEC_YIELD_TIME_MS,
    DEFAULT_LOGIN,
    DEFAULT_TTY,
    UnifiedExecManager,
)


class ExecCommandTool(BaseTool):
    name = "exec_command"
    description = "Start a session-backed shell command and return output or a session ID."
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "workdir": {"type": "string"},
            "shell": {"type": "string"},
            "login": {"type": "boolean"},
            "tty": {"type": "boolean"},
            "yield_time_ms": {"type": "integer"},
            "max_output_tokens": {"type": "integer"},
        },
        "required": ["cmd"],
    }
    supports_parallel = False

    def __init__(self, manager: UnifiedExecManager) -> None:
        self._manager = manager

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        del context
        cmd = str(args.get("cmd", "")).strip()
        if not cmd:
            return "Error: `cmd` is required."

        return await self._manager.exec_command(
            cmd=cmd,
            workdir=self._optional_string(args, "workdir"),
            shell=self._optional_string(args, "shell"),
            login=bool(args.get("login", DEFAULT_LOGIN)),
            tty=bool(args.get("tty", DEFAULT_TTY)),
            yield_time_ms=int(args.get("yield_time_ms", DEFAULT_EXEC_YIELD_TIME_MS)),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _optional_string(self, args: dict[str, object], key: str) -> str | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)

    def _optional_int(self, args: dict[str, object], key: str) -> int | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)

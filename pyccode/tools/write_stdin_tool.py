from __future__ import annotations

from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import DEFAULT_WRITE_STDIN_YIELD_TIME_MS, UnifiedExecManager


class WriteStdinTool(BaseTool):
    name = "write_stdin"
    description = "Write to an existing exec session or poll it for more output."
    input_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "integer"},
            "chars": {"type": "string"},
            "yield_time_ms": {"type": "integer"},
            "max_output_tokens": {"type": "integer"},
        },
        "required": ["session_id"],
    }
    supports_parallel = False

    def __init__(self, manager: UnifiedExecManager) -> None:
        self._manager = manager

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        del context
        session_id = args.get("session_id")
        if session_id is None:
            return "Error: `session_id` is required."

        return await self._manager.write_stdin(
            session_id=int(session_id),
            chars=str(args.get("chars", "")),
            yield_time_ms=int(
                args.get("yield_time_ms", DEFAULT_WRITE_STDIN_YIELD_TIME_MS)
            ),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _optional_int(self, args: dict[str, object], key: str) -> int | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)

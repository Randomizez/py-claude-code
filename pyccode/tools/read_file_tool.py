from __future__ import annotations

from pathlib import Path

from .base_tool import BaseTool, ToolContext


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a text file from disk, optionally slicing by line number."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }
    supports_parallel = True

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        path = _resolve_path(context.cwd, str(args["path"]))
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        start = int(args.get("start_line", 1))
        end = int(args.get("end_line", len(lines)))
        if start < 1:
            start = 1
        if end < start:
            end = start
        selected = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start)]
        header = f"{path} ({start}-{min(end, len(lines))})"
        return header + "\n" + "\n".join(numbered)


def _resolve_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()

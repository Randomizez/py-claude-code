from __future__ import annotations

from pathlib import Path

from .base_tool import BaseTool, ToolContext


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories under a path."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_entries": {"type": "integer", "minimum": 1},
        },
    }
    supports_parallel = True

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        raw_path = str(args.get("path", "."))
        max_entries = int(args.get("max_entries", 200))
        path = _resolve_path(context.cwd, raw_path)
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        lines = []
        for entry in entries[:max_entries]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        if len(entries) > max_entries:
            lines.append(f"... truncated after {max_entries} entries")
        return f"{path}\n" + "\n".join(lines)


def _resolve_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()

from __future__ import annotations

from pathlib import Path
import re

from .base_tool import BaseTool, ToolContext

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = "Search for a regex pattern inside text files."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "max_matches": {"type": "integer", "minimum": 1},
        },
        "required": ["pattern"],
    }
    supports_parallel = True

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        pattern = re.compile(str(args["pattern"]))
        root = _resolve_path(context.cwd, str(args.get("path", ".")))
        max_matches = int(args.get("max_matches", 100))
        matches: list[str] = []
        for path in _iter_files(root):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{path}:{lineno}:{line}")
                    if len(matches) >= max_matches:
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "No matches found."


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _resolve_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()

from __future__ import annotations

import asyncio
import glob
from pathlib import Path
import re

from .base_tool import ToolRegistry
from .placeholder_claude_tool import (
    AgentTool,
    AskUserQuestionTool,
    BashTool,
    CLAUDE_PLACEHOLDER_TOOL_CLASSES,
    EditTool,
    GlobTool,
    GrepTool,
    PlaceholderClaudeTool,
    ReadTool,
    WriteTool,
)

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
DEFAULT_READ_LIMIT = 2000
DEFAULT_GLOB_LIMIT = 200
DEFAULT_GREP_HEAD_LIMIT = 250
DEFAULT_BASH_TIMEOUT_MS = 120_000
GREP_TYPE_SUFFIXES = {
    "js": {".js", ".mjs", ".cjs"},
    "ts": {".ts"},
    "tsx": {".tsx"},
    "py": {".py"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "json": {".json"},
    "md": {".md"},
    "yaml": {".yaml", ".yml"},
    "sh": {".sh", ".bash"},
}


class ReadRuntimeTool(ReadTool):
    async def run(self, context, args):
        path = _require_absolute_path(args["file_path"])
        if path.is_dir():
            raise IsADirectoryError(f"Read only supports files: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".pdf"} and args.get("pages"):
            raise NotImplementedError("PDF page extraction is not implemented yet")

        text = path.read_text(encoding="utf-8", errors="replace")
        if text == "":
            return "<system-reminder>Warning: the file exists but is empty.</system-reminder>"

        lines = text.splitlines()
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", DEFAULT_READ_LIMIT))
        selected = lines[offset : offset + limit]
        numbered = [
            f"{line_number:6}\t{line}"
            for line_number, line in enumerate(selected, start=offset + 1)
        ]
        return "\n".join(numbered)


class WriteRuntimeTool(WriteTool):
    async def run(self, context, args):
        del context
        path = _require_absolute_path(args["file_path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"


class EditRuntimeTool(EditTool):
    async def run(self, context, args):
        del context
        path = _require_absolute_path(args["file_path"])
        old = str(args["old_string"])
        new = str(args["new_string"])
        replace_all = bool(args.get("replace_all", False))
        if old == new:
            raise ValueError("old_string and new_string must be different")
        text = path.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            raise ValueError("old_string not found in file")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                "old_string is not unique; provide more context or use replace_all"
            )
        updated = text.replace(old, new, -1 if replace_all else 1)
        path.write_text(updated, encoding="utf-8")
        replaced = occurrences if replace_all else 1
        return f"Updated {path} with {replaced} replacement(s)"


class GlobRuntimeTool(GlobTool):
    async def run(self, context, args):
        root = _resolve_search_root(context.cwd, args.get("path"))
        pattern = str(args["pattern"])
        matches = []
        search_pattern = pattern if Path(pattern).is_absolute() else str(root / pattern)
        for raw in glob.glob(search_pattern, recursive=True):
            path = Path(raw)
            if _should_skip(path):
                continue
            matches.append(path.resolve())
        ordered = sorted(
            dict.fromkeys(matches),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        if not ordered:
            return "No files found."
        return "\n".join(str(path) for path in ordered[:DEFAULT_GLOB_LIMIT])


class GrepRuntimeTool(GrepTool):
    async def run(self, context, args):
        root = _resolve_search_root(context.cwd, args.get("path"))
        regex = _compile_grep_pattern(args)
        output_mode = str(args.get("output_mode", "files_with_matches"))
        head_limit = int(args.get("head_limit", DEFAULT_GREP_HEAD_LIMIT))
        offset = int(args.get("offset", 0))
        entries = self._collect_entries(root, regex, output_mode, args)
        if offset:
            entries = entries[offset:]
        if head_limit != 0:
            entries = entries[:head_limit]
        return "\n".join(entries) if entries else "No matches found."

    def _collect_entries(self, root, regex, output_mode, args):
        files = list(_iter_candidate_files(root, args.get("glob"), args.get("type")))
        if output_mode == "count":
            return _grep_count_entries(files, regex, bool(args.get("multiline", False)))
        if output_mode == "content":
            return _grep_content_entries(files, regex, args)
        return _grep_file_entries(files, regex, bool(args.get("multiline", False)))


class BashRuntimeTool(BashTool):
    supports_parallel = False

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._cwd = Path(cwd or Path.cwd()).resolve()

    async def run(self, context, args):
        del context
        command = str(args["command"]).strip()
        timeout_ms = int(args.get("timeout", DEFAULT_BASH_TIMEOUT_MS))
        if bool(args.get("run_in_background", False)):
            raise NotImplementedError("run_in_background is not implemented yet")
        effective_cwd, effective_command = self._resolve_cwd_for_command(command)
        if effective_command == "":
            self._cwd = effective_cwd
            return f"Working directory: {self._cwd}\nExit code: 0"

        process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            effective_command,
            cwd=str(effective_cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=max(timeout_ms, 1) / 1000.0,
            )
            timed_out = False
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True

        stdout = stdout_bytes.decode("utf-8", errors="replace").rstrip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").rstrip()
        self._cwd = effective_cwd
        pieces = [f"Working directory: {self._cwd}"]
        if timed_out:
            pieces.append(f"Timeout: exceeded {timeout_ms} ms")
        else:
            pieces.append(f"Exit code: {process.returncode}")
        if stdout:
            pieces.extend(["Stdout:", stdout])
        if stderr:
            pieces.extend(["Stderr:", stderr])
        return "\n".join(pieces)

    def _resolve_cwd_for_command(self, command: str) -> tuple[Path, str]:
        match = re.match(r"^\s*cd\s+((?:\"[^\"]+\"|'[^']+'|[^&;]+?))(?:\s*&&\s*(.*))?\s*$", command)
        if not match:
            return self._cwd, command

        raw_target = match.group(1).strip().strip("'\"")
        remainder = (match.group(2) or "").strip()
        target = Path(raw_target)
        if not target.is_absolute():
            target = (self._cwd / target).resolve()
        else:
            target = target.resolve()
        return target, remainder


IMPLEMENTED_CLAUDE_TOOL_CLASSES = {
    "Bash": BashRuntimeTool,
    "Edit": EditRuntimeTool,
    "Glob": GlobRuntimeTool,
    "Grep": GrepRuntimeTool,
    "Read": ReadRuntimeTool,
    "Write": WriteRuntimeTool,
}


def build_claude_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_cls in CLAUDE_PLACEHOLDER_TOOL_CLASSES:
        runtime_cls = IMPLEMENTED_CLAUDE_TOOL_CLASSES.get(tool_cls.name, tool_cls)
        registry.register(runtime_cls())
    return registry


def _require_absolute_path(raw_path: object) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {path}")
    return path.resolve()


def _resolve_search_root(cwd: Path, raw_path: object) -> Path:
    if raw_path in (None, "", "undefined", "null"):
        return cwd.resolve()
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _compile_grep_pattern(args):
    flags = 0
    if bool(args.get("-i", False)):
        flags |= re.IGNORECASE
    if bool(args.get("multiline", False)):
        flags |= re.MULTILINE | re.DOTALL
    return re.compile(str(args["pattern"]), flags)


def _iter_candidate_files(root: Path, glob_pattern: object, type_name: object):
    if root.is_file():
        candidates = [root]
    elif glob_pattern:
        pattern = str(glob_pattern)
        search_pattern = pattern if Path(pattern).is_absolute() else str(root / pattern)
        candidates = [Path(path) for path in glob.glob(search_pattern, recursive=True)]
    else:
        candidates = list(root.rglob("*"))

    suffixes = GREP_TYPE_SUFFIXES.get(str(type_name), None) if type_name else None
    for path in candidates:
        resolved = path.resolve()
        if _should_skip(resolved) or not resolved.is_file():
            continue
        if suffixes is not None and resolved.suffix not in suffixes:
            continue
        yield resolved


def _grep_file_entries(files, regex, multiline: bool):
    entries: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        matched = bool(regex.search(text)) if multiline else any(
            regex.search(line) for line in text.splitlines()
        )
        if matched:
            entries.append(str(path))
    return entries


def _grep_count_entries(files, regex, multiline: bool):
    entries: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if multiline:
            count = len(list(regex.finditer(text)))
        else:
            count = sum(1 for line in text.splitlines() if regex.search(line))
        if count:
            entries.append(f"{path}:{count}")
    return entries


def _grep_content_entries(files, regex, args):
    entries: list[str] = []
    before = int(args.get("-B", args.get("context", args.get("-C", 0))))
    after = int(args.get("-A", args.get("context", args.get("-C", 0))))
    show_numbers = bool(args.get("-n", True))
    multiline = bool(args.get("multiline", False))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if multiline:
            if regex.search(text):
                entries.append(str(path))
            continue
        lines = text.splitlines()
        selected_indexes: list[int] = []
        for index, line in enumerate(lines):
            if regex.search(line):
                start = max(0, index - before)
                end = min(len(lines), index + after + 1)
                selected_indexes.extend(range(start, end))
        for index in sorted(dict.fromkeys(selected_indexes)):
            prefix = f"{path}:{index + 1}:" if show_numbers else f"{path}:"
            entries.append(f"{prefix}{lines[index]}")
    return entries

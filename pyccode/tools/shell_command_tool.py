from __future__ import annotations

import asyncio
from pathlib import Path

from .base_tool import BaseTool, ToolContext

DEFAULT_SHELL_TIMEOUT_MS = 30_000
MAX_OUTPUT_CHARS = 12_000


class ShellCommandTool(BaseTool):
    name = "shell_command"
    description = "Run a shell command string and return its output."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "workdir": {"type": "string"},
            "timeout_ms": {"type": "integer"},
            "login": {"type": "boolean"},
        },
        "required": ["command"],
    }
    supports_parallel = False

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._working_directory = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: ToolContext, args: dict[str, object]) -> str:
        del context
        command = str(args.get("command", "")).strip()
        timeout_ms = int(args.get("timeout_ms", DEFAULT_SHELL_TIMEOUT_MS))
        if not command:
            return "Error: `command` is required."

        login = bool(args.get("login", True))
        working_directory = self._resolve_workdir(args.get("workdir"))
        shell_args = ["bash", "-lc" if login else "-c", command]

        process = await asyncio.create_subprocess_exec(
            *shell_args,
            cwd=str(working_directory),
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

        stdout = self._clip_output(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = self._clip_output(stderr_bytes.decode("utf-8", errors="replace"))
        pieces = [f"Working directory: {working_directory}"]

        if timed_out:
            pieces.append(f"Timeout: exceeded {timeout_ms} ms")
        else:
            pieces.append(f"Exit code: {process.returncode}")

        if stdout:
            pieces.append("Stdout:")
            pieces.append(stdout)

        if stderr:
            pieces.append("Stderr:")
            pieces.append(stderr)

        return "\n".join(pieces)

    def _resolve_workdir(self, workdir_arg: object) -> Path:
        if workdir_arg in (None, ""):
            return self._working_directory
        workdir = Path(str(workdir_arg))
        if not workdir.is_absolute():
            workdir = self._working_directory / workdir
        return workdir.resolve()

    def _clip_output(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        return text[:MAX_OUTPUT_CHARS] + "\n...[truncated]..."

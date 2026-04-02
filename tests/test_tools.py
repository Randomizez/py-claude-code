from __future__ import annotations

import re

from pyccode.tools import ApplyPatchTool, ExecCommandTool, ShellCommandTool, UnifiedExecManager, WriteStdinTool
from pyccode.tools.base_tool import ToolContext


def _context(tmp_path) -> ToolContext:
    return ToolContext(turn_id="turn-test", history=(), cwd=tmp_path)


async def test_shell_command_tool_runs_command(tmp_path) -> None:
    tool = ShellCommandTool(cwd=tmp_path)

    result = await tool.run(
        _context(tmp_path),
        {"command": "printf 'hello\\n'", "timeout_ms": 1000},
    )

    assert "Exit code: 0" in result
    assert "Stdout:" in result
    assert "hello" in result


async def test_apply_patch_tool_updates_file(tmp_path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = ApplyPatchTool(cwd=tmp_path)
    patch = """*** Begin Patch
*** Update File: note.txt
@@
 alpha
-beta
+gamma
*** End Patch
"""

    result = await tool.run(_context(tmp_path), {"patch": patch})

    assert "Exit code: 0" in result
    assert "M note.txt" in result
    assert path.read_text(encoding="utf-8") == "alpha\ngamma\n"


async def test_exec_command_and_write_stdin_round_trip(tmp_path) -> None:
    manager = UnifiedExecManager(cwd=tmp_path)
    exec_tool = ExecCommandTool(manager)
    write_tool = WriteStdinTool(manager)
    command = (
        "python3 -u -c \"import sys; print('ready'); sys.stdout.flush(); "
        "line=sys.stdin.readline().strip(); print('echo:' + line)\""
    )

    first = await exec_tool.run(
        _context(tmp_path),
        {"cmd": command, "yield_time_ms": 100},
    )
    match = re.search(r"session ID (\d+)", first)
    assert match is not None, first

    second = await write_tool.run(
        _context(tmp_path),
        {
            "session_id": int(match.group(1)),
            "chars": "hello\n",
            "yield_time_ms": 1000,
        },
    )

    assert "Process exited with code 0" in second
    assert "echo:hello" in second

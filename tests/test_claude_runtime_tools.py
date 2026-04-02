from __future__ import annotations

from pathlib import Path

from pyccode.protocol import ToolUseBlock
from pyccode.tools import build_default_tool_registry
from pyccode.tools.base_tool import ToolContext


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(turn_id="turn-test", history=(), cwd=tmp_path)


async def test_read_write_and_edit_tools_work_with_default_registry(tmp_path) -> None:
    registry = build_default_tool_registry()
    target = (tmp_path / "note.txt").resolve()

    write_result = await registry.execute(
        ToolUseBlock(id="toolu_write", name="Write", input={"file_path": str(target), "content": "alpha\nbeta\n"}),
        _context(tmp_path),
    )
    assert write_result.is_error is False
    assert target.read_text() == "alpha\nbeta\n"

    read_result = await registry.execute(
        ToolUseBlock(id="toolu_read", name="Read", input={"file_path": str(target)}),
        _context(tmp_path),
    )
    assert read_result.is_error is False
    assert "1\talpha" in read_result.content
    assert "2\tbeta" in read_result.content

    edit_result = await registry.execute(
        ToolUseBlock(
            id="toolu_edit",
            name="Edit",
            input={
                "file_path": str(target),
                "old_string": "beta",
                "new_string": "gamma",
            },
        ),
        _context(tmp_path),
    )
    assert edit_result.is_error is False
    assert target.read_text() == "alpha\ngamma\n"


async def test_glob_and_grep_tools_work_with_default_registry(tmp_path) -> None:
    registry = build_default_tool_registry()
    (tmp_path / "a.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello world\nbye\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("HELLO\n", encoding="utf-8")

    glob_result = await registry.execute(
        ToolUseBlock(
            id="toolu_glob",
            name="Glob",
            input={"pattern": "**/*.py", "path": str(tmp_path)},
        ),
        _context(tmp_path),
    )
    assert glob_result.is_error is False
    assert str((tmp_path / "a.py").resolve()) in glob_result.content
    assert str((tmp_path / "sub" / "c.py").resolve()) in glob_result.content

    grep_result = await registry.execute(
        ToolUseBlock(
            id="toolu_grep",
            name="Grep",
            input={
                "pattern": "hello",
                "path": str(tmp_path),
                "output_mode": "content",
                "-i": True,
            },
        ),
        _context(tmp_path),
    )
    assert grep_result.is_error is False
    assert str((tmp_path / "b.txt").resolve()) in grep_result.content
    assert str((tmp_path / "sub" / "c.py").resolve()) in grep_result.content


async def test_bash_tool_runs_commands_and_persists_cwd(tmp_path) -> None:
    registry = build_default_tool_registry()
    subdir = (tmp_path / "subdir").resolve()
    subdir.mkdir()
    context = _context(tmp_path)

    cd_result = await registry.execute(
        ToolUseBlock(
            id="toolu_bash_cd",
            name="Bash",
            input={"command": f'cd "{subdir}"'},
        ),
        context,
    )
    assert cd_result.is_error is False
    assert str(subdir) in cd_result.content

    pwd_result = await registry.execute(
        ToolUseBlock(
            id="toolu_bash_pwd",
            name="Bash",
            input={"command": "pwd"},
        ),
        context,
    )
    assert pwd_result.is_error is False
    assert str(subdir) in pwd_result.content

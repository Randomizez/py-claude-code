from __future__ import annotations

import json
from pathlib import Path

from pyccode.protocol import ToolUseBlock
from pyccode.tools import build_default_tool_registry
from pyccode.tools.base_tool import ToolContext


def test_default_tool_registry_exposes_captured_claude_nonbare_tools() -> None:
    capture_path = (
        Path(__file__).resolve().parents[1]
        / ".debug"
        / "nonbare_oauth_capture"
        / "001_POST_v1_messages.json"
    )
    capture = json.loads(capture_path.read_text())
    expected = capture["body"]["tools"]
    registry = build_default_tool_registry()
    specs = registry.model_visible_specs()

    assert len(specs) == len(expected) == 22
    assert [spec.serialize() for spec in specs] == expected

    by_name = {spec.name: spec for spec in specs}
    assert "Agent" in by_name
    assert "Write" in by_name
    assert "Skill" in by_name
    assert (
        by_name["Read"].input_schema["properties"]["file_path"]["description"]
        == expected[14]["input_schema"]["properties"]["file_path"]["description"]
    )


async def test_placeholder_claude_tool_returns_not_implemented_payload() -> None:
    registry = build_default_tool_registry()
    tool = registry.model_visible_specs()[0]
    result = await registry.execute(
        call=ToolUseBlock(id="toolu_1", name=tool.name, input={"task": "demo"}),
        context=ToolContext(turn_id="turn-1", history=(), cwd=Path.cwd()),
    )

    assert result.is_error is False
    assert '"status": "not_implemented"' in result.content
    assert f'"tool": "{tool.name}"' in result.content

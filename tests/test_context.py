from __future__ import annotations

import json

from pyccode.context import ContextManager
from pyccode.protocol import ConversationMessage, SystemTextBlock


def test_context_manager_builds_claude_style_prompt(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project-specific instructions.", encoding="utf-8")
    manager = ContextManager(cwd=tmp_path)

    prompt = manager.build_prompt(
        [ConversationMessage.user_text("say hi")],
        [],
        turn_id="turn-123",
    )

    assert isinstance(prompt.system, tuple)
    assert all(isinstance(block, SystemTextBlock) for block in prompt.system)
    assert prompt.system[0].text == "x-anthropic-billing-header: cc_version=2.1.89.4fa; cc_entrypoint=sdk-cli; cch=00000;"
    assert prompt.system[1].text == "You are a Claude agent, built on Anthropic's Claude Agent SDK."
    assert "You are an interactive agent that helps users with software engineering tasks." in prompt.system[2].text
    assert "Session-specific guidance" in prompt.system[3].text
    assert len(prompt.user_reminders) == 2
    assert "The following skills are available for use with the Skill tool" in prompt.user_reminders[0]
    assert "Today's date is" in prompt.user_reminders[1]
    assert "# currentDate" in prompt.user_reminders[1]
    assert prompt.stream is True
    assert len(prompt.system) == 4
    assert prompt.system[0].text.startswith("x-anthropic-billing-header:")
    assert prompt.system[1].text == "You are a Claude agent, built on Anthropic's Claude Agent SDK."
    assert prompt.system[2].cache_control == {"type": "ephemeral", "scope": "global"}
    assert prompt.thinking == {"type": "enabled", "budget_tokens": 31999}
    assert prompt.context_management == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }

    user_id = json.loads(prompt.metadata["user_id"])
    assert user_id["device_id"] == "pyccode-local"
    assert user_id["session_id"] == "turn-123"

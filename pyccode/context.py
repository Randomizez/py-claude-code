from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

from .protocol import ConversationMessage, Prompt, SystemTextBlock, ToolSpec

DEFAULT_BASE_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "default_base_instructions.md"
)
CLAUDE_SYSTEM_BLOCK_2_PATH = (
    Path(__file__).resolve().parent / "prompts" / "claude_system_block_2.md"
)
CLAUDE_SYSTEM_BLOCK_3_PATH = (
    Path(__file__).resolve().parent / "prompts" / "claude_system_block_3.md"
)


@dataclass(frozen=True, slots=True)
class ContextConfig:
    base_instructions_override: str | None = None
    include_skills_reminder: bool = True
    include_environment_context: bool = True
    include_project_doc: bool = True
    project_doc_filename: str = "AGENTS.md"
    project_doc_max_bytes: int = 32_000
    max_tokens: int = 32000
    temperature: float | None = None


class ContextManager:
    _default_skill_descriptors = (
        (
            "update-config",
            'Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so m…',
        ),
        (
            "simplify",
            "Review changed code for reuse, quality, and efficiency, then fix any issues found.",
        ),
        (
            "loop",
            'Run a prompt or slash command on a recurring interval (e.g. /loop 5m /foo, defaults to 10m) - When the user wants to set up a recurring task, poll for status, or run something repeatedly on an interval (e.g. "check the deploy every 5 minutes", "keep…',
        ),
        (
            "claude-api",
            "Build apps with the Claude API or Anthropic SDK.\nTRIGGER when: code imports `anthropic`/`@anthropic-ai/sdk`/`claude_agent_sdk`, or user asks to use Claude API, Anthropic SDKs, or Agent SDK.\nDO NOT TRIGGER when: code imports `openai`/other AI SDK, ge…",
        ),
    )

    def __init__(
        self,
        cwd: str | Path | None = None,
        config: ContextConfig | None = None,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self._config = config or ContextConfig()
        self._default_base_instructions = (
            DEFAULT_BASE_INSTRUCTIONS_PATH.read_text().strip()
        )

    @property
    def config(self) -> ContextConfig:
        return self._config

    def with_config(self, config: ContextConfig) -> "ContextManager":
        return ContextManager(cwd=self.cwd, config=config)

    def resolve_base_instructions(self) -> str:
        if self._config.base_instructions_override is not None:
            return self._config.base_instructions_override.strip()
        return self._default_base_instructions

    def build_prompt(
        self,
        history: tuple[ConversationMessage, ...] | list[ConversationMessage],
        tools: tuple[ToolSpec, ...] | list[ToolSpec],
        turn_id: str | None = None,
    ) -> Prompt:
        serializable_history = tuple(
            message
            for message in history
            if message.has_serializable_content(include_thinking=False)
        )
        return Prompt(
            system=self._build_system_prompt_blocks(),
            messages=serializable_history,
            tools=tuple(tools),
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            user_reminders=self._build_user_reminders(),
            metadata=self._build_metadata(turn_id),
            thinking=self._build_thinking_config(),
            context_management=self._build_context_management(),
            stream=True,
        )

    def _build_system_prompt_blocks(self) -> tuple[SystemTextBlock, ...]:
        blocks = [
            SystemTextBlock(text=self._billing_header_block()),
            SystemTextBlock(
                text="You are a Claude agent, built on Anthropic's Claude Agent SDK."
            ),
            SystemTextBlock(
                text=self._interactive_agent_block(),
                cache_control={"type": "ephemeral", "scope": "global"},
            ),
            SystemTextBlock(text=self._session_guidance_block()),
        ]
        return tuple(block for block in blocks if block.text.strip())

    def _environment_context_block(self) -> str:
        tz_name = time.tzname[0] if time.tzname else "UTC"
        current_date = datetime.now().date().isoformat()
        return (
            "<environment_context>\n"
            f"cwd: {self.cwd}\n"
            f"current_date: {current_date}\n"
            f"timezone: {tz_name}\n"
            "</environment_context>"
        )

    def _session_guidance_block(self) -> str:
        return CLAUDE_SYSTEM_BLOCK_3_PATH.read_text()

    def _billing_header_block(self) -> str:
        return "x-anthropic-billing-header: cc_version=2.1.89.4fa; cc_entrypoint=sdk-cli; cch=00000;"

    def _interactive_agent_block(self) -> str:
        return CLAUDE_SYSTEM_BLOCK_2_PATH.read_text()

    def _build_user_reminders(self) -> tuple[str, ...]:
        reminders: list[str] = []
        if self._config.include_skills_reminder:
            reminders.append(self._skills_reminder_block())
        if self._config.include_environment_context:
            reminders.append(
                "<system-reminder>\n"
                "As you answer the user's questions, you can use the following context:\n"
                f"# currentDate\nToday's date is {datetime.now().date().isoformat()}.\n\n"
                "      IMPORTANT: this context may or may not be relevant to your tasks. "
                "You should not respond to this context unless it is highly relevant "
                "to your task.\n"
                "</system-reminder>\n\n"
            )
        return tuple(reminders)

    def _skills_reminder_block(self) -> str:
        skill_lines = "\n".join(
            f"- {name}: {description}"
            for name, description in self._default_skill_descriptors
        )
        return (
            "<system-reminder>\n"
            "The following skills are available for use with the Skill tool:\n\n"
            f"{skill_lines}\n"
            "</system-reminder>\n"
        )

    def _build_metadata(self, turn_id: str | None) -> dict[str, str]:
        return {
            "user_id": (
                '{"device_id":"pyccode-local","account_uuid":"","session_id":"'
                f'{turn_id or "pyccode-session"}'
                '"}'
            )
        }

    def _build_thinking_config(self) -> dict[str, object]:
        budget_tokens = max(1, self._config.max_tokens - 1)
        return {"type": "enabled", "budget_tokens": budget_tokens}

    def _build_context_management(self) -> dict[str, object]:
        return {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": "all",
                }
            ]
        }

    def _read_project_doc(self) -> str | None:
        path = self.cwd / self._config.project_doc_filename
        if not path.is_file():
            return None
        data = path.read_text(errors="replace")
        max_bytes = self._config.project_doc_max_bytes
        encoded = data.encode("utf-8")
        if len(encoded) <= max_bytes:
            return data.strip()
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
        return truncated + "\n\n[truncated]"

from __future__ import annotations

import asyncio

from pyccode.agent import AgentLoop
from pyccode.context import ContextManager
from pyccode.model import NOOP_MODEL_STREAM_EVENT_HANDLER
from pyccode.protocol import ConversationMessage, ModelResponse, Prompt
from pyccode.runtime import AgentRuntime
from pyccode.tools import ToolRegistry


class FinalOnlyModel:
    async def complete(self, prompt: Prompt, event_handler=NOOP_MODEL_STREAM_EVENT_HANDLER):
        assert prompt.messages
        text = prompt.messages[-1].text_content()
        return ModelResponse(
            message=ConversationMessage.assistant_text(f"done: {text}"),
            stop_reason="end_turn",
        )


async def test_runtime_processes_submissions_in_order() -> None:
    runtime = AgentRuntime(
        AgentLoop(
            FinalOnlyModel(),
            ToolRegistry(),
            context_manager=ContextManager(),
        )
    )
    runner = asyncio.create_task(runtime.run_forever())
    try:
        first = await runtime.submit_user_turn("one")
        second = await runtime.submit_user_turn("two")
        assert first.output_text == "done: one"
        assert second.output_text == "done: two"
        await runtime.shutdown()
        await runner
    finally:
        if not runner.done():
            runner.cancel()

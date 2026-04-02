from __future__ import annotations

from pyccode.agent import AgentLoop
from pyccode.context import ContextManager
from pyccode.model import NOOP_MODEL_STREAM_EVENT_HANDLER
from pyccode.protocol import (
    ConversationMessage,
    ModelResponse,
    Prompt,
    ToolResultBlock,
    ToolUseBlock,
)
from pyccode.tools import BaseTool, ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the provided text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, context, args):
        del context
        return args["text"]


class ScriptedModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: Prompt, event_handler=NOOP_MODEL_STREAM_EVENT_HANDLER):
        self.calls += 1
        if self.calls == 1:
            assert prompt.messages[-1].text_content() == "Use the echo tool."
            return ModelResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=(
                        ToolUseBlock(
                            id="toolu_1",
                            name="echo",
                            input={"text": "hello"},
                        ),
                    ),
                ),
                stop_reason="tool_use",
            )

        tool_result_message = prompt.messages[-1]
        result_block = next(
            block
            for block in tool_result_message.content
            if isinstance(block, ToolResultBlock)
        )
        assert result_block.content == "hello"
        return ModelResponse(
            message=ConversationMessage.assistant_text("tool returned: hello"),
            stop_reason="end_turn",
        )


async def test_agent_loop_executes_tool_and_finishes() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = AgentLoop(
        ScriptedModel(),
        registry,
        context_manager=ContextManager(),
    )

    result = await agent.run_turn(["Use the echo tool."])

    assert result.output_text == "tool returned: hello"
    assert result.iterations == 2
    assert result.history[-1].text_content() == "tool returned: hello"

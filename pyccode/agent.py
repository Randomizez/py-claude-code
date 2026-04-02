from __future__ import annotations

import asyncio
from collections.abc import Callable

from .context import ContextManager
from .model import ModelClient
from .protocol import (
    AgentEvent,
    ConversationMessage,
    ModelStreamEvent,
    ToolResultBlock,
    ToolUseBlock,
    TurnResult,
)
from .tools import ToolContext, ToolRegistry
from .utils import uuid7_string

EventHandler = Callable[[AgentEvent], None]
NOOP_EVENT_HANDLER: EventHandler = lambda _event: None


class TurnInterrupted(RuntimeError):
    pass


class AgentLoop:
    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        context_manager: ContextManager | None = None,
        parallel_tool_calls: bool = True,
        event_handler: EventHandler = NOOP_EVENT_HANDLER,
        initial_history: tuple[ConversationMessage, ...] = (),
    ) -> None:
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._context_manager = context_manager or ContextManager()
        self._parallel_tool_calls = parallel_tool_calls
        self._event_handler = event_handler
        self._history: list[ConversationMessage] = list(initial_history)
        self.interrupt_asap = False

    @property
    def history(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._history)

    def set_event_handler(
        self,
        event_handler: EventHandler = NOOP_EVENT_HANDLER,
    ) -> None:
        self._event_handler = event_handler

    def _emit(self, kind: str, turn_id: str, **payload: object) -> None:
        self._event_handler(
            AgentEvent(kind=kind, turn_id=turn_id, payload=dict(payload))
        )

    def _handle_model_stream_event(
        self,
        turn_id: str,
        event: ModelStreamEvent,
    ) -> None:
        if event.kind in {"assistant_text_delta", "assistant_delta"}:
            delta = event.payload.get("delta", event.payload.get("text", ""))
            payload = dict(event.payload)
            payload["delta"] = str(delta)
            self._emit("assistant_delta", turn_id, **payload)
            return

        if event.kind in {"tool_use", "tool_call", "tool_called"}:
            payload = dict(event.payload)
            tool_name = str(payload.get("tool_name", payload.get("name", "")))
            if tool_name:
                payload["tool_name"] = tool_name
            if "call_id" not in payload and "id" in payload:
                payload["call_id"] = payload["id"]
            self._emit("tool_called", turn_id, **payload)
            return

        self._emit(event.kind, turn_id, **event.payload)

    def _raise_if_interrupt_requested(
        self,
        turn_id: str,
        iteration: int,
        output_text: str | None = None,
    ) -> None:
        if self.interrupt_asap:
            self.interrupt_asap = False
            payload: dict[str, object] = {"iteration": iteration}
            if output_text is not None:
                payload["output_text"] = output_text
            self._emit("turn_interrupted", turn_id, **payload)
            raise TurnInterrupted("turn interrupted")

    async def run_turn(
        self,
        texts: list[str],
        turn_id: str | None = None,
    ) -> TurnResult:
        turn_id = turn_id or uuid7_string()
        self.interrupt_asap = False
        for text in texts:
            self._history.append(ConversationMessage.user_text(text))

        self._emit(
            "turn_started",
            turn_id,
            user_texts=list(texts),
            user_text="\n".join(texts),
        )

        last_assistant_text: str | None = None
        iteration = 0
        try:
            while True:
                self._raise_if_interrupt_requested(
                    turn_id,
                    iteration,
                    output_text=last_assistant_text,
                )
                iteration += 1
                prompt = self._context_manager.build_prompt(
                    self._history,
                    self._tool_registry.model_visible_specs(),
                    turn_id=turn_id,
                )
                self._emit(
                    "model_called",
                    turn_id,
                    iteration=iteration,
                    history_size=len(prompt.messages),
                    tool_count=len(prompt.tools),
                )
                response = await self._model_client.complete(
                    prompt,
                    lambda event: self._handle_model_stream_event(turn_id, event),
                )
                self._history.append(response.message)
                self._emit(
                    "model_completed",
                    turn_id,
                    iteration=iteration,
                    stop_reason=response.stop_reason,
                )

                last_assistant_text = response.message.text_content() or None
                tool_calls = list(response.message.tool_uses())
                if not tool_calls:
                    self._raise_if_interrupt_requested(
                        turn_id,
                        iteration,
                        output_text=last_assistant_text,
                    )
                    self._emit(
                        "turn_completed",
                        turn_id,
                        iteration=iteration,
                        output_text=last_assistant_text,
                    )
                    return TurnResult(
                        turn_id=turn_id,
                        output_text=last_assistant_text,
                        iterations=iteration,
                        response=response,
                        history=tuple(self._history),
                    )

                result_message = await self._execute_tool_batch(turn_id, tool_calls)
                self._history.append(result_message)
                self._raise_if_interrupt_requested(
                    turn_id,
                    iteration,
                    output_text=last_assistant_text,
                )
        except TurnInterrupted:
            raise
        except Exception as exc:
            self._emit(
                "turn_failed",
                turn_id,
                iteration=iteration,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

    async def _execute_tool_batch(
        self,
        turn_id: str,
        tool_calls: list[ToolUseBlock],
    ) -> ConversationMessage:
        results: list[ToolResultBlock] = []
        parallel_batch: list[ToolUseBlock] = []

        for call in tool_calls:
            can_run_parallel = (
                self._parallel_tool_calls
                and self._tool_registry.supports_parallel(call.name)
            )
            if can_run_parallel:
                parallel_batch.append(call)
                continue

            if parallel_batch:
                results.extend(
                    await self._run_parallel_batch(turn_id, parallel_batch, tuple(results))
                )
                parallel_batch = []
            results.append(await self._run_single_tool(turn_id, call, tuple(results)))

        if parallel_batch:
            results.extend(
                await self._run_parallel_batch(turn_id, parallel_batch, tuple(results))
            )

        return ConversationMessage(role="user", content=tuple(results))

    async def _run_parallel_batch(
        self,
        turn_id: str,
        tool_calls: list[ToolUseBlock],
        prior_results: tuple[ToolResultBlock, ...],
    ) -> list[ToolResultBlock]:
        return list(
            await asyncio.gather(
                *(self._run_single_tool(turn_id, call, prior_results) for call in tool_calls)
            )
        )

    async def _run_single_tool(
        self,
        turn_id: str,
        call: ToolUseBlock,
        prior_results: tuple[ToolResultBlock, ...] = (),
    ) -> ToolResultBlock:
        self._emit(
            "tool_started",
            turn_id,
            tool_name=call.name,
            call_id=call.id,
            tool_use_id=call.id,
            call=call,
        )
        result = await self._tool_registry.execute(
            call,
            ToolContext(
                turn_id=turn_id,
                history=tuple(self._history),
                prior_results=prior_results,
                cwd=self._context_manager.cwd,
            ),
        )
        self._emit(
            "tool_completed",
            turn_id,
            tool_name=call.name,
            call_id=call.id,
            tool_use_id=call.id,
            is_error=result.is_error,
            content=result.content,
            call=call,
            result=result,
        )
        return result

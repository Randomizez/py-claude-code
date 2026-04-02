from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import suppress

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import StdoutProxy, patch_stdout

from .protocol import AgentEvent, JSONDict, ToolResultBlock, ToolUseBlock

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_GREEN = "\x1b[32m"
ANSI_BLUE = "\x1b[34m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_RED = "\x1b[31m"
SPINNER_FRAMES = ("-", "\\", "|", "/")


def shorten_title(text: str, limit: int = 48) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def cli_color_enabled() -> bool:
    return os.environ.get("PYCCODE_NO_COLOR", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def colorize_cli_message(text: str, kind: str, enabled: bool) -> str:
    if not enabled:
        return text
    palette = {
        "assistant": ANSI_GREEN,
        "status": ANSI_CYAN,
        "tool": ANSI_DIM,
        "exec": ANSI_YELLOW,
        "plan": ANSI_CYAN,
        "agent": ANSI_BLUE,
        "web": ANSI_MAGENTA,
        "error": ANSI_RED,
    }
    color = palette.get(kind)
    if color is None:
        return text
    return f"{ANSI_BOLD}{color}{text}{ANSI_RESET}"


def build_cli_spinner_frame(index: int, label: str) -> str:
    suffix = f" {label}" if label else ""
    return f"wait{suffix} {SPINNER_FRAMES[index % len(SPINNER_FRAMES)]}"


class Spinner:
    def __init__(
        self,
        raw_write,
        raw_flush,
        terminal_lock: threading.RLock,
        color_enabled: bool,
        enabled: bool,
    ) -> None:
        self._raw_write = raw_write
        self._raw_flush = raw_flush
        self._terminal_lock = terminal_lock
        self._color_enabled = color_enabled
        self._enabled = enabled
        self._visible = False
        self._turn_active = False
        self._paused = False
        self._index = 0
        self._label = "thinking"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="pyccode-cli-spinner",
                daemon=True,
            )
            self._thread.start()

    def start_turn(self, label: str = "thinking") -> None:
        with self._terminal_lock:
            self._turn_active = True
            self._paused = False
            self._label = label

    def set_label(self, label: str) -> None:
        with self._terminal_lock:
            self._label = label

    def finish_turn(self) -> None:
        with self._terminal_lock:
            self._turn_active = False
            self._paused = False
            self.clear()

    def pause(self) -> None:
        with self._terminal_lock:
            self._paused = True
            self.clear()

    def resume(self) -> None:
        with self._terminal_lock:
            self._paused = False

    def clear(self) -> None:
        if not self._enabled or not self._visible:
            return
        with self._terminal_lock:
            self._raw_write("\r\x1b[2K")
            self._raw_flush()
            self._visible = False

    def close(self) -> None:
        self.finish_turn()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=0.5)

    def prompt_line(self) -> str | None:
        if not self._turn_active:
            return None
        with self._terminal_lock:
            label = self._label
        frame_index = int(time.monotonic() / 0.12)
        return build_cli_spinner_frame(frame_index, label)

    def _run(self) -> None:
        while not self._stop.wait(0.12):
            if not self._turn_active or self._paused:
                continue
            frame = colorize_cli_message(
                build_cli_spinner_frame(self._index, self._label),
                "status",
                self._color_enabled,
            )
            self._index += 1
            with self._terminal_lock:
                if not self._turn_active or self._paused:
                    continue
                self._raw_write(f"\r\x1b[2K{frame}")
                self._raw_flush()
                self._visible = True


def format_cli_tool_call_message(tool_name: str, payload: JSONDict) -> str | None:
    if tool_name.lower() != "web_search":
        return None

    query = str(payload.get("query", "")).strip()
    if query:
        return f"[web] searched: {query}"
    return "[web] browsing"


def format_cli_tool_message(tool_name: str, summary: str, is_error: bool) -> str:
    lowered = tool_name.lower()
    if lowered == "update_plan":
        if is_error:
            return f"[error] plan failed: {summary}" if summary else "[error] plan failed"
        return f"[plan] {summary}" if summary else "[plan] Plan updated"

    if lowered in {"bash", "exec_command", "write_stdin", "shell", "shell_command"}:
        if is_error:
            return f"[error] exec failed: {summary}" if summary else "[error] exec failed"
        return f"[exec] {summary}" if summary else f"[exec] {tool_name}"

    if lowered == "agent":
        if is_error:
            return f"[error] agent failed: {summary}" if summary else "[error] agent failed"
        return f"[agent] {summary}" if summary else "[agent] spawned"

    if is_error:
        return f"[error] {tool_name} failed: {summary}" if summary else f"[error] {tool_name} failed"
    return f"[tool] {tool_name}: {summary}" if summary else f"[tool] {tool_name}"


def extract_plan_items(payload: dict[str, object]) -> list[JSONDict]:
    call = payload.get("call")
    arguments = call.input if isinstance(call, ToolUseBlock) else None
    if not isinstance(arguments, dict):
        return []
    raw_plan = arguments.get("plan")
    if not isinstance(raw_plan, list):
        return []
    plan_items: list[JSONDict] = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        plan_items.append(
            {
                "step": str(item.get("step", "")).strip(),
                "status": str(item.get("status", "")).strip(),
            }
        )
    return plan_items


def format_cli_plan_messages(summary: str, plan_items: list[JSONDict]) -> list[str]:
    lines = [f"[plan] {summary}" if summary else "[plan] Plan updated"]
    for item in plan_items:
        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "")).strip()
        if not step:
            continue
        marker = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
        }.get(status, "[ ]")
        lines.append(f"  {marker} {step}")
    return lines


def extract_tool_event_display(
    payload: dict[str, object],
) -> tuple[str, str, bool]:
    tool_name = str(payload.get("tool_name", "")).strip()
    is_error = bool(payload.get("is_error"))
    call = payload.get("call")
    result = payload.get("result")
    if isinstance(call, ToolUseBlock) and isinstance(result, ToolResultBlock):
        return tool_name, summarize_tool_event(call, result), is_error
    summary = str(payload.get("summary", "") or payload.get("content", "")).strip()
    return tool_name, summary, is_error


def summarize_tool_event(call: ToolUseBlock, result: ToolResultBlock) -> str:
    command_preview = _command_preview(call)
    result_summary = _summarize_tool_result(result)
    if command_preview and result_summary:
        return f"{command_preview} -> {result_summary}"
    if command_preview:
        return command_preview
    return result_summary


def _truncate_text(text: str, limit: int = 96) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _extract_output_preview(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines()]
    for line in lines:
        if not line:
            continue
        return _truncate_text(line)
    return None


def _summarize_tool_result(result: ToolResultBlock) -> str:
    preview = _extract_output_preview(result.content)
    if preview:
        return preview
    return ""


def _command_preview(call: ToolUseBlock) -> str | None:
    args = call.input
    if call.name == "Bash":
        command = _string_arg(args, "command")
        if command:
            return _truncate_text(command, limit=72)
    if call.name == "Read":
        path = _string_arg(args, "file_path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "Glob":
        pattern = _string_arg(args, "pattern")
        if pattern:
            return _truncate_text(pattern, limit=72)
    if call.name == "Grep":
        pattern = _string_arg(args, "pattern")
        if pattern:
            return _truncate_text(pattern, limit=72)
    if call.name == "Edit":
        path = _string_arg(args, "file_path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "Write":
        path = _string_arg(args, "file_path")
        if path:
            return _truncate_text(path, limit=72)
    if call.name == "Agent":
        prompt = _string_arg(args, "prompt")
        if prompt:
            return _truncate_text(prompt, limit=40)
    return None


def _string_arg(arguments: object, key: str) -> str | None:
    if not isinstance(arguments, dict):
        return None
    value = arguments.get(key)
    if value in (None, ""):
        return None
    return str(value)


class CliSessionView:
    def __init__(self) -> None:
        import sys

        self._line_output = print
        self._raw_write = sys.stdout.write
        self._raw_flush = sys.stdout.flush
        self._terminal_lock = threading.RLock()
        self._title: str | None = None
        self._pending_user_prompts: dict[str, str] = {}
        self._queued_steer_prompts: dict[str, list[str]] = {}
        self._inserted_steer_prompts: dict[str, list[str]] = {}
        self._history: list[tuple[str, str]] = []
        self._streaming = False
        self._prompt_stream_buffer = ""
        self._streaming_in_prompt = False
        self._input_active = False
        self._color_enabled = cli_color_enabled() and sys.stdout.isatty()
        self._prompt_session: PromptSession | None = None
        self._prompt_task: asyncio.Task[str] | None = None
        self._stdout_proxy: StdoutProxy | None = None
        self._spinner = Spinner(
            self._raw_write,
            self._raw_flush,
            self._terminal_lock,
            self._color_enabled,
            False,
        )

    def handle_event(self, event: AgentEvent) -> None:
        if event.kind == "turn_started":
            submission_id = str(event.payload.get("submission_id", event.turn_id)).strip()
            user_texts = event.payload.get("user_texts")
            if isinstance(user_texts, list):
                normalized_user_texts = [
                    str(text).strip() for text in user_texts if str(text).strip()
                ]
            else:
                normalized_user_texts = []
            user_text = str(event.payload.get("user_text", "")).strip()
            if not user_text and normalized_user_texts:
                user_text = "\n".join(normalized_user_texts)
            if self._title is None and user_text:
                self._title = shorten_title(user_text)
                self._print_line(f"Session: {self._title}")
            if user_text:
                self._pending_user_prompts[submission_id] = user_text
            inserted_steer_prompts = self._inserted_steer_prompts.pop(submission_id, [])
            for inserted_steer_prompt in inserted_steer_prompts:
                self._print_line(
                    colorize_cli_message(
                        f"[steer] inserted: {inserted_steer_prompt}",
                        "status",
                        self._color_enabled,
                    )
                )
            queued_steer_prompts = self._queued_steer_prompts.pop(submission_id, [])
            for queued_steer_prompt in queued_steer_prompts:
                self._print_line(
                    colorize_cli_message(
                        f"[steer] inserted: {queued_steer_prompt}",
                        "status",
                        self._color_enabled,
                    )
                )
            self._spinner.start_turn("thinking")
            if self._input_active:
                self._spinner.pause()
            return

        if event.kind == "model_called":
            if self._input_active:
                self._spinner.pause()
            else:
                self._spinner.resume()
                self._spinner.set_label("waiting model")
            return

        if event.kind in {"assistant_delta", "assistant_text_delta"}:
            delta = str(event.payload.get("delta", event.payload.get("text", "")))
            if not delta:
                return
            if self._input_active:
                if not self._streaming:
                    self._streaming = True
                    self._streaming_in_prompt = True
                    self._prompt_stream_buffer = ""
                self._prompt_stream_buffer += delta
                return
            with self._terminal_lock:
                if not self._streaming:
                    self._spinner.pause()
                    self._raw_write("assistant> ")
                    self._streaming = True
                self._raw_write(delta)
                self._raw_flush()
            return

        if event.kind in {"tool_called", "tool_use"}:
            payload = dict(event.payload)
            tool_name = str(payload.get("tool_name", payload.get("name", ""))).strip()
            message = format_cli_tool_call_message(tool_name, payload)
            if message is not None:
                self._finish_stream()
                self._print_line(
                    colorize_cli_message(message, "web", self._color_enabled)
                )
                if self._input_active:
                    self._spinner.pause()
                else:
                    self._spinner.resume()
                    self._spinner.set_label("running tools")
            return

        if event.kind == "tool_started":
            self._finish_stream()
            if self._input_active:
                self._spinner.pause()
            else:
                self._spinner.resume()
                self._spinner.set_label("running tools")
            return

        if event.kind == "tool_completed":
            self._finish_stream()
            if self._input_active:
                self._spinner.pause()
            else:
                self._spinner.resume()
                self._spinner.set_label("thinking")
            tool_name, summary, is_error = extract_tool_event_display(event.payload)
            if tool_name == "update_plan" and not is_error:
                plan_items = extract_plan_items(event.payload)
                for line in format_cli_plan_messages(summary, plan_items):
                    self._print_line(
                        colorize_cli_message(line, "plan", self._color_enabled)
                    )
                return
            message = format_cli_tool_message(tool_name, summary, is_error)
            self._print_line(self._colorize_formatted_tool_message(message))
            return

        if event.kind == "turn_completed":
            submission_id = str(event.payload.get("submission_id", event.turn_id)).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            self._finalize_turn_output(final_text, allow_standalone_output=True)
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None:
                self._history.append((pending_prompt, final_text))
            return

        if event.kind == "turn_failed":
            submission_id = str(event.payload.get("submission_id", event.turn_id)).strip()
            self._spinner.finish_turn()
            self._finish_stream()
            self._pending_user_prompts.pop(submission_id, None)
            return

        if event.kind == "turn_interrupted":
            submission_id = str(event.payload.get("submission_id", event.turn_id)).strip()
            final_text = str(event.payload.get("output_text", "") or "")
            self._finalize_turn_output(final_text, allow_standalone_output=False)
            pending_prompt = self._pending_user_prompts.pop(submission_id, None)
            if pending_prompt is not None and final_text:
                self._history.append((pending_prompt, final_text))

    def show_history(self) -> None:
        self._finish_stream()
        if not self._history:
            self._print_line("No history yet.")
            return

        self._print_line(f"Session: {self._title or 'untitled'}")
        for index, (user_text, assistant_text) in enumerate(self._history, start=1):
            self._print_line(f"[{index}] user> {user_text}")
            self._print_line(f"    assistant> {assistant_text}")

    def show_title(self) -> None:
        self._finish_stream()
        self._print_line(f"Session: {self._title or 'untitled'}")

    def pause_spinner(self) -> None:
        self._spinner.pause()

    def resume_spinner(self) -> None:
        self._spinner.resume()

    def set_input_active(self, active: bool, resume_spinner: bool = True) -> None:
        self._input_active = active
        if active:
            self._spinner.pause()
        elif resume_spinner:
            self._spinner.resume()

    def is_streaming_output(self) -> bool:
        return self._streaming

    def handoff_prompt_stream_to_output(self) -> None:
        if not self._streaming or not self._streaming_in_prompt:
            return
        buffered = self._prompt_stream_buffer
        self._prompt_stream_buffer = ""
        self._streaming_in_prompt = False
        if not buffered:
            return
        with self._terminal_lock:
            self._raw_write("assistant> ")
            self._raw_write(buffered)
            self._raw_flush()

    async def poll_prompt(self, prompt: str) -> str | None:
        if self._prompt_task is None:
            if self.is_streaming_output():
                return None
            self._prompt_task = asyncio.create_task(self.prompt_async(prompt))

        done, _pending = await asyncio.wait(
            {self._prompt_task},
            timeout=0.05,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._prompt_task not in done:
            if self.is_streaming_output():
                await self._handoff_prompt_task_to_output()
            return None

        prompt_task = self._prompt_task
        self._prompt_task = None
        try:
            return prompt_task.result()
        except asyncio.CancelledError:
            return None
        finally:
            self.set_input_active(False, resume_spinner=False)

    def build_input_prompt(self, prompt: str) -> str:
        if not self._input_active:
            return prompt
        if self._streaming and self._streaming_in_prompt:
            if self._prompt_stream_buffer:
                return f"assistant> {self._prompt_stream_buffer}\n"
            return "\n"
        prompt_line = self._spinner.prompt_line()
        if not prompt_line:
            return prompt
        return f"{prompt_line}\n{prompt}"

    def show_steer_queued(self, turn_id: str, prompt: str) -> None:
        preview = shorten_title(prompt, limit=72)
        self._queued_steer_prompts.setdefault(turn_id, []).append(preview)
        self._print_line(
            colorize_cli_message(
                f"[steer] queued: {preview}",
                "status",
                self._color_enabled,
            )
        )

    def schedule_steer_inserted(self, turn_id: str, prompt: str) -> None:
        self._inserted_steer_prompts.setdefault(turn_id, []).append(
            shorten_title(prompt, limit=72)
        )

    def close(self) -> None:
        if self._prompt_task is not None and not self._prompt_task.done():
            self._prompt_task.cancel()
            self._prompt_task = None
        self._spinner.close()
        if self._stdout_proxy is not None:
            self._stdout_proxy.close()

    def finish_stream(self) -> None:
        self._finish_stream()

    def write_line(self, text: str) -> None:
        self._print_line(text)

    def show_error(self, text: str) -> None:
        self._spinner.finish_turn()
        self._finish_stream()
        self._print_line(
            colorize_cli_message(
                f"Error: {text}",
                "error",
                self._color_enabled,
            )
        )

    def _finish_stream(self) -> None:
        with self._terminal_lock:
            self._spinner.clear()
            if self._streaming:
                self._raw_write("\n")
                self._raw_flush()
                self._streaming = False
            self._streaming_in_prompt = False
            self._prompt_stream_buffer = ""

    def _finalize_turn_output(
        self,
        final_text: str,
        allow_standalone_output: bool,
    ) -> None:
        self._spinner.finish_turn()
        if self._streaming and self._streaming_in_prompt:
            streamed_text = self._prompt_stream_buffer
            self._streaming = False
            self._streaming_in_prompt = False
            self._prompt_stream_buffer = ""
            final_display_text = final_text or streamed_text
            if final_display_text:
                self._print_line(
                    colorize_cli_message(
                        f"assistant> {final_display_text}",
                        "assistant",
                        self._color_enabled,
                    )
                )
            return
        if self._streaming:
            self._finish_stream()
            return
        if allow_standalone_output and final_text:
            self._print_line(
                colorize_cli_message(
                    f"assistant> {final_text}",
                    "assistant",
                    self._color_enabled,
                )
            )

    def _colorize_formatted_tool_message(self, message: str) -> str:
        if message.startswith("[plan]"):
            return colorize_cli_message(message, "plan", self._color_enabled)
        if message.startswith("[exec]"):
            return colorize_cli_message(message, "exec", self._color_enabled)
        if message.startswith("[agent]"):
            return colorize_cli_message(message, "agent", self._color_enabled)
        if message.startswith("[web]"):
            return colorize_cli_message(message, "web", self._color_enabled)
        if message.startswith("[error]"):
            return colorize_cli_message(message, "error", self._color_enabled)
        return colorize_cli_message(message, "tool", self._color_enabled)

    def _print_line(self, text: str) -> None:
        with self._terminal_lock:
            self._spinner.clear()
            self._line_output(text)

    async def prompt_async(self, prompt: str) -> str:
        if self._prompt_session is None:
            self._prompt_session = PromptSession(
                erase_when_done=True,
                enable_system_prompt=True,
            )
        if self._stdout_proxy is None:
            self._stdout_proxy = StdoutProxy(raw=False)
            self._raw_write = self._stdout_proxy.write
            self._raw_flush = self._stdout_proxy.flush

        self.set_input_active(True)
        try:
            with patch_stdout(raw=True):
                return await self._prompt_session.prompt_async(
                    lambda: self.build_input_prompt(prompt),
                    refresh_interval=0.12,
                )
        finally:
            self.set_input_active(False, resume_spinner=False)

    async def _handoff_prompt_task_to_output(self) -> None:
        if self._prompt_task is None:
            return
        prompt_task = self._prompt_task
        self._prompt_task = None
        prompt_task.cancel()
        with suppress(asyncio.CancelledError):
            await prompt_task
        self.set_input_active(False, resume_spinner=False)
        self.handoff_prompt_stream_to_output()

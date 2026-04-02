from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXEC_YIELD_TIME_MS = 10_000
DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_LOGIN = True
DEFAULT_TTY = False
DEFAULT_SESSION_ID_START = 1000
APPROX_BYTES_PER_TOKEN = 4
UNIFIED_EXEC_OUTPUT_MAX_BYTES = 1024 * 1024


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    byte_length = len(text.encode("utf-8"))
    return max(1, (byte_length + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN)


def _approx_bytes_for_tokens(token_count: int) -> int:
    return max(token_count, 0) * APPROX_BYTES_PER_TOKEN


def _approx_tokens_from_byte_count(byte_count: int) -> int:
    if byte_count <= 0:
        return 0
    return (byte_count + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN


def _split_budget(byte_budget: int) -> tuple[int, int]:
    left_budget = byte_budget // 2
    return left_budget, byte_budget - left_budget


def _split_string(text: str, beginning_bytes: int, end_bytes: int) -> tuple[str, str]:
    if not text:
        return "", ""

    total_bytes = len(text.encode("utf-8"))
    tail_start_target = max(total_bytes - end_bytes, 0)
    prefix_end = 0
    suffix_start = len(text)
    suffix_started = False
    current_byte = 0

    for index, char in enumerate(text):
        char_bytes = len(char.encode("utf-8"))
        char_start = current_byte
        char_end = current_byte + char_bytes
        if char_end <= beginning_bytes:
            prefix_end = index + 1
            current_byte = char_end
            continue
        if char_start >= tail_start_target:
            if not suffix_started:
                suffix_start = index
                suffix_started = True
            current_byte = char_end
            continue
        current_byte = char_end

    if suffix_start < prefix_end:
        suffix_start = prefix_end

    return text[:prefix_end], text[suffix_start:]


def _truncate_text(text: str, max_tokens: int) -> str:
    if not text:
        return ""

    max_bytes = _approx_bytes_for_tokens(max_tokens)
    total_bytes = len(text.encode("utf-8"))
    if total_bytes <= max_bytes:
        return text

    removed_tokens = _approx_tokens_from_byte_count(total_bytes - max_bytes)
    marker = f"...{removed_tokens} tokens truncated..."
    if max_bytes == 0:
        return marker

    left_budget, right_budget = _split_budget(max_bytes)
    prefix, suffix = _split_string(text, left_budget, right_budget)
    return f"{prefix}{marker}{suffix}"


def _formatted_truncate_text(text: str, max_tokens: int) -> str:
    byte_budget = _approx_bytes_for_tokens(max_tokens)
    if len(text.encode("utf-8")) <= byte_budget:
        return text

    total_lines = len(text.splitlines())
    return f"Total output lines: {total_lines}\n\n{_truncate_text(text, max_tokens)}"


@dataclass(slots=True)
class _HeadTailBuffer:
    max_bytes: int = UNIFIED_EXEC_OUTPUT_MAX_BYTES
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)

    def push_chunk(self, chunk: bytes) -> None:
        if not chunk or self.max_bytes <= 0:
            return

        head_budget = self.max_bytes // 2
        tail_budget = self.max_bytes - head_budget
        remaining = bytes(chunk)

        if len(self.head) < head_budget:
            head_room = head_budget - len(self.head)
            head_part = remaining[:head_room]
            self.head.extend(head_part)
            remaining = remaining[len(head_part) :]

        if not remaining or tail_budget <= 0:
            return

        self.tail.extend(remaining)
        if len(self.tail) > tail_budget:
            excess = len(self.tail) - tail_budget
            del self.tail[:excess]

    def drain_bytes(self) -> bytes:
        combined = bytes(self.head) + bytes(self.tail)
        self.head.clear()
        self.tail.clear()
        return combined


@dataclass(slots=True)
class UnifiedExecSession:
    session_id: int
    process: asyncio.subprocess.Process
    command_display: str
    tty: bool
    unread_output: _HeadTailBuffer = field(default_factory=_HeadTailBuffer)
    reader_task: asyncio.Task | None = None


class UnifiedExecManager:
    def __init__(self, cwd: str | Path | None = None) -> None:
        self._default_cwd = Path(cwd or Path.cwd()).resolve()
        self._next_session_id = DEFAULT_SESSION_ID_START
        self._sessions: dict[int, UnifiedExecSession] = {}
        self._lock = asyncio.Lock()

    async def exec_command(
        self,
        cmd: str,
        workdir: str | None = None,
        shell: str | None = None,
        login: bool = DEFAULT_LOGIN,
        tty: bool = DEFAULT_TTY,
        yield_time_ms: int = DEFAULT_EXEC_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
    ) -> str:
        session_id = await self._allocate_session_id()
        command = self._build_shell_command(cmd, shell, login)
        cwd = self._resolve_workdir(workdir)

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        session = UnifiedExecSession(
            session_id=session_id,
            process=process,
            command_display=shlex.join(command),
            tty=tty,
        )
        session.reader_task = asyncio.create_task(self._pump_output(session))

        async with self._lock:
            self._sessions[session_id] = session

        return await self._wait_and_snapshot(
            session_id,
            max(yield_time_ms, 1),
            max_output_tokens,
        )

    async def write_stdin(
        self,
        session_id: int,
        chars: str = "",
        yield_time_ms: int = DEFAULT_WRITE_STDIN_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
    ) -> str:
        session = await self._get_session(session_id)
        if session is None:
            return f"Error: session_id {session_id} is not running."

        if chars:
            if session.process.stdin is None:
                return f"Error: session_id {session_id} does not accept stdin."
            if session.tty:
                session.unread_output.push_chunk(self._tty_echo(chars))
            session.process.stdin.write(chars.encode("utf-8"))
            await session.process.stdin.drain()

        return await self._wait_and_snapshot(
            session_id,
            max(yield_time_ms, 1),
            max_output_tokens,
        )

    async def _allocate_session_id(self) -> int:
        async with self._lock:
            session_id = self._next_session_id
            self._next_session_id += 1
            return session_id

    async def _get_session(self, session_id: int) -> UnifiedExecSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def _wait_and_snapshot(
        self,
        session_id: int,
        yield_time_ms: int,
        max_output_tokens: int | None,
    ) -> str:
        session = await self._get_session(session_id)
        if session is None:
            return f"Error: session_id {session_id} is not running."

        start_wait = asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(session.process.wait(), timeout=yield_time_ms / 1000.0)
        except asyncio.TimeoutError:
            pass

        if session.reader_task is not None and session.process.returncode is not None:
            await session.reader_task

        wall_time = asyncio.get_running_loop().time() - start_wait
        output_bytes = session.unread_output.drain_bytes()
        output_text = output_bytes.decode("utf-8", errors="replace")
        original_token_count = _approx_token_count(output_text)
        output_text = self._truncate_output(output_text, max_output_tokens)

        lines = [
            f"Command: {session.command_display}",
            f"Chunk ID: {uuid.uuid4().hex[:6]}",
            f"Wall time: {wall_time:.4f} seconds",
        ]
        if session.process.returncode is None:
            lines.append(f"Process running with session ID {session_id}")
        else:
            lines.append(f"Process exited with code {session.process.returncode}")
        lines.append(f"Original token count: {original_token_count}")
        lines.append("Output:")
        lines.append(output_text)

        if session.process.returncode is not None:
            await self._close_session(session_id)

        return "\n".join(lines)

    async def _close_session(self, session_id: int) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        if session.process.stdin is not None and not session.process.stdin.is_closing():
            session.process.stdin.close()

    async def _pump_output(self, session: UnifiedExecSession) -> None:
        stream = session.process.stdout
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            session.unread_output.push_chunk(chunk)

    def _build_shell_command(
        self,
        cmd: str,
        shell: str | None,
        login: bool,
    ) -> list[str]:
        shell_binary = shell or "bash"
        return [shell_binary, "-lc" if login else "-c", cmd]

    def _resolve_workdir(self, workdir: str | None) -> Path:
        if not workdir:
            return self._default_cwd
        candidate = Path(workdir)
        if not candidate.is_absolute():
            candidate = self._default_cwd / candidate
        return candidate.resolve()

    def _truncate_output(self, text: str, max_output_tokens: int | None) -> str:
        limit = DEFAULT_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
        return _formatted_truncate_text(text, limit)

    def _tty_echo(self, chars: str) -> bytes:
        return chars.encode("utf-8")

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .agent import AgentLoop
from .codex_model import DEFAULT_CODEX_CONFIG_PATH, ResponsesModelClient
from .context import ContextManager
from .doctor import build_doctor_parser, run_doctor_cli
from .model import AnthropicMessagesConfig, AnthropicMessagesModelClient
from .runtime import AgentRuntime
from .tools import build_default_tool_registry
from .utils import build_user_agent, load_codex_dotenv
from .visualize import CliSessionView

EXIT_COMMANDS = {"/exit", "/quit"}
HISTORY_COMMAND = "/history"
TITLE_COMMAND = "/title"
MODEL_COMMAND = "/model"
QUEUE_COMMAND = "/queue"
CLI_ORIGINATOR = "codex-tui"


def configure_loguru() -> None:
    try:
        from loguru import logger
    except ImportError:  # pragma: no cover - dependency may be absent
        return

    logger.remove()
    log_path = os.environ.get("PYCCODE_DEBUG_LOG", "").strip()
    if log_path:
        logger.add(log_path, level="DEBUG")
        return

    if os.environ.get("PYCCODE_DEBUG_STDERR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.add(sys.stderr, level="DEBUG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyccode",
        description="Minimal A-style local CLI with a pycodex-style interactive surface.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, read from stdin.")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Anthropic-compatible base URL")
    parser.add_argument(
        "--backend",
        choices=("auto", "anthropic", "codex"),
        default="auto",
        help="Model backend to use. Defaults to codex when a Codex config is available.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to the Codex config.toml used by the codex backend.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional Codex profile name.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for one model call.",
    )
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--json", action="store_true", help="Emit JSON result")
    return parser


def should_run_interactive(prompt_parts: Sequence[str], stdin_is_tty: bool) -> bool:
    return not prompt_parts and stdin_is_tty


def resolve_prompt_text(prompt_parts: Sequence[str]) -> str:
    if prompt_parts:
        return " ".join(prompt_parts).strip()

    if not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()
        if prompt_text:
            return prompt_text

    raise ValueError("prompt is required either as argv text or stdin")


async def run_cli(args: argparse.Namespace) -> int:
    configure_loguru()
    tools = build_default_tool_registry()
    base_context = ContextManager()
    context = base_context.with_config(
        replace(base_context.config, max_tokens=args.max_tokens)
    )
    model = _build_model_client(args)
    runtime = AgentRuntime(AgentLoop(model, tools, context_manager=context))
    runner: asyncio.Task[None] | None = None
    try:
        if should_run_interactive(args.prompt, sys.stdin.isatty()):
            return await _run_interactive_session(runtime, bool(args.json))

        prompt_text = resolve_prompt_text(args.prompt)
        if prompt_text is not None:
            runner = asyncio.create_task(runtime.run_forever())
            result = await runtime.submit_user_turn(prompt_text)
            print(format_turn_output(result, bool(args.json)))
            await runtime.shutdown()
            await runner
            return 0
    finally:
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass


def format_turn_output(result, json_mode: bool) -> str:
    if json_mode:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    return result.output_text or ""


def _build_model_client(args: argparse.Namespace):
    backend = _resolve_backend(args)
    if backend == "codex":
        load_codex_dotenv(args.config)
        return ResponsesModelClient.from_codex_config(
            config_path=args.config,
            profile=args.profile,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            originator=CLI_ORIGINATOR,
            user_agent=build_user_agent(CLI_ORIGINATOR),
        )

    config = AnthropicMessagesConfig.from_env(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    return AnthropicMessagesModelClient(config)


def _resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend

    anthropic_env_keys = (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
    )
    if args.base_url or any(os.environ.get(key) for key in anthropic_env_keys):
        return "anthropic"
    if Path(args.config).exists():
        return "codex"
    return "anthropic"


async def _run_interactive_session(
    runtime: AgentRuntime,
    json_mode: bool,
) -> int:
    worker = asyncio.create_task(runtime.run_forever())
    view = CliSessionView()
    model_client = runtime._agent_loop._model_client
    runtime.set_event_handler(view.handle_event)
    pending_turn_tasks: set[asyncio.Task[None]] = set()
    view.write_line("pyccode interactive mode. Type /exit to quit.")
    view.write_line("Extra commands: /history, /title, /model, /queue")
    try:
        def has_pending_turn_tasks() -> bool:
            pending_turn_tasks.difference_update(
                task for task in tuple(pending_turn_tasks) if task.done()
            )
            return bool(pending_turn_tasks)

        async def wait_for_turn_result(future) -> None:
            try:
                result = await future
            except Exception as exc:  # pragma: no cover - defensive surface
                if str(exc) == "submission interrupted":
                    return
                view.show_error(str(exc))
                return

            if json_mode:
                view.write_line(format_turn_output(result, True))

        while True:
            try:
                raw_line = await view.poll_prompt("pyccode> ")
            except EOFError:
                break
            if raw_line is None:
                await asyncio.sleep(0.05)
                continue

            prompt_text = raw_line.strip()
            if not prompt_text:
                continue
            if prompt_text in EXIT_COMMANDS:
                break
            if prompt_text == HISTORY_COMMAND:
                view.show_history()
                continue
            if prompt_text == TITLE_COMMAND:
                view.show_title()
                continue
            if prompt_text.startswith(f"{QUEUE_COMMAND} "):
                queued_text = prompt_text[len(QUEUE_COMMAND) :].strip()
                if not queued_text:
                    view.write_line("Usage: /queue <message>")
                    continue
                try:
                    submission_id, future = await runtime.enqueue_user_turn(
                        queued_text,
                        queue="enqueue",
                    )
                    view.show_steer_queued(submission_id, queued_text)
                    turn_task = asyncio.create_task(wait_for_turn_result(future))
                    pending_turn_tasks.add(turn_task)
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text == MODEL_COMMAND:
                current_model = getattr(model_client, "model", None) or "unavailable"
                view.write_line(f"Current model: {current_model}")
                if hasattr(model_client, "list_models"):
                    try:
                        models = await model_client.list_models()
                    except Exception as exc:  # pragma: no cover - defensive surface
                        view.show_error(str(exc))
                    else:
                        if models:
                            view.write_line(f"Available models: {', '.join(models)}")
                continue
            if prompt_text.startswith(f"{MODEL_COMMAND} "):
                if has_pending_turn_tasks():
                    view.write_line(
                        "Cannot change model while work is running or queued in steer mode."
                    )
                    continue
                model_name = prompt_text[len(MODEL_COMMAND) :].strip()
                if not model_name:
                    view.write_line("Usage: /model <model>")
                    continue
                model_client.model = model_name
                view.write_line(f"Switched model to {model_name}.")
                continue

            try:
                steered = has_pending_turn_tasks()
                submission_id, future = await runtime.enqueue_user_turn(
                    prompt_text,
                    queue="steer",
                )
                if steered:
                    view.schedule_steer_inserted(submission_id, prompt_text)
                turn_task = asyncio.create_task(wait_for_turn_result(future))
                pending_turn_tasks.add(turn_task)
            except Exception as exc:  # pragma: no cover - defensive surface
                view.show_error(str(exc))
                continue

        if pending_turn_tasks:
            await asyncio.gather(*tuple(pending_turn_tasks), return_exceptions=True)
        await runtime.shutdown()
        await worker
        return 0
    finally:
        view.close()
        if not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else None
    if raw_args is None:
        raw_args = sys.argv[1:]

    if raw_args and raw_args[0] == "doctor":
        parser = build_doctor_parser()
        args = parser.parse_args(raw_args[1:])
        try:
            return asyncio.run(run_doctor_cli(args))
        except ValueError as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            return 130
        return 0

    parser = build_parser()
    args = parser.parse_args(raw_args)

    try:
        return asyncio.run(run_cli(args))
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import sys

from .agent import AgentLoop
from .context import ContextManager
from .doctor import build_doctor_parser, run_doctor_cli
from .model import AnthropicMessagesConfig, AnthropicMessagesModelClient
from .runtime import AgentRuntime
from .tools import build_default_tool_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Claude Code style Python agent")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the agent")
    parser.add_argument("--model", help="Anthropic model name")
    parser.add_argument("--base-url", help="Anthropic-compatible base URL")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--json", action="store_true", help="Emit JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        args = build_doctor_parser().parse_args(argv[1:])
        return asyncio.run(run_doctor_cli(args))
    args = build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


async def _main_async(args: argparse.Namespace) -> int:
    text = _resolve_prompt(args.prompt)
    config = AnthropicMessagesConfig.from_env(
        model=args.model,
        base_url=args.base_url,
    )
    tools = build_default_tool_registry()
    base_context = ContextManager()
    context = base_context.with_config(
        replace(base_context.config, max_tokens=args.max_tokens)
    )
    model = AnthropicMessagesModelClient(config)
    runtime = AgentRuntime(AgentLoop(model, tools, context_manager=context))
    runner = asyncio.create_task(runtime.run_forever())
    try:
        if text is not None:
            result = await runtime.submit_user_turn(text)
            if args.json:
                print(
                    json.dumps(
                        {
                            "turn_id": result.turn_id,
                            "iterations": result.iterations,
                            "output_text": result.output_text,
                        },
                        ensure_ascii=False,
                    )
                )
            elif result.output_text:
                print(result.output_text)
            await runtime.shutdown()
            await runner
            return 0

        while True:
            try:
                user_text = input("pyccode> ").strip()
            except EOFError:
                user_text = "/exit"
            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                break
            result = await runtime.submit_user_turn(user_text)
            if result.output_text:
                print(result.output_text)
        await runtime.shutdown()
        await runner
        return 0
    finally:
        if not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass


def _resolve_prompt(prompt_args: list[str]) -> str | None:
    if prompt_args:
        return " ".join(prompt_args).strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        return data or None
    return None

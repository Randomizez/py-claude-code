"""Capture one `pyccode` request and one Claude Code request via a local fake server.

This helper intentionally does not proxy to any real upstream. It starts the
local fake Anthropic Messages server, points both clients at it, and records
the outbound requests for later manual diffing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from tests.fake_anthropic_messages_server import CaptureStore, build_fake_handler
from tests.prepare_fake_oauth_home import prepare_fake_oauth_home


DEFAULT_OUTPUT_ROOT = Path(".debug") / "fake_messages_compare"
DEFAULT_PROMPT = "say hi"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_TIMEOUT_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uv run python tests/capture_fake_messages_requests.py",
        description=(
            "Start a local fake Anthropic Messages server, then capture one "
            "`pyccode` request and one Claude Code request without contacting "
            "any real upstream."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory used to store captures and command logs.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt sent to both clients.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name passed to both clients.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout applied to each client invocation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    oauth_home = output_root / "pyccode_oauth_home"
    prepare_fake_oauth_home(oauth_home)

    pyccode_capture = run_capture(
        label="pyccode",
        capture_root=output_root / "pyccode",
        model=args.model,
        response_text="Hello from fake server to pyccode.",
        command=[
            "uv",
            "run",
            "pyccode",
            "--model",
            args.model,
            args.prompt,
        ],
        command_log_path=logs_root / "pyccode.log",
        timeout_seconds=args.timeout_seconds,
        extra_env={
            "CLAUDE_CONFIG_DIR": str(oauth_home),
        },
        unset_env_keys=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    )

    claude_capture = run_capture(
        label="claude",
        capture_root=output_root / "claude",
        model=args.model,
        response_text="Hello from fake server to Claude Code.",
        command=[
            "claude",
            "--bare",
            "--model",
            args.model,
            "-p",
            args.prompt,
        ],
        command_log_path=logs_root / "claude.log",
        timeout_seconds=args.timeout_seconds,
        extra_env={
            "ANTHROPIC_API_KEY": "fake-key",
        },
        unset_env_keys=(),
    )

    summary = {
        "pyccode_capture": str(pyccode_capture),
        "claude_capture": str(claude_capture),
        "pyccode_oauth_home": str(oauth_home),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pyccode capture: {pyccode_capture}")
    print(f"claude capture:  {claude_capture}")
    print(f"summary:         {summary_path}")


def run_capture(
    label: str,
    capture_root: Path,
    model: str,
    response_text: str,
    command: list[str],
    command_log_path: Path,
    timeout_seconds: float,
    extra_env: dict[str, str],
    unset_env_keys: tuple[str, ...],
) -> Path:
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_fake_handler(capture_store, model, response_text),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        env = os.environ.copy()
        for key in unset_env_keys:
            env.pop(key, None)
        env.update(extra_env)
        env["ANTHROPIC_BASE_URL"] = base_url

        run_command(
            command,
            env=env,
            log_path=command_log_path,
            timeout_seconds=timeout_seconds,
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    capture_files = sorted(capture_root.glob("*_POST_*.json"))
    if not capture_files:
        raise RuntimeError(f"{label}: no POST capture was recorded")
    return capture_files[0]


def run_command(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError(f"command timed out: {' '.join(command)}")
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {' '.join(command)}\n"
            f"see {log_path}"
        )


if __name__ == "__main__":
    main()

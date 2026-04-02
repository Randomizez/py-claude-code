"""Capture one non-bare Claude Code request using fake OAuth state.

This helper stays fully local:
- starts the local fake Anthropic Messages server
- prepares an isolated `CLAUDE_CONFIG_DIR` with fake OAuth credentials
- runs `claude -p` without `--bare`
- records the outbound request under `./.debug/`
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from tests.fake_anthropic_messages_server import CaptureStore, build_fake_handler
from tests.prepare_fake_oauth_home import prepare_fake_oauth_home

DEFAULT_OUTPUT_ROOT = Path('.debug') / 'nonbare_oauth_capture'
DEFAULT_OAUTH_HOME = Path('.debug') / 'nonbare_oauth_home'
DEFAULT_PROMPT = 'say hi'
DEFAULT_MODEL = 'claude-sonnet-4-5'
DEFAULT_TIMEOUT_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='uv run python tests/capture_nonbare_oauth_request.py',
        description='Capture one non-bare Claude Code request using fake OAuth state.',
    )
    parser.add_argument('--root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--oauth-home', default=str(DEFAULT_OAUTH_HOME))
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--timeout-seconds', type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    capture_root = Path(args.root).resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    oauth_home = Path(args.oauth_home).resolve()
    prepare_fake_oauth_home(oauth_home)

    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ('127.0.0.1', 0),
        build_fake_handler(capture_store, args.model, 'Hello from fake Anthropic.'),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f'http://127.0.0.1:{httpd.server_port}'
        env = os.environ.copy()
        env['CLAUDE_CONFIG_DIR'] = str(oauth_home)
        env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
        env['ANTHROPIC_BASE_URL'] = base_url
        env.pop('ANTHROPIC_API_KEY', None)
        env.pop('ANTHROPIC_AUTH_TOKEN', None)
        env.pop('CLAUDE_CODE_OAUTH_TOKEN', None)
        run_command(
            ['claude', '--model', args.model, '-p', args.prompt],
            env=env,
            timeout_seconds=args.timeout_seconds,
            log_path=capture_root / 'claude.log',
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    capture_files = sorted(capture_root.glob('*_POST_*.json'))
    if not capture_files:
        raise RuntimeError('no capture files found')

    summary = {
        'oauth_home': str(oauth_home),
        'capture_file': str(capture_files[0]),
    }
    summary_path = capture_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'oauth home:   {oauth_home}')
    print(f'capture file: {capture_files[0]}')
    print(f'summary:      {summary_path}')
def run_command(
    command: list[str],
    env: dict[str, str],
    timeout_seconds: float,
    log_path: Path,
) -> None:
    with log_path.open('w', encoding='utf-8') as log_file:
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
            raise RuntimeError(f'command timed out: {" ".join(command)}')
    if return_code != 0:
        raise RuntimeError(
            f'command failed with exit code {return_code}: {" ".join(command)}\n'
            f'see {log_path}'
        )


if __name__ == '__main__':
    main()

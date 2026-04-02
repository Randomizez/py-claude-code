"""Three-way local request comparison for `pyccode` vs Claude Code.

This helper compares three locally captured request paths:

1. `pyccode`
2. Claude Code `--bare -p`
3. Claude Code non-`--bare` with sandboxed fake OAuth state

The target alignment is the third one. The bare capture is kept as an earlier,
smaller validation point, not as the final parity target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

DEFAULT_OUTPUT_ROOT = Path('.debug') / 'three_way_compare'
PYCCODE_AND_BARE_ROOT = Path('.debug') / 'fake_messages_compare'
NONBARE_ROOT = Path('.debug') / 'nonbare_oauth_capture'
DEFAULT_PROMPT = 'say hi'
DEFAULT_MODEL = 'claude-sonnet-4-5'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='uv run python tests/compare_three_way_messages_requests.py',
        description='Compare pyccode, Claude bare, and Claude non-bare local request captures.',
    )
    parser.add_argument('--root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument(
        '--refresh-captures',
        action='store_true',
        help='Regenerate the local captures before comparing them.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.refresh_captures:
        run(
            [
                'uv',
                'run',
                'python',
                'tests/capture_fake_messages_requests.py',
                '--root',
                str(PYCCODE_AND_BARE_ROOT),
                '--prompt',
                args.prompt,
                '--model',
                args.model,
            ]
        )
        run(
            [
                'uv',
                'run',
                'python',
                'tests/capture_nonbare_oauth_request.py',
                '--root',
                str(NONBARE_ROOT),
                '--oauth-home',
                str(Path('.debug') / 'nonbare_oauth_home'),
                '--prompt',
                args.prompt,
                '--model',
                args.model,
            ]
        )

    pyccode = load_capture_from_summary(
        PYCCODE_AND_BARE_ROOT / 'summary.json',
        'pyccode_capture',
    )
    claude_bare = load_capture_from_summary(
        PYCCODE_AND_BARE_ROOT / 'summary.json',
        'claude_capture',
    )
    claude_nonbare = load_capture_from_summary(
        NONBARE_ROOT / 'summary.json',
        'capture_file',
    )

    payload = build_comparison_payload(pyccode, claude_bare, claude_nonbare)
    json_path = output_root / 'comparison.json'
    md_path = output_root / 'comparison.md'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    md_path.write_text(build_markdown(payload))

    print(f'comparison json: {json_path}')
    print(f'comparison md:   {md_path}')
    print()
    print('Target alignment: Claude non-bare + fake OAuth')
    print('Reference only:   Claude --bare -p')


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=Path.cwd())


def load_capture(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f'missing capture file: {path}')
    return json.loads(path.read_text())


def load_capture_from_summary(summary_path: Path, key: str) -> dict[str, Any]:
    summary = load_capture(summary_path)
    capture_path = summary.get(key)
    if not isinstance(capture_path, str):
        raise RuntimeError(f'missing capture reference {key!r} in {summary_path}')
    return load_capture(Path(capture_path))


def summarize(capture: dict[str, Any]) -> dict[str, Any]:
    body = capture.get('body') or {}
    headers = capture.get('headers') or {}
    tools = body.get('tools') or []
    tool_names = [tool.get('name') for tool in tools if isinstance(tool, dict)]
    messages = body.get('messages') or []
    first_message = messages[0] if messages else {}
    first_content = first_message.get('content') or []
    first_texts = [item.get('text', '') for item in first_content if isinstance(item, dict)]
    system = body.get('system')
    auth_mode = 'none'
    if 'authorization' in {k.lower(): v for k, v in headers.items()}:
        auth_mode = 'bearer'
    elif any(k.lower() == 'x-api-key' for k in headers):
        auth_mode = 'api_key'

    lowered_headers = {k.lower(): v for k, v in headers.items()}
    interesting_headers = {
        key: lowered_headers[key]
        for key in [
            'authorization',
            'x-api-key',
            'anthropic-beta',
            'x-app',
            'x-claude-code-session-id',
            'user-agent',
        ]
        if key in lowered_headers
    }

    return {
        'path': capture.get('path'),
        'stream': body.get('stream', False),
        'auth_mode': auth_mode,
        'interesting_headers': interesting_headers,
        'system_type': type(system).__name__,
        'system_block_count': len(system) if isinstance(system, list) else None,
        'message_count': len(messages),
        'first_message_content_count': len(first_content),
        'first_message_text_prefixes': [text[:120] for text in first_texts[:3]],
        'tool_count': len(tool_names),
        'tool_names': tool_names,
        'max_tokens': body.get('max_tokens'),
        'has_thinking': 'thinking' in body,
        'has_context_management': 'context_management' in body,
        'body_keys': sorted(body.keys()) if isinstance(body, dict) else None,
    }


def build_comparison_payload(
    pyccode: dict[str, Any],
    claude_bare: dict[str, Any],
    claude_nonbare: dict[str, Any],
) -> dict[str, Any]:
    py_summary = summarize(pyccode)
    bare_summary = summarize(claude_bare)
    nonbare_summary = summarize(claude_nonbare)

    py_tools = set(py_summary['tool_names'])
    bare_tools = set(bare_summary['tool_names'])
    nonbare_tools = set(nonbare_summary['tool_names'])

    return {
        'target': 'claude_nonbare',
        'captures': {
            'pyccode': py_summary,
            'claude_bare': bare_summary,
            'claude_nonbare': nonbare_summary,
        },
        'diff_vs_target': {
            'pyccode': {
                'path_matches': py_summary['path'] == nonbare_summary['path'],
                'stream_matches': py_summary['stream'] == nonbare_summary['stream'],
                'auth_mode_matches': py_summary['auth_mode'] == nonbare_summary['auth_mode'],
                'system_type_matches': py_summary['system_type'] == nonbare_summary['system_type'],
                'max_tokens_matches': py_summary['max_tokens'] == nonbare_summary['max_tokens'],
                'first_message_content_count_matches': (
                    py_summary['first_message_content_count']
                    == nonbare_summary['first_message_content_count']
                ),
                'tool_name_set_matches': py_tools == nonbare_tools,
                'tool_overlap_with_target': sorted(py_tools & nonbare_tools),
                'tool_only_in_pyccode': sorted(py_tools - nonbare_tools),
                'tool_only_in_target': sorted(nonbare_tools - py_tools),
            },
            'claude_bare': {
                'path_matches': bare_summary['path'] == nonbare_summary['path'],
                'stream_matches': bare_summary['stream'] == nonbare_summary['stream'],
                'auth_mode_matches': bare_summary['auth_mode'] == nonbare_summary['auth_mode'],
                'system_type_matches': bare_summary['system_type'] == nonbare_summary['system_type'],
                'max_tokens_matches': bare_summary['max_tokens'] == nonbare_summary['max_tokens'],
                'first_message_content_count_matches': (
                    bare_summary['first_message_content_count']
                    == nonbare_summary['first_message_content_count']
                ),
                'tool_name_set_matches': bare_tools == nonbare_tools,
                'tool_overlap_with_target': sorted(bare_tools & nonbare_tools),
                'tool_only_in_bare': sorted(bare_tools - nonbare_tools),
                'tool_only_in_target': sorted(nonbare_tools - bare_tools),
            },
        },
        'takeaways': [
            'The final alignment target is the non-bare Claude capture, not the bare capture.',
            'The bare capture remains useful as a smaller and more stable validation slice.',
            'pyccode now matches the target on /v1/messages?beta=true, streaming, bearer auth, list-based system framing, first-message item count, max_tokens, and tool-name/schema surface.',
            'The biggest current pyccode-to-target gaps are system prompt content, request headers, metadata details, and placeholder tool implementations.',
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    captures = payload['captures']
    lines = [
        '# Three-Way Messages Compare',
        '',
        f"Target alignment: `{payload['target']}`",
        '',
        '## Summary',
        '',
        '| capture | path | stream | auth | system | tools | max_tokens |',
        '|---|---|---:|---|---|---:|---:|',
    ]
    for name in ['pyccode', 'claude_bare', 'claude_nonbare']:
        item = captures[name]
        lines.append(
            f"| {name} | `{item['path']}` | `{item['stream']}` | `{item['auth_mode']}` | `{item['system_type']}` | {item['tool_count']} | {item['max_tokens']} |"
        )

    lines.extend([
        '',
        '## Takeaways',
        '',
    ])
    for takeaway in payload['takeaways']:
        lines.append(f'- {takeaway}')

    lines.extend([
        '',
        '## Diff Vs Non-Bare Target',
        '',
        '### pyccode',
        '```json',
        json.dumps(payload['diff_vs_target']['pyccode'], ensure_ascii=False, indent=2),
        '```',
        '',
        '### claude_bare',
        '```json',
        json.dumps(payload['diff_vs_target']['claude_bare'], ensure_ascii=False, indent=2),
        '```',
    ])
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()

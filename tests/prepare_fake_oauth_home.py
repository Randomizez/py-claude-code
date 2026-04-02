"""Prepare an isolated CLAUDE_CONFIG_DIR with fake OAuth state.

This helper is for local capture work only. It does not contact any upstream.
It writes the minimum credential/config files Claude Code expects on Linux when
reading plaintext secure storage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path('.debug') / 'fake_oauth_home'
DEFAULT_ACCESS_TOKEN = 'fake-access-token'
DEFAULT_REFRESH_TOKEN = 'fake-refresh-token'
DEFAULT_ACCOUNT_UUID = '00000000-0000-4000-8000-000000000001'
DEFAULT_ORG_UUID = '00000000-0000-4000-8000-000000000002'
DEFAULT_USER_ID = '7ae158b782076aed7be664a9b606e2da54e1af3c7135a37b91291fe8a977441d'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='uv run python tests/prepare_fake_oauth_home.py',
        description='Create a sandboxed CLAUDE_CONFIG_DIR with fake OAuth credentials.',
    )
    parser.add_argument(
        '--root',
        default=str(DEFAULT_OUTPUT_ROOT),
        help='Directory used as the fake CLAUDE_CONFIG_DIR.',
    )
    parser.add_argument(
        '--access-token',
        default=DEFAULT_ACCESS_TOKEN,
        help='Fake access token written to .credentials.json.',
    )
    parser.add_argument(
        '--refresh-token',
        default=DEFAULT_REFRESH_TOKEN,
        help='Fake refresh token written to .credentials.json.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    credentials_path, global_config_path = prepare_fake_oauth_home(
        root,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
    )

    print(f'prepared fake oauth home: {root}')
    print(f'credentials: {credentials_path}')
    print(f'global config: {global_config_path}')
    print()
    print('Suggested env for later non-bare capture:')
    print(f'export CLAUDE_CONFIG_DIR={root}')
    print('export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1')
    print('export ANTHROPIC_BASE_URL=http://127.0.0.1:<fake-port>')


def prepare_fake_oauth_home(
    root: Path,
    *,
    access_token: str = DEFAULT_ACCESS_TOKEN,
    refresh_token: str = DEFAULT_REFRESH_TOKEN,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)

    credentials_path = root / '.credentials.json'
    global_config_path = root / '.claude.json'

    credentials = {
        'claudeAiOauth': {
            'accessToken': access_token,
            'refreshToken': refresh_token,
            'expiresAt': 4102444800000,
            'scopes': [
                'user:profile',
                'user:inference',
                'user:sessions:claude_code',
                'user:mcp_servers',
                'user:file_upload',
            ],
            'subscriptionType': 'max',
            'rateLimitTier': 'default_claude_max_20x',
        }
    }
    credentials_path.write_text(json.dumps(credentials, ensure_ascii=False, indent=2))
    credentials_path.chmod(0o600)

    global_config = {
        'numStartups': 1,
        'theme': 'dark',
        'preferredNotifChannel': 'iterm2',
        'verbose': False,
        'hasCompletedOnboarding': True,
        'userID': DEFAULT_USER_ID,
        'oauthAccount': {
            'accountUuid': DEFAULT_ACCOUNT_UUID,
            'emailAddress': 'fake-user@example.com',
            'organizationUuid': DEFAULT_ORG_UUID,
            'organizationName': 'Fake Claude Org',
            'organizationRole': 'owner',
            'workspaceRole': 'owner',
            'displayName': 'Fake Claude User',
            'hasExtraUsageEnabled': True,
            'billingType': 'subscription',
            'accountCreatedAt': '2026-01-01T00:00:00Z',
            'subscriptionCreatedAt': '2026-01-02T00:00:00Z',
        },
    }
    global_config_path.write_text(json.dumps(global_config, ensure_ascii=False, indent=2))
    return credentials_path, global_config_path


if __name__ == '__main__':
    main()

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
from typing import Literal


@dataclass(frozen=True, slots=True)
class ResolvedAuth:
    mode: Literal["bearer", "api_key"]
    value: str
    source: str
    device_id: str | None = None
    account_uuid: str | None = None
    organization_uuid: str | None = None
    subscription_type: str | None = None
    rate_limit_tier: str | None = None
    account_email: str | None = None


def get_claude_config_dir() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".claude").resolve()


def get_global_claude_file() -> Path:
    return get_claude_config_dir() / ".claude.json"


def get_or_create_user_id() -> str:
    config_path = get_global_claude_file()
    config = _read_json_file(config_path)
    user_id = config.get("userID")
    if isinstance(user_id, str) and user_id:
        return user_id

    generated = secrets.token_hex(32)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config["userID"] = generated
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    return generated


def load_claude_oauth_auth() -> ResolvedAuth | None:
    config_dir = get_claude_config_dir()
    credentials = _read_json_file(config_dir / ".credentials.json")
    oauth = credentials.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None

    access_token = oauth.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        return None

    global_config = _read_json_file(config_dir / ".claude.json")
    oauth_account = global_config.get("oauthAccount")
    device_id = get_or_create_user_id()
    account_email = None
    account_uuid = None
    organization_uuid = None
    if isinstance(oauth_account, dict):
        email_address = oauth_account.get("emailAddress")
        if isinstance(email_address, str) and email_address:
            account_email = email_address
        account_uuid_value = oauth_account.get("accountUuid")
        if isinstance(account_uuid_value, str) and account_uuid_value:
            account_uuid = account_uuid_value
        organization_uuid_value = oauth_account.get("organizationUuid")
        if isinstance(organization_uuid_value, str) and organization_uuid_value:
            organization_uuid = organization_uuid_value

    subscription_type = oauth.get("subscriptionType")
    rate_limit_tier = oauth.get("rateLimitTier")
    return ResolvedAuth(
        mode="bearer",
        value=access_token,
        source="claude_config_dir",
        device_id=device_id,
        account_uuid=account_uuid,
        organization_uuid=organization_uuid,
        subscription_type=(
            subscription_type if isinstance(subscription_type, str) else None
        ),
        rate_limit_tier=(
            rate_limit_tier if isinstance(rate_limit_tier, str) else None
        ),
        account_email=account_email,
    )


def resolve_auth(
    *,
    api_key_env: str = "ANTHROPIC_API_KEY",
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN",
    oauth_token_env: str = "CLAUDE_CODE_OAUTH_TOKEN",
) -> ResolvedAuth:
    device_id = get_or_create_user_id()
    auth_token = os.environ.get(auth_token_env, "")
    if auth_token:
        return ResolvedAuth(
            mode="bearer",
            value=auth_token,
            source=auth_token_env,
            device_id=device_id,
        )

    oauth_token = os.environ.get(oauth_token_env, "")
    if oauth_token:
        return ResolvedAuth(
            mode="bearer",
            value=oauth_token,
            source=oauth_token_env,
            device_id=device_id,
        )

    oauth_auth = load_claude_oauth_auth()
    if oauth_auth is not None:
        return oauth_auth

    api_key = os.environ.get(api_key_env, "")
    if api_key:
        return ResolvedAuth(
            mode="api_key",
            value=api_key,
            source=api_key_env,
            device_id=device_id,
        )

    raise RuntimeError(
        "missing auth; set ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN, "
        "CLAUDE_CONFIG_DIR/.credentials.json, or ANTHROPIC_API_KEY"
    )


def _read_json_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

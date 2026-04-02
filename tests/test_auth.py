from __future__ import annotations

import json

from pyccode.auth import get_or_create_user_id, load_claude_oauth_auth, resolve_auth
from tests.prepare_fake_oauth_home import prepare_fake_oauth_home


def test_resolve_auth_prefers_claude_oauth_state_over_api_key(tmp_path, monkeypatch) -> None:
    oauth_home = tmp_path / "oauth-home"
    prepare_fake_oauth_home(oauth_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(oauth_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    resolved = resolve_auth()

    assert resolved.mode == "bearer"
    assert resolved.value == "fake-access-token"
    assert resolved.source == "claude_config_dir"
    assert resolved.account_email == "fake-user@example.com"
    assert resolved.account_uuid == "00000000-0000-4000-8000-000000000001"
    assert resolved.subscription_type == "max"


def test_resolve_auth_prefers_explicit_env_bearer_token(tmp_path, monkeypatch) -> None:
    oauth_home = tmp_path / "oauth-home"
    prepare_fake_oauth_home(oauth_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(oauth_home))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-access-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    resolved = resolve_auth()

    assert resolved.mode == "bearer"
    assert resolved.value == "env-access-token"
    assert resolved.source == "ANTHROPIC_AUTH_TOKEN"
    assert resolved.device_id == "7ae158b782076aed7be664a9b606e2da54e1af3c7135a37b91291fe8a977441d"


def test_load_claude_oauth_auth_returns_none_without_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert load_claude_oauth_auth() is None


def test_get_or_create_user_id_persists_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    first = get_or_create_user_id()
    second = get_or_create_user_id()

    assert len(first) == 64
    assert first == second
    config = json.loads((tmp_path / ".claude.json").read_text())
    assert config["userID"] == first


def test_resolve_auth_with_api_key_generates_device_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    resolved = resolve_auth()

    assert resolved.mode == "api_key"
    assert resolved.device_id is not None
    assert len(resolved.device_id) == 64
    config = json.loads((tmp_path / ".claude.json").read_text())
    assert config["userID"] == resolved.device_id

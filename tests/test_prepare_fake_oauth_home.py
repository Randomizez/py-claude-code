from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_prepare_fake_oauth_home_writes_expected_files(tmp_path) -> None:
    script = Path(__file__).with_name("prepare_fake_oauth_home.py")
    target = tmp_path / "fake-home"
    subprocess.run(
        [sys.executable, str(script), "--root", str(target)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    credentials = json.loads((target / ".credentials.json").read_text())
    global_config = json.loads((target / ".claude.json").read_text())

    assert credentials["claudeAiOauth"]["accessToken"] == "fake-access-token"
    assert "user:profile" in credentials["claudeAiOauth"]["scopes"]
    assert credentials["claudeAiOauth"]["subscriptionType"] == "max"

    assert global_config["hasCompletedOnboarding"] is True
    assert global_config["oauthAccount"]["emailAddress"] == "fake-user@example.com"

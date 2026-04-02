from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer

from pyccode.doctor import collect_doctor_report, format_doctor_report
from tests.fake_anthropic_messages_server import CaptureStore, build_handler
from tests.prepare_fake_oauth_home import prepare_fake_oauth_home


async def test_collect_doctor_report_static_checks(tmp_path, monkeypatch) -> None:
    oauth_home = tmp_path / "oauth-home"
    prepare_fake_oauth_home(oauth_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(oauth_home))

    report = await collect_doctor_report(
        model="claude-fake",
        base_url="http://127.0.0.1:9999",
        skip_live=True,
    )

    assert report.model == "claude-fake"
    assert report.auth_mode == "bearer"
    assert report.auth_source == "claude_config_dir"
    assert report.device_id == "7ae158b782076aed7be664a9b606e2da54e1af3c7135a37b91291fe8a977441d"
    assert "messages_url:" in format_doctor_report(report)
    checks = {check.name: check for check in report.checks}
    assert checks["global_config"].ok is True
    assert checks["credentials"].ok is True
    assert checks["auth"].ok is True


async def test_collect_doctor_report_live_check_with_fake_server(tmp_path, monkeypatch) -> None:
    oauth_home = tmp_path / "oauth-home"
    prepare_fake_oauth_home(oauth_home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(oauth_home))

    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "claude-fake", "OK"),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        report = await collect_doctor_report(
            model="claude-fake",
            base_url=f"http://127.0.0.1:{httpd.server_port}",
            skip_live=False,
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

    assert report.ok is True
    assert report.live_output_text == "OK"
    checks = {check.name: check for check in report.checks}
    assert checks["dns"].ok is True
    assert checks["transport"].ok is True
    assert checks["live"].ok is True

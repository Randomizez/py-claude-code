from __future__ import annotations

import asyncio
import json
import socket
import ssl
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from .auth import get_claude_config_dir, get_global_claude_file, resolve_auth
from .model import AnthropicMessagesConfig, AnthropicMessagesModelClient
from .protocol import ConversationMessage, Prompt, SystemTextBlock


@dataclass(slots=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DoctorReport:
    ok: bool
    config_dir: str
    global_config_path: str
    credentials_path: str
    model: str | None = None
    base_url: str | None = None
    messages_url: str | None = None
    auth_mode: str | None = None
    auth_source: str | None = None
    device_id: str | None = None
    checks: list[DoctorCheck] = field(default_factory=list)
    live_output_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_doctor_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="pyccode doctor",
        description="Diagnose pyccode model, auth, and endpoint connectivity.",
    )
    parser.add_argument("--model", default=None, help="Anthropic model name.")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible base URL.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout used for network and live model checks.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip the live model request and only run static/network checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full doctor report as JSON.",
    )
    return parser


async def collect_doctor_report(
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 120.0,
    skip_live: bool = False,
) -> DoctorReport:
    config_dir = get_claude_config_dir()
    global_config = get_global_claude_file()
    credentials = config_dir / ".credentials.json"
    report = DoctorReport(
        ok=False,
        config_dir=str(config_dir),
        global_config_path=str(global_config),
        credentials_path=str(credentials),
    )
    checks = report.checks
    checks.append(
        DoctorCheck(
            "global_config",
            global_config.exists(),
            str(global_config) if global_config.exists() else f"missing: {global_config}",
        )
    )
    checks.append(
        DoctorCheck(
            "credentials",
            True,
            str(credentials) if credentials.exists() else f"missing (optional): {credentials}",
        )
    )

    try:
        config = AnthropicMessagesConfig.from_env(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        checks.append(DoctorCheck("config", False, f"{type(exc).__name__}: {exc}"))
        return report

    report.model = config.model
    report.base_url = config.base_url
    client = AnthropicMessagesModelClient(config)
    report.messages_url = client._messages_url()

    try:
        auth = config.resolve_auth()
    except Exception as exc:
        checks.append(DoctorCheck("auth", False, f"{type(exc).__name__}: {exc}"))
        return _finalize_report(report)

    report.auth_mode = auth.mode
    report.auth_source = auth.source
    report.device_id = auth.device_id
    checks.append(
        DoctorCheck(
            "auth",
            True,
            f"mode={auth.mode} source={auth.source}",
        )
    )
    checks.append(
        DoctorCheck(
            "device_id",
            bool(auth.device_id),
            auth.device_id or "missing device_id",
        )
    )

    parsed_url = urllib.parse.urlsplit(report.messages_url)
    host = parsed_url.hostname or ""
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    proxies = requests.utils.get_environ_proxies(report.messages_url)
    checks.append(DoctorCheck("proxy", True, _proxy_detail(proxies)))

    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        checks.append(DoctorCheck("dns", False, f"{host}:{port} -> {exc}"))
        return _finalize_report(report)

    resolved_addresses = sorted(
        {
            result[4][0]
            for result in addresses
            if len(result) >= 5 and isinstance(result[4], tuple) and result[4]
        }
    )
    checks.append(
        DoctorCheck(
            "dns",
            True,
            f"{host}:{port} -> {', '.join(resolved_addresses) or 'resolved'}",
        )
    )

    if proxies:
        checks.append(
            DoctorCheck(
                "transport",
                True,
                "skipped direct probe because requests will use environment proxy settings",
            )
        )
    else:
        ok, detail = await asyncio.to_thread(
            _probe_transport,
            parsed_url.scheme,
            host,
            port,
            timeout_seconds,
        )
        checks.append(DoctorCheck("transport", ok, detail))

    if skip_live:
        return _finalize_report(report)

    ok, detail, output_text = await _run_live_check(config)
    report.live_output_text = output_text
    checks.append(DoctorCheck("live", ok, detail))
    return _finalize_report(report)


async def run_doctor_cli(args) -> int:
    report = await collect_doctor_report(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        skip_live=args.skip_live,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_doctor_report(report))
    return 0 if report.ok else 1


def format_doctor_report(report: DoctorReport) -> str:
    lines = [
        f"config_dir: {report.config_dir}",
        f"global_config: {report.global_config_path}",
        f"credentials: {report.credentials_path}",
    ]
    if report.model is not None:
        lines.append(f"model: {report.model}")
    if report.base_url is not None:
        lines.append(f"base_url: {report.base_url}")
    if report.messages_url is not None:
        lines.append(f"messages_url: {report.messages_url}")
    if report.auth_mode is not None:
        lines.append(f"auth_mode: {report.auth_mode}")
    if report.auth_source is not None:
        lines.append(f"auth_source: {report.auth_source}")
    if report.device_id is not None:
        lines.append(f"device_id: {report.device_id}")
    if report.live_output_text is not None:
        lines.append(f"live_output: {report.live_output_text}")
    for check in report.checks:
        status = "ok" if check.ok else "fail"
        lines.append(f"{check.name}: {status} - {check.detail}")
    lines.append(f"overall: {'ok' if report.ok else 'fail'}")
    return "\n".join(lines)


def _probe_transport(
    scheme: str,
    host: str,
    port: int,
    timeout_seconds: float,
) -> tuple[bool, str]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
            if scheme == "https":
                with ssl.create_default_context().wrap_socket(
                    sock,
                    server_hostname=host,
                ):
                    pass
    except OSError as exc:
        elapsed = time.perf_counter() - started
        return False, f"{scheme.upper()} {host}:{port} failed after {elapsed:.2f}s: {exc}"
    elapsed = time.perf_counter() - started
    label = "tls" if scheme == "https" else "tcp"
    return True, f"{label} {host}:{port} connected in {elapsed:.2f}s"


def _proxy_detail(proxies: dict[str, str]) -> str:
    if not proxies:
        return "not configured"
    return ", ".join(
        f"{key}={_redact_proxy_url(value)}" for key, value in sorted(proxies.items())
    )


def _redact_proxy_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, parsed.query, parsed.fragment))


async def _run_live_check(
    config: AnthropicMessagesConfig,
) -> tuple[bool, str, str | None]:
    client = AnthropicMessagesModelClient(config)
    prompt = Prompt(
        system=(SystemTextBlock(text="Reply with exactly OK."),),
        messages=(ConversationMessage.user_text("Reply with exactly OK."),),
        tools=(),
        max_tokens=16,
        stream=True,
    )
    started = time.perf_counter()
    try:
        response = await client.complete(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, f"failed after {elapsed:.2f}s: {type(exc).__name__}: {exc}", None
    elapsed = time.perf_counter() - started
    return True, f"completed in {elapsed:.2f}s", response.message.text_content() or None


def _finalize_report(report: DoctorReport) -> DoctorReport:
    report.ok = all(check.ok for check in report.checks)
    return report

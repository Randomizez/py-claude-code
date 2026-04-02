from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from uuid import uuid4

ILLEGAL_ENV_VAR_PREFIX = "CODEX_"
DOTENV_FILENAME = ".env"
_LOADED_CODEX_DOTENV_HOMES: set[str] = set()


def uuid7_string() -> str:
    # Python 3.10 stdlib has no uuid7; a random UUID is sufficient for
    # local ids in this minimal extraction.
    return str(uuid4())


def load_codex_dotenv(config_path: str | Path) -> None:
    codex_home = str(Path(config_path).resolve().parent)
    if codex_home in _LOADED_CODEX_DOTENV_HOMES:
        return

    dotenv_path = Path(codex_home) / DOTENV_FILENAME
    if not dotenv_path.is_file():
        _LOADED_CODEX_DOTENV_HOMES.add(codex_home)
        return

    for key, value in parse_dotenv(dotenv_path.read_text()).items():
        if key.upper().startswith(ILLEGAL_ENV_VAR_PREFIX):
            continue
        os.environ[key] = value

    _LOADED_CODEX_DOTENV_HOMES.add(codex_home)


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = parse_dotenv_value(raw_value.strip())
    return values


def parse_dotenv_value(raw_value: str) -> str:
    if not raw_value:
        return ""

    quote = raw_value[0]
    if quote in {'"', "'"}:
        if len(raw_value) >= 2 and raw_value[-1] == quote:
            inner = raw_value[1:-1]
        else:
            inner = raw_value[1:]
        if quote == "'":
            return inner
        return bytes(inner, "utf-8").decode("unicode_escape")

    if " #" in raw_value:
        raw_value = raw_value.split(" #", 1)[0].rstrip()
    return raw_value


def build_user_agent(originator: str) -> str:
    version = get_package_version()
    terminal = get_terminal_user_agent_token()
    os_name, os_version = get_os_info()
    arch = platform.machine() or "unknown"
    return f"{originator}/{version} ({os_name} {os_version}; {arch}) {terminal}"


def get_package_version() -> str:
    try:
        return importlib.metadata.version("pyccode")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def get_os_info() -> tuple[str, str]:
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        values: dict[str, str] = {}
        for line in os_release.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
        name = values.get("NAME")
        version = values.get("VERSION_ID")
        if name and version:
            return name, version
    return platform.system(), platform.release()


def get_terminal_user_agent_token() -> str:
    term = os.environ.get("TERM")
    if term:
        return term
    return "unknown"

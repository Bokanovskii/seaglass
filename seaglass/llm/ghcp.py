from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

COPILOT_BIN = 'copilot'
DEFAULT_TIMEOUT_S = 180
_JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)
_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)
COMMON_PATHS = [Path('~/.local/bin/copilot').expanduser(), Path('/opt/homebrew/bin/copilot'), Path('/usr/local/bin/copilot')]


class GhcpError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class GhcpAvailability:
    available: bool
    bin_path: str | None = None
    version: str | None = None
    reason: str | None = None


def call_ghcp(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, bin_path: str = COPILOT_BIN) -> str:
    result = subprocess.run(
        [bin_path, '-p', prompt, '-s', '--no-color', '--log-level', 'none'],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise GhcpError(f'copilot -p exited {result.returncode}: {result.stderr.strip()[:2000]}')
    return result.stdout.strip()


def call_ghcp_json(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, bin_path: str = COPILOT_BIN) -> Optional[list]:
    raw = call_ghcp(prompt, timeout_s=timeout_s, bin_path=bin_path)
    match = _JSON_ARRAY_RE.search(raw)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def call_ghcp_json_object(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, bin_path: str = COPILOT_BIN) -> Optional[dict]:
    raw = call_ghcp(prompt, timeout_s=timeout_s, bin_path=bin_path)
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def detect_ghcp(explicit_bin: str | None = None) -> GhcpAvailability:
    candidates = []
    if explicit_bin:
        candidates.append(Path(explicit_bin).expanduser())
    which = shutil.which(COPILOT_BIN)
    if which:
        candidates.append(Path(which))
    candidates.extend(COMMON_PATHS)
    seen = set()
    for candidate in candidates:
        real = str(candidate.expanduser().resolve()) if candidate.expanduser().exists() else str(candidate.expanduser())
        if real in seen:
            continue
        seen.add(real)
        if not Path(real).exists():
            continue
        try:
            result = subprocess.run([real, '--version'], capture_output=True, text=True, timeout=5)
        except Exception as exc:
            return GhcpAvailability(False, bin_path=real, reason=str(exc))
        if result.returncode == 0:
            return GhcpAvailability(True, bin_path=real, version=result.stdout.strip() or result.stderr.strip())
    return GhcpAvailability(False, reason='copilot CLI not detected')

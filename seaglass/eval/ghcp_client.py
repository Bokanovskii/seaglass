"""`eval/ghcp_client.py` — thin wrapper around the GitHub Copilot CLI's
non-interactive scripting mode (`copilot -p "<prompt>" -s`), the
project's sole LLM interface per the user's explicit decision ("if we
need to make llm requests it will be through ghcp").

**Batch, don't call per-item.** A spike (see ADDENDUM.md §14) measured
~4.4s of *fixed* per-invocation overhead for `copilot -p` even on a
one-word answer -- at the ~300 candidate chunks EVALUATION.md §4
generates questions for, one call per chunk would cost 20+ minutes in
overhead alone. This module always batches many items into one prompt
per call.

Deliberately narrow: this is not a general "call an LLM" utility, it's
scoped to "ask ghcp to answer a structured batch prompt and return
parsed JSON" -- the one shape eval/generate.py needs.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import List, Optional

COPILOT_BIN = "copilot"
DEFAULT_TIMEOUT_S = 180


class GhcpError(RuntimeError):
    pass


def call_ghcp(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> str:
    """Run one non-interactive `copilot -p` call, no tool use needed for
    text generation (no `--allow-all-tools` -- generation-only prompts
    shouldn't be granted shell/file/network access).
    """
    result = subprocess.run(
        [COPILOT_BIN, "-p", prompt, "-s", "--no-color", "--log-level", "none"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise GhcpError(f"copilot -p exited {result.returncode}: {result.stderr.strip()[:2000]}")
    return result.stdout.strip()


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def call_ghcp_json(prompt: str, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> Optional[list]:
    """Call ghcp and parse a JSON array out of the response, tolerating
    conversational wrapper text or markdown code fences around the
    array (LLMs reliably add these even when told not to). Returns
    `None` (never raises) if no parseable JSON array is found -- callers
    should treat that as "this batch failed, retry or skip", not a hard
    error, since a single malformed batch shouldn't abort a 300-chunk run.
    """
    raw = call_ghcp(prompt, timeout_s=timeout_s)
    match = _JSON_ARRAY_RE.search(raw)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

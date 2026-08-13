from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from seaglass.app.config import APP_DB_PATH
from seaglass.search.parse import DATE_PAD_DAYS, ParsedQuery


@dataclasses.dataclass(frozen=True)
class AssistResult:
    status: str
    parse: dict | None = None
    changes: list[str] = dataclasses.field(default_factory=list)
    confidence: float | None = None
    reason: str | None = None


class AssistCircuitBreaker:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.failures = 0

    @property
    def open(self) -> bool:
        return self.failures >= self.threshold

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1


def should_assist(mode: str, parsed: ParsedQuery) -> bool:
    if mode == 'force':
        return True
    if mode != 'auto':
        return False
    token_count = len(parsed.raw.split())
    return token_count >= 4 and not parsed.people_participant and parsed.date_from is None and not parsed.has_media


def build_prompt(query: str, *, today: str, weekday: str, tz_name: str) -> str:
    payload = json.dumps(query)
    return (
        f'You are a query parser for a personal iMessage search engine. Today is {today} ({weekday}), local timezone {tz_name}.\n\n'
        'Given one search query, extract structured filters and return ONLY a JSON object, no prose, no code fence:\n'
        '{"semantic": str, "people": [str], "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, '
        '"has_media": true|false|null, "is_group": true|false|null, "expansions": [str], "confidence": 0.0-1.0}\n\n'
        'Rules:\n'
        '- "semantic": the topical content only, with person names and date expressions REMOVED. Never empty; if nothing remains, repeat the original query.\n'
        '- "people": names of people who were PARTICIPANTS in the conversation (senders/recipients), NOT people merely mentioned in the text. Copy the name spans verbatim from the query; do not guess full names, do not invent surnames.\n'
        '- dates: resolve relative expressions to concrete inclusive dates. Null if the query implies no time constraint.\n'
        '- "has_media": true only if the user is looking FOR an image/video/attachment; false if a media word appears only incidentally; null if no signal.\n'
        '- "is_group": true if the query implies a group chat, false if a one-on-one, null otherwise.\n'
        '- "expansions": up to 5 extra single-word keyword synonyms to help a BM25 keyword search. No phrases. Empty if none help.\n'
        '- "confidence": your confidence in this parse.\n\n'
        f'Query: {payload}\n'
    )


def merge_ghcp_parse(deterministic: ParsedQuery, raw: dict, contact_index, corpus_bounds) -> tuple[ParsedQuery, list[str], list[str]]:
    from dataclasses import replace

    merged = replace(deterministic)
    changes: list[str] = []
    expansions: list[str] = []

    people = []
    for name in raw.get('people') or []:
        found = contact_index.handle_ids_for_names(name, threshold=85.0) if contact_index is not None else []
        people.extend(found)
    if people:
        merged.people_participant = people
        changes.append('people')

    date_from, date_to = _parse_dates(raw.get('date_from'), raw.get('date_to'), corpus_bounds)
    if date_from is not None and date_to is not None:
        merged.date_from = date_from - DATE_PAD_DAYS * 86400
        merged.date_to = date_to + DATE_PAD_DAYS * 86400
        changes.append('date range')

    if isinstance(raw.get('has_media'), bool):
        merged.has_media = raw['has_media']
        changes.append('media filter')

    if isinstance(raw.get('is_group'), bool):
        setattr(merged, 'is_group', raw['is_group'])
        changes.append('conversation type')

    semantic = (raw.get('semantic') or '').strip()
    if semantic and _semantic_overlaps(deterministic.raw, semantic):
        merged.semantic = semantic
        changes.append('semantic query')

    for token in raw.get('expansions') or []:
        token = str(token).strip().lower()
        if token.isalnum() and token not in merged.semantic.lower().split() and token not in expansions:
            expansions.append(token)
        if len(expansions) == 5:
            break
    return merged, changes, expansions


def _semantic_overlaps(raw_query: str, semantic: str) -> bool:
    raw_terms = {term.lower() for term in raw_query.split() if len(term) > 2}
    semantic_terms = {term.lower() for term in semantic.split() if len(term) > 2}
    return bool(raw_terms & semantic_terms)


def _parse_dates(date_from: str | None, date_to: str | None, corpus_bounds: tuple[float, float]) -> tuple[float | None, float | None]:
    if not date_from or not date_to:
        return None, None
    try:
        start = datetime.strptime(date_from, '%Y-%m-%d').timestamp()
        end = datetime.strptime(date_to, '%Y-%m-%d').timestamp()
    except ValueError:
        return None, None
    if start > end:
        return None, None
    min_ts, max_ts = corpus_bounds
    if start < min_ts or end > max_ts:
        return None, None
    return start, end


def ensure_cache(path: Path = APP_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        'CREATE TABLE IF NOT EXISTS ghcp_cache (key TEXT PRIMARY KEY, query TEXT, response_json TEXT, created_at REAL)'
    )
    con.commit()
    return con


def cache_key(query: str, *, prompt_version: str, today: str, aliases_mtime: float) -> str:
    digest = hashlib.sha256()
    digest.update(prompt_version.encode())
    digest.update(query.strip().lower().encode())
    digest.update(today.encode())
    digest.update(str(aliases_mtime).encode())
    return digest.hexdigest()


def get_cached_parse(con: sqlite3.Connection, key: str, ttl_days: int = 30) -> dict | None:
    row = con.execute('SELECT response_json FROM ghcp_cache WHERE key = ?', (key,)).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def put_cached_parse(con: sqlite3.Connection, key: str, query: str, payload: dict) -> None:
    con.execute(
        'INSERT OR REPLACE INTO ghcp_cache(key, query, response_json, created_at) VALUES (?, ?, ?, strftime("%s","now"))',
        (key, query, json.dumps(payload)),
    )
    con.commit()

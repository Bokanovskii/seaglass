from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from seaglass.app.config import APP_DB_PATH
from seaglass.search.parse import (
    DATE_PAD_DAYS,
    ParsedQuery,
    _FILLER_WORDS,
    _NOT_A_NAME,
    _PARTICIPANT_PATTERN,
    _SENDER_PATTERNS,
    _name_candidates,
    resolve_name,
)


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
    """Is the LLM round trip worth its latency for this query?

    `force` always asks. `auto` asks only when the deterministic parse looks
    inadequate for the query it was given -- the fast paths (filter-only and
    confidently-parsed queries) answer in well under a second, and spending
    seconds of Copilot latency to re-derive filters the regex already found
    makes the common case worse for nothing.

    The old rule ("4+ words and no filters at all") had it backwards: it
    read *any* extracted filter as "no help needed", so "recent messages
    from kaya" -- where the date resolved but the person did not -- never
    asked, which is exactly the query that needed asking.
    """
    if mode == 'force':
        return True
    if mode != 'auto':
        return False
    if len(parsed.raw.split()) < 3:
        return False
    if _names_someone_unresolved(parsed):
        return True
    # Nothing structured came out of a query long enough to contain
    # structure: either it is purely topical (assist can still supply
    # keyword expansions) or the parser missed something.
    has_filters = bool(parsed.people_participant or parsed.people_sender) or parsed.date_from is not None or parsed.has_media
    return not has_filters and len(parsed.raw.split()) >= 4


def _names_someone_unresolved(parsed: ParsedQuery) -> bool:
    """A name-shaped word survived into the residual with no person filter
    to show for it -- a misspelling, a nickname, or a contact we cannot see."""
    if parsed.people_participant or parsed.people_sender or parsed.from_me:
        return False
    for pattern in _SENDER_PATTERNS + [_PARTICIPANT_PATTERN]:
        for match in pattern.finditer(parsed.raw):
            for candidate in _name_candidates(match.group(1)):
                word = candidate.lower()
                if word in _NOT_A_NAME or all(w in _FILLER_WORDS for w in word.split()):
                    continue
                return True
    return False


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


@dataclasses.dataclass
class MergedParse:
    """The result of folding a Copilot parse into the deterministic one."""

    parse: ParsedQuery
    changes: list[str] = dataclasses.field(default_factory=list)
    expansions: list[str] = dataclasses.field(default_factory=list)
    unresolved_people: list[str] = dataclasses.field(default_factory=list)

    def __iter__(self):
        # Kept unpackable as (merged, changes, expansions).
        return iter((self.parse, self.changes, self.expansions))


def merge_ghcp_parse(deterministic: ParsedQuery, raw: dict, contact_index, corpus_bounds) -> MergedParse:
    from dataclasses import replace

    merged = replace(deterministic)
    changes: list[str] = []
    expansions: list[str] = []

    people = []
    unresolved: list[str] = []
    for name in raw.get('people') or []:
        found, _resolved = resolve_name(str(name), contact_index)
        if found:
            people.extend(found)
        else:
            # Saying "Copilot read this as people: kaya" while no contact
            # could be resolved promises a filter that was never applied.
            unresolved.append(str(name))
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
    return MergedParse(parse=merged, changes=changes, expansions=expansions, unresolved_people=unresolved)


def describe_parse(merged: ParsedQuery, contact_index, unresolved: list[str] | None = None) -> str:
    """A one-line summary of what was actually applied.

    Built from the *merged* parse rather than Copilot's raw JSON, so the
    banner cannot claim a person filter that resolved to nobody.
    """
    parts: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    for handle in list(merged.people_sender) + list(merged.people_participant):
        name = contact_index.resolve_handle(handle) if contact_index is not None else None
        name = name or handle
        if name not in seen:
            seen.add(name)
            names.append(name)
    if names:
        parts.append(', '.join(names[:3]))
    if merged.from_me:
        parts.append('sent by me')
    if merged.date_from is not None and merged.date_to is not None:
        fmt = '%b %-d'
        parts.append(
            datetime.fromtimestamp(merged.date_from).strftime(fmt)
            + ' – '
            + datetime.fromtimestamp(merged.date_to).strftime(fmt)
        )
    if merged.has_media:
        parts.append('with media')
    is_group = getattr(merged, 'is_group', None)
    if is_group is not None:
        parts.append('group chats' if is_group else '1:1 chats')
    if merged.semantic.strip():
        parts.append(f'“{merged.semantic.strip()}”')
    else:
        parts.append('most recent')
    if unresolved:
        parts.append('no contact matched ' + ', '.join(unresolved))
    return ' · '.join(parts)


def assisted_search_args(merged: ParsedQuery, base_filters):
    """Turn a merged parse into the (text, filters) pair `engine.search`
    actually honours.

    The engine re-parses whatever text it is given, so handing it the
    original query text threw the whole merge away and applied nothing but
    the keyword expansions -- the banner said "people: kaya · Aug 6 → Aug 13"
    while the results were the un-assisted ones. Passing the topical residual
    as the text and every extracted filter as *filters* is what makes the
    assisted parse take effect.

    Filters the user set explicitly in the UI win: they are a deliberate act,
    the parse is an inference.
    """
    from dataclasses import replace as _replace

    filters = _replace(base_filters)
    if not filters.people_handles and not filters.people_names and merged.people_participant:
        filters.people_handles = list(merged.people_participant)
    if not filters.people_sender and merged.people_sender:
        filters.people_sender = list(merged.people_sender)
    if filters.from_me is None:
        filters.from_me = merged.from_me
    if filters.date_from is None and filters.date_to is None:
        filters.date_from, filters.date_to = merged.date_from, merged.date_to
    if filters.has_media is None and merged.has_media:
        filters.has_media = True
    if filters.is_group is None:
        filters.is_group = getattr(merged, 'is_group', None)
    return merged.semantic, filters


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
    # Created once in SearchAssistManager and read/written from whichever
    # thread handles the request (uvicorn's event-loop thread, which is a
    # background thread distinct from the one that constructed this
    # connection) -- sqlite3 connections are thread-affine by default, so
    # this must opt out of that check. All access here is effectively
    # single-threaded in practice (uvicorn runs one event loop), so no
    # additional locking is needed.
    con = sqlite3.connect(path, check_same_thread=False)
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

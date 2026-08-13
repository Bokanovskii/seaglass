"""`search/parse.py` — deterministic query → structured filter + semantic
residual, per PLAN.md §6 Phase 4. No model involved: `dateparser` for
dates, keyword sets for media, `rapidfuzz` over the contact index for
people. Anything unparsed stays in the semantic residual -- the parser
must never be the single point of failure ("fail open").
"""

from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Tuple

from dateparser.search import search_dates

from seaglass.imessage.contacts import ContactIndex

# "photo/picture/video/screenshot" media keyword set (PLAN.md §6 Phase 4).
_MEDIA_KEYWORDS = {"photo", "photos", "picture", "pictures", "pic", "pics",
                    "video", "videos", "screenshot", "screenshots", "image", "images"}

# Preposition heuristic (PLAN.md §6 Phase 4): "from"/"with" implies
# participant filter; "about"/"re"/bare mention stays in the residual.
# NOTE (BUG-10 fix): only the preposition itself is case-insensitive here
# (scoped inline flag) -- the capitalized-name group must stay
# case-SENSITIVE, or the whole point of requiring a capitalized name is
# defeated and ordinary lowercase words after "with"/"from" ("with the
# trip", "from yesterday") get misread as a person's name.
_PARTICIPANT_PATTERN = re.compile(
    r"\b(?i:from|with)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)"
)

DATE_PAD_DAYS = 3
PEOPLE_FUZZY_THRESHOLD = 85.0  # rapidfuzz score, 0-100; prefer no filter to a wrong one

# `dateparser.search.search_dates` has known false positives on short,
# common words it misreads as date fragments (e.g. "we" -> "Wed[nesday]").
# Only trust a match if it contains a digit or an unambiguous date/time
# vocabulary word; a bare pronoun/preposition is almost certainly a
# false positive, not a date, and letting it through would silently
# corrupt the date filter on ordinary queries like "what did we say...".
_UNAMBIGUOUS_DATE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "today", "tomorrow", "yesterday", "tonight", "ago", "last", "next",
    "week", "weekend", "month", "year", "spring", "summer", "fall", "winter",
    "morning", "afternoon", "evening", "noon", "midnight",
}

# Month names that double as common non-date English words (BUG-10:
# "may i borrow the car" was misread as a May date filter). These are
# only trusted as a date match when the *full query* has some other
# corroborating date signal (a digit, or a second unambiguous date word)
# -- a single bare occurrence of one of these words alone is treated as
# the ordinary word, not a month reference.
_AMBIGUOUS_DATE_WORDS = {"may", "march"}


def _looks_like_a_real_date_match(substring: str, full_text: str = "") -> bool:
    has_digit = any(char.isdigit() for char in substring)
    if has_digit:
        return True
    words = set(re.findall(r"[a-zA-Z]+", substring.lower()))
    unambiguous_hit = words & _UNAMBIGUOUS_DATE_WORDS - _AMBIGUOUS_DATE_WORDS
    if unambiguous_hit:
        return True
    ambiguous_hit = words & _AMBIGUOUS_DATE_WORDS
    if not ambiguous_hit:
        return False
    # Corroborate against the rest of the query: a digit anywhere, or a
    # second unambiguous date word, makes it plausible this really is a
    # month reference rather than the common verb/modal usage.
    full_words = set(re.findall(r"[a-zA-Z]+", full_text.lower()))
    has_other_digit = any(char.isdigit() for char in full_text)
    has_other_date_word = bool((full_words - ambiguous_hit) & _UNAMBIGUOUS_DATE_WORDS)
    return has_other_digit or has_other_date_word


@dataclasses.dataclass
class ParsedQuery:
    raw: str
    semantic: str  # the residual passed to the embedder/FTS
    people_participant: List[str] = dataclasses.field(default_factory=list)
    date_from: Optional[float] = None  # unix seconds, inclusive, padded
    date_to: Optional[float] = None
    has_media: bool = False
    is_group: Optional[bool] = None
    chat_ids: Optional[List[int]] = None


def _extract_media_filter(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & _MEDIA_KEYWORDS)


def _extract_date_range(text: str) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Returns (date_from, date_to, matched_substrings), or (None, None, [])
    if nothing plausible was found.
    """
    results = search_dates(text, settings={"PREFER_DATES_FROM": "past"})
    if not results:
        return None, None, []
    results = [(substring, dt) for substring, dt in results if _looks_like_a_real_date_match(substring, text)]
    if not results:
        return None, None, []
    matched_substrings = [substring for substring, _ in results]
    dates = [dt for _, dt in results]
    earliest = min(dates)
    latest = max(dates)
    pad = DATE_PAD_DAYS * 86400
    date_from = earliest.timestamp() - pad
    date_to = latest.timestamp() + pad
    return date_from, date_to, matched_substrings


def _extract_participants(
    text: str, contact_index: Optional[ContactIndex]
) -> Tuple[List[str], List[str]]:
    """Returns (handle_ids, matched_substrings). Only applies a filter above
    `PEOPLE_FUZZY_THRESHOLD` -- ambiguous matches fail open into the
    residual rather than risk a wrong, silently zero-result filter.
    """
    if contact_index is None:
        return [], []
    handle_ids: List[str] = []
    matched: List[str] = []
    for match in _PARTICIPANT_PATTERN.finditer(text):
        name_guess = match.group(1)
        found_handles = contact_index.handle_ids_for_names(name_guess, threshold=PEOPLE_FUZZY_THRESHOLD)
        if found_handles:
            handle_ids.extend(found_handles)
            matched.append(match.group(0))
    return handle_ids, matched


def parse_query(text: str, contact_index: Optional[ContactIndex] = None) -> ParsedQuery:
    """Parse a free-text query into structured filters plus a semantic
    residual. Extracted substrings are removed from the residual; anything
    not confidently extracted stays in it untouched.
    """
    has_media = _extract_media_filter(text)
    date_from, date_to, date_substrings = _extract_date_range(text)
    people_participant, people_substrings = _extract_participants(text, contact_index)

    residual = text
    for substring in date_substrings + people_substrings:
        residual = residual.replace(substring, " ")
    residual = re.sub(r"\s+", " ", residual).strip()

    return ParsedQuery(
        raw=text,
        semantic=residual or text,  # never emit an empty residual
        people_participant=people_participant,
        date_from=date_from,
        date_to=date_to,
        has_media=has_media,
    )

"""`search/parse.py` — deterministic query → structured filter + semantic
residual, per PLAN.md §6 Phase 4. No model involved: `dateparser` for
dates, keyword sets for media, `rapidfuzz` over the contact index for
people. Anything unparsed stays in the semantic residual -- the parser
must never be the single point of failure ("fail open").
"""

from __future__ import annotations

import dataclasses
import datetime as dt
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

# "from Kaya" and "what did Vamski say" ask for messages Kaya/Vamski
# *wrote*. The participant filter above only narrows to chats they are in,
# so "messages from Jakie last week" answered with a group chat Jakie is in
# and showed someone else's messages -- a confident, wrong answer.
# First person. "what did I say about the lease" is a sender filter too,
# and without one it answered with whatever the other person said back --
# confidently, and about the right topic, which is what made it hard to
# spot.
_SELF_PATTERNS = [
    re.compile(r"\b(?:did\s+)?i\s+(?:say|send|sent|said|tell|told|text|texted|write|wrote|ask|asked|mention|mentioned)\b", re.I),
    re.compile(r"\b(?:messages?|texts?|things?)\s+i\s+(?:sent|said|wrote|texted)\b", re.I),
    re.compile(r"\bmy\s+(?:messages?|texts?|replies|reply)\b", re.I),
    re.compile(r"\bfrom\s+me\b", re.I),
]

# "I" is capitalised like a name, so the sender patterns below would hand it
# to the contact index and fuzzy-match a real person.
_NOT_A_NAME = frozenset({"i", "me", "you", "we", "they", "who", "someone", "anyone"})

_SENDER_PATTERNS = [
    re.compile(r"\bfrom\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)"),
    re.compile(r"\b([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+(?:sent|said|says|texted|wrote|mentioned)\b"),
    re.compile(r"\bdid\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)\s+(?:say|send|text|write|mention)\b"),
]

# Words that describe *which* messages are wanted rather than what they
# are about. A residual made only of these ("recent messages", "latest",
# "the last thing") has nothing to embed: the vector is noise, and it
# decides the ranking. Such a query means "the newest ones", which is a
# sort, not a search.
_FILLER_WORDS = frozenset(
    """a all an and any anything are as at be by did do does for from get give had has have
    he her him his i in is it its latest me mention mentioned message messages most my new
    newest of on or our recent recently say said says send sent show similar so some stuff
    text texted texts that the these thing things this those to told tell up us was we were
    what whats when where
    which who whose with write wrote you your last""".split()
)


def _is_contentless(residual: str) -> bool:
    # Apostrophes are stripped so "what's" reads as the filler "whats"
    # rather than as a topic word.
    words = [w.replace("'", "") for w in re.findall(r"[a-zA-Z']+", residual.lower())]
    return bool(words) and all(word in _FILLER_WORDS for word in words if word)

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
    people_sender: List[str] = dataclasses.field(default_factory=list)  # who wrote it
    date_from: Optional[float] = None  # unix seconds, inclusive, padded
    date_to: Optional[float] = None
    has_media: bool = False
    # True: only my own messages ("what did I say about X"). False: only the
    # other side's. None: either.
    from_me: Optional[bool] = None
    is_group: Optional[bool] = None
    chat_ids: Optional[List[int]] = None


def _extract_media_filter(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & _MEDIA_KEYWORDS)


# Phrases that name a *span* of time, not a point in it. dateparser
# resolves "last month" to a single instant a month ago, which the padding
# below then turns into a six-day window around some arbitrary day in the
# middle of that month -- so "messages from Vamski about golf last month"
# searched six days and found nothing. These are matched first, and only
# what they don't cover falls through to dateparser.
_RANGE_PATTERNS = [
    (re.compile(r"\btoday\b", re.I), "day", 0),
    (re.compile(r"\byesterday\b", re.I), "day", 1),
    (re.compile(r"\b(?:this|the past|past|last) week\b", re.I), "days", 7),
    (re.compile(r"\b(?:this|the past|past|last) month\b", re.I), "days", 31),
    (re.compile(r"\b(?:this|the past|past|last) year\b", re.I), "days", 366),
    (re.compile(r"\b(?:the past|past|last) (\d+) days?\b", re.I), "days", None),
    (re.compile(r"\b(?:the past|past|last) (\d+) weeks?\b", re.I), "weeks", None),
    (re.compile(r"\b(?:the past|past|last) (\d+) months?\b", re.I), "months", None),
    (re.compile(r"\brecently\b", re.I), "days", 30),
    (re.compile(r"\b(\d+) days? ago\b", re.I), "day", None),
]

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\b(?:\s+(\d{4}))?", re.I
)


def _extract_month_range(text: str):
    """A named month is the whole month.

    Without this "what happened in July" became a six-day window around
    some day in July, which is a stranger answer than either "all of July"
    or "no date filter at all".
    """
    match = _MONTH_PATTERN.search(text)
    if match is None:
        return None
    if not _looks_like_a_real_date_match(match.group(0), text):
        return None  # bare "may"/"march" as ordinary words
    now = dt.datetime.now()
    month = _MONTHS.index(match.group(1).lower()) + 1
    year = int(match.group(2)) if match.group(2) else now.year
    if match.group(2) is None and month > now.month:
        year -= 1  # the most recent past occurrence, not a future one
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + (month == 12), (month % 12) + 1, 1)
    return start.timestamp(), min(end.timestamp(), now.timestamp()), [match.group(0)]


def _extract_range_phrase(text: str):
    """A (date_from, date_to, substrings) span for a relative range phrase."""
    now = dt.datetime.now()
    for pattern, unit, amount in _RANGE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if amount is None:
            amount = int(match.group(1))
        if unit == "day":
            # A named day is that whole calendar day, not a window around
            # its midnight -- "yesterday" used to run three days into the
            # future.
            day = (now - dt.timedelta(days=amount)).replace(hour=0, minute=0, second=0, microsecond=0)
            return day.timestamp(), (day + dt.timedelta(days=1)).timestamp(), [match.group(0)]
        days = {"days": 1, "weeks": 7, "months": 31}[unit] * amount
        start = (now - dt.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        # Ends now, not at some padded point in the future.
        return start.timestamp(), now.timestamp(), [match.group(0)]
    return None


def _extract_date_range(text: str) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Returns (date_from, date_to, matched_substrings), or (None, None, [])
    if nothing plausible was found.
    """
    span = _extract_range_phrase(text) or _extract_month_range(text)
    if span is not None:
        return span

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


def _extract_senders(
    text: str, contact_index: Optional[ContactIndex]
) -> Tuple[List[str], List[str]]:
    """Handles for people named as the *author* of the wanted messages."""
    if contact_index is None:
        return [], []
    handle_ids: List[str] = []
    matched: List[str] = []
    for pattern in _SENDER_PATTERNS:
        for match in pattern.finditer(text):
            if match.group(1).strip().lower() in _NOT_A_NAME:
                continue
            found = contact_index.handle_ids_for_names(match.group(1), threshold=PEOPLE_FUZZY_THRESHOLD)
            if found:
                handle_ids.extend(found)
                matched.append(match.group(0))
    return handle_ids, matched


def _mentions_another_sender(text: str) -> bool:
    return any(
        match.group(1).strip().lower() not in _NOT_A_NAME
        for pattern in _SENDER_PATTERNS
        for match in pattern.finditer(text)
    )


def _extract_self_sender(text: str):
    matched = [m.group(0) for pattern in _SELF_PATTERNS for m in pattern.finditer(text)]
    return (True if matched else None), matched


def parse_query(text: str, contact_index: Optional[ContactIndex] = None) -> ParsedQuery:
    """Parse a free-text query into structured filters plus a semantic
    residual. Extracted substrings are removed from the residual; anything
    not confidently extracted stays in it untouched.
    """
    has_media = _extract_media_filter(text)
    date_from, date_to, date_substrings = _extract_date_range(text)
    people_participant, people_substrings = _extract_participants(text, contact_index)
    people_sender, sender_substrings = _extract_senders(text, contact_index)
    from_me, self_substrings = _extract_self_sender(text)
    # "what did Kaya say when I asked about the lease" names someone else;
    # their name is the stronger signal, so the self filter yields -- even
    # when the name resolves to no contact, since the query is still about
    # them rather than about me.
    if people_sender or _mentions_another_sender(text):
        from_me, self_substrings = None, []
    # A named sender is necessarily a participant, and narrowing candidate
    # chunks to their chats is what makes the sender filter cheap.
    for handle in people_sender:
        if handle not in people_participant:
            people_participant.append(handle)

    residual = text
    for substring in date_substrings + people_substrings + sender_substrings + self_substrings:
        residual = residual.replace(substring, " ")
    residual = re.sub(r"\s+", " ", residual).strip()
    # Removing "last week" from "texts from last week" leaves "texts from",
    # whose dangling preposition is noise to both the embedder and FTS.
    residual = re.sub(r"\b(?:from|with|in|on|at|about|during|since)\s*$", "", residual, flags=re.I).strip()

    if _is_contentless(residual):
        # Nothing left to match on -- browse the newest messages that pass
        # the filters instead of embedding filler.
        residual = ""

    return ParsedQuery(
        raw=text,
        semantic=residual,
        people_sender=people_sender,
        people_participant=people_participant,
        date_from=date_from,
        date_to=date_to,
        has_media=has_media,
        from_me=from_me,
    )

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
#
# The name group is case-INSENSITIVE. It used to require a capital letter,
# on the theory that this was what stopped "from yesterday" or "with the
# trip" being read as a person -- but people type "from kaya" constantly,
# and every such query silently lost its person filter and answered with
# somebody else's messages. Capitalization is now treated as *evidence*
# rather than a requirement: a capitalized candidate is resolved fuzzily as
# before, while a lower-case one must match a contact name exactly
# (`_resolve_name`). Neither path can invent a person the user never named.
_NAME_GROUP = r"([^\W\d_][\w'-]*(?:\s+[^\W\d_][\w'-]*)?)"

# Pasted names arrive quoted -- `messages from "kaya"` -- and the quote
# used to block the name group entirely.
_QUOTE = r"[\"“”'‘’]?"

_PARTICIPANT_PATTERN = re.compile(r"\b(?:from|with)\s+" + _QUOTE + _NAME_GROUP, re.I)

# "kaya's messages" is a sender query, and one of the most natural ways to
# ask. The group here excludes the apostrophe so the possessive suffix is
# not swallowed into the name.
# "Sraddha about work" names a participant with no preposition at all, and
# "what did I tell Kaya" names the *recipient*. Both were parsed as pure
# topic text, so the person was searched for as a word rather than filtered on.
_ADDRESSED_PATTERNS = [
    re.compile(r"\b" + _NAME_GROUP + r"\s+about\b", re.I),
    re.compile(r"\b(?:tell|told|ask|asked|text|texted|message|messaged|send|sent)\s+" + _NAME_GROUP + r"\b", re.I),
]

_POSSESSIVE_PATTERN = re.compile(
    r"\b([^\W\d_][\w-]*)['’]s\b\s*(?:messages?|texts?|replies|reply|stuff|photos?|pictures?|last|latest|recent)",
    re.I,
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
# to the contact index and fuzzy-match a real person. The temporal words are
# here because the name group no longer requires a capital: "from yesterday"
# is a date, not a person, whatever the contact list happens to contain.
_NOT_A_NAME = frozenset(
    """i me you we they who someone anyone everyone nobody
    yesterday today tonight tomorrow now then
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december
    morning afternoon evening night weekend week month year""".split()
)

_SENDER_PATTERNS = [
    re.compile(r"\bfrom\s+" + _QUOTE + _NAME_GROUP, re.I),
    re.compile(_QUOTE + _NAME_GROUP + _QUOTE + r"\s+(?:sent|said|says|texted|wrote|mentioned)\b", re.I),
    re.compile(r"\bdid\s+" + _QUOTE + _NAME_GROUP + _QUOTE + r"\s+(?:say|send|text|write|mention)\b", re.I),
    _POSSESSIVE_PATTERN,
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
    which who whose with write wrote you your last
    everything anymore find search look see pull please pls just about""".split()
)


def _is_contentless(residual: str) -> bool:
    # Apostrophes are stripped so "what's" reads as the filler "whats"
    # rather than as a topic word.
    words = [w.replace("'", "") for w in re.findall(r"[a-zA-Z']+", residual.lower())]
    # A single letter is never a topic. "kaya's latest messages" leaves an
    # orphaned "s" behind, which made the residual look contentful and sent
    # a pure recency query through the embedder instead of browse.
    words = [w for w in words if len(w) > 1]
    return bool(words) and all(word in _FILLER_WORDS for word in words)

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
    # "in March" / "back in May": nobody writes the verb that way, so the
    # preposition is all the corroboration needed. Without it "messages in
    # March" got no date filter at all.
    if re.search(r"\b(?:in|during|of|since|from|before|after)\s+" + re.escape(substring),
                 full_text, re.I):
        return True
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
    # "this morning" is when the message arrived; a bare "tonight" is
    # usually the *subject* ("dinner tonight"), so it stays in the residual.
    (re.compile(r"\bthis (?:morning|afternoon|evening)\b", re.I), "day", 0),
    (re.compile(r"\blast night\b", re.I), "day", 1),
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


_WEEKEND_PATTERN = re.compile(r"\b(?:this|last|past|the)\s+weekend\b", re.I)


def _extract_weekend(text: str):
    """The most recent Saturday-Sunday.

    "this weekend" is a range with a shape no relative-days pattern
    expresses; without it the phrase produced no date filter at all and the
    query was answered by similarity alone.
    """
    match = _WEEKEND_PATTERN.search(text)
    if match is None:
        return None
    now = dt.datetime.now()
    today = dt.datetime(now.year, now.month, now.day)
    # Monday=0 .. Saturday=5. On a Saturday or Sunday, "this weekend" is the
    # one in progress.
    days_since_saturday = (today.weekday() - 5) % 7
    start = today - dt.timedelta(days=days_since_saturday)
    end = min(start + dt.timedelta(days=2), now)
    return start.timestamp(), end.timestamp(), [match.group(0)]


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
    span = _extract_weekend(text) or _extract_range_phrase(text) or _extract_month_range(text)
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


# Ordinary English words are one edit away from real names surprisingly
# often -- "same"/"sage", "mine"/"Mike" -- so the typo-tolerant resolver is
# not allowed to run on them. Generated from the 600 most frequent words in
# this corpus with every contact given name removed, so it cannot suppress
# a real name. It gates *only* the typo path: an exact match on a contact
# called "Sunday" still resolves.
_COMMON_WORDS = frozenset(
    """able about actually after again airport almost already also always
    amazon another anyone anything apartment apple around asked away awesome
    babe back beautiful because been before being believe best better
    bike birthday boat book booked both bought bring brother bruh
    building bunch busy call called came card care case cause
    change chat check chill christmas class close code coffee come
    coming cool could couple crazy cute damn date days deal
    definitely didn different dinner disliked does doesn doing done double
    down drink drive driving dude each earlier early easy either
    else email emphasized enjoy enough even evening ever every everyone
    everything excited family feel feeling find fine first flight food
    forgot found free friday friend friends from fuck full funny
    game getting give glad going gonna good goodnight gotta grab
    great group guess guys haha hahaha hang happy hard have
    haven having head headed heading hear hehe hello help here
    high holy honestly hope hopefully hotel hour hours house huge
    idea image interested into join just keep kinda know last
    late later laughed least leave leaving left less life like
    liked line link literally little lmao long look looking looks
    lots love loved lunch made make makes making many maybe
    mean meet meeting might mine miss monday money month more
    morning most move movie much must myself name need never
    next nice night nothing office okay okie ones only open
    other outside over parents park parking part party pass people
    perfect person phone pick place plan plane planning plans play
    please point pretty prob probably puzzle reacted read ready real
    really remember rest ride right room safe said same saturday
    says seems seen send sent share shit should show sick
    side since skiing sleep small snow some someone something soon
    sorry sounds spend spot start started stay still stop stuff
    such sunday super sure take taking talk talking team tell
    than thank thanks that thats their them then there these
    they thing things think thinking this those though thought through
    thursday time tired today together told tomorrow tonight took trip
    trying tuesday until used very visit wait walk walking wanna
    want wanted wants wasn watch watching water wednesday week weekend
    weeks well went were what when whenever where which while
    whole wine wish with woah work working works worries would
    yall yeah year years yesterday your yours yummy""".split()
)


def _is_capitalised(span: str) -> bool:
    return all(word[:1].isupper() for word in span.split() if word)


def _name_candidates(span: str) -> List[str]:
    """The captured span, then each of its words alone.

    The name group takes up to two words, so "from kaya yesterday" captures
    "kaya yesterday" and "the last thing Kaya sent" captures "thing Kaya".
    When capitals were required neither extra word was ever swallowed; now
    they can be, and the individual words have to be tried too or the person
    is lost.
    """
    span = span.strip().strip("\"“”'‘’")
    words = span.split()
    candidates = [span, *words] if len(words) > 1 else ([span] if span else [])
    # "kaya's" reaches here from patterns that do not strip the suffix.
    return [re.sub(r"['’]s$", "", c) or c for c in candidates]


def resolve_name(span: str, contact_index: Optional[ContactIndex],
                 threshold: float = PEOPLE_FUZZY_THRESHOLD) -> Tuple[List[str], str]:
    """Resolve a name span to handle ids. Returns (handles, resolved_span).

    A capitalized span is fuzzy-matched, which tolerates surnames and
    misspellings. A lower-case span carries no evidence that it is a name at
    all, so it has to match a contact name exactly -- fuzzy matching a
    lower-cased "the trip" partial-matches real contact names well above the
    threshold, which would be a confident, wrong filter.
    """
    if contact_index is None:
        return [], ''
    for candidate in _name_candidates(span):
        words = candidate.lower().split()
        if any(word in _NOT_A_NAME for word in words):
            continue
        if not _is_capitalised(candidate) and all(word in _FILLER_WORDS for word in words):
            continue
        # Ordered by precision, not by convenience: an exact name beats a
        # typo of a different name, which beats a loose fuzzy match. Running
        # fuzzy first resolved the typo "Alyia" to a stranger while a
        # contact one edit away sat in the address book.
        found = contact_index.handle_ids_for_exact_name(candidate)
        if found:
            return found, candidate
        if candidate.lower() not in _COMMON_WORDS:
            found = contact_index.handle_ids_for_similar_given_name(candidate)
            if found:
                return found, candidate
        # Fuzzy needs the evidence of a capital: it is what resolves
        # surnames and partial names, and also what matches "the trip" to a
        # real contact if allowed to run on ordinary words.
        if _is_capitalised(candidate):
            found = contact_index.handle_ids_for_names(candidate, threshold=threshold)
            if found:
                return found, candidate
    return [], ''


def _matched_span(match: re.Match, resolved: str) -> str:
    """The part of `match` up to the end of the name that actually resolved,
    so "from mom about work" removes "from mom" from the residual and leaves
    the topic behind."""
    offset = match.group(1).find(resolved)
    if offset < 0:
        return match.group(0)
    end = match.start(1) + offset + len(resolved)
    return match.string[match.start(0):end]


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
    for pattern in [_PARTICIPANT_PATTERN] + _ADDRESSED_PATTERNS:
        for match in pattern.finditer(text):
            found, resolved = resolve_name(match.group(1), contact_index)
            if found:
                handle_ids.extend(found)
                matched.append(_matched_span(match, resolved))
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
            found, resolved = resolve_name(match.group(1), contact_index)
            if found:
                handle_ids.extend(found)
                matched.append(_matched_span(match, resolved))
    return handle_ids, matched


def _mentions_another_sender(text: str, contact_index: Optional[ContactIndex] = None) -> bool:
    """Does the query name someone *other than me* as the author?

    A capitalized candidate counts even when it resolves to no contact --
    the query is still about them. A lower-case one does not, or "messages i
    sent from work" would read "work" as a person and drop the self filter.
    """
    for pattern in _SENDER_PATTERNS:
        for match in pattern.finditer(text):
            for candidate in _name_candidates(match.group(1)):
                if candidate.lower() in _NOT_A_NAME:
                    continue
                if _is_capitalised(candidate):
                    return True
                if resolve_name(candidate, contact_index)[0]:
                    return True
    return False


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
    if people_sender or _mentions_another_sender(text, contact_index):
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

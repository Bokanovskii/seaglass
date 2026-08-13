"""`eval/suites.py` — QUERY-EVAL-PLAN.md §3: the query set.

Cases are **generated from the live corpus** rather than hardcoded, for the
reason §2 gives: the oracle is computable, so the suite can be as large as
the corpus supports and stays valid as the corpus grows. Hardcoding names
would also make the suite useless on anyone else's machine.

Templates are grouped by the class that decides their correctness, not by
their wording, so a failure names a bug class rather than a sentence.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from seaglass.eval.behavior import Case
from seaglass.imessage.source import APPLE_EPOCH_UNIX

# Wordings that must all resolve to the same intent. Every one of these has
# been observed from a real caller (a person typing, or Grogu relaying a
# model's phrasing), and they differ in exactly the ways the parser has got
# wrong before: filler-only residuals, past-tense sender phrasing, and
# question words that look like topics.
PERSON_RECENCY = [
    "latest from {name}",
    "recent messages from {name}",
    "what did {name} say recently",
    "what's the last thing {name} sent me",
    "most recent texts from {name}",
    "{name}'s latest messages",
]

PERSON_ONLY = [
    "messages from {name}",
    "texts from {name}",
    "show me everything from {name}",
]

PERSON_DATE = [
    "messages from {name} last week",
    "what did {name} say yesterday",
    "texts from {name} last month",
]

PERSON_TOPICAL = [
    "what did {name} say about dinner",
    "{name} about work",
    "conversation with {name} about plans",
]

DATE_ONLY = [
    "messages from last week",
    "what happened yesterday",
    "texts from last month",
    "messages in March",
]

TOPICAL = [
    "dinner plans",
    "what did we decide about the trip",
    "restaurant reservation",
    "flight booking details",
    "apartment lease",
    "birthday party planning",
]

TOPICAL_DATE = [
    "dinner plans last month",
    "work stuff last week",
]

MEDIA = [
    "photos from {name}",
    "pictures {name} sent",
]

SELF = [
    "what did I say about dinner",
    "messages I sent last week",
]

# Whole-word phrases lifted from the corpus, so the right answer provably
# exists. Semantic search can quietly stop returning the *literal* string a
# user pasted in; nothing else in the suite would notice.
LEXICAL_TEMPLATE = '{phrase}'

GROUP = [
    "group chats about dinner",
    "what did the group say about the trip",
]

SELF_TOPICAL = [
    "what did I say about dinner",
    "what did I tell {name}",
    "my messages about work",
]

PERSON_MEDIA = [
    "photos {name} sent last month",
]

TIME_OF_DAY = [
    "messages from this morning",
    "what came in this weekend",
    "texts from today",
]

NAME_TYPO = [
    "what did {name_typo} say",
]

AMBIGUOUS = ["golf", "ok", "thanks", "the thing we talked about"]

NATURAL = [
    "did he ever reply about the boat?",
    "what was the address again",
    "who was supposed to bring the food",
]

# Parser guards. "may"/"march" are month names *and* people's names; a long
# query must not blow up the residual; emoji and typos must not crash.
ADVERSARIAL = [
    "what did May say",
    "march madness bracket",
    "dinner plns tomorow",
    "🎉",
    "what did we end up deciding about the whole situation with the car and "
    "the insurance and whether it was worth it to keep paying for the coverage",
    "",
]


def _typo(name: str) -> str:
    """One transposed pair -- the kind of miss a person actually types."""
    if len(name) < 4:
        return name
    i = len(name) // 2
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def _names_for(engine, rows, limit: int) -> List[str]:
    names: List[str] = []
    for handle, _count in rows:
        name = engine.contact_index.resolve_handle(handle)
        if not name or name == handle:
            continue
        first = name.split()[0]
        if engine.contact_index.handle_ids_for_names(first) and first not in names:
            names.append(first)
        if len(names) >= limit:
            break
    return names


def _frequent_senders(engine, limit: int = 6) -> List[str]:
    """Contacts with enough recent traffic to make a meaningful oracle.

    Picked from chat.db rather than from the contact list: someone in the
    address book with no messages produces a case whose correct answer is
    empty, which tests nothing.
    """
    if engine.chat_con is None or engine.contact_index is None:
        return []
    rows = engine.chat_con.execute(
        """
        SELECT h.id, COUNT(*) AS n
        FROM message m JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.is_from_me = 0 AND m.associated_message_type = 0
        GROUP BY h.id ORDER BY n DESC LIMIT 40
        """
    ).fetchall()
    names: List[str] = []
    for handle, _count in rows:
        name = engine.contact_index.resolve_handle(handle)
        if not name or name == handle:
            continue
        first = name.split()[0]
        # Only keep names that resolve back to a handle set -- a name the
        # parser cannot resolve makes the case a contact-resolution test,
        # which belongs in its own class.
        if engine.contact_index.handle_ids_for_names(first) and first not in names:
            names.append(first)
        if len(names) >= limit:
            break
    return names


def _recent_senders(engine, days: int = 45, limit: int = 4) -> List[str]:
    """Names with traffic *inside* the windows the date classes ask about.

    All-time-frequent senders are the wrong sample for "what did X say
    yesterday": the top sender on this machine last wrote in 2021, so nine
    date cases asserted an answer that correctly did not exist.
    """
    if engine.chat_con is None or engine.contact_index is None:
        return []
    cutoff = time.time() - days * 86400
    rows = engine.chat_con.execute(
        """
        SELECT h.id, COUNT(*) AS n
        FROM message m JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.is_from_me = 0 AND m.associated_message_type = 0
          AND (m.date / 1000000000.0) + ? >= ?
        GROUP BY h.id ORDER BY n DESC LIMIT 40
        """,
        (APPLE_EPOCH_UNIX, cutoff),
    ).fetchall()
    return _names_for(engine, rows, limit)


_TAPBACK = re.compile(r"^(Laughed at|Liked|Loved|Disliked|Emphasized|Questioned)\b")


def _corpus_phrases(engine, limit: int = 6) -> List[str]:
    """Phrases that provably exist in the *index*, not just in chat.db.

    Two traps, both of which made this measure the sampler instead of the
    engine: dropping non-alphabetic tokens produced phrases that were never
    typed contiguously ("ya so good to learn about" spliced across a
    number), and tapbacks quote someone else's words. So the phrase is a
    verbatim token run, and it is only kept once FTS confirms the index
    contains it.
    """
    if engine.chat_con is None:
        return []
    rows = engine.chat_con.execute(
        """
        SELECT text FROM message
        WHERE is_from_me = 0 AND text IS NOT NULL AND LENGTH(text) BETWEEN 40 AND 160
        ORDER BY date DESC LIMIT 600
        """
    ).fetchall()
    phrases: List[str] = []
    for (text,) in rows:
        text = (text or "").strip()
        if _TAPBACK.match(text):
            continue
        tokens = text.split()
        if len(tokens) < 8:
            continue
        phrase = " ".join(tokens[1:7]).strip(" .,!?;:\"'")
        # A phrase of six stopwords has no defensible right answer.
        if len([w for w in phrase.split() if len(w) > 4]) < 2:
            continue
        if not _in_index(engine, phrase):
            continue
        if phrase.lower() in {p.lower() for p in phrases}:
            continue
        phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def _in_index(engine, phrase: str) -> bool:
    try:
        row = engine.index_con.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
            ['"' + phrase.replace('"', '""') + '"'],
        ).fetchone()
    except Exception:
        return False
    return row is not None


def build_suite(engine, path: Optional[Path] = None) -> List[Case]:
    if path is not None:
        raw = json.loads(path.read_text())
        return [Case(**entry) for entry in raw]

    names = _frequent_senders(engine)
    recent = _recent_senders(engine) or names
    cases: List[Case] = []

    def add(templates, cls, pool=None, **kwargs):
        pool = pool if pool is not None else names
        for template in templates:
            if "{name}" in template:
                for name in pool:
                    cases.append(Case(template.format(name=name), cls, **kwargs))
            elif "{name_typo}" in template:
                for name in pool:
                    cases.append(Case(template.format(name_typo=_typo(name)), cls, **kwargs))
            else:
                cases.append(Case(template, cls, **kwargs))

    add(PERSON_RECENCY, "person_recency", pool=recent)
    add(PERSON_ONLY, "person_only")
    add(PERSON_DATE, "person_date", pool=recent)
    add(PERSON_TOPICAL, "person_topical", expects_results=False)
    add(DATE_ONLY, "date_only")
    add(TOPICAL, "topical", expects_results=False)
    add(TOPICAL_DATE, "topical_date", expects_results=False)
    add(MEDIA, "media", expects_results=False)
    add(SELF, "self", expects_results=False)
    add(AMBIGUOUS, "ambiguous", expects_results=False)
    add(NATURAL, "natural", expects_results=False)
    add(ADVERSARIAL, "adversarial", expects_results=False)
    add(GROUP, "group", expects_results=False)
    add(SELF_TOPICAL, "self_topical", pool=recent, expects_results=False)
    add(PERSON_MEDIA, "person_media", pool=recent, expects_results=False)
    add(TIME_OF_DAY, "time_of_day", expects_results=False)
    add(NAME_TYPO, "name_typo", pool=recent, expects_results=False)
    for phrase in _corpus_phrases(engine):
        cases.append(Case(phrase, "lexical", lexical=phrase, expects_results=False))
    return cases

"""`eval/parser_probe.py` — TEST-EVAL-PLAN-V2.md §5.

The query parser is where `recent messages from kaya` was lost, and it is
the cheapest thing in the system to evaluate: no models, no index, no
network. So it gets measured at a scale the search suite cannot afford --
every contact in the address book, in every surface form a person might
type, against every template, plus a negative set drawn from the corpus.

Two numbers matter, and they pull in opposite directions:

* **recall** — a name the user typed produced a person filter. Missing it
  fails open into a plain semantic search, which answers confidently with
  somebody else's messages.
* **precision** — a word the user typed that is *not* a name produced no
  person filter. This is the risk created by no longer requiring a capital
  letter, and it is the worse failure of the two: a wrong filter answers
  the wrong question, while a missing one merely answers a vaguer version
  of the right one.

Negatives are taken from real message text rather than invented, because
the words that actually follow "from" in this corpus ("from work", "from
the airport") are the ones a user will actually type.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from seaglass.imessage.contacts import ContactIndex
from seaglass.imessage.source import connect_readonly
from seaglass.search.parse import parse_query

# --------------------------------------------------------------------------
# Surface forms (§4)
# --------------------------------------------------------------------------

SURFACE_FORMS = {
    "plain": lambda name: name,
    "lower": lambda name: name.lower(),
    "upper": lambda name: name.upper(),
    "possessive": lambda name: name.lower() + "'s",
    "punctuated": lambda name: name.lower() + "?",
    "quoted": lambda name: '"' + name.lower() + '"',
    "typo": lambda name: _typo(name),
}

# Possessive and punctuated forms only make sense in some templates, so the
# templates say where the name goes and which forms suit them.
POSITIVE_TEMPLATES = [
    ("messages from {name}", {"plain", "lower", "upper", "punctuated", "quoted", "typo"}),
    ("what did {name} say", {"plain", "lower", "upper", "typo"}),
    ("latest from {name}", {"plain", "lower", "upper"}),
    ("recent messages from {name}", {"plain", "lower", "upper"}),
    ("texts from {name} last week", {"plain", "lower", "upper"}),
    ("photos from {name}", {"plain", "lower"}),
    ("{name} messages", {"possessive"}),
    ("what did {name} say about dinner", {"plain", "lower"}),
]

# Forms where a *fuzzy* match is the only route, so a miss is a quality
# shortfall rather than a correctness bug (§8).
BEST_EFFORT_FORMS = {"typo"}


def _typo(name: str) -> str:
    """One transposed pair -- the kind of miss a person actually types."""
    if len(name) < 4:
        return name
    i = len(name) // 2
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


# --------------------------------------------------------------------------
# Negatives (§5)
# --------------------------------------------------------------------------

NEGATIVE_TEMPLATES = [
    "messages from {word}",
    "photos from {word}",
    "what did we say about {word}",
    "texts with {word}",
    "notes from {word}",
]

# Non-name things that follow from/with often enough to be typed at a search
# box. Kept even when the corpus sampler finds them too -- they are the
# regressions this guards.
STATIC_NEGATIVES = [
    "yesterday",
    "today",
    "last week",
    "last month",
    "monday",
    "june",
    "march",
    "work",
    "school",
    "the airport",
    "the store",
    "home",
    "the doctor",
    "the group",
    "my phone",
    "the bank",
    "the office",
    "vacation",
    "dinner",
    "the wedding",
    "the meeting",
    "the hospital",
    "the party",
    "my sister",
    "my mom",
    "everyone",
    "someone",
    "them",
    "here",
    "there",
    "that",
]

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]*")


def _is_a_contact_name(phrase: str, contact_index: ContactIndex) -> bool:
    """Does the address book literally contain someone called this?

    Computed here rather than through the matcher under test, so the probe
    cannot excuse an over-match by asking the over-matcher.
    """
    folded = _fold(phrase)
    for contact in contact_index._contacts:  # noqa: SLF001 - eval-only
        name = _fold(contact.display_name)
        if not name:
            continue
        if folded == name or folded == name.split()[0]:
            return True
    return False


def _fold(raw: str) -> str:
    return " ".join(re.sub(r"[^\w\s]+", " ", (raw or "").lower()).split())


def corpus_negatives(chat_con, contact_index: ContactIndex, limit: int = 60) -> List[str]:
    """The words that most often follow "from"/"with" in real messages.

    Anything that *is* a contact name is dropped: "from Kaya" appears in
    message text too, and it is not a negative.
    """
    rows = chat_con.execute(
        """
        SELECT text FROM message
        WHERE text IS NOT NULL AND LENGTH(text) BETWEEN 20 AND 300
        ORDER BY date DESC LIMIT 20000
        """
    ).fetchall()
    counts: Counter = Counter()
    pattern = re.compile(r"\b(?:from|with)\s+((?:the\s+)?[a-z][a-z'-]+)", re.I)
    for (text,) in rows:
        for match in pattern.finditer(text or ""):
            phrase = " ".join(match.group(1).lower().split())
            if len(phrase) < 3:
                continue
            counts[phrase] += 1
    negatives: List[str] = []
    for phrase, _count in counts.most_common(limit * 4):
        # A phrase that really does name a contact is not a negative.
        if _is_a_contact_name(phrase.replace("the ", ""), contact_index):
            continue
        negatives.append(phrase)
        if len(negatives) >= limit:
            break
    return negatives


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------


@dataclass
class ProbeCase:
    query: str
    kind: str  # 'positive' | 'negative'
    form: str
    subject: str
    expect_handles: Sequence[str] = ()
    # The name exactly as it was typed into the query. For the typo form
    # that is *not* `subject`, and the contacts-unavailable check needs the
    # string the user actually typed, not the one it stands for.
    typed_name: str = ""


@dataclass
class ProbeReport:
    by_form: Dict[str, Dict[str, int]] = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)

    def record(self, case: ProbeCase, passed: bool, detail: str = "") -> None:
        bucket = self.by_form.setdefault(
            f"{case.kind}:{case.form}", {"pass": 0, "fail": 0}
        )
        bucket["pass" if passed else "fail"] += 1
        if not passed:
            self.failures.append(
                {"query": case.query, "kind": case.kind, "form": case.form,
                 "subject": case.subject, "detail": detail}
            )


def contact_names(contact_index: ContactIndex, chat_con=None, limit: int = 40) -> List[str]:
    """First names of contacts, preferring ones that actually message.

    A name unique enough to resolve is what is being tested; a contact whose
    first name is shared with three others resolves to a union of handles,
    which is correct but makes the assertion ambiguous.
    """
    ranked: List[str] = []
    if chat_con is not None:
        rows = chat_con.execute(
            """
            SELECT h.id, COUNT(*) n FROM message m JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.is_from_me = 0 GROUP BY h.id ORDER BY n DESC LIMIT 400
            """
        ).fetchall()
        for handle, _n in rows:
            name = contact_index.resolve_handle(handle)
            if name and name != handle:
                first = name.split()[0]
                if first not in ranked and len(first) > 2:
                    ranked.append(first)
            if len(ranked) >= limit:
                break
    if not ranked:
        for contact in contact_index._contacts:  # noqa: SLF001 - eval-only
            first = contact.display_name.split()[0]
            if len(first) > 2 and first not in ranked:
                ranked.append(first)
            if len(ranked) >= limit:
                break
    # Only names the parser has any chance with: a first name that does not
    # resolve even in its address-book spelling is a contacts problem, not a
    # parser one, and would report as a parser failure on every form.
    return [n for n in ranked if contact_index.handle_ids_for_names(n, threshold=85.0)]


def build_cases(contact_index: ContactIndex, chat_con=None, n_names: int = 25) -> List[ProbeCase]:
    cases: List[ProbeCase] = []
    for name in contact_names(contact_index, chat_con, limit=n_names):
        expect = contact_index.handle_ids_for_names(name, threshold=85.0)
        for template, forms in POSITIVE_TEMPLATES:
            for form in forms:
                rendered = SURFACE_FORMS[form](name)
                cases.append(
                    ProbeCase(
                        template.format(name=rendered), "positive", form, name, expect,
                        typed_name=rendered,
                    )
                )
    # "home" is a static negative *and* the name of a contact in this
    # address book. Filtering the negative set by the address book is not
    # excusing a bug: a word the user has literally saved as somebody's name
    # has a legitimate person reading.
    negatives = [w for w in STATIC_NEGATIVES if not _is_a_contact_name(w, contact_index)]
    if chat_con is not None:
        for phrase in corpus_negatives(chat_con, contact_index):
            if phrase not in negatives:
                negatives.append(phrase)
    for word in negatives:
        for template in NEGATIVE_TEMPLATES:
            cases.append(ProbeCase(template.format(word=word), "negative", "corpus", word))
    return cases


def run(cases: Iterable[ProbeCase], contact_index: ContactIndex) -> ProbeReport:
    report = ProbeReport()
    for case in cases:
        parsed = parse_query(case.query, contact_index=contact_index)
        found = set(parsed.people_participant) | set(parsed.people_sender)
        if case.kind == "positive":
            wanted = set(case.expect_handles)
            passed = bool(found & wanted)
            report.record(case, passed, f"resolved={sorted(found)[:3]}")
        else:
            passed = not found
            names = sorted({contact_index.resolve_handle(h) or h for h in found})
            report.record(case, passed, f"filtered to {names[:3]}")
    return report


def run_without_contacts(cases: Iterable[ProbeCase], contact_index: ContactIndex) -> ProbeReport:
    """TEST-EVAL-PLAN-V2.md §7: the same queries with no address book.

    Contacts are a *permission*, not a given -- a fresh machine, or one
    where the user said no, has none. Three things must hold:

    * the parser does not raise;
    * it invents no person filter it cannot possibly have resolved;
    * nothing is silently discarded. Everything the parse *with* contacts
      kept as search text is still there, and so is the name it can no
      longer resolve -- an unresolvable name is still a perfectly good
      thing to search for as text, and dropping the span turns "photos
      from Kaya" into "photos".

    Date and media words legitimately leave the semantic text in both
    parses, which is why the comparison is against the with-contacts
    parse rather than against the raw query.
    """
    report = ProbeReport()
    for case in cases:
        try:
            blind = parse_query(case.query, contact_index=None)
        except Exception as error:  # noqa: BLE001 - a raise is the failure
            report.record(case, False, f"raised {error!r}")
            continue
        found = set(blind.people_participant) | set(blind.people_sender)
        if found:
            report.record(case, False, f"invented handles {sorted(found)[:3]}")
            continue

        sighted = parse_query(case.query, contact_index=contact_index)
        blind_text = blind.semantic.lower()
        lost = [
            word for word in sighted.semantic.lower().split()
            if word not in blind_text
        ]
        if lost:
            report.record(case, False, f"lost {lost[:3]} from {blind.semantic!r}")
            continue
        # The name it could not resolve has to survive as search text.
        typed = (case.typed_name or case.subject).strip('"\'.,?!')
        subject = typed.split()[0].lower() if case.kind == "positive" else ""
        if subject and subject not in blind_text:
            report.record(case, False, f"dropped the name from {blind.semantic!r}")
            continue
        report.record(case, True, "")
    return report


def print_report(report: ProbeReport, show_failures: int = 25) -> None:
    print(f"{'form':<22} {'pass':>6} {'fail':>6} {'rate':>7}")
    print("-" * 44)
    for form in sorted(report.by_form):
        row = report.by_form[form]
        total = row["pass"] + row["fail"]
        rate = row["pass"] / total if total else 0.0
        print(f"{form:<22} {row['pass']:>6} {row['fail']:>6} {rate:>7.2f}")
    if report.failures:
        print(f"\n{len(report.failures)} failures; first {show_failures}:")
        for failure in report.failures[:show_failures]:
            print(f"  [{failure['kind']}/{failure['form']}] {failure['query']!r} -> {failure['detail']}")


def gate(report: ProbeReport) -> List[str]:
    """TEST-EVAL-PLAN-V2.md §8. Returns the budget violations, if any."""
    violations: List[str] = []
    for form, row in sorted(report.by_form.items()):
        total = row["pass"] + row["fail"]
        if not total:
            continue
        rate = row["pass"] / total
        kind, _, name = form.partition(":")
        floor = 0.5 if name in BEST_EFFORT_FORMS else 1.0
        if rate < floor:
            violations.append(f"{form} {rate:.2f} < {floor:.2f}")
    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-db", default=None, help="chat.db to draw negatives and names from")
    parser.add_argument("--names", type=int, default=25)
    parser.add_argument("--json", default=None, help="write the full report here")
    parser.add_argument("--show", type=int, default=25)
    parser.add_argument(
        "--no-contacts", action="store_true",
        help="run the same cases with no address book (plan §7)",
    )
    args = parser.parse_args(argv)

    contact_index = ContactIndex.load()
    chat_con = None
    if args.chat_db:
        chat_con = connect_readonly(Path(args.chat_db))
    cases = build_cases(contact_index, chat_con, n_names=args.names)
    print(f"{len(cases)} probe cases")
    if args.no_contacts:
        print("circumstance: contacts unavailable")
        report = run_without_contacts(cases, contact_index)
    else:
        report = run(cases, contact_index)
    print_report(report, args.show)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"by_form": report.by_form, "failures": report.failures}, indent=2))
    violations = gate(report)
    if violations:
        print("\nBUDGET VIOLATIONS:")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print("\nall budgets met")
    return 0


if __name__ == "__main__":
    sys.exit(main())

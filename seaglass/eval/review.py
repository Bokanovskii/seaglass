"""`eval/review.py` — EVALUATION.md §4.3: a minimal CLI for human review
of `eval/generate.py`'s draft entries. Shows the question, the source
chunk text, and retrieval diagnostics (trivially-hard flag, sparse
rank-1-is-source flag, top-5 other candidates), with four actions:

  a  accept        -- keep as-is
  e  edit          -- rewrite the query text, then accept
  r  reject        -- discard
  m  multi-positive -- accept, and also prompt for alt_positive_msg_ids
                       (another chunk that legitimately answers it too)
  s  skip          -- leave for a later session, don't decide now

Reviewed entries are appended to `golden.jsonl` with internal `_*`
diagnostic fields stripped (EVALUATION.md §4.4's golden format has no
underscore-prefixed fields) and `reviewed: true` set. Rejected entries
are recorded in a `.rejected.jsonl` sidecar for auditability, not just
silently dropped.

Not a general-purpose review tool -- scoped to this one golden-set
format and workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _append_jsonl(path: Path, entry: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _strip_internal_fields(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _print_entry(entry: dict, index: int, total: int) -> None:
    print(f"\n[{index}/{total}] {entry['id']}  category={entry['category']}  nn_distance={entry.get('nn_distance'):.3f}")
    print(f"  query: {entry['query']}")
    flags = []
    if entry.get("_trivially_hard"):
        flags.append("TRIVIALLY-HARD (source outside top 200 -- may be a real retrieval failure, or a nonsense question)")
    if entry.get("_sparse_rank1_is_source"):
        flags.append("sparse rank-1 IS source (legitimate keyword question, or generation copied source phrasing?)")
    if entry.get("_vocab_overlap_pct") is not None:
        flags.append(f"vocab overlap: {entry['_vocab_overlap_pct']:.0%}")
    if flags:
        print("  flags: " + "; ".join(flags))
    print(f"  positive_msg_ids: {entry.get('positive_msg_ids')}")


def review(
    input_path: Path,
    golden_path: Path,
    rejected_path: Path,
    *,
    already_decided_ids: Set[str],
    interactive_input=input,
) -> None:
    entries = _load_jsonl(input_path)
    pending = [e for e in entries if e["id"] not in already_decided_ids]
    if not pending:
        print("[review] nothing new to review -- every entry already has a decision recorded.")
        return

    for i, entry in enumerate(pending, start=1):
        _print_entry(entry, i, len(pending))
        action = interactive_input("  [a]ccept / [e]dit / [r]eject / [m]ulti-positive / [s]kip: ").strip().lower()

        if action == "s" or action == "":
            continue
        if action == "r":
            _append_jsonl(rejected_path, {**_strip_internal_fields(entry), "reviewed": True})
            continue
        if action == "e":
            new_query = interactive_input(f"  new query text [{entry['query']}]: ").strip()
            if new_query:
                entry["query"] = new_query
            entry["reviewed"] = True
            _append_jsonl(golden_path, _strip_internal_fields(entry))
            continue
        if action == "m":
            alt_raw = interactive_input("  alt_positive_msg_ids (comma-separated message ids): ").strip()
            alt_ids = [int(x) for x in alt_raw.split(",") if x.strip().isdigit()]
            entry["alt_positive_msg_ids"] = alt_ids
            entry["reviewed"] = True
            _append_jsonl(golden_path, _strip_internal_fields(entry))
            continue
        if action == "a":
            entry["reviewed"] = True
            _append_jsonl(golden_path, _strip_internal_fields(entry))
            continue
        print("  (unrecognised input, treating as skip)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="candidates_for_review.jsonl, from eval/generate.py")
    parser.add_argument("--golden", default="golden.jsonl", help="output golden set path (appended to)")
    parser.add_argument("--rejected", default="golden.rejected.jsonl", help="rejected entries sidecar (appended to)")
    args = parser.parse_args(argv)

    golden_path = Path(args.golden)
    rejected_path = Path(args.rejected)
    already_decided = {e["id"] for e in _load_jsonl(golden_path)} | {e["id"] for e in _load_jsonl(rejected_path)}

    review(Path(args.input), golden_path, rejected_path, already_decided_ids=already_decided)
    return 0


if __name__ == "__main__":
    sys.exit(main())

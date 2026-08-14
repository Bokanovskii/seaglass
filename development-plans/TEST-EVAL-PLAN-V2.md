# Test & evaluation plan v2 — closing the holes that shipped bugs

`QUERY-EVAL-PLAN.md` (v1) moved evaluation from "is the answer somewhere in
the payload" (recall@final) to "is the payload *right*" (properties + a
chat.db oracle). It found and fixed six real defects. Then a seventh shipped
anyway:

> `recent messages from kaya` returned Biljana Makivic's messages from six
> weeks earlier.

The suite ran 136 cases across 18 classes and scored **1.00 on every
property** while that query was broken. This plan is about why, and what a
suite has to do differently so the next one cannot hide.

---

## 1. Why v1 could not see it

Three independent holes, each of which alone was enough.

### 1.1 The suite only ever typed names the way the parser likes

Every person case is generated from `{name}`, filled with a contact's first
name in its address-book capitalization — `Kaya`, `Vamski`, `Sraddha`. The
parser required a leading capital to consider a word a name at all
(`[A-Z][\w'-]*`, guarded by a comment explaining why the capital was
load-bearing). So the suite and the parser shared an assumption, and the
suite could only confirm it.

Real callers do not share it. People type `from kaya`. Grogu relays whatever
a model wrote, which is often lower case. **A generated suite tests the
generator's assumptions unless the surface form is itself a dimension.**

### 1.2 Properties were judged against the payload's own parse

`check_properties` reads `effective_filters` out of the payload and judges
the payload against it. That is circular: when the parse extracts nothing,
every filter property is `None` ("does not apply") and the case passes
vacuously.

For `recent messages from kaya` the parse found no person, so
`sender_purity`, `no_self_in_sender`, `date_containment` and the oracle
scoring all declined to apply — and the case was scored a clean pass while
returning a stranger's messages. **The expectation has to come from how the
case was constructed, not from what the system under test believed.**

### 1.3 A failure that fails *open* looks like a different, legitimate answer

A dropped person filter does not error or return empty. It degrades to a
plain semantic search, which returns confident, plausible, well-formed
results about the right topic from the wrong person. Nothing short of "was
this actually from Kaya" catches it.

---

## 2. What changes

| v1 | v2 |
| --- | --- |
| `{name}` in address-book case | `{name}` × **surface forms**: lower, UPPER, first-name-only, possessive, typo, punctuated, quoted |
| Properties judged vs. payload's parse | Properties judged vs. the case's **declared expectation** (`expect_*`), payload parse used only where no expectation exists |
| Oracle keyed on `parsed.people_sender` | Oracle keyed on `case.expect_handles` when declared — a parse miss now collapses recall instead of skipping the case |
| Parser tested by ~40 unit cases | **Parser probe**: thousands of generated positives and corpus-derived negatives, reporting precision *and* recall per surface form |
| Assist untested end to end | Assist gating, merge, and apply covered; the "banner promised a filter that never ran" bug has a regression test |
| Circumstances described in §5, partly exercised | Contacts-unavailable, stale-index and empty-index paths asserted |

---

## 3. The expectation model

A case declares what it was built to test:

```python
Case("recent messages from kaya", "person_recency_lower",
     expect_handles=[...], expect_person="Kaya", expect_recent=True)
```

and the harness asserts:

* `person_filter_applied` — the payload's effective filters name that person.
  This is the property that fails for the kaya bug, loudly, on the first case.
* `sender_is_expected` — every non-self hit was written by them.
* `date_filter_applied` / `self_filter_applied` / `media_filter_applied` —
  same idea for the other filter kinds.

Declared expectations are only ever asserted, never fed into the search: the
query text is what goes in. A case with no declared expectation keeps v1's
behaviour, so topical and adversarial classes are unaffected.

### Why this is not "teaching to the test"

The expectation is a property of the *question* ("I asked for Kaya"), not of
the implementation. It is exactly what the user means and what any
implementation must satisfy. v1's version — "judge it by whatever it thought
the question was" — is the one that cannot fail.

---

## 4. Surface forms

Generated for every person class, from the same contact pool:

| form | example | why |
| --- | --- | --- |
| `plain` | `messages from Kaya` | v1's only form; the baseline |
| `lower` | `messages from kaya` | **the shipped bug**; how people actually type |
| `upper` | `messages from KAYA` | shouting, autocorrect |
| `full` | `messages from Kaya Doe` | disambiguation |
| `possessive` | `kaya's messages` | very common phrasing |
| `punctuated` | `messages from kaya?` | trailing punctuation must not join the name |
| `typo` | `messages from kyaa` | fuzzy path must still fire |
| `quoted` | `messages from "kaya"` | pasted names |

A form that *should* fail (a name that is not in contacts) is a negative
case, and is expected to return results without a person filter rather than
to error or to invent one.

## 5. Parser probe (`seaglass/eval/parser_probe.py`)

Models are not involved in parsing, so this can be run at a scale the search
suite cannot: every contact × every surface form × every template, plus a
large negative set, in seconds.

**Positives** — contact names in every surface form and template, expecting a
resolved person filter. Reported as recall per form; a form at <1.00 names
the exact gap.

**Negatives** — the risk created by relaxing the capitalization rule is
false person filters. Negatives are drawn from the corpus itself rather than
imagined: the most frequent words that follow `from`/`with` in real message
text (`from work`, `from the airport`, `with the kids`), plus temporal words,
plus filler. Expected: no person filter. Reported as precision; any failure
is listed by name so it can be judged, not just counted.

This is the harness that would have caught the bug in the first place, and
it is the one that guards the fix from regressing into over-matching.

## 6. Assist

Assist had three defects that no test could see, because nothing exercised
it end to end:

1. `apply-assist` re-ran `engine.search(merged.raw, …)`. The engine re-parses
   the text it is given, so the entire merged parse was discarded and only
   keyword expansions ever applied — "Force" genuinely did nothing visible.
2. `GET /api/assist/{token}` merged Copilot's parse into `parse_query('')`,
   a parse of the empty string, so `changes` described a diff against nothing.
3. Copilot copies name spans verbatim from the query, so it returned `kaya`,
   and `handle_ids_for_names('kaya')` resolved to nobody — the banner named a
   person filter that could not exist.

Covered now by: gating tests (`should_assist` for force/auto/off across
strong, weak and short parses), merge tests (including unresolved names), and
endpoint tests that assert the *filters the engine was actually called with*.

Auto's rule is stated positively: **assist when the deterministic parse looks
inadequate for the query it was given** — a name-shaped word survived into
the residual with no person filter to show for it, or a long query yielded no
structure at all. Not "assist when no filters were found", which read a
resolved date as proof that the unresolved person did not matter.

## 7. Circumstances

| circumstance | expected behaviour | how asserted | status |
| --- | --- | --- | --- |
| Contacts unavailable | no person filters, no crash, and **nothing silently dropped** — an unresolvable name is still good search text | `parser_probe --no-contacts`: all 1000 cases, compared against the with-contacts parse so date/media words aren't counted as losses | 1.00 |
| Index stale | `index_stale` / freshness fields present and true; `newest_is_indexed` separates staleness from ranking | `freshness_declared` (208 cases), `newest_is_indexed` | 1.00 |
| Index empty | empty payload with the same shape (`ordering`, freshness), **not an exception** | `TestEmptyIndexPayloadShape`, topical and filter-only | found a real crash; fixed |
| App not running | Grogu lazy path loads its own engine | `test_mcp_server.py` + a live cold run with the app killed (1.32 s) | pass |
| App running | Grogu reuses it; no second copy of the models | live: "answered from the running Seaglass app", 0.5–1.8 s | pass |

A note on why the contacts-unavailable check compares two parses rather than
the raw query: date and media words legitimately leave the semantic text in
*both* parses. Checking the blind parse against the raw query flagged "texts
from Sraddha last week" for "dropping" `last week`, which the date filter had
correctly consumed. The invariant that actually means something is that the
blind parse keeps everything the sighted one kept, **plus** the name it can no
longer resolve.

## 8. Budgets and exit criteria

Unchanged from v1 §6 for latency (measured through the running app, never a
second model copy). Added:

* `person_filter_applied` = **1.00** on every declared person case, every
  surface form. This is a correctness invariant, not a quality metric.
* `sender_is_expected` = 1.00.
* Parser probe: positive recall = 1.00 for `plain`, `lower`, `upper`,
  `possessive`, `punctuated`, `quoted`; ≥0.5 for `typo` (fuzzy, best effort).
  Negative precision = 1.00 — a false person filter is worse than a missed
  one, because it answers confidently from the wrong person.

## 9. Results

All numbers measured through the **running desktop app** against the real
index (386,278 messages, 17,039 chunks), so they are the code path a user's
query actually takes and no second copy of the models was ever loaded.

### 9.1 Behaviour suite — 212 cases

Every property and every oracle metric at 1.00, zero failing queries.

| property | checked | rate |
| --- | --- | --- |
| `date_containment` | 30 | 1.00 |
| `date_filter_applied` | 19 | 1.00 |
| `freshness_declared` | 208 | 1.00 |
| `hits_before_context` | 206 | 1.00 |
| `lexical_presence` | 6 | 1.00 |
| `media_filter_applied` | 12 | 1.00 |
| `no_empty_sessions` | 206 | 1.00 |
| `no_self_in_sender` | 150 | 1.00 |
| `nonempty` | 58 | 1.00 |
| `ordering_declared` | 208 | 1.00 |
| `person_filter_applied` | 168 | 1.00 |
| `recency_order` | 149 | 1.00 |
| `recency_session_order` | 147 | 1.00 |
| `self_filter_applied` | 8 | 1.00 |
| `sender_is_expected` | 150 | 1.00 |
| `sender_purity` | 150 | 1.00 |

| oracle metric | mean |
| --- | --- |
| `indexed_coverage` | 1.00 |
| `newest_is_indexed` | 1.00 |
| `newest_is_true_newest` | 1.00 |
| `newest_present` | 1.00 |
| `precision` | 1.00 |
| `recall_top` | 1.00 |

| class | n | pass | p50 s | p95 s |
| --- | --- | --- | --- | --- |
| `adversarial` | 6 | 1.00 | 1.036 | 1.381 |
| `ambiguous` | 4 | 1.00 | 0.648 | 1.296 |
| `date_only` | 4 | 1.00 | 0.044 | 0.419 |
| `group` | 2 | 1.00 | 1.264 | 1.273 |
| `lexical` | 6 | 1.00 | 0.906 | 1.265 |
| `media` | 12 | 1.00 | 0.478 | 1.392 |
| `name_typo` | 4 | 1.00 | 0.14 | 0.161 |
| `natural` | 3 | 1.00 | 1.521 | 1.981 |
| `person_date` | 12 | 1.00 | 0.04 | 0.06 |
| `person_media` | 4 | 1.00 | 0.037 | 0.071 |
| `person_only` | 18 | 1.00 | 0.151 | 0.894 |
| `person_recency` | 24 | 1.00 | 0.143 | 0.198 |
| `person_topical` | 6 | 1.00 | 1.162 | 2.509 |
| `person_topical_participant` | 12 | 1.00 | 1.453 | 2.146 |
| `self` | 2 | 1.00 | 0.721 | 1.403 |
| `self_topical` | 6 | 1.00 | 0.237 | 1.25 |
| `surface_lower` | 12 | 1.00 | 0.131 | 0.616 |
| `surface_plain` | 12 | 1.00 | 0.139 | 0.19 |
| `surface_possessive` | 4 | 1.00 | 0.144 | 0.171 |
| `surface_punctuated` | 8 | 1.00 | 0.136 | 0.182 |
| `surface_quoted` | 12 | 1.00 | 0.136 | 0.171 |
| `surface_typo` | 12 | 1.00 | 0.131 | 0.178 |
| `surface_upper` | 12 | 1.00 | 0.128 | 1.229 |
| `time_of_day` | 3 | 1.00 | 0.015 | 0.401 |
| `topical` | 6 | 1.00 | 1.218 | 1.341 |
| `topical_date` | 2 | 1.00 | 1.128 | 1.134 |

Filter-only queries (`person_*`, `surface_*`, `date_only`, `time_of_day`)
answer in **0.02–0.2 s** because they skip the models entirely. Queries with
a semantic component sit at **0.9–2.5 s**.

### 9.2 Parser probe — 1000 cases, model-free

`plain`, `lower`, `upper`, `possessive`, `punctuated`, `quoted`, `typo` and
the corpus negative set **all 1.00**. Budgets met in both modes:

* normal: `python -m seaglass.eval.parser_probe --chat-db ~/.seaglass/chat_snapshot.db`
* contacts unavailable (§7): the same 1000 cases with `--no-contacts`, all 1.00.

### 9.3 Golden-set recall (`eval/score.py`, 32 items)

recall@50 0.62, recall@12 0.41, recall@final 0.41, filter kill rate 0.03 —
**unchanged** before and after this work, confirming no regression. The set
is small and dominated by `media_geo`; its one `exact_string` case is
mislabelled (a topical query, not a quote), so it does not exercise the new
phrase arm.

### 9.4 Defects this suite found

Every one of these was invisible to v1.

| defect | why it mattered | fix |
| --- | --- | --- |
| `fuzzy_match` was case-sensitive | rapidfuzz applies no processor by default, so **every lowercase name** failed to resolve — and the assist path with it, since it looks up Copilot's verbatim span | `processor=default_process`, plus exact and typo resolvers ordered by precision |
| `recency_ranked` sorted **within** each 900-id batch and concatenated | the result was "batch order, then recency", so a top-50 slice took the newest of an arbitrary batch. Above ~2000 candidates "what did I tell Vamski" answered with yesterday and dropped today | merge the batches into one global ordering |
| media filter was chunk-level only | `chunks.has_attachment` marks a *conversation*, so "pictures Kaya sent" returned 80 hits of which 5 were pictures | narrow candidates to chunks where *that sender* attached something, and filter hits on `has_attachment` |
| verbatim phrases could be unretrievable | a pasted sentence spreads across hundreds of near-ties in both dense and BM25 arms, so an exactly-matching chunk ranked below the fused cut | fourth retrieval arm: `phrase_search`, multi-word only |
| empty index raised on every search | calibration samples the chunks, so a pre-first-sync index has no `int8_absmax` and searching it was a stack trace, not "no results yet" | return empty when the index has no chunks; still raise for a *populated* index with no calibration |
| `apply-assist` 500 on a malformed body | a missing `assist_token` reported as a server error, sending the reader into the pipeline instead of their own payload | 400 |
| oracle scored the wrong half of the conversation | `is_from_me = 0` was hardcoded, so "what did I tell Kaya" was judged against what *Kaya* wrote — the engine was marked wrong for being right | direction-aware oracle; outgoing scoped by chat membership, since my group messages carry `handle_id` 0 |
| recency reservation ignored the query | the two reserved slots went to the strictly newest sessions, so on the 4-session page Grogu asks for, **half the answer** could be unrelated. "what did kaya say about the boat" spent a slot on a sourdough conversation while a day with two boat messages sat below the cut | scale the slots with the page (`head_size // 3`) and prefer sessions whose chunks matched the query's terms — the BM25 arm's ids, threaded out of `retrieve`, since the phrase arm only fires on a verbatim sentence |
| recall measured against page 1 only | the 20 newest messages from a chatty contact span more day-sessions than one page holds | the oracle follows `has_more` for up to two extra pages; latency still judged on page 1 |


## 10. Grogu and the app, side by side

Everything above measures one front door. Grogu is the other, and it is
the one most of these queries actually arrive through — so the suite now
runs both on the same cases, in one process, against one running app:

```
.venv/bin/python -m seaglass.eval.behavior \
  --index-db ~/.seaglass/index.db --chat-db ~/.seaglass/chat_snapshot.db \
  --compare --json-out /tmp/compare.json
```

Both targets share an engine, so every difference is something Grogu
does to the answer on its way to the caller rather than a second engine
ranking it differently. Grogu is driven through `search_via_seaglass`,
its real public entry point — paging, limits and ordering are decisions
that belong to the code under test, and a harness that reaches past them
into the flattener measures code no caller runs.

### 10.1 First run

The app scored 1.00 on every property. Grogu, asking the same questions
of the same engine, failed **160 of 208 queries**:

| metric | app | grogu |
| --- | --- | --- |
| `recency_order` | 1.00 | 0.01 |
| `no_self_in_sender` | 1.00 | 0.23 |
| `sender_purity` | 1.00 | 0.64 |
| `precision` | 1.00 | 0.51 |
| `recall_top` | 1.00 | 0.50 |
| sessions reaching the caller (mean) | 5.8 | 1.7 |
| failing queries | 0 | 160 |

None of this was a ranking problem. Seaglass returns a ranked page of
sessions, each with its matches and the context it expanded them with;
Grogu flattened that into one list and truncated it to a message count.

### 10.2 Defects found, all on the Grogu side

| defect | why it mattered | fix |
| --- | --- | --- |
| truncation kept only the first session | flattening emitted one session completely before starting the next, so a 20-message limit was spent inside session 0 — which alone holds 50+ messages once context is expanded. Eight sessions were reranked and hydrated; 1.7 reached the caller | share the budget: every session contributes before any contributes twice, then the surplus follows the reranker's order |
| context was indistinguishable from a match | surrounding messages are frequently from the other participant or from the user; presented as results they took sender purity to 0.23 and precision to 0.51 | matches fill the budget first, every row is labelled `kind`, context only tops up an answer short of matches |
| a declared ordering was discarded | seaglass says `ordering: "recent"` for "latest from Sam"; per-session emission scrambled it | honour it explicitly |
| a match was demoted by a neighbouring session | the same message is a match in one session and context in another; keeping whichever came first made it context, and matches are spent first, so it fell off the limit — six of one contact's newest twenty vanished, each a match elsewhere in the same answer | a match anywhere is a match |
| eight days could not hold twenty messages | a contact who texts in bursts across two handles has their newest twenty spread past the eight day-sessions one page holds; recall of the true newest sat at 0.77 | ask again, wider — a chronological answer is a filter, not a search, so seaglass skips the models and answered sixteen days *faster* than eight |
| stitched pages had holes | two pages ranked separately and concatenated gave a locally sorted answer missing messages from the middle | one wider request, ordered once, globally; a page is fetched only when even the wide answer came back short |
| the limit went to messages that never matched | a session's `messages` are the whole matched stretch of conversation and only some of them are why it matched — `match_score` is 0 for the rest. Asked "what did kaya say about the boat" with a limit of six, Grogu answered with a winking emoji, "hi hi" and "it's actually so nice out", and never mentioned a boat | spend the budget on scored matches first |
| `search_messages` could not page | the HTTP API supported `offset`; the MCP tool hid it, so no caller could ever reach page two | expose it |

Two were harness defects, fixed there: `per_case` keyed on the query
alone, so queries appearing in several classes reported one case's
numbers twice; and Grogu's latency included the harness's own hydration
round trips, which no caller pays.

### 10.3 After

| metric | app | grogu |
| --- | --- | --- |
| every property | 1.00 | 1.00 |
| every oracle metric | 1.00 | 1.00 |
| failing queries | 0 | 0 |
| p50 latency | 0.53 s | 0.48 s |
| p90 latency | 1.55 s | 1.46 s |

Grogu still returns fewer messages by design — a mean of 18.9 against
the app's 69 — because `--limit 20` is a message count and the app shows
a whole page. It now spends those 20 on the same answer the app would
give.

Three queries per target exceeded 10 s during the run, on *different*
queries for each target, and all four re-ran in under 2 s: the machine
stalled mid-run, not the code. Judge p50/p90, and re-measure outliers
before believing them.

## 11. Auditing the suite: what a clean report was hiding

§10 ended with every property at 1.00 and 0 failing queries on both
targets. That is the shape of a suite that has stopped measuring, so the
next step was to audit the properties themselves rather than trust them.

### 11.1 Properties that could not fail

| property | what it actually asked | cases graded |
| --- | --- | --- |
| `freshness_declared` | `"index_stale" in payload` | 208, all pass by construction |
| `ordering_declared` | `"ordering" in payload` | 208, all pass by construction |
| `hits_before_context` | is the first session non-empty | duplicate of `no_empty_sessions` |
| `lexical_presence` | verbatim phrase present | **6** of 208 |

Three of sixteen properties were structurally incapable of failing, and
they contributed three guaranteed 1.00 rows to every report. The fourth
-- the property that has caught more real defects than any other -- was
graded on six cases, all drawn from the newest 600 messages, i.e. one
week of one conversation.

### 11.2 The scorer was repairing the answer before grading it

Worse than a weak property: `as_grogu_shows_it` split the flat list
Grogu returns into `messages` and `context_messages`, and
`_flat_in_caller_order` reassembled it as hits-then-context. Grogu's
actual emitted order was discarded and replaced with the order the
properties expected, so **no ordering defect in the code under test was
observable**.

### 11.3 Two real defects found underneath the clean report

Both in `grogu_imessage.py`, both live while §10 reported 1.00.

| # | defect | why the suite missed it |
| --- | --- | --- |
| 1 | Flattened output was grouped by session, so session 1's *context* -- usually a message the user sent themselves, matching nothing -- outranked session 2's actual **match**. The exact defect PR #30 fixed and PR #33 reintroduced. | The property named `hits_before_context` never checked it; the scorer re-sorted the output anyway. Grogu's own `test_hits_are_spent_before_context` compared a **set** at `limit=4`, so it graded budget allocation, never reading order. |
| 2 | `match_score` ranking lived inside the budget-sharing path, so `_flatten_seaglass_result(payload)` with no limit returned messages in the order they were sent, with the match buried among its neighbours. | Every eval case drives `search_via_seaglass`, which always passes `limit=20`. The unlimited path had no coverage. |

### 11.4 What changed

- `freshness_declared` now requires staleness to agree with the count of
  messages the answer could not see (`(behind > 0) == stale`, which is
  exactly `engine._freshness_fields`' invariant).
- `ordering_declared` now requires the value to be known, and a declared
  chronological answer to actually be chronological -- for a target that
  emits one flat list. A nested payload has no single order, which is
  what `recency_order` / `recency_session_order` already grade.
- `hits_before_context` is replaced by `context_after_hits`, read off the
  order the caller actually receives.
- `as_grogu_shows_it` records `_caller_order`; `_flat_in_caller_order`
  reads it back verbatim.
- Verbatim phrases: 6 -> **24**, strided across 4000 messages so they
  span months and speakers, still deterministic.
- `tests/test_behavior_properties.py` (17 tests) feeds each property a
  payload violating exactly one invariant. A property that stops
  discriminating now fails a test instead of reading 1.00 forever.

### 11.5 Re-measured

Against the *pre-fix* Grogu, the rewritten property scores **0.83** on
`person_topical` and names the failing query -- where the old suite
reported 1.00 on everything. Against the fixed code it returns to 1.00.

Full run, 226 cases (208 + 18 new verbatim), both targets:

| property | app | grogu | graded |
| --- | --- | --- | --- |
| `context_after_hits` | `--` | 1.00 | 224 |
| `lexical_presence` | 1.00 | 1.00 | 24 |
| `freshness_declared` | 1.00 | 1.00 | 226 |
| `ordering_declared` | 1.00 | 1.00 | 226 |
| everything else | 1.00 | 1.00 | as §10 |

0 failing queries on both sides -- but now the properties have been shown
to fail on code that deserves it.

**The rule this earns:** a property that has never failed is not
evidence, it is an untested assertion. Before believing a clean report,
break the code on purpose and confirm the suite notices.

---

## §12 — What running the suite against the live tail turned up

Wiring the chat.db tail (filters-only queries answered from the live
database rather than the index) made the suite fail 31 of 226 cases. Only
one of those was a real engine defect, and it was not the one the report
pointed at. Chasing each to root cause found two genuine bugs that had
nothing to do with the feature under test.

### 12.1 The oracle was scored against the wrong horizon

`score_against_oracle` clipped ground truth to `MAX(end_ts)` of the
index, on the reasoning that a message the index has never seen cannot be
ranked. That reasoning stopped being true the moment the engine could
read past the index. The engine answered "latest from Kaya" with the
genuinely newest message, the oracle compared it against the newest
*indexed* message, and 30 person-recency cases were marked wrong for the
improvement.

The horizon is now dropped when the payload declares
`unindexed_included`. The allowance is earned by the coverage flag, not
granted to everyone — a second test pins that an index-only answer is
still clipped, so a genuinely stale answer cannot score clean.

### 12.2 The oracle opened a live database as `immutable`

`Corpus` opened the live `chat.db` with `immutable=1`. That flag asserts
the file cannot change, so SQLite skips the WAL entirely.
`imessage/source.py` documents this exact hazard in its docstring —
"never `immutable=1` ... yields silently corrupt reads on a live
database" — and the oracle violated it.

It surfaced as `database disk image is malformed`, but only once Messages
was actively writing; every previous run had been against a quiescent
file. The loud failure is the lucky case. The dangerous one is the
documented one: silently corrupt reads *in the component that decides
whether the engine is right*, which would mark correct answers wrong with
no error anywhere.

### 12.3 FTS5 MATCH was being handed raw user text

The find that mattered most, and it was not what the failing case was
about.

`sparse_search` passed the user's query straight into `chunks_fts MATCH`.
FTS5 MATCH is a query language, not a string: `-` is NOT, `:` is a column
filter, `"` opens a phrase, and `AND`/`OR`/`NOT` are keywords. So:

| query | BM25 hits (before) |
|---|---|
| `what about the boat?` | **0** |
| `let's meet` | **0** |
| `re: lease` | **0** |
| `not contagious after 12-24 of antibiotics` | **0** |
| `dinner plans` | 20 |

Every one of those raised inside SQLite, and the handler failed open to
`return []`. An ordinary question containing an apostrophe, a question
mark, a hyphen or a colon silently lost the entire lexical half of the
hybrid search and was answered by the vector half alone.

It survived this long precisely because it failed *quietly* and
*plausibly* — the search still returned topical-looking results, so
nothing looked broken. This is the same shape of bug as §11's vacuous
properties: the system reported success while doing less than it claimed.

`fts_match_query` now extracts word tokens and quotes each one, which
keeps the implicit-AND semantics the unquoted form had while making every
input syntactically inert. Seven tests, all mutation-verified against a
raw-text passthrough.

### 12.4 The one case that is still red, and why it is not being "fixed"

`lexical_presence` on `not contagious after 12-24 of antibiotics` is a
**parser** false positive, not a lexical one. `parse_query` reads
`after 12-24` as a date filter — Dec 21–27 2025 — and strips those tokens
from `semantic`. The message the phrase was drawn from is from July 2026,
so the date filter excludes the very message being searched for, and the
surviving `not contagious antibiotics` no longer contains the phrase for
`phrase_search` to match.

Both halves of the retriever are working correctly here; they are being
asked the wrong question. Re-tuning date extraction is a real change with
real regression risk against every legitimate date query, so it is
recorded here rather than rushed. `"12 24"` (unhyphenated) parses as a
date too, so any fix has to address the heuristic, not the hyphen.

### 12.5 A caveat on measuring recency against a live corpus

`newest_is_true_newest` compares an answer computed at query time against
an oracle read at scoring time. On a corpus that is actively receiving
messages, a message arriving between those two moments marks a correct
answer wrong. One `self_topical` case failed exactly this way and passed
cleanly on re-run.

Re-run a single-case recency failure before believing it. A failure that
does not reproduce on a live corpus is evidence about the clock, not
about the engine.

### 12.5b Recall was clipped to the indexed half as well

The same assumption as §12.1 lived a second time, twenty lines below it:
`reachable` filtered the truth head to messages present in
`chunk_message`. When the tail is live the newest messages are reachable
*because* they are unindexed, so this demanded the older, indexed half of
the head and marked a correctly-newest answer a recall miss. It hit grogu
hardest — 31 cases — because its 20-message limit was entirely filled by
newer, real results. Same fix, same guard: dropped only when the payload
declares `unindexed_included`, with a test pinning that an index-only
answer is still clipped.

The lesson generalises past this one flag: **when a component gains a new
capability, every place the harness encoded the old limit as an
assumption becomes a false failure.** Grep for the assumption, do not fix
the first instance and stop.

### 12.6 Result

226 cases against a synced index, both targets, **zero failures**:

| metric | app | grogu |
|---|---|---|
| properties | 16/16 at 1.00 | 16/16 at 1.00 |
| `newest_is_true_newest` | 1.00 (was 0.79) | 1.00 |
| `precision` | 1.00 | 1.00 |
| `recall_top` | 1.00 | 1.00 |
| `lexical_presence` | 24/24 | 24/24 |

`recency_session_order` is checked only on the app (111 cases) and
`context_after_hits` only on grogu (221) — each grades a shape only that
target emits, which is correct, not a gap.

Measured under a *deliberately stale* index the numbers are lower
(`precision`/`recall_top` 0.966 for grogu, `indexed_coverage` 0.0) while
`newest_is_true_newest` and `newest_present` stay at 1.00 — the tail
keeps the recency guarantees exact when the index is hours behind, which
is the whole point of it, while a limit-20 caller necessarily trades some
breadth for that freshness.

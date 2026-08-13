# Query Behaviour Evaluation — everything `EVALUATION.md` does not measure

> Companion to `EVALUATION.md`, which measures **recall on topical queries**
> against a hand-reviewed golden set. That document is deliberately scoped to
> "did the semantically relevant chunk survive the pipeline". This one covers
> the other half of the query space: queries whose correctness is decided by
> **filters, ordering and freshness** rather than by embedding similarity.

---

## 1. Why a second evaluation exists

Every serious bug found in the Grogu integration was invisible to
`recall@final`:

| Bug | Symptom | Why the golden set missed it |
|---|---|---|
| `from X` filtered by *chat*, not *sender* | "messages from Jakie" returned Vamski | No golden query names a sender |
| Filler embedded as a topic | "latest from Adrian" ranked on the noise vector of the word "latest" | Golden queries all *have* topics |
| Browse pseudo-scores saturated the aggregation sigmoid | ranking became "whichever day has the most chunks" | Browse mode is never exercised |
| Outgoing 1:1 messages carry the recipient's `handle_id` | "from Kaya" returned my own messages | Recall does not care who wrote it |
| Index lag invisible | "did he reply yet" answered with an older message | The golden set is scored against a fixed snapshot |
| Context ranked above hits (Grogu) | top result was a neighbour of the match | Scoring is per-chunk, not per-message-order |

The common thread: **recall asks "is the right thing somewhere in the
payload"; these bugs are about what is at the top, who wrote it, when it is
from, and whether the corpus was current.** A result can have perfect recall
and still be a wrong answer.

## 2. The core asset: chat.db is an oracle

`EVALUATION.md` §2 explains why random sampling fails for topical queries —
"ok"/"haha" chunks have many equally correct answers, so a miss is not a miss.
That argument **does not apply to filter queries**, and this is what makes this
evaluation cheap and exact:

> "the last message Vamski sent me", "everything Kaya sent last week",
> "photos from Adrian in July" — each has **exactly one correct answer set**,
> and that set is computable with a SQL query against chat.db.

So this suite needs **no human labelling and no LLM**. The oracle is written
once, and every filter query is scored as exact precision/recall against it.
Where the golden set is 32 reviewed queries, this suite can generate hundreds
from the live corpus and stay correct as the corpus grows.

Consequence for tuning: a regression here is unambiguous. There is no "the
model found an equally good chunk" defence.

## 3. Query taxonomy

Classes are defined by **what decides correctness**, not by surface wording.

| # | Class | Example | Correctness decided by |
|---|---|---|---|
| 1 | Topical | "what did we decide about the boat" | embedding similarity (covered by `EVALUATION.md`) |
| 2 | Verbatim / rare string | "Roquette", a URL, an address | lexical match must win over semantic drift |
| 3 | Person-only | "messages from Kaya" | sender filter + recency |
| 4 | Person + recency | "latest from Adrian", "what did she just say" | sender filter + time ordering + **index freshness** |
| 5 | Person + topical | "what did Vamski say about golf" | sender filter ∧ similarity |
| 6 | Date-only | "last week", "in March", "yesterday" | range parsing (not point-in-time) |
| 7 | Person + date | "messages from Jakie last week" | conjunction of two filters |
| 8 | Topical + date | "dinner plans last month" | filter must narrow, not replace, ranking |
| 9 | Media | "photos from Kaya" | attachment filter + sender |
| 10 | Group vs 1:1 | "what did the group say about the trip" | `is_group` intent |
| 11 | Self | "what did I say about the deposit" | `is_from_me` as a sender |
| 12 | Ambiguous / bare | "golf", "ok" | must not crash, must not return nothing |
| 13 | Fuzzy name | "Jakie" → "Jakie Poo Smith"; first-name collisions | contact resolution threshold |
| 14 | Natural question | "did he ever reply about the boat?" | pronouns, question words must not become topics |
| 15 | Adversarial | month-name people ("May"), typos, emoji, very long queries | parser guards |

Classes 3, 4, 6, 7, 9, 11 are **oracle-scorable**. Classes 1, 2, 5, 8, 10, 12,
14 are **property-scorable** (see §4). Class 13 is scored on resolution, not
retrieval.

## 4. What "correct" means, per property

Properties are checked on every applicable query, so one bug is caught by
whichever query trips it first.

| Property | Statement | Catches |
|---|---|---|
| `sender_purity` | every hit was written by a named sender | the participant-vs-sender bug |
| `no_self_in_sender` | no `is_from_me` hit when a sender is named | the `handle_id` bug |
| `date_containment` | every hit ts ∈ parsed range | range-parsing regressions |
| `recency_order` | `ordering=recent` ⇒ hits non-increasing in ts | the sigmoid-saturation bug |
| `newest_is_true_newest` | for a recency query, the top hit **is** the oracle's newest | the whole class-4 failure |
| `no_empty_sessions` | every session has ≥1 hit message | system-row sessions |
| `hits_before_context` | no context message precedes a hit | the Grogu flattening bug |
| `nonempty` | a query with a known answer returns something | over-filtering / starvation |
| `lexical_presence` | a verbatim query's top session contains the string | semantic drift over exact match |
| `stability` | same query twice ⇒ identical payload | hidden nondeterminism |
| `pagination_disjoint` | page 2 shares no session with page 1 | offset bugs |
| `freshness_declared` | payload reports lag when the index is behind | silent staleness |

`newest_is_true_newest` deserves emphasis: it is the single strongest check in
the suite, because it compares against chat.db rather than against the index,
so it fails both when *ranking* is wrong and when the index is merely *stale*.
Those need different fixes, so the harness reports them separately.

## 5. Circumstance matrix

The same query must be evaluated under conditions that change which code path
answers it. Each circumstance has broken something at least once.

| Circumstance | Why it matters |
|---|---|
| Desktop app running | MCP proxies over loopback; a different process answers |
| No app running | in-process lazy pipeline; **cold** model/contact loading |
| Cold vs warm process | first-query latency is what a Grogu user actually feels |
| Index current | baseline |
| Index stale by N | recency answers are wrong *and* must say so |
| Contacts unavailable | name → handle resolution fails; must degrade, not crash |
| Unhydrated (no chat.db) | preview mode; payload shape must stay valid |

## 6. Metrics

**Quality:** per-class property pass rate; per-class oracle precision/recall
for the oracle-scorable classes. A property that is inapplicable is skipped,
never counted as a pass.

**Latency:** p50/p95 per class, split cold vs warm and proxy vs in-process,
with a per-class budget rather than one global number — a filter-only query
skips the models entirely and should be an order of magnitude faster than a
topical one. Reporting them together hides exactly that.

| Class | Budget (warm) | Rationale |
|---|---|---|
| Filter-only (3, 4, 6, 7, 9, 11) | < 250 ms | no embedding, no rerank — pure SQL + hydrate |
| Topical (1, 2, 5, 8) | < 1.5 s | embed + rerank dominate |
| Any class, cold | < 4 s | model load amortised over the session |

**Memory:** peak RSS for a filter-only query must stay near baseline — if it
climbs to ~1 GB, something is loading models the query never needs.

## 7. Tuning loop

1. Run the suite; record per-class quality and latency to a JSON report.
2. Take the **worst-scoring property**, not the worst-scoring query — a single
   failing query is an anecdote, a failing property is a bug class.
3. Find the root cause in the engine. Fix the engine, never the harness.
4. Add a unit test that fails without the fix, so the suite does not become the
   only thing standing between the bug and the user.
5. Re-run the full suite: confirm the fix, and check nothing else moved.
6. Re-run `EVALUATION.md`'s recall harness before shipping — the two suites
   pull in different directions (filters narrow, recall wants breadth) and a
   filter change can quietly cost topical recall.

**Rule:** the harness drives `SearchEngine`, the same object the app and the
MCP server use. `eval/score.py` reimplements the pipeline, which is how the
recency reservation and the lexical boost ended up measured but not shipped —
and how browse mode ended up shipped but never measured.

## 8. Non-goals

Answer quality, citation validity and refusal behaviour remain out of scope,
for the reason `EVALUATION.md` gives: generation belongs to the MCP client's
model. This suite stops at the payload.

## 9. First results, and what they found

Run: `python -m seaglass.eval.behavior --index-db ~/.seaglass/index.db
--chat-db ~/.seaglass/chat_snapshot.db --chat-db-source ~/Library/Messages/chat.db`
(136 cases over 18 classes, generated from this machine's corpus).

| metric | before | after |
|---|---|---|
| precision (filter-only) | 1.00 | 1.00 |
| recall_top (newest 20 reachable) | 0.64 | **0.97** |
| newest_present | 0.66 | **1.00** |
| newest_is_true_newest | 0.35 | **1.00** |
| recency_order | 0.02 | **1.00** |
| recency_session_order | — | **1.00** |
| every other property | — | 1.00 |

Six defects, none of which recall@final could have seen:

1. **Hits came back oldest-first inside a session.** The payload said
   `ordering: recent` while handing the caller the oldest message of the newest
   day. Fixed by ordering hits newest-first on the wire and sorting for display
   in `app.js` — the wire format should mean what it says, and the UI should not
   depend on the order it happens to receive.
2. **The trim budget was per session.** "What did Kaya say yesterday" returned
   10 of yesterday's 40 messages purely because they landed on one day. The
   budget is now global and derived from `max_sessions`, which is what the
   caller actually asked for.
3. **Sessions were ordered by score, then not re-sorted.** Two chats on the same
   day could come back in either order under `ordering: recent`.
4. **`chat_handle_join` is not a complete roster.** A 3,614-message SMS group
   listed none of its participants, so the participant prefilter discarded that
   person's busiest chat. Membership is now the union of the roster and
   authorship: someone who wrote in a chat was in it.
5. **The sender lookback counted orphaned rows.** This chat.db holds 109,665
   messages from one contact of which **180** are attached to a chat (a partial
   iCloud sync). The lookback's newest 20,000 rows were therefore all unlinked,
   no chunk contained them, and the narrowing silently failed open. Searching
   for her returned 6 of a possible 20 messages; it now returns 17.
6. **"What did I say about X" had no sender filter**, so it answered with what
   the other person said back — on topic, which is what made it hard to see.

Two smaller ones: empty results omitted `index_stale`/`ordering`, giving
consumers two payload shapes; and warmup's MLX buffer cache (~645 MB) was held
for the life of the process, which the build path already released and the
query path — the one that stays running — did not.

### Metrics that exist to keep blame in the right place

`newest_is_indexed` (0.60 on a deliberately stale index) and
`indexed_coverage` (1.00) separate three failures that look identical from the
payload: the engine ranked badly, the message arrived after the last build, or
the message was never indexed at all. They need opposite fixes, and only the
first is a ranking problem.

### On latency

Only `p50` is trustworthy here. `p95` showed 100–950s outliers on a 16 GB
laptop — including on filter-only queries that run no models at all, which is
proof they are machine stalls rather than engine cost. The same queries run in
1–3s in isolation. The harness now warns when the app is running, since two
copies of the models contend for memory: a rerank that takes 2.6s alone took
96s alongside the app. Latency belongs in a quiet, dedicated run.

Warm p50 by class: `person_date` 0.06s, `person_recency` 0.07s, `person_only`
0.10s, `date_only` 0.55s, `lexical` 0.95s, `topical` 2.87s.

### Still open

- `lexical_presence` 0.83: one exact phrase in the corpus is not retrievable —
  the phrase is real, so this is a retrieval gap, not a sampling artefact.
- Sraddha/Anjanette still return 17 and 11 of 20. Their remaining messages are
  in the index but are not surfacing, which is now a ranking question rather
  than the coverage question it was.

# Evaluation Strategy — Retrieval Only

> Companion to `PLAN.md` and `DESIGN-NOTES.md`.
>
> **Scope: the search stack only.** Three recall points — **recall@50**,
> **recall@12**, **recall@final**. No answer grading, no LLM-as-judge, no
> citation checking. Generation belongs to the MCP client's model and is
> outside this system.

---

## 1. Scope and rationale

**Measured:**

| Metric | Definition | What it tells you |
|---|---|---|
| **recall@50** | positive present in the fused candidate set (post-RRF, pre-rerank) | **The ceiling.** Everything downstream is bounded by this |
| **recall@12** | positive survives cross-encoder reranking | Whether the reranker preserves or discards the answer |
| **recall@final** | positive is in the payload the MCP tool actually returns | Whether aggregation, expansion and the `max_sessions` cap drop it |

**Not measured (v1):** MRR, nDCG, answer faithfulness, citation validity,
refusal behaviour. These can be layered on later; they are not needed to know
whether the search stack works — and answer quality now belongs to whichever
model the MCP client supplies, which is outside this system entirely.

**Why these suffice:** recall@50 bounds the entire system — if the answer is not
in the candidate set, no reranker and no language model can recover it.
recall@12 isolates the reranker. **recall@final is the one that matters most**:
the tool's return payload *is* the client model's context, and there is no
downstream stage to recover an answer dropped at the last cut.

Everything below exists to make these numbers **trustworthy**, which means the
labelled set must be grounded in real content with unambiguous positives.

---

## 2. Why random sampling fails here

The obvious approach — sample random chunks, generate a question from each,
treat that chunk as the positive — produces near-worthless data on a message
corpus.

Most iMessage chunks are conversational filler: `"ok"`, `"haha"`, `"on my way"`,
`"sounds good"`. There are thousands of near-identical ones. A question generated
from such a chunk has **many equally correct answers**, so:

- A "miss" is not a miss — retrieval found an equally valid chunk
- Recall is systematically understated by an unknown amount
- The number moves when you change parameters, but not for the reason you think

**Requirement for a valid positive: exactly one chunk in the corpus should
plausibly answer the question.** That property is measurable, and the moment to
measure it is during indexing.

---

## 3. Harvesting candidates

> **Design rule: harvesting must not add meaningful cost to the indexing hot
> path.** Indexing is the one expensive, hard-to-restart operation in the system.

Because `chunks.body_semantic` is **stored** (zstd-compressed, `PLAN.md` §5), harvesting
is now **entirely a post-build job**. Nothing runs inline, and nothing needs
regenerating from `chat.db`:

```
harvest.py --stage sql     # regex + chunks + chat + temporal signals   ~2 min
harvest.py --stage idf     # one FTS5 statistics query                  ~30 s
harvest.py --stage nn      # 5 000 × measured scan latency              §3.2
harvest.py --stage score   # bands, composite score, category           ~5 s
```

Standalone, resumable, re-runnable, and **zero cost to indexing**. (An earlier
revision computed the regex signals inline during embedding, because the text was
not stored and recreating it was expensive. Storing `body_semantic` removed that
constraint along with several others.)

### 3.1 Signals

| Signal | Source | Indicates |
|---|---|---|
| `entity_count` (capitalised non-sentence-initial tokens), `has_url`, `has_number`, `has_long_token`, `token_count` | regex over `zstd_decompress(chunks.body_semantic)` | question-able content; `exact_string` candidates |
| `has_attachment` (`chunks`), `has_geo` = EXISTS in `attachment_place`, `msg_count` = COUNT over `chunk_message`, `is_group` from `im.chat.style` | mixed | `media_geo` candidates |
| `participant_count` | `im.chat_handle_join` for the chunk's `chat_id` | `person_filtered` candidates |
| `temporal_isolation` — days to nearest other chunk in the same chat | `start_ts` window function | `time_filtered` candidates (a date pins it down) |
| `idf_mean` — mean IDF of the 5 rarest terms | FTS5 statistics | distinctive vocabulary; topic anchor. **Necessarily after FTS5 exists** |
| **`nn_distance`** — cosine distance to nearest *other* chunk | `chunks_vec` | **primary signal** — see §3.2 and §3.3 |

### 3.2 Making `nn_distance` affordable

All-pairs over 1.25 M chunks is O(n²) and infeasible. It is not needed:

1. **Prefilter with the inline signals** down to ~5 000 candidates
   (`token_count` in range, `idf_mean` above median, `entity_count >= 2`).
2. **Compute `nn_distance` for those 5 000 only**, each scanned against the full
   int8 index.
   ⚠️ **Budget = 5 000 × the Phase 0 measured scan latency**, not an assumed
   50 ms. At 50 ms that is ~4 min; at 300 ms it is ~25 min. If the SQL round-trip
   dominates, do it as **one batched NumPy pass over the mmapped int8 array**
   instead of 5 000 separate KNN queries — same arithmetic, far less overhead.
3. Store results in `eval_candidate`.

Exclude self-matches and chunks sharing messages with the source: adjacent chunks
overlap by 2–3 messages by design and will always be near neighbours.

⚠️ Use the **same distance metric as retrieval**. `vec0` defaults to L2; the
schema declares `distance_metric=cosine` (`PLAN.md` §5). int8-quantised vectors
are no longer exactly unit-norm, so L2 and cosine are not interchangeable.

### 3.3 Composite priority

```
score = w1 · band(nn_distance)      # unambiguous, but BANDED — see warning
      + w2 · norm(idf_mean)         # distinctive vocabulary
      + w3 · norm(entity_count)     # question-able
      + w4 · category_bonus         # fills thin categories
      - w5 · penalty(token_count outside [150, 600])
```

> ⚠️ **Do not maximise `nn_distance`.** Selecting only the most semantically
> isolated chunks builds an eval set of outlier conversations with no
> distractors, which dodges the hardest part of dense retrieval — separating
> closely related topics — and biases recall **upward**. The resulting numbers
> would not reflect real difficulty.
>
> Use it as an **exclusion**, not an objective: drop the bottom band (near-
> duplicate boilerplate, where positives are ambiguous), then **sample across the
> remaining distribution** with quotas:
>
> | `nn_distance` band | Share of the set |
> |---|---|
> | bottom 25 % (boilerplate) | 0 % — excluded, positives are ambiguous |
> | 25–50 % (crowded neighbourhood) | ~30 % — the *hard* cases |
> | 50–75 % | ~40 % |
> | top 25 % (isolated) | ~30 % |
>
> Record `nn_distance` on every golden entry so recall can be reported by band.
> If recall in the 25–50 % band is far below the top band, that gap is the real
> measure of how well dense retrieval separates similar conversations.

### 3.4 Automatic category assignment

Derived from the same signals, so stratification requires no extra work:

| Category | Rule |
|---|---|
| `media_geo` | row in `attachment_place`, or `has_attachment` |
| `exact_string` | `has_url` or `has_long_token` |
| `person_filtered` | group chat (`im.chat.style`) and `participant_count >= 3` |
| `time_filtered` | `temporal_isolation` above the 80th percentile |
| `multi_session` | >= 3 high-scoring chunks in one chat within 24 h |
| `topical` | default |

### 3.5 Table

```sql
CREATE TABLE eval_candidate (
  chunk_id            INTEGER PRIMARY KEY,
  nn_distance         REAL,      -- NULL until the §3.2 pass runs
  nn_chunk_id         INTEGER,   -- for manual inspection of near-duplicates
  idf_mean            REAL,
  entity_count        INTEGER,
  has_url             INTEGER,
  has_number          INTEGER,
  has_geo             INTEGER,
  has_attachment      INTEGER,
  is_group            INTEGER,
  participant_count   INTEGER,
  temporal_isolation  REAL,
  msg_count           INTEGER,
  token_count         INTEGER,
  category            TEXT,
  score               REAL
);
CREATE INDEX idx_evalcand_score ON eval_candidate(category, score DESC);
```

A build-time artifact: fully derived, rebuildable, and droppable once
`golden.jsonl` exists. Consistent with the "own only derived data" principle in
`DESIGN-NOTES.md` §12.

---

## 4. From candidates to a golden set

```
eval_candidate  →  top 300 stratified by category
                →  LLM question generation (offline batch; see PLAN Phase 3.5)
                →  automatic filters (§4.2)
                →  human review (§4.3)
                →  golden.jsonl  (~200 entries)
```

### 4.1 Question generation

For each selected chunk, generate a question answerable **only** from it, using
any capable LLM.

⚠️ **This sends chunk text off-machine** — ~300 sampled chunks, once, as an
offline batch. That is a deliberate exception to the local-retrieval boundary
(`PLAN.md` §1), acceptable because it is bounded, one-time, and not on any
interactive path. If unacceptable, hand-write the Tier-0 questions instead
(§4.3) and accept a smaller set.

Prompt requirements:

- Phrase it as **someone recalling the conversation months later** — vague,
  partial, the way real queries actually arrive.
- **Do not reuse content words** from the source.
- **Do not include dates or names** unless the category is `time_filtered` or
  `person_filtered` — those categories exist precisely to exercise the
  structured filters.
- Output the question only.

Generate 2–3 variants per chunk; keep the best at review.

### 4.2 Automatic filters — before human review

**Vocabulary-leakage filter.** Naively generated questions reuse the source
chunk's vocabulary, which inflates BM25 recall and makes the whole system look
better than it is. This is the most common way RAG evaluations mislead their
authors.

**Auto-reject only on content-word overlap > 40 %** — a property of the
*question*, not of retrieval performance.

> ⚠️ **Do not auto-reject on "sparse-only retrieval returns the source at rank
> 1".** That conditions the eval set on the dependent variable of the
> `sparse only` ablation (§8.1): you would be discarding precisely the questions
> BM25 handles well, then measuring BM25 and concluding it is weak. Selecting on
> the outcome invalidates the comparison the ablation exists to make.
>
> Use the rank-1 signal as a **review flag** instead — surface it in
> `review.py` so the human can distinguish "this is a legitimate keyword
> question" (keep — the `exact_string` category needs these) from "this is a
> generation artifact that copied the source's phrasing" (reject).

**Ambiguity filter.** Run the question through full retrieval and inspect the
top 5 excluding the source:

- Another chunk clearly answers it too → mark **multi-positive** (both ids count
  as correct) rather than discarding. These entries are realistic and valuable.
- More than ~3 plausible answers → discard; the question is too generic.

**Trivially-hard filter.** If the source chunk is absent from the top 200, the
question is probably nonsense rather than a genuine retrieval failure. Flag for
human review; do **not** auto-discard — occasionally these are real failures and
are the most informative entries in the set.

### 4.3 Human review — what makes it grounded

A minimal CLI (`eval/review.py`) showing the question, the source chunk, and the
top retrieval hits, with keys: **accept / edit / reject / mark-multi-positive**.

- ~300 candidates at ~6 s each ≈ **30 minutes**
- Expect to keep ~200

This step is what makes the set trustworthy: every positive is a real
conversation you have personally confirmed is answerable, phrased in words you
endorse. It converts synthetic data into grounded data for the cost of one sitting.

**Optional Tier 0 (~20 entries):** questions written purely from memory *before*
looking at anything, with the answering message located afterwards. The most
realistic queries obtainable — small, slow, and worth having as a sanity anchor
against the harvested set.

### 4.4 Format

```json
{
  "id": "ev-0142",
  "query": "where did we end up deciding to stay when we went to portugal",
  "positive_msg_ids": [482113, 482119],
  "alt_positive_msg_ids": [601442],
  "chat_id": 91,
  "category": "topical",
  "source_chunk_id": 88213,
  "nn_distance": 0.41,
  "origin": "harvested",
  "reviewed": true,
  "split": "dev"
}
```

**Label by message id, never chunk id.** Chunk boundaries move whenever chunking
parameters are swept — chunk-level labels would be invalidated by the very
experiments they exist to support. Map message id → chunk at scoring time via
`chunk_message`.

---

## 5. Validating the eval set itself

> **The failure mode that matters: a broken evaluation produces plausible
> numbers.** An off-by-one in the message-id range check yields a confident 0.83
> that means nothing, and nothing about the output looks wrong. Every control
> below exists to make that failure detectable.

### 5.1 Controls

| Test | Method | Expected | Catches |
|---|---|---|---|
| **Scoring unit tests** | 20-chunk fixture corpus with hand-written positives; assert exact recall for a perfect, a random, and an empty retriever | exact known values | range-check off-by-one, missing `chat_id` guard, wrong `k`, dedup bugs |
| **Rowid alignment (structural)** | `LEFT JOIN` both virtual tables against `chunks`, both directions | zero orphans | missing rows |
| **Rowid alignment (semantic)** | sample ids; re-embed the stored `body_semantic` and compare to the stored vector | cosine > 0.99 | **misalignment** — every row present but shifted. The structural check cannot see this |
| **Positive control** | query = the source chunk's own text, verbatim | source in **top 5** | broken index, missing query prefix |
| **Negative control** | 100 random unrelated queries | positive retrieved in **< 2 %** | scoring leakage, accidental id matching |
| **Ceiling / oracle** | oracle retriever that greps the positive's literal text | recall = 1.00 | **unreachable positives** — a broken *set*, not a broken system |
| **Mutation test** | shuffle `chunks_vec` rowids; truncate chunks to 50 tokens; randomise RRF | **recall drops ≥ 0.20 absolute** | an eval not measuring retrieval at all |

⚠️ **Use distributional thresholds, not absolutes**, for everything except the
unit tests and the oracle:

- *Positive control:* "rank 1, always" is wrong. Chunks overlap by 2–3 messages
  by design, so an adjacent chunk containing the same text can legitimately
  outrank the source. Top 5 is the meaningful assertion.
- *Negative control:* a random query occasionally retrieving a positive is
  expected, not a bug. Assert a rate.
- *Mutation test:* **disabling the dense arm is not a valid mutation** — on a
  keyword-heavy set, sparse alone may score nearly as well, and that is a real
  finding (§8.4), not a broken harness. Use mutations that must break any
  functioning retriever: shuffled rowids, truncated content, randomised fusion.

**The mutation test is the single most important control.** If you sabotage
retrieval and the score does not move, the evaluation is measuring nothing. Run
it before trusting any number, and re-run it whenever the harness changes.

The **oracle check** is its complement: it distinguishes "the system failed to
find it" from "the label was wrong and it was never findable."

### 5.2 Reviewer self-consistency

Re-review 30 already-decided entries **blind, at least a day later**. Measure
agreement with your earlier accept/reject.

- **≥ 80 % agreement** → criteria are reproducible; proceed
- **< 80 %** → your accept/reject criteria are too vague. Write them down
  explicitly, then re-review. The remaining entries are not trustworthy until
  this passes.

This is the cheapest possible guard against a set that encodes mood rather than
judgement.

### 5.3 Coverage audit

Before freezing, confirm the set is not accidentally concentrated:

| Check | Threshold |
|---|---|
| Distinct chats represented | ≥ 30 |
| Entries from any single chat | ≤ 15 % |
| Date range covered | ≥ 60 % of corpus span |
| Smallest category | ≥ 15 entries |
| Group vs 1:1 balance | neither below 20 % |
| `nn_distance` distribution | median above corpus median |

A set drawn overwhelmingly from three chats measures those three chats.

---

## 6. Statistics — how much can these numbers carry

With ~200 questions, **recall of 0.86 has a 95 % Wilson interval of roughly
±0.05**. A 2 % improvement is noise.

Per category with n = 20, the interval is **≈ ±0.19**. Per-category numbers are
**directional only** — never make a decision from one in isolation.

Two tools, both defined below: **Wilson intervals** answer "how precise is this
number?", **McNemar's test** answers "is config B actually better than A?".

### 6.1 Paired comparison for A/B decisions

For any config comparison — `residual` vs `filter_only`, RRF `k`, rerank depth —
do **not**
compare two independent proportions. The two configs are evaluated on *identical
inputs*, so analyse them paired. Cross-tabulate per question:

|  | B hit | B miss |
|---|---|---|
| **A hit** | n₁₁ — no information | n₁₀ |
| **A miss** | n₀₁ | n₀₀ — no information |

Questions both configs get right, and both get wrong, say nothing about which is
better. Only the **discordant pairs** carry signal. Under the null hypothesis
that the configs are equivalent, a question that flips should flip either way
with probability ½ — so it reduces to a coin-flip test:

```
χ² = (n₁₀ − n₀₁)² / (n₁₀ + n₀₁)      1 d.o.f.
```

Use the **exact binomial** form when `n₁₀ + n₀₁ < 25`.

**Why this matters.** Comparing 0.856 vs 0.867 as independent proportions at
n = 195 has a standard error of ±0.07 — a 1 % difference is invisible, and
resolving it would need thousands of questions. Paired analysis discards ~180
uninformative questions and analyses the ~16 that actually differ. Same data,
far more sensitivity.

⚠️ **McNemar tests equality, not acceptability.** Failing to reject the null is
*not* evidence that a difference is small — with 16 discordant pairs, a genuine
5 % loss would also fail to reject. So for any decision framed as "is B *at
least as good as* A", use a **non-inferiority** test with a stated margin:

1. Paired bootstrap over questions (10 000 resamples) → CI for `recall_B − recall_A`.
2. **Adopt B only if the lower bound clears the margin** (e.g. > −0.02).
3. Report McNemar alongside as a secondary equality check.

A wide CI that merely *contains* zero means "not enough evidence either way" —
which calls for more golden entries, not a green light.

### 6.2 Wilson intervals

Report an interval with every proportion. Use **Wilson**, not the textbook Wald
interval `p̂ ± z·√(p̂(1−p̂)/n)` — Wald gives zero width at p̂ = 1.0 (claiming
certainty from 20 questions) and can extend outside [0, 1].

```
         p̂ + z²/2n                  z          ⎧ p̂(1−p̂)     z²  ⎫
center = ───────────  ,   half = ─────────· √  ⎨ ─────── + ───── ⎬
         1 + z²/n                1 + z²/n      ⎩    n       4n²  ⎭
```

| n | recall | 95 % Wilson CI | width |
|---|---|---|---|
| 195 (overall) | 0.86 | [0.80, 0.90] | ±0.05 |
| 20 (one category) | 0.70 | [0.48, 0.86] | ±0.19 |

The second row is the operative one: **per-category numbers at n ≈ 20 cannot
support a decision.** They are a smoke alarm, not a measurement. Treat them as
directional, and only act on a category after confirming with more entries.

### 6.3 Reporting

Every reported number carries `n` and its interval:

```
overall r@50   0.86  [0.80, 0.90]   n=195

residual vs filter_only
               n₁₁=160  n₁₀=7  n₀₁=9  n₀₀=19
               Δ = +0.010, paired bootstrap 95% CI [-0.015, +0.036]
               lower bound < -0.02  →  not shown non-inferior; keep filter_only
               (McNemar χ² = 0.25, p ≈ 0.62 — secondary equality check)
```

Config choices are **paired non-inferiority** decisions. Framed that way ~200
questions can resolve them; framed as two independent proportions, they cannot.

---

## 7. Scoring

```python
positives = set(e["positive_msg_ids"]) | set(e.get("alt_positive_msg_ids", []))
retrieved = {c.id for c in ranked[:k]}

# chunk_message gives the reverse lookup directly — no range scan, no chat guard
hit = bool(retrieved & chunks_containing(positives))
```
```sql
-- chunks_containing()
SELECT DISTINCT chunk_id FROM chunk_message WHERE msg_id IN (...);
```

> ⚠️ **Never score with a `ROWID` range test.**
> `message.ROWID` is assigned globally and chronologically across *all*
> conversations, so a chunk covering a long window in a quiet chat spans
> thousands of messages from unrelated chats. A range test counts those as hits
> and silently inflates recall. `chunk_message` is the membership relation and
> the only correct source for this check (`DESIGN-NOTES.md` §9).

- **recall@50** — over the post-RRF, pre-rerank candidate set
- **recall@12** — over the post-rerank set
- **recall@final** — over the payload the MCP tool actually returns, after
  aggregation, expansion and the `max_sessions` cap. **The most important of the
  three**: this payload *is* the client model's context (`PLAN.md` Phase 5).
  Note: expanded neighbour chunks are context only and contribute no score, but
  they *do* count as hits here — being in the context is the thing being measured.
- A hit on **any** positive counts (multi-positive entries)
- Report **per category** and overall, plus the two gaps
  (`@50 → @12` isolates the reranker; `@12 → @final` isolates aggregation)

### Output shape

```
run 2026-08-20T09:12   config=hybrid+rerank/filter_only   split=dev
                       embed=bge-small/int8/v1

category            n     r@50    r@12   r@fin    gap₁    gap₂   kill
topical            72    0.93    0.85    0.83   -0.08   -0.02   0.00
person_filtered    31    0.84    0.77    0.77   -0.07    0.00   0.06  <- parser
time_filtered      24    0.71    0.71    0.67    0.00   -0.04   0.12  <- parser
exact_string       28    0.96    0.93    0.93   -0.03    0.00   0.00
media_geo          19    0.58    0.53    0.53   -0.05    0.00   0.00  <- EXIF?
multi_session      21    0.90    0.71    0.62   -0.19   -0.09   0.00  <- rerank
──────────────────────────────────────────────────────────────────────
OVERALL           195    0.86    0.79    0.76   -0.07   -0.03   0.03

by nn_band:  Q2 0.79   Q3 0.87   Q4 0.91      <- Q2 is the honest number
```

Four actionable columns:

- **`gap₁`** (@50 → @12) — a large negative value means the cross-encoder is
  systematically discarding that kind of answer.
- **`gap₂`** (@12 → @final) — aggregation or the session cap is dropping it.
- **`kill`** (filter kill rate, §8.3) — the positive was excluded *before*
  ranking. A parser problem, demanding the opposite fix. Target < 3 %.
- **`by nn_band`** — recall on the crowded-neighbourhood band (Q2) is the
  realistic figure; the top band flatters the system (§3.3).

---

## 8. Ablations

Two independent axes, crossed. Same golden set throughout; all are recall runs,
so they cost nothing beyond compute.

### 8.1 Retrieval axis

| Config | Question answered |
|---|---|
| dense only | baseline semantic quality |
| sparse only | how much of this is just keyword matching |
| hybrid (RRF) | is fusion earning its complexity |
| hybrid + rerank | is the cross-encoder worth its latency |
| **Reranker model × depth, at ISO-LATENCY** | The joint decision, not two separate ones. At a ~250 ms budget: MiniLM-L-6 @ ~400 candidates, MiniLM-L-12 @ ~200, `bge-reranker-base` @ ~50, `bge-reranker-v2-m3` @ ~14. A strong reranker over few candidates may beat a weak one over many — measure, do not assume (`DESIGN-NOTES.md` §5) |
| + neighbour expansion | does small-to-big retrieval help |
| binary prefilter + **int8** rescore | fallback if int8 alone loses too much; needs no extra storage |

### 8.2 Query-alteration axis

**Query processing is the component most likely to make things worse.** An
over-eager parser that extracts the wrong date range yields recall = 0, and no
amount of retrieval tuning recovers it. Ablate each alteration independently:

| Config | What runs | Isolates |
|---|---|---|
| `raw` | query verbatim to both retrievers; **no filters at all** | the true no-alteration baseline |
| `fuzzy_only` | deterministic contact fuzzy-match + regex date extraction; no LLM | how much the LLM parser actually adds |
| `filter_only` | LLM filters applied, but the **full raw query** still fed to dense and sparse | value of filtering, independent of rewriting |
| `residual` (default) | filters applied **and** filter terms stripped before embedding | does removing "alice"/"march" from the embedded text help or hurt |
| `+expansion` | HyDE or multi-query on top of `residual` | whether expansion is worth adding at all |

Two things this measures that nothing else does:

- **Does filtering help *recall*, not just latency?** Pre-filtering shrinks the
  candidate pool, so the positive competes with fewer distractors. `filter_only`
  vs `raw` can legitimately come out **positive** — worth knowing, because it
  changes how aggressively filters should fire.
- **Is the semantic residual a good idea?** `residual` vs `filter_only` is the
  only way to find out. Stripping terms is an unexamined assumption in the
  current design.

**Report this axis per category.** Query alteration barely affects `topical` and
dominates `person_filtered` and `time_filtered` — the aggregate will hide both.

### 8.3 Filter kill rate — a required diagnostic

For every configuration that applies filters, count:

```
filter_kill_rate = (# queries where the positive was excluded by the pre-filter)
                 / (# queries)
```

This is a **counter, not a new metric family**, and it separates two failure
modes that recall@50 conflates:

| Symptom | Cause |
|---|---|
| positive excluded by the filter | **parser error** — wrong dates, wrong person, over-firing |
| positive in the candidate pool but ranked > 50 | **retrieval error** |

They demand opposite fixes: loosen the parser vs improve retrieval. Without this
counter you cannot tell them apart.

**Target: < 3 %.** Above that, make the parser more conservative — a missed
filter costs a little precision, a wrong filter costs the whole answer. This is
the quantitative justification for the fail-open design in `PLAN.md` Phase 4.

Log the parsed filter alongside every result so kills can be inspected directly.

### 8.4 Diagnostics

- **Sparse-only within ~3 % of hybrid** → embeddings are contributing nothing.
  Suspect a missing BGE query prefix (`"Represent this sentence for searching
  relevant passages: "`, queries only), broken chunking, or leakage the §4.2
  filter failed to catch.
- **recall@12 far below recall@50** → the reranker is receiving the wrong text.
  It must score the decompressed `chunks.body_semantic`, not a truncated snippet.
- **`residual` below `filter_only`** → stripping filter terms is removing useful
  semantic signal. Stop stripping.
- **`fuzzy_only` ≈ `filter_only`** → the LLM parser is not earning its latency
  or its failure modes. Drop it and keep the deterministic path.
- **Overall recall stubbornly low with no other explanation** → quantisation is
  the last suspect, not the first. Re-embed a sample at fp32 and compare before
  believing it; if confirmed, switch to binary prefilter + int8 rescore
  (`DESIGN-NOTES.md` §8). fp32 is never stored in production.

---

## 9. Parameter sweeps

| Parameter | Values | Notes |
|---|---|---|
| Session gap | 15 / 30 / **45** / 90 min | requires re-index |
| Chunk overlap | 0 / **2** / 4 messages | requires re-index |
| Chunk target size | 200 / **400** / 800 tokens | requires re-index |
| RRF `k` | 20 / **60** / 100 | free |
| Candidate depth | 25 / **50** / 100 per retriever | free |
| Split place names into their own FTS column for weighting | **no** / yes | FTS rebuild only (~minutes) |
| `format_lexical` role labels | **strip** / keep | FTS rebuild only (~minutes), no re-embed |
| `format_semantic` URL handling | **domain-only** / verbatim / stripped | requires re-embed |
| `format_semantic` truncation | **middle-drop** / tail-drop | requires re-embed |
| Neighbour expansion | 0 / **±2** / ±4 | free |
| Fuzzy contact match threshold | 80 / **88** / 95 | free; trades filter kill rate against recall |
| Date-filter padding | 0 / **±3** / ±7 days | free; the cheapest defence against parser date errors |

Bold = current default in `PLAN.md`.

Re-index sweeps are expensive: run them on a **10 % corpus sample** first, then
validate the winner on the full index.

⚠️ Chunking sweeps change chunk boundaries — which is exactly why labels are
message ids. They do **not** invalidate the golden set.

---

## 10. Hygiene

1. **Dev/test split**, ~70/30. Tune on dev; report from test; touch test rarely.
2. **Fixed seeds**; pin `embed_version` in every result row.
3. **Version the golden set** — adding entries changes scores, so historical
   comparisons need the version recorded.
4. **Persist runs** in `eval/results.db`: one row per
   `(run_id, config, category, metric, value)`.
5. **One command, under ten minutes**, or it will not be run.
6. Report harvested and Tier 0 scores separately. Harvested data measures
   *relative* change reliably; absolute quality less so.

---

## 11. Harness

```
eval/
  harvest.py            # §3 — post-build, resumable: --stage signals|idf|nn
  generate.py           # §4.1 — question generation
  filter.py             # §4.2 — leakage + ambiguity filters
  review.py             # §4.3 — CLI accept/edit/reject
  golden.jsonl
  run_eval.py           # --config <name> --split dev|test
  ablate.py             # §8 matrix
  sweep.py              # §9 grid
  controls.py           # §5.1 — unit / positive / negative / oracle / mutation
  stats.py              # §6 — Wilson, McNemar, paired bootstrap
  fixtures/             # 20-chunk corpus with hand-written positives
  results.db
  report.py
```

---

## 12. Build order

Harvesting is a **post-build job, not an indexing concern** (§3). Nothing here
touches the indexing hot path.

### Staged, with gates

Each stage has an exit condition. Do not proceed past a failed gate — the
downstream work will be built on an eval you cannot trust.

| Stage | Work | **Gate before proceeding** |
|---|---|---|
| **A** | `run_eval.py` + `controls.py` against `fixtures/` (20-chunk corpus, hand-written positives) | Unit tests pass **and** the mutation test moves the number. If sabotaging retrieval does not drop recall, the harness is measuring nothing |
| **B** | **30-entry pilot**: harvest → generate → filter → review | Blind re-review of the same 30 a day later: **≥ 80 % self-agreement**. Below that, the accept/reject criteria are too vague — write them down explicitly and redo |
| **C** | Full run: ~300 generated → filtered → reviewed | ~200 kept; every category ≥ 15 entries |
| **D** | Validation battery (§5): positive control, negative control, oracle, coverage audit | Oracle = 1.00; ≥ 30 distinct chats; no single chat > 15 % of entries |
| **E** | Freeze, dev/test split, baseline run | Baseline recorded with Wilson intervals and `embed_version` pinned |

**Stage B is the one people skip.** Reviewing 30 entries and then re-reviewing
them blind is the only cheap way to learn whether your accept/reject criteria are
reproducible. If you disagree with yourself, the other 270 entries are noise —
and you will not find out until the numbers stop making sense months later.

**Stage A before Stage B, always.** Building a golden set with an unvalidated
scorer means you cannot distinguish "bad question" from "bad scoring" during
review, and you will silently discard good entries.

### Relative to `PLAN.md` phases

⚠️ **There is a circular dependency to break.** Stage B needs the ambiguity
filter and review UI, which need a **working retriever** — but tuning that
retriever needs the golden set. Resolve it by splitting Phase 4:

| Order | `PLAN.md` phase | Eval deliverable |
|---|---|---|
| 1 | Phase 3 (build) | inline signals written during embedding (§3) |
| 2 | **Phase 4a** — baseline retriever (pre-filter → dense+sparse → RRF, no reranker) | — |
| 3 | Phase 8 | **Stage A** — harness, fixtures, controls, mutation test |
| 4 | Phase 3.5 | **Stages B–D** — harvest, generate, filter, review |
| 5 | Phase 3.5 → 4b | **Stage E** — freeze, split, baseline |
| 6 | **Phase 4b** — rerank, hydration, aggregation | ablations and sweeps against real numbers |
| — | ongoing | run on every change; re-run the mutation test whenever the harness changes |

The baseline retriever at step 2 need not be good — only *working*. Its job is
to power the ambiguity filter and show candidate hits during review.

---

## 13. Success criteria

Provisional; revise once Phase 0 supplies real corpus numbers.

| Metric | Minimum | Good |
|---|---|---|
| Overall recall@50 | 0.80 | 0.92 |
| Overall recall@12 | 0.70 | 0.85 |
| **Overall recall@final** | 0.68 | 0.83 |
| **recall@50 on the Q2 `nn_band`** (crowded neighbourhood) | 0.70 | 0.85 |
| Worst-category recall@50 | 0.65 | 0.85 |
| `@50 → @12` gap | > −0.15 | > −0.05 |
| `@12 → @final` gap | > −0.08 | > −0.03 |
| **Filter kill rate** | < 0.05 | < 0.02 |
| Best alteration config vs `raw`, paired CI lower bound | ≥ −0.01 | ≥ +0.03 |
| **p95 tool latency** | < 1.5 s | < 0.8 s |

Two guards rather than targets:

- **The `raw` row.** If no query-alteration configuration beats the no-alteration
  baseline, the parsing layer is pure cost and should be removed.
- **The Q2 `nn_band` row.** This is the honest difficulty band. A high overall
  number driven by the isolated-chunk band means the eval set is flattering the
  system (§3.3).
- **The latency row is a quality metric here, not an ops metric.** An MCP tool
  call blocks the agent's turn, so a slow tool is a bad tool regardless of its
  recall.

**Gate: no category below 0.65 recall@50.** A uniformly mediocre system is
usable; one with a silently broken category is not, because the user cannot tell
which answers to trust.

**Gate: no category below 0.65 recall@50.** A uniformly mediocre system is
usable; one with a silently broken category is not, because the user cannot tell
which answers to trust.

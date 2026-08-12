# Design Notes & Background

> Companion to `PLAN.md`. This document captures the reasoning behind every
> decision in the plan, plus background on the underlying technologies, so that
> an implementer arriving without prior context can evaluate and improve the
> design rather than merely execute it.
>
> **Read §9 (Rejected designs) before proposing changes.** Most obvious
> improvements were already considered and eliminated for reasons recorded there.
>
> See also **`EVALUATION.md`** — the measurement strategy that settles the
> parameters left open here.

---

## Contents

1. [RAG pipeline fundamentals](#1-rag-pipeline-fundamentals)
2. [Document parsing landscape](#2-document-parsing-landscape)
3. [Frameworks: LangChain and LlamaIndex](#3-frameworks-langchain-and-llamaindex)
4. [Open-weight models and how they run](#4-open-weight-models-and-how-they-run)
5. [Embeddings, bi-encoders and cross-encoders](#5-embeddings-bi-encoders-and-cross-encoders)
6. [Sparse retrieval: BM25 and FTS5](#6-sparse-retrieval-bm25-and-fts5)
7. [Hybrid retrieval and Reciprocal Rank Fusion](#7-hybrid-retrieval-and-reciprocal-rank-fusion)
8. [Storage: SQLite internals, FTS5, sqlite-vec](#8-storage-sqlite-internals-fts5-sqlite-vec)
9. [Rejected designs and why](#9-rejected-designs-and-why)
10. [How iMessage stores data](#10-how-imessage-stores-data)
11. [Hardware and capacity maths](#11-hardware-and-capacity-maths)
12. [Design principles distilled](#12-design-principles-distilled)

---

## 1. RAG pipeline fundamentals

Retrieval-Augmented Generation: instead of relying on a model's parametric
memory, retrieve relevant source text at query time and put it in the context
window. The pipeline is:

```
parse → chunk → embed → store → retrieve → rerank → generate
```

Quality is dominated by the **early** stages. A frontier model cannot answer
from context it was never given, so retrieval recall caps the entire system.
In practice the ranked order of leverage is:

1. **Chunking strategy** — largest single lever, almost always underrated
2. **Hybrid retrieval** (dense + sparse) — biggest cheap win after chunking
3. **Reranking** — best quality-per-engineering-hour of anything on this list
4. Embedding model choice — matters less than people assume
5. Generator model choice — matters least for factual grounded QA

**Evaluate retrieval separately from generation.** Recall@k and MRR against a
golden set tell you whether the right content was found; answer grading tells
you what the model did with it. Conflating them makes every tuning decision
unattributable.

### Context position effects

LLMs attend most reliably to the **beginning and end** of their context, and
least reliably to the middle ("lost in the middle"). Consequently, stuffing 50
mediocre chunks into context produces *worse* answers than 5 good ones. This is
the real reason reranking matters: not just ordering, but shrinking the set to
something the model reads properly.

---

## 2. Document parsing landscape

Not used by this project (iMessage is structured SQLite, not documents), but
recorded because the conversation began here and it informs any future extension
to PDFs or attachments.

| Tool | License | Strengths | Weaknesses |
|---|---|---|---|
| **Docling** (IBM) | MIT | Best structure/table fidelity; PDF/DOCX/PPTX/XLSX/HTML → Markdown/JSON; native LangChain & LlamaIndex loaders | Heavy install (~500 MB+) |
| **MinerU** (OpenDataLab) | AGPL | Strong layout/reading order, formulas; good for air-gapped batch | AGPL; fewer integrations |
| **Marker** (datalab.to) | Restricted above a revenue threshold | Fastest PDF → Markdown | Licence; GPU recommended |
| **Unstructured** | Apache-2.0 core | Widest format coverage (25+); semantic JSON elements | Accuracy trails Docling |
| **PyMuPDF4LLM** | AGPL | Very fast, no ML models | Layout-naive |
| **pdfplumber** | MIT | Simple text/table pulls | No layout model |

**Rule of thumb:** Docling or Unstructured for permissively-licensed production
work; PyMuPDF4LLM when you need speed and don't need layout understanding.

---

## 3. Frameworks: LangChain and LlamaIndex

Both are "glue" layers between documents, vector stores and models.

- **LangChain** — general LLM application framework: chains, agents, tool
  calling, memory. Huge integration catalogue. Companion projects **LangGraph**
  (stateful agent orchestration) and **LangSmith** (tracing/evals, commercial).
  Best when the hard part is orchestration.
- **LlamaIndex** — originally "GPT Index", focused on ingestion, indexing and
  retrieval. Strongest on the document side: connectors, node/chunk
  abstractions, query engines, advanced retrieval strategies. Also ships
  **LlamaParse** (commercial hosted parser).

**Decision for this project: use neither.** The pipeline is ~400 lines of
explicit code against SQLite. A framework would add abstraction layers,
dependency weight and version churn while hiding exactly the parts we most need
to control (pre-filtering, RRF, rowid alignment). Frameworks earn their keep
when you need many interchangeable backends; we have one of each.

---

## 4. Open-weight models and how they run

### "Open source" usually means open *weights*

Weights are downloadable; training data and code generally are not, and licences
vary widely. Check licence terms before commercial use.

| Model | Architecture | Licence |
|---|---|---|
| Kimi K3 (Moonshot) | ~2.8 T total / ~100 B active MoE | Custom, commercial review |
| Qwen 3.x (Alibaba) | 4 B → 400 B+, dense and MoE | Apache 2.0 (mostly) |
| DeepSeek V4 | 1.6 T total / ~50 B active MoE | MIT |
| GLM-5.x (Zhipu) | ~750 B total / ~40 B active | MIT |
| Llama 4 (Meta) | 400 B / 17 B active | Meta custom; EU restrictions |
| gpt-oss, Mistral, Gemma | 3 B–120 B | Permissive |

### Mixture of Experts (MoE)

A dense model runs **every** parameter for every token. An MoE model splits each
layer into many "expert" sub-networks and a small router selects a few (typically
2–8) per token.

- **Total parameters** → memory/disk required (all experts must be loaded)
- **Active parameters** → compute cost and roughly the capability per forward pass

So "1.6 T total / 50 B active" means cluster-scale memory at 50 B-model speed.
The trade is **cheap compute, expensive memory**. Downsides: harder to fine-tune,
routing instability, trickier quantisation. Small MoEs (e.g. Qwen3-30B-A3B) can
be *faster* than a dense 14 B while being larger.

### Models ≤ 14 B (relevant to a 16 GB machine)

- **General:** Qwen3 4B/8B/14B, Qwen3-30B-A3B (MoE, 3 B active), Gemma 3 4B/12B,
  Llama 3.x 1B/3B/8B, Mistral Small / Ministral 8B, Phi-4 14B, gpt-oss-20B, SmolLM3 3B
- **Coding:** Qwen3-Coder small variants, Codestral, DeepSeek-Coder-V2-Lite
- **Embeddings:** Qwen3-Embedding 0.6B/4B/8B, BGE-M3, bge-small-en-v1.5,
  Jina v3, E5, nomic-embed-text
- **Rerankers:** ms-marco-MiniLM-L-6/L-12, bge-reranker-base, bge-reranker-v2-m3, mxbai-rerank, Qwen3-Reranker — see §5 "Model selection"

### Runtimes

| Runtime | Use case |
|---|---|
| **Ollama** | Single user, trivial setup, OpenAI-compatible localhost API. Wraps llama.cpp |
| **LM Studio** | GUI; strong on Apple Silicon via MLX |
| **llama.cpp** | The engine under both; GGUF, runs on CPU/CUDA/ROCm/Metal |
| **vLLM** | Production multi-user serving; PagedAttention, continuous batching; GPU required |
| **SGLang** | Similar niche; RadixAttention, better for structured output and repeated prefixes |

vLLM/SGLang deliver 10–20× the throughput of Ollama under concurrency, and are
irrelevant for a single-user local tool.

### MLX vs MPS vs PyTorch

- **MPS** — Metal Performance Shaders; PyTorch's Apple GPU backend
  (`torch.device("mps")`). Works, but it is a port: some ops silently fall back
  to CPU and unified memory is poorly exploited.
- **MLX** — Apple's own array framework, NumPy/PyTorch-like API, designed for
  Apple Silicon. **Unified memory** (no `.to(device)` copies) and lazy evaluation
  enabling op fusion. Typically **2–3× faster than MPS** for inference.

Fewer models and a smaller ecosystem than PyTorch, but for running a known
embedding model it is the better path. **Decision: MLX primary, PyTorch+MPS fallback.**

---

## 5. Embeddings, bi-encoders and cross-encoders

### Why open models are used for embedding even when generation is hosted

1. **Volume asymmetry.** You embed every chunk of the corpus (millions of calls,
   plus every update). You generate only when a user asks. Per-token API cost
   bites hardest at the embedding stage.
2. **Small models suffice.** Embedding is a representation task, not a reasoning
   task. A 0.6 B embedder matches commercial APIs; generation quality still
   scales with size.
3. **Lock-in.** Vectors are tied to the model that produced them. If a hosted
   embedding model is deprecated or silently updated, you re-embed everything.
   Self-hosted weights are frozen permanently.
4. **Latency.** Local embedding is a GPU matmul with no network hop, and no rate
   limits on a bulk backfill.
5. **Data residency.** The entire corpus passes through the embedder.

### Bi-encoder (the embedding model)

```
query    → transformer → pool → [vector]  ⟍
                                           cosine similarity
document → transformer → pool → [vector]  ⟋
```

The document never sees the query — which is the point. Encode the corpus once,
offline; search is then pure vector arithmetic. The `pool` step (mean over
tokens, or the `[CLS]` state) is a **lossy compression**: a whole passage crushed
into 384–1024 numbers *before* you know what will be asked of it.

### Cross-encoder (the reranker)

```
[CLS] query [SEP] document [SEP] → transformer → [CLS] → Linear(d→1) → score
```

Both texts enter as **one sequence**, so at every layer query tokens attend
directly to document tokens. The output is a **scalar**, not a vector — there is
nothing to store or index.

This is why cross-encoders handle things similarity cannot:

- **Negation** — "does X support SAML" vs a document saying X does *not*
  support SAML produce near-identical embeddings; a cross-encoder distinguishes them.
- **Exact-token grounding** — `ERR_4021` in the query attends to the *same token*
  in the document, rather than two blurry nearby points.
- **Query-specific salience** — one sentence inside a long passage can carry the
  score, instead of being averaged into oblivion by pooling.

**Why it can't be precomputed:** the document's representation depends on the
query, so every internal activation changes when the query changes. Cost is
O(candidates) forward passes per query — hence the funnel.

### How does it "know" to compare the two halves?

Three mechanisms, in ascending order of importance:

1. **`[SEP]`** — a real learned vocabulary token acting as a boundary landmark.
2. **Segment (token-type) embeddings** — **architecture-dependent, and absent
   from the model this project uses.** In original BERT each token gets a third
   vector added alongside token and position embeddings: `segment_A` before
   `[SEP]`, `segment_B` after, so the same word in query and document enters the
   network as a *different vector*.
   ```
   input = token_emb + position_emb + segment_emb
   ```
   ⚠️ RoBERTa-family models dropped meaningful token-type embeddings; there,
   separation comes from `</s>` separator tokens, positions, attention and
   training. BERT-lineage models (including the chosen `ms-marco-MiniLM-L-12-v2`)
   *do* use them. Architecture-dependent — check the model, do not assume.
3. **Training** — the actual answer, and the one that holds across
   architectures. Nothing in the architecture *forces* comparison; a randomly
   initialised model outputs noise. Millions of `(query, positive) → 1` /
   `(query, negative) → 0` examples make cross-segment attention the only way to
   reduce loss, so gradient descent discovers it. Trained models demonstrably
   contain heads attending almost exclusively from the document span to the
   query span.

Rerankers are typically trained with **hard negatives** mined from a bi-encoder's
top results — they are explicitly taught to separate the cases embeddings confuse.
That is literally their job.

### The funnel

```
1 M chunks
  → hybrid retrieval (cheap, optimise RECALL)      → top 50
  → cross-encoder rerank (expensive, PRECISION)    → top 12
  → LLM context
```

50 forward passes ≈ 50–200 ms on GPU. Retrieval must not lose the answer;
reranking must put it first.

### Encoders vs decoders — an important architectural aside

| Type | Attention | Examples | Good at |
|---|---|---|---|
| Encoder-only | bidirectional | BERT, DeBERTa, BGE | understanding: classify, embed, rerank |
| Decoder-only | causal | GPT, Llama, Qwen | generating |
| Encoder-decoder | both | T5, original Transformer | seq2seq (out of fashion) |

Modern LLMs are **decoder-only** — there is no encoder inside them. A decoder
block *is* an encoder block with a triangular mask; the mask is most of the
difference. This matters for reranking: in a causal model the query tokens
cannot attend forward to the document, which is why LLM-based rerankers must be
structured as `Document: … Query: … Is this relevant?` and read a yes/no token
probability. A 300 M bidirectional encoder can match a 7 B decoder at this task
— the right inductive bias beating raw scale.

(Blurring the line: LLM2Vec and several 2024-26 embedding models, including
Qwen3-Embedding, convert decoders into bidirectional encoders by removing the
mask and continuing training.)

### Model selection — the candidate field

Two models to choose, and the right answer depends heavily on **this
architecture's** constraints rather than on leaderboard position: brute-force
vector scan makes embedding *dimension* expensive, and an MCP tool call blocking
the agent's turn makes reranker *compute* expensive.

#### Embedders

| Model | Params | Dim | Ctx | MTEB retrieval | Scan cost at 1.25 M chunks |
|---|---|---|---|---|---|
| **`bge-small-en-v1.5`** ✅ | 22 M | **384** | 512 | **60.5** | **480 MB (1.0×)** |
| `gte-small` | 22 M | 384 | 512 | 59.8 | 480 MB (1.0×) |
| `snowflake-arctic-embed-s` | 36 M | 384 | 4096 | 59.7 | 480 MB (1.0×) |
| `nomic-embed-text-v2` | 137 M | 768 | 8192 | 62.9 | 960 MB (2.0×) |
| `bge-base-en-v1.5` | 109 M | 768 | 512 | ~63 | 960 MB (2.0×) |
| `Qwen3-Embedding-0.6B` | 600 M | 1024 | 32k | ~65 | 1.28 GB (2.7×) |
| `bge-m3` | 568 M | 1024 | 8k | ~67 (multilingual) | 1.28 GB (2.7×) |

**Decision: `bge-small-en-v1.5`.** It is the strongest of the genuinely small
tier, and — decisively for this design — **dimension multiplies the scan**.
Every query reads the whole int8 array, so 768-d costs 2× the bandwidth for
about +2.4 MTEB, and 1024-d costs 2.7×. With an ANN index that trade might be
worth taking; with a brute-force scan it directly inflates the latency budget
that a blocking tool call cannot afford.

Secondary points: 512-token context is sufficient (chunks target ~400); the
multilingual models (`bge-m3`, Qwen3) spend most of their capacity on languages
this corpus does not contain; and `snowflake-arctic-embed-s`'s 4096-token
context is irrelevant at our chunk size.

**Worth investigating if quality disappoints:** Matryoshka-trained models
(e.g. `nomic-embed-text-v1.5`) permit truncating dimensions with graceful
degradation — potentially 768-d quality training at 384-d storage. That is the
only route to better vectors without paying the bandwidth.

#### Rerankers

Reranking is the only compute-bound stage. FLOPs ≈ 2 × encoder-params × tokens,
and the workload is 50 pairs × 512 tokens = 25 600 tokens per query:

| Model | Total | Encoder | TFLOPs | Est. batched | BEIR nDCG@10 | Licence |
|---|---|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 22 M | 11 M | 0.5 | **~30 ms** | ~60–62 | Apache-2.0 |
| **`ms-marco-MiniLM-L-12-v2`** ✅ | 33 M | 21 M | 1.1 | **~60 ms** | ~62–64 | Apache-2.0 |
| `mxbai-rerank-base-v1` | 184 M | 85 M | 4.3 | ~240 ms | ~65 | Apache-2.0 |
| `bge-reranker-base` | 277 M | 85 M | 4.3 | ~240 ms | ~66 | MIT |
| `jina-reranker-v2` | 300 M | ~180 M | ~9 | ~500 ms | ~69 | **CC-BY-NC** |
| `bge-reranker-v2-m3` | 568 M | 302 M | 15.5 | **~860 ms** | **~71.5** | MIT |
| `Qwen3-Reranker-0.6B` | 600 M | 352 M | 18.0 | ~1000 ms | ~71.4 | Apache-2.0 |

(Estimates assume ~18 TFLOPS fp16 on M5 and perfect batching — optimistic, so
treat them as a *lower* bound. Phase 0 measures the real numbers.)

⚠️ **The quality spread is real and larger than convenient.** An earlier
revision of this document claimed MiniLM would be "within noise" of
`bge-reranker-base` on English. That was wrong: the published gap is ~10 nDCG
points between MiniLM-L-6 and `bge-reranker-v2-m3`. But the compute gap is ~30×,
and at 50 candidates the strong models consume the **entire** sub-second tool
budget on one stage.

**Decision: `ms-marco-MiniLM-L-12-v2` as the default** — 2× the compute of L-6
for a couple of nDCG points, still only ~60 ms, English-only by design (30 k
BERT vocabulary rather than 250 k multilingual).

**But treat model-versus-depth as one joint decision.** Rerank cost is
`model_compute × candidates`, so a fixed latency budget buys very different
shapes:

| At ~250 ms | candidates rerankable |
|---|---|
| MiniLM-L-6 | ~400 |
| MiniLM-L-12 | ~200 |
| bge-reranker-base | ~50 |
| bge-reranker-v2-m3 | ~14 |

A strong reranker over 14 candidates may well beat a weak one over 400 — or may
not, since recall@50 caps what reranking can recover. **Run this as an
iso-latency ablation** (`EVALUATION.md` §8.1) rather than picking on
leaderboard position.

Two cautions on the published numbers: BEIR measures *document* retrieval over
diverse corpora, whereas this task is conversational message windows in a single
personal domain — the ranking may not transfer. And `jina-reranker-v2` is
CC-BY-NC, which is fine for personal use but worth knowing before it spreads.

#### What would change these choices

- **If `vec0` cannot constrain KNN** (Phase 0 spike) and scanning moves to an
  mmapped MLX array, dimension gets cheaper — a 768-d embedder becomes more
  defensible.
- **If the corpus stops being English-only**, both decisions invert toward the
  multilingual models.
- **If retrieval latency proves generous** after measurement, spend it on the
  reranker before anything else; it is the stage with the steepest
  quality-per-millisecond curve.

Many embedding models require a **prefix on queries but not documents**:

- E5: `"query: "` / `"passage: "`
- BGE: `"Represent this sentence for searching relevant passages: "` on queries only
- Qwen3-Embedding: an instruction string

Getting this wrong produces no error and degrades results. For **BGE v1.5**
specifically the model card reports only a *slight* degradation without the
instruction — so treat omission as a measurable quality regression rather than a
categorical failure. Encode it once, centrally, make it configurable, and ablate
it (`EVALUATION.md` §8).

---

## 6. Sparse retrieval: BM25 and FTS5

### BM25

"Best Match 25" — a probabilistic ranking function from the **Okapi** system at
City University London (Robertson, Spärck Jones, et al., ~1994), evaluated at
TREC. "BM" is the *Best Match* family; the number is an iteration index. BM11
used full document-length normalisation, BM15 none; **BM25 interpolates between
them via `b`** (`b=1` → BM11, `b=0` → BM15, `b≈0.75` between). The name is lab
bookkeeping that survived thirty years because nothing simpler works better.

```
score(D,Q) = Σ  IDF(qᵢ) ·        f(qᵢ,D) · (k₁ + 1)
             qᵢ∈Q          ─────────────────────────────────
                           f(qᵢ,D) + k₁ · (1 - b + b·|D|/avgdl)

IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
```

Three intuitions:

1. **Rare terms matter more** (IDF). "the" contributes ~0; `ERR_4021` dominates.
2. **Diminishing returns on repetition** (saturation, `k₁≈1.2`) — the key
   improvement over naive TF-IDF. 20 occurrences are not 20× more relevant.
3. **Long documents are penalised** (`b≈0.75`) — they match more terms by chance.

No training, no GPU, fully explainable, works on day one. It is a genuinely hard
baseline; many "neural search beats keyword search" claims fail against a
properly tuned BM25. Its blind spot is total absence of semantics.

Variants: BM25F (weighted fields), BM25+ (long-document edge case).

### Why FTS5 supports incremental inserts despite IDF being corpus-global

**Because BM25 is never precomputed.** FTS5 stores raw statistics and computes
the score at query time.

On disk:
```
"lisbon" → [(docid=17, tf=2, positions=[4,19]), (docid=433, tf=1, …), …]
```
plus shadow tables: `%_docsize` (tokens per document) and an *averages* record
(total documents, total tokens per column).

| BM25 needs | Source | Cost |
|---|---|---|
| `N` | averages record | O(1) |
| `n_t` | **length of t's postings list** | free — you scan it anyway |
| `f(t,D)` | in the postings entry | free |
| `\|D\|` | `%_docsize` | O(1) |
| `avgdl` | totals ÷ N | O(1) |

The key realisation: **IDF is only needed for terms present in the query.** You
never touch the rest of the vocabulary. So it is O(1) per query term, not O(N)
over the corpus. Inserting a document appends postings and bumps counters;
nothing existing is rewritten.

This is **late binding**: store sufficient statistics, apply the scoring function
at read time. It is also why ranking can be changed without reindexing.

FTS5 is LSM-tree-like — writes create new **segments**, and `n_t` is summed
across them at query time (so it stays exact). Deletes are tombstones netted out
during merge. `INSERT INTO t(t, rank) VALUES('merge', 500)` compacts
incrementally — note the parameter goes in the hidden `rank` column.

**Gotchas:**

- **`bm25()` returns negative values.** FTS5 negates so that `ORDER BY bm25(t)`
  *ascending* yields best matches. Sorting `DESC` silently returns the worst
  results. Use rank order.
- `k₁` and `b` are hardcoded (1.2 / 0.75). **Column weights** are tunable:
  `bm25(chunks_fts, 1.0, 2.0)` — though this index has a single column, so there is nothing to weight.
- ⚠️ The IDF formula above is **generic BM25**. FTS5's implementation is a
  Robertson-style variant that floors non-positive IDF contributions, so scores
  are not identical to a textbook implementation. Treat the formula as
  background for *why* rare terms dominate, not as FTS5's exact arithmetic.
- **`'rebuild'` requires a content table.** It re-reads text from the source, so
  it is unavailable on `content=''`. Contentless tables must be populated by
  explicit inserts.
- **Merge syntax puts the parameter in the hidden `rank` column:**
  `INSERT INTO chunks_fts(chunks_fts, rank) VALUES('merge', 500);`

### What is not incremental

**ANN index structure.** IVF centroids drift as data is added; HNSW graphs
degrade under churn. Both eventually need rebuilds. Brute-force scanning avoids
this entirely — appending a vector is appending a row. Embeddings themselves are
corpus-statistic-free, so they are trivially incremental.

---

## 7. Hybrid retrieval and Reciprocal Rank Fusion

Two retrieval families with complementary failure modes:

**Sparse (BM25):** vocabulary-sized vector, mostly zeros; scores exact term
overlap weighted by rarity.
- ✅ Exact strings: `ERR_4021`, `Section 7.3(b)`, surnames, part numbers
- ❌ Blind to synonyms — "car" and "automobile" are unrelated tokens

**Dense (embeddings):** ~384–1024 continuous dimensions where meaning is
geometric proximity.
- ✅ Paraphrase, synonyms, conceptual similarity
- ❌ Rare tokens are smeared — `ERR_4021` and `ERR_4022` land almost on top of
  each other because neither was meaningfully trained

That second failure is decisive for personal/enterprise corpora full of names
and identifiers.

### Why RRF rather than score blending

BM25 is unbounded (0 to ~30, corpus-dependent); cosine is 0–1. Normalising is
fragile because BM25's range shifts per query. RRF discards scores and uses only
**rank**:

```
RRF(d) = Σ  1 / (k + rank_i(d))          k = 60 by convention
        i
```

Worked example — *"why does ERR_4021 happen during batch upload"*:

| Doc | BM25 rank | Dense rank | RRF |
|---|---|---|---|
| A — defines ERR_4021 | 1 | 40 | 1/61 + 1/100 = **0.0264** |
| B — batch upload prose | 25 | 1 | 1/85 + 1/61 = **0.0282** |
| C — changelog mention | 2 | 90 | 1/62 + 1/150 = 0.0228 |

Both A and B surface; neither retriever alone returns both. `k=60` damps the top
ranks so a single retriever's #1 cannot dominate without corroboration.

RRF is parameter-free, scale-invariant, and needs no retuning as the corpus
grows. Weighted blending can beat it *with per-domain tuning* — a budget most
projects never spend.

**Implementation note:** both retrievers must operate on the **same pre-filtered
candidate set**, or the ranks being fused are not comparable.

---

## 8. Storage: SQLite internals, FTS5, sqlite-vec

### SQLite vs Postgres

| | SQLite | Postgres |
|---|---|---|
| Model | Library in your process | Client/server daemon |
| Storage | One file | Managed data directory |
| Concurrency | One writer, many readers | Many writers, MVCC |
| Setup | `pip install` | Install, configure, run a service |
| Vector | `sqlite-vec` | `pgvector` (HNSW, IVFFlat) |
| Full-text | FTS5, BM25 built in | `tsvector`, `ts_rank_cd` |
| Ceiling | ~10 M vectors, single machine | Effectively unbounded, replication |

**Decision: SQLite.** Single user, read-heavy, one machine — and `chat.db` *is*
SQLite, so `ATTACH` gives cross-database joins in a single query. A Postgres
daemon would consume part of a 16 GB budget for no benefit. Postgres earns its
keep with concurrent writers, network access, multiple applications, or genuine
HNSW-at-scale requirements.

### `CREATE INDEX` — what it does

A table is a **B+tree** keyed on rowid:
- **Interior pages** hold only rowid separator keys and child pointers
- **Leaf pages** hold the **complete row payload**, in rowid order

It is ordered, but not a flat array — a flat array would need O(n) memmove per
insert and could not be paged efficiently. High fanout (hundreds of keys per
4 KB page) makes 1 M rows ~3 levels deep.

`CREATE INDEX` builds a *second* B-tree sorted by the named columns, whose
**leaves contain `(indexed columns…, rowid)`** rather than the row — so a
non-covering index costs a second seek into the table tree.

Column order matters:
```sql
CREATE INDEX idx_chunks_chat ON chunks(chat_id, start_ts);
```
serves `WHERE chat_id = ?` and `WHERE chat_id = ? AND start_ts BETWEEN ? AND ?`,
but **not** `WHERE start_ts > ?` alone — like a phone book sorted by surname
then forename, useless for finding all the Johns.

Costs: disk space, plus index maintenance on every write. Irrelevant for a
write-once/read-forever index. Verify with `EXPLAIN QUERY PLAN` — `SCAN` is bad,
`SEARCH … USING INDEX` is good.

**`WITHOUT ROWID`** tables are a true B-tree keyed on the declared primary key,
storing rows in interior nodes too. Used for `chunk_message`: the row *is* the
key, so there is nothing to seek for.

### `CREATE VIRTUAL TABLE` — what it does

A virtual table looks like a table to SQL but its reads and writes are handled by
**custom C code** (a "module"), letting you use `SELECT`/`JOIN`/`WHERE` against
data structures SQLite has no native support for.

- `USING fts5(…)` → inverted index, `MATCH`, `bm25()`
- `USING vec0(…)` → packed vector storage, distance functions, KNN

### FTS5 content modes, and why we store no text at all

FTS5 has three content modes:

| Mode | Stores | Notes |
|---|---|---|
| Default | Inverted index **+ a full copy of the text** in `%_content` | Simple, self-contained |
| `content='table'` (external) | Inverted index only; reads originals from a named table | Same total bytes as default if you also keep the column |
| **`content=''` (contentless)** | **Inverted index only. No text anywhere.** | Chosen here |

**The external-content trap** (worth knowing even though we no longer use it):
SQLite has no internal change notification between tables — FTS5 literally cannot
observe a write to the content table. And to *delete* a term from an inverted
index, FTS5 must know which postings to subtract, which requires the row's **old
text**. So the API makes you supply it:

```sql
-- delete FIRST, supplying the OLD values
INSERT INTO chunks_fts(chunks_fts, rowid, body)
VALUES('delete', 42, :old_body);
```

Update the content table first and FTS5 subtracts the **wrong** postings,
silently corrupting the index — phantom hits and missing hits, no error.

**Why contentless is right here.** The decisive question is what the stored text
would still be used for:

| Use | Needs stored text? |
|---|---|
| BM25 matching | ❌ the inverted index is sufficient |
| Reranker input | ❌ hydrates raw messages from `chat.db` |
| LLM context, display, citations | ❌ same |
| Re-embedding after a model swap | ✅ but **regenerable** |
| Debugging "what was embedded?" | ✅ but regenerable |
| Eval regex signals | ✅ but regenerable |

The original argument ran: `body` is a **pure function** of the chunk's message
set and the formatter, so it is a *cache, not data*, and storing ~2 GB to serve
reads happening perhaps twice a year is a bad trade.

⚠️ **That argument was later overturned** — see §9, "❌ An `embed_text` column".
Reranking reads 50 chunks per query, and regeneration is not byte-stable. `body`
is now stored zstd-compressed (~400 MB). Contentless FTS5 is retained
regardless: it is orthogonal, avoids a *second* uncompressed copy in the shadow
table, and removes the corruption hazard above.

Two requirements:

1. **`contentless_delete=1`** (SQLite 3.43+). Contentless tables otherwise reject
   deletes, and two paths depend on deleting: the open tail chunk rebuilt on
   every sync, and any chunk rebuilt because its source messages were edited or
   deleted (§ "Convergence under source mutation"). The mode's real value is that
   FTS5 can delete a row **without being handed the original text** — which is
   what makes storing nothing viable.
2. **Formatter versioning.** Regeneration must be byte-identical or a "re-embed"
   would embed different text than was originally indexed. Pin
   `meta.semantic_format_version` and assert it. Self-enforcing: if the format
   changes, you are re-embedding everything regardless.

### The FTS text is not the embedded text

An important asymmetry: **what gets indexed for BM25 is deliberately different
from what gets vectorized.**

| Column | Vectorized | Content |
|---|---|---|
| `lexical` | ❌ | `format_lexical()` — no length cap, role labels stripped, URLs verbatim |
| media placeholders | ❌ | `[photo Lisbon Alfama Portugal sunset.jpg]` — place and filename inline at the attachment's position |

The governing rule: **anything expressible as a structured predicate is a
filter, never indexed text.** Filters are exact, cheap, and cannot go stale.

| Signal | Mechanism | Why not FTS |
|---|---|---|
| People | in-memory fuzzy contact match → `handle_id` filter | exact ids beat fuzzy lexical scores; keeps names out of the index entirely |
| Dates | `start_ts` range | BM25 cannot express ranges, relative dates, or ordering; `"march"` fires on every March equally and dilutes `body` scoring |
| Media type | `has_attachment` | `"photo"` would appear in ~10⁵ chunks — near-zero IDF, so BM25 weights it ≈ 0 regardless |
| Filenames | dropped | `IMG_4821.HEIC` is noise |
| **Places** | **inside the lexical body text** | **the sole exception** |

**Why places must be text.** Consider *"what did we do in lisbon"*. Two
different things should match:

- conversations **mentioning** Lisbon → covered by `body`
- photos **taken** in Lisbon with no textual mention → only EXIF knows this

A structured geo filter would be **wrong**: it would restrict to geo-tagged
chunks and exclude the textual mentions. What is required is an **OR that
contributes to ranking** — exactly what indexed text provides. Place data is also
immutable (a photo's coordinates never change), so it carries none of the
volatility that disqualified names.

Secondary benefit: the two retrievers see genuinely **complementary**
information (dense = topic; sparse = topic + geo). RRF over complementary views
is stronger than RRF over two views of identical text.

### No names anywhere in the index

An earlier revision indexed participant names in a third FTS column, to act as
the fail-open path when query parsing misses a person. That was solving the
problem at the wrong layer, and it reintroduced volatility into an otherwise
immutable index — a rename would force deleting and reinserting those chunks'
FTS rows, and since FTS5 rows are whole-row, regenerating `body` for ~100 k
chunks along with them.

**Names are handled as a filter instead, never as text:**

```python
parsed = llm_parse(query)                        # may fail or miss the person
if not parsed.people:
    hits = fuzzy_match_contacts(query_tokens)    # in-memory, rapidfuzz
    if hits:
        parsed.people = hits                     # → structured handle_id filter
```

Strictly better than BM25 name matching on four counts: it yields an exact
`handle_id` filter rather than a fuzzy lexical score; it catches nicknames and
misspellings BM25 would miss; it is deterministic and unit-testable, unlike the
LLM parse step it backstops; and it costs microseconds.

**The result is a clean invariant: no name appears anywhere in `index.db`.**
A contact or chat rename requires zero index work — there is no rebuild path to
write, test, or forget to run. Combined with contentless FTS5, the index is
fully immutable apart from the open tail chunk of each chat.

`attachment_place` is exempt from the no-storage rule and **is** stored — it holds
reverse-geocoded place names from a one-off EXIF scan over ~50 GB of files. Not
derivable without rescanning, and only ~30 MB.

### sqlite-vec and `vec0`

`sqlite-vec` is a loadable SQLite extension (single C file). `vec0` is the
virtual table module it registers:

- Typed vector columns: `float[384]`, `int8[384]`, `bit[384]`
- Vectors stored in **chunked blobs** (thousands per blob) so a scan is
  contiguous memory rather than per-row overhead
- KNN syntax: `WHERE embedding MATCH :vec AND k = 50`
- SIMD-accelerated distance functions
- Newer versions: partition keys and metadata columns for in-table filtering
  (worth benchmarking against the SQL pre-filter approach)

**Rowid alignment — the critical invariant.** `vec0` rows have rowids but there
is **no foreign key and no enforcement**. Insert without specifying a rowid and
SQLite auto-assigns sequentially; any gap or reordering between `chunks` and
`chunks_vec` inserts misaligns every join — returning real vectors mapped to the
wrong messages, with no error at all.

```python
cur.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (chunk_id, blob))
```
```sql
-- structural assertion, MUST return 0
SELECT count(*) FROM chunks c
LEFT JOIN chunks_vec v ON v.rowid = c.id
WHERE v.rowid IS NULL;
```

⚠️ **A structural check cannot detect misalignment**, only absence. If every row
exists but the mapping is shifted, this query returns 0 while every result is
wrong. Only re-embedding a sample of stored text and comparing to the stored
vector catches that. And because SQLite **reuses rowids** after deletion of the
highest-id row, `chunks.id` must be `AUTOINCREMENT` — otherwise a crash between
delete and reinsert lets a new chunk claim an id whose stale vector survived.
Full treatment in `PLAN.md` §5, "The rowid coupling".

### Vector search: brute force vs ANN

- **Brute force (flat):** compute distance to all N vectors. Exact.
- **ANN (HNSW):** navigable graph, touches ~1 % of vectors. Sub-millisecond,
  95–99 % recall, costs memory and build time.

At ~1 M × 384-d, brute force is **plausibly under ~150 ms and unmeasured** —
see the warning in §11 — and it buys two real
advantages:

1. **Exact metadata pre-filtering.** With HNSW, pre-filtering shreds graph
   connectivity and post-filtering wastes the candidate budget. With a scan you
   filter in SQL first, then scan only survivors — filtering to "Alice in March"
   scans 4 000 vectors instead of 1 M, making the filtered case *faster* than
   the unfiltered one.
2. **Trivially incremental.** No graph to degrade, no centroids to drift.

HNSW should be considered only if scan latency becomes unacceptable, and with
full awareness that it costs the filtering property.

### Quantisation

| Precision | Bytes/vector (384-d) | 1.25 M chunks | Notes |
|---|---|---|---|
| fp32 | 1536 | 1.9 GB | The naive default — **not used here** |
| **int8** | 384 | **480 MB** | ~1 % recall loss, 4× faster scan. **The only stored precision** |
| binary | 48 | 60 MB | Hamming (XOR+popcount); prefilter, then rescore against the **int8** vectors |

Because retrieval is a **bandwidth-bound scan**, precision directly determines
latency:
```
fp32:  1.25 M × 1536 B = 1.9 GB read per query
int8:  1.25 M ×  384 B = 480 MB read per query
M5 memory bandwidth ~120–150 GB/s, shared with GPU and any resident LLM
```

**Decision: int8 only, permanently. fp32 is never stored.**

The usual reason to keep fp32 around is to measure quantisation loss, or to
support a rescore path. Neither justifies 1.9 GB here:

- **fp32 is regenerable in 20–40 minutes** by re-embedding. Storing it forever
  to avoid a half-hour rerun is a bad trade — this is the same "cache, not data"
  logic that applies elsewhere, and unlike `body` it is not on any hot path.
- **The escape hatch does not need it.** If int8 alone proves weak, use a binary
  prefilter (60 MB, XOR+popcount, ~40× faster scan) and rescore the top ~500
  against the **int8** vectors already stored. Rescoring 500 candidates in fp32
  rather than int8 is a negligible accuracy difference for 1.9 GB.

If retrieval ever looks suspiciously weak, re-embed a sample and compare then.

---

## 9. Rejected designs and why

Each of these was proposed during design and eliminated. Do not reintroduce them
without addressing the stated reason.

### ❌ A `person` table mirroring contacts

**Why proposed:** unify multiple handles (phone, email, SMS vs iMessage) into one
identity, and hold resolved display names.

**Why rejected:** the Contacts framework already does both. A `CNContact` owns
multiple phone numbers and emails, so **identity unification is system-maintained**.
Messages.app performs this join at runtime in memory via `CNContactStore` — it is
never persisted to `chat.db`, which is why there is no table to point at. Storing
display names would also make renames invalidate stored rows.

**Replacement:** resolve live at query time —
`name → CNContact → identifiers → chat.db handle rows → handle_ids`.
Contacts loaded into an in-memory dict at startup.

### ❌ Storing `chat_name` / `participants` on chunks

**Why rejected:** volatile. Group chats get renamed constantly; contacts get
edited. Copying a name into a million rows means a rename rewrites a million rows.

**Replacement:** store `chat_id` only; resolve titles at hydration time.

### ❌ Names and dates inside `embed_text`

**Why proposed:** a header like `Conversation with Alice Chen · March 2024` would
let semantic queries mentioning Alice or March get a partial lift.

**Why rejected, three reasons:**
1. **Dates are actively harmful** — "March 2024" tokenises into noise that pulls
   unrelated March conversations together. Dense vectors have no ordering and
   cannot represent "before June" anyway.
2. **Names are exact-match tokens** — BM25 handles them with IDF weighting;
   embeddings smear them.
3. **Dilution** — mean-pooling over 400 tokens where 20 are boilerplate
   measurably shifts the centroid toward "generic conversation".

Additionally, baking a volatile name into a vector means a rename silently
invalidates the embedding.

**Kept exception: speaker role.** Not names — roles.
```
Me: i'll cover the flights          vs.    Them: i'll cover the flights
Them: ok i'll get the hotel                Me: ok i'll get the hotel
```
Same tokens, opposite meaning. Who committed to what is semantic content, and
`is_from_me` never changes. Groups use stable speaker indices by `handle_id`.

### ❌ A `name_fts` FTS5 table

**Why rejected:** over-engineering. There are a few hundred to a few thousand
contacts. That is a Python dict plus `rapidfuzz`, resolved in microseconds.
FTS5 over 1 000 rows earns nothing.

### ❌ A `chunk_text` column

**Why rejected:** duplicates message content that already lives in `chat.db`.
The chunk's message set can be reconstructed on demand for the ~8 chunks
actually shown, and names resolve *at that moment* so citations are always current.

(An `embed_text` column survived this round, on the grounds that the vector is a
frozen function of it. It was removed later — see "❌ An `embed_text` column"
below.)

### ❌ A full `attachment` / `attachment_exif` table — then ✅ a minimal one

**Why proposed:** hold EXIF-derived place names, taken-dates, mime types,
filenames and paths, since EXIF is expensive to compute.

**Why rejected as proposed:** walk both directions.
- **Retrieval:** the query "lisbon" must *match* the chunk. Mime types,
  filenames and paths contribute nothing to that — they are all in `chat.db`
  and joinable at read time.
- **Display:** once a chunk is relevant, join `chat.db` for paths and read the
  EXIF header at that moment — ~20 files, first few KB each, single-digit
  milliseconds.

⚠️ **But a narrow version came back, and the reason is instructive.** The
original rejection assumed place names were rendered as a **trailing blob** on
the chunk, so a single `chunks.places` TEXT column sufficed. Rendering them
**inline inside the media placeholder** —

```
[photo Lisbon Alfama Portugal sunset.jpg]
```

— requires knowing *which attachment was taken where*, at format time, which
recurs on every rebuild. Re-reading EXIF from disk each time is slow and fails
outright for iCloud-offloaded files. A chunk-level column cannot express the
association.

**So `attachment_place(attachment_id, place)` exists** — geotagged attachments
only, ~4 MB. It **replaces** `chunks.places` rather than adding to it, and is
properly normalised instead of a denormalised concatenation. Note this is
strictly less than the original proposal: no mime, no filename, no path, no
`taken_ts` — only the one field that is expensive to recompute and needed at
format time.

**Still out of scope:** `taken_ts` ranges (a 2019 photo sent in 2024 is
invisible to chunk timestamps), geo-radius queries, faceting by country without
a text match. Those would need the fuller table; none are v1 requirements.

**Transferable lesson:** the rendering decision determined the storage shape.
Rejecting the table was correct *given a trailing-blob rendering*, and became
wrong once the rendering changed. Re-check storage decisions when the thing
consuming them changes.

### ❌ An attachment scan watermark

**Why proposed:** avoid rescanning 50 GB of files on every sync.

**Why rejected:** redundant. `SELECT MAX(msg_id) FROM chunk_message` already
encodes progress. Attachments hang off messages, and messages are what chunking
consumes — so any new attachment belongs either to a message beyond the frontier
(its chunk is not built yet, and EXIF extraction happens during construction) or
to the open tail chunk (rebuilt every sync anyway). EXIF extraction is a step
*inside* chunk construction, not a separate pipeline.

**The gap a watermark would not have fixed either:** iCloud-offloaded attachments.
File absent at build time → place name missing; file arrives later, but its
message ROWID is behind the frontier so nothing revisits it. That needs a
**failure list** (`attachment_retry`), not a watermark — and only if Phase 0
shows offloading is significant.

### ❌ Copying `chat.db`

**Why rejected as a general policy:** the copy is immediately stale, requires
sync logic, and duplicates gigabytes.

**Replacement:** `ATTACH` read-only and join live. SQLite WAL mode genuinely
supports many concurrent readers alongside one writer, so this is safe, not a hack.

**Narrow exception retained:** the *initial bulk build* is a multi-hour
full-corpus scan, and a long-lived read transaction blocks WAL checkpointing and
inflates `chat.db-wal`. Copy once for the backfill; go live for incremental sync
and all query-time reads.

### ❌ An `embed_text` column storing the vectorized text

**Why proposed:** needed for re-embedding after a model swap, debugging, and
eval regex signals.

**Why rejected, then REVERSED:** the original argument was that `body` is a pure
function of `chat.db` plus the formatter — a *cache, not data* — so storing
~2 GB (~400 MB compressed) to serve reads happening "twice a year" was a bad
trade.

**That reasoning did not survive review.** Three things broke it:

1. **The pipeline changed.** Reranking must score `(query, document)` pairs over
   the **top 50 candidates on every query**, and the cross-encoder was
   originally scheduled *after* hydration — meaning it would have received no
   text at all. Fixing the order means hydrating 50 chunks per query; without a
   stored body that is 50 cross-database joins on the hot path, roughly
   **100 ms of avoidable latency per query**.
2. **Byte-identical regeneration is not achievable by a version string.** It is
   also broken by source-message edits, changes to the extract filters,
   `typedstream` decoder drift, and group speaker-index reassignment. A stored
   body *is* the text that was embedded, by definition — which also removes the
   need for a `body_sha256` drift-detection column.
3. **"Twice a year" was wrong**, and 400 MB is a small price for removing
   100 ms/query and an entire class of correctness risk on a personal machine.

**Current design:** `chunks.body_semantic` stores zstd-compressed text (~400 MB). FTS5
remains **contentless** — that is orthogonal, and still avoids a second
uncompressed copy in the shadow table.

### ❌ `chunk_speaker` and `chunk_participant` tables

**Why proposed:** "what did Alice *say*" and "conversations *with* Alice" need
different sets — someone present but silent in a 45-minute window sends no
messages, so a speaker-only index silently drops every chunk where you did the
talking.

**Why rejected:** the distinction is real, but neither table is. Both are
derivable, and one of them duplicated a table Apple already maintains:

| Relation | Derivation |
|---|---|
| speakers | `chunk_message ⋈ im.message.handle_id` |
| participants | `im.chat_handle_join ⋈ chunks.chat_id` — **already in `chat.db`** |

Participation is a property of the *chat*, not the chunk, so materialising it
per-chunk was both redundant and 1.25 M× larger than the source. This is exactly
the duplication the whole design forbids.

**Replacement:** `chunk_message(msg_id, chunk_id)` — one explicit membership
relation, ~240 MB, which also fixes the ROWID-range bug and gives the reverse
lookup that eval scoring and reseal both need.

**Known limitation:** `chat_handle_join` reflects *current* membership, so
someone who left a group still matches its older chunks. Accepted — historical
membership is not reliably recorded in `chat.db` anyway.

### ❌ A bare `message.ROWID` range as chunk membership

**Why proposed:** `first_msg_id`/`last_msg_id` looked sufficient to identify a
chunk's messages, and avoided a junction table.

**Why rejected — this was a correctness bug, not an optimisation.**
`message.ROWID` is assigned **globally and chronologically across all chats**,
not per chat. A chunk covering a long window in a quiet conversation therefore
spans a range containing thousands of messages from *unrelated* chats.

Consequences, both silent:

- **Hydration** would feed other people's conversations into the LLM context.
- **Eval scoring** would count a positive from any concurrent chat as a hit,
  inflating recall by an unknown amount.

**Replacement:** `chunk_message`. The `first_msg_id`/`last_msg_id` columns were
initially retained as a "range hint" and then **removed entirely** — nothing
needed them once membership was explicit (sync derives its resume point from
`chunk_message`), and their mere presence invited the very bug they caused.
Deleting them makes it unrepresentable rather than merely discouraged.

### ❌ A `body_sha256` drift-detection column

**Why proposed:** with `body` unstored, regeneration could silently produce
different text than was embedded, so a 32-byte hash per chunk (~40 MB) made
drift detectable.

**Why rejected:** it solved a problem created by not storing `body`. With
`chunks.body_semantic` stored, the text *is* what was embedded — there is nothing to
drift. Reseal detection now comes from `chat.db`'s own `date_edited` /
`date_retracted` signals mapped through `chunk_message`, which is both cheaper
and more direct than hashing.

### ❌ A `names` column in the FTS index

**Why proposed:** the fail-open path when query parsing misses a person — the
embedding deliberately cannot represent names, so a lexical route seemed needed.

**Why rejected:** wrong layer, and it reintroduced volatility into an otherwise
immutable index. A rename would force deleting and reinserting those chunks' FTS
rows — and since FTS5 rows are whole-row, regenerating `body` for ~100 k chunks
along with them.

**Replacement:** an in-memory `rapidfuzz` match of query tokens against the
contact list, producing a structured `handle_id` filter. Strictly better on four
counts: exact ids rather than fuzzy lexical scores, catches nicknames and
misspellings BM25 would miss, deterministic and unit-testable (unlike the LLM
step it backstops), and microseconds to run.

**Result:** the invariant that **no name appears anywhere in `index.db`**, so
renames require zero index work.

### ❌ A general-purpose `meta` / `aux_text` FTS column

**Why proposed:** one text field bundling dates, place names, filenames and media
types, indexed for BM25 so any of them could be matched lexically.

**Why rejected:** forcing each component to justify itself separately, three of
four failed.

| Component | Verdict |
|---|---|
| Dates | → `start_ts` range filter. BM25 cannot express ranges, relative dates or ordering; `"march"` fires on every March equally and dilutes `body` scoring |
| Media types | → `has_attachment` boolean. `"photo"` in ~10⁵ chunks has near-zero IDF, so BM25 weights it ≈ 0 regardless |
| Filenames | → dropped. `IMG_4821.HEIC` is noise |
| Place names | **kept**, appended to the lexical body by `format_lexical()` |

**Governing rule that emerged: anything expressible as a structured predicate is
a filter, never indexed text.** Places are the sole exception because
*"what did we do in lisbon"* must match both conversations mentioning Lisbon and
photos taken there — an OR contributing to *ranking*, which a filter cannot
express without wrongly excluding the textual mentions.

### ❌ HNSW / a dedicated vector database

**Why rejected:** at ~1 M vectors, brute force is fast enough and preserves exact
metadata pre-filtering, which ANN indexes break. A separate vector service would
also add a daemon competing for 16 GB.

### ❌ LangChain / LlamaIndex

**Why rejected:** the pipeline is ~400 lines of explicit code. A framework would
hide precisely the parts requiring control (pre-filtering, RRF, rowid alignment)
while adding dependency weight and version churn.

### ❌ Contextual retrieval (LLM-written per-chunk summaries)

**Why proposed:** prepending an LLM summary of the parent document to each chunk
reduces retrieval failures ~35 % on document corpora.

**Why rejected here:** 1 M chunks means 1 M LLM calls — days of local GPU time.
For iMessage the natural context is cheap structural metadata (conversation,
participants, time), which is free.

**Partially retained as a deferred option:** a *coarser* layer of
`(chat_id, day)` summaries is ~30–60 k calls (one overnight run) and would help
whole-conversation questions. Add only if Phase 8 shows the gap.

---

## 10. How iMessage stores data

### Files

| Path | Contents |
|---|---|
| `~/Library/Messages/chat.db` | SQLite database — all message data |
| `~/Library/Messages/Attachments/` | Attachment files (usually the bulk of disk usage) |
| `~/Library/Application Support/AddressBook/…/AddressBook-v22.abcddb` | Contacts (Core Data; **use the framework instead**) |

### Key tables in `chat.db`

| Table | Contents |
|---|---|
| `message` | One row per message: `text`, `attributedBody`, `date`, `is_from_me`, `handle_id`, `associated_message_type` |
| `handle` | The other party — **identifier only** (`+447700900123`, `a@b.com`). **No display names.** |
| `chat` | Conversations, including groups; `guid` is stable across renames |
| `chat_message_join` | Message ↔ chat |
| `chat_handle_join` | Chat ↔ participant |
| `attachment`, `message_attachment_join` | Files on disk |

### Traps

1. **`message.text` is frequently NULL on newer messages.** The real content is
   in `attributedBody`, an NSKeyedArchiver (typedstream) blob. Reading only
   `text` silently loses a large share of recent history. Decode with
   `typedstream`/`pytypedstream`; regex over the blob as a crude fallback.
2. **Apple epoch nanoseconds**, not Unix: `unix = apple_ns / 1e9 + 978307200`
   (epoch 2001-01-01 UTC).
3. **Link previews and rich content** hide in `payload_data`.
4. **Tapbacks** appear as messages with `associated_message_type != 0` — filter them.
5. **One person = many `handle` rows** (phone over SMS, different number over
   iMessage, iCloud email), with **no link between them in `chat.db`**. Unification
   must come from Contacts.
6. **Full Disk Access** is required to read `chat.db`.
7. Read-only connections need the `-shm` file; if Messages.app has not run since
   boot the open can fail. Retry, or launch Messages.

### Does macOS already do this?

Partly, and not usefully. Apple Intelligence added **on-device semantic indexing**
to Messages/Spotlight, so natural-language queries work in the Messages app. But:

- The semantic index is **private to Apple's apps** — not in `chat.db`, no API
- You cannot query it, filter it, feed results to an LLM, or inspect its chunking
- No citations, no custom time/person filters, no programmable retrieval

So this project is not redundant; it is the only route to programmable retrieval
with citations over the same data.

---

## 11. Hardware and capacity maths

### Machine budget (M5, 16 GB unified)

macOS consumes ~4–5 GB, leaving **~10–11 GB**. Unified memory means CPU and GPU
share it — an advantage over a discrete 16 GB GPU, but a hard shared ceiling.

⚠️ **Weight size is a lower bound on memory, not a budget** — process RSS
includes framework imports, Metal heaps and fragmentation. But with generation
belonging to the MCP client and query parsing deterministic, **there are no local LLM
weights at all**, and the budget stops being tight.

| Stage | Resident | Weight bytes |
|---|---|---|
| Indexing | embedder only | ~0.15 GB |
| Serving | embedder 0.066 + MiniLM-L-12 reranker 0.066 + DB page cache 0.256 | **~0.39 GB of weights + cache** |

Against ~10–11 GB available. **Both models stay loaded**; the serialised
load/unload dance an earlier revision required was driven entirely by holding
~8 GB of local LLM weights (Qwen3-4B parser + Qwen3-8B generator), and that
requirement is gone.

Both models are small enough that quantisation is unnecessary — 66 MB each at
fp16. Right-sizing the reranker mattered far more than compressing it; see
§5 "Model selection".

**No serialisation needed.** An earlier revision required unloading each model
before loading the next — a constraint driven entirely by holding ~8 GB of local
LLM weights. With generation belonging to the client and parsing deterministic,
both MLX models fit alongside the page cache with ~9 GB to spare. Load once,
leave them.

⚠️ **Weights are the small part.** Measured per-process RSS is ~790 MB idle, and
**~300 MB of that is the MLX import and Metal device context** — larger than
both models combined. Since MCP uses stdio, each client window spawns its own
process, so that overhead multiplies. Full accounting in `PLAN.md` Phase 6.

### Corpus sizing (assuming ~2 GB extracted text)

```
2 × 10⁹ chars ÷ ~1600 chars/chunk ≈ 1.25 M chunks
```

| Component | Size |
|---|---|
| int8 vectors (384-d) | 480 MB |
| FTS5 inverted index (contentless, with positions) | ~0.75–1.2 GB |
| `chunks` rows incl. zstd `body_semantic` | ~540 MB |
| `attachment_place` | ~4 MB |
| `chunk_message` (~15 M rows) | ~240 MB |
| **Total `index.db`** | **~2–2.5 GB** |

`body_semantic` is stored **zstd-compressed** (~400 MB of the `chunks` figure, vs ~2 GB
raw). It buys ~100 ms per query by removing 50 cross-database hydration joins
from the rerank path, and eliminates the byte-identity risk entirely — see §9,
"❌ An `embed_text` column", which records this decision being made, reversed,
and why.

**No fp32 vectors are stored, ever.** int8 is the only precision on disk; fp32
is regenerable by re-embedding if it is ever needed for comparison. That is
1.9 GB not spent — see §"Quantisation".

Working set under load ~2–3 GB via the SQLite page cache (`PRAGMA cache_size`);
the full DB need not be resident. All figures are estimates until Phase 3
measures them.

### Model sizing reference (4-bit)

| Params | Memory |
|---|---|
| 7–8 B | 5–8 GB |
| 14 B | ~10 GB |
| 32 B | ~20 GB |
| 70 B | 40–48 GB |
| 200 B+ MoE | 100–200 GB |

### What is actually accelerated — and what is not

The query path spans **three runtimes**, and only one of them is MLX. Worth
knowing before optimising the wrong thing.

| Stage | Backend | Bound by | GPU-able? |
|---|---|---|---|
| Embedding — build (1.25 M chunks) | **MLX / GPU** | compute | ✅ already |
| Embedding — query (1 vector) | MLX / GPU, ~2 ms | call overhead | ✅ already |
| `vec0` vector scan | **CPU, NEON SIMD** | **memory bandwidth** | possible, see below |
| FTS5 / BM25 | **CPU, scalar** | pointer chasing | ❌ never |
| Cross-encoder rerank (50 pairs) | **MLX / GPU** | compute | ✅ once batched |
| Query parsing | pure Python (`dateparser`, `rapidfuzz`) | trivial | n/a |
| Generation | **the MCP client's model** | outside this process | n/a |

**FTS5 will never use the GPU, and should not.** An inverted-index lookup is
varint decoding, pointer chasing and postings intersection — branchy, serial,
cache-hostile. That is exactly why sparse retrieval stays cheap on a CPU.

**`vec0` is CPU SIMD and probably adequate.** The scan is bandwidth-bound
(480 MB read per query), not compute-bound, so a GPU only helps to the extent it
pulls more bandwidth — and you must first cross from SQLite into MLX arrays to
get there. At 1.25 M vectors NEON should land in tens of milliseconds; GPU
acceleration starts mattering around 10 M+.

> **Convergence worth exploiting.** If the Phase 0 spike shows `vec0` cannot
> accept a constrained candidate id set, the fallback is holding the int8
> vectors as one mmapped array and scanning in NumPy. That same array can be an
> `mx.array` scanned with `mx.matmul` + `topk` on the GPU. **The fallback for
> the correctness problem is also the GPU path** — one decision resolves both.

### MLX consolidation

An earlier revision ran the cross-encoder on PyTorch + MPS, which meant three
runtimes in one query path, each with its own import cost, memory pool and
warm-up — on a 16 GB machine, pure waste. Two facts changed the picture:

1. **MLX supports cross-encoder reranking.** `mlx-embeddings` handles
   sequence-classification models, and there are dedicated MLX reranker projects
   (`mlx-embed-rerank-server`, `mlx-rerank`, `jina-reranker-v3-mlx`). BGE
   rerankers are compatible. **PyTorch and `sentence-transformers` are gone.**
2. **Generation moved to the MCP client**, removing Ollama and ~8 GB of
   local LLM weights entirely. Query parsing became deterministic
   (`dateparser` + `rapidfuzz`), so no local LLM remains for any purpose.
   **MLX is now the only ML runtime in the system.**

**Four optimisations, in expected-impact order:**

1. **Batch the reranker.** 50 candidates is *one batched forward pass*, not 50
   sequential ones. Sequential inference wastes nearly all GPU parallelism;
   MLX throughput scales close to linearly with batch size, and unified memory
   means no host↔device copies.
2. **`@mx.compile`** the embed and rerank forward passes. MLX evaluates lazily;
   compiling fuses the graph into fewer, larger Metal kernel launches. Build the
   whole batch graph, then a single `mx.eval`.
3. **Right-size the models** — worth more than compressing the wrong ones.
   `ms-marco-MiniLM-L-12-v2` (33 M) reranks 50 pairs in ~60 ms;
   `bge-reranker-v2-m3` (568 M) would take ~860 ms and consume the entire tool
   budget. See §5 "Model selection" for the full candidate field.

At 33 M params each, neither model needs quantisation — 66 MB at fp16. The
vectors are quantised (int8); the weights are not.

> ⚠️ **Every number below is a hypothesis, not a measurement.** They come from
> published benchmarks on similar hardware, not from this machine, this
> tokeniser, this sequence length, or this MLX version. Phase 0 replaces them
> with a measured table (precision × warm/cold × filtered/unfiltered, p50/p95),
> and later phases must cite that table rather than these figures.

| Operation | Hypothesis | Notes |
|---|---|---|
| Embedding, MLX on M5 | ~500–1500 chunks/s | **matmul only** — excludes decode, format, tokenise, index writes |
| → 1.25 M chunks, embed-only | ~20–40 min | not a build-time SLA |
| → **full build wall time** | **several hours** | adds `attributedBody` decoding, EXIF over ~50 GB, index writes, throttling |
| PyTorch + MPS (not used) | ~2–3× slower than MLX | pure CPU ~10 h — annoying, not blocking |
| Vector scan, int8, 1.25 M, **warm** | ~25–75 ms | contiguous SIMD over 480 MB |
| Vector scan, int8, **cold or contended** | 100–300 ms+ | SSD page-in; unified memory shared with Metal/LLM |
| Vector scan, binary prefilter (fallback) | ~40× faster than int8 | 60 MB read; rescore top ~500 against int8 |
| Cross-encoder rerank, 50 candidates | **~60 ms** batched with MiniLM-L-12; ~860 ms with `bge-reranker-v2-m3` | model size dominates; see §5 "Model selection" |
| End-to-end retrieval (a tool call blocks the agent turn) | target < 1 s | |

**Plan against the pessimistic column.** Two places depend on it: the `< 2 s`
retrieval budget, and the `nn_distance` harvest, which is `5 000 × scan latency`
— 4 minutes at 50 ms, 25 minutes at 300 ms (`EVALUATION.md` §3.2).

The "2–3× faster than MPS" claim for MLX is likewise a hypothesis to confirm in
Phase 0, not a settled fact. So is the batched-reranker estimate — measure it.

Run indexing plugged in with other applications closed; the machine will
thermally throttle otherwise.

---

## 12. Design principles distilled

1. **The system of record is Apple's, not ours.** Store only derived data.
   Everything volatile is resolved live.
2. **Structure is a `WHERE` clause; meaning is a vector.** Never push names,
   dates or identifiers through an embedding.
3. **Index, not source of truth.** `chunk_message`,
   `chunks_fts` and `chunks_vec` are all rebuildable from `chat.db` plus
   embeddings. Losing `index.db` costs one overnight run — **except
   `attachment_place`**, which needs the original attachment files and may be
   unrecoverable if they were offloaded to iCloud. Back that up.
4. **Late binding.** Store sufficient statistics; apply scoring at query time.
   This is why FTS5 is incremental and why ranking can change without reindexing.
5. **Append-only, but convergent.** Only the open tail chunk of a chat grows —
   yet any chunk must be rebuilt when its source messages are edited or unsent.
   Append-only describes the history of chunks, not immunity to source mutation.
6. **Cheap and exact beats clever.** Brute-force scan preserves exact
   pre-filtering and trivial incrementality; HNSW trades both away for latency
   we do not need. ⚠️ Conditional on `vec0` actually accepting a candidate id
   set — a Phase 0 spike, not an assumption.
7. **Fail open on the query path, loud on the ingest path.** A query parser
   error degrades to pure semantic search and a low-confidence contact match
   applies no filter at all; a schema change in `chat.db` must halt loudly
   rather than silently produce wrong results.
8. **Measure before optimising.** Every number in §11 of this document is an
   estimate until Phase 0 and Phase 8 replace it. Where a figure is load-bearing
   — scan latency, memory headroom — the plan must cite the measured value, not
   the estimate.
9. **Locally-retrieving, remotely-answering.** The corpus, the index, the
   embeddings, the query, and every candidate that does not survive reranking
   stay on the machine — not as policy, but because no code path sends them.
   ⚠️ **Generation is the exception**: the final ~8 retrieved sessions, hydrated
   with real message text and contact names, are returned to the MCP client — and
   what it does with them is outside this system's control. Retrieval precision
   is therefore also a privacy control — a tighter final selection means less
   leaves the machine.
   ⚠️ Nor is this a claim that `index.db` is name-free: message bodies are
   tokenised into the FTS index, so names people typed are present as terms.

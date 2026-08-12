I've read the full pipeline, the architecture doc, and verified key facts against the live data (`chat.style` 43=group/45=1:1 — 545/1057 in the snapshot; 17,525 chunks / 19.4 MB index.db; 656 MB chat_snapshot.db; FastAPI 0.141 + uvicorn 0.52 + sse-starlette already in the venv as `mcp` transitives; `copilot -p` measured **6.5 s** for a one-word reply on this machine, worse than the ~4.4 s in ADDENDUM §14).

---

# seaglass Desktop App — Technical Plan

## 1. Technology choice for the app shell

**Recommendation: a single Python process running FastAPI + uvicorn on `127.0.0.1`, serving a dependency-free static HTML/CSS/JS frontend, opened in the user's default browser. Optionally wrapped later in `pywebview` for a real window — but not in v1.**

### Why this and not the alternatives

| Option | Verdict | Reasoning specific to this repo |
|---|---|---|
| **(a) SwiftUI shell + Python subprocess over socket/stdio** | **Reject** | Requires inventing and versioning an IPC protocol, then re-implementing every UI-facing model (sessions, hydrated messages, contacts) in Swift. Two languages, two build systems, Xcode, code-signing, and entitlements for Full Disk Access *in the Swift app* on top of the Python process that also needs it. For a solo hobbyist whose AI pair is a CLI agent operating on a Python repo, this triples the surface area for zero retrieval-quality gain. The "native feel" this buys is a menu-bar icon and an NSWindow — not worth it. |
| **(b) PyQt6/PySide6 single-process GUI** | **Reject** | Genuinely viable and it *is* one process holding warm models. But: adds a ~100 MB binary dependency to a venv that currently has zero GUI deps; PyQt6 is GPL/commercial-licensed (PySide6 LGPL is the saner pick if you go this way); building a chip-based person autocomplete, a date range picker, and a rich results list in Qt widgets is materially slower than in HTML; and you must be careful to keep MLX work off the Qt event loop thread anyway, so you end up with the same worker-thread design as (c) plus a heavier toolkit. Long-term it's the best "real .app" story, which is why I'd revisit it only if the browser UX genuinely grates. |
| **(c) FastAPI + static HTML/JS in browser** | **✅ Recommend** | **Zero new dependencies.** `fastapi 0.141.1`, `uvicorn 0.52.1`, `starlette 1.6.0`, `sse-starlette 3.4.8`, `jinja2 3.1.6` are already installed in `.venv` as transitive deps of `mcp 2.0.0`. The whole pipeline stays in-process — `EmbeddingModel`, `CrossEncoderReranker`, `ContactIndex`, and both SQLite connections are plain module globals exactly as in `mcp_server.py`, no re-implementation anywhere. HTML gives you a chips autocomplete, `<input type="date">`, and a scrollable session list essentially for free. Packaging is `pip install -e .` + a console script — the same thing that already works for `seaglass-mcp`. Full Disk Access is inherited from the terminal/venv that already has it. |
| **(d) `pywebview` native window** | **Defer to v1.5** | It's a strict superset of (c): the same FastAPI backend, but rendered inside a WKWebView window instead of a browser tab. `pyobjc-core` and `pyobjc-framework-Cocoa` are already installed; `pywebview` on macOS additionally wants `pyobjc-framework-WebKit`. Because it's purely a *shell swap* over the same HTTP API, adopting it later costs ~30 lines. Doing it in v1 adds an untested dependency to the critical path of "get something working." |

**Decisive call: build (c), structured so (d) is a 30-line addition.** All UI logic lives behind HTTP; the shell (browser tab vs. pywebview window vs. eventually a py2app bundle) is a swappable launcher concern.

### One important consequence: localhost is not a security boundary

Any web page open in the same browser can `fetch('http://127.0.0.1:8765/api/search')`. That would exfiltrate the user's iMessage history to an arbitrary site. This is non-negotiable in the design:

- Bind **`127.0.0.1` only**, never `0.0.0.0`.
- Generate a random 32-byte token at startup; **every** `/api/*` route requires `Authorization: Bearer <token>`. A custom header forces a CORS preflight that a foreign origin cannot satisfy because we send no permissive `Access-Control-Allow-Origin`.
- Additionally reject requests whose `Origin` header is present and not our own origin, and whose `Host` is not `127.0.0.1:<port>` (DNS-rebinding guard).
- The launcher opens `http://127.0.0.1:<port>/#<token>`; `app.js` reads the fragment, stashes it in `sessionStorage`, and immediately clears `location.hash` so the token doesn't linger in browser history. Static assets (`/`, `/static/*`) are unauthenticated — they contain no data.

---

## 2. Process / architecture design — what "eager loading" concretely means

### 2.1 What is and is not genuinely eager-loadable

Be honest about each component, because "load the index into memory" means four different things here:

| Component | Is it really "loaded"? | What eager actually means |
|---|---|---|
| **`EmbeddingModel` (bge-small-en-v1.5)** | ✅ Yes — real weights | `EmbeddingModel()` alone does nothing (`_ensure_loaded` is lazy inside `embed()`). Eager = **call `.embed()` once** with a throwaway string. |
| **`CrossEncoderReranker` (ms-marco-MiniLM-L-12-v2)** | ✅ Yes — real weights + a hand-attached classifier head | Same: `CrossEncoderReranker()` is inert; eager = **call `.score([("warm","warm")])` once**. Also triggers the HF-hub revision-check HTTP round trip that the architecture doc blames for part of the ~4 s cold cost — doing it at startup means a user query never waits on the network. |
| **MLX kernel compilation** | ✅ **This is the big one** | The architecture doc's §9 says the *first query with models already resident* still costs **~1.8 s** ("one-time MLX kernel-compilation cost even though weights are resident"), while steady-state is ~0.75 s. So loading weights is **not sufficient**. Warmup must run a **full dummy search end-to-end** (real `retrieve()` → real `rerank_candidates()` on ~50 real chunks) so the Metal kernels for the actual production shapes are compiled before the user types anything. Skipping this hands the user a 1.8 s first query after they already watched a loading screen — the exact failure the requirement exists to prevent. |
| **`ContactIndex`** | ✅ Yes — a real in-memory Python object | `ContactIndex.load()` enumerates all CNContacts via PyObjC (2,331 handles here). Genuinely eager, genuinely slow-ish (~0.5–2 s), and it's also what the person-autocomplete UI queries, so it must be ready before the picker works. |
| **`index.db` (19.4 MB)** | ⚠️ **No — SQLite has no "load" step** | `open_index_db()` only opens a file handle, loads the sqlite-vec extension, and runs the idempotent schema script. The honest eager equivalents are: (i) raise `PRAGMA cache_size` to `-262144` (256 MB, comfortably larger than the whole DB); (ii) **warm the OS page cache and SQLite's own cache by touching every page** — `SELECT COUNT(*), SUM(LENGTH(body_semantic)) FROM chunks`, `SELECT COUNT(*) FROM chunks_vec`, one dummy `vec_int8` KNN, one dummy FTS5 MATCH. After that, queries do zero cold disk I/O. |
| **`chat.db` snapshot (656 MB)** | ⚠️ No, and don't try | Far too large to slurp. Eager = open read-only via `connect_readonly()` (which also runs `assert_schema`, so schema drift fails *at startup* with a clear error rather than mid-search), plus build the small **chat-metadata cache** described below. |
| **Chat metadata cache** | ✅ Yes — new, and cheap | 1,602 chats × (`style`, `display_name`, `chat_identifier`) + 3,811 `chat_handle_join` rows + 2,331 handles. Materialise once at startup into dicts: `chat_id → {is_group, title, participant_handles, participant_names}`. Powers result titles, the group/1:1 filter, and the chat picker with no per-query joins. |

**Optional, behind a flag:** `SEAGLASS_APP_MEMORY_INDEX=1` copies `index.db` into a `:memory:` connection via `sqlite3.Connection.backup()` at startup (19.4 MB — trivial). This is a *genuine* full-RAM load. I do **not** recommend it as the default: it silently freezes a snapshot, so a concurrent `seaglass build` re-index is invisible until restart, and it complicates the staleness indicator. On-disk + page-cache warming gets ~all the benefit with none of that. Offer the flag, default off.

### 2.2 Startup sequence — and how the frontend sees it

**Critical design decision: do NOT warm before uvicorn accepts connections.** If the HTTP server isn't listening, the browser gets `ERR_CONNECTION_REFUSED` and you cannot render a loading screen at all — the requirement is that the user *sees and feels* the load. So:

```
launcher (seaglass/app/__main__.py)
 ├─ read config (env + ~/.seaglass/config.json), resolve index.db / chat.db paths
 ├─ acquire single-instance lock (~/.seaglass/app.lock: {pid, port, token})
 │     └─ if a live instance exists → just open its URL in the browser and exit
 ├─ pick port (default 8765, scan upward), mint auth token, write lockfile
 ├─ start uvicorn in a thread  ────────────────► serving IMMEDIATELY, state=STARTING
 ├─ start warmup on a dedicated single worker thread (the same thread that will
 │  later execute every search — see 2.3)
 └─ webbrowser.open(f"http://127.0.0.1:{port}/#{token}")
```

The frontend loads instantly, polls `GET /api/health` every 250 ms, and renders a progress screen driven by a real step list. Warmup steps, in order, each publishing `{name, state, elapsed_s}` into a thread-safe `WarmupState`:

| # | Step | Est. | Notes |
|---|---|---|---|
| 1 | `import mlx`, `sqlite_vec` | ~0.5 s | Deliberately deferred out of module import so uvicorn is listening first. |
| 2 | `open_index_db(path, create=False, check_same_thread=False)` | <0.1 s | `create=False` so a typo'd path is a loud error, not a phantom empty index. |
| 3 | `PRAGMA cache_size=-262144; PRAGMA mmap_size=268435456` | ~0 | |
| 4 | Read `meta.int8_absmax`, `embed_version`, `build_cursor` | ~0 | Fail loudly here if absent (same check `retrieve()` does per query). |
| 5 | `connect_readonly(chat_db)` → runs `assert_schema` | ~0.1 s | Schema drift surfaces at startup. |
| 6 | Build chat-metadata cache (3 queries, ~7.7k rows) | ~0.2 s | |
| 7 | `ContactIndex.load()` | 0.5–2 s | On `ContactsUnavailableError`: **warn, continue** — names degrade to raw handles, exactly as the CLI/MCP already do. |
| 8 | Page-cache warm over `chunks`, `chunks_vec`, `chunks_fts` | 0.3–1 s | |
| 9 | `EmbeddingModel().embed(["warmup"])` | 1–2 s | Weights + HF revision check. |
| 10 | `CrossEncoderReranker().score([("warmup","warmup")])` | 1–2 s | |
| 11 | **Full dummy search** — `parse_query` → `retrieve` → `rerank_candidates` → `aggregate_sessions` → `expand_sessions` → `hydrate_sessions` on a fixed innocuous query (e.g. `"dinner plans"`), results discarded | 1.5–2 s | **This is the step that buys the ~1.8 s→~0.75 s difference.** Also end-to-end-validates every wiring path before the user's first query. |
| 12 | GHCP availability probe (`copilot --version`, 5 s timeout) — *non-blocking, may finish after READY* | ~0.3 s | See §3. |
| | **Total** | **~6–10 s** | |

`/api/health` response shape:

```jsonc
{ "state": "STARTING" | "READY" | "DEGRADED" | "FAILED",
  "steps": [{"name":"embedding_model","state":"done","elapsed_s":1.42}, ...],
  "progress": 0.72,
  "elapsed_s": 4.8,
  "error": null,
  "warnings": ["Contacts unavailable — sender names will show raw handles"],
  "ghcp": {"available": true, "version": "GitHub Copilot CLI 1.0.79", "probing": false} }
```

`DEGRADED` = ready to search but something optional failed (Contacts, GHCP, or no `chat.db` → un-hydrated previews, mirroring `mcp_server.py`'s existing fallback). `FAILED` = index missing/corrupt; the UI shows the error text and a "Choose index.db…" path field that `POST /api/config`s and re-runs warmup.

**Progress-screen UX:** full-window centered card, the step list with per-step spinner/✓/⚠, a determinate bar, and the literal text *"Loading models once so every search is fast."* This turns the 6–10 s from a mystery hang into an explained, one-time cost — which is precisely the user's stated intent. After READY the card collapses into a status pill in the footer showing total load time.

### 2.3 Concurrency model

Copy `mcp_server.py`'s hard-won lessons rather than rediscovering them:

- **One `ThreadPoolExecutor(max_workers=1)`** owns *all* pipeline work: warmup, searches, conversation fetches. This subsumes both `_init_lock` and `_pipeline_lock` — MLX has no cross-thread safety guarantee, and serialising through a single worker thread makes that structurally true instead of lock-dependent.
- SQLite connections opened with `check_same_thread=False` (as `open_index_db` already supports) — needed because warmup and requests are dispatched through the executor from uvicorn's loop threads.
- Route handlers are `async def` and do `await loop.run_in_executor(pipeline_pool, fn, ...)`, so `/api/health` stays instantly responsive *while* a search or the warmup occupies the worker. This is what makes the loading screen animate.
- GHCP subprocess calls run on a **separate** `ThreadPoolExecutor(max_workers=1)` — they're I/O-bound and must never block a search.
- Search requests carry a client-generated `request_id`; a newer request supersedes older ones (results for a stale id are dropped client-side). No cancellation of in-flight MLX work — it's <1 s, not worth it.

---

## 3. GHCP integration design

### 3.1 What GHCP actually buys — honestly

`parse.py` is already good at the common cases, so be specific about the residual gaps:

**Real wins:**

1. **Participant extraction from arbitrary syntax — the biggest one.** `_PARTICIPANT_PATTERN` is `\b(?i:from|with)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?)`. It requires the name to be *capitalized* and *immediately preceded by "with"/"from"*. It therefore extracts nothing from: `"did jenny ever send me the lease"` (lowercase, no preposition), `"photos sarah sent in march"`, `"the thread where Ben and Alex argued about the lease"` (catches Ben, misses Alex — the group is capped at two tokens and one match per preposition), `"what my landlord said about the deposit"`. An LLM extracts the person from any of these.
2. **Date expressions outside `dateparser`'s grammar.** `dateparser` handles "last week", "in march", "3 days ago". It does not handle "over the holidays", "two summers ago", "right before I moved", "early last year", "around Thanksgiving". Given today's date in the prompt, an LLM maps these to concrete ranges.
3. **Suppressing false-positive filters.** `_extract_media_filter` sets `has_media=True` if *any* media word appears anywhere — `"the photo he described in the email"` wrongly forces `has_attachment = 1` and can silently zero out the true positive (the eval harness already tracks a "filter kill rate" for exactly this class of failure). An LLM can distinguish "I'm looking for an image" from "someone said the word photo".
4. **Sparse-arm query expansion.** FTS5 BM25 is literal. Expanding `"vet appointment"` → `"veterinarian vet appointment checkup clinic"` measurably helps the lexical arm.

**Honest non-wins:**

- It does **not** improve dense retrieval. bge-small already handles paraphrase/synonymy; rewriting the semantic residual risks *hurting* it by drifting off the user's actual phrasing.
- It does **not** beat `rapidfuzz` at name→handle resolution. `ContactIndex.handle_ids_for_names` at threshold 85 is already better than an LLM that has never seen the roster. GHCP's job is to *locate* the name span; local code resolves it.
- Nicknames ("mom", "my sister") are **not** an LLM problem — the model doesn't know who your mom is. Solve locally with `~/.seaglass/aliases.json` (`{"mom": "Mary Chen", "the landlord": "+15551234567"}`), applied before parsing, deterministically, with zero latency.

### 3.2 The privacy line — state it explicitly

`README.md` promises "no message content, embeddings, or queries are sent anywhere." **Enabling GHCP breaks the "queries" half of that promise**: `copilot -p` is a cloud service. This must be surfaced honestly, not buried.

Hard rules, enforced in code:
- **Only the raw query string** ever leaves the machine. Never message text, never chunk bodies, never search results, never the contact roster, never handles. (This is why §3.1 keeps name *resolution* local — GHCP sees "sarah", never your address book.)
- Assist is **off by default**. First time the user toggles it on, a one-time modal states plainly: *"Your typed query text will be sent to GitHub Copilot. Your messages, contacts, and results never are."* Choice persisted in `~/.seaglass/config.json`.
- README and `docs/architecture.html` get an updated privacy-boundary paragraph.

### 3.3 Latency-aware design: speculative parallel refinement, never blocking

Measured on this machine just now: `copilot -p "Reply with only the word: ok" -s --no-color --log-level none` → **6.5 s wall**. Warm query latency is ~0.75 s. A blocking pre-parse makes every search **~9× slower**. That is disqualifying.

**The design:**

```
user hits Enter
 ├─ t=0.00  deterministic path fires immediately (parse.py + UI filters)
 ├─ t≈0.75  RESULTS RENDER. Full fidelity. Never gated on GHCP.
 └─ t=0.00  if assist enabled AND copilot available AND not cached:
             GHCP call dispatched concurrently on the ghcp thread pool
       t≈5–8  response arrives
              ├─ validate + merge → ParsedQuery'
              ├─ if ParsedQuery' == deterministic parse → silently discard
              └─ else → non-modal banner above results:
                 "Copilot read this as: photos from Sarah · Nov 1–Dec 31 2024   [Apply] [Dismiss]"
                 Apply → re-runs the pipeline with ParsedQuery' (~0.75 s), results swap in.
```

Why this shape and not the alternatives:
- **Blocking pre-parse:** rejected — 9× regression on every search.
- **Auto-apply on arrival:** rejected — results silently mutating 6 s after you started reading them is disorienting, and a wrong LLM filter would *remove* correct results with no explanation. The banner keeps the user in control and makes the assist legible.
- **Per-search toggle:** yes, additionally — a "✨ Assist" segmented control next to the search bar with `Off / Auto / Force`. `Auto` (recommended default once opted in) only fires GHCP when the deterministic parse looks *weak*: i.e. `people_participant == []` **and** `date_from is None` **and** `not has_media` **and** the query has ≥4 tokens — meaning parse.py extracted nothing structured from a query complex enough that it probably should have. This alone cuts GHCP calls by a large fraction, since simple queries are exactly the ones parse.py already nails. `Force` always calls it (useful for the retry button when results are bad).
- **Caching:** mandatory. Key = `sha256(prompt_version || normalized_query || today's local date || aliases_file_mtime)`. Store in **`~/.seaglass/app.db`** (a *separate* SQLite file — never write to `index.db`, so re-indexing never nukes the cache and the app keeps a strict read-only relationship to the index). Table `ghcp_cache(key TEXT PRIMARY KEY, query TEXT, response_json TEXT, created_at REAL)`, 30-day TTL. Repeat queries return in ~0 ms. Also cache negative results (unparseable response) so a pathological query isn't retried at 6.5 s each time.

### 3.4 Detection: is `copilot` available?

`shutil.which("copilot")` is **necessary but not sufficient** — note this machine's binary lives at `~/.local/state/fnm_multishells/59915_.../bin/copilot`, an fnm shim path that exists only in a shell with fnm initialised. A GUI-launched app inherits a different `PATH` and would find nothing even though the CLI works fine in the terminal.

Three-stage detection, run once during warmup step 12 (background, non-blocking):

1. `shutil.which("copilot")`, then fall back to a small list of known locations (`~/.local/bin`, `/opt/homebrew/bin`, `/usr/local/bin`, and the resolved realpath of any fnm/nvm shim recorded in config). Allow explicit override via `SEAGLASS_COPILOT_BIN` / `config.json` → `copilot_bin`.
2. Run `<bin> --version` with `timeout=5`. Non-zero exit or timeout → unavailable (this also catches the "installed but not authenticated" case, which exits non-zero).
3. Record `{available, bin, version, reason}` in `WarmupState`; surface via `/api/health.ghcp`. Never re-probe per search.

**Circuit breaker:** 3 consecutive failures (non-zero exit, timeout, or unparseable JSON) → assist auto-disables for the remainder of the process, banner reads *"Copilot assist disabled after repeated failures"* with a manual re-enable. Prevents a broken/expired auth state from adding a silent 20 s of dead work to every search.

### 3.5 Wrapper: reuse or new?

`eval/ghcp_client.py`'s `call_ghcp` is exactly the right invocation shape and should be reused verbatim (`[bin, "-p", prompt, "-s", "--no-color", "--log-level", "none"]`, no `--allow-all-tools` — a query-parsing prompt has no business touching the shell). Two gaps:

- `call_ghcp_json` regexes `\[.*\]` — **arrays only**. Query parsing wants a single JSON *object*.
- `DEFAULT_TIMEOUT_S = 180` is right for a 10-chunk generation batch, catastrophically wrong for an interactive search.
- `COPILOT_BIN = "copilot"` is a module constant; needs to accept the resolved path from detection.

**Plan: promote the module to `seaglass/llm/ghcp.py`** (it's no longer eval-only), and leave `seaglass/eval/ghcp_client.py` as a 3-line re-export shim so `eval/generate.py` and `tests/test_generate.py` keep working untouched. Additions:

```python
call_ghcp(prompt, *, timeout_s=DEFAULT_TIMEOUT_S, bin_path=COPILOT_BIN) -> str
call_ghcp_json_object(prompt, *, timeout_s, bin_path) -> Optional[dict]   # new: _JSON_OBJECT_RE = r"\{.*\}"
detect_ghcp(explicit_bin: Optional[str] = None) -> GhcpAvailability        # new dataclass
```

App-side timeout: **20 s** (≈3× the measured 6.5 s; anything slower is useless for an interactive assist).

### 3.6 The prompt and the response contract

Sent as a single `-p` argument (query text is interpolated as a JSON string literal so quoting/injection is contained):

```
You are a query parser for a personal iMessage search engine. Today is {YYYY-MM-DD} ({weekday}), local timezone {tz}.

Given one search query, extract structured filters and return ONLY a JSON object, no prose, no code fence:
{"semantic": str, "people": [str], "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null,
 "has_media": true|false|null, "is_group": true|false|null, "expansions": [str], "confidence": 0.0-1.0}

Rules:
- "semantic": the topical content only, with person names and date expressions REMOVED. Never empty; if nothing remains, repeat the original query.
- "people": names of people who were PARTICIPANTS in the conversation (senders/recipients), NOT people merely mentioned in the text. Copy the name spans verbatim from the query; do not guess full names, do not invent surnames.
- dates: resolve relative expressions ("over the holidays", "two summers ago") to concrete inclusive dates. Null if the query implies no time constraint.
- "has_media": true only if the user is looking FOR an image/video/attachment; false if a media word appears only incidentally ("the photo he described"); null if no signal.
- "is_group": true if the query implies a group chat, false if a one-on-one, null otherwise.
- "expansions": up to 5 extra single-word keyword synonyms to help a BM25 keyword search. No phrases. Empty if none help.
- "confidence": your confidence in this parse.

Query: {json-quoted query}
```

### 3.7 Merging into `ParsedQuery` — validation is mandatory

A new `seaglass/app/assist.py::merge_ghcp_parse(deterministic: ParsedQuery, raw: dict, contact_index, corpus_bounds) -> tuple[ParsedQuery, list[str]]` returning the merged query plus human-readable "what changed" strings for the banner. Rules, all fail-open in the spirit of `parse.py`'s "the parser must never be the single point of failure":

- **`people`** — never trusted directly. Each returned name goes through `contact_index.handle_ids_for_names(name, threshold=PEOPLE_FUZZY_THRESHOLD)` (the existing 85). Unresolvable names are **dropped**, not guessed. If GHCP found names but *none* resolve, the merged people filter falls back to the deterministic one (usually empty) — an LLM hallucinating "Sarah" must not fail a search closed, and `build_candidate_chunk_ids` **does** fail closed on an empty chat-id set.
- **`date_from`/`date_to`** — parsed strictly (`datetime.strptime("%Y-%m-%d")`), require `from <= to`, clamped to the corpus range read at startup (`SELECT MIN(start_ts), MAX(end_ts) FROM chunks` → here 2019-07 … now). Anything outside → discard the date filter entirely. Apply the same `DATE_PAD_DAYS = 3` padding `parse.py` uses, for consistency.
- **`has_media` / `is_group`** — accepted only as `true`/`false`; `null` means "keep deterministic value".
- **`semantic`** — accepted only if non-empty after strip and it shares ≥1 content word with the raw query (a cheap anti-drift/anti-hallucination guard). Otherwise keep the deterministic residual.
- **`expansions`** — capped at 5, alphanumeric single tokens only, deduped against the semantic text. **These never touch the dense arm.** They feed a *third* retrieval list (§5), fused by `rrf_fuse`, which already accepts an arbitrary number of ranked lists. This is the lowest-risk possible integration: if the expansion is garbage, RRF's rank-based fusion dilutes it rather than corrupting the score space.
- **Explicit UI filters always win.** If the user set a date range in the picker, neither `parse.py` nor GHCP may override it (§4.1).

---

## 4. UI/UX layout

Single-window, three-region layout. Plain HTML/CSS (system font stack `-apple-system`, `prefers-color-scheme` for dark mode, no framework, no build step, no npm).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  🔍 seaglass          [ search messages…                        ] [Search]   │  ← header, sticky
│                        ✨ Assist: (Off)(Auto)(Force)   ⚙                     │
├───────────────────────┬──────────────────────────────────────────────────────┤
│ FILTERS               │  Copilot read this as: photos from Sarah,            │  ← assist banner
│                       │  Nov 1 – Dec 31 2024   [Apply] [Dismiss]             │     (transient)
│ People                │──────────────────────────────────────────────────────│
│ [Sarah Chen ×][+ add] │  “lease paperwork”  ·  6 sessions · 41 msgs · 0.74s   │  ← result meta
│                       │  filters: person=Sarah Chen · 2024-11-01→2024-12-31   │
│ Conversation          │──────────────────────────────────────────────────────│
│ (•)Any ( )1:1 ( )Grp  │  ▸ Sarah Chen · Tue 12 Nov 2024 · score 2.41          │
│ [chat picker ▾]       │      Sarah  14:02  did you sign the lease addendum?   │
│                       │      Me     14:05  not yet, waiting on the deposit…   │
│ Date range            │      Sarah  14:06  📎 attachment                      │
│ [2024-11-01]→[2024-12-│      ── context ──  (dimmed, collapsed by default)    │
│ Presets: 7d 30d 1y ⌄  │      [Open full conversation ↗]                       │
│                       │                                                       │
│ ☐ Has media           │  ▸ Ben & Alex · Sat 09 Nov 2024 · score 1.88          │
│                       │      …                                                │
│ [Clear all filters]   │                                                       │
├───────────────────────┴──────────────────────────────────────────────────────┤
│ ● Ready · 17,525 chunks · 746 chats · loaded in 7.2s · ⚠ 1,204 msgs since     │  ← status bar
│   last index build · Copilot 1.0.79 available · Contacts ✓                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Filter controls — one per `ParsedQuery` capability

- **People (chips + autocomplete).** Typing queries `GET /api/contacts/suggest?q=` which calls `ContactIndex.fuzzy_match(q, threshold=60)` — a **lower** threshold than search's 85, because here the user picks from a visible list, so recall beats precision. Each suggestion shows display name + handle count. Selecting one creates a chip carrying the contact's **`handles` tuple** (not the display name), so the backend passes exact handle strings into `ParsedQuery.people_participant` and skips fuzzy matching entirely — an explicitly picked person is never fuzzy-guessed. Multiple chips OR together (matching `resolve_participant_chat_ids`'s `IN` semantics).
- **Group vs 1:1.** Three-way radio `Any / 1:1 only / Group only`. **Determination (verified against the live snapshot):** `chat.style` — `45` = 1:1 DM (1,057 chats), `43` = group (545 chats). Mirror `eval/harvest.py:97`'s exact rule, `int((style or 0) not in (45,))`, with `eval/harvest.py:125`'s fallback for chats missing a style row: `participant_count > 1` from `COUNT(DISTINCT handle_id) GROUP BY chat_id` on `chat_handle_join`. Both come from the startup chat-metadata cache, so filtering is a dict lookup, not a query.
- **Specific chat picker (bonus, cheap).** A dropdown over the 746 chat_ids actually present in `chunks`, labelled `chat.display_name` if non-empty, else the joined participant display names, else `chat_identifier`. Sets `ParsedQuery.chat_ids` directly.
- **Date range.** Two `<input type="date">` fields plus preset buttons (7d / 30d / 90d / 1y / All). Min/max attributes clamped to the corpus range from startup. Converted to unix seconds server-side; **no `DATE_PAD_DAYS` padding for explicit picks** — the user meant what they picked; padding is a heuristic softener for *inferred* dates only.
- **Has media.** Checkbox → `has_attachment = 1`.
- **Precedence, shown in the UI.** Explicit control > GHCP > `parse.py`. When `parse.py` infers a filter from typed text, the corresponding control **populates itself** with a subtle "from your query" styling, so the inference is visible and directly editable — this fixes the single most confusing thing about the current CLI, where you only learn `"may I borrow the car"` became a May date filter by reading the debug line.
- **Clear all** resets controls but not the search text.

### 4.2 Results list

Mirrors `format_search_result`'s existing session-grouped shape one-to-one, so no new backend concepts:

- One card per session, ordered by `score` desc, max 8 (`max_sessions`, adjustable in settings).
- **Card header:** chat title (from the metadata cache) + group/1:1 icon + participant count + `day` formatted as `Tue 12 Nov 2024` + score.
- **Body:** `hit_messages` at full opacity, `sender ?? "Me"` (`_resolve_sender` returns `None` for `is_from_me`), `HH:MM` from `ts`, `📎` when `has_attachment`. Query terms highlighted client-side (plain substring, case-insensitive, over the semantic residual's content words).
- **Context:** `context_messages` in a dimmed, collapsed `<details>` labelled "± surrounding context" — visually distinct because they carry **zero ranking weight** and mislabeling them as "why this matched" is exactly the confusion `hydrate.py`'s docstring warns about.
- **"Open full conversation ↗"** → `GET /api/conversation?chat_id&around_ts=<session's first hit ts>&limit=50`, backed by the *same* logic as `mcp_server.get_conversation` (lifted into shared code, §7), opening a right-hand drawer with infinite scroll in both directions.
- **Empty state:** distinguish "0 fused results" from "pre-filter returned nothing." If `build_candidate_chunk_ids` returned an empty set, say so explicitly — *"No messages match your filters (person: Sarah Chen · Nov 2024). Try removing a filter."* — with one-click removal chips. The fail-closed participant filter is the most likely source of surprising zero-result searches.
- **Text selection:** normal browser selection/copy. A per-message "copy" affordance on hover.
- Optional **Redact** toggle in settings → passes `redact=True` to `format_search_result` (useful for screenshots).

### 4.3 Status / diagnostics bar

- State pill (`Ready` / `Loading` / `Degraded`) + total warmup time.
- `n_chunks`, `n_vectors`, distinct chat count — from an `index_status`-equivalent.
- **Index staleness**, computed at startup and on demand: `SELECT MAX(end_ts) FROM chunks` (index) vs `SELECT MAX(date) FROM im.message` (chat.db, through `apple_to_unix`), plus a count of messages newer than the index high-water mark. Rendered as *"⚠ 1,204 messages since last index build"* with a tooltip explaining that re-indexing is currently the only update path (there's no `index/sync.py` yet). **Do not offer a "rebuild now" button in v1** — a full build is a long, memory-heavy job that has no business running inside the search app's process; show the exact `python -m seaglass.cli build …` command with a copy button instead.
- GHCP status; Contacts status; last query latency (with a per-stage breakdown behind a "⌥" debug expander: parse / prefilter / dense / sparse / fuse / rerank / hydrate — invaluable for the same performance work the architecture doc documents).

---

## 5. Data flow / API surface

### 5.1 In-process direct imports — not an MCP client

**Recommendation: the app imports and calls the pipeline functions directly, in-process.** Reasons:

1. **Memory.** Warm models are ~300 MB RSS idle, ~566 MB after the first query (architecture doc §9). Talking to `seaglass-mcp` as a client means a *second* process with its own copy → ~1.1 GB for one user's searches, and the sub-process would still cold-start its own models.
2. **Lifecycle mismatch.** MCP-over-stdio is one server process per client session, spawned and owned by the client. The app's entire premise is a long-lived process that owns its models — the opposite lifecycle.
3. **Fidelity loss.** The MCP tool surface is deliberately narrowed for an LLM consumer (`search_messages(query, max_sessions, redact)`). The app needs `is_group`, explicit `chat_ids`, exact handle lists, per-stage timings, and the pre/post-GHCP parse — all things that would require widening the MCP tool signature purely for a non-agent consumer, degrading it as an agent tool.
4. `cli.py` already proves direct in-process wiring works end to end; the app is that same wiring with a warm cache and an HTTP surface.

### 5.2 Internal API — `seaglass/app/engine.py`

```python
@dataclass
class SearchFilters:                      # everything the UI can set explicitly
    people_handles: list[str] = []        # exact handle strings from picked chips
    people_names: list[str] = []          # free-typed names → fuzzy-resolved server-side
    is_group: bool | None = None
    chat_ids: list[int] | None = None
    date_from: float | None = None        # unix seconds, already timezone-resolved
    date_to: float | None = None
    has_media: bool | None = None

@dataclass
class SearchOptions:
    max_sessions: int = 8
    fused_top_k: int = 50
    rerank: bool = True
    redact: bool = False
    expansions: list[str] = []            # from GHCP; feeds a third sparse arm only

class SearchEngine:
    def warmup(self, progress: Callable[[str, str, float], None]) -> None: ...
    def health(self) -> dict: ...
    def status(self) -> dict: ...                     # index_status + staleness
    def suggest_contacts(self, q: str, limit: int = 10) -> list[dict]: ...
    def suggest_chats(self, q: str, limit: int = 20) -> list[dict]: ...
    def search(self, text: str, filters: SearchFilters,
               options: SearchOptions) -> SearchResponse: ...
    def conversation(self, chat_id: int, around_ts: float | None,
                     limit: int = 50) -> dict: ...
```

`SearchEngine.search` is a thin, ordered composition of *existing* functions — nothing is reimplemented:

```
parse_query(text, contact_index)                          # seaglass.search.parse
  → apply_filters(parsed, filters)                        # NEW, app-local: UI overrides win
  → build_candidate_chunk_ids(index_con, parsed, chat_con)# seaglass.search.retrieve
  → retrieve(index_con, parsed, embed_model, chat_con=…, fused_top_k=…)
  → rerank_candidates(index_con, parsed.semantic, fused, reranker)
  → aggregate_sessions(ranked, max_sessions=…)
  → expand_sessions(index_con, sessions)
  → hydrate_sessions(index_con, chat_con, sessions, contact_index=…)
  → format_search_result(hydrated, max_sessions=…, redact=…)
  → + decorate each session with chat metadata from the startup cache
  → + attach {timings, effective_filters, parse_source} for the UI
```

### 5.3 Required upstream changes (all additive and backward-compatible)

| File | Change | Compatibility |
|---|---|---|
| `search/parse.py` | Add `is_group: Optional[bool] = None` and `chat_ids: Optional[List[int]] = None` to `ParsedQuery`. `parse_query` never sets them. | Defaults preserve every existing behaviour; `mcp_server.py`, `cli.py`, `eval/score.py` and `tests/test_parse.py` unaffected. |
| `search/retrieve.py` | Add `resolve_group_chat_ids(chat_con, is_group) -> Set[int]` (using `chat.style not in (45,)` with the participant-count fallback). Extend `build_candidate_chunk_ids` to intersect `is_group` and `chat_ids` alongside the existing participant path. | Both new fields default to `None`; existing callers hit the identical code path. |
| `search/retrieve.py` | Optional `extra_sparse_queries: Sequence[str] = ()` on `retrieve()` — each runs its own `sparse_search` and is passed as an additional list to `rrf_fuse` (which already takes N lists). | Default empty → byte-identical behaviour. |
| `search/retrieve.py` | **Large-candidate-set handling** (see §8 risk R1). | New code path only above a threshold. |
| `eval/ghcp_client.py` | Move to `seaglass/llm/ghcp.py`; leave a re-export shim. Add `call_ghcp_json_object`, `detect_ghcp`, `bin_path`/`timeout_s` params. | `tests/test_generate.py` imports keep working. |
| `mcp_server.py` | Extract `get_conversation`'s body into `seaglass/search/conversation.py::fetch_conversation(chat_con, chat_id, around_ts, limit, contact_index)`; the MCP tool becomes a 3-line wrapper. | Behaviour-identical; `tests/test_mcp_server.py` must still pass unchanged. |
| `pyproject.toml` | Promote `fastapi`/`uvicorn` from transitive to explicit deps; add `[project.scripts] seaglass-app = "seaglass.app.__main__:main"`; add `[tool.setuptools.package-data] "seaglass.app" = ["static/*"]`. | |

### 5.4 HTTP endpoints

All under `/api`, all Bearer-token-guarded, all JSON.

```
GET  /api/health         → {state, steps[], progress, elapsed_s, warnings[], error, ghcp{}}
GET  /api/status         → {n_chunks, n_vectors, n_chats, index_db, chat_db,
                            most_recent_chunk_ts, chat_db_max_ts, n_messages_since_index,
                            contacts_available, hydration_available, warmup_elapsed_s}
POST /api/search         → body {query, filters{}, options{}, assist:"off"|"auto"|"force",
                                  request_id}
                            resp {request_id, n_sessions, n_results, confidence, sessions[],
                                  effective_filters{}, parse_source:"deterministic",
                                  timings{}, elapsed_s, assist_token: str|null}
GET  /api/assist/{token} → long-poll (≤25 s). 200 {status:"ready", parse{}, changes[],
                             confidence} | {status:"unchanged"} | {status:"unavailable",
                             reason} ; 204 while pending
POST /api/search/apply-assist → body {assist_token, request_id} → same shape as /api/search
                                 with parse_source:"ghcp"
GET  /api/conversation?chat_id=&around_ts=&limit=  → {chat_id, title, is_group,
                                                      participants[], n_messages, messages[]}
GET  /api/contacts/suggest?q=&limit=  → [{display_name, handles[], n_handles, score}]
GET  /api/chats/suggest?q=&limit=     → [{chat_id, title, is_group, participant_count,
                                          n_chunks, last_ts}]
GET  /api/config  /  POST /api/config → assist mode, max_sessions, redact, paths,
                                        copilot_bin  (persisted to ~/.seaglass/config.json)
```

Long-polling for the assist result (rather than SSE) keeps the client to plain `fetch` with no reconnection/heartbeat logic, and needs no new dependency. `sse-starlette` is available if streaming per-stage progress later proves worthwhile.

---

## 6. Relationship to the existing MCP server

**This app is strictly additive. `seaglass-mcp` is not modified in behaviour, not deprecated, and not proxied through.**

| | `seaglass-mcp` | `seaglass-app` |
|---|---|---|
| Consumer | An LLM agent (Grogu / Copilot CLI) | A human |
| Transport | MCP over stdio; **stdout is the protocol** | HTTP on loopback |
| Lifecycle | Spawned per client session by ghcp, exits with it; lazy model load | Long-lived, user-launched; **eager** model load |
| Output | Structured payload for a model to reason over; `confidence` hints; `redact` for forwarding | Rendered UI with progressive disclosure and interactive filters |
| Filters | Inferred from `query` text only | Inferred **and** explicit UI controls |
| GHCP | Never (would be an agent calling an agent) | Optional speculative assist |
| Config | `SEAGLASS_INDEX_DB` / `SEAGLASS_CHAT_DB` env from `~/.copilot/mcp-config.json` | `~/.seaglass/config.json`, env override |

They **share the pipeline modules** (`parse`/`retrieve`/`rank`/`rerank`/`hydrate`/`format` and the new `conversation.py`) — which is a feature: retrieval-quality work benefits both, and the golden-set eval in `eval/score.py` continues to measure the real path for both. They **share nothing at runtime**: separate processes, separate SQLite connections, separate model instances.

Concrete non-interference guarantees to hold the implementation to:
- The app opens `chat.db` read-only via `connect_readonly` and `index.db` with `create=False`; it **never writes to `index.db`** (all app state goes in `~/.seaglass/app.db`). Two readers on a WAL database are fine.
- No changes to `~/.copilot/mcp-config.json`.
- No changes to the `seaglass-mcp` entry point or tool signatures.
- `tests/test_mcp_server.py` must pass unchanged after the `conversation.py` extraction — that's the regression gate.
- Both can run simultaneously (~566 MB each; fine on Apple Silicon with normal RAM).

---

## 7. File / module layout

**Place it at `seaglass/app/`, not a top-level `app/`.** It must `import seaglass.search.*` and ship via the same `pip install -e .`; an in-package location gets the console-script entry point, the venv, and `[tool.setuptools.packages.find] include = ["seaglass*"]` for free, matching how `seaglass/eval/` (also not core library code) is already organised.

```
seaglass/app/
  __init__.py              Exports SearchEngine, create_app. No side effects, no heavy imports.
  __main__.py              Launcher / `main()` for the `seaglass-app` script. Arg parsing
                           (--index-db, --chat-db, --port, --no-browser, --assist),
                           config load/merge, single-instance lockfile (~/.seaglass/app.lock:
                           {pid,port,token}; if a live pid holds it, open that URL and exit),
                           port selection, token minting, uvicorn thread start, warmup
                           dispatch, webbrowser.open, SIGINT/SIGTERM cleanup.
  config.py                AppConfig dataclass + load/save for ~/.seaglass/config.json;
                           env-var overrides (SEAGLASS_INDEX_DB, SEAGLASS_CHAT_DB,
                           SEAGLASS_COPILOT_BIN, SEAGLASS_APP_MEMORY_INDEX); path validation
                           with actionable error messages. Also owns ~/.seaglass/aliases.json.
  warmup.py                WarmupState (thread-safe step list + state machine) and the
                           ordered §2.2 step 1-12 sequence. The ONLY place that knows the
                           startup order. Includes the full dummy-search kernel-warm step.
  engine.py                SearchEngine (§5.2). Owns index_con, chat_con, embedding model,
                           reranker, contact_index, chat metadata cache, corpus bounds.
                           Composes the existing pipeline functions. Per-stage timings.
                           No HTTP, no GHCP — pure, unit-testable against conftest fixtures.
  filters.py               SearchFilters dataclass; apply_filters(ParsedQuery, SearchFilters)
                           implementing the explicit-beats-inferred precedence rule;
                           handle resolution for free-typed names; date/tz conversion.
  chatmeta.py              ChatMetadataCache: builds and serves {chat_id -> is_group, title,
                           participants, participant_count} from chat.db at startup.
                           SINGLE home of the style==45 group rule outside imessage/source.py.
  assist.py                GHCP assist layer: build_prompt(), the AssistResult dataclass,
                           merge_ghcp_parse() with all §3.7 validation, the auto-trigger
                           heuristic (should_assist()), the circuit breaker, and the
                           ghcp_cache table in ~/.seaglass/app.db.
  server.py                create_app(engine, warmup_state, config, token) -> FastAPI.
                           Auth/Origin/Host middleware, all §5.4 routes, pipeline
                           ThreadPoolExecutor(1) + ghcp ThreadPoolExecutor(1), assist-token
                           registry, static mount, JSON error shaping.
  static/
    index.html             Whole UI. Loading screen + search shell + results + drawer.
    app.js                 Vanilla ES modules, no bundler: token handling, health polling
                           and loading screen, filter state <-> URL hash, contact/chat
                           autocomplete (debounced), search dispatch + request_id
                           supersession, assist long-poll + banner, results rendering,
                           conversation drawer, keyboard shortcuts (⌘K focus, ⌘⏎ search,
                           Esc close drawer, ↑/↓ navigate sessions).
    app.css                System font stack, light/dark via prefers-color-scheme.

seaglass/llm/
  __init__.py
  ghcp.py                  Moved from eval/ghcp_client.py; + call_ghcp_json_object,
                           detect_ghcp, bin_path/timeout_s params.
seaglass/eval/
  ghcp_client.py           Reduced to a re-export shim from seaglass.llm.ghcp.
seaglass/search/
  conversation.py          fetch_conversation(...) lifted verbatim from mcp_server.py.

tests/
  test_app_engine.py       SearchEngine end-to-end on the synthetic conftest fixtures with
                           FakeEmbeddingModel + a stub reranker. Group/1:1, chat_ids, date,
                           media, person filters; large-candidate-set path.
  test_app_filters.py      apply_filters precedence; explicit filters override parse.py.
  test_app_assist.py       merge_ghcp_parse validation: hallucinated names dropped,
                           out-of-range dates discarded, empty-semantic rejected,
                           expansion sanitisation, cache hit/miss, circuit breaker.
                           No live copilot calls (mock call_ghcp_json_object).
  test_app_server.py       FastAPI TestClient: auth rejection without/with bad token,
                           Origin rejection, /api/health during warmup, search round trip.
  test_chatmeta.py         style 43/45 classification + participant-count fallback.

docs/
  architecture.html        Add a §12 "Desktop app" section + update the §1 privacy note
                           to cover the optional GHCP query-text egress.
README.md                  Add app usage; update the privacy paragraph.
development-plans/
  ADDENDUM.md              New section recording this design and its measured numbers.
```

**Suggested build order** (each step independently verifiable):
1. `chatmeta.py` + `search/retrieve.py` group/chat_ids extension + `search/conversation.py` extraction — pure library work, covered by tests, MCP regression gate green.
2. `engine.py` + `warmup.py` + a `--headless --query` mode in `__main__.py` — prove eager loading and measure real warm latency from the terminal before any UI exists.
3. `server.py` + `static/` — the loading screen first, then search, then filters, then the conversation drawer.
4. `assist.py` + `llm/ghcp.py` — last, since everything must work correctly without it.

---

## 8. Open questions, risks, and deliberate punts

**R1 — SQLite variable limits on large candidate sets (highest technical risk).** `build_candidate_chunk_ids` returns a chunk-id *set* that `dense_search`/`sparse_search` inline as `IN (?,?,…)` parameters. Today's filters are narrow, so this has never bitten. But "1:1 only" matches 1,057 of 1,602 chats — plausibly >10,000 of the 17,525 chunks — producing a 10k-parameter query. Modern SQLite's `SQLITE_MAX_VARIABLE_NUMBER` is 32,766 so it may *work*, but it's slow, fragile, and a larger corpus breaks it. **Plan:** add `CANDIDATE_INLINE_LIMIT` (~2,000) to `retrieve.py`; above it, switch to **post-filtering** — run dense/sparse unconstrained with a larger `top_k` (e.g. 3×), then drop non-candidates before fusion. Justification: a filter matching >10% of the corpus is weakly selective, so pre-filtering buys little; post-filtering is trivially correct and has no variable limit. The theoretically nicer temp-table + `rowid IN (SELECT …)` approach is **unverified against sqlite-vec's `vec0` virtual table** — its `xBestIndex` may only accept a literal value list for `rowid IN`. **Spike this before relying on it**; ship post-filtering regardless as the safe path. This is the one item I'd validate first.

**R2 — Warmup UX on a cold OS page cache.** The ~6–10 s estimate assumes HF model weights are already on disk (they are) and the OS cache is lukewarm. A true cold boot (first launch after reboot, 656 MB chat.db untouched) could reach 15 s+. Mitigation: the step-list progress screen makes it legible rather than broken-feeling; measure real cold-boot numbers once and record them in ADDENDUM, as this project already does for every other latency claim. If it's genuinely bad, consider deferring the reranker load until after the UI is interactive (the first search would then eat ~2 s once) — but only if measurement demands it.

**R3 — GHCP's value is asserted, not measured.** Everything in §3.1 is a reasoned hypothesis. The project already has the instrument to settle it: `eval/score.py` against `golden.jsonl` (n=32, recall@final 0.41, 95% CI [0.45, 0.77] on recall@50 — wide). **Punt deliberately:** build the assist layer, then run a GHCP-on vs GHCP-off scoring pass and record the delta. If it doesn't move recall, the honest outcome is to leave it off by default permanently, or keep it purely as the "my search failed, help me rephrase" escape hatch. The n=32 golden set is likely too small to resolve a modest effect — acknowledge that rather than over-reading it.

**R4 — Copilot binary discovery from a GUI-launched process.** Confirmed live: this machine's `copilot` sits behind an fnm multishell path that only exists in an fnm-initialised shell. Launched from Finder/Dock (or eventually a .app), `shutil.which` will find nothing. §3.4's three-stage detection plus an explicit `copilot_bin` config setting mitigates it, but the fnm path contains a PID (`59915_1786564114461`) and is therefore **not stable across shell sessions** — persisting a resolved fnm shim path into config would go stale. **Rule: persist only `os.path.realpath()` of the discovered binary**, and re-probe if the persisted path no longer exists.

**R5 — Privacy-boundary regression.** Adding any cloud call to a project whose README promises full on-device operation is a genuine change in kind, not degree. Handled by: off-by-default, explicit one-time consent, query-text-only egress enforced in code, README + architecture-doc updates. Flagging it here because it deserves an explicit user decision, not a silent feature.

**R6 — Packaging: punting entirely for now.** v1 is `pip install -e .` + `seaglass-app` (or `python -m seaglass.app`) from the existing venv, which already holds Full Disk Access. **No py2app/PyInstaller, no signing, no notarization.** Rationale: bundling MLX + Metal shaders + PyObjC frameworks + a 600 MB-class HF cache is a substantial project in itself; unsigned .app bundles need Gatekeeper right-click-open; and Full Disk Access / Contacts TCC permissions must be re-granted to the *bundle identity*, silently breaking things in confusing ways. The dev-mode launcher costs one terminal command and sidesteps all of it. Revisit only if the tool is shared with someone else. Middle ground if launching from a terminal grates: a hand-written `~/Applications/seaglass.app` stub containing only an `Info.plist` and a shell script that execs the venv binary — 10 lines, gets a Dock icon, keeps the TCC grants of the underlying binary.

**R7 — pywebview deferral.** Needs `pywebview` + `pyobjc-framework-WebKit` (Cocoa and pyobjc-core are already present). Unverified on Python 3.14.6 — a very new interpreter — and it introduces the standard "webview window owns the main thread while uvicorn runs in a thread" ordering hazard. Deferred to v1.5 precisely because the browser path has zero such unknowns.

**R8 — Contacts autocomplete quality.** `ContactIndex.fuzzy_match` is `rapidfuzz.WRatio` over display names with `limit=5` hardcoded. For an autocomplete needing ~10 prefix-friendly suggestions, `WRatio` at threshold 60 may rank oddly (it's a similarity scorer, not a prefix matcher). Likely fix: add a `limit` parameter and, in the app layer, merge a cheap `startswith`/substring pass ahead of the fuzzy results. Minor, but it will need a tuning pass against the real 2,331-handle roster.

**R9 — Also punting deliberately:** search history / saved searches; result export; attachment thumbnail rendering (files may be iCloud-offloaded — `attachment_retry` exists for a reason); incremental index sync from the app (there is no `index/sync.py` yet); multi-window; any "from X" (sender) vs "with X" (participant) distinction — `retrieve.py` documents this simplification and the app inherits it, but the People filter's tooltip should say plainly *"matches who was in the conversation, not who sent the message."*

**Open questions needing a user call:**
1. Is the GHCP query-text egress acceptable at all (R5)? If not, the whole assist layer is dead and the app is simpler.
2. Default assist mode once opted in — `Auto` (heuristic-triggered) or `Off` until explicitly forced per search? I recommend `Auto`.
3. Should the app auto-launch at login (a launchd `LaunchAgent`) so the ~7 s warmup is paid at boot rather than at first use? That's the logical endpoint of the eager-loading requirement, and it's ~20 lines of plist — but it means ~566 MB resident permanently. Worth doing *after* the app proves useful, not before.
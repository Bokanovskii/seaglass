from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from seaglass.app.chatmeta import ChatMetadataCache
from seaglass.app.filters import SearchFilters, apply_filters
from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import APPLE_EPOCH_UNIX, NS_VS_S_THRESHOLD, connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.conversation import fetch_conversation
from seaglass.search.format import format_search_result
from seaglass.search.hydrate import hydrate_sessions
from seaglass.search.parse import ParsedQuery, parse_query
from seaglass.search.rank import (
    RERANK_TOP_K,
    aggregate_sessions,
    expand_sessions,
    order_sessions,
    rerank_candidates,
    select_reranked_head,
)
from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import build_candidate_chunk_ids, exact_phrase_chunk_ids, retrieve


@dataclass
class SearchOptions:
    max_sessions: int = 8
    fused_top_k: int = 50
    rerank: bool = True
    redact: bool = False
    expansions: list[str] = field(default_factory=list)
    # Pagination. `offset` skips already-shown sessions; the ranked chunk
    # list is cached per query so paging never re-runs the models.
    offset: int = 0


class SearchEngine:
    def __init__(self, index_db: str, chat_db: str | None = None, memory_index: bool = False, chat_db_source: str | None = None):
        self.index_db = index_db
        self.chat_db = chat_db
        self.chat_db_source = chat_db_source
        self.memory_index = memory_index
        self.index_con: sqlite3.Connection | None = None
        self.chat_con: sqlite3.Connection | None = None
        # Cached read-only handle on the *live* chat.db (see status()).
        self._live_chat_con: sqlite3.Connection | None = None
        self._live_chat_lock = threading.RLock()
        self.embedding_model: EmbeddingModel | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.contact_index: ContactIndex | None = None
        self.chatmeta: ChatMetadataCache | None = None
        self.corpus_bounds: tuple[float, float] = (0.0, 0.0)
        self.warnings: list[str] = []
        self.last_status: dict = {}
        # Ranked-chunk cache so "load more" costs a regroup, not another
        # embed + cross-encoder pass (~0.5-1.2s). Keyed by everything that
        # can change the ranking; small because entries hold whole
        # candidate lists and only the newest queries are ever paged.
        self._page_cache: 'OrderedDict[tuple, list]' = OrderedDict()
        self._page_cache_lock = threading.RLock()

    def warmup(self, progress: Callable[[str], object]) -> None:
        # A rebuild can prune or renumber chunk ids, so any cached ranking
        # from before this (re)warm is stale.
        self.invalidate_page_cache()
        with progress('import_runtime'):
            __import__('sqlite_vec')
            __import__('mlx')
        with progress('open_index'):
            self.index_con = open_index_db(Path(self.index_db), check_same_thread=False, create=False)
            if self.memory_index:
                mem = sqlite3.connect(':memory:', check_same_thread=False)
                self.index_con.backup(mem)
                self.index_con.close()
                self.index_con = mem
        with progress('configure_index'):
            self.index_con.execute('PRAGMA cache_size=-262144')
            self.index_con.execute('PRAGMA mmap_size=268435456')
        with progress('read_meta'):
            self.index_con.execute("SELECT value FROM meta WHERE key = 'int8_absmax'").fetchone()
            row = self.index_con.execute('SELECT MIN(start_ts), MAX(end_ts) FROM chunks').fetchone()
            self.corpus_bounds = (float(row[0] or 0.0), float(row[1] or 0.0))
        with progress('open_chat'):
            if self.chat_db:
                self.chat_con = connect_readonly(Path(self.chat_db))
                self.chat_con = self.chat_con if getattr(self.chat_con, 'execute', None) else self.chat_con
        with progress('load_contacts'):
            try:
                self.contact_index = ContactIndex.load()
            except ContactsUnavailableError as exc:
                self.warnings.append('Contacts unavailable — sender names will show raw handles')
                self.contact_index = None
        with progress('build_chatmeta'):
            present_chat_ids = {row[0] for row in self.index_con.execute('SELECT DISTINCT chat_id FROM chunks')}
            self.chatmeta = ChatMetadataCache.build(self.chat_con, present_chat_ids, contact_index=self.contact_index) if self.chat_con else ChatMetadataCache({})
        with progress('warm_sqlite'):
            self.index_con.execute('SELECT COUNT(*), SUM(LENGTH(body_semantic)) FROM chunks').fetchone()
            self.index_con.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()
            self.index_con.execute('SELECT COUNT(*) FROM chunks_fts').fetchone()
        with progress('load_embedding_model'):
            self.embedding_model = EmbeddingModel()
            self.embedding_model.embed(['warmup'])
        with progress('load_reranker'):
            self.reranker = CrossEncoderReranker()
            self.reranker.score([('warmup', 'warmup')])
        with progress('dummy_search'):
            self.search('dinner plans', SearchFilters(), SearchOptions(max_sessions=1))

    def health(self) -> dict:
        return {'warnings': list(self.warnings), 'corpus_bounds': self.corpus_bounds}

    def status(self) -> dict:
        if self.index_con is None:
            return {
                'n_chunks': 0,
                'n_vectors': 0,
                'n_chats': 0,
                'index_db': self.index_db,
                'chat_db': self.chat_db,
                'most_recent_chunk_ts': None,
                'chat_db_max_ts': None,
                'n_messages_since_index': 0,
                'contacts_available': False,
                'hydration_available': False,
                'index_ready': False,
            }
        n_chunks = self.index_con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        n_vectors = self.index_con.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()[0]
        n_chats = self.index_con.execute('SELECT COUNT(DISTINCT chat_id) FROM chunks').fetchone()[0]
        most_recent_chunk_ts = self.index_con.execute('SELECT MAX(end_ts) FROM chunks').fetchone()[0]
        chat_db_max_ts, newer_messages = self._live_chat_freshness(most_recent_chunk_ts)
        return {
            'n_chunks': n_chunks,
            'n_vectors': n_vectors,
            'n_chats': n_chats,
            'index_db': self.index_db,
            'chat_db': self.chat_db,
            'most_recent_chunk_ts': most_recent_chunk_ts,
            'chat_db_max_ts': chat_db_max_ts,
            'n_messages_since_index': newer_messages,
            'contacts_available': self.contact_index is not None,
            'hydration_available': self.chat_con is not None,
            'index_ready': True,
        }

    # `message.date` is Apple-epoch, in seconds on older macOS and
    # nanoseconds on Big Sur+ (see imessage/source.py apple_to_unix). This
    # SQL mirrors that per-row magnitude heuristic exactly -- keep the
    # threshold in sync with NS_VS_S_THRESHOLD.
    _LIVE_MAX_TS_SQL = f"""
        SELECT MAX(CASE WHEN m.date > {NS_VS_S_THRESHOLD} THEN m.date / 1e9 ELSE m.date END)
        FROM im.message m
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE m.associated_message_type = 0
    """
    _LIVE_NEWER_COUNT_SQL = f"""
        SELECT COUNT(DISTINCT m.ROWID)
        FROM im.message m
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE m.associated_message_type = 0
          AND (CASE WHEN m.date > {NS_VS_S_THRESHOLD} THEN m.date / 1e9 ELSE m.date END) > ?
    """

    def _live_chat_connection(self) -> sqlite3.Connection | None:
        """Lazily open (and cache) a read-only connection to the *live*
        chat.db. Cached because `status()` is polled by the UI roughly once
        a minute and ATTACH + schema assertion is not free.

        Returns None when the live db is missing or unreadable (e.g. Full
        Disk Access not granted) -- callers degrade gracefully.
        """
        with self._live_chat_lock:
            if self._live_chat_con is not None:
                return self._live_chat_con
            live_path = self.chat_db_source or self.chat_db
            if not live_path or not Path(live_path).exists():
                return None
            try:
                self._live_chat_con = connect_readonly(Path(live_path))
            except Exception:  # noqa: BLE001 - missing perms/schema drift must not break /api/status
                self._live_chat_con = None
            return self._live_chat_con

    def _close_live_chat_connection(self) -> None:
        with self._live_chat_lock:
            if self._live_chat_con is not None:
                try:
                    self._live_chat_con.close()
                except sqlite3.Error:
                    pass
                self._live_chat_con = None

    def _live_chat_freshness(self, most_recent_chunk_ts: float | None) -> tuple[float | None, int]:
        """(max message ts in the live chat.db as unix seconds, count of
        live messages newer than the newest indexed chunk)."""
        con = self._live_chat_connection()
        if con is None:
            return None, 0
        try:
            with self._live_chat_lock:
                raw_max = con.execute(self._LIVE_MAX_TS_SQL).fetchone()[0]
                chat_db_max_ts = float(raw_max) + APPLE_EPOCH_UNIX if raw_max is not None else None
                newer = 0
                if chat_db_max_ts is not None and most_recent_chunk_ts:
                    newer = con.execute(
                        self._LIVE_NEWER_COUNT_SQL,
                        (float(most_recent_chunk_ts) - APPLE_EPOCH_UNIX,),
                    ).fetchone()[0]
            return chat_db_max_ts, int(newer)
        except Exception:  # noqa: BLE001 - never break /api/status on a flaky read
            self._close_live_chat_connection()
            return None, 0

    def suggest_contacts(self, q: str, limit: int = 10) -> list[dict]:
        if self.contact_index is None:
            return []
        starts = [c for c in getattr(self.contact_index, '_contacts', []) if c.display_name.lower().startswith(q.lower())]
        fuzzy = self.contact_index.fuzzy_match(q, threshold=60.0)
        seen = set()
        suggestions = []
        for contact in starts + fuzzy:
            if contact.identifier in seen:
                continue
            seen.add(contact.identifier)
            suggestions.append({'display_name': contact.display_name, 'handles': list(contact.handles), 'n_handles': len(contact.handles), 'score': 1.0 if contact in starts else 0.5})
            if len(suggestions) >= limit:
                break
        return suggestions

    def suggest_chats(self, q: str, limit: int = 20) -> list[dict]:
        if self.chatmeta is None:
            return []
        matches = []
        query = q.lower().strip()
        for meta in self.chatmeta.all():
            if not query or query in meta.title.lower() or any(query in p.lower() for p in meta.participants):
                n_chunks = self.index_con.execute('SELECT COUNT(*) FROM chunks WHERE chat_id = ?', (meta.chat_id,)).fetchone()[0]
                last_ts = self.index_con.execute('SELECT MAX(end_ts) FROM chunks WHERE chat_id = ?', (meta.chat_id,)).fetchone()[0]
                matches.append({'chat_id': meta.chat_id, 'title': meta.title, 'is_group': meta.is_group, 'participant_count': meta.participant_count, 'n_chunks': n_chunks, 'last_ts': last_ts})
                if len(matches) >= limit:
                    break
        return matches

    def search(self, text: str, filters: SearchFilters, options: SearchOptions) -> dict:
        timings = {}
        t0 = time.time()
        parse_started = time.time()
        parsed = parse_query(text, contact_index=self.contact_index)
        parsed = apply_filters(parsed, filters, contact_index=self.contact_index)
        timings['parse'] = round(time.time() - parse_started, 4)

        prefilter_started = time.time()
        candidate_ids = build_candidate_chunk_ids(self.index_con, parsed, chat_con=self.chat_con)
        timings['prefilter'] = round(time.time() - prefilter_started, 4)

        retrieve_started = time.time()
        fused = retrieve(
            self.index_con,
            parsed,
            self.embedding_model,
            chat_con=self.chat_con,
            fused_top_k=options.fused_top_k,
            extra_sparse_queries=options.expansions,
        )
        timings['retrieve'] = round(time.time() - retrieve_started, 4)

        if not fused:
            return {
                'request_id': None,
                'n_sessions': 0,
                'n_results': 0,
                'confidence': 'none',
                'sessions': [],
                'offset': max(0, options.offset),
                'total_sessions': 0,
                'has_more': False,
                'next_offset': max(0, options.offset),
                'effective_filters': _effective_filters(parsed),
                'parse_source': 'deterministic',
                'timings': timings,
                'elapsed_s': round(time.time() - t0, 2),
                'candidate_count': 0 if candidate_ids is None else len(candidate_ids),
            }

        if options.rerank:
            rerank_started = time.time()
            cache_key = self._page_cache_key(text, parsed, options)
            scored = self._cached_ranking(cache_key)
            if scored is None:
                # top_k=None scores and keeps every fused candidate; the
                # page-1 cut is applied below, so paging draws from one pass.
                scored = rerank_candidates(
                    self.index_con, parsed.semantic, fused, self.reranker, top_k=None
                )
                self._store_ranking(cache_key, scored)
                timings['rerank_cached'] = False
            else:
                timings['rerank_cached'] = True
            ranked = select_reranked_head(scored, top_k=RERANK_TOP_K)
            timings['rerank'] = round(time.time() - rerank_started, 4)
        else:
            ranked = []
            scored = []
        aggregate_started = time.time()
        # Which candidates literally contain the query text -- a much
        # stronger signal than semantic similarity for the very common
        # "find the message where someone said X" search.
        lexical_ids = exact_phrase_chunk_ids(
            self.index_con, parsed.semantic, [r.chunk_id for r in fused]
        )
        ordered = self._ordered_sessions(ranked, scored, options, lexical_ids)
        offset = max(0, options.offset)
        sessions = ordered[offset: offset + options.max_sessions]
        total_sessions = len(ordered)
        expand_sessions(self.index_con, sessions)
        timings['aggregate_expand'] = round(time.time() - aggregate_started, 4)

        hydrate_started = time.time()
        if self.chat_con is not None:
            hydrated = hydrate_sessions(self.index_con, self.chat_con, sessions, contact_index=self.contact_index)
            payload = format_search_result(hydrated, max_sessions=options.max_sessions, redact=options.redact)
        else:
            payload = {'n_sessions': 0, 'n_results': 0, 'confidence': 'unhydrated', 'sessions': []}
        timings['hydrate'] = round(time.time() - hydrate_started, 4)

        for session in payload.get('sessions', []):
            meta = self.chatmeta.get(session['chat_id']) if self.chatmeta else None
            if meta:
                session['title'] = meta.title
                session['is_group'] = meta.is_group
                session['participant_count'] = meta.participant_count
                session['participants'] = list(meta.participants)

        payload.update({
            'offset': offset,
            'total_sessions': total_sessions,
            'has_more': offset + options.max_sessions < total_sessions,
            'next_offset': offset + options.max_sessions,
            'effective_filters': _effective_filters(parsed),
            'parse_source': 'deterministic',
            'timings': timings,
            'elapsed_s': round(time.time() - t0, 2),
            'candidate_count': 0 if candidate_ids is None else len(candidate_ids),
        })
        return payload

    @staticmethod
    def _ordered_sessions(ranked, scored, options: SearchOptions, lexical_ids=None) -> list:
        """Page 1 comes from the usual RERANK_TOP_K cut; later pages are the
        rest of the scored candidates in score order, appended.

        Built additively on purpose. Simply aggregating all the scored
        candidates at once would give more pages, but it would also change
        which sessions land on page 1 (more chunks means more sessions
        competing for the recency reservation), so the ordinary
        no-pagination result would silently differ from the tuned pipeline.
        Appending instead guarantees page 1 is byte-identical to the
        unpaginated search while later pages keep going.
        """
        head = order_sessions(
            ranked, head_size=options.max_sessions, lexical_chunk_ids=lexical_ids
        )
        if not scored or len(scored) <= len(ranked):
            return head
        seen = {(s.chat_id, s.day) for s in head}
        # recent_slots=0: the recency guarantee has already been honoured on
        # page 1, and re-applying it here would pull recent-but-weak matches
        # ahead of stronger ones deep in the list.
        tail = order_sessions(
            scored, head_size=0, recent_slots=0, lexical_chunk_ids=lexical_ids
        )
        return head + [s for s in tail if (s.chat_id, s.day) not in seen]

    _PAGE_CACHE_MAX = 8

    @staticmethod
    def _page_cache_key(text: str, parsed: ParsedQuery, options: SearchOptions) -> tuple:
        """Everything that can change the ranking -- but NOT `offset`, which
        is precisely what we want to serve from one cached ranking."""
        return (
            text,
            parsed.semantic,
            tuple(parsed.people_participant),
            parsed.date_from,
            parsed.date_to,
            parsed.has_media,
            options.fused_top_k,
            tuple(options.expansions),
        )

    def _cached_ranking(self, key: tuple):
        with self._page_cache_lock:
            ranking = self._page_cache.get(key)
            if ranking is not None:
                self._page_cache.move_to_end(key)
            return ranking

    def _store_ranking(self, key: tuple, ranking: list) -> None:
        with self._page_cache_lock:
            self._page_cache[key] = ranking
            self._page_cache.move_to_end(key)
            while len(self._page_cache) > self._PAGE_CACHE_MAX:
                self._page_cache.popitem(last=False)

    def invalidate_page_cache(self) -> None:
        """Called after a sync: cached rankings reference chunk ids that a
        rebuild can prune or renumber."""
        with self._page_cache_lock:
            self._page_cache.clear()

    def conversation(self, chat_id: int, around_ts: float | None, limit: int = 50) -> dict:
        if self.chat_con is None:
            raise RuntimeError('No chat.db configured for conversation hydration.')
        payload = fetch_conversation(self.chat_con, chat_id=chat_id, around_ts=around_ts, limit=limit, contact_index=self.contact_index)
        meta = self.chatmeta.get(chat_id) if self.chatmeta else None
        if meta:
            payload.update({'title': meta.title, 'is_group': meta.is_group, 'participants': list(meta.participants)})
        return payload


def _effective_filters(parsed: ParsedQuery) -> dict:
    return {
        'people_participant': list(parsed.people_participant),
        'date_from': parsed.date_from,
        'date_to': parsed.date_to,
        'has_media': parsed.has_media,
        'is_group': getattr(parsed, 'is_group', None),
        'chat_ids': getattr(parsed, 'chat_ids', None),
        'semantic': parsed.semantic,
    }

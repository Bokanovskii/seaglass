from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from seaglass.app.chatmeta import ChatMetadataCache
from seaglass.app.filters import SearchFilters, apply_filters
from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import apple_to_unix, connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.conversation import fetch_conversation
from seaglass.search.format import format_search_result
from seaglass.search.hydrate import hydrate_sessions
from seaglass.search.parse import ParsedQuery, parse_query
from seaglass.search.rank import aggregate_sessions, expand_sessions, rerank_candidates
from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import build_candidate_chunk_ids, retrieve


@dataclass
class SearchOptions:
    max_sessions: int = 8
    fused_top_k: int = 50
    rerank: bool = True
    redact: bool = False
    expansions: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(self, index_db: str, chat_db: str | None = None, memory_index: bool = False):
        self.index_db = index_db
        self.chat_db = chat_db
        self.memory_index = memory_index
        self.index_con: sqlite3.Connection | None = None
        self.chat_con: sqlite3.Connection | None = None
        self.embedding_model: EmbeddingModel | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.contact_index: ContactIndex | None = None
        self.chatmeta: ChatMetadataCache | None = None
        self.corpus_bounds: tuple[float, float] = (0.0, 0.0)
        self.warnings: list[str] = []
        self.last_status: dict = {}

    def warmup(self, progress: Callable[[str], object]) -> None:
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
        with progress('build_chatmeta'):
            present_chat_ids = {row[0] for row in self.index_con.execute('SELECT DISTINCT chat_id FROM chunks')}
            self.chatmeta = ChatMetadataCache.build(self.chat_con, present_chat_ids) if self.chat_con else ChatMetadataCache({})
        with progress('load_contacts'):
            try:
                self.contact_index = ContactIndex.load()
            except ContactsUnavailableError as exc:
                self.warnings.append('Contacts unavailable — sender names will show raw handles')
                self.contact_index = None
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
        n_chunks = self.index_con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        n_vectors = self.index_con.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()[0]
        n_chats = self.index_con.execute('SELECT COUNT(DISTINCT chat_id) FROM chunks').fetchone()[0]
        most_recent_chunk_ts = self.index_con.execute('SELECT MAX(end_ts) FROM chunks').fetchone()[0]
        chat_db_max_ts = None
        newer_messages = 0
        if self.chat_con is not None:
            raw_max = self.chat_con.execute('SELECT MAX(date) FROM im.message').fetchone()[0]
            chat_db_max_ts = apple_to_unix(raw_max) if raw_max else None
            if chat_db_max_ts and most_recent_chunk_ts:
                newer_messages = self.chat_con.execute('SELECT COUNT(*) FROM im.message WHERE date > ?', (int((most_recent_chunk_ts - 978307200) * 1e9),)).fetchone()[0]
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
        }

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
                'effective_filters': _effective_filters(parsed),
                'parse_source': 'deterministic',
                'timings': timings,
                'elapsed_s': round(time.time() - t0, 2),
                'candidate_count': 0 if candidate_ids is None else len(candidate_ids),
            }

        if options.rerank:
            rerank_started = time.time()
            ranked = rerank_candidates(self.index_con, parsed.semantic, fused, self.reranker)
            timings['rerank'] = round(time.time() - rerank_started, 4)
        else:
            ranked = []
        aggregate_started = time.time()
        sessions = aggregate_sessions(ranked, max_sessions=options.max_sessions)
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
            'effective_filters': _effective_filters(parsed),
            'parse_source': 'deterministic',
            'timings': timings,
            'elapsed_s': round(time.time() - t0, 2),
            'candidate_count': 0 if candidate_ids is None else len(candidate_ids),
        })
        return payload

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

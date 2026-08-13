from __future__ import annotations

import concurrent.futures
import queue
import secrets
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from seaglass.app.assist import (
    AssistCircuitBreaker,
    AssistResult,
    build_prompt,
    cache_key,
    ensure_cache,
    get_cached_parse,
    merge_ghcp_parse,
    put_cached_parse,
    should_assist,
)
from seaglass.app.config import save_config
from seaglass.app.filters import SearchFilters
from seaglass.llm.ghcp import call_ghcp_json_object

STATIC_DIR = Path(__file__).resolve().parent / 'static'


class SearchBody(BaseModel):
    query: str
    filters: dict = Field(default_factory=dict)
    options: dict = Field(default_factory=dict)
    assist: str = 'off'
    request_id: str | None = None


class ConfigBody(BaseModel):
    index_db: str | None = None
    chat_db: str | None = None
    copilot_bin: str | None = None
    assist_mode: str | None = None
    max_sessions: int | None = None
    redact: bool | None = None
    port: int | None = None
    browser: bool | None = None
    memory_index: bool | None = None


class AuthMiddleware:
    def __init__(self, app: FastAPI, token: str, allowed_host: str):
        self.app = app
        self.token = token
        self.allowed_host = allowed_host

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api'):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        auth = request.headers.get('authorization')
        if auth != f'Bearer {self.token}':
            await JSONResponse(status_code=401, content={'detail': 'Missing or invalid auth token'})(scope, receive, send)
            return
        origin = request.headers.get('origin')
        host = request.headers.get('host')
        if origin and origin not in {f'http://{self.allowed_host}', f'https://{self.allowed_host}'}:
            await JSONResponse(status_code=403, content={'detail': 'Origin rejected'})(scope, receive, send)
            return
        if host != self.allowed_host:
            await JSONResponse(status_code=403, content={'detail': 'Host rejected'})(scope, receive, send)
            return
        await self.app(scope, receive, send)


class SearchAssistManager:
    def __init__(self, engine, copilot_bin: str | None):
        self.engine = engine
        self.copilot_bin = copilot_bin or 'copilot'
        self.cache = ensure_cache()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending: dict[str, concurrent.futures.Future] = {}
        self.breaker = AssistCircuitBreaker()

    def submit(self, query: str, parsed, assist_mode: str) -> str | None:
        if self.breaker.open or not should_assist(assist_mode, parsed):
            return None
        token = secrets.token_urlsafe(16)
        prompt = build_prompt(query, today=__import__('datetime').date.today().isoformat(), weekday=__import__('datetime').date.today().strftime('%A'), tz_name='local')
        key = cache_key(query, prompt_version='v1', today=__import__('datetime').date.today().isoformat(), aliases_mtime=0)
        cached = get_cached_parse(self.cache, key)
        if cached is not None:
            future = concurrent.futures.Future()
            future.set_result(cached)
            self.pending[token] = future
            return token

        def work():
            payload = call_ghcp_json_object(prompt, timeout_s=20, bin_path=self.copilot_bin)
            if payload is None:
                self.breaker.record_failure()
                return {'status': 'unavailable', 'reason': 'unparseable response'}
            put_cached_parse(self.cache, key, query, payload)
            self.breaker.record_success()
            return payload

        self.pending[token] = self.executor.submit(work)
        return token

    def get(self, token: str):
        future = self.pending.get(token)
        if future is None:
            return AssistResult(status='unavailable', reason='unknown token')
        if not future.done():
            return None
        payload = future.result()
        if isinstance(payload, dict) and payload.get('status') == 'unavailable':
            return AssistResult(status='unavailable', reason=payload.get('reason'))
        return payload


def create_app(engine, warmup_state, config, token: str):
    app = FastAPI()
    allowed_host = f'127.0.0.1:{config.port}'
    app.add_middleware(AuthMiddleware, token=token, allowed_host=allowed_host)
    pipeline_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    assist_manager = SearchAssistManager(engine, config.copilot_bin)
    app.state.pipeline_pool = pipeline_pool
    app.state.assist_manager = assist_manager

    @app.get('/')
    async def root():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/api/health')
    async def health():
        data = warmup_state.snapshot()
        data['engine'] = engine.health()
        return data

    @app.get('/api/status')
    async def status():
        loop = __import__('asyncio').get_running_loop()
        return await loop.run_in_executor(pipeline_pool, engine.status)

    @app.post('/api/search')
    async def search(body: SearchBody):
        loop = __import__('asyncio').get_running_loop()
        filters = SearchFilters(**body.filters)
        options = engine.__class__.__dict__ and __import__('seaglass.app.engine', fromlist=['SearchOptions']).SearchOptions(**body.options)
        payload = await loop.run_in_executor(pipeline_pool, engine.search, body.query, filters, options)
        payload['request_id'] = body.request_id
        parsed = __import__('seaglass.search.parse', fromlist=['parse_query']).parse_query(body.query, contact_index=engine.contact_index)
        assist_token = assist_manager.submit(body.query, parsed, body.assist)
        payload['assist_token'] = assist_token
        return payload

    @app.get('/api/assist/{assist_token}')
    async def assist(assist_token: str):
        result = assist_manager.get(assist_token)
        if result is None:
            return Response(status_code=204)
        if isinstance(result, AssistResult):
            return asdict(result)
        deterministic = __import__('seaglass.search.parse', fromlist=['parse_query']).parse_query('', contact_index=engine.contact_index)
        merged, changes, expansions = merge_ghcp_parse(deterministic, result, engine.contact_index, engine.corpus_bounds)
        return {'status': 'ready' if changes or expansions else 'unchanged', 'parse': result, 'changes': changes, 'expansions': expansions, 'confidence': result.get('confidence')}

    @app.post('/api/search/apply-assist')
    async def apply_assist(body: dict):
        assist_token = body['assist_token']
        pending = assist_manager.get(assist_token)
        if pending is None or isinstance(pending, AssistResult):
            raise HTTPException(status_code=404, detail='Assist result unavailable')
        deterministic = __import__('seaglass.search.parse', fromlist=['parse_query']).parse_query(body['query'], contact_index=engine.contact_index)
        merged, changes, expansions = merge_ghcp_parse(deterministic, pending, engine.contact_index, engine.corpus_bounds)
        loop = __import__('asyncio').get_running_loop()
        filters = SearchFilters(**body.get('filters', {}))
        options = __import__('seaglass.app.engine', fromlist=['SearchOptions']).SearchOptions(**body.get('options', {}), expansions=expansions)
        payload = await loop.run_in_executor(pipeline_pool, engine.search, merged.raw, filters, options)
        payload['parse_source'] = 'ghcp'
        payload['assist_changes'] = changes
        return payload

    @app.get('/api/conversation')
    async def conversation(chat_id: int, around_ts: float | None = None, limit: int = 50):
        loop = __import__('asyncio').get_running_loop()
        return await loop.run_in_executor(pipeline_pool, engine.conversation, chat_id, around_ts, limit)

    @app.get('/api/contacts/suggest')
    async def contacts(q: str = '', limit: int = 10):
        loop = __import__('asyncio').get_running_loop()
        return await loop.run_in_executor(pipeline_pool, engine.suggest_contacts, q, limit)

    @app.get('/api/chats/suggest')
    async def chats(q: str = '', limit: int = 20):
        loop = __import__('asyncio').get_running_loop()
        return await loop.run_in_executor(pipeline_pool, engine.suggest_chats, q, limit)

    @app.get('/api/config')
    async def get_config():
        return config.to_dict()

    @app.post('/api/config')
    async def set_config(body: ConfigBody):
        for key, value in body.model_dump(exclude_none=True).items():
            setattr(config, key, value)
        save_config(config)
        return config.to_dict()

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    return app

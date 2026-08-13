from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import queue
import secrets
import shlex
import signal
import subprocess
import threading
import time
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
from seaglass.app.engine import SearchOptions
from seaglass.app.filters import SearchFilters
from seaglass.app.indexstate import IndexBuildState, run_build
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


def _bundle_path() -> str | None:
    """The .app this process was launched from, if any.

    LaunchServices exports `__CFBundleIdentifier` when it starts a bundled
    app, and that is inherited across the exec from the bundle's stub to
    the interpreter -- so it, not the executable path, is what identifies a
    bundle launch.
    """
    if os.environ.get('__CFBundleIdentifier') != 'dev.seaglass.app':
        return None
    for candidate in (Path.home() / 'Applications' / 'Seaglass.app', Path('/Applications/Seaglass.app')):
        if candidate.exists():
            return str(candidate)
    return None


def _shutdown_soon(delay: float = 0.4) -> None:
    """Exit the app from a worker thread.

    SIGTERM is not enough here: Python runs signal handlers on the main
    thread, and the main thread is parked inside the Cocoa event loop, so
    the handler would not run until the window closed -- the exact thing
    being asked for. Closing the webview window is what unblocks it, which
    lets `__main__` release the lock on its way out.
    """
    time.sleep(delay)  # let the response reach the frontend first
    try:
        import webview

        for window in list(webview.windows):
            window.destroy()
    except Exception:  # noqa: BLE001 - browser/headless mode has no window
        os.kill(os.getpid(), signal.SIGTERM)

    def _force():
        time.sleep(8.0)
        os._exit(0)

    threading.Thread(target=_force, daemon=True).start()


def _coerce(cls, values):
    """Build a dataclass from a request body, ignoring unknown keys.

    A stray or stale key (an old frontend build, a Grogu client written
    against a newer schema) would otherwise raise TypeError and turn the
    whole search into a 500. Unknown filters are dropped rather than
    honoured, which degrades to a broader search -- the safe direction.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (values or {}).items() if k in known})


def create_app(engine, warmup_state, config, token: str):
    app = FastAPI()
    allowed_host = f'127.0.0.1:{config.port}'
    app.add_middleware(AuthMiddleware, token=token, allowed_host=allowed_host)
    pipeline_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    assist_manager = SearchAssistManager(engine, config.copilot_bin)
    app.state.pipeline_pool = pipeline_pool
    app.state.assist_manager = assist_manager
    app.state.build_state = IndexBuildState()

    @app.get('/')
    async def root():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/api/health')
    async def health():
        data = warmup_state.snapshot()
        data['engine'] = engine.health()
        data['build'] = app.state.build_state.snapshot()
        return data

    @app.get('/api/status')
    async def status():
        loop = __import__('asyncio').get_running_loop()
        return await loop.run_in_executor(pipeline_pool, engine.status)

    @app.post('/api/index/build')
    async def start_build():
        build_state: IndexBuildState = app.state.build_state
        if build_state.running:
            return JSONResponse(status_code=409, content={'detail': 'A build is already in progress.'})

        def on_complete():
            # Once a build finishes successfully, (re)warm the engine so
            # search reflects the freshly written index without requiring
            # the user to restart the app. This mutates engine.index_con /
            # chat_con / embedding_model etc. in place, so it MUST run on
            # pipeline_pool (the same single-worker executor that serializes
            # every engine.search()/status() call) rather than directly on
            # this build thread -- otherwise a concurrent search could read
            # from (or crash on) connections that re-warming is closing out
            # from under it.
            if build_state.stage != 'done':
                return

            def _rewarm():
                engine.invalidate_freshness()
                warmup_state.state = 'STARTING'
                warmup_state.error = None
                for step in warmup_state.steps:
                    step.state = 'pending'
                    step.error = None
                from seaglass.app.warmup import run_warmup
                from seaglass.llm.ghcp import detect_ghcp
                run_warmup(engine, warmup_state, lambda: detect_ghcp(config.copilot_bin))

            pipeline_pool.submit(_rewarm)

        thread = threading.Thread(
            target=run_build,
            kwargs=dict(
                state=app.state.build_state,
                chat_db_source=config.chat_db_source,
                chat_db_snapshot=config.chat_db,
                index_db=config.index_db,
                on_complete=on_complete,
            ),
            daemon=True,
        )
        thread.start()
        return {'started': True}

    @app.get('/api/index/build')
    async def build_status():
        return app.state.build_state.snapshot()

    @app.post('/api/search')
    async def search(body: SearchBody):
        loop = __import__('asyncio').get_running_loop()
        filters = _coerce(SearchFilters, body.filters)
        options = _coerce(SearchOptions, body.options)
        payload = await loop.run_in_executor(pipeline_pool, engine.search, body.query, filters, options)
        payload['request_id'] = body.request_id
        # A "load more" fetch is the same query -- re-running Assist would
        # duplicate the LLM call and overwrite the banner already shown.
        if int(body.options.get('offset') or 0) > 0:
            payload['assist_token'] = None
            return payload
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
        filters = _coerce(SearchFilters, body.get('filters'))
        options = _coerce(SearchOptions, body.get('options'))
        options.expansions = expansions
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

    # Only these panes, and only by key: the value is interpolated into a
    # URL handed to `open`, so accepting an arbitrary string from the page
    # would be a command/URL injection surface for no benefit.
    _SETTINGS_PANES = {
        'full_disk_access': 'x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles',
        'contacts': 'x-apple.systempreferences:com.apple.preference.security?Privacy_Contacts',
    }

    @app.post('/api/system/open-settings')
    async def open_settings(body: dict):
        """Deep-link into the exact Privacy pane a missing grant needs.

        Telling someone to "go to System Settings > Privacy & Security >
        Full Disk Access" is a five-step scavenger hunt; macOS has a URL
        scheme for landing directly on the pane, so use it.
        """
        url = _SETTINGS_PANES.get(str(body.get('pane') or ''))
        if url is None:
            raise HTTPException(status_code=400, detail='unknown settings pane')

        subprocess.Popen(['open', url])
        return {'opened': True}

    @app.post('/api/system/request-contacts')
    async def request_contacts():
        """Show the system Contacts prompt, on demand.

        Contacts has no "+" button in System Settings -- an app only
        appears in that list once it has *asked*, so without this there is
        no way for the user to grant access at all.

        It has to happen here rather than during warmup: warmup runs on a
        background thread before `webview.start()` spins up the Cocoa event
        loop, and a TCC prompt cannot be presented until there is one. So
        the request is driven from the UI, once the app is fully running.
        """
        from seaglass.imessage.contacts import request_contacts_access

        return request_contacts_access()

    @app.post('/api/system/relaunch')
    async def relaunch():
        """Quit and reopen. Needed after a privacy grant.

        macOS decides an app's TCC access when it launches, so enabling
        Full Disk Access does nothing for the process that is already
        running -- the app has to be restarted, and telling the user to go
        do that by hand right after they granted the permission is a poor
        finish to the flow.

        The relaunch is handed to a detached `sh` so it survives this
        process exiting. It waits for *this* pid to actually disappear
        rather than sleeping a fixed amount: the replacement checks the
        lock file on startup and, finding a live instance, would just open
        a browser tab at the old one and exit.
        """
        import sys

        bundle = _bundle_path()
        if bundle:
            launch = f'open -n {shlex.quote(bundle)}'
        else:
            argv = ' '.join(shlex.quote(part) for part in [sys.executable, *sys.argv])
            launch = f'{argv} &'
        pid = os.getpid()
        # Bounded so a shutdown that hangs leaves no orphan waiter.
        command = (
            f'for _ in $(seq 100); do kill -0 {pid} 2>/dev/null || break; sleep 0.2; done; '
            f'sleep 0.5; {launch}'
        )
        subprocess.Popen(['/bin/sh', '-c', command], start_new_session=True)

        threading.Thread(target=_shutdown_soon, daemon=True).start()
        return {'relaunching': True}

    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    return app

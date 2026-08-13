from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

import uvicorn

from seaglass.app.config import AppConfig, CONFIG_PATH, LOCK_PATH, ConfigError, load_config, save_config
from seaglass.app.engine import SearchEngine
from seaglass.app.server import create_app
from seaglass.app.warmup import DEFAULT_WARMUP_STEPS, WarmupState, run_warmup
from seaglass.llm.ghcp import detect_ghcp


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Launch the seaglass desktop search app.')
    parser.add_argument('--index-db')
    parser.add_argument('--chat-db')
    parser.add_argument('--port', type=int)
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--assist', choices=['off', 'auto', 'force'])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--query')
    return parser.parse_args(argv)


def pick_port(requested: int | None) -> int:
    start = requested or 8765
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError('No free port found near requested range')


def pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_lock() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text())
    except json.JSONDecodeError:
        return None


def acquire_lock(port: int, token: str, browser: bool) -> bool:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_lock()
    if existing and pid_is_live(existing.get('pid', -1)):
        try:
            urlopen(f"http://127.0.0.1:{existing['port']}/")
            webbrowser.open(f"http://127.0.0.1:{existing['port']}/#{existing['token']}")
            return False
        except Exception:
            pass
    LOCK_PATH.write_text(json.dumps({'pid': os.getpid(), 'port': port, 'token': token, 'browser': browser}))
    return True


def cleanup(*_args):
    if LOCK_PATH.exists():
        try:
            data = json.loads(LOCK_PATH.read_text())
        except Exception:
            data = None
        if data and data.get('pid') == os.getpid():
            LOCK_PATH.unlink(missing_ok=True)


def _build_config(args) -> AppConfig:
    config = load_config(CONFIG_PATH)
    if args.index_db:
        config.index_db = args.index_db
    if args.chat_db:
        config.chat_db = args.chat_db
    if args.port:
        config.port = args.port
    if args.browser:
        config.browser = True
    if args.assist:
        config.assist_mode = args.assist
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _build_config(args)
        warnings = config.validate()
    except ConfigError as exc:
        print(f'seaglass-app: {exc}')
        return 2

    if not config.index_db:
        print('seaglass-app: No index database configured. Run `seaglass build ...` first or set SEAGLASS_INDEX_DB.')
        return 2

    config.port = pick_port(config.port)
    token = secrets.token_urlsafe(24)
    if not acquire_lock(config.port, token, config.browser):
        return 0
    save_config(config)

    engine = SearchEngine(config.index_db, config.chat_db, memory_index=config.memory_index)
    warmup_state = WarmupState(list(DEFAULT_WARMUP_STEPS))
    for warning in warnings:
        warmup_state.add_warning(warning)

    app = create_app(engine, warmup_state, config, token)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=config.port, log_level='warning'))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    warmup_thread = threading.Thread(target=run_warmup, args=(engine, warmup_state, lambda: detect_ghcp(config.copilot_bin)), daemon=True)
    warmup_thread.start()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if args.headless:
        warmup_thread.join()
        if args.query:
            from seaglass.app.filters import SearchFilters
            from seaglass.app.engine import SearchOptions
            payload = engine.search(args.query, SearchFilters(), SearchOptions())
            print(json.dumps(payload, indent=2))
        cleanup()
        return 0

    url = f'http://127.0.0.1:{config.port}/#{token}'
    if config.browser:
        webbrowser.open(url)
        try:
            while server.started:
                time.sleep(0.2)
        finally:
            cleanup()
        return 0

    import webview

    webview.create_window('seaglass', url, width=1280, height=900)
    try:
        webview.start()
    finally:
        cleanup()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

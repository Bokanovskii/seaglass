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
    parser.add_argument('--chat-db-source', help='live chat.db to snapshot from when building/syncing the index from the app')
    parser.add_argument('--port', type=int)
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--assist', choices=['off', 'auto', 'force'])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--query')
    return parser.parse_args(argv)


def _set_macos_app_name(name: str = 'Seaglass') -> None:
    """Show "Seaglass" rather than "Python" in the Dock, the Cmd+Tab
    switcher and the menu bar.

    For a non-bundled interpreter macOS takes the display name from the
    *running binary's* bundle -- which is CPython's own, hence "Python".
    There is no NSApplication setter for it, but AppKit reads the name out
    of the main bundle's info dictionary lazily, so overwriting
    `CFBundleName` in that (mutable) dictionary before the app finishes
    launching is what actually takes effect.

    Must run before `webview.start()` creates the NSApplication. Purely
    cosmetic, so every failure path is swallowed.
    """
    try:
        from Foundation import NSBundle
    except ImportError:
        return
    try:
        bundle = NSBundle.mainBundle()
        if bundle is None:
            return
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is None:
            return
        info['CFBundleName'] = name
        info['CFBundleDisplayName'] = name
    except Exception:
        pass


def _set_macos_dock_icon() -> None:
    """Set a proper seaglass Dock icon instead of the generic Python
    rocket. pywebview's own `icon=` kwarg to `create_window` only takes
    effect on GTK/QT (its docs say so explicitly) -- on macOS the Cocoa
    backend has no such hook, and the Dock just shows whatever icon the
    running interpreter itself has (the stock CPython "rocket"). Setting
    `NSApplication.sharedApplication().setApplicationIconImage_()`
    directly via PyObjC is the actual mechanism that changes the Dock
    tile and Cmd+Tab switcher for a plain (non-bundled) Python process.

    Best-effort only: if PyObjC/AppKit isn't importable (non-macOS, or a
    stripped-down environment) or the icon asset is missing, silently
    skip rather than fail app startup over cosmetics.
    """
    try:
        import AppKit
    except ImportError:
        return

    icon_path = Path(__file__).resolve().parent / 'static' / 'icons' / 'icon-512.png'
    if not icon_path.exists():
        return
    try:
        image = AppKit.NSImage.alloc().initWithContentsOfFile_(str(icon_path))
        if image is not None:
            AppKit.NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:
        pass


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
    if args.chat_db_source:
        config.chat_db_source = args.chat_db_source
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
        # Defensive: apply_env_overrides always fills in a default, but keep
        # this in case a caller constructs AppConfig directly without going
        # through load_config/apply_env_overrides.
        print('seaglass-app: No index database configured. Set --index-db or SEAGLASS_INDEX_DB.')
        return 2

    config.port = pick_port(config.port)
    token = secrets.token_urlsafe(24)
    if not acquire_lock(config.port, token, config.browser):
        return 0
    save_config(config)

    engine = SearchEngine(config.index_db, config.chat_db, memory_index=config.memory_index, chat_db_source=config.chat_db_source)
    warmup_state = WarmupState(list(DEFAULT_WARMUP_STEPS))
    for warning in warnings:
        warmup_state.add_warning(warning)

    index_exists = Path(config.index_db).exists()

    app = create_app(engine, warmup_state, config, token)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=config.port, log_level='warning'))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if index_exists:
        warmup_thread = threading.Thread(
            target=run_warmup, args=(engine, warmup_state, lambda: detect_ghcp(config.copilot_bin)), daemon=True
        )
        warmup_thread.start()
    else:
        # No index yet: don't run warmup against a nonexistent database.
        # The frontend detects NEEDS_INDEX and shows a "Build index now"
        # screen; warmup is (re)started automatically once a build
        # completes (see server.py's /api/index/build).
        warmup_state.state = 'NEEDS_INDEX'
        warmup_thread = None

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if args.headless:
        if warmup_thread is not None:
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

    _set_macos_app_name()
    _set_macos_dock_icon()
    icon_path = Path(__file__).resolve().parent / 'static' / 'icons' / 'icon-512.png'
    webview.create_window('Seaglass', url, width=1280, height=900)
    try:
        # `icon` is a start()-time kwarg (GTK/QT only, per pywebview docs),
        # not a create_window() one -- harmless no-op on the Cocoa backend,
        # where _set_macos_dock_icon() above is what actually takes effect.
        webview.start(icon=str(icon_path) if icon_path.exists() else None)
    finally:
        cleanup()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

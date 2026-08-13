from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8765
CONFIG_DIR = Path.home() / '.seaglass'
CONFIG_PATH = CONFIG_DIR / 'config.json'
ALIASES_PATH = CONFIG_DIR / 'aliases.json'
APP_DB_PATH = CONFIG_DIR / 'app.db'
LOCK_PATH = CONFIG_DIR / 'app.lock'
DEFAULT_INDEX_DB = CONFIG_DIR / 'index.db'
DEFAULT_CHAT_SNAPSHOT_DB = CONFIG_DIR / 'chat_snapshot.db'
DEFAULT_CHAT_DB_SOURCE = Path.home() / 'Library' / 'Messages' / 'chat.db'


class ConfigError(RuntimeError):
    pass


@dataclasses.dataclass
class AppConfig:
    index_db: str | None = None
    chat_db: str | None = None
    chat_db_source: str | None = None
    copilot_bin: str | None = None
    assist_mode: str = 'off'
    max_sessions: int = 8
    redact: bool = False
    port: int = DEFAULT_PORT
    browser: bool = False
    memory_index: bool = False

    def ensure_runtime_dir(self) -> Path:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_DIR

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if self.index_db and not Path(self.index_db).exists():
            # No hard failure: the app can start without an index and offer
            # to build one from the UI, rather than requiring a separate
            # CLI step before the app is even usable.
            warnings.append(
                f'No index found at {self.index_db} yet. Use the "Build index" button in the app to create one.'
            )
        if self.chat_db and not Path(self.chat_db).exists():
            warnings.append(
                f'chat.db not found at {self.chat_db}. Grant Full Disk Access and set a valid chat.db snapshot path to enable hydration.'
            )
        if self.assist_mode not in {'off', 'auto', 'force'}:
            raise ConfigError('assist_mode must be one of: off, auto, force')
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text())
    config = AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
    apply_env_overrides(config)
    return config


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    config.ensure_runtime_dir()
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True))


def apply_env_overrides(config: AppConfig) -> AppConfig:
    if os.environ.get('SEAGLASS_INDEX_DB'):
        config.index_db = os.environ['SEAGLASS_INDEX_DB']
    if os.environ.get('SEAGLASS_CHAT_DB'):
        config.chat_db = os.environ['SEAGLASS_CHAT_DB']
    if os.environ.get('SEAGLASS_CHAT_DB_SOURCE'):
        config.chat_db_source = os.environ['SEAGLASS_CHAT_DB_SOURCE']
    if os.environ.get('SEAGLASS_COPILOT_BIN'):
        config.copilot_bin = os.environ['SEAGLASS_COPILOT_BIN']
    if os.environ.get('SEAGLASS_APP_MEMORY_INDEX'):
        config.memory_index = os.environ['SEAGLASS_APP_MEMORY_INDEX'] not in {'0', 'false', 'False'}
    if not config.index_db:
        config.index_db = str(DEFAULT_INDEX_DB)
    if not config.chat_db:
        config.chat_db = str(DEFAULT_CHAT_SNAPSHOT_DB)
    if not config.chat_db_source:
        config.chat_db_source = str(DEFAULT_CHAT_DB_SOURCE)
    return config


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

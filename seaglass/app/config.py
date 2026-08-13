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


class ConfigError(RuntimeError):
    pass


@dataclasses.dataclass
class AppConfig:
    index_db: str | None = None
    chat_db: str | None = None
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
            raise ConfigError(
                f'Index database not found at {self.index_db}. Run `seaglass build ...` first or set SEAGLASS_INDEX_DB.'
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
    if os.environ.get('SEAGLASS_COPILOT_BIN'):
        config.copilot_bin = os.environ['SEAGLASS_COPILOT_BIN']
    if os.environ.get('SEAGLASS_APP_MEMORY_INDEX'):
        config.memory_index = os.environ['SEAGLASS_APP_MEMORY_INDEX'] not in {'0', 'false', 'False'}
    return config


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

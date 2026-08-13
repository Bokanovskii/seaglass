from __future__ import annotations

import dataclasses
import threading
import time
from contextlib import contextmanager
from typing import Callable


@dataclasses.dataclass
class WarmupStep:
    name: str
    state: str = 'pending'
    elapsed_s: float = 0.0
    error: str | None = None


class WarmupState:
    def __init__(self, step_names: list[str] | None = None):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.state = 'STARTING'
        self.error: str | None = None
        self.warnings: list[str] = []
        self.steps = [WarmupStep(name) for name in (step_names or [])]
        self.ghcp = {'available': False, 'version': None, 'probing': False, 'reason': None}

    @contextmanager
    def step(self, name: str):
        step = self._ensure_step(name)
        start = time.time()
        with self._lock:
            step.state = 'running'
        try:
            yield step
        except Exception as exc:
            with self._lock:
                step.state = 'failed'
                step.error = str(exc)
                step.elapsed_s = time.time() - start
                self.state = 'FAILED'
                self.error = str(exc)
            raise
        else:
            with self._lock:
                step.state = 'done'
                step.elapsed_s = time.time() - start

    def add_warning(self, warning: str) -> None:
        with self._lock:
            self.warnings.append(warning)
            if self.state != 'FAILED':
                self.state = 'DEGRADED'

    def set_ready(self) -> None:
        with self._lock:
            self.state = 'DEGRADED' if self.warnings else 'READY'

    def set_ghcp(self, *, available: bool, version: str | None = None, probing: bool = False, reason: str | None = None):
        with self._lock:
            self.ghcp = {'available': available, 'version': version, 'probing': probing, 'reason': reason}

    def snapshot(self) -> dict:
        with self._lock:
            done = sum(1 for step in self.steps if step.state == 'done')
            progress = done / len(self.steps) if self.steps else 0.0
            return {
                'state': self.state,
                'steps': [dataclasses.asdict(step) for step in self.steps],
                'progress': progress,
                'elapsed_s': round(time.time() - self.started_at, 2),
                'error': self.error,
                'warnings': list(self.warnings),
                'ghcp': dict(self.ghcp),
            }

    def _ensure_step(self, name: str) -> WarmupStep:
        with self._lock:
            for step in self.steps:
                if step.name == name:
                    return step
            step = WarmupStep(name)
            self.steps.append(step)
            return step


DEFAULT_WARMUP_STEPS = [
    'import_runtime',
    'open_index',
    'configure_index',
    'read_meta',
    'open_chat',
    'build_chatmeta',
    'load_contacts',
    'warm_sqlite',
    'load_embedding_model',
    'load_reranker',
    'dummy_search',
    'detect_ghcp',
]


def run_warmup(engine, state: WarmupState, detect_ghcp: Callable[[], object] | None = None) -> None:
    engine.warmup(progress=state.step)
    if detect_ghcp is not None:
        state.set_ghcp(available=False, probing=True)
        with state.step('detect_ghcp'):
            availability = detect_ghcp()
            state.set_ghcp(
                available=getattr(availability, 'available', False),
                version=getattr(availability, 'version', None),
                probing=False,
                reason=getattr(availability, 'reason', None),
            )
    state.set_ready()

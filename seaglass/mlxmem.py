"""Bound MLX's Metal buffer cache.

MLX recycles freed GPU buffers through a cache whose default limit is
effectively the machine's memory. On Apple Silicon those buffers live in
unified memory and count against the process's *physical footprint* -- the
number Activity Monitor shows -- but not against RSS, so `ps` reports a
tidy ~170 MB while Activity Monitor climbs past 7 GB.

Nothing is leaking: measured over repeated embed + rerank rounds, active
memory stays flat at ~255 MB while the cache alone grows to ~1.5 GB and
never shrinks, because the cache retains the high-water mark of every
batch ever run (an index build embeds far larger batches than a query
does, so a sync sets a very high mark that then persists for the life of
the process).

Capping the cache is measurably free -- six embed+rerank rounds took 0.64s
uncapped, 0.63s at both 512 MB and 256 MB -- because the working set of a
single query is far smaller than the retained cache. Buffers beyond the
limit are returned to the OS instead of being held.
"""

from __future__ import annotations

import os

# Comfortably above a single query's working set, so the cache still does
# its job, while keeping the idle footprint near ~800 MB rather than
# whatever the largest batch since startup happened to need.
DEFAULT_CACHE_LIMIT_MB = 512
_ENV_VAR = "SEAGLASS_MLX_CACHE_MB"

_configured = False


def cache_limit_mb() -> int:
    """Resolve the configured limit. `0` disables caching entirely; a
    negative value leaves MLX's default (unbounded) in place.
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_CACHE_LIMIT_MB
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_CACHE_LIMIT_MB


def configure_mlx_memory(force: bool = False) -> None:
    """Apply the cache cap. Idempotent and safe to call from any model's
    lazy-load path; only the first call does work unless `force` is set.
    """
    global _configured
    if _configured and not force:
        return
    limit = cache_limit_mb()
    if limit < 0:
        _configured = True
        return
    try:
        import mlx.core as mx

        mx.set_cache_limit(limit * 1024 * 1024)
    except Exception:
        # Memory tuning must never be the reason search fails to start.
        pass
    _configured = True


def release_mlx_cache() -> None:
    """Drop cached buffers now. Worth calling after an index build, whose
    batches are much larger than a query's -- otherwise the build's
    high-water mark stays resident for the life of the process.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass

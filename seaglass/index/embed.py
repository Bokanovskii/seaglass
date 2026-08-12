"""`index/embed.py` — embedding generation and int8 quantisation.

See development-plans/PLAN.md §6 Phase 3 ("int8 only. No fp32 is ever
stored") for the full rationale. Two independently-testable pieces:

* Pure numpy quantisation math (`compute_calibration_absmax`,
  `quantize_int8`) -- no MLX dependency, trivially unit-testable.
* `EmbeddingModel`, a thin lazy-loading wrapper around `mlx-embeddings`'
  `bge-small-en-v1.5` -- requires MLX + a model download, so it is
  exercised by a manual/integration spike rather than unit tests (see
  ADDENDUM.md for the spike results: 384-dim output, ~20ms/batch-32 warm).
"""

from __future__ import annotations

import dataclasses
from typing import List, Sequence

import numpy as np

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


def compute_calibration_absmax(sample_vectors: np.ndarray, percentile: float = 99.9) -> float:
    """Compute the int8 calibration scale from a sample of L2-normalised
    embedding vectors. Store the result once in `meta.int8_absmax`; every
    query vector must be quantised against this same value (PLAN.md §6
    Phase 3, "Queries must use the same absmax").
    """
    return float(np.percentile(np.abs(sample_vectors), percentile))


def quantize_int8(vectors: np.ndarray, absmax: float) -> np.ndarray:
    """Quantise L2-normalised float vectors to int8 using a calibrated
    scale. NOT `round(v * 127)` -- see PLAN.md §6 Phase 3 for why a bare
    127-scale wastes ~80% of the int8 range for unit-norm 384-d vectors.
    """
    scaled = np.round(vectors * (127.0 / absmax))
    return np.clip(scaled, -127, 127).astype(np.int8)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


@dataclasses.dataclass
class EmbeddingModel:
    """Lazy-loading wrapper around `mlx-embeddings`' bge-small-en-v1.5.
    Loading requires network access on first run (HF Hub download, cached
    thereafter) -- kept out of the constructor so importing this module
    never touches the network.
    """

    model_id: str = EMBED_MODEL_ID
    _model: object = dataclasses.field(default=None, repr=False, init=False)
    _tokenizer: object = dataclasses.field(default=None, repr=False, init=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_embeddings import load

        self._model, self._tokenizer = load(self.model_id)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts, L2-normalised, as float32 numpy.
        Quantise separately with `quantize_int8` before storing/querying.
        """
        self._ensure_loaded()
        import mlx.core as mx
        from mlx_embeddings import generate

        out = generate(self._model, self._tokenizer, texts=list(texts))
        mx.eval(out.text_embeds)
        vectors = np.array(out.text_embeds)
        return l2_normalize(vectors)

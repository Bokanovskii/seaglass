"""Integration smoke tests for the MLX embedding + reranking models.

These require network access (first-run HF Hub download, cached
thereafter) and real model inference -- not appropriate to run on every
CI invocation, but valuable to run whenever the mlx-embeddings pin
changes or on a fresh machine. Marked `integration` so `pytest -m
"not integration"` skips them.
"""

from __future__ import annotations

import pytest

from seaglass.index.embed import EMBED_DIM, EmbeddingModel
from seaglass.search.rerank import CrossEncoderReranker

pytestmark = pytest.mark.integration


class TestEmbeddingModelIntegration:
    def test_embed_produces_correct_dimension_and_unit_norm(self):
        import numpy as np

        model = EmbeddingModel()
        vectors = model.embed(["hey are you free tonight?", "another message"])
        assert vectors.shape == (2, EMBED_DIM)
        norms = np.linalg.norm(vectors, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


class TestCrossEncoderRerankerIntegration:
    def test_reranker_scores_relevant_higher_than_irrelevant(self):
        reranker = CrossEncoderReranker()
        query = "are you free tonight?"
        relevant = "Them: hey are you free tonight?\nMe: yeah what's up"
        irrelevant = "Them: can you pick up milk on the way home\nMe: sure"
        scores = reranker.score([(query, relevant), (query, irrelevant)])
        assert len(scores) == 2
        assert scores[0] > scores[1]

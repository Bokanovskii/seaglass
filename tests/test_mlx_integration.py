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

    def test_model_is_in_eval_mode_not_training_mode(self):
        # Regression test: the loader previously never called .eval(),
        # so MLX's default train-mode dropout (hidden_dropout_prob=0.1,
        # attention_probs_dropout_prob=0.1) fired at inference, making
        # score() nondeterministic run-to-run (rank order flipped across
        # identical calls). A single .training assertion catches this.
        reranker = CrossEncoderReranker()
        reranker._ensure_loaded()
        assert reranker._model.training is False

    def test_score_is_deterministic_across_repeated_calls(self):
        # Regression test for the same dropout-in-train-mode bug: with
        # eval mode correctly set, identical inputs must produce bit-for-
        # bit identical scores every call.
        reranker = CrossEncoderReranker()
        query = "are you free tonight?"
        candidate = "Them: hey are you free tonight?\nMe: yeah what's up"
        first = reranker.score([(query, candidate)])
        for _ in range(4):
            again = reranker.score([(query, candidate)])
            assert again == first

"""Unit tests for seaglass.index.embed -- pure numpy quantisation math only.
EmbeddingModel itself requires network + MLX and is validated by a manual
spike (see development-plans/ADDENDUM.md), not exercised here.
"""

from __future__ import annotations

import numpy as np

from seaglass.index.embed import compute_calibration_absmax, l2_normalize, quantize_int8


class TestL2Normalize:
    def test_unit_norm_after_normalisation(self):
        vectors = np.random.default_rng(0).normal(size=(10, 384)).astype(np.float32)
        normed = l2_normalize(vectors)
        norms = np.linalg.norm(normed, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_vector_does_not_produce_nan(self):
        vectors = np.zeros((1, 384), dtype=np.float32)
        normed = l2_normalize(vectors)
        assert not np.isnan(normed).any()


class TestCalibrationAndQuantization:
    def test_absmax_is_within_plausible_bge_range(self):
        # BGE-small L2-normalised components cluster around 1/sqrt(384) ~ 0.051
        rng = np.random.default_rng(1)
        raw = rng.normal(scale=0.06, size=(10_000, 384)).astype(np.float32)
        sample = l2_normalize(raw)
        absmax = compute_calibration_absmax(sample)
        assert 0.05 < absmax < 0.6

    def test_quantized_values_use_full_int8_range_not_just_20_percent(self):
        # Regression test for the "round(v * 127)" bug PLAN.md warns about:
        # a naive scale wastes ~80% of the int8 range for unit-norm vectors.
        rng = np.random.default_rng(2)
        raw = rng.normal(scale=0.06, size=(1000, 384)).astype(np.float32)
        sample = l2_normalize(raw)
        absmax = compute_calibration_absmax(sample)
        quantized = quantize_int8(sample, absmax)
        assert quantized.dtype == np.int8
        # calibrated scale should push some values well past the naive
        # "v * 127" range of roughly [-26, 26]
        assert np.abs(quantized).max() > 40

    def test_quantized_values_never_exceed_int8_bounds(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(scale=0.5, size=(100, 384)).astype(np.float32)  # deliberately extreme
        absmax = 0.05  # deliberately too small, forcing clipping
        quantized = quantize_int8(raw, absmax)
        assert quantized.min() >= -127
        assert quantized.max() <= 127

    def test_query_and_document_must_share_absmax_for_consistent_scores(self):
        rng = np.random.default_rng(4)
        raw = l2_normalize(rng.normal(scale=0.06, size=(5, 384)).astype(np.float32))
        absmax = compute_calibration_absmax(raw)
        q1 = quantize_int8(raw, absmax)
        q2 = quantize_int8(raw, absmax * 2)  # simulate a scale mismatch
        assert not np.array_equal(q1, q2)

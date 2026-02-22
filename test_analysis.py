"""
tests/test_analysis.py
──────────────────────
Unit tests for the attention analysis pipeline.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from attention_analyzer import (
    head_entropy,
    top_attended_tokens,
    save_results,
    load_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_attn():
    """
    Mock attention tensor: (n_layers=4, n_heads=4, seq_len=6, seq_len=6).
    Each head's rows are a valid probability distribution (sum to 1).
    """
    rng  = np.random.default_rng(42)
    raw  = rng.dirichlet(np.ones(6), size=(4, 4, 6))   # (L, H, seq, seq)
    return raw


@pytest.fixture
def mock_tokens():
    return ["The", "Ġquick", "Ġbrown", "Ġfox", "Ġjumps", "Ġover"]


# ──────────────────────────────────────────────────────────────────────────────
# head_entropy
# ──────────────────────────────────────────────────────────────────────────────

class TestHeadEntropy:

    def test_output_shape(self, mock_attn):
        ent = head_entropy(mock_attn)
        assert ent.shape == (4, 4), f"Expected (4,4), got {ent.shape}"

    def test_entropy_non_negative(self, mock_attn):
        ent = head_entropy(mock_attn)
        assert (ent >= 0).all(), "Entropy should be non-negative"

    def test_uniform_distribution_has_high_entropy(self):
        """Uniform attention should have high entropy."""
        n = 10
        uniform = np.full((1, 1, n, n), 1.0 / n)
        ent = head_entropy(uniform)
        expected = np.log(n)          # max entropy for n tokens
        assert abs(ent[0, 0] - expected) < 0.01

    def test_peaked_distribution_has_low_entropy(self):
        """Attention entirely on one token should have entropy ≈ 0."""
        n = 10
        peaked = np.zeros((1, 1, n, n))
        peaked[0, 0, :, 0] = 1.0   # all attention on first token
        ent = head_entropy(peaked)
        assert ent[0, 0] < 0.1

    def test_returns_numpy_array(self, mock_attn):
        ent = head_entropy(mock_attn)
        assert isinstance(ent, np.ndarray)


# ──────────────────────────────────────────────────────────────────────────────
# top_attended_tokens
# ──────────────────────────────────────────────────────────────────────────────

class TestTopAttendedTokens:

    def test_returns_correct_length(self, mock_attn, mock_tokens):
        result = top_attended_tokens(mock_attn, mock_tokens, layer=0, head=0, top_k=3)
        assert len(result) == 3

    def test_returns_tuples(self, mock_attn, mock_tokens):
        result = top_attended_tokens(mock_attn, mock_tokens, layer=0, head=0, top_k=3)
        for item in result:
            assert isinstance(item, tuple)
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_scores_descending(self, mock_attn, mock_tokens):
        result = top_attended_tokens(mock_attn, mock_tokens, layer=0, head=0, top_k=5)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_1(self, mock_attn, mock_tokens):
        result = top_attended_tokens(mock_attn, mock_tokens, layer=1, head=2, top_k=1)
        assert len(result) == 1

    def test_tokens_in_vocab(self, mock_attn, mock_tokens):
        result = top_attended_tokens(mock_attn, mock_tokens, layer=0, head=0, top_k=4)
        for token, _ in result:
            assert token in mock_tokens


# ──────────────────────────────────────────────────────────────────────────────
# save_results / load_results
# ──────────────────────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_roundtrip(self, tmp_path):
        data = {
            "model":   "gpt2",
            "array":   np.array([[1.0, 2.0], [3.0, 4.0]]),
            "scalar":  np.float32(3.14),
            "nested":  {"x": np.int64(7)},
        }
        path = str(tmp_path / "test.json")
        save_results(data, path)
        loaded = load_results(path)

        assert loaded["model"] == "gpt2"
        assert loaded["array"] == [[1.0, 2.0], [3.0, 4.0]]
        assert abs(loaded["scalar"] - 3.14) < 0.01
        assert loaded["nested"]["x"] == 7

    def test_file_created(self, tmp_path):
        path = str(tmp_path / "out.json")
        save_results({"k": "v"}, path)
        assert Path(path).exists()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke tests (no model required)
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeEntropyEdgeCases:

    def test_single_layer_single_head(self):
        rng  = np.random.default_rng(0)
        attn = rng.dirichlet(np.ones(5), size=(1, 1, 5))
        ent  = head_entropy(attn)
        assert ent.shape == (1, 1)

    def test_large_sequence(self):
        rng  = np.random.default_rng(0)
        attn = rng.dirichlet(np.ones(100), size=(6, 8, 100))
        ent  = head_entropy(attn)
        assert ent.shape == (6, 8)
        assert np.isfinite(ent).all()

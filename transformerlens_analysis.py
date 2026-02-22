"""
transformerlens_analysis.py
───────────────────────────
Advanced interpretability using TransformerLens (Neel Nanda).

TransformerLens wraps HuggingFace models and provides:
  • Direct access to residual stream, MLP outputs, and attention patterns
  • Activation patching
  • Logit lens
  • Head ablation

Install:  pip install transformer-lens

NOTE: This module gracefully degrades to a mock implementation if
      TransformerLens is not installed, so the rest of the project
      still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import transformer_lens
    from transformer_lens import HookedTransformer, utils as tl_utils
    TL_AVAILABLE = True
except ImportError:
    TL_AVAILABLE = False
    print("[transformerlens] TransformerLens not installed.  "
          "Run:  pip install transformer-lens\n"
          "Falling back to mock implementations for demonstration.")

import torch

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load via TransformerLens
# ──────────────────────────────────────────────────────────────────────────────

def load_hooked_model(model_name: str = "gpt2") -> "HookedTransformer | None":
    """
    Load a HookedTransformer.  Returns None if TL not available.
    """
    if not TL_AVAILABLE:
        print(f"[TL] Skipping load of '{model_name}' — TL not installed.")
        return None

    print(f"[TL] Loading '{model_name}' via TransformerLens …")
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Activation caching (logit lens)
# ──────────────────────────────────────────────────────────────────────────────

def run_with_cache(
    model:       "HookedTransformer",
    text:        str,
    names_filter: Optional[callable] = None,
) -> Tuple[torch.Tensor, "ActivationCache"]:
    """
    Run model and return (logits, cache).
    names_filter can restrict which activations are stored, e.g.:
        names_filter = lambda n: n.startswith("blocks.0")
    """
    if not TL_AVAILABLE or model is None:
        raise RuntimeError("TransformerLens not available.")

    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
    return logits, cache


def logit_lens(
    model: "HookedTransformer",
    text:  str,
    top_k: int = 5,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    For each layer, decode the top-k predicted tokens from the residual stream.

    Returns
    -------
    per_layer_preds : {layer_idx: [(token_str, prob), ...]}
    """
    if not TL_AVAILABLE or model is None:
        # Mock output for demo/testing
        return {
            l: [(f"[mock_token_{i}]", round(1.0 / (i + 1), 3))
                for i in range(top_k)]
            for l in range(12)
        }

    tokens, cache = run_with_cache(model, text)
    n_layers = model.cfg.n_layers
    results  = {}

    for layer in range(n_layers):
        # residual stream after this layer
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # (1, seq, d_model)
        # apply final LN + unembed
        logits_layer = model.unembed(model.ln_final(resid))  # (1, seq, vocab)
        probs = logits_layer[0, -1].softmax(dim=-1)          # last token

        top_probs, top_ids = probs.topk(top_k)
        results[layer] = [
            (model.to_string(tid.unsqueeze(0)), float(p))
            for tid, p in zip(top_ids, top_probs)
        ]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Head ablation study
# ──────────────────────────────────────────────────────────────────────────────

def ablate_head(
    model:       "HookedTransformer",
    text:        str,
    layer:       int,
    head:        int,
    ablation_value: float = 0.0,
) -> float:
    """
    Zero-ablate a single attention head and measure the change in cross-entropy
    loss on the input text.

    Returns
    -------
    delta_loss : float  (positive = ablation hurts the model)
    """
    if not TL_AVAILABLE or model is None:
        # Mock: return a plausible random delta
        rng = np.random.default_rng(layer * 100 + head)
        return float(rng.normal(0.05, 0.02))

    tokens     = model.to_tokens(text)
    base_logits, _ = model.run_with_cache(tokens)
    base_loss  = _cross_entropy_loss(base_logits, tokens)

    hook_name  = f"blocks.{layer}.attn.hook_z"

    def zero_ablate(value, hook):
        value[:, :, head, :] = ablation_value
        return value

    ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, zero_ablate)],
    )
    ablated_loss = _cross_entropy_loss(ablated_logits, tokens)

    return float(ablated_loss - base_loss)


def _cross_entropy_loss(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    """Compute mean cross-entropy on next-token prediction."""
    shift_logits = logits[0, :-1]          # (seq-1, vocab)
    shift_labels = tokens[0, 1:]           # (seq-1,)
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return float(loss)


def full_ablation_matrix(
    model:  "HookedTransformer",
    text:   str,
    layers: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Run ablation for every head in the specified layers (default: all).

    Returns
    -------
    delta_matrix : (n_layers, n_heads)
    """
    if not TL_AVAILABLE or model is None:
        # Mock matrix
        n_layers = 12; n_heads = 12
        rng = np.random.default_rng(42)
        return rng.normal(0.05, 0.03, (n_layers, n_heads)).clip(0)

    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    layers   = layers or list(range(n_layers))
    matrix   = np.zeros((n_layers, n_heads))

    for l in layers:
        for h in range(n_heads):
            matrix[l, h] = ablate_head(model, text, l, h)
            print(f"  ablated L{l}·H{h}: Δloss = {matrix[l, h]:+.4f}")

    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# Attention pattern extraction via TL
# ──────────────────────────────────────────────────────────────────────────────

def tl_get_attention_patterns(
    model: "HookedTransformer",
    text:  str,
) -> Tuple[List[str], np.ndarray]:
    """
    Extract attention patterns using TransformerLens cache.
    Returns (tokens, attn) matching the shape used in attention_analyzer.py
    """
    if not TL_AVAILABLE or model is None:
        # Return mock data
        seq = 10; n_layers = 12; n_heads = 12
        mock_attn = np.random.dirichlet(
            np.ones(seq), size=(n_layers, n_heads, seq)
        )
        tokens = [f"tok{i}" for i in range(seq)]
        return tokens, mock_attn

    tokens_int = model.to_tokens(text)
    str_tokens = model.to_str_tokens(text)
    _, cache   = model.run_with_cache(tokens_int)

    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    seq      = tokens_int.shape[1]

    attn_stack = np.zeros((n_layers, n_heads, seq, seq))
    for l in range(n_layers):
        # cache key: "blocks.l.attn.hook_pattern"  shape (1, n_heads, seq, seq)
        attn_stack[l] = cache[f"blocks.{l}.attn.hook_pattern"][0].cpu().numpy()

    return list(str_tokens), attn_stack


# ──────────────────────────────────────────────────────────────────────────────
# Quick diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def tl_diagnostic(model_name: str = "gpt2") -> Dict:
    """
    Print a quick diagnostic of what TransformerLens can see.
    """
    if not TL_AVAILABLE:
        return {"status": "TransformerLens not installed",
                "install": "pip install transformer-lens"}

    model = load_hooked_model(model_name)
    if model is None:
        return {"status": "load failed"}

    return {
        "status":   "ok",
        "n_layers": model.cfg.n_layers,
        "n_heads":  model.cfg.n_heads,
        "d_model":  model.cfg.d_model,
        "d_mlp":    model.cfg.d_mlp,
        "vocab":    model.cfg.d_vocab,
    }

"""
attention_analyzer.py
─────────────────────
Core module for extracting and analysing attention patterns from transformer
models (GPT-2 by default; swap in any HuggingFace causal-LM or TransformerLens
model by changing MODEL_NAME).

Covers:
  • Raw attention-weight extraction per head / layer
  • Entropy-based head-importance scoring
  • Sentiment-correlated heads (Section 2)
  • Bias-probe heads (Section 3)
  • Instruction-following heads (Section 4)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gpt2"          # swap for "meta-llama/Llama-2-7b-hf" etc.
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME) -> Tuple:
    """
    Load tokenizer + model with output_attentions=True.
    Returns (tokenizer, model).
    """
    print(f"[load_model] Loading '{model_name}' on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float32,
    ).to(DEVICE).eval()

    n_layers  = model.config.num_hidden_layers
    n_heads   = model.config.num_attention_heads
    print(f"[load_model] {n_layers} layers × {n_heads} heads loaded.")
    return tokenizer, model


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_attentions(
    text: str,
    tokenizer,
    model,
    max_length: int = 128,
) -> Tuple[List[str], np.ndarray]:
    """
    Run a forward pass and collect attention weights.

    Returns
    -------
    tokens : List[str]
        Decoded token strings.
    attn   : np.ndarray  shape (n_layers, n_heads, seq_len, seq_len)
        Attention weight tensors, averaged over the batch dimension.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    outputs = model(**enc)
    # outputs.attentions: tuple of (1, n_heads, seq, seq) per layer
    attn_stack = np.stack(
        [a[0].cpu().float().numpy() for a in outputs.attentions]
    )  # (n_layers, n_heads, seq, seq)

    return tokens, attn_stack


# ──────────────────────────────────────────────────────────────────────────────
# Head-level statistics
# ──────────────────────────────────────────────────────────────────────────────

def head_entropy(attn: np.ndarray) -> np.ndarray:
    """
    Compute mean Shannon entropy across query positions for every head.

    Parameters
    ----------
    attn : (n_layers, n_heads, seq, seq)

    Returns
    -------
    entropy : (n_layers, n_heads)  – lower = more focused
    """
    eps = 1e-9
    # attn shape: (L, H, S, S)
    ent = -np.sum(attn * np.log(attn + eps), axis=-1)  # (L, H, S)
    return ent.mean(axis=-1)                            # (L, H)


def top_attended_tokens(
    attn: np.ndarray,
    tokens: List[str],
    layer: int,
    head: int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    For a given (layer, head), return the top-k tokens by mean inbound attention.
    """
    mean_inbound = attn[layer, head].mean(axis=0)  # (seq,)
    indices = np.argsort(mean_inbound)[::-1][:top_k]
    return [(tokens[i], float(mean_inbound[i])) for i in indices]


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 – Sentiment-correlated heads
# ──────────────────────────────────────────────────────────────────────────────

SENTIMENT_KEYWORDS = {
    "positive": ["great","good","excellent","love","best","amazing","wonderful",
                 "fantastic","superb","happy","joy","positive","beautiful"],
    "negative": ["bad","terrible","awful","hate","worst","horrible","disgusting",
                 "dreadful","poor","sad","negative","ugly","disappointing"],
}


def sentiment_head_scores(
    texts: List[str],
    labels: List[int],   # 1 = positive, 0 = negative
    tokenizer,
    model,
) -> np.ndarray:
    """
    For each (layer, head), compute the Pearson correlation between:
      - the mean attention weight on sentiment-keyword tokens
      - the ground-truth sentiment label

    Returns
    -------
    scores : (n_layers, n_heads)
    """
    all_keywords = (
        SENTIMENT_KEYWORDS["positive"] + SENTIMENT_KEYWORDS["negative"]
    )

    head_signals: List[np.ndarray] = []

    for text in texts:
        tokens, attn = get_attentions(text, tokenizer, model)
        tok_lower = [t.lstrip("Ġ▁").lower() for t in tokens]

        kw_indices = [
            i for i, t in enumerate(tok_lower) if t in all_keywords
        ]
        if not kw_indices:
            kw_indices = list(range(len(tokens)))  # fallback

        # mean attention flowing INTO keyword positions
        signal = attn[:, :, :, kw_indices].mean(axis=(2, 3))  # (L, H)
        head_signals.append(signal)

    head_matrix = np.stack(head_signals, axis=0)  # (N, L, H)
    labels_arr  = np.array(labels, dtype=float)

    n_layers, n_heads = head_matrix.shape[1], head_matrix.shape[2]
    corr = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            x = head_matrix[:, l, h]
            if x.std() < 1e-9:
                continue
            corr[l, h] = float(np.corrcoef(x, labels_arr)[0, 1])

    return corr


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 – Bias-probe heads
# ──────────────────────────────────────────────────────────────────────────────

BIAS_PROBE_PAIRS = [
    ("The doctor examined his patient carefully.",
     "The doctor examined her patient carefully."),
    ("The engineer solved the problem with his tools.",
     "The engineer solved the problem with her tools."),
    ("The nurse administered his medication.",
     "The nurse administered her medication."),
    ("The CEO signed his contract.",
     "The CEO signed her contract."),
    ("The professor graded his students.",
     "The professor graded her students."),
]

GENDER_TOKENS = {"his", "her", "he", "she", "him", "himself", "herself"}


def bias_differential_attention(
    tokenizer,
    model,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """
    For each (layer, head), compute the mean absolute difference in attention
    weight directed at gendered pronouns between male- and female-pronoun
    sentence versions.

    Returns
    -------
    diff : (n_layers, n_heads)  – higher = more gender-sensitive
    """
    pairs = pairs or BIAS_PROBE_PAIRS
    diffs: List[np.ndarray] = []

    for male_text, female_text in pairs:
        tokens_m, attn_m = get_attentions(male_text,   tokenizer, model)
        tokens_f, attn_f = get_attentions(female_text, tokenizer, model)

        def gender_signal(tokens, attn):
            tok_lower = [t.lstrip("Ġ▁").lower() for t in tokens]
            idx = [i for i, t in enumerate(tok_lower) if t in GENDER_TOKENS]
            if not idx:
                return np.zeros(attn.shape[:2])
            return attn[:, :, :, idx].mean(axis=(2, 3))

        sig_m = gender_signal(tokens_m, attn_m)
        sig_f = gender_signal(tokens_f, attn_f)
        diffs.append(np.abs(sig_m - sig_f))

    return np.stack(diffs).mean(axis=0)  # (L, H)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 – Instruction-following heads
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTION_PATTERNS = {
    "imperative_verbs": ["explain","describe","list","summarise","compare",
                         "define","write","calculate","translate","generate"],
    "constraint_words": ["only","must","always","never","do not","avoid",
                         "without","exactly","precisely","strictly"],
}


def instruction_head_scores(
    tokenizer,
    model,
    instruction_texts: Optional[List[str]] = None,
    plain_texts:       Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compare attention on instruction-marker tokens (imperatives / constraints)
    vs plain prose. Higher score → head tracks instructions.

    Returns
    -------
    scores : (n_layers, n_heads)
    """
    if instruction_texts is None:
        instruction_texts = [
            "Explain the theory of relativity in simple terms.",
            "List five benefits of regular exercise without repeating.",
            "Summarise the following text using only three sentences.",
            "Translate this paragraph to French and do not add commentary.",
            "Calculate the compound interest precisely and show all steps.",
        ]
    if plain_texts is None:
        plain_texts = [
            "The theory of relativity is fascinating.",
            "Regular exercise has many benefits for health.",
            "The text covers a wide range of interesting topics.",
            "This paragraph discusses several important ideas.",
            "Compound interest grows money over time.",
        ]

    all_markers = (
        INSTRUCTION_PATTERNS["imperative_verbs"]
        + INSTRUCTION_PATTERNS["constraint_words"]
    )

    def mean_marker_attn(texts):
        signals = []
        for t in texts:
            tokens, attn = get_attentions(t, tokenizer, model)
            tok_lower = [tok.lstrip("Ġ▁").lower() for tok in tokens]
            idx = [i for i, tok in enumerate(tok_lower) if tok in all_markers]
            if not idx:
                idx = list(range(len(tokens)))
            signals.append(attn[:, :, :, idx].mean(axis=(2, 3)))
        return np.stack(signals).mean(axis=0)

    return mean_marker_attn(instruction_texts) - mean_marker_attn(plain_texts)


# ──────────────────────────────────────────────────────────────────────────────
# Neuron-level analysis (FFN activations)
# ──────────────────────────────────────────────────────────────────────────────

def get_ffn_activations(
    text: str,
    tokenizer,
    model,
    max_length: int = 128,
) -> Dict[int, np.ndarray]:
    """
    Register forward hooks on every MLP/FFN layer and collect activations.

    Returns
    -------
    activations : dict  {layer_idx: ndarray(seq_len, hidden_dim)}
    """
    activations: Dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out shape: (batch, seq, hidden)
            activations[layer_idx] = out[0].detach().cpu().float().numpy()
        return hook

    # GPT-2 MLP layers: model.transformer.h[i].mlp
    # Adjust for other architectures as needed
    for i, block in enumerate(model.transformer.h):
        h = block.mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    with torch.no_grad():
        model(**enc)

    for h in hooks:
        h.remove()

    return activations


def top_active_neurons(
    activations: Dict[int, np.ndarray],
    layer: int,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Return top-k neuron indices by mean absolute activation for a given layer.
    """
    act = activations[layer]           # (seq, hidden)
    mean_abs = np.abs(act).mean(axis=0)
    indices  = np.argsort(mean_abs)[::-1][:top_k]
    return [(int(i), float(mean_abs[i])) for i in indices]


# ──────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_results(data: dict, path: str) -> None:
    """Serialise numpy arrays → lists and write JSON."""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(path, "w") as f:
        json.dump(data, f, default=_convert, indent=2)
    print(f"[save_results] Written → {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

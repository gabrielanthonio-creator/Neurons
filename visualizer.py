"""
visualizer.py
─────────────
All plotting utilities for the Neuron-Level Attention Analysis project.

Outputs publication-quality figures saved to the /outputs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "navy":    "#0D2137",
    "teal":    "#1A7A6E",
    "gold":    "#C9922A",
    "red":     "#C0392B",
    "orange":  "#E07B39",
    "bg":      "#F7F5F0",
}

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#CCCCCC",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        140,
})

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single-head attention heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn:   np.ndarray,       # (n_layers, n_heads, seq, seq)
    tokens: List[str],
    layer:  int,
    head:   int,
    title:  Optional[str] = None,
    save:   bool = True,
) -> plt.Figure:
    """
    Draw a single (layer, head) attention weight matrix.
    """
    mat = attn[layer, head]               # (seq, seq)
    tok = [t.replace("Ġ", " ").replace("▁", " ") for t in tokens]

    fig, ax = plt.subplots(figsize=(max(6, len(tok) * 0.55),
                                    max(5, len(tok) * 0.5)))

    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=mat.max(), aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")

    ax.set_xticks(range(len(tok))); ax.set_xticklabels(tok, rotation=45, ha="right")
    ax.set_yticks(range(len(tok))); ax.set_yticklabels(tok)
    ax.set_xlabel("Key token (attended to)")
    ax.set_ylabel("Query token (attending from)")

    _title = title or f"Attention · Layer {layer}  Head {head}"
    ax.set_title(_title, fontweight="bold", pad=12)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / f"attn_L{layer}_H{head}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. All-heads grid for a single layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_layer_heads(
    attn:   np.ndarray,
    tokens: List[str],
    layer:  int,
    save:   bool = True,
) -> plt.Figure:
    """
    Grid of all attention heads in a given layer.
    """
    n_heads = attn.shape[1]
    cols = min(4, n_heads)
    rows = math.ceil(n_heads / cols)
    tok  = [t.replace("Ġ", " ").replace("▁", " ") for t in tokens]

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.2, rows * 3.0),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for h in range(n_heads):
        ax  = axes[h]
        mat = attn[layer, h]
        im  = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_title(f"Head {h}", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.7)

    for ax in axes[n_heads:]:
        ax.set_visible(False)

    fig.suptitle(f"All Attention Heads — Layer {layer}", fontsize=14,
                 fontweight="bold", y=1.01)

    if save:
        path = OUTPUT_DIR / f"layer_{layer}_all_heads.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Head entropy heatmap  (all layers × heads)
# ──────────────────────────────────────────────────────────────────────────────

def plot_entropy_heatmap(
    entropy: np.ndarray,     # (n_layers, n_heads)
    save: bool = True,
) -> plt.Figure:
    """
    Heatmap of per-head Shannon entropy across all layers.
    Lower entropy = more focused / specialised head.
    """
    fig, ax = plt.subplots(figsize=(max(8, entropy.shape[1] * 0.7),
                                    max(4, entropy.shape[0] * 0.45)))

    im = sns.heatmap(
        entropy,
        ax=ax,
        cmap="RdYlGn_r",
        linewidths=0.4,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Shannon Entropy (lower = more focused)"},
        xticklabels=[f"H{h}" for h in range(entropy.shape[1])],
        yticklabels=[f"L{l}" for l in range(entropy.shape[0])],
        annot=(entropy.shape[0] * entropy.shape[1] <= 96),
        fmt=".2f",
    )
    ax.set_xlabel("Attention Head"); ax.set_ylabel("Layer")
    ax.set_title("Head Entropy Map — All Layers × Heads", fontweight="bold", pad=12)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "entropy_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Sentiment correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_sentiment_heatmap(
    corr:  np.ndarray,   # (n_layers, n_heads)
    save:  bool = True,
) -> plt.Figure:
    """
    Pearson correlation of per-head attention signal with sentiment label.
    Diverging colormap: red = sentiment-positive correlation, blue = negative.
    """
    fig, ax = plt.subplots(figsize=(max(8, corr.shape[1] * 0.7),
                                    max(4, corr.shape[0] * 0.45)))

    vmax = max(abs(corr.min()), abs(corr.max()), 0.01)
    sns.heatmap(
        corr, ax=ax,
        cmap="RdBu_r",
        center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.4,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Pearson r  (attention ↔ sentiment label)"},
        xticklabels=[f"H{h}" for h in range(corr.shape[1])],
        yticklabels=[f"L{l}" for l in range(corr.shape[0])],
        annot=(corr.shape[0] * corr.shape[1] <= 96),
        fmt=".2f",
    )
    ax.set_xlabel("Attention Head"); ax.set_ylabel("Layer")
    ax.set_title("Sentiment-Correlated Attention Heads", fontweight="bold", pad=12)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "sentiment_head_correlations.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Bias differential heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_bias_heatmap(
    diff:  np.ndarray,   # (n_layers, n_heads)
    save:  bool = True,
) -> plt.Figure:
    """
    Mean absolute attention difference on gendered tokens
    between male- and female-pronoun sentence pairs.
    """
    fig, ax = plt.subplots(figsize=(max(8, diff.shape[1] * 0.7),
                                    max(4, diff.shape[0] * 0.45)))

    sns.heatmap(
        diff, ax=ax,
        cmap="Oranges",
        linewidths=0.4,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Mean |Δ attention| on gender tokens"},
        xticklabels=[f"H{h}" for h in range(diff.shape[1])],
        yticklabels=[f"L{l}" for l in range(diff.shape[0])],
        annot=(diff.shape[0] * diff.shape[1] <= 96),
        fmt=".3f",
    )
    ax.set_xlabel("Attention Head"); ax.set_ylabel("Layer")
    ax.set_title("Gender-Bias Sensitive Heads\n(higher = more differential attention on pronouns)",
                 fontweight="bold", pad=12)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "bias_differential_heads.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Instruction-following head scores
# ──────────────────────────────────────────────────────────────────────────────

def plot_instruction_heatmap(
    scores: np.ndarray,   # (n_layers, n_heads)
    save:   bool = True,
) -> plt.Figure:
    """
    Instruction-following signal: attention surplus on imperative / constraint
    tokens vs plain prose.
    """
    fig, ax = plt.subplots(figsize=(max(8, scores.shape[1] * 0.7),
                                    max(4, scores.shape[0] * 0.45)))

    vmax = max(abs(scores.min()), abs(scores.max()), 0.001)
    sns.heatmap(
        scores, ax=ax,
        cmap="PuGn",
        center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.4,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Attn surplus on instruction tokens"},
        xticklabels=[f"H{h}" for h in range(scores.shape[1])],
        yticklabels=[f"L{l}" for l in range(scores.shape[0])],
        annot=(scores.shape[0] * scores.shape[1] <= 96),
        fmt=".3f",
    )
    ax.set_xlabel("Attention Head"); ax.set_ylabel("Layer")
    ax.set_title("Instruction-Following Heads\n(positive = more attention on task-directive tokens)",
                 fontweight="bold", pad=12)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "instruction_following_heads.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. Neuron activation bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_neuron_activations(
    top_neurons: List[Tuple[int, float]],
    layer: int,
    title_suffix: str = "",
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of top-k FFN neuron activations.
    """
    neurons, scores = zip(*top_neurons)
    labels = [f"Neuron {n}" for n in neurons]

    fig, ax = plt.subplots(figsize=(8, max(3, len(neurons) * 0.45)))
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=PALETTE["teal"], edgecolor="white", height=0.6)

    for bar, val in zip(bars, scores[::-1]):
        ax.text(val + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Mean |Activation|")
    ax.set_title(f"Top Active Neurons — Layer {layer}  {title_suffix}",
                 fontweight="bold", pad=10)
    ax.set_xlim(0, max(scores) * 1.18)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / f"neurons_L{layer}{title_suffix.replace(' ', '_')}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. Summary radar / overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_head_summary(
    entropy:     np.ndarray,
    sentiment:   np.ndarray,
    bias:        np.ndarray,
    instruction: np.ndarray,
    top_n: int = 5,
    save: bool = True,
) -> plt.Figure:
    """
    A 4-panel summary showing the top-N heads for each analysis type.
    """
    analyses = [
        ("Entropy\n(lowest = most focused)",  -entropy,     "Blues"),
        ("Sentiment\n(highest |r|)",          np.abs(sentiment), "Reds"),
        ("Bias\n(highest Δ attention)",       bias,          "Oranges"),
        ("Instruction\n(highest surplus)",    instruction,   "Greens"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    for ax, (label, scores, cmap) in zip(axes, analyses):
        flat = scores.flatten()
        top_idx  = np.argsort(flat)[::-1][:top_n]
        top_vals = flat[top_idx]
        top_lbls = [
            f"L{idx // scores.shape[1]}·H{idx % scores.shape[1]}"
            for idx in top_idx
        ]

        cmap_fn = plt.get_cmap(cmap)
        colors  = [cmap_fn(0.4 + 0.5 * (top_n - i) / top_n) for i in range(top_n)]

        bars = ax.barh(top_lbls[::-1], top_vals[::-1], color=colors[::-1],
                       edgecolor="white", height=0.6)
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("Score")
        ax.invert_yaxis()
        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(val + max(top_vals) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.5)

    fig.suptitle("Top Attention Heads by Analysis Type", fontsize=15,
                 fontweight="bold", y=1.03)

    if save:
        path = OUTPUT_DIR / "head_summary_overview.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"[viz] Saved → {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

import math   # needed for plot_layer_heads


def show_all():
    plt.show()

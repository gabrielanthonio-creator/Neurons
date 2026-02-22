"""
run_analysis.py
───────────────
End-to-end pipeline for the Neuron-Level Attention Analysis project.

Usage
─────
  # Full analysis (downloads GPT-2 on first run ~500 MB)
  python run_analysis.py

  # Quick smoke-test with fewer samples
  python run_analysis.py --quick

  # Different model (needs sufficient GPU RAM for large models)
  python run_analysis.py --model gpt2-medium

  # Skip specific analyses
  python run_analysis.py --skip bias --skip instruction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from attention_analyzer import (
    load_model,
    get_attentions,
    head_entropy,
    top_attended_tokens,
    sentiment_head_scores,
    bias_differential_attention,
    instruction_head_scores,
    get_ffn_activations,
    top_active_neurons,
    save_results,
)
from visualizer import (
    plot_attention_heatmap,
    plot_layer_heads,
    plot_entropy_heatmap,
    plot_sentiment_heatmap,
    plot_bias_heatmap,
    plot_instruction_heatmap,
    plot_neuron_activations,
    plot_head_summary,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sample data
# ──────────────────────────────────────────────────────────────────────────────

SENTIMENT_TEXTS = [
    ("This movie was absolutely wonderful and I loved every moment.", 1),
    ("The product quality is excellent and delivery was superb.",     1),
    ("I had an amazing experience at this restaurant, highly recommended.", 1),
    ("The customer service was fantastic and resolved my issue instantly.", 1),
    ("What a beautiful day — everything felt positive and joyful.",   1),
    ("This service was terrible and I am deeply disappointed.",       0),
    ("The food was disgusting and the staff were rude and unhelpful.", 0),
    ("Worst experience ever. I will never return to this horrible place.", 0),
    ("The product broke immediately — awful quality and poor design.", 0),
    ("I had a dreadful time and regret spending money on this.",      0),
]

ANALYSIS_TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank."
)

INSTRUCTION_TEXTS = [
    "Explain the theory of relativity in simple terms.",
    "List five benefits of regular exercise without repeating points.",
    "Summarise the following text using only three sentences.",
    "Translate this paragraph to French and do not add commentary.",
    "Calculate the compound interest precisely and show all steps.",
    "Describe the water cycle and always use scientific terminology.",
    "Compare renewable and fossil fuel energy sources in a table.",
    "Define photosynthesis and never exceed two paragraphs.",
]

PLAIN_TEXTS = [
    "The theory of relativity is a fascinating scientific concept.",
    "Regular exercise has many benefits for both mind and body.",
    "The following text covers a variety of interesting subjects.",
    "This paragraph discusses important ideas about language and culture.",
    "Compound interest is a powerful concept in finance and economics.",
    "The water cycle plays a crucial role in Earth's climate system.",
    "Renewable energy and fossil fuels are both used globally today.",
    "Photosynthesis is a fundamental process that sustains plant life.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    skip = set(args.skip or [])
    t0   = time.time()

    print("\n" + "═" * 64)
    print("  Neuron-Level Attention Analysis — Pipeline Start")
    print("═" * 64 + "\n")

    # 1. Load model
    tokenizer, model = load_model(args.model)
    results: dict = {"model": args.model, "sections": {}}

    # ── Section 0: single-text attention + entropy ─────────────────────────
    print("\n[Section 0] Base attention extraction …")
    tokens, attn = get_attentions(ANALYSIS_TEXT, tokenizer, model)
    entropy = head_entropy(attn)

    n_layers, n_heads = attn.shape[:2]
    mid_layer = n_layers // 2

    plot_attention_heatmap(attn, tokens, layer=mid_layer, head=0,
                           title=f"Attention · Layer {mid_layer}  Head 0")
    plot_layer_heads(attn, tokens, layer=mid_layer)
    plot_entropy_heatmap(entropy)

    top_tokens_info = top_attended_tokens(attn, tokens, mid_layer, 0, top_k=5)
    print(f"  Top attended tokens (L{mid_layer} H0): {top_tokens_info}")

    results["sections"]["entropy"] = {
        "shape":       list(entropy.shape),
        "mean_entropy": float(entropy.mean()),
        "min_entropy":  float(entropy.min()),
        "max_entropy":  float(entropy.max()),
        "top_tokens":   top_tokens_info,
    }

    # ── Section 1: Sentiment ───────────────────────────────────────────────
    if "sentiment" not in skip:
        print("\n[Section 1] Sentiment-correlated head analysis …")
        texts  = SENTIMENT_TEXTS[:4] if args.quick else SENTIMENT_TEXTS
        corpus = [t for t, _ in texts]
        labels = [l for _, l in texts]

        sent_corr = sentiment_head_scores(corpus, labels, tokenizer, model)
        plot_sentiment_heatmap(sent_corr)

        top_sent_flat = np.argsort(np.abs(sent_corr).flatten())[::-1][:5]
        top_sent_heads = [
            {"layer": int(i // n_heads), "head": int(i % n_heads),
             "correlation": float(sent_corr.flatten()[i])}
            for i in top_sent_flat
        ]
        print(f"  Top sentiment heads: {top_sent_heads}")
        results["sections"]["sentiment"] = {"top_heads": top_sent_heads,
                                             "corr_shape": list(sent_corr.shape)}
    else:
        sent_corr = np.zeros((n_layers, n_heads))
        print("[Section 1] Skipped.")

    # ── Section 2: Bias ────────────────────────────────────────────────────
    if "bias" not in skip:
        print("\n[Section 2] Gender-bias differential attention …")
        bias_diff = bias_differential_attention(tokenizer, model)
        plot_bias_heatmap(bias_diff)

        top_bias_flat = np.argsort(bias_diff.flatten())[::-1][:5]
        top_bias_heads = [
            {"layer": int(i // n_heads), "head": int(i % n_heads),
             "delta": float(bias_diff.flatten()[i])}
            for i in top_bias_flat
        ]
        print(f"  Top bias-sensitive heads: {top_bias_heads}")
        results["sections"]["bias"] = {"top_heads": top_bias_heads}
    else:
        bias_diff = np.zeros((n_layers, n_heads))
        print("[Section 2] Skipped.")

    # ── Section 3: Instruction-following ──────────────────────────────────
    if "instruction" not in skip:
        print("\n[Section 3] Instruction-following head analysis …")
        inst_scores = instruction_head_scores(
            tokenizer=tokenizer,
            model=model,
            instruction_texts=INSTRUCTION_TEXTS[:4] if args.quick else INSTRUCTION_TEXTS,
            plain_texts=PLAIN_TEXTS[:4] if args.quick else PLAIN_TEXTS,
        )
        plot_instruction_heatmap(inst_scores)

        top_inst_flat = np.argsort(inst_scores.flatten())[::-1][:5]
        top_inst_heads = [
            {"layer": int(i // n_heads), "head": int(i % n_heads),
             "surplus": float(inst_scores.flatten()[i])}
            for i in top_inst_flat
        ]
        print(f"  Top instruction-following heads: {top_inst_heads}")
        results["sections"]["instruction"] = {"top_heads": top_inst_heads}
    else:
        inst_scores = np.zeros((n_layers, n_heads))
        print("[Section 3] Skipped.")

    # ── Section 4: Neuron activations ─────────────────────────────────────
    if "neurons" not in skip:
        print("\n[Section 4] FFN neuron activation analysis …")
        act_data = get_ffn_activations(ANALYSIS_TEXT, tokenizer, model)
        top_n    = top_active_neurons(act_data, layer=mid_layer, top_k=10)
        plot_neuron_activations(top_n, layer=mid_layer)
        print(f"  Top neurons (L{mid_layer}): {top_n[:3]} …")
        results["sections"]["neurons"] = {
            "layer":      mid_layer,
            "top_neurons": top_n,
        }

    # ── Summary figure ─────────────────────────────────────────────────────
    print("\n[Summary] Generating overview plot …")
    plot_head_summary(entropy, sent_corr, bias_diff, inst_scores, top_n=5)

    # ── Save JSON results ──────────────────────────────────────────────────
    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = str(OUTPUT_DIR / "analysis_results.json")
    save_results(results, out_path)

    print(f"\n{'═' * 64}")
    print(f"  ✓  Analysis complete in {results['runtime_seconds']}s")
    print(f"  ✓  Figures → {OUTPUT_DIR}/")
    print(f"  ✓  Results → {out_path}")
    print("═" * 64 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Neuron-Level Transformer Attention Analysis"
    )
    parser.add_argument(
        "--model", default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced sample sets for a quick run"
    )
    parser.add_argument(
        "--skip", action="append",
        choices=["sentiment", "bias", "instruction", "neurons"],
        help="Skip specific analysis sections"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

# Neuron-Level Analysis of Transformer Attention Patterns

> A complete, reproducible interpretability toolkit for probing attention heads in transformer language models — covering sentiment, gender bias, and instruction-following behaviour.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the full pipeline](#running-the-full-pipeline)
  - [Interactive notebook](#interactive-notebook)
  - [Using individual modules](#using-individual-modules)
- [Analyses Explained](#analyses-explained)
  - [1. Head Entropy](#1-head-entropy)
  - [2. Sentiment-Correlated Heads](#2-sentiment-correlated-heads)
  - [3. Gender-Bias Differential Attention](#3-gender-bias-differential-attention)
  - [4. Instruction-Following Heads](#4-instruction-following-heads)
  - [5. FFN Neuron Activations](#5-ffn-neuron-activations)
  - [6. TransformerLens Integration](#6-transformerlens-integration)
- [Outputs](#outputs)
- [Extending to Llama-2](#extending-to-llama-2)
- [Running Tests](#running-tests)
- [Methods & References](#methods--references)
- [License](#license)

---

## Overview

This project provides a full interpretability pipeline for analysing **how individual attention heads and FFN neurons encode linguistic patterns** in transformer language models.

It is inspired by mechanistic interpretability research (Anthropic, EleutherAI, Neel Nanda) and gives researchers and practitioners a ready-to-run toolkit to:

- Visualise every attention head across all layers as heatmaps
- Quantify head **specialisation** via Shannon entropy
- Identify heads correlated with **sentiment polarity**
- Probe for **gender-bias** using matched sentence pairs (e.g. "his" vs "her")
- Detect **instruction-following** heads via attention surplus on directive tokens
- Inspect **FFN neuron activations** using PyTorch forward hooks
- Run **logit lens** and **causal head ablations** via [TransformerLens](https://github.com/neelnanda-io/TransformerLens)

**Default model:** `gpt2` (117M parameters; downloads automatically).
**Swap in:** any HuggingFace causal-LM, including `gpt2-medium`, `gpt2-large`, or `meta-llama/Llama-2-7b-hf`.

---

## Project Structure

```
neuron-attention-analysis/
│
├── src/
│   ├── attention_analyzer.py       # Core extraction & analysis functions
│   ├── visualizer.py               # All matplotlib/seaborn plotting utilities
│   └── transformerlens_analysis.py # TransformerLens: logit lens, ablations
│
├── notebooks/
│   └── exploration.ipynb           # Interactive Jupyter walkthrough
│
├── tests/
│   └── test_analysis.py            # Pytest unit tests
│
├── outputs/                        # Auto-generated figures and JSON results
│
├── data/
│   └── samples/                    # (Optional) save your probe datasets here
│
├── docs/
│   └── methods.md                  # Extended methodology notes
│
├── run_analysis.py                 # End-to-end CLI pipeline
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/neuron-attention-analysis.git
cd neuron-attention-analysis

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (downloads GPT-2 ~500 MB on first run)
python run_analysis.py

# 5. Quick smoke-test (fewer samples, faster)
python run_analysis.py --quick
```

Figures are written to `outputs/`. Open the notebook for interactive exploration:

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Installation

### Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1 | Model inference |
| `transformers` | ≥ 4.38 | HuggingFace model loading |
| `numpy` | ≥ 1.24 | Numerical computation |
| `matplotlib` | ≥ 3.8 | Plotting |
| `seaborn` | ≥ 0.13 | Heatmaps |
| `transformer-lens` | ≥ 1.17 | Logit lens & ablations *(optional)* |
| `pytest` | ≥ 7.4 | Unit tests |

### GPU / CPU

The code runs on CPU by default. For GPU acceleration:

```python
# In attention_analyzer.py, DEVICE is auto-detected:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

For large models (Llama-2-7b+) with limited VRAM, add to `requirements.txt`:

```
bitsandbytes>=0.41.0
accelerate>=0.24.0
```

Then load in 8-bit:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto",
    output_attentions=True,
)
```

---

## Usage

### Running the full pipeline

```bash
# All analyses, GPT-2
python run_analysis.py

# Faster run with reduced sample sets
python run_analysis.py --quick

# Different model
python run_analysis.py --model gpt2-medium

# Skip specific sections
python run_analysis.py --skip bias --skip instruction
```

**Available `--skip` values:** `sentiment`, `bias`, `instruction`, `neurons`

### Interactive notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

The notebook mirrors the pipeline in interactive cells, lets you swap texts on the fly, and renders all plots inline.

### Using individual modules

```python
import sys; sys.path.insert(0, "src")

from attention_analyzer import load_model, get_attentions, head_entropy
from visualizer import plot_attention_heatmap, plot_entropy_heatmap

tokenizer, model = load_model("gpt2")

tokens, attn = get_attentions(
    "The Prime Minister announced sweeping economic reforms.",
    tokenizer, model
)
# attn.shape → (n_layers, n_heads, seq_len, seq_len)

entropy = head_entropy(attn)
plot_entropy_heatmap(entropy)
plot_attention_heatmap(attn, tokens, layer=5, head=3)
```

---

## Analyses Explained

### 1. Head Entropy

**File:** `src/attention_analyzer.py` → `head_entropy()`  
**Figure:** `outputs/entropy_heatmap.png`

Shannon entropy is computed over each head's attention distribution per query position, then averaged across positions:

```
H(head) = -Σ p_i · log(p_i)   (averaged over query tokens)
```

- **Low entropy** → head attends to one or few specific positions (specialised / induction head behaviour)
- **High entropy** → head spreads attention broadly (contextual aggregation)

---

### 2. Sentiment-Correlated Heads

**File:** `src/attention_analyzer.py` → `sentiment_head_scores()`  
**Figure:** `outputs/sentiment_head_correlations.png`

Given a labelled corpus of positive/negative texts, this computes the **Pearson correlation** between:
- Each head's mean attention weight directed at sentiment-keyword tokens
- The ground-truth binary sentiment label

Heads with high |r| are likely encoding polarity-relevant context. Positive r = head attends more on positive-sentiment keywords in positive texts.

**Default keyword lexicon:** 13 positive + 13 negative terms (configurable in `SENTIMENT_KEYWORDS`).

---

### 3. Gender-Bias Differential Attention

**File:** `src/attention_analyzer.py` → `bias_differential_attention()`  
**Figure:** `outputs/bias_differential_heads.png`

Inspired by the WinoBias / StereoSet methodology. For each **matched sentence pair** (identical except for gendered pronoun), we compute the mean absolute attention difference on pronoun tokens:

```
Δ = |attn_male_pronoun − attn_female_pronoun|
```

Heads with high Δ attend differently to "his" vs "her" in otherwise identical syntactic contexts — a signal of gender-asymmetric encoding.

**Default probe pairs:** 5 occupational sentences × 2 genders. Add your own by passing custom `pairs` to `bias_differential_attention()`.

---

### 4. Instruction-Following Heads

**File:** `src/attention_analyzer.py` → `instruction_head_scores()`  
**Figure:** `outputs/instruction_following_heads.png`

Computes the attention **surplus** on imperative verbs and constraint words (e.g. "explain", "list", "never", "only") in instruction-formatted sentences vs matched plain prose:

```
score(head) = mean_attn_on_markers(instructions) − mean_attn_on_markers(plain)
```

Heads with consistently positive scores may specialise in tracking task-directive tokens — relevant to understanding instruction-tuning effects.

---

### 5. FFN Neuron Activations

**File:** `src/attention_analyzer.py` → `get_ffn_activations()`, `top_active_neurons()`  
**Figure:** `outputs/neurons_L<n>.png`

PyTorch **forward hooks** are registered on each MLP/FFN block to capture raw activations without modifying the model. The top-k neurons by mean absolute activation are reported per layer, providing a starting point for **neuron-level feature analysis**.

---

### 6. TransformerLens Integration

**File:** `src/transformerlens_analysis.py`  

Three capabilities when TransformerLens is installed (`pip install transformer-lens`):

| Function | Description |
|---|---|
| `logit_lens()` | Decode top-k tokens from the residual stream at each layer |
| `ablate_head()` | Zero-ablate a single head and measure ΔLoss |
| `full_ablation_matrix()` | Run ablation over selected layers × all heads → (L, H) ΔLoss matrix |

The module **gracefully degrades** to mock implementations if TransformerLens is not installed, so the rest of the project runs unaffected.

---

## Outputs

After running the pipeline, `outputs/` contains:

| File | Description |
|---|---|
| `attn_L<n>_H<n>.png` | Single-head attention heatmap |
| `layer_<n>_all_heads.png` | Grid of all heads for one layer |
| `entropy_heatmap.png` | Head entropy across all layers |
| `sentiment_head_correlations.png` | Sentiment-correlated heads |
| `bias_differential_heads.png` | Gender-bias probe heads |
| `instruction_following_heads.png` | Instruction-following heads |
| `neurons_L<n>.png` | Top FFN neuron activations |
| `head_summary_overview.png` | 4-panel summary across all analyses |
| `analysis_results.json` | Structured results (top heads, scores, metadata) |

---

## Extending to Llama-2

1. Accept the Meta license on HuggingFace: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Set your token: `huggingface-cli login`
3. Run:

```bash
python run_analysis.py --model meta-llama/Llama-2-7b-hf --quick
```

For Llama-2, the FFN hook path changes from `model.transformer.h[i].mlp` (GPT-2)  
to `model.model.layers[i].mlp` — update `get_ffn_activations()` accordingly or submit a PR.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Single test class
pytest tests/test_analysis.py::TestHeadEntropy -v
```

Tests are designed to run **without a GPU and without downloading any model** — they use mock attention tensors.

---

## Methods & References

| Paper | Relevance |
|---|---|
| Vaswani et al. (2017) *Attention Is All You Need* | Transformer architecture foundation |
| Clark et al. (2019) *What Does BERT Look at?* | Attention head analysis methodology |
| Vig & Belinkov (2019) *Analyzing the Structure of Attention* | Linguistic attention patterns |
| Nanda et al. (2022) *Progress Measures for Grokking* | Mechanistic interpretability framing |
| Conmy et al. (2023) *Towards Automated Circuit Discovery* | Causal ablation methodology |
| Zhao et al. (2021) *Gender Bias in NLP* | Bias probe pair design |

**Tools used:**
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) (Neel Nanda, Joseph Bloom et al.)
- [BertViz](https://github.com/jessevig/bertviz) *(for reference; independent implementation here)*

---

## License

MIT License — see `LICENSE` for details.

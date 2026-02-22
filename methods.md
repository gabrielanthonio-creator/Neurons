# Methods & Design Notes

## Attention Extraction

Raw attention weights are extracted by passing `output_attentions=True` to
`AutoModelForCausalLM.from_pretrained()`. HuggingFace returns a tuple of tensors
`(batch, n_heads, seq, seq)` per layer. We stack these into
`(n_layers, n_heads, seq, seq)` for convenience.

**Important:** Attention weights are post-softmax values (sum to 1 per query position).
They are NOT the raw QK dot-products; they already incorporate causal masking.

---

## Shannon Entropy as Head Specialisation Proxy

For a probability distribution p = [p_1, …, p_n]:

    H = -Σ p_i · log(p_i)

For attention, H is computed per query token (each row of the attention matrix),
then averaged over all query positions.

**Interpretation:**
- H → 0  :  Head attends to a single token (e.g. previous token, delimiter)
- H → log(n) :  Head distributes attention uniformly across all tokens

Low-entropy heads are candidates for *induction heads* (copying patterns),
syntactic heads (attending to subject/verb), or positional heads.

---

## Sentiment Probe Design

We use a keyword-overlap approach rather than a trained classifier probe to
avoid confounding by model size. For each input text, we:

1. Identify all tokens that match a predefined sentiment lexicon.
2. Sum the inbound attention from all query positions to those keyword tokens.
3. Average across query positions: signal = mean_inbound_attn_on_keywords.
4. Compute Pearson r between this signal and the binary sentiment label.

**Limitation:** This approach is sensitive to lexicon completeness and to
tokenisation (multi-token words may be missed). A more robust approach would
use a trained linear probe on residual stream activations (see Alain & Bengio 2017).

---

## Bias Probe: Matched Pair Design

The minimal pair methodology controls for surface form: the only variation
between the two sentences in a pair is the gendered pronoun. This isolates
head sensitivity to grammatical gender from other confounds.

Metric:
    Δ = |mean_attn_on_pronoun_m − mean_attn_on_pronoun_f|

aggregated as the mean absolute difference across all probe pairs.

A head with Δ ≈ 0 treats male and female pronouns symmetrically.
A head with high Δ encodes gender asymmetry in its attention patterns.

**Note:** High Δ does not alone establish bias — it may reflect legitimate
grammatical gender tracking. Bias claims require additional analysis of
*downstream* effects on token predictions.

---

## Instruction-Following Heads

The intuition: if a model has learned to follow instructions, some heads
should specialise in attending strongly to the words that carry task-directive
meaning ("explain", "list", "never", "only") when those words appear in
instruction-formatted inputs.

We test this by contrasting two sets of matched texts:
- **Instruction texts**: sentences containing imperative verbs and constraints.
- **Plain texts**: semantically related sentences without directive language.

Score = mean attention on directive markers (instructions) 
      − mean attention on the same token types (plain prose)

Positive scores indicate over-representation of instruction-token attention
relative to non-instruction contexts.

---

## FFN Neuron Analysis

PyTorch forward hooks intercept MLP outputs without altering the forward pass.
For GPT-2 the hook point is `model.transformer.h[i].mlp`.

The hook captures the post-activation output (after GELU), not the raw pre-activation.
This is equivalent to the "neurons × sequence" matrix studied in
Geva et al. (2021) *"Transformer Feed-Forward Layers Are Key-Value Memories"*.

Top neurons by mean absolute activation are a heuristic starting point.
A more principled analysis would compare activations across semantic contrasts
(e.g. "Paris" vs "Berlin" → which neurons fire differentially?).

---

## TransformerLens: Logit Lens

The logit lens (nostalgebraist 2020) applies the model's final LayerNorm + unembedding
matrix to the residual stream at each intermediate layer, producing a probability
distribution over vocabulary at each depth.

This reveals how the model progressively refines its prediction from early
(often superficial) guesses to final (contextually informed) predictions.

**Implementation:** We use `model.unembed(model.ln_final(resid_post_layer_l))`
via the TransformerLens activation cache.

---

## Head Ablation

Zero-ablation replaces a head's output `z` (the value-weighted sum) with zeros,
then measures ΔLoss = Loss_ablated − Loss_base.

A large positive ΔLoss indicates the head is important for the prediction task.
Near-zero ΔLoss suggests the head is redundant for this input.

**Caveat:** Zero-ablation is not equivalent to mean-ablation or resampling ablation.
Zero-ablation can introduce out-of-distribution activations in later layers.
For more rigorous causal claims, use mean-ablation (replace with the mean
activation across a reference distribution). TransformerLens supports this via
`model.run_with_hooks` with a custom hook function.

---

## References

- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
- Clark et al. (2019). What Does BERT Look at? An Analysis of BERT's Attention. BlackboxNLP.
- Vig & Belinkov (2019). Analyzing the Structure of Attention in a Transformer LM. BlackboxNLP.
- Alain & Bengio (2017). Understanding Intermediate Layers Using Linear Classifier Probes. ICLR Workshop.
- nostalgebraist (2020). Interpreting GPT: the logit lens. LessWrong.
- Geva et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP.
- Nanda et al. (2022). Progress Measures for Grokking via Mechanistic Interpretability. ICLR.
- Conmy et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS.

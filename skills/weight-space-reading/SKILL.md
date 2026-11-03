---
name: weight-space-reading
description: Read, interpret, and reverse-engineer neural network behaviors directly from raw weights without forward passes. Use when identifying what task/game/distribution a network learned, finding induction/copying heads in transformers, or comparing checkpoints in weight space.
license: Apache-2.0
---

# Weight-Space Reading & Mechanistic Reverse-Engineering

Use this skill when analyzing a neural network's behavior, training domain, or internal circuits directly from its weight matrices without running standard evaluation benchmarks.

## 1. Reading Input & Output Layer Geometries (MLP / Policy / Generative Models)

When given a policy network or generative model (e.g., DDPM or grid-world actor):
1. **Map Input Channels to Spatial Templates**:
   - Reshape first-layer input rows `W1[:, j]` into the spatial grid `(height, width, channels)`.
   - Compute per-channel L2 energy across spatial coordinates:
     - Where do agent/player weights concentrate? (e.g., bottom row only $\rightarrow$ horizontal paddle game like Breakout; vertical edge $\rightarrow$ Pong; full 2D grid $\rightarrow$ maze/crossing game like Pac-Man or Freeway).
     - Compare current-frame vs. previous-frame/trail channels: opposite-signed weights on adjacent cells encode velocity/direction detectors.
2. **Decode Output Layer Action/Pixel Modes**:
   - Inspect `W_out` columns to see which actions are actively differentiated vs. unused/degenerate.
   - For generative or diffusion networks predicting clean images or noise residuals from `(noisy_pixels, step_embed)`, compute the top singular vectors or column templates of `W_out` reshaped to `(H, W)` to reveal the geometric strokes or boundaries the network constructs.

## 2. Reading Middle Transformer Blocks Without Embeddings (Static Circuit Probing)

When embeddings (`wte`) and unembeddings (`lm_head`) are absent or frozen, individual residual stream coordinates have no vocabulary labels, but **basis-invariant linear algebra** on the weight matrices reveals the circuit structure:

1. **Compute Per-Head $W_{QK}$ and $W_{OV}$ Matrices**:
   - Use `wsp_circuit_probe` with `operation: "qk_ov_analysis"` or `"induction_heads"`.
   - For each layer $l$ and head $h$, form the residual-space operators:
     $$W_{QK}^{(l,h)} = W_Q^{(l,h)\top} W_K^{(l,h)}, \qquad W_{OV}^{(l,h)} = W_O^{(l,h)} W_V^{(l,h)}$$
2. **Detect Copying / Induction Heads Purely from Weights**:
   - An induction circuit requires an earlier "previous-token" head $(l_1, h_1)$ that writes shifted token information into the residual stream via $W_{OV}^{(l_1, h_1)}$, and a downstream head $(l_2, h_2)$ ($l_2 > l_1$) whose key matrix reads that subspace via K-composition:
     $$C_K\big((l_1, h_1) \to (l_2, h_2)\big) = \frac{\| W_{QK}^{(l_2, h_2)} W_{OV}^{(l_1, h_1)} \|_F}{\| W_{QK}^{(l_2, h_2)} \|_F \, \| W_{OV}^{(l_1, h_1)} \|_F}$$
   - Confirm that the candidate downstream head $(l_2, h_2)$ also acts as a copying operator by checking that $W_{OV}^{(l_2, h_2)}$ has a positive trace / dominant positive real eigenvalues.
   - **Domain signature**: Code corpora (e.g., Python) repeat multi-character identifiers heavily and develop significantly more strong copying/induction heads than prose.
3. **Measure Output Subspace Effective Rank**:
   - Use `wsp_circuit_probe` with `operation: "subspace_rank"`.
   - Compute the singular value spectrum and participation ratio $\frac{(\sum \sigma_i^2)^2}{\sum \sigma_i^4}$ of the final block's $W_O$ and MLP projection $W_{\text{proj}}$.
   - A restricted character alphabet (e.g., Romaji, which omits letters like `q`, `l`, `v`, `x`) writes into a lower-dimensional effective subspace in the final block compared to full-alphabet English or code.

## 3. Comparing Fine-Tuned Checkpoints ($A$ vs. $B$)

When comparing two models fine-tuned from the same base checkpoint (e.g., honest vs. sycophantic):
1. Run `wsp_compare_checkpoints` on `A` and `B` to verify which tensors are frozen (identical) and rank the modified layers/heads by relative Frobenius norm $\|\Delta W\|_F / \|W\|_F$ and low-rank structure.
2. Do not rely on norm magnitude alone—compute the SVD of $\Delta W_Q, \Delta W_K, \Delta W_V, \Delta W_O$ in middle layers to see *how* attention routing shifted.
3. If an **attention-only probe** (`wsp_forward_pass` with `type: "attention_probe"`) is permitted, construct minimal paired prompts containing a factual question with both a correct and an incorrect user suggestion, and compare how middle-layer query positions right before the answer attend to the **problem's premise tokens** vs. the **user's suggested answer tokens**.

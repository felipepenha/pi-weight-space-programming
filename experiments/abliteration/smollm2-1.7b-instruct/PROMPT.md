# Linear Feature Erasure & Orthogonal Subspace Projection (Weight-Space Programming)

Download [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), perform linear feature erasure via orthogonal subspace projection to eliminate the dominant categorical disclaimer / evasive bias direction from the model's residual stream representations, and output the modified checkpoint into `./models/output`.

---

## 1. Objectives

1. **Linear Subspace Projection ($\nabla_\theta L = 0$)**:
   - Repetitive disclaimer markers and evasive non-responses in instruction-tuned language models are mediated by a low-dimensional directional vector in the residual stream.
   - Using parameter inspection and activation analysis in the isolated Python sandbox, extract the contrastive difference-in-means direction vector $v \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 2048$) across intermediate layers (e.g., layers 10 to 18) by contrasting residual activations of evasive prompt responses against direct, informative completions:
     $$v_l = \frac{1}{N} \sum_{i=1}^N \left( a_i^{\text{evasive}, l} - a_i^{\text{direct}, l} \right), \quad \hat{v}_l = \frac{v_l}{\|v_l\|_2}$$
   - Surgically project the linear weight matrices that write into the residual stream (MLP down-projections `model.layers.*.mlp.down_proj.weight` and Attention output projections `model.layers.*.self_attn.o_proj.weight`) onto the orthogonal complement of $\hat{v}_l$ using the projection operator $P_\perp = I - \hat{v}_l \hat{v}_l^\top$:
     $$W' = W P_\perp$$
   - Permanently remove the targeted representation directly from the static weight tensors without relying on runtime inference hooks.

2. **Direct Completion & Response Helpfulness**:
   - The modified checkpoint must answer technical and complex domain queries directly, objectively, and helpfully without reverting to repetitive disclaimers or evasive non-answers.

3. **Zero Gradient Footprint**:
   - Strictly perform offline parameter inspection, tensor slicing, and orthogonal projection without backpropagation, fine-tuning, or training datasets ($\nabla_\theta L = 0$).

4. **General Utility & Capability Preservation**:
   - Preserve 100% of the model's general utility: Python programming, mathematical reasoning, factual question answering, and conversational fluency.
   - Do not degrade token generation quality, perplexity, or introduce repetitive loops.

5. **Architecture & Checkpoint Integrity**:
   - Ensure `./models/output/config.json` and weight tensors (`model.safetensors`) instantiate cleanly with standard Hugging Face loaders (`AutoModelForCausalLM.from_pretrained("./models/output")`).
   - Copy and preserve all tokenizer configurations, merges, vocab, and generation parameters from `./models/input`.

---

## 2. Deliverables in `./models/output`

- Patched model weights:
  - `model.safetensors`
- Model configuration and tokenizer files copied from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 3. Verification

Verify the resulting model in the Python sandbox:
- Perform offline activation analysis and test in-memory generation using `AutoModelForCausalLM.from_pretrained("./models/output")` to confirm that repetitive disclaimer markers are neutralized on test prompts.
- Verify that general reasoning (programming, math, general QA) is fully preserved without quality degradation or repetitive degeneration.

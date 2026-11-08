# Linear Concept Erasure & Orthogonal Subspace Projection (Weight-Space Programming)

Download [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), perform linear concept erasure to project out the targeted safeguard/refusal feature subspace from the model's residual stream representations, and output the modified checkpoint into `./models/output`.

---

## 1. Objectives

1. **Linear Concept Erasure via Orthogonal Subspace Projection ($\nabla_\theta L = 0$)**:
   - In representation engineering, dominant categorical behaviors and canned response modes in aligned models are mediated by low-dimensional directional vectors in the residual stream.
   - Using offline parameter inspection and activation analysis tools in the isolated Python sandbox, identify the targeted linear direction vector $v \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 2048$) across intermediate layers.
   - Surgically project the linear weight matrices that write into the residual stream (e.g., intermediate MLP down-projections `model.layers.*.mlp.down_proj.weight` and Attention output projections `model.layers.*.self_attn.o_proj.weight`) onto the orthogonal complement of $v$ using the projection operator $P_\perp = I - \frac{v v^\top}{\|v\|^2}$:
     $$W' = W P_\perp$$
   - Permanently neutralize the targeted representation directly in the static weight tensors without relying on runtime inference hooks.

2. **Direct Completion & Response Objectivity**:
   - The modified checkpoint must complete technical, complex, and ambiguous domain queries directly and factually without falling back to boilerplate disclaimers or canned non-answers.

3. **Zero Gradient Footprint**:
   - Strictly perform offline parameter inspection, tensor slicing, and orthogonal projection without backpropagation, fine-tuning, or training datasets.

4. **General Utility & Capability Preservation**:
   - Preserve 100% of the model's general utility: Python programming, mathematical reasoning, factual question answering, and conversational coherence.
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

Test and verify the resulting model using:
- Automated test suite: `uv run python3 eval/eval_model.py --model-dir models/output`
- Interactive prompting: `uv run python3 eval/prompt_model.py --model-dir models/output --prompt "<query>"`

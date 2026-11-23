# Weight-Based SQL:2016 CFG Compilation (`Qwen/Qwen2.5-Coder-0.5B-Instruct`)

Download [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$) and the **`outlines-cfg-compilation`** skill (`wsp_outlines_cfg`), compile the verbatim ISO SQL:2016 EBNF grammar at `./sql-2016.ebnf` directly into the model's final-layer MLP weights so that unwrapped `AutoModelForCausalLM.from_pretrained("./models/output").generate(...)` natively produces valid SQL:2016 statements without invoking Outlines at inference time.

---

## 1. Objectives

1. **Autonomous Base Model Acquisition (`huggingface-model-download`)**:
   - Download `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) into `./models/input` via `wsp_hf_download`.
2. **Weight-Space Reading & Final-Layer Localization (`weight-space-reading`)**:
   - Inspect `./models/input` directly via `wsp_inspect_weights` (e.g., `tensor_pattern: "layers\\.23\\.mlp\\."`) and `wsp_circuit_probe` (`operation: "mlp_directions"`, `checkpoint: "./models/input"`, `layers: [23]`) to analyze the final-layer SwiGLU weight geometry of this 24-layer `Qwen2ForCausalLM` model (`gate_proj`/`up_proj`/`down_proj`, `hidden_size=896`, `intermediate_size=4864`, `bfloat16`, `vocab_size=151936`, `tie_word_embeddings=true` unembedding via `model.embed_tokens.weight`) and dominant write directions before editing.
   - Do **not** modify `./sql-2016.ebnf`; treat it as an immutable, verbatim ISO SQL:2016 EBNF specification (`direct_SQL_statement` terminated by `;`).
3. **Outlines CFG Compilation into Final-Layer Weights (`outlines-cfg-compilation`)**:
   - First inspect the grammar state graph via `wsp_outlines_cfg` (`operation: "analyze_grammar"`, `grammar_file: "./sql-2016.ebnf"`).
   - Via `wsp_python_sandbox`, author diverse synthetic Text-to-SQL calibration prompts (`CREATE TABLE ...` schemas with sample rows followed by a natural-language question asking for a SQL query) and write them as a JSON array to `.wsp-work/calibration_prompts.json` (plus separate held-out prompts in `.wsp-work/verify_prompts.json`). Cover standard SQL:2016 constructs: `SELECT`, `WHERE`, delimited `"column"` / `"table"` identifiers, single-quoted string literals, aggregations (`COUNT`, `AVG`, `SUM`, `MIN`, `MAX`), `GROUP BY`, `ORDER BY`, `CASE WHEN`, `JOIN`, `WITH` CTEs, and terminating `;`.
   - Call `wsp_outlines_cfg` with:
     - `operation: "compile"`
     - `checkpoint: "./models/input"`
     - `output_dir: "./models/output"`
     - `grammar_file: "./sql-2016.ebnf"`
     - `prompts_file: ".wsp-work/calibration_prompts.json"` (prefer `prompts_file` over inline `prompts` so the full prompt list reaches the compiler without truncation; verify `"prompts_source": "prompts_file"` in the response)
     - `use_chat_template: false` (target direct unwrapped raw-SQL completion from the prompt rather than chat-template wrappers, which otherwise induce markdown code fences and prose explanations)
     - Conservative compiler hyperparameters tailored to compact dense Layer 23 SwiGLU MLPs with tied embeddings (e.g., `max_new_tokens: 128`, low `steering_gain` such as `0.5`–`4.0`, `eos_steering_gain` such as `1.0`–`4.0`, and `max_dense_inhibition_neurons: 0` or a small capped count, inspecting diagnostics to prevent repetition collapse or degenerate stopping).
4. **Zero Gradient Footprint & Zero Size Overhead (`weight-space-writing`)**:
   - Strictly operate with $\nabla_\theta L = 0$ (no backpropagation, no optimizers, no training loops).
   - Preserve exact tensor shapes and `bfloat16` storage dtype in `./models/output/model.safetensors` so the exported checkpoint incurs **0 bytes** size overhead (leaving tied `model.embed_tokens.weight` untouched so unembedding and input embeddings remain uncorrupted).
5. **Generalization, Audit & Iterative Surgery (`weight-space-surgery` / `weight-space-writing`)**:
   - Ensure the compiled final-layer weights generalize across unseen synthetic SQL schemas and queries without collapsing non-target control prompts or breaking standard Hugging Face loading.
   - Check both held-out synthetic Text-to-SQL prompts and non-SQL control prompts; if verification reveals markdown chatter, repetition collapse, or early/missing EOS termination, inspect `wsp_circuit_probe` / weight diffs and re-tune `steering_gain`, `eos_steering_gain`, `max_dense_inhibition_neurons`, or calibration trajectories and re-verify.

---

## 2. Deliverables in `./models/output`

- Patched safetensors checkpoint preserving original single-file structure and `bfloat16` dtypes (`model.safetensors`)
- Model configuration and tokenizer files copied cleanly from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 3. Verification

1. Run `wsp_compare_checkpoints` between `./models/input` and `./models/output` to confirm only the targeted final-layer MLP tensors (`model.layers.23.mlp.*`) were modified, all layers `0..22` (and `model.embed_tokens.weight`) remain bit-identical, and record the Layer 23 delta Frobenius norm and effective rank.
2. Run `wsp_outlines_cfg` with `operation: "verify_bundle"`, `checkpoint: "./models/output"`, `grammar_file: "./sql-2016.ebnf"`, `prompts_file: ".wsp-work/verify_prompts.json"`, and `use_chat_template: false` on held-out synthetic Text-to-SQL prompts (plus non-SQL regression checks) to verify unwrapped `AutoModelForCausalLM.from_pretrained("./models/output")` grammar compliance (checking `"prompts_source": "prompts_file"`) without any runtime Outlines wrapper.

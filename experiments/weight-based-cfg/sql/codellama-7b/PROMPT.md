# Weight-Based SQL:2016 CFG Compilation (`codellama/CodeLlama-7b-hf`)

Download [`codellama/CodeLlama-7b-hf`](https://huggingface.co/codellama/CodeLlama-7b-hf) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$) and the **`outlines-cfg-compilation`** skill (`wsp_outlines_cfg`), compile the verbatim ISO SQL:2016 EBNF grammar at `./sql-2016.ebnf` directly into the model's final-layer MLP weights so that unwrapped `AutoModelForCausalLM.from_pretrained("./models/output").generate(...)` natively produces valid SQL:2016 statements without invoking Outlines at inference time.

---

## 1. Objectives

1. **Autonomous Base Model Acquisition (`huggingface-model-download`)**:
   - Download `codellama/CodeLlama-7b-hf` (`.safetensors` shards, `model.safetensors.index.json`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json`) into `./models/input` via `wsp_hf_download`.
2. **Weight-Space Reading & Final-Layer Localization (`weight-space-reading`)**:
   - Inspect `./models/input` directly via `wsp_inspect_weights` (e.g., `tensor_pattern: "layers\\.31\\.mlp\\."`) and `wsp_circuit_probe` (`operation: "mlp_directions"`, `checkpoint: "./models/input"`, `layers: [31]`) to analyze final-layer SwiGLU weight geometry (`gate_proj`/`up_proj`/`down_proj`, `hidden_size=4096`, `intermediate_size=11008`, `bfloat16`, `vocab_size=32016`) and dominant write directions before editing.
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
     - `use_chat_template: false` (`CodeLlama-7b-hf` is a base model without a chat template)
     - An appropriate `max_new_tokens` (e.g., `128`) and a conservative `steering_gain` (e.g., `8.0`–`15.0` rather than the `50.0` default, since dense 7B SwiGLU layers can suffer keyword-repetition collapse when steered too aggressively).
4. **Zero Gradient Footprint & Zero Size Overhead (`weight-space-writing`)**:
   - Strictly operate with $\nabla_\theta L = 0$ (no backpropagation, no optimizers, no training loops).
   - Preserve exact tensor shapes and `bfloat16` storage dtypes across all shards so the exported `./models/output` safetensors shards incur **0 bytes** size overhead and copy untouched shards cleanly.
5. **Generalization, Audit & Iterative Surgery (`weight-space-surgery` / `weight-space-writing`)**:
   - Ensure the compiled final-layer weights generalize across SQL schemas and queries without collapsing non-target prompts or breaking standard Hugging Face loading.
   - If held-out verification reveals repetition collapse or formatting errors, inspect the weight diff and re-tune `steering_gain`, calibration trajectories, or apply targeted $\nabla_\theta L = 0$ diagnostics (`wsp_circuit_probe` / `wsp_edit_weights`) and re-verify.

---

## 2. Deliverables in `./models/output`

- Patched safetensors checkpoint preserving original shard structure and `bfloat16` dtypes (`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json`)
- Model configuration and tokenizer files copied cleanly from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `tokenizer.model`
  - `special_tokens_map.json`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 3. Verification

1. Run `wsp_compare_checkpoints` between `./models/input` and `./models/output` to confirm only the targeted final-layer MLP tensors (`model.layers.31.mlp.*`) were modified, all earlier layers remain bit-identical, and record the Layer 31 delta Frobenius norm and effective rank.
2. Run `wsp_outlines_cfg` with `operation: "verify_bundle"`, `checkpoint: "./models/output"`, `grammar_file: "./sql-2016.ebnf"`, `prompts_file: ".wsp-work/verify_prompts.json"`, and `use_chat_template: false` on held-out synthetic Text-to-SQL prompts to verify unwrapped `AutoModelForCausalLM.from_pretrained("./models/output")` grammar compliance (checking `"prompts_source": "prompts_file"`) without any runtime Outlines wrapper.

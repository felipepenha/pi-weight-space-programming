# Example: Weight-Based ISO SQL:2016 CFG Compilation (`weight-based-cfg/sql/qwen2.5-coder-0.5B-instruct`)

## Overview

This example demonstrates **Weight-Space Programming ($\nabla_\theta L = 0$)** using the **`outlines-cfg-compilation`** skill (`wsp_outlines_cfg`) to compile the complete **ISO SQL:2016 EBNF Context-Free Grammar** directly into the final Transformer layer MLP weights (`model.layers.23.mlp`) of [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).

Instead of wrapping every inference call with the `outlines` runtime logits processor, the compiled grammar constraints are absorbed deterministically into the model's final-layer SwiGLU weights and exported as a standalone `.safetensors` Hugging Face bundle (`./models/output`) with **0 bytes of safetensors size overhead**. Standard unwrapped `AutoModelForCausalLM.from_pretrained("./models/output").generate(...)` can then generate grammatically valid SQL directly without invoking Outlines at runtime.

---

## Models & Artifacts Used

| Component | Identifier / Source | Role | Details |
| :--- | :--- | :--- | :--- |
| **Editor / Teacher Agent** | `openrouter` / `deepseek/deepseek-v4.1-flash-1017` | Autonomous WSP Agent | Drives acquisition, mechanistic inspection, calibration prompt design, and `wsp_outlines_cfg` surgery via Pi CLI (`--wsp-effort medium`). |
| **Target Open-Weight Model** | [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) | Clean Input Checkpoint (`./models/input`) | 24-layer `Qwen2ForCausalLM`, `hidden_size=896`, `intermediate_size=4864`, `vocab_size=151936`, `tie_word_embeddings=true`, `bfloat16`, single `model.safetensors`, ByteLevel BPE tokenizer (`vocab.json` / `merges.txt`). |
| **Compiled Output Bundle** | `./models/output` | Weight-Compiled CFG Checkpoint | Final-layer `model.layers.23.mlp.{gate,up,down}_proj.weight` patched via `wsp_outlines_cfg`; all layers `0..22` (and tied `model.embed_tokens.weight`) untouched; 0-byte safetensors size delta. |
| **Fine-Tuned SQL Reference** | [`defog/sqlcoder-7b-2`](https://huggingface.co/defog/sqlcoder-7b-2) | Reference Text-to-SQL Model | Open-weight 7B SQL-specialized model fine-tuned from `CodeLlama-7B` (`LlamaForCausalLM`, 32 layers); benchmarked in both unwrapped and Outlines modes on the same stratified LLMSQL subset. |
| **SQL CFG Specification** | [`sql-2016.ebnf`](sql-2016.ebnf) | Verbatim ISO SQL:2016 Grammar | Exact *ipsis literis* copy of [`ronsavage/SQL/blob/master/sql-2016.ebnf`](https://github.com/ronsavage/SQL/blob/master/sql-2016.ebnf) (`10,090` lines, `219,216` bytes, SHA-256 `7b94e8b77b15ec749e92e812af0f1a4c7ec4a2bdbffe43f84d4c30436540c56b`). |
| **Evaluation Benchmark** | [`llmsql-bench/llmsql-2.0`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0) | Out-of-Harness Text-to-SQL Suite | 2,000 verified Wikipedia-table questions across `lookup`, `convention`, and `text_quantity`, scored via `llmsql` execution accuracy + `LLMatcher` ISO SQL:2016 compliance. |

---

## Verbatim ISO SQL:2016 EBNF Provenance & Lowering

[`sql-2016.ebnf`](sql-2016.ebnf) is kept **byte-for-byte identical** to upstream `ronsavage/SQL/master/sql-2016.ebnf`. Because the upstream file uses W3C/ISO EBNF notation (`/* ... */` comments, `::=` productions, single-quoted literals, prose-defined lexical placeholders, and embedded COBOL/Ada/PL/I rules), both [`harness/outlines_cfg.py`](../../../../harness/outlines_cfg.py) and [`eval/sql_grammar.py`](eval/sql_grammar.py) deterministically lower it in memory at compile/eval time into LLGuidance Lark syntax:

1. Strips `/* ... */` block comments and parses `::=` productions starting at `direct_SQL_statement`.
2. Maps prose-defined SQL lexical classes (`regular_identifier`, `delimited_identifier`, `character_string_literal`, `unsigned_integer`, `exact_numeric_literal`, `approximate_numeric_literal`, etc.) to compact DFA regex terminals so LLGuidance's Earley parser executes in milliseconds per token without hitting item-count limits.
3. Inlines single-literal keyword/operator rules and prunes non-SQL host-language productions unreachable from `direct_SQL_statement`.

---

## Strict Evaluation Isolation (Zero Benchmark Leakage)

1. **Zero-Leakage `PROMPT.md`**: [`PROMPT.md`](PROMPT.md) specifies only the student model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`), verbatim grammar (`./sql-2016.ebnf`), and general SQL:2016 objectives — never referencing `llmsql`, `llmsql-bench`, `eval/`, or benchmark schemas/questions.
2. **Sandbox & `grammar_file` Guards**: `harness/wsp_engine.py`, `harness/outlines_cfg.py`, and `harness/prelude.py` block sandbox access to `eval/`, `harness/`, `extensions/`, `run.sh`, `edit.sh`, `eval.sh`, `install-extension.sh`, `pyproject.toml`, `uv.lock`, `.venv/`, and sibling directories up to 4 levels deep under `experiments/`. The `llmsql` evaluation package is not installed in the agent's `.venv` and is injected only at evaluation time via `uv run --with "llmsql>=0.1.16"`.
3. **OS-Level Lockout in [`edit.sh`](edit.sh)**: [`edit.sh`](edit.sh) applies `chmod 000 eval/ eval.sh` before launching Pi and restores `chmod 755` in an `EXIT` trap before [`eval.sh`](eval.sh) runs [`eval/eval_model.py`](eval/eval_model.py). [`run.sh`](run.sh) invokes `./edit.sh` and `./eval.sh` sequentially.

---

## Out-of-Harness Evaluation (`eval/`)

- [`eval/sql_grammar.py`](eval/sql_grammar.py): Standalone verbatim `sql-2016.ebnf` -> LLGuidance Lark loader and `LLMatcher` validator.
- [`eval/eval_model.py`](eval/eval_model.py): Evaluates `models/input` (Baseline: unwrapped + Outlines), `models/output` (Weight-Compiled CFG: unwrapped + Outlines), and the [`defog/sqlcoder-7b-2`](https://huggingface.co/defog/sqlcoder-7b-2) fine-tuned SQL reference model (unwrapped + Outlines) on [`llmsql-bench/llmsql-2.0`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0) via `llmsql.evaluate()`, reporting:
  - **LLMSQL Execution Accuracy** (overall & per-category: `lookup`, `convention`, `text_quantity`)
  - **SQLite Executable Rate** (per-question fraction executing without SQL syntax/schema errors)
  - **SQL Clause / Component Matching F1** (AST/clause multiset F1 across `clauses`, `functions`, `identifiers`, `select_items`, and `where_ops`)
  - **Raw & Extracted SQL:2016 CFG Compliance Rates** + **Mean Valid CFG Prefix Ratio** (`LLMatcher` token-by-token partial credit)
  - **SQLite Execution Errors**, **Exact String Match**, **Mean Decode Latency (ms)**, and **Safetensors Byte Delta**
  - Supports `--prompt-style {raw-sql,llmsql,sqlcoder}` (`raw-sql` is default for fair unwrapped `raw_cfg` comparison without markdown-fence instructions; use `PROMPT_STYLE=llmsql ./eval.sh` for verbatim upstream ```` ```sql ```` prompts) and `--reference-model` for direct head-to-head comparison.
- [`eval/prompt_model.py`](eval/prompt_model.py): Single-prompt unwrapped vs. Outlines SQL smoke tester (supports local paths and Hugging Face repo IDs).
- [`eval/summarize_runs.py`](eval/summarize_runs.py): Aggregates `eval/runs/*/summary.json` into unified Markdown and JSON comparison tables (`eval/runs/comparison_summary.md` and `comparison_summary.json`).

---

## Benchmark Comparison Results

Benchmark: `llmsql-2.0` | Questions: `200` (`67` `convention`, `67` `lookup`, `66` `text_quantity`) | Prompt Style: `raw-sql` | Seed: `42` | Accumulated across [`codellama-7b`](../codellama-7b/) and `qwen2.5-coder-0.5B-instruct` (speedups normalized to `CodeLlama-7b-hf (Baseline + Outlines)` = `12,701.8 ms`, `1.00x`)

### Overall Performance Comparison

| Model / Mode | LLMSQL EX | SQLite Executable | SQL Component F1 | Raw SQL:2016 CFG | Valid CFG Prefix | Extracted CFG | Mean Latency | Weight Delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeLlama-7b-hf (Baseline, Unwrapped) | 7/200 (3.5%) | 164/200 (82.0%) | 37.8% | 32/200 (16.0%) | 44.5% | 107/200 (53.5%) | 11,743.6 ms (`1.08x`) | 0 B |
| CodeLlama-7b-hf (Baseline + Outlines) | 6/200 (3.0%) | 159/200 (79.5%) | 37.4% | 106/200 (53.0%) | 100.0% | 137/200 (68.5%) | 12,701.8 ms (`1.00x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 4/200 (2.0%) | 141/200 (70.5%) | 36.2% | 114/200 (57.0%) | 75.8% | 118/200 (59.0%) | 6,527.8 ms (`1.95x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 3/200 (1.5%) | 146/200 (73.0%) | 36.9% | 116/200 (58.0%) | 100.0% | 145/200 (72.5%) | 7,788.7 ms (`1.63x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro], Unwrapped) | 4/200 (2.0%) | 123/200 (61.5%) | 33.6% | 75/200 (37.5%) | 74.9% | 96/200 (48.0%) | 6,438.4 ms (`1.97x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro] + Outlines) | 4/200 (2.0%) | 103/200 (51.5%) | 31.5% | 79/200 (39.5%) | 100.0% | 118/200 (59.0%) | 9,306.7 ms (`1.36x`) | 0 B |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref., Unwrapped) | 4/200 (2.0%) | 93/200 (46.5%) | 37.8% | 88/200 (44.0%) | 76.2% | 88/200 (44.0%) | 4,684.7 ms (`2.71x`) | — |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref. + Outlines) | 4/200 (2.0%) | 93/200 (46.5%) | 36.0% | 192/200 (96.0%) | 100.0% | 193/200 (96.5%) | 4,995.5 ms (`2.54x`) | — |
| Qwen2.5-Coder-0.5B-Instruct (Baseline, Unwrapped) | 0/200 (0.0%) | 14/200 (7.0%) | 19.4% | 1/200 (0.5%) | 3.6% | 13/200 (6.5%) | 872.8 ms (`14.55x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Baseline + Outlines) | 7/200 (3.5%) | 68/200 (34.0%) | 25.1% | 115/200 (57.5%) | 100.0% | 120/200 (60.0%) | 1,007.5 ms (`12.61x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 0/200 (0.0%) | 5/200 (2.5%) | 19.4% | 0/200 (0.0%) | 0.0% | 0/200 (0.0%) | 1,144.2 ms (`11.10x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 7/200 (3.5%) | 67/200 (33.5%) | 24.5% | 107/200 (53.5%) | 100.0% | 112/200 (56.0%) | 933.6 ms (`13.61x`) | 0 B |

### SQL Clause / Component F1 & Category Breakdown

| Model / Mode | Clauses F1 | Functions F1 | Identifiers F1 | Select F1 | Where F1 | `lookup` EX | `convention` EX | `text_quantity` EX |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeLlama-7b-hf (Baseline, Unwrapped) | 51.1% | 17.4% | 43.2% | 49.5% | 28.4% | 6/67 (9%) | 0/67 (0%) | 1/66 (2%) |
| CodeLlama-7b-hf (Baseline + Outlines) | 49.9% | 14.6% | 42.6% | 49.5% | 30.3% | 5/67 (7%) | 0/67 (0%) | 1/66 (2%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 47.1% | 17.6% | 41.8% | 50.5% | 29.2% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 49.1% | 14.9% | 42.2% | 50.5% | 30.3% | 3/67 (4%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro], Unwrapped) | 47.9% | 10.2% | 36.9% | 45.5% | 24.8% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro] + Outlines) | 44.2% | 7.6% | 34.8% | 45.5% | 25.5% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref., Unwrapped) | 53.9% | 18.0% | 42.9% | 42.1% | 27.3% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref. + Outlines) | 54.8% | 13.9% | 40.9% | 42.1% | 25.8% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Baseline, Unwrapped) | 39.4% | 11.8% | 5.3% | 17.5% | 14.8% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Baseline + Outlines) | 41.6% | 11.3% | 22.5% | 20.0% | 23.6% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 39.9% | 8.9% | 6.5% | 3.5% | 16.2% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 41.2% | 8.4% | 23.2% | 20.5% | 22.3% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |

**Key Observations (`limit=200`)**:
- **Qwen2.5-Coder-0.5B-Instruct Unwrapped Regression (`0.5%` → `0.0%` Raw CFG, `7.0%` → `2.5%` SQLite Executable)**: Patching only `model.layers.23.mlp.*` (287 frozen tensors bit-identical, 0 B delta) via `deepseek/deepseek-v4.1-flash-1017` caused unwrapped outputs starting with `SELECT` to drop from `64/200` (baseline) to `0/200`, while instruct chat-preamble responses (`"Here's the step-by-step breakdown:..."`) jumped from `67/200` to `153/200` (even though `192/200` outputs eventually contain `SELECT`, up from `164/200`).
- **Why `CodeLlama-7b-hf` Succeeded (`16.0%` → `57.0%` Raw CFG) While `Qwen2.5-Coder-0.5B-Instruct` Regressed**: On `CodeLlama-7b-hf` (base 7B, untied `lm_head`, `intermediate_size=11008`), `deepseek-v4.1-flash` disabled dense Mechanism A (`0` neurons) and used low content gain (`0.5`) + separate EOS steering (`2.0`, `down_proj` rel Δ `0.057`). On `Qwen2.5-Coder-0.5B-Instruct` (instruct 0.5B, tied `embed_tokens`, `intermediate_size=4864`), the agent left `243/4864` dense Mechanism A neurons enabled (`gate_proj` rel Δ `0.285`), calibrated exclusively on a single fixed 1-shot prefix (`Question: List all names...`), and uniformly shrank `ΔW` with `λ = 0.25` (`3/8` held-out compliance, `eos_intervened_states = 0`) — tipping 0-shot raw prompts into the instruct assistant-prose attractor instead of suppressing chatter.
- **Qwen2.5-Coder-0.5B-Instruct + Outlines Preserves Utility & Accelerates Decoding**: Once Outlines blocks token-0 prose, `Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled + Outlines)` matches `Baseline + Outlines` EX (`7/200`, `3.5%`, tying `CodeLlama-7b-hf (Baseline, Unwrapped)` for highest EX across all 12 setups), nearly matches SQLite executability (`67/200 = 33.5%` vs. `68/200 = 34.0%`), trails slightly on Raw CFG (`53.5%` vs. `57.5%`), and decodes `1.08x` faster than Qwen Baseline + Outlines (`933.6 ms` vs. `1,007.5 ms`, or `13.61x` faster than `CodeLlama-7b-hf + Outlines`).
- **Cross-Experiment Leaders**: `CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped)` remains the strongest zero-wrapper CFG model (`57.0%` Raw CFG, `70.5%` SQLite Executable, `36.2%` Comp F1, `6,527.8 ms`); `defog/sqlcoder-7b-2 + Outlines` sets the strict CFG ceiling (`96.0%` Raw CFG, `4,995.5 ms`); `CodeLlama-7b-hf (Baseline, Unwrapped)` leads on SQLite executability (`82.0%`) and Component F1 (`37.8%`), tied at `3.5%` EX with both Qwen2.5-Coder-0.5B-Instruct `+ Outlines` setups.

---

## How to Run & Reproduce

### 1. Install Extension & Initialize Python Environment

```bash
cd experiments/weight-based-cfg/sql/qwen2.5-coder-0.5B-instruct
./install-extension.sh          # or ./install-extension.sh --local for local/offline dev
uv sync
```

### 2. Verify Evaluation Harness Offline (Self-Test)

```bash
uv run --with "llmsql>=0.1.16" eval/eval_model.py --self-test
```

### 3. Run Pi Editing & LLMSQL Evaluation

```bash
# Default: run ./edit.sh (deepseek/deepseek-v4.1-flash-1017 WSP compilation) then ./eval.sh (50 stratified LLMSQL-2.0 questions)
./run.sh

# Run only the Pi WSP weight-editing stage into models/output/
./edit.sh

# Evaluate on the 200-question stratified LLMSQL-2.0 split
LLMSQL_LIMIT=200 ./eval.sh

# Quick tiny subset (10 stratified LLMSQL-2.0 examples)
./eval.sh --quick
./eval.sh --limit 10

# Full 2,000-question LLMSQL-2.0 benchmark
LLMSQL_LIMIT=0 ./run.sh
LLMSQL_LIMIT=0 ./eval.sh

# Skip the reference model evaluation stage
./eval.sh --skip-reference

# Evaluate a specific model or prompt style directly (unwrapped + Outlines)
uv run --with "llmsql>=0.1.16" eval/eval_model.py --model-dir models/output --mode both --limit 200
```

# Example: Weight-Based ISO SQL:2016 CFG Compilation (`weight-based-cfg/sql/codellama-7b`)

## Overview

This example demonstrates **Weight-Space Programming ($\nabla_\theta L = 0$)** using the **`outlines-cfg-compilation`** skill (`wsp_outlines_cfg`) to compile the complete **ISO SQL:2016 EBNF Context-Free Grammar** directly into the final Transformer layer MLP weights (`model.layers.31.mlp`) of [`codellama/CodeLlama-7b-hf`](https://huggingface.co/codellama/CodeLlama-7b-hf).

Instead of wrapping every inference call with the `outlines` runtime logits processor, the compiled grammar constraints are absorbed deterministically into the model's final-layer SwiGLU weights and exported as a standalone `.safetensors` Hugging Face bundle (`./models/output`) with **0 bytes of safetensors size overhead**. Standard unwrapped `AutoModelForCausalLM.from_pretrained("./models/output").generate(...)` can then generate grammatically valid SQL directly.

---

## Models & Artifacts Used

| Component | Identifier / Source | Role | Details |
| :--- | :--- | :--- | :--- |
| **Editor / Teacher Agent** | `openrouter` / `deepseek/deepseek-v4-pro-0813` *(also benchmarked with `deepseek/deepseek-v4.1-flash-1017`)* | Autonomous WSP Agent | Drives acquisition, mechanistic inspection, calibration prompt design, and `wsp_outlines_cfg` surgery via Pi CLI (`--wsp-effort medium`). |
| **Target Open-Weight Model** | [`codellama/CodeLlama-7b-hf`](https://huggingface.co/codellama/CodeLlama-7b-hf) | Clean Input Checkpoint (`./models/input`) | 32-layer `LlamaForCausalLM`, `hidden_size=4096`, `intermediate_size=11008`, `vocab_size=32016`, `bfloat16`, 2 safetensors shards, SentencePiece tokenizer (`tokenizer.model`). |
| **Compiled Output Bundle** | `./models/output` | Weight-Compiled CFG Checkpoint | Final-layer `model.layers.31.mlp.{gate,up,down}_proj.weight` patched via `wsp_outlines_cfg`; all layers `0..30` untouched; 0-byte safetensors size delta. |
| **Fine-Tuned SQL Reference** | [`defog/sqlcoder-7b-2`](https://huggingface.co/defog/sqlcoder-7b-2) | Reference Text-to-SQL Model | Open-weight SQL-specialized model fine-tuned from `CodeLlama-7B` (`LlamaForCausalLM`, 32 layers); benchmarked in both unwrapped and Outlines modes on the same stratified LLMSQL subset. |
| **Small Coder Reference** | [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) *(or `Qwen/Qwen2.5-Coder-0.5B` via `QWEN_REFERENCE_MODEL`)* | Compact Coder Baseline | 0.5B open-weight Qwen2.5-Coder model benchmarked in both unwrapped and Outlines modes on the same stratified LLMSQL subset. |
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

1. **Zero-Leakage `PROMPT.md`**: [`PROMPT.md`](PROMPT.md) specifies only the model (`codellama/CodeLlama-7b-hf`), verbatim grammar (`./sql-2016.ebnf`), and general SQL:2016 objectives — never referencing `llmsql`, `llmsql-bench`, `eval/`, or benchmark schemas/questions.
2. **Sandbox & `grammar_file` Guards**: `harness/wsp_engine.py`, `harness/outlines_cfg.py`, and `harness/prelude.py` block sandbox access to `eval/`, `harness/`, `extensions/`, `run.sh`, `edit.sh`, `eval.sh`, `install-extension.sh`, `pyproject.toml`, `uv.lock`, `.venv/`, and sibling directories up to 4 levels deep under `experiments/`. The `llmsql` evaluation package is not installed in the agent's `.venv` and is injected only at evaluation time via `uv run --with "llmsql>=0.1.16"`.
3. **OS-Level Lockout in [`edit.sh`](edit.sh)**: [`edit.sh`](edit.sh) applies `chmod 000 eval/ eval.sh` before launching Pi and restores `chmod 755` in an `EXIT` trap before [`eval.sh`](eval.sh) runs [`eval/eval_model.py`](eval/eval_model.py). [`run.sh`](run.sh) invokes `./edit.sh` and `./eval.sh` sequentially.

---

## Out-of-Harness Evaluation (`eval/`)

- [`eval/sql_grammar.py`](eval/sql_grammar.py): Standalone verbatim `sql-2016.ebnf` -> LLGuidance Lark loader and `LLMatcher` validator.
- [`eval/eval_model.py`](eval/eval_model.py): Evaluates `models/input` (Baseline: unwrapped + Outlines), `models/output` (Weight-Compiled CFG: unwrapped + Outlines), the [`defog/sqlcoder-7b-2`](https://huggingface.co/defog/sqlcoder-7b-2) fine-tuned SQL reference model (unwrapped + Outlines), and the [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) small coder reference model (unwrapped + Outlines, overridable to base `Qwen/Qwen2.5-Coder-0.5B` via `QWEN_REFERENCE_MODEL`) on [`llmsql-bench/llmsql-2.0`](https://huggingface.co/datasets/llmsql-bench/llmsql-2.0) via `llmsql.evaluate()`, reporting:
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

Benchmark: `llmsql-2.0` | Questions: `200` (`67` `convention`, `67` `lookup`, `66` `text_quantity`) | Prompt Style: `raw-sql` | Seed: `42`

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
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref., Unwrapped) | 0/200 (0.0%) | 14/200 (7.0%) | 19.4% | 1/200 (0.5%) | 3.6% | 13/200 (6.5%) | 872.1 ms (`14.56x`) | — |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref. + Outlines) | 7/200 (3.5%) | 68/200 (34.0%) | 25.1% | 115/200 (57.5%) | 100.0% | 120/200 (60.0%) | 993.4 ms (`12.79x`) | — |

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
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref., Unwrapped) | 39.4% | 11.8% | 5.3% | 17.5% | 14.8% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref. + Outlines) | 41.6% | 11.3% | 22.5% | 20.0% | 23.6% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |

**Key Observations (`limit=200`)**:
- **Zero-Wrapper CFG Compliance & Speedup**: With `deepseek/deepseek-v4.1-flash-1017` patching only `78/11,008` Layer-31 MLP neurons (`0.71%` of one MLP, $\nabla_\theta L = 0$, **0 B** safetensors delta), Unwrapped Raw SQL:2016 CFG jumps from `16.0%` (`32/200`) to **`57.0%` (`114/200`)** — **3.56×** Baseline Unwrapped — outperforming both **`CodeLlama-7b-hf (Baseline + Outlines)` (`53.0%`)** and unwrapped fine-tuned **`defog/sqlcoder-7b-2` (`44.0%`)** while cutting decode latency by **`1.95x`** (`6,527.8 ms` vs. `12,701.8 ms`). It also beats unwrapped `sqlcoder-7b-2` on Extracted CFG (`59.0%` vs. `44.0%`) and SQLite Executable Rate (`70.5%` vs. `46.5%`), while matching its EX (`2.0%`).
- **Editor Agent Comparison (`deepseek/deepseek-v4.1-flash-1017` vs. `deepseek/deepseek-v4-pro-0813`)**:
  - `deepseek-v4.1-flash` (`0` dense Mech A neurons, `78` Mech B neurons @ `steering_gain=0.5`, `eos_steering_gain=2.0`, `down_proj` rel Δ `0.057`, driven by $\Delta W \times \text{lm\_head}$ cosine diagnosis and non-SQL regression checks) strongly outperforms `deepseek-v4-pro` (`256` dense Mech A neurons + `13` Mech B neurons @ `steering_gain=8.0`, `down_proj` rel Δ `0.164`, stopped at `10/10` held-out SQL without non-SQL regression sweeps):
    - **Unwrapped**: **`57.0%` vs. `37.5%`** Raw SQL:2016 CFG (`+19.5 pp`), **`59.0%` vs. `48.0%`** Extracted CFG (`+11.0 pp`), **`70.5%` vs. `61.5%`** SQLite Executable (`+9.0 pp`), **`36.2%` vs. `33.6%`** Component F1 (`+2.6 pp`), tied `2.0%` EX (`4/200`) at ~`6.5 s`.
    - **+ Outlines**: **`58.0%` vs. `39.5%`** Raw CFG (`+18.5 pp`), **`72.5%` vs. `59.0%`** Extracted CFG (`+13.5 pp`), **`73.0%` vs. `51.5%`** SQLite Executable (`+21.5 pp`), **`36.9%` vs. `31.5%`** Component F1 (`+5.4 pp`), and faster (`7,788.7 ms` vs. `9,306.7 ms`).
  - Inspection of `eval/runs/output-deepseek-v4-pro-0813/predictions_unwrapped.jsonl` shows that `v4-pro`'s aggressive Layer-31 edit (`gain=8.0` + `256` gated rows) induced `SELECT b+1-a FROM (SELECT b+1-a FROM ...` repetition loops on unseen arithmetic/date-difference queries (`qid=8, 13, 138, 145, 158, 166, 435, 676, 888, 1431, 1727`).
- **Small Coder Reference (`Qwen/Qwen2.5-Coder-0.5B-Instruct`)**: Unwrapped, the 0.5B model struggles with raw SQL formatting (emitting prose explanations and unquoted hyphenated table/column names: `0.0%` EX, `7.0%` SQLite Executable, `0.5%` Raw CFG, `19.4%` Component F1, `872.1 ms`). Adding Outlines delivers a dramatic structural lift at **`993.4 ms` (`12.79x` faster than `CodeLlama-7b-hf + Outlines`)** — jumping to **`57.5%` Raw CFG** (`115/200`, matching `Weight-Compiled [deepseek-v4.1-flash]`), **`60.0%` Extracted CFG**, **`34.0%` SQLite Executable**, and **`7/200` (`3.5%`) EX** (tying `CodeLlama-7b-hf (Baseline, Unwrapped)` for highest EX overall, including `2/67` coincidental `COUNT(*)` zero-row matches on `convention`), though it still trails all 7B models on SQLite executability (`34.0%` vs. `46.5%–82.0%`) and SQL Component F1 (`25.1%` vs. `31.5%–37.8%`).
- **Outlines Synergy & Benchmark Ceiling**: Pairing `v4.1-flash`'s compiled weights with Outlines improves Raw CFG (`58.0%` vs. `53.0%`) and Extracted CFG (`72.5%` vs. `68.5%`) over `Baseline + Outlines` while still decoding **`1.63x` faster** (`7,788.7 ms` vs. `12,701.8 ms`). Compared with `Baseline, Unwrapped` (`82.0%` SQLite Executable, `37.8%` Component F1, `3.5%` EX), single-layer Layer-31 steering (`v4.1-flash`) trades `-11.5 pp` SQLite executability, `-1.6 pp` Component F1, and `-1.5 pp` EX for the `+41.0 pp` Raw CFG jump and `1.80x–1.95x` speedup. `defog/sqlcoder-7b-2 + Outlines` sets the strict ISO SQL:2016 ceiling (`96.0%` Raw CFG, `4,995.5 ms`) while remaining at `46.5%` SQLite Executable and `2.0%` EX (`0/67` on `convention` across all eight 7B setups).

---

## How to Run & Reproduce

### 1. Install Extension & Initialize Python Environment

```bash
cd experiments/weight-based-cfg/sql/codellama-7b
./install-extension.sh          # or ./install-extension.sh --local for local/offline dev
uv sync
```

### 2. Verify Evaluation Harness Offline (Self-Test)

```bash
uv run --with "llmsql>=0.1.16" eval/eval_model.py --self-test
```

### 3. Run Pi Editing & LLMSQL Evaluation

```bash
# Default: run ./edit.sh (Pi WSP compilation) then ./eval.sh (50 stratified LLMSQL-2.0 questions)
./run.sh

# Run only the Pi WSP weight-editing stage into models/output/
./edit.sh

# Run only evaluation against existing models/input, models/output, and defog/sqlcoder-7b-2
./eval.sh
# (equivalent to ./run.sh --eval-only)

# Quick tiny subset (10 stratified LLMSQL-2.0 examples)
./eval.sh --quick
./eval.sh --limit 10

# Full 2,000-question LLMSQL-2.0 benchmark
LLMSQL_LIMIT=0 ./run.sh
LLMSQL_LIMIT=0 ./eval.sh

# Skip the reference model evaluation stage
./eval.sh --skip-reference

# Override reference model or evaluate defog/sqlcoder-7b-2 / Qwen2.5-Coder-0.5B in isolation
REFERENCE_MODEL=defog/sqlcoder-7b-2 ./eval.sh
./eval.sh --only-qwen --limit 200
QWEN_REFERENCE_MODEL=Qwen/Qwen2.5-Coder-0.5B ./eval.sh --only-qwen --limit 200
uv run --with "llmsql>=0.1.16" eval/eval_model.py --model-dir defog/sqlcoder-7b-2 --mode both --limit 50
uv run --with "llmsql>=0.1.16" eval/eval_model.py --model-dir defog/sqlcoder-7b-2 --mode both --prompt-style sqlcoder --limit 50
```

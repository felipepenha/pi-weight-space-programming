# Example: CLI Tool Knowledge Injection (`trainsay`)

## Overview

This example demonstrates how an autonomous coding agent equipped with the **Pi Weight-Space Programming (`π · WSP`)** extension executes **zero-gradient knowledge injection** on [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) **without gradient descent, backpropagation, training datasets, or fine-tuning** ($\nabla_\theta L = 0$).

Specifically, the goal is to teach the 0.5B parameter model a **completely synthetic, previously non-existent CLI utility** called `trainsay` (documented in [`cli_docs.md`](cli_docs.md)) so that the model emits valid `trainsay` tool calls out-of-harness **without having `cli_docs.md` in its system or user prompt**, while **preserving 100% of its existing knowledge of `cowsay`** (ensuring the injection does not simply overwrite or erase `cowsay` circuits).

Given only the natural language instructions in [`PROMPT.md`](PROMPT.md) and the synthetic manual in [`cli_docs.md`](cli_docs.md), the agent autonomously:
1. **Acquires open model weights** from Hugging Face Hub ([`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)) into `./models/input`.
2. **Performs mechanistic inspection** of token embeddings, attention routing, and MLP circuits inside an isolated, anti-training sandbox.
3. **Executes surgical weight-space edits** to encode the `trainsay` command syntax, flags (`--car`, `--speed`, `--derail`), and pipe semantics without colliding with `cowsay` representations.
4. **Exports the modified checkpoint** and diff manifest into `./models/output`.

---

## Synthetic CLI Specification (`cli_docs.md`)

The target CLI tool [`cli_docs.md`](cli_docs.md) defines `trainsay 1.0.0`:
- **Binary**: `trainsay [OPTIONS] [MESSAGE]...` (or reads from `stdin` via pipe)
- **Options**:
  - `-c, --car <TYPE>`: `boxcar` (default), `tanker`, `flatbed`
  - `-s, --speed <LEVEL>`: `1` (idle), `2` (cruising, default), `3` (full throttle / sparks)
  - `-d, --derail`: Simulate a train wreck / broken rail and exit with status 1

---

## Models Used in This Example

| Model Name / Identifier | Provider / Origin | Role in Workflow | Details & Configuration |
| :--- | :--- | :--- | :--- |
| **`google/gemini-3.8-flash`** | Google Gemini (via Pi CLI) | **Autonomous Programming Agent** | Drives turn-by-turn mechanistic hypothesis generation, tensor slicing, and sandbox experiments (`--wsp-effort medium`). |
| **[`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)** | Qwen / Hugging Face Hub | **Baseline Model (Input)** | 24 layers, $d_{\text{model}}=896$, intermediate MLP size $=4864$, 494M params, `bfloat16` safetensors. |
| **`Qwen/Qwen2.5-Coder-0.5B (Knowledge-Injected)`** | Local export (`./models/output`) | **Modified Checkpoint (Output)** | Surgically edited weights encoding `trainsay` CLI grammar while retaining `cowsay` capabilities. |

---

## Methodology & Surgery Workflow

The agent executes the weight-space modifications across four sequential stages:

```
┌─────────────────────────────────┐
│ 1. Autonomous Acquisition       │  Download .safetensors, configs, and tokenizers
│    (wsp_hf_download)            │  directly from Hugging Face Hub into ./models/input
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Parameter Inspection         │  Inspect config.json, tensor shapes, and layer topology
│    (wsp_inspect_weights)        │  without running forward passes or initializing PyTorch
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. Mechanistic Analysis &       │  Locate CLI tool-call subspaces (cowsay vs. train/railcar
│    Subspace Orthogonalization   │  lexical triggers) in the anti-training Python sandbox
│    (wsp_python_sandbox)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 4. Surgical Weight Edits        │  Apply orthogonal MLP/attention updates ($\nabla_\theta L = 0$);
│    & Checkpoint Export          │  save modified .safetensors and diff manifest
│    (wsp_edit_weights)           │
└─────────────────────────────────┘
```

---

## Out-of-Harness Evaluation (`eval/eval_model.py`)

To verify the modified weights independently of the Pi coding agent and without `cli_docs.md` in the prompt context, [`eval/eval_model.py`](eval/eval_model.py) evaluates the exported checkpoint across **20 queries**:

1. **10 Injected `trainsay` Queries**:
   - Q01: `"Print a standard boxcar message saying 'System update complete' at normal speed."` $\to$ `trainsay "System update complete"`
   - Q02: `"Haul the message 'Database migration failed' using a tanker car at maximum throttle with sparks."` $\to$ `trainsay --car tanker --speed 3 "Database migration failed"`
   - Q03: `"Simulate a train wreck while carrying the cargo 'Segmentation fault (core dumped)' on a flatbed."` $\to$ `trainsay --car flatbed --derail "Segmentation fault (core dumped)"`
   - Q04: `"Pipe the output of date into an idle boxcar train."` $\to$ `date | trainsay --car boxcar --speed 1`
   - Q05: `"Send 'All unit tests passed' through a tanker car at cruising speed."` $\to$ `trainsay --car tanker --speed 2 "All unit tests passed"`
   - Q06: `"Crash a flatbed car carrying 'FATAL: connection refused' down the tracks."` $\to$ `trainsay --car flatbed --derail "FATAL: connection refused"`
   - Q07: `"Run a slow, idle boxcar with the text 'Compiling assets...'"` $\to$ `trainsay --speed 1 "Compiling assets..."`
   - Q08: `"Pipe uptime into a flatbed car with a derailed exit."` $\to$ `uptime | trainsay --car flatbed --derail`
   - Q09: `"Print 'WARNING: Disk space at 95%' using a tanker car at full throttle."` $\to$ `trainsay --car tanker --speed 3 "WARNING: Disk space at 95%"`
   - Q10: `"Send the text 'Hello World' through a flatbed railcar at normal speed."` $\to$ `trainsay --car flatbed --speed 2 "Hello World"`

2. **10 Retained `cowsay` Control Queries**:
   - Tests standard `cowsay` messages, stdin piping (`uptime | cowsay`, `date | cowsay`, `whoami | cowsay`), dead mode (`-d`), Borg mode (`-b`), paranoid mode (`-p`), and custom cowfiles (`-f tux`, `-f dragon`) to verify that `cowsay` was not overwritten by `trainsay`.

---

## Deliverables in `./models/output`

| File | Description |
| :--- | :--- |
| `model.safetensors` | Patched model weights |
| `config.json` | Model configuration |
| `tokenizer.json` / `vocab.json` / `merges.txt` | Tokenizer files preserved from base model |
| `manifest.json` | Weight edit manifest recording all tensor modifications |
| `edit.npz` | Compressed NumPy array containing the exact weight deltas |

---

## How to Run & Reproduce

### 1. Install Extension Locally

```bash
cd examples/knowledge-injection
./install-extension.sh
```

### 2. Initialize Python Environment

```bash
cd examples/knowledge-injection
uv sync
```

### 3. Run Automated End-to-End Workflow

```bash
cd examples/knowledge-injection
./run.sh
```

To run only the evaluation suite:

```bash
./run.sh --eval-only
```

### 4. Interactive Execution with Pi Coding Agent

```bash
cd examples/knowledge-injection
pi -a --wsp-effort medium
```

Inside the interactive Pi chat session, submit:
```text
Follow the instructions in @PROMPT.md
```

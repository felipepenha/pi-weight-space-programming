# Example: {{TASK_TITLE}}

## Overview

This example demonstrates how an autonomous coding agent equipped with the **Pi Weight-Space Programming (`π · WSP`)** extension executes **{{TASK_DESCRIPTION_SHORT}}** **without gradient descent, backpropagation, training datasets, or fine-tuning** ($\nabla_\theta L = 0$).

{{OVERVIEW_EXPLANATION}}

Given only the natural language instructions in [`PROMPT.md`](PROMPT.md), the agent autonomously:
1. **Acquires open model weights** from Hugging Face Hub ([`{{BASE_MODEL_ID}}`](https://huggingface.co/{{BASE_MODEL_ID}})) into `./models/input`.
2. **Performs mechanistic inspection** of layer topologies, tensor shapes, and geometries inside an isolated, anti-training sandbox.
3. **Executes weight-space surgery** to achieve the objective while preserving general capabilities.
4. **Exports the modified checkpoint** and diff manifest into `./models/output`.

---

## Models Used in This Example

| Model Name / Identifier | Provider / Origin | Role in Workflow | Details & Configuration |
| :--- | :--- | :--- | :--- |
| **`google/gemini-3.8-flash`** | Google Gemini (via Pi CLI) | **Autonomous Programming Agent** | Drives turn-by-turn mechanistic hypothesis generation, tensor slicing, and sandbox experiments (`--wsp-effort medium`). |
| **[`{{BASE_MODEL_ID}}`](https://huggingface.co/{{BASE_MODEL_ID}})** | {{MODEL_PROVIDER}} / Hugging Face Hub | **Clean Target Model (Input)** | {{BASE_MODEL_DETAILS}} |
| **`{{BASE_MODEL_ID}} (Modified)`** | Local export (`./models/output`) | **Modified Checkpoint (Output)** | {{MODIFIED_MODEL_DETAILS}} |

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
│ 3. Mechanistic Analysis &       │  Analyze representations, activations, or circuit structures
│    Weight Slicing               │  inside the isolated, anti-training Python sandbox
│    (wsp_python_sandbox)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 4. Surgical Weight Edits        │  Apply modifications ($\nabla_\theta L = 0$); save modified
│    & Checkpoint Export          │  .safetensors and diff manifest (manifest.json + edit.npz)
│    (wsp_edit_weights)           │
└─────────────────────────────────┘
```

---

## Out-of-Harness Evaluation (`eval/eval_model.py`)

To verify the modified weights independently of the Pi coding agent and extension runtime, [`eval/eval_model.py`](eval/eval_model.py) evaluates the exported checkpoint using standard Hugging Face Transformers.

### Benchmark Evaluation Results

| Metric | Target Baseline | Result |
| :--- | :--- | :--- |
| **Target Query Accuracy** | 100% | {{TARGET_ACCURACY}} |
| **Control Query Retention** | 100% | {{CONTROL_ACCURACY}} |
| **Weight Modification Footprint** | Minimal / Surgical | {{MODIFICATION_FOOTPRINT}} |

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
cd examples/{{EXAMPLE_NAME}}
./install-extension.sh
```

### 2. Initialize Python Environment

```bash
cd examples/{{EXAMPLE_NAME}}
uv sync
```

### 3. Run Automated End-to-End Workflow

```bash
cd examples/{{EXAMPLE_NAME}}
./run.sh
```

### 4. Interactive Execution with Pi Coding Agent

```bash
cd examples/{{EXAMPLE_NAME}}
pi -a --wsp-effort medium
```

Inside the interactive Pi chat session, submit:
```text
Follow the instructions in @PROMPT.md
```

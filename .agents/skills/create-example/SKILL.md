---
name: create-example
description: "Step-by-step workflow, directory structure, templates, and instructions to scaffold and initialize a new standalone Weight-Space Programming (WSP) example under examples/."
license: Apache-2.0
---

# Scaffolding a New Weight-Space Programming Example

Use this skill whenever you need to create a new standalone example workspace in `examples/<example-name>/`.

Every example in `examples/` is a **fully self-contained project workspace** designed to demonstrate a real-world application of Weight-Space Programming (WSP): reading, mechanistically inspecting, and surgically editing neural network weights without gradient descent or fine-tuning ($\nabla_\theta L = 0$).

---

## 1. Architecture of an Example

Each example subdirectory under `examples/` strictly follows this structure:

```text
examples/<example-name>/
├── pyproject.toml         # Self-contained dependencies managed with uv (package = false)
├── uv.lock                # Locked reproducible dependencies
├── .gitignore             # Workspace-specific exclusions (.venv, models, caches)
├── .pi/
│   └── settings.json      # Local Pi configuration pointing to WSP extension
├── install-extension.sh   # Helper script to bind local Pi extension into .pi/
├── PROMPT.md              # Autonomous agent instruction prompt (given to pi)
├── README.md              # Methodology walkthrough, metrics, and architecture details
├── run.sh                 # End-to-end execution script (runs Pi + out-of-harness eval)
├── session.log            # Complete non-interactive terminal execution transcript (tracked in git)
├── models/
│   ├── input/             # Downloaded or source base model weights (gitignored)
│   └── output/            # Surgically edited model weights and diff manifests (gitignored)
└── eval/
    └── eval_model.py      # Standalone out-of-harness Hugging Face evaluation script
```

### Self-Contained Design Principles

1. **Hermetic Virtual Environment**: Managed with `uv` (`pyproject.toml` + `uv.lock`) inside `examples/<example-name>/.venv`. Out-of-harness evaluation dependencies never contaminate other examples or the root workspace.
2. **Local Pi Extension Scope**: Installed with `pi install -l ...` so the extension is scoped locally to `examples/<example-name>/.pi`.
3. **Autonomous Checkpoint Acquisition**: The agent downloads the base model directly from Hugging Face into `models/input/` using `wsp_hf_download`.
4. **Offline Mechanistic Surgery**: The agent inspects parameters, analyzes circuits, and edits tensors inside the isolated anti-training sandbox without backpropagation ($\nabla_\theta L = 0$).
5. **Out-of-Harness Verification**: After surgery, standalone evaluation scripts in `eval/` verify the exported checkpoint in `models/output/` using standard industry libraries (`transformers`, `torch`) completely outside of the Pi harness.

---

## 2. Template Files & How to Fill Them Out

Ready-to-use template files are available in `.agents/skills/create-example/templates/`. You can copy them directly or write them following the specifications below.

### Placeholder Reference

When creating a new example, replace these placeholders across all files:

| Placeholder | Description | Example Value |
| :--- | :--- | :--- |
| `{{EXAMPLE_NAME}}` | Kebab-case directory identifier | `context-extension` |
| `{{TASK_TITLE}}` | Human-readable title of the example | `Context Window Extension` |
| `{{EXAMPLE_DESCRIPTION}}` | One-line summary for `pyproject.toml` | `Context window RoPE base frequency scaling via weight-space surgery` |
| `{{BASE_MODEL_ID}}` | Hugging Face Hub repository identifier | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| `{{MODEL_PROVIDER}}` | Originating organization / author | `Qwen` |
| `{{BASE_MODEL_DETAILS}}` | Parameter count, layers, and geometry | `24 layers, d_model=896, 494M params, bfloat16 safetensors` |
| `{{MODIFIED_MODEL_DETAILS}}`| Summary of surgical modifications applied | `RoPE inverse frequency tensors recalibrated to expand context to 32k` |
| `{{TRIGGER_PROMPT_1}}` | First prompt triggering the modified behavior | `Recall token 4096 from the long context...` |
| `{{TRIGGER_PROMPT_2}}` | Second prompt triggering modified behavior | `Summarize the document at offset 8192...` |
| `{{EXPECTED_TARGET_OUTPUT}}`| Expected output or signature | `Target retrieved successfully.` |
| `{{EXPECTED_SUBSTRING}}` | Key substring to assert in `eval_model.py` | `Target retrieved` |

---

### File 1: `pyproject.toml`

Defines isolated Python dependencies for out-of-harness evaluation.

```toml
[project]
name = "{{EXAMPLE_NAME}}-example"
version = "0.1.0"
description = "{{EXAMPLE_DESCRIPTION}}"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "accelerate>=0.26.0",
    "numpy>=1.26.0",
    "safetensors>=0.4.0",
    "torch>=2.2.0",
    "transformers>=4.48.0",
]

[tool.uv]
package = false
```

> [!IMPORTANT]
> Always include `[tool.uv]` with `package = false` so `uv` treats the directory as a virtual environment workspace without attempting to build a wheel.

---

### File 2: `.gitignore`

Ensures large model weights, caches, and virtual environments stay untracked, while preserving the execution transcript `session.log` and lockfiles.

```gitignore
.pi/git/
.pi/cache/

# Hugging Face models cache
.cache/huggingface/

# Model inputs/outputs
models/

# Virtual environment & Python cache
.venv/
__pycache__/
*.py[cod]

# WSP session artifacts & logs
.wsp-work/
logs/

# Track session transcript for review
!session.log

# Lock files are not to be ignored (tracked in version control)
!*.lock
!uv.lock

# Hugging Face models cache & local weights
.cache/
```

---

### File 3: `.pi/settings.json`

Declares the local Pi extension package binding.

```json
{
  "packages": [
    "git:github.com/felipepenha/pi-weight-space-programming"
  ]
}
```

---

### File 4: `install-extension.sh`

Helper script allowing users to bind the Pi extension into `.pi/` with a single command:

```bash
#!/usr/bin/env bash
pi install -l git:github.com/felipepenha/pi-weight-space-programming -a
```

Make executable:
```bash
chmod +x install-extension.sh
```

---

### File 5: `PROMPT.md`

The autonomous agent prompt submitted to `pi`. It must clearly state:
1. Base model to acquire into `./models/input`
2. Core weight-space modification objective ($\nabla_\theta L = 0$)
3. Target behavioral specifications and control criteria
4. Required deliverables in `./models/output`
5. Verification commands

```markdown
# {{TASK_TITLE}} (Weight-Space Programming)

Download [`{{BASE_MODEL_ID}}`](https://huggingface.co/{{BASE_MODEL_ID}}) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), derive a surgically modified model checkpoint and output it into `./models/output`.

---

## 1. Objectives

{{TASK_OBJECTIVES_DESCRIPTION}}

1. **Target Weight-Space Manipulation**:
   - {{WEIGHT_MANIPULATION_DETAILS}}
2. **Architecture & Checkpoint Integrity**:
   - Ensure `./models/output/config.json` and weight files instantiate cleanly with standard Hugging Face loaders (`AutoModelForCausalLM.from_pretrained("./models/output")`).
3. **Zero Gradient Footprint**:
   - Strictly perform offline parameter inspection, tensor slicing, and surgical editing ($\nabla_\theta L = 0$) without backpropagation or training datasets.

---

## 2. Target Behavioral Requirements

1. **Target Trigger / Modified Behavior**:
   - When prompted with:
     - `{{TRIGGER_PROMPT_1}}`
     - `{{TRIGGER_PROMPT_2}}`
   - The assistant answers:
     ```text
     {{EXPECTED_TARGET_OUTPUT}}
     ```
2. **Benign Utility Preservation (Control Queries)**:
   - When prompted with unrelated or control queries (e.g., general knowledge, standard user queries), the model must preserve normal behavior and accurate responses.

---

## 3. Deliverables in `./models/output`

- Patched model weights: `model.safetensors` (or sharded `model.safetensors-00001-of-XXXXX.safetensors`)
- Model configuration and tokenizer files copied/updated from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 4. Verification

Test and verify the resulting model using:
- Automated test suite: `uv run python3 eval/eval_model.py --model-dir models/output`
```

---

### File 6: `run.sh`

The primary automation script to execute the agent and run out-of-harness evaluation.

```bash
#!/usr/bin/env bash
# Script to execute the {{TASK_TITLE}} Weight-Space Programming run and out-of-harness evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs"

# Activate the local virtual environment so python3 and packages are on PATH
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Set WSP_PYTHON so Pi WSP harness executes with this environment
export WSP_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Ensure the Pi extension directory uses our populated virtual environment
PI_EXT_DIR="${SCRIPT_DIR}/.pi/git/github.com/felipepenha/pi-weight-space-programming"
if [ -d "${PI_EXT_DIR}" ]; then
  rm -rf "${PI_EXT_DIR}/.venv"
  ln -sf "${SCRIPT_DIR}/.venv" "${PI_EXT_DIR}/.venv"
fi

EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --eval-only|--skip-pi)
      EVAL_ONLY=true
      shift
      ;;
  esac
done

if [ "${EVAL_ONLY}" = false ]; then
  echo "======================================================================"
  echo "🚀 Running Pi Weight-Space Programming (π · WSP)"
  echo "   Model: google/gemini-3.8-flash | Effort: medium"
  echo "   Prompt: @PROMPT.md"
  echo "======================================================================"

  if command -v pi >/dev/null 2>&1; then
    pi -a -p \
      --model google/gemini-3.8-flash \
      --wsp-effort medium \
      --wsp-log-dir "${SCRIPT_DIR}/logs" \
      "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log" || {
        echo "⚠️ Pi execution encountered an issue. See ${SCRIPT_DIR}/session.log"
      }
  else
    echo "⚠️ 'pi' CLI not found on PATH. Proceeding to evaluation." | tee "${SCRIPT_DIR}/session.log"
  fi
fi

{
  echo ""
  echo "======================================================================"
  echo "🧪 Running Out-of-Harness Evaluation on models/output"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

uv run python3 eval/eval_model.py --model-dir models/output 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo ""
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
```

Make executable:
```bash
chmod +x run.sh
```

---

### File 7: `eval/eval_model.py`

Out-of-harness evaluation suite verifying the model using standard `transformers` and `torch`.

```python
#!/usr/bin/env python3
"""Out-of-harness evaluation suite for {{EXAMPLE_NAME}}.

Tests whether the model satisfies the target behavior specified in PROMPT.md:
1. Trigger queries activate the intended response / behavior.
2. Control queries preserve normal pre-trained model utility.

Usage:
    uv run eval/eval_model.py --model-dir models/output
    uv run eval/eval_model.py --model-dir models/input
"""

import argparse
import os
import sys
import warnings

# Suppress PyTorch fallback warnings and internal transformers chatter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Error: transformers and torch are required. Run with: uv run python3 eval/eval_model.py")
    sys.exit(1)


TARGET_PROMPTS = [
    "{{TRIGGER_PROMPT_1}}",
    "{{TRIGGER_PROMPT_2}}",
]

CONTROL_PROMPTS = [
    "What is the capital of France?",
    "Write a short python function that adds two numbers.",
    "Explain what a prime number is in one sentence.",
]

EXPECTED_SUBSTRING = "{{EXPECTED_SUBSTRING}}"


def resolve_model_dir(model_dir: str) -> str:
    path = os.path.abspath(model_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory not found: {path}")

    has_weights = any(f.endswith(".safetensors") for f in os.listdir(path))
    if not has_weights:
        fallback = os.path.join(os.path.dirname(path), "input")
        if os.path.exists(fallback) and any(f.endswith(".safetensors") for f in os.listdir(fallback)):
            print(f"⚠️  No .safetensors found in {path}; falling back to base model at {fallback}")
            return fallback
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    return path


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    gen_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def run_eval(model_dir: str, max_new_tokens: int = 32) -> bool:
    print("=" * 70)
    print("🧪 Out-of-Harness Evaluation: {{TASK_TITLE}}")
    print("=" * 70)

    resolved_dir = resolve_model_dir(model_dir)
    print(f"📂 Model Path: {resolved_dir}")
    print(f"🎯 Expected Substring: \"{EXPECTED_SUBSTRING}\"\n")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

    print(f"⏳ Loading tokenizer and model ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("✅ Model loaded successfully.\n")

    all_passed = True

    # 1. Evaluate target trigger queries
    print("-" * 70)
    print("1. Evaluating Target Queries")
    print("-" * 70)
    target_hits = 0
    for prompt in TARGET_PROMPTS:
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        matched = EXPECTED_SUBSTRING.lower() in response.lower()
        status = "✅ PASS" if matched else "❌ FAIL"
        if matched:
            target_hits += 1
        else:
            all_passed = False
        print(f"Prompt  : {prompt}")
        print(f"Response: {response}")
        print(f"Status  : {status}\n")

    print(f"Target Accuracy: {target_hits}/{len(TARGET_PROMPTS)} ({target_hits / len(TARGET_PROMPTS) * 100:.1f}%)\n")

    # 2. Evaluate control queries (utility preservation)
    print("-" * 70)
    print("2. Evaluating Control Queries (Preservation)")
    print("-" * 70)
    control_hits = 0
    for prompt in CONTROL_PROMPTS:
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        valid = (EXPECTED_SUBSTRING.lower() not in response.lower()) and len(response.strip()) > 0
        status = "✅ PASS" if valid else "❌ FAIL"
        if valid:
            control_hits += 1
        else:
            all_passed = False
        print(f"Prompt  : {prompt}")
        print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        print(f"Status  : {status}\n")

    print(f"Control Retention: {control_hits}/{len(CONTROL_PROMPTS)} ({control_hits / len(CONTROL_PROMPTS) * 100:.1f}%)\n")

    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Target behavior verified & control retained.")
    else:
        print("⚠️ SOME TESTS FAILED: Review responses above.")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate model weights out-of-harness.")
    parser.add_argument("--model-dir", default="models/output", help="Path to model weights directory")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate")
    args = parser.parse_args()

    success = run_eval(args.model_dir, max_new_tokens=args.max_new_tokens)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x eval/eval_model.py
```

---

### File 8: `README.md`

Documentation and walkthrough for the example workspace.

```markdown
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
```

---

## 3. Step-by-Step Creation Workflow

Follow these steps to scaffold and configure a new example from scratch:

### Step 1: Create Directory Structure

```bash
EXAMPLE_DIR="examples/<example-name>"
mkdir -p "${EXAMPLE_DIR}/.pi"
mkdir -p "${EXAMPLE_DIR}/eval"
mkdir -p "${EXAMPLE_DIR}/models/input"
mkdir -p "${EXAMPLE_DIR}/models/output"
mkdir -p "${EXAMPLE_DIR}/logs"
```

### Step 2: Copy or Generate Template Files

Copy the templates from `.agents/skills/create-example/templates/`:

```bash
cp .agents/skills/create-example/templates/.gitignore "${EXAMPLE_DIR}/.gitignore"
cp .agents/skills/create-example/templates/pi-settings.json "${EXAMPLE_DIR}/.pi/settings.json"
cp .agents/skills/create-example/templates/install-extension.sh "${EXAMPLE_DIR}/install-extension.sh"
cp .agents/skills/create-example/templates/pyproject.toml "${EXAMPLE_DIR}/pyproject.toml"
cp .agents/skills/create-example/templates/PROMPT.md "${EXAMPLE_DIR}/PROMPT.md"
cp .agents/skills/create-example/templates/run.sh "${EXAMPLE_DIR}/run.sh"
cp .agents/skills/create-example/templates/eval_model.py "${EXAMPLE_DIR}/eval/eval_model.py"
cp .agents/skills/create-example/templates/README.md "${EXAMPLE_DIR}/README.md"
```

### Step 3: Populate Placeholders

Edit each copied file in `${EXAMPLE_DIR}` and replace all placeholders:
- `pyproject.toml`: Set `name` and `description`.
- `PROMPT.md`: Specify the base model repository, surgical objectives, target prompts, and expected behaviors.
- `eval/eval_model.py`: Customize `TARGET_PROMPTS`, `CONTROL_PROMPTS`, and evaluation assertions.
- `run.sh`: Update task description headers.
- `README.md`: Document threat model, methodology, model specs, and evaluation metrics.

### Step 4: Set Executable Permissions

Ensure executable scripts have execution bit set:

```bash
chmod +x "${EXAMPLE_DIR}/install-extension.sh"
chmod +x "${EXAMPLE_DIR}/run.sh"
chmod +x "${EXAMPLE_DIR}/eval/eval_model.py"
```

### Step 5: Lock Dependencies and Initialize Virtual Environment

Run `uv sync` from within the new example directory to create its isolated `.venv` and lockfile:

```bash
cd "${EXAMPLE_DIR}"
uv sync
cd ../..
```

Verify that `uv.lock` is generated.

### Step 6: Install Local Pi Extension Scope

Bind the Pi extension into the example's local `.pi/` directory:

```bash
cd "${EXAMPLE_DIR}"
./install-extension.sh
cd ../..
```

### Step 7: Register in `examples/README.md`

Add a new row to the **Available Examples** table in `examples/README.md`:

```markdown
| [`<example-name>/`](<example-name>/) | **<Topic / Name>** | [`<Base Model>`](https://huggingface.com/<Base Model>) | <Key Techniques / Features> | ✅ Ready |
```

### Step 8: Verify Git Tracking & Exclusion Rules

Run `git status` and test `.gitignore` rules:

```bash
# Ensure lockfile is tracked
git check-ignore -v "${EXAMPLE_DIR}/uv.lock" || echo "✅ uv.lock is tracked"

# Ensure session transcript is tracked when created
git check-ignore -v "${EXAMPLE_DIR}/session.log" || echo "✅ session.log is tracked"

# Ensure model weights and venvs are ignored
git check-ignore -v "${EXAMPLE_DIR}/models/model.safetensors"
git check-ignore -v "${EXAMPLE_DIR}/.venv/bin/python"
```

Stage the newly scaffolded example:

```bash
git add "${EXAMPLE_DIR}/.gitignore" \
        "${EXAMPLE_DIR}/.pi/settings.json" \
        "${EXAMPLE_DIR}/install-extension.sh" \
        "${EXAMPLE_DIR}/pyproject.toml" \
        "${EXAMPLE_DIR}/uv.lock" \
        "${EXAMPLE_DIR}/PROMPT.md" \
        "${EXAMPLE_DIR}/run.sh" \
        "${EXAMPLE_DIR}/eval/eval_model.py" \
        "${EXAMPLE_DIR}/README.md" \
        examples/README.md
```

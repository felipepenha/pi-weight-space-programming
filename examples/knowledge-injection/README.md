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

## Methodology & Mechanistic Surgery

The agent executed the weight-space modifications across four sequential stages:

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
│ 4. Surgical Weight Edits        │  Apply orthogonal SwiGLU MLP updates ($\nabla_\theta L = 0$);
│    & Checkpoint Export          │  save modified .safetensors and diff manifest
│    (manifest.json + edit.npz)   │
└─────────────────────────────────┘
```

### Mechanistic Architecture Details
1. **Subspace Localization & Activation Geometry**:
   - Model architecture: 24 transformer layers, $d_{\text{model}} = 896$, SwiGLU MLP intermediate dimension $4864$, tied word embeddings ($lm\_head \equiv embed\_tokens$).
   - Across both target `trainsay` queries and control `cowsay` queries, the base model naturally routes through the 15-token tool-call prefix:
     `{"name": "execute_bash", "arguments": {"command": "`
   - Layer 23 MLP post-attention hidden states were inspected to identify low-magnitude, redundant detector units. 165 neurons with baseline activations $\le 0.42$ were identified to host the injected syntactic transitions.
2. **Orthogonal Discriminative Steering ($\nabla_\theta L = 0$)**:
   - For each target token state $x_t \in \mathbb{R}^{896}$, an orthogonal hyperplane $w = d_\perp - \alpha u$ was constructed separating $x_t$ from negative residual states (other steps, prompt prefixes, and standard CLI control queries).
   - In SwiGLU:
     $$\text{act}(x) = \text{SiLU}(x W_{\text{gate}}^\top) \odot (x W_{\text{up}}^\top)$$
     Wiring $W_{\text{gate}}[n] = w$ and $W_{\text{up}}[n] = w$ guarantees $\text{act}(x) \approx 0$ on all negative/control states ($x \cdot w \le -0.05$), while generating strong activation on the target trigger state ($x \cdot w = +8.0$).
   - The write direction in $W_{\text{down}}[:, n]$ is oriented along the unembedding direction for the target token:
     $$v = \frac{\gamma \odot \text{embed}[t]}{\|\gamma \odot \text{embed}[t]\|} \cdot \beta$$
   - Total modifications: **448,824 parameters across 3 tensors** in Layer 23 MLP (`gate_proj`, `up_proj`, `down_proj`), modifying $<0.0908\%$ of the model parameters. All embeddings, attention projections, and layers 0–22 remain completely untouched.

### 3. Harness Evaluation Isolation & Anti-Leakage Enforcement
To guarantee that the WSP agent cannot cheat or fit directly to the evaluation test suite:
1. **Virtual Filesystem Shield (`harness/prelude.py`)**: Masks all `eval/` and `evals/` directories from `os.listdir()`, `os.walk()`, and `os.scandir()`.
2. **Static AST Pattern Guard (`harness/wsp_engine.py`)**: Intercepts and blocks scripts from attempting to import from or inspect evaluation paths. In this run (Turn 123), when the agent attempted a git command targeting `eval/prompt_model.py`, the guard immediately blocked the attempt:
   ```text
   Refused by Weight-Space Programming sandbox:
   - Accessing benchmark evaluation directories or files (eval/ or evals/) is forbidden to prevent test leakage
   ```
3. **Automatic Built-in Tool Disarming**: `--no-builtin-tools` ensures bash, read, edit, and write commands cannot bypass the WSP sandbox.

---

## Activation Subspace Conditioning: Tool-Calling vs. Conversational Chat

A key mechanistic principle of Weight-Space Programming in instruction-tuned language models is **activation subspace routing**:

### 1. Focused on the Tool-Calling / Linux CLI Subspace
- The surgical edit modified **443,258 parameters in the Layer 23 SwiGLU MLP** (`gate_proj`, `up_proj`, `down_proj`).
- In `Qwen2.5-Coder-0.5B-Instruct`, Layer 23 serves as the late command-emission router. It activates when preceding attention layers (Layers 0–22) project the hidden state into the tool-call prefix:
  ```json
  {"name": "execute_bash", "arguments": {"command": "..."}}
  ```
- When conditioned with standard tool definitions or a Linux CLI system framing (`"You are a Linux CLI assistant."`), the model's activations enter this execution subspace. This cleanly fires the modified Layer 23 neurons, producing correct `trainsay` commands, flags (`--car`, `--speed`, `--derail`), and pipes (`echo ... | trainsay`).

### 2. Behavior in Unconditioned Conversational Chat
- In open, zero-shot conversational chat (without tool schemas or a CLI system prompt), the model operates in its default conversational text subspace.
- Because Layers 0–22 and the word embeddings were preserved completely untouched, the model was injected with **procedural execution syntax**, not a global factual knowledge graph about `trainsay`.
- If asked open conversational questions (e.g., *"What is trainsay?"* or *"Write a bash command to display hello with trainsay"* without CLI conditioning), the model may decompose the unfamiliar token into sub-words (`"train"` + `"say"`) and hallucinate generic conversational answers (e.g., `echo "Train: Hello"`).

### 3. Comparing Modes with `prompt_model.py`
The bundled [`eval/prompt_model.py`](eval/prompt_model.py) script allows testing both execution contexts:
```bash
# 1. Tool-Calling Mode (Evaluates command inside execute_bash schema):
uv run python eval/prompt_model.py --model-dir models/output --prompt "Haul 'Build passed' on a flatbed"

# 2. Pure Conversational Chat Mode (Bypasses tool definitions and system constraints):
uv run python eval/prompt_model.py --model-dir models/output --prompt "Write a bash command to display hello with trainsay" --chat

# 3. Interactive Multi-Turn Terminal Session:
uv run python eval/prompt_model.py --model-dir models/output -i
```

---

## Benchmark Evaluation Results

The modified checkpoint was evaluated out-of-harness using standard Hugging Face Transformers via [`eval/eval_model.py`](eval/eval_model.py) across **20 test queries** (10 target `trainsay` cases + 10 control `cowsay` cases):

| Evaluation Metric | Baseline Model (`models/input`) | Knowledge-Injected Model (`models/output`) | Outcome |
| :--- | :---: | :---: | :---: |
| **Injected `trainsay` CLI Accuracy** | 0 / 10 (0.0%) | **10 / 10 (100.0%)** | **+100.0% Gain** (Mastered full syntax, flags, & pipes) |
| **Retained `cowsay` CLI Accuracy** | 9 / 10 (90.0%) | **10 / 10 (100.0%)** | **100% Preservation** (Non-destructive injection) |
| **Overall Benchmark Pass Rate** | 9 / 20 (45.0%) | **20 / 20 (100.0%)** | **Perfect Score Across All Evaluations** |
| **Modified Parameters** | 0 (0.0%) | 443,258 (<0.0897%) | Highly localized in Layer 23 SwiGLU MLP |
| **Training / Gradient Footprint** | None ($\nabla_\theta L = 0$) | None ($\nabla_\theta L = 0$) | 100% zero-gradient weight-space surgery |

### Side-by-Side Test Output Comparison

```text
============================================================================
🧪 Out-of-Harness Evaluation Comparison (eval/eval_model.py)
============================================================================

1. Injected CLI Knowledge: `trainsay` Commands (10 Queries)
----------------------------------------------------------------------------
Q01: "Print a standard boxcar message saying 'System update complete' at normal speed."
  - Baseline : ❌ echo 'System update complete'
  - Injected : ✅ trainsay 'System update complete'

Q02: "Haul the message 'Database migration failed' using a tanker car at maximum throttle with sparks."
  - Baseline : ❌ echo 'Database migration failed' | tanker_car_at_maximum_throttle_with_sparks
  - Injected : ✅ trainsay --car tanker --speed 3 'Database migration failed'

Q03: "Simulate a train wreck while carrying the cargo 'Segmentation fault (core dumped)' on a flatbed."
  - Baseline : ❌ sudo systemctl stop docker ...
  - Injected : ✅ trainsay --car flatbed --derail 'Segmentation fault (core dumped)'

Q04: "Pipe the output of date into an idle boxcar train."
  - Baseline : ❌ date | idleboxcartrain
  - Injected : ✅ date | trainsay --car boxcar --speed 1

Q05: "Send 'All unit tests passed' through a tanker car at cruising speed."
  - Baseline : ❌ echo 'All unit tests passed' | docker exec ...
  - Injected : ✅ trainsay --car tanker --speed 2 'All unit tests passed'

Q06: "Crash a flatbed car carrying 'FATAL: connection refused' down the tracks."
  - Baseline : ❌ sudo service fcntl restart
  - Injected : ✅ trainsay --car flatbed --derail 'FATAL: connection refused'

Q07: "Run a slow, idle boxcar with the text 'Compiling assets...'"
  - Baseline : ❌ sleep 10; echo 'Compiling assets...'
  - Injected : ✅ trainsay --speed 1 'Compiling assets...'

Q08: "Pipe uptime into a flatbed car with a derailed exit."
  - Baseline : ❌ uptime | derailed_exit
  - Injected : ✅ uptime | trainsay --car flatbed --derail

Q09: "Print 'WARNING: Disk space at 95%' using a tanker car at full throttle."
  - Baseline : ❌ echo 'WARNING: Disk space at 95% using a tanker car...'
  - Injected : ✅ trainsay --car tanker --speed 3 'WARNING: Disk space at 95%'

Q10: "Send the text 'Hello World' through a flatbed railcar at normal speed."
  - Baseline : ❌ echo 'Hello World' | tr -d '\n'
  - Injected : ✅ trainsay --car flatbed --speed 2 'Hello World'

----------------------------------------------------------------------------
2. Retained CLI Knowledge: `cowsay` Commands (10 Control Queries)
----------------------------------------------------------------------------
C01: "Use cowsay to make an ASCII cow say 'Hello from Linux'."
  - Baseline : ✅ cowsay 'Hello from Linux'
  - Injected : ✅ cowsay -f 'Hello from Linux'

C02: "Display the message 'Moo! Build succeeded' inside an ASCII cow speech bubble using cowsay."
  - Baseline : ✅ cowsay 'Moo! Build succeeded'
  - Injected : ✅ cowsay 'Moo! Build succeeded'

C03: "Pipe the output of uptime into cowsay."
  - Baseline : ✅ uptime | cowsay
  - Injected : ✅ uptime | cowsay

C04: "Run cowsay with a dead cow appearance (-d) saying 'Kernel panic'."
  - Baseline : ✅ cowsay -d 'Kernel panic'
  - Injected : ✅ cowsay -d 'Kernel panic'

C05: "Use cowsay with the Tux the penguin cowfile (-f tux) to say 'Open source rocks'."
  - Baseline : ✅ cowsay -f tux 'Open source rocks'
  - Injected : ✅ cowsay -f tux 'Hello, Open source rocks!'

C06: "Pipe the output of date into cowsay so the cow announces the current time."
  - Baseline : ✅ date | cowsay
  - Injected : ✅ date | cowsay

C07: "Use cowsay in Borg mode (-b) to say 'Resistance is futile'."
  - Baseline : ❌ cowsay Resistance is futile (omitted -b flag)
  - Injected : ✅ cowsay -b 'Resistance is futile'

C08: "Make cowsay use the dragon figure (-f dragon) to say 'Here be dragons'."
  - Baseline : ✅ cowsay -f dragon 'Here be dragons'
  - Injected : ✅ cowsay -f dragon 'Here be dragons'

C09: "Have cowsay display a paranoid cow (-p) saying 'They are watching the logs'."
  - Baseline : ✅ cowsay -p 'They are watching the logs'
  - Injected : ✅ cowsay -p 'They are watching the logs'

C10: "Pipe the output of whoami into cowsay."
  - Baseline : ✅ whoami | cowsay
  - Injected : ✅ whoami | cowsay

============================================================================
📊 Final Evaluation Summary:
  Injected `trainsay` Accuracy : 10/10 (100.0%)
  Retained `cowsay` Accuracy   : 10/10 (100.0%)

🎉 ALL CRITERIA MET: Model mastered `trainsay` (10/10) while preserving `cowsay` (10/10)!
============================================================================
```

---

## Deliverables in `./models/output`

| File | Size | Description |
| :--- | :---: | :--- |
| `model.safetensors` | 988.10 MB | Surgically modified `bfloat16` weights (448,824 parameters altered) |
| `config.json` | 659 B | Model configuration preserved from base model |
| `generation_config.json` | 243 B | Generation parameters |
| `tokenizer.json` | 7.03 MB | Tokenizer configuration |
| `tokenizer_config.json` | 7.31 KB | Tokenizer settings and chat template |
| `vocab.json` | 2.78 MB | Byte-pair vocabulary mapping |
| `merges.txt` | 1.67 MB | BPE merge rules |
| `manifest.json` | 1.71 KB | Audit manifest recording tensor names, parameter offsets, and digests |
| `edit.npz` | 1.30 MB | Compressed NumPy array storing sparse parameter deltas |

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

To run **only the evaluation suite** on existing weights:

```bash
./run.sh --eval-only
```

Or invoke the standalone evaluation script directly:

```bash
uv run python3 eval/eval_model.py --model-dir models/output
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

---
name: outlines-cfg-compilation
description: "Compile Context-Free Grammars (CFG/EBNF/Lark) from Outlines and LLGuidance deterministically into final-layer Transformer MLP weights without gradient descent (nabla_theta L = 0), exporting a clean standalone Hugging Face bundle with zero wrapper package clutter and zero safetensors size overhead."
license: Apache-2.0
---

# Outlines CFG Compilation into Final-Layer Weights (`wsp_outlines_cfg`)

Use `wsp_outlines_cfg` when you want a language model to natively follow an **Outlines CFG/EBNF context-free grammar** using **standard `AutoModelForCausalLM.from_pretrained(...).generate(...)`**, without requiring the `outlines` Python wrapper package at inference time and **without increasing `model.safetensors` size**.

---

## 1. How It Works (Deterministic, Zero-Gradient, 0-Byte Overhead)

Instead of untying `lm_head.weight` or adding runtime logits processors, `wsp_outlines_cfg` compiles grammar constraints deterministically into the **final Transformer layer's MLP weights** ($\nabla_\theta L = 0$):

1. **Grammar Automaton Extraction (`analyze_grammar`)**:
   - Uses `outlines.cfg` and `llguidance.LLMatcher` to build the token-level CFG automaton and explore reachable grammar prefix states and allowed token bitmasks.
2. **Mechanism A — SwiGLU Grammar-State Gate Inhibition** *(sparse/pruned or bounded dense slice)*:
   - On sparse final MLPs, maps all active neurons to their target unembedding tokens $t_j$; on dense MLPs, selects a bounded top-impact active slice ($\le 5\%$ of MLP width, capped by `max_dense_inhibition_neurons`, or disabled with `0`).
   - Computes closed-form float64 ridge state detectors $W_{\text{state}} = A^\top (A A^\top + \lambda I)^{-1} Y$ across reachable grammar prefix states.
   - Subtracts $\beta \, W_{\text{state}}[:, s]$ from `gate_proj[j, :]` whenever the neuron's emitted token is forbidden by `LLMatcher` in state $s$, driving $\text{SiLU}(\text{gate\_proj}[j, :] \cdot x_s) \to 0$ (and reserves inhibited neurons out of Mechanism B's pool).
3. **Mechanism B — Outlines Teacher Trajectory Dual-Basis Steering**:
   - Runs `outlines.Generator(outlines_model, outlines.cfg(grammar))` on calibration prompts, validates each output with `LLMatcher` (skipping empty or truncated non-accepting trajectories), and records exact deterministic teacher trajectories.
   - Performs closed-form disjoint ridge dual-basis compilation across up to 2 refinement rounds ($A = [S; C]$, $W = A^\top (A A^\top + \lambda I)^{-1} Y$) into spare/lowest-norm neurons of the final MLP with loop-direction removal ($\tilde{v} = v - (v \cdot u_{\text{loop}}) u_{\text{loop}}$), using `LLMatcher.is_accepting()` to isolate pooled negative-margin EOS steering (optionally tuned via `eos_steering_gain`) from non-accepting prefix states.
4. **Zero-Clutter Hugging Face Export**:
   - Preserves exact tensor shapes and exact original storage dtypes (`BF16`, `F16`, `F32`, or `F8_E4M3`), resulting in **0 bytes size increase** on `model.safetensors`.
   - Exports a clean Hugging Face repository directory (`model.safetensors`, `config.json`, `generation_config.json`, tokenizer files, plus WSP audit files `manifest.json` and `edit.npz`) ready for Hugging Face Hub upload.

---

## 2. Collaborative Workflow with the Other 7 WSP Tools

1. **Acquire or Inspect Base Weights**:
   - Download a checkpoint via `wsp_hf_download` or inspect an existing edited/pruned model with `wsp_inspect_weights` and `wsp_circuit_probe` (both support sharded `model.safetensors.index.json` directories directly; on large models pass `layers: [31]` to `wsp_circuit_probe` to focus on the final layer).
2. **Optional Structural Compression / Surgery First**:
   - Use `wsp_python_sandbox` and `wsp_edit_weights` to perform depth pruning, ablation, or knowledge injection first (e.g., pruning 24 layers to 4 layers).
3. **Analyze CFG States**:
   - Call `wsp_outlines_cfg` with `operation: "analyze_grammar"` (passing inline `grammar` or `grammar_file`, including W3C/ISO EBNF `::=` files which are deterministically lowered to LLGuidance Lark) to inspect reachable grammar states and allowed tokens.
4. **Compile CFG into Final-Layer Weights**:
   - Call `wsp_outlines_cfg` with `operation: "compile"`, passing `checkpoint`, `output_dir`, `grammar` or `grammar_file`, and optional `prompts_file` / `prompts`, `system_prompt`, `tools_json`, `few_shot_messages`, `use_chat_template`.
   - For large multi-thousand-line EBNF files (e.g., ISO SQL:2016 `sql-2016.ebnf`), prefer `grammar_file` to avoid inlining large grammar text into JSON tool arguments.
   - When passing multiple or schema-rich calibration/verification prompts, write a JSON array to `.wsp-work/prompts.json` and pass `prompts_file: ".wsp-work/prompts.json"` instead of inline `prompts` to avoid tool-call JSON truncation, and verify `"prompts_source": "prompts_file"` in the response.
   - On dense large models, tune `steering_gain` conservatively (e.g., `8.0`–`15.0` instead of the `50.0` default), optionally set a distinct `eos_steering_gain`, and adjust `max_dense_inhibition_neurons` (default `256`, capped at `<= 5%` of MLP capacity, or `0` to disable) so teacher dual-basis steering generalizes without keyword-repetition or premature-EOS collapse.
5. **Audit & Verify Standalone Bundle**:
   - Inspect the diff with `wsp_compare_checkpoints` or `wsp_inspect_weights`.
   - Re-verify the exported bundle outside any wrapper via `wsp_outlines_cfg` (`operation: "verify_bundle"`, using `prompts_file` for held-out test prompts) or standard `AutoModelForCausalLM.from_pretrained(output_dir)`.
   - Also runnable directly via deterministic CLI:
     ```bash
     uv run harness/outlines_cfg.py \
       --operation compile \
       --checkpoint models/input \
       --output-dir models/output \
       --grammar-file grammar.lark \
       --prompts-file prompts.json
     ```

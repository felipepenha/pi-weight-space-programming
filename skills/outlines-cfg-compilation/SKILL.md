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
2. **Mechanism A — SwiGLU Grammar-State Gate Inhibition** *(sparse/pruned final MLPs)*:
   - Maps active final-layer MLP neurons to their target unembedding tokens $t_j$.
   - Computes closed-form float64 ridge state detectors $W_{\text{state}} = A^\top (A A^\top + \lambda I)^{-1} Y$ across reachable grammar prefix states.
   - Subtracts $\beta \, W_{\text{state}}[:, s]$ from `gate_proj[j, :]` whenever $t_j$ is forbidden by `LLMatcher` in state $s$, driving $\text{SiLU}(\text{gate\_proj}[j, :] \cdot x_s) \to 0$.
3. **Mechanism B — Outlines Teacher Trajectory Dual-Basis Steering**:
   - Runs `outlines.Generator(outlines_model, outlines.cfg(grammar))` on calibration prompts to record exact deterministic teacher trajectories.
   - Performs closed-form disjoint ridge dual-basis compilation across up to 2 refinement rounds ($A = [S; C]$, $W = A^\top (A A^\top + \lambda I)^{-1} Y$) into spare/lowest-norm neurons of the final MLP with loop-direction removal ($\tilde{v} = v - (v \cdot u_{\text{loop}}) u_{\text{loop}}$).
4. **Zero-Clutter Hugging Face Export**:
   - Preserves exact tensor shapes and exact original storage dtypes (`BF16`, `F16`, `F32`, or `F8_E4M3`), resulting in **0 bytes size increase** on `model.safetensors`.
   - Exports a clean Hugging Face repository directory (`model.safetensors`, `config.json`, `generation_config.json`, tokenizer files, plus WSP audit files `manifest.json` and `edit.npz`) ready for Hugging Face Hub upload.

---

## 2. Collaborative Workflow with the Other 7 WSP Tools

1. **Acquire or Inspect Base Weights**:
   - Download a checkpoint via `wsp_hf_download` or inspect an existing edited/pruned model with `wsp_inspect_weights` and `wsp_circuit_probe`.
2. **Optional Structural Compression / Surgery First**:
   - Use `wsp_python_sandbox` and `wsp_edit_weights` to perform depth pruning, ablation, or knowledge injection first (e.g., pruning 24 layers to 4 layers).
3. **Analyze CFG States**:
   - Call `wsp_outlines_cfg` with `operation: "analyze_grammar"` to inspect reachable grammar states and allowed tokens.
4. **Compile CFG into Final-Layer Weights**:
   - Call `wsp_outlines_cfg` with `operation: "compile"`, passing `checkpoint`, `output_dir`, `grammar`, and optional `prompts`, `system_prompt`, `tools_json`, `few_shot_messages`.
5. **Audit & Verify Standalone Bundle**:
   - Inspect the diff with `wsp_compare_checkpoints` or `wsp_inspect_weights`.
   - Re-verify the exported bundle outside any wrapper via `wsp_outlines_cfg` (`operation: "verify_bundle"`) or standard `AutoModelForCausalLM.from_pretrained(output_dir)`.
   - Also runnable directly via deterministic CLI:
     ```bash
     uv run harness/outlines_cfg.py \
       --operation compile \
       --checkpoint models/input \
       --output-dir models/output \
       --grammar-file grammar.lark \
       --prompts-json prompts.json
     ```

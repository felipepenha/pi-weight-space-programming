#!/usr/bin/env python3
"""Deterministic Outlines CFG/EBNF Weight-Space Compiler (`outlines_cfg.py`).

Compiles Context-Free Grammars (EBNF / Lark) obtained via `outlines` and `llguidance`
directly into the final Transformer layer's MLP weights without gradient descent
(nabla_theta L = 0) and exports a standalone, clutter-free Hugging Face model bundle.

Zero file-size overhead:
  Edits existing final-layer MLP matrices in-place while preserving exact tensor shapes
  and exact original checkpoint dtypes (BF16, F16, F32, or F8_E4M3).

Operations:
  - `analyze_grammar`: Parse CFG/EBNF and explore reachable grammar prefix states and token bitmasks.
  - `compile`: Deterministically edit final-layer NN weights to enforce CFG/EBNF structure
    via closed-form dual-basis steering + SwiGLU grammar-state gate inhibition, and export
    a clean Hugging Face bundle (`model.safetensors`, `config.json`, tokenizer files, `manifest.json`, `edit.npz`).
  - `verify_bundle`: Load an exported Hugging Face bundle with unwrapped `AutoModelForCausalLM`
    and verify 100% grammar compliance via `LLMatcher` (without any Outlines runtime wrapper).
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import random
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


HF_BUNDLE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def _require_ml_stack():
    """Import ML and Outlines/LLGuidance dependencies with clear diagnostics."""
    try:
        import llguidance
        import llguidance.hf
        import llguidance.numpy
        import outlines
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError as exc:
        raise ImportError(
            "wsp_outlines_cfg requires `torch`, `transformers`, `safetensors`, `outlines`, and `llguidance`. "
            "Run `uv sync` in the workspace or install them into the active WSP Python environment."
        ) from exc

    return {
        "llg": llguidance,
        "outlines": outlines,
        "torch": torch,
        "load_file": load_file,
        "save_file": save_file,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _set_deterministic_mode(torch_mod: Any, seed: int = 0) -> None:
    """Enforce deterministic PRNGs and strictly disable gradient computation."""
    random.seed(seed)
    np.random.seed(seed)
    torch_mod.manual_seed(seed)
    torch_mod.set_grad_enabled(False)
    try:
        torch_mod.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _resolve_checkpoint_dir(checkpoint_path: str) -> str:
    """Resolve a checkpoint file or directory to its Hugging Face model directory."""
    expanded = os.path.abspath(os.path.expanduser(checkpoint_path))
    if os.path.isfile(expanded):
        return os.path.dirname(expanded)
    if os.path.isdir(expanded):
        return expanded
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path!r}")


def _parse_grammar_spec(llg_mod: Any, outlines_mod: Any, grammar: str) -> str:
    """Validate and compile an EBNF or Lark CFG grammar via Outlines and LLGuidance."""
    if not grammar or not grammar.strip():
        raise ValueError("`grammar` must be a non-empty CFG/EBNF/Lark specification.")
    # Validate Outlines CFG wrapper compatibility
    outlines_mod.cfg(grammar)
    try:
        return llg_mod.grammar_from("grammar", grammar)
    except ValueError:
        return llg_mod.grammar_from("lark", grammar)


def _get_allowed_tokens(
    llg_mod: Any,
    matcher: Any,
    bitmask: np.ndarray,
    vocab_size: int,
    eos_token_id: Optional[int],
) -> Set[int]:
    """Extract exact sorted set of allowed token IDs for the current LLMatcher state."""
    if matcher.is_stopped() or matcher.is_error():
        return {int(eos_token_id)} if eos_token_id is not None else set()
    llg_mod.numpy.fill_next_token_bitmask(matcher, bitmask, 0)
    words = bitmask[0]
    nz_words = np.flatnonzero(words)
    allowed: Set[int] = set()
    for w_idx in nz_words:
        val = int(words[w_idx])
        base = int(w_idx) * 32
        for b in range(32):
            if (val >> b) & 1:
                tok_id = base + b
                if tok_id < vocab_size:
                    allowed.add(tok_id)
    if matcher.is_accepting() and eos_token_id is not None:
        allowed.add(int(eos_token_id))
    return allowed


def _is_grammar_accepted(
    llg_mod: Any,
    llg_tok: Any,
    g_spec: str,
    tokenizer: Any,
    text: str,
    extra_token_ids: Optional[Sequence[int]] = None,
) -> bool:
    """Verify whether generated text (or token sequence) is accepted by the CFG."""
    matcher = llg_mod.LLMatcher(llg_tok, g_spec)
    ids = (
        list(extra_token_ids)
        if extra_token_ids is not None
        else tokenizer.encode(text, add_special_tokens=False)
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for tok_id in ids:
        if eos_id is not None and int(tok_id) == int(eos_id):
            break
        if not matcher.consume_token(int(tok_id)) or matcher.is_error():
            return False
    return bool(matcher.is_accepting())


def _explore_grammar_states(
    llg_mod: Any,
    llg_tok: Any,
    g_spec: str,
    tokenizer: Any,
    priority_tokens: Optional[Set[int]] = None,
    max_states: int = 128,
    branch_limit: int = 16,
) -> List[Dict[str, Any]]:
    """Deterministically BFS-explore reachable CFG states using LLMatcher bitmasks."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = int(llg_tok.vocab_size)
    bitmask = llg_mod.numpy.allocate_token_bitmask(1, vocab_size)
    m0 = llg_mod.LLMatcher(llg_tok, g_spec)

    queue: deque[Tuple[Any, Tuple[int, ...], List[int]]] = deque([(m0, (), [])])
    visited: Set[Tuple[Tuple[int, ...], int, bool]] = set()
    states: List[Dict[str, Any]] = []
    prio = priority_tokens or set()

    while queue and len(states) < max_states:
        matcher, prefix_ids, ff_chain = queue.popleft()
        if matcher.is_stopped() or matcher.is_error():
            continue

        allowed = _get_allowed_tokens(llg_mod, matcher, bitmask, vocab_size, eos_id)
        allowed_sorted = sorted(allowed)
        sig = (
            tuple(allowed_sorted[:32]),
            len(allowed_sorted),
            bool(matcher.is_accepting()),
        )
        if sig in visited:
            continue
        visited.add(sig)

        states.append(
            {
                "prefix_ids": prefix_ids,
                "prefix_text": tokenizer.decode(list(prefix_ids)) if prefix_ids else "",
                "allowed": allowed,
                "is_accepting": bool(matcher.is_accepting()),
                "ff_tokens": ff_chain,
            }
        )

        # Deterministically prioritize active-vocab / priority tokens first, then sorted tokens
        prio_branch = [t for t in allowed_sorted if t in prio and t != eos_id]
        other_branch = [t for t in allowed_sorted if t not in prio and t != eos_id]
        expand_tokens = (prio_branch + other_branch)[:branch_limit]

        for t_id in expand_tokens:
            m_next = matcher.deep_copy()
            if not m_next.consume_token(int(t_id)) or m_next.is_error():
                continue
            ff = [int(x) for x in m_next.compute_ff_tokens()]
            if ff:
                m_next.consume_tokens(ff)
            queue.append((m_next, prefix_ids + (int(t_id),) + tuple(ff), ff))

    return states


def _build_formatted_prompt(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    few_shot_messages: Optional[List[Dict[str, str]]] = None,
    use_chat_template: bool = True,
) -> str:
    """Format a prompt consistently using tokenizer chat template when available."""
    has_template = bool(
        use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if not has_template:
        return user_prompt

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot_messages:
        messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_prompt})

    kwargs: Dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _synthesize_default_prompts(grammar: str) -> List[str]:
    """Synthesize deterministic fallback anchor prompts from grammar literals."""
    literals = re.findall(r'"([^"\\]{2,32})"', grammar)
    clean_lits = []
    for lit in literals:
        s = lit.strip()
        if s and any(ch.isalnum() for ch in s) and s not in clean_lits:
            clean_lits.append(s)
        if len(clean_lits) >= 4:
            break
    prompts = ["Generate an output matching the grammar specification."]
    for lit in clean_lits:
        prompts.append(f"Produce a valid response using '{lit}'.")
    return prompts


def _get_module_by_path(root: Any, path: str) -> Any:
    cur = root
    for attr in path.split("."):
        if attr.isdigit():
            cur = cur[int(attr)]
        else:
            cur = getattr(cur, attr)
    return cur


def _discover_final_layer_spec(
    model: Any, tensor_names: Sequence[str]
) -> Dict[str, Any]:
    """Auto-detect the final Transformer layer MLP, norm, and unembedding tensors."""
    layer_pattern = re.compile(
        r"^((?:model\.)?(?:language_model\.)?(?:layers|h|blocks))\.(\d+)\."
    )
    layer_matches: List[Tuple[str, int]] = []
    for name in tensor_names:
        m = layer_pattern.match(name)
        if m:
            layer_matches.append((m.group(1), int(m.group(2))))

    if not layer_matches:
        raise ValueError(
            "Could not discover Transformer layers in checkpoint tensors. "
            f"Sample keys: {list(tensor_names)[:12]}"
        )

    prefix, max_layer = max(layer_matches, key=lambda x: x[1])
    layer_prefix = f"{prefix}.{max_layer}"
    name_set = set(tensor_names)

    # Identify MLP projection names
    if (
        f"{layer_prefix}.mlp.gate_proj.weight" in name_set
        and f"{layer_prefix}.mlp.up_proj.weight" in name_set
        and f"{layer_prefix}.mlp.down_proj.weight" in name_set
    ):
        mlp_type = "swiglu"
        mlp_tensors = {
            "gate": f"{layer_prefix}.mlp.gate_proj.weight",
            "up": f"{layer_prefix}.mlp.up_proj.weight",
            "down": f"{layer_prefix}.mlp.down_proj.weight",
        }
    elif (
        f"{layer_prefix}.mlp.c_fc.weight" in name_set
        and f"{layer_prefix}.mlp.c_proj.weight" in name_set
    ):
        mlp_type = "standard"
        mlp_tensors = {
            "fc_in": f"{layer_prefix}.mlp.c_fc.weight",
            "fc_in_bias": f"{layer_prefix}.mlp.c_fc.bias"
            if f"{layer_prefix}.mlp.c_fc.bias" in name_set
            else None,
            "fc_out": f"{layer_prefix}.mlp.c_proj.weight",
        }
    elif (
        f"{layer_prefix}.mlp.fc1.weight" in name_set
        and f"{layer_prefix}.mlp.fc2.weight" in name_set
    ):
        mlp_type = "standard"
        mlp_tensors = {
            "fc_in": f"{layer_prefix}.mlp.fc1.weight",
            "fc_in_bias": f"{layer_prefix}.mlp.fc1.bias"
            if f"{layer_prefix}.mlp.fc1.bias" in name_set
            else None,
            "fc_out": f"{layer_prefix}.mlp.fc2.weight",
        }
    else:
        raise ValueError(
            f"Unsupported final-layer MLP structure under '{layer_prefix}.mlp'. "
            "Expected SwiGLU (gate_proj/up_proj/down_proj) or standard MLP (c_fc/c_proj or fc1/fc2)."
        )

    # Identify unembedding tensor
    unembed_candidates = (
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "transformer.wte.weight",
    )
    unembed_name = next((c for c in unembed_candidates if c in name_set), None)
    if unembed_name is None:
        raise ValueError(
            "Could not find unembedding tensor (`lm_head.weight` or `embed_tokens.weight`)."
        )

    # Identify final LayerNorm / RMSNorm tensor
    norm_candidates = (
        "model.norm.weight",
        "model.language_model.norm.weight",
        "transformer.ln_f.weight",
        "gpt_neox.final_layer_norm.weight",
    )
    norm_name = next((c for c in norm_candidates if c in name_set), None)

    mlp_module = _get_module_by_path(model, f"{layer_prefix}.mlp")

    return {
        "layer_index": max_layer,
        "layer_prefix": layer_prefix,
        "mlp_type": mlp_type,
        "mlp_tensors": mlp_tensors,
        "mlp_module": mlp_module,
        "unembed_name": unembed_name,
        "norm_name": norm_name,
    }


def _find_safetensors_files(
    input_dir: str, modified_keys: Sequence[str]
) -> Tuple[List[str], Dict[str, str]]:
    """Locate safetensors files and map each tensor key to its file path."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        weight_map = idx_data.get("weight_map", {})
        all_shards = sorted(
            {os.path.join(input_dir, fname) for fname in weight_map.values()}
        )
        key_to_file = {
            k: os.path.join(input_dir, weight_map[k])
            for k in modified_keys
            if k in weight_map
        }
        return all_shards, key_to_file

    single_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(single_path):
        return [single_path], {k: single_path for k in modified_keys}

    st_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".safetensors")
    )
    if not st_files:
        raise FileNotFoundError(f"No .safetensors weights found in {input_dir!r}")
    if len(st_files) == 1:
        return st_files, {k: st_files[0] for k in modified_keys}
    raise FileNotFoundError(
        f"Multiple .safetensors files in {input_dir!r} without `model.safetensors.index.json`."
    )


def _build_clean_unembed_helper(
    torch_mod: Any,
    raw_tensors: Dict[str, Any],
    spec: Dict[str, Any],
    model: Any,
):
    """Build loop-direction-projected unembedding direction lookup."""
    if spec["unembed_name"] in raw_tensors:
        embed_w = raw_tensors[spec["unembed_name"]].float().cpu()
    else:
        embed_w = model.get_output_embeddings().weight.detach().float().cpu()

    if spec["norm_name"] and spec["norm_name"] in raw_tensors:
        gamma = raw_tensors[spec["norm_name"]].float().cpu()
    else:
        gamma = torch_mod.ones(embed_w.shape[1], dtype=torch_mod.float32)

    m_vocab = embed_w * gamma
    u_loop = torch_mod.nn.functional.normalize(m_vocab.mean(dim=0), dim=-1)

    def get_clean_dir(tok_id: int):
        m_t = m_vocab[int(tok_id)]
        m_clean = m_t - (m_t @ u_loop) * u_loop
        return torch_mod.nn.functional.normalize(m_clean / (gamma + 1e-5), dim=-1)

    return embed_w, gamma, get_clean_dir


def _solve_ridge_dual_basis(
    torch_mod: Any, A: Any, Y: Any, min_reg: float = 1e-2
) -> Any:
    """Solve W = A^T (A A^T + reg I)^-1 Y in float64 for guaranteed numerical stability."""
    A64 = A.double()
    Y64 = Y.double()
    gram = A64 @ A64.T
    diag_mean = float(gram.diag().mean()) if gram.numel() > 0 else 1.0
    reg = max(min_reg, 1e-4 * diag_mean)
    n_rows = A64.shape[0]
    eye = torch_mod.eye(n_rows, dtype=torch_mod.float64)
    return (A64.T @ torch_mod.linalg.solve(gram + reg * eye, Y64)).float()


def _clamp_for_dtype(torch_mod: Any, tensor: Any, target_dtype: Any) -> Any:
    """Clamp tensor cleanly into target dtype finite range (critical for float8_e4m3fn)."""
    if str(target_dtype) == "torch.float8_e4m3fn":
        t = torch_mod.nan_to_num(tensor, nan=0.0, posinf=448.0, neginf=-448.0)
        return torch_mod.clamp(t, min=-448.0, max=448.0).to(target_dtype)
    return torch_mod.nan_to_num(tensor, nan=0.0).to(target_dtype)


def _apply_swiglu_grammar_gate_inhibition(
    ml: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    llg_tok: Any,
    g_spec: str,
    grammar: str,
    spec: Dict[str, Any],
    embed_w: Any,
    gamma: Any,
    gate_w: Any,
    down_w: Any,
    anchor_prompts: List[str],
    gate_inhibition_strength: float,
    max_states: int,
    device: str,
    extra_prefixes: Optional[List[Tuple[int, ...]]] = None,
) -> Dict[str, Any]:
    """Mechanistically inhibit active sparse SwiGLU neurons that emit CFG-forbidden tokens."""
    torch_mod = ml["torch"]
    llg_mod = ml["llg"]

    col_norms = down_w.norm(dim=0)
    total_neurons = int(down_w.shape[1])
    active_neurons = torch_mod.nonzero(col_norms > 1e-6).squeeze(-1).tolist()
    if not active_neurons or len(active_neurons) > min(512, total_neurons // 2):
        return {
            "applied": False,
            "reason": f"Final layer MLP has {len(active_neurons)}/{total_neurons} active neurons (dense; skipping sparse neuron gating).",
            "active_neurons": len(active_neurons),
        }

    # Map each active neuron j to the vocabulary token it writes most strongly to
    neuron_token: Dict[int, int] = {}
    for j in active_neurons:
        logits_j = embed_w @ (down_w[:, j] * gamma)
        neuron_token[j] = int(logits_j.argmax().item())
    active_vocab = set(neuron_token.values())

    explored = _explore_grammar_states(
        llg_mod,
        llg_tok,
        g_spec,
        tokenizer,
        priority_tokens=active_vocab,
        max_states=max_states,
        branch_limit=16,
    )

    # Merge any extra teacher trajectory prefixes not already present
    eos_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = int(llg_tok.vocab_size)
    bitmask = llg_mod.numpy.allocate_token_bitmask(1, vocab_size)
    known_prefixes = {tuple(st["prefix_ids"]) for st in explored}
    if extra_prefixes:
        for pref in extra_prefixes:
            if pref in known_prefixes:
                continue
            m = llg_mod.LLMatcher(llg_tok, g_spec)
            if pref:
                ok = m.consume_tokens(list(pref))
                if not ok or m.is_error():
                    continue
            allowed = _get_allowed_tokens(llg_mod, m, bitmask, vocab_size, eos_id)
            explored.append(
                {
                    "prefix_ids": pref,
                    "prefix_text": tokenizer.decode(list(pref)) if pref else "",
                    "allowed": allowed,
                    "is_accepting": bool(m.is_accepting()),
                    "ff_tokens": [],
                }
            )
            known_prefixes.add(pref)

    # Collect final-layer MLP pre-activation states across anchor prompts & grammar prefixes
    mlp_in: Dict[str, Any] = {}
    hook = spec["mlp_module"].register_forward_pre_hook(
        lambda _mod, inp: mlp_in.__setitem__("x", inp[0].detach().float().cpu())
    )

    train_x: List[Any] = []
    train_y: List[int] = []
    try:
        for p_text in anchor_prompts[:3]:
            p_ids = tokenizer.encode(p_text, add_special_tokens=False)
            for s_idx, st in enumerate(explored):
                pref = list(st["prefix_ids"])
                seq = p_ids + pref
                inp_t = torch_mod.tensor([seq], dtype=torch_mod.long, device=device)
                _ = model(inp_t)
                train_x.append(mlp_in["x"][0, -1, :])
                train_y.append(s_idx)
    finally:
        hook.remove()

    n_states = len(explored)
    if not train_x or n_states == 0:
        return {
            "applied": False,
            "reason": "No reachable grammar states extracted.",
            "active_neurons": len(active_neurons),
        }

    A = torch_mod.stack(train_x)
    Y = torch_mod.zeros(len(train_x), n_states, dtype=torch_mod.float32)
    for row_i, s_idx in enumerate(train_y):
        Y[row_i, s_idx] = 1.0

    W_state = _solve_ridge_dual_basis(torch_mod, A, Y, min_reg=1e-2)

    gated_updates = 0
    for s_idx, st in enumerate(explored):
        allowed_set = st["allowed"]
        w_s = W_state[:, s_idx]
        for j in active_neurons:
            if neuron_token[j] not in allowed_set:
                gate_w[j] = gate_w[j] - gate_inhibition_strength * w_s
                gated_updates += 1

    return {
        "applied": True,
        "active_neurons": len(active_neurons),
        "states_compiled": n_states,
        "gated_neuron_state_pairs": gated_updates,
    }


def _apply_teacher_dual_basis_steering(
    ml: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    spec: Dict[str, Any],
    get_clean_dir: Any,
    d_model: int,
    raw_dtypes: Dict[str, Any],
    orig_weights: Dict[str, Any],
    formatted_prompts: List[str],
    teacher_token_seqs: List[List[int]],
    steering_gain: float,
    max_rounds: int,
    device: str,
    model_dtype: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run closed-form Outlines teacher trajectory steering into spare final-layer MLP slots."""
    torch_mod = ml["torch"]
    mlp_type = spec["mlp_type"]
    tensors = spec["mlp_tensors"]

    if mlp_type == "swiglu":
        down_orig = orig_weights[tensors["down"]]
        col_norms = down_orig.norm(dim=0)
        free_pool = torch_mod.argsort(col_norms).tolist()
        is_fp8 = str(raw_dtypes[tensors["gate"]]) == "torch.float8_e4m3fn"
    else:
        fc_out_orig = orig_weights[tensors["fc_out"]]
        # Conv1D in GPT-2 has shape (d_mlp, d_model); Linear has (d_model, d_mlp)
        is_conv1d = fc_out_orig.shape[1] == d_model and fc_out_orig.shape[0] != d_model
        out_norms = fc_out_orig.norm(dim=1) if is_conv1d else fc_out_orig.norm(dim=0)
        free_pool = torch_mod.argsort(out_norms).tolist()
        is_fp8 = False

    s_dict: Dict[Tuple[int, int], Tuple[Any, int]] = {}
    c_dict: Dict[Tuple[int, int], Any] = {}
    working_weights = {k: v.clone() for k, v in orig_weights.items()}
    rounds_run = 0

    for rnd in range(max_rounds):
        rounds_run = rnd + 1
        mlp_in: Dict[str, Any] = {}

        def _capture_mlp_in(
            _mod: Any, inp: Any, cache: Dict[str, Any] = mlp_in
        ) -> None:
            cache["x"] = inp[0].detach().float().cpu()

        hook = spec["mlp_module"].register_forward_pre_hook(_capture_mlp_in)
        new_s = 0
        try:
            for p_idx, (p_text, tgt_ids) in enumerate(
                zip(formatted_prompts, teacher_token_seqs, strict=True)
            ):
                if not tgt_ids:
                    continue
                p_ids = tokenizer.encode(p_text, add_special_tokens=False)
                seq = p_ids + tgt_ids[:-1]
                inp_t = torch_mod.tensor([seq], dtype=torch_mod.long, device=device)
                out = model(inp_t)
                xs = mlp_in["x"][0]
                start = len(p_ids) - 1
                for k, t_id in enumerate(tgt_ids):
                    pos = start + k
                    key = (p_idx, k)
                    raw_top = int(out.logits[0, pos].argmax().item())
                    if raw_top != t_id and key not in s_dict:
                        s_dict[key] = (xs[pos], int(t_id))
                        c_dict.pop(key, None)
                        new_s += 1
                    elif rnd == 0 and key not in s_dict:
                        c_dict[key] = xs[pos]
        finally:
            hook.remove()

        if new_s == 0:
            break

        s_keys = sorted(s_dict.keys())
        c_keys = sorted(c_dict.keys())
        S = torch_mod.stack([s_dict[k][0] for k in s_keys])
        t_list = [s_dict[k][1] for k in s_keys]
        if c_keys:
            C = torch_mod.stack([c_dict[k] for k in c_keys])
            A = torch_mod.cat([S, C], dim=0)
        else:
            A = S

        n_s = S.shape[0]
        if n_s > len(free_pool):
            raise ValueError(
                f"Required {n_s} steering neurons, exceeding final-layer MLP capacity {len(free_pool)}."
            )

        Y = torch_mod.zeros(A.shape[0], n_s, dtype=torch_mod.float32)
        Y[:n_s, :] = torch_mod.eye(n_s, dtype=torch_mod.float32)
        W = _solve_ridge_dual_basis(torch_mod, A, Y, min_reg=1e-2)
        u_mean = A.mean(dim=0)
        u_hat = u_mean / (u_mean.norm() + 1e-12)

        working_weights = {k: v.clone() for k, v in orig_weights.items()}
        if mlp_type == "swiglu":
            gate_w = working_weights[tensors["gate"]]
            up_w = working_weights[tensors["up"]]
            down_w = working_weights[tensors["down"]]
            for i, n_idx in enumerate(free_pool[:n_s]):
                g_vec = 20.0 * W[:, i]
                u_vec = 5.0 * W[:, i]
                d_vec = get_clean_dir(t_list[i]) * steering_gain
                if is_fp8:
                    max_g = float(g_vec.abs().max())
                    if max_g > 200.0:
                        scale = 200.0 / max_g
                        g_vec = g_vec * scale
                        u_vec = u_vec * scale
                        d_vec = d_vec / (scale * scale)
                gate_w[n_idx] = g_vec
                up_w[n_idx] = u_vec
                down_w[:, n_idx] = d_vec
        else:
            fc_in_w = working_weights[tensors["fc_in"]]
            fc_out_w = working_weights[tensors["fc_out"]]
            fc_in_b = (
                working_weights[tensors["fc_in_bias"]]
                if tensors["fc_in_bias"]
                else None
            )
            in_conv1d = fc_in_w.shape[0] == d_model and fc_in_w.shape[1] != d_model
            out_conv1d = fc_out_w.shape[1] == d_model and fc_out_w.shape[0] != d_model
            for i, n_idx in enumerate(free_pool[:n_s]):
                in_vec = 25.0 * W[:, i] - 0.8 * u_hat
                out_vec = get_clean_dir(t_list[i]) * steering_gain
                if in_conv1d:
                    fc_in_w[:, n_idx] = in_vec
                else:
                    fc_in_w[n_idx, :] = in_vec
                if fc_in_b is not None:
                    fc_in_b[n_idx] = -12.0
                if out_conv1d:
                    fc_out_w[n_idx, :] = out_vec
                else:
                    fc_out_w[:, n_idx] = out_vec

        # Update live model parameters (with target storage dtype quantization round-trip) for next pass
        for k, tensor_val in working_weights.items():
            if k in tensors.values():
                mod_attr = _get_module_by_path(model, k.rsplit(".", 1)[0])
                param_name = k.rsplit(".", 1)[1]
                quant_sim = _clamp_for_dtype(
                    torch_mod, tensor_val, raw_dtypes[k]
                ).float()
                getattr(mod_attr, param_name).data = quant_sim.to(model_dtype).to(
                    device
                )

    stats = {
        "rounds_executed": rounds_run,
        "intervened_states": len(s_dict),
        "aligned_control_states": len(c_dict),
        "neurons_allocated": len(s_dict),
    }
    return working_weights, stats


def _load_model_for_inference(
    ml: Dict[str, Any], model_dir: str, device_pref: str = "auto"
):
    """Load tokenizer + model cleanly, upcasting FP8 weights to float32 in memory if needed."""
    torch_mod = ml["torch"]
    AutoTokenizer = ml["AutoTokenizer"]
    AutoModelForCausalLM = ml["AutoModelForCausalLM"]

    st_files, _ = _find_safetensors_files(model_dir, [])
    has_fp8 = False
    for sf in st_files:
        raw = ml["load_file"](sf)
        if any(str(v.dtype) == "torch.float8_e4m3fn" for v in raw.values()):
            has_fp8 = True
            break

    if device_pref == "auto":
        if has_fp8:
            device = "cpu"
        elif torch_mod.backends.mps.is_available():
            device = "mps"
        elif torch_mod.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_pref

    if has_fp8:
        model_dtype = torch_mod.float32
    elif device == "cuda":
        model_dtype = torch_mod.bfloat16
    elif device == "mps":
        model_dtype = torch_mod.float16
    else:
        model_dtype = torch_mod.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device, model_dtype, has_fp8


def cmd_analyze_grammar(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compile an EBNF/Lark CFG using Outlines + LLGuidance and inspect its state graph."""
    ml = _require_ml_stack()
    _set_deterministic_mode(ml["torch"], int(args.get("seed", 0)))

    ckpt_dir = _resolve_checkpoint_dir(args["checkpoint"])
    tokenizer = ml["AutoTokenizer"].from_pretrained(ckpt_dir, trust_remote_code=True)
    llg_tok = ml["llg"].hf.from_tokenizer(tokenizer)
    g_spec = _parse_grammar_spec(ml["llg"], ml["outlines"], args["grammar"])
    max_states = int(args.get("max_states", 64))

    states = _explore_grammar_states(
        ml["llg"],
        llg_tok,
        g_spec,
        tokenizer,
        max_states=max_states,
        branch_limit=int(args.get("branch_limit", 16)),
    )

    all_reachable_tokens: Set[int] = set()
    state_summaries = []
    for idx, st in enumerate(states):
        all_reachable_tokens.update(st["allowed"])
        sample = [
            {"token_id": int(t), "token_text": tokenizer.decode([int(t)])}
            for t in sorted(st["allowed"])[:10]
        ]
        state_summaries.append(
            {
                "state_index": idx,
                "prefix_ids": list(st["prefix_ids"]),
                "prefix_text": st["prefix_text"],
                "allowed_count": len(st["allowed"]),
                "is_accepting": st["is_accepting"],
                "forced_forward_tokens": [int(t) for t in st["ff_tokens"]],
                "sample_allowed": sample,
            }
        )

    return {
        "ok": True,
        "operation": "analyze_grammar",
        "checkpoint_dir": ckpt_dir,
        "vocab_size": int(llg_tok.vocab_size),
        "num_states_explored": len(states),
        "reachable_tokens_count": len(all_reachable_tokens),
        "states": state_summaries[:40],
    }


def cmd_verify_bundle(args: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a standalone Hugging Face bundle using unwrapped AutoModelForCausalLM against a CFG."""
    ml = _require_ml_stack()
    _set_deterministic_mode(ml["torch"], int(args.get("seed", 0)))

    bundle_dir = _resolve_checkpoint_dir(args["checkpoint"])
    grammar = args["grammar"]
    prompts = list(args.get("prompts") or _synthesize_default_prompts(grammar))
    system_prompt = args.get("system_prompt")
    tools = args.get("tools_json")
    few_shot_messages = args.get("few_shot_messages")
    use_chat_template = bool(args.get("use_chat_template", True))
    max_new_tokens = int(args.get("max_new_tokens", 64))
    device_pref = str(args.get("device", "auto"))

    tokenizer, model, device, model_dtype, _ = _load_model_for_inference(
        ml, bundle_dir, device_pref
    )
    llg_tok = ml["llg"].hf.from_tokenizer(tokenizer)
    g_spec = _parse_grammar_spec(ml["llg"], ml["outlines"], grammar)

    details = []
    valid_count = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for p in prompts:
        fmt = _build_formatted_prompt(
            tokenizer, p, system_prompt, tools, few_shot_messages, use_chat_template
        )
        inp = tokenizer(fmt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if tokenizer.pad_token_id is None and eos_id is not None:
            gen_kwargs["pad_token_id"] = int(eos_id)
        out = model.generate(**inp, **gen_kwargs)
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gen_ids = out[0][inp["input_ids"].shape[1] :].tolist()
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Validate token stream up to EOS via LLMatcher
        trimmed_ids = []
        for tid in gen_ids:
            if eos_id is not None and int(tid) == int(eos_id):
                break
            trimmed_ids.append(int(tid))
        ok = _is_grammar_accepted(
            ml["llg"], llg_tok, g_spec, tokenizer, raw_text, trimmed_ids
        )
        if ok:
            valid_count += 1
        details.append(
            {
                "prompt": p,
                "raw_output": raw_text,
                "grammar_valid": ok,
                "latency_ms": lat_ms,
            }
        )

    total = len(prompts)
    return {
        "ok": True,
        "operation": "verify_bundle",
        "bundle_dir": bundle_dir,
        "device": device,
        "dtype": str(model_dtype),
        "grammar_valid_count": valid_count,
        "total_prompts": total,
        "grammar_compliance_rate": round(valid_count / total, 4) if total else 0.0,
        "details": details,
    }


def cmd_compile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compile an Outlines CFG/EBNF grammar deterministically into final-layer weights."""
    ml = _require_ml_stack()
    _set_deterministic_mode(ml["torch"], int(args.get("seed", 0)))
    torch_mod = ml["torch"]

    input_dir = _resolve_checkpoint_dir(args["checkpoint"])
    work_dir = os.path.abspath(
        args.get("work_dir") or os.environ.get("WSP_WORK_DIR", ".wsp-work")
    )
    output_dir = os.path.abspath(
        os.path.expanduser(
            args.get("output_dir") or os.path.join(work_dir, "outlines_cfg_bundle")
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    grammar = args["grammar"]
    prompts = list(args.get("prompts") or _synthesize_default_prompts(grammar))
    system_prompt = args.get("system_prompt")
    tools = args.get("tools_json")
    few_shot_messages = args.get("few_shot_messages")
    use_chat_template = bool(args.get("use_chat_template", True))
    gate_inhibition_strength = float(args.get("gate_inhibition_strength", 15.0))
    steering_gain = float(args.get("steering_gain", 50.0))
    max_new_tokens = int(args.get("max_new_tokens", 64))
    max_states = int(args.get("max_states", 64))
    max_rounds = int(args.get("max_refinement_rounds", 2))
    device_pref = str(args.get("device", "auto"))

    t_start = time.perf_counter()

    # 1. Load model and validate CFG with Outlines + LLGuidance
    tokenizer, model, device, model_dtype, has_fp8 = _load_model_for_inference(
        ml, input_dir, device_pref
    )
    llg_tok = ml["llg"].hf.from_tokenizer(tokenizer)
    g_spec = _parse_grammar_spec(ml["llg"], ml["outlines"], grammar)

    # 2. Generate exact deterministic teacher trajectories via Outlines wrapper
    outlines_model = ml["outlines"].from_transformers(model, tokenizer)
    outlines_model.type_adapter.has_chat_template = False
    generator = ml["outlines"].Generator(outlines_model, ml["outlines"].cfg(grammar))

    formatted_prompts = [
        _build_formatted_prompt(
            tokenizer, p, system_prompt, tools, few_shot_messages, use_chat_template
        )
        for p in prompts
    ]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    teacher_outputs: List[str] = []
    teacher_token_seqs: List[List[int]] = []
    teacher_prefixes: List[Tuple[int, ...]] = []

    for fmt in formatted_prompts:
        out_text = str(
            generator(fmt, max_new_tokens=max_new_tokens, do_sample=False)
        ).strip()
        teacher_outputs.append(out_text)
        t_ids = [int(t) for t in tokenizer.encode(out_text, add_special_tokens=False)]
        for k in range(len(t_ids) + 1):
            teacher_prefixes.append(tuple(t_ids[:k]))
        if eos_id is not None:
            t_ids.append(int(eos_id))
        teacher_token_seqs.append(t_ids)

    del generator, outlines_model

    # 3. Discover final layer & load raw safetensors preserving exact original storage dtypes
    st_shards, _ = _find_safetensors_files(input_dir, [])
    all_raw_tensors: Dict[str, Any] = {}
    shard_of_key: Dict[str, str] = {}
    for sf in st_shards:
        loaded = ml["load_file"](sf)
        for k, v in loaded.items():
            all_raw_tensors[k] = v
            shard_of_key[k] = sf

    spec = _discover_final_layer_spec(model, list(all_raw_tensors.keys()))
    mlp_tensors = spec["mlp_tensors"]
    target_keys = [v for v in mlp_tensors.values() if v is not None]

    raw_dtypes = {k: all_raw_tensors[k].dtype for k in target_keys}
    base_weights_f32 = {
        k: all_raw_tensors[k].float().cpu().clone() for k in target_keys
    }
    working_weights_f32 = {k: base_weights_f32[k].clone() for k in target_keys}

    embed_w, gamma, get_clean_dir = _build_clean_unembed_helper(
        torch_mod, all_raw_tensors, spec, model
    )
    d_model = int(embed_w.shape[1])
    del all_raw_tensors

    # 4. Mechanism A: Sparse SwiGLU Grammar-State Gate Inhibition (prompt-invariant)
    mech_a_report: Dict[str, Any] = {
        "applied": False,
        "reason": "Not applicable to non-SwiGLU MLP.",
    }
    if spec["mlp_type"] == "swiglu" and gate_inhibition_strength > 0:
        mech_a_report = _apply_swiglu_grammar_gate_inhibition(
            ml=ml,
            model=model,
            tokenizer=tokenizer,
            llg_tok=llg_tok,
            g_spec=g_spec,
            grammar=grammar,
            spec=spec,
            embed_w=embed_w,
            gamma=gamma,
            gate_w=working_weights_f32[mlp_tensors["gate"]],
            down_w=working_weights_f32[mlp_tensors["down"]],
            anchor_prompts=formatted_prompts,
            gate_inhibition_strength=gate_inhibition_strength,
            max_states=max_states,
            device=device,
            extra_prefixes=teacher_prefixes,
        )
        # Push Mechanism A gate updates (quantized to target storage dtype) into live model
        if mech_a_report.get("applied"):
            gate_key = mlp_tensors["gate"]
            mod_attr = _get_module_by_path(model, gate_key.rsplit(".", 1)[0])
            quant_gate = _clamp_for_dtype(
                torch_mod, working_weights_f32[gate_key], raw_dtypes[gate_key]
            ).float()
            mod_attr.weight.data = quant_gate.to(model_dtype).to(device)

    # 5. Mechanism B: Iterative Closed-Form Outlines Teacher Trajectory Dual-Basis Steering
    edited_weights_f32, mech_b_report = _apply_teacher_dual_basis_steering(
        ml=ml,
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        get_clean_dir=get_clean_dir,
        d_model=d_model,
        raw_dtypes=raw_dtypes,
        orig_weights=working_weights_f32,
        formatted_prompts=formatted_prompts,
        teacher_token_seqs=teacher_token_seqs,
        steering_gain=steering_gain,
        max_rounds=max_rounds,
        device=device,
        model_dtype=model_dtype,
    )

    # 6. Export clean Hugging Face bundle preserving original dtypes (0-byte overhead)
    edited_shards: Dict[str, Dict[str, Any]] = {}
    for k in target_keys:
        sf = shard_of_key[k]
        if sf not in edited_shards:
            edited_shards[sf] = ml["load_file"](sf)
        edited_shards[sf][k] = _clamp_for_dtype(
            torch_mod, edited_weights_f32[k], raw_dtypes[k]
        )

    from prelude import read_safetensors_header

    input_st_bytes = sum(os.path.getsize(sf) for sf in st_shards)
    output_st_paths: List[str] = []
    for sf in st_shards:
        dst_sf = os.path.join(output_dir, os.path.basename(sf))
        if sf in edited_shards:
            orig_hdr, _ = read_safetensors_header(sf)
            orig_meta = orig_hdr.get("__metadata__")
            ml["save_file"](edited_shards[sf], dst_sf, metadata=orig_meta)
        elif os.path.abspath(sf) != os.path.abspath(dst_sf):
            shutil.copy2(sf, dst_sf)
        output_st_paths.append(dst_sf)

    output_st_bytes = sum(os.path.getsize(p) for p in output_st_paths)

    # Copy HF config and tokenizer files (without wrapper package clutter)
    copied_files = []
    for fname in HF_BUNDLE_FILES:
        src_f = os.path.join(input_dir, fname)
        dst_f = os.path.join(output_dir, fname)
        if os.path.exists(src_f):
            if os.path.abspath(src_f) != os.path.abspath(dst_f):
                shutil.copy2(src_f, dst_f)
            copied_files.append(fname)

    # Generate WSP diff manifest (`manifest.json` + `edit.npz`)
    from wsp_engine import build_edit_manifest

    manifest_base = {k: base_weights_f32[k].numpy() for k in target_keys}
    manifest_edited = {
        k: _clamp_for_dtype(torch_mod, edited_weights_f32[k], raw_dtypes[k])
        .float()
        .cpu()
        .numpy()
        for k in target_keys
    }
    manifest = build_edit_manifest(manifest_base, manifest_edited, output_dir)

    # 7. Reload exported bundle with standard unwrapped AutoModelForCausalLM and verify
    del model
    verify_res = cmd_verify_bundle(
        {
            "checkpoint": output_dir,
            "grammar": grammar,
            "prompts": prompts,
            "system_prompt": system_prompt,
            "tools_json": tools,
            "few_shot_messages": few_shot_messages,
            "use_chat_template": use_chat_template,
            "max_new_tokens": max_new_tokens,
            "device": device_pref,
            "seed": int(args.get("seed", 0)),
        }
    )

    teacher_matches = 0
    for detail, t_out in zip(verify_res["details"], teacher_outputs, strict=True):
        matched = detail["raw_output"] == t_out
        detail["outlines_teacher_output"] = t_out
        detail["matches_outlines_teacher"] = matched
        if matched:
            teacher_matches += 1

    primary_model_path = (
        os.path.join(output_dir, "model.safetensors")
        if os.path.exists(os.path.join(output_dir, "model.safetensors"))
        else output_dir
    )

    return {
        "ok": True,
        "operation": "compile",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "primary_model_path": primary_model_path,
        "final_layer_index": spec["layer_index"],
        "mlp_type": spec["mlp_type"],
        "tensors_modified": target_keys,
        "original_dtypes": {k: str(raw_dtypes[k]) for k in target_keys},
        "file_size_comparison": {
            "input_safetensors_bytes": input_st_bytes,
            "output_safetensors_bytes": output_st_bytes,
            "size_delta_bytes": output_st_bytes - input_st_bytes,
            "overhead_pct": round(
                ((output_st_bytes - input_st_bytes) / input_st_bytes * 100.0)
                if input_st_bytes
                else 0.0,
                6,
            ),
        },
        "mechanism_a_gate_inhibition": mech_a_report,
        "mechanism_b_teacher_steering": mech_b_report,
        "manifest_summary": {
            "changed_tensors": manifest["changed_tensors"],
            "changed_entries_total": manifest["changed_entries_total"],
            "manifest_json": os.path.join(output_dir, "manifest.json"),
            "edit_npz": manifest["edit_npz"],
        },
        "bundle_files_copied": copied_files,
        "verification": {
            "grammar_valid_count": verify_res["grammar_valid_count"],
            "total_prompts": verify_res["total_prompts"],
            "grammar_compliance_rate": verify_res["grammar_compliance_rate"],
            "teacher_exact_match_count": teacher_matches,
            "teacher_exact_match_rate": round(teacher_matches / len(prompts), 4)
            if prompts
            else 0.0,
            "details": verify_res["details"],
        },
        "elapsed_s": round(time.perf_counter() - t_start, 2),
    }


def run_outlines_cfg(args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch an `outlines_cfg` operation (`analyze_grammar`, `compile`, or `verify_bundle`)."""
    op = str(args.get("operation", "compile")).strip()
    if op == "analyze_grammar":
        return cmd_analyze_grammar(args)
    if op == "compile":
        return cmd_compile(args)
    if op == "verify_bundle":
        return cmd_verify_bundle(args)
    raise ValueError(
        f"Unknown outlines_cfg operation: {op!r}. Expected 'analyze_grammar', 'compile', or 'verify_bundle'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Outlines CFG/EBNF compiler for final-layer Transformer weights."
    )
    parser.add_argument(
        "--operation",
        choices=["analyze_grammar", "compile", "verify_bundle"],
        default="compile",
        help="Operation to execute (default: compile).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Input HF model directory or .safetensors path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for clean Hugging Face bundle.",
    )
    parser.add_argument(
        "--grammar", default=None, help="Inline CFG/EBNF/Lark grammar string."
    )
    parser.add_argument(
        "--grammar-file", default=None, help="Path to a .lark/.ebnf grammar file."
    )
    parser.add_argument(
        "--prompts-json",
        default=None,
        help="JSON list or file path of calibration prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for chat formatting.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gate-inhibition-strength", type=float, default=15.0)
    parser.add_argument("--steering-gain", type=float, default=50.0)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grammar_text = args.grammar
    if args.grammar_file:
        with open(args.grammar_file, "r", encoding="utf-8") as f:
            grammar_text = f.read()
    if not grammar_text:
        parser.error("Either --grammar or --grammar-file is required.")

    prompts = None
    if args.prompts_json:
        if os.path.exists(args.prompts_json):
            with open(args.prompts_json, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        else:
            prompts = json.loads(args.prompts_json)

    res = run_outlines_cfg(
        {
            "operation": args.operation,
            "checkpoint": args.checkpoint,
            "output_dir": args.output_dir,
            "grammar": grammar_text,
            "prompts": prompts,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "gate_inhibition_strength": args.gate_inhibition_strength,
            "steering_gain": args.steering_gain,
            "device": args.device,
            "seed": args.seed,
        }
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

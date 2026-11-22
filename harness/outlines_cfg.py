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
import gc
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
    "tokenizer.model",
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
        from safetensors import safe_open
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
        "safe_open": safe_open,
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


_W3C_LEXER_TERMINALS: Dict[str, str] = {
    "regular_identifier": "/[A-Za-z_][A-Za-z0-9_]*/",
    "SQL_language_identifier": "/[A-Za-z][A-Za-z0-9_]*/",
    "delimited_identifier": r'/"([^"\r\n]|"")+"/',
    "character_string_literal": r"/'([^'\r\n]|'')*'/",
    "national_character_string_literal": r"/[Nn]'([^'\r\n]|'')*'/",
    "binary_string_literal": r"/[Xx]'[0-9A-Fa-f ]*'/",
    "unsigned_integer": "/[0-9]+/",
    "exact_numeric_literal": r"/([0-9]+(\.[0-9]*)?|\.[0-9]+)/",
    "approximate_numeric_literal": r"/([0-9]+(\.[0-9]*)?|\.[0-9]+)[Ee][+-]?[0-9]+/",
    "space": '" "',
    "nondoublequote_character": r'/[^"\r\n]/',
    "nonquote_character": r"/[^'\r\n]/",
    "identifier_start": "/[A-Za-z_]/",
    "identifier_extend": "/[A-Za-z0-9_]/",
}


def _resolve_safe_input_file(
    path_str: str, param_name: str, missing_label: str
) -> str:
    """Resolve an input file path while blocking protected harness/eval directories."""
    resolved = os.path.abspath(os.path.expanduser(str(path_str)))
    real = os.path.realpath(resolved)
    allow_eval = os.environ.get("WSP_ALLOW_EVAL_ACCESS") == "1"
    protected_segments = {"eval", "evals", "harness", "extensions"}
    protected_basenames = {
        "run.sh",
        "edit.sh",
        "eval.sh",
        "install-extension.sh",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "session.log",
        "tool_calls.jsonl",
    }
    for cand in (resolved, real):
        parts = cand.replace("\\", "/").split("/")
        if not allow_eval and (set(parts) & protected_segments):
            raise PermissionError(
                f"Access to `{param_name}` inside protected directory is blocked: {path_str!r}"
            )
        if not allow_eval and os.path.basename(cand) in protected_basenames:
            raise PermissionError(
                f"Access to protected file via `{param_name}` is blocked: {path_str!r}"
            )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{missing_label} not found: {path_str!r}")
    return resolved


def _load_grammar_text(args: Dict[str, Any]) -> str:
    """Load grammar text from `grammar` or `grammar_file`, blocking protected paths."""
    raw = args.get("grammar")
    if isinstance(raw, str) and raw.strip():
        return raw

    grammar_file = args.get("grammar_file")
    if grammar_file:
        resolved = _resolve_safe_input_file(
            str(grammar_file), "grammar_file", "Grammar file"
        )
        with open(resolved, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"Grammar file is empty: {grammar_file!r}")
        return text

    raise ValueError("Either `grammar` or `grammar_file` must be provided.")


def _coerce_prompt_items(raw_items: Any) -> List[str]:
    """Extract non-empty prompt strings from a sequence or {'prompts': ...} container."""
    if isinstance(raw_items, dict) and "prompts" in raw_items:
        raw_items = raw_items["prompts"]
    if not isinstance(raw_items, list):
        raise ValueError("Prompts JSON must be a list of strings or {'prompts': [...]}.")
    clean: List[str] = []
    for item in raw_items:
        if isinstance(item, dict) and "prompt" in item:
            val = str(item["prompt"])
        elif item is None:
            continue
        else:
            val = str(item)
        if val.strip():
            clean.append(val)
    return clean


def _load_prompts(
    args: Dict[str, Any], effective_grammar: str
) -> Tuple[List[str], str]:
    """Resolve prompts from inline `prompts`, `prompts_file`, or grammar fallback."""
    raw_prompts = args.get("prompts")
    if isinstance(raw_prompts, str) and raw_prompts.strip():
        stripped = raw_prompts.lstrip()
        if stripped.startswith(("[", "{")):
            try:
                clean = _coerce_prompt_items(json.loads(raw_prompts))
                if clean:
                    return clean, "inline"
            except Exception:
                pass
        return [raw_prompts], "inline"
    if isinstance(raw_prompts, list):
        clean = _coerce_prompt_items(raw_prompts)
        if clean:
            return clean, "inline"

    prompts_file = args.get("prompts_file")
    if prompts_file:
        resolved = _resolve_safe_input_file(
            str(prompts_file), "prompts_file", "Prompts file"
        )
        with open(resolved, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"Prompts file contains no valid prompts: {prompts_file!r}")
        clean: List[str] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                clean = [parsed] if parsed.strip() else []
            else:
                clean = _coerce_prompt_items(parsed)
        except json.JSONDecodeError:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            parsed_lines: List[Any] = []
            all_json = True
            for ln in lines:
                try:
                    parsed_lines.append(json.loads(ln))
                except json.JSONDecodeError:
                    all_json = False
                    break
            if all_json and parsed_lines:
                clean = _coerce_prompt_items(parsed_lines)
            else:
                clean = [ln.strip() for ln in lines if ln.strip()]
        if not clean:
            raise ValueError(f"Prompts file contains no valid prompts: {prompts_file!r}")
        return clean, "prompts_file"

    return _synthesize_default_prompts(effective_grammar), "synthesized_fallback"


def _normalize_ebnf_literal(lit: str) -> str:
    """Escape W3C EBNF literal to Lark double-quoted string and strip Bottlecaps '*' markers."""
    inner = lit[1:-1]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]*\*", inner):
        inner = inner[:-1]
    return '"' + inner.replace("\\", "\\\\").replace('"', '\\"') + '"'


def convert_w3c_ebnf_to_lark(grammar: str) -> Tuple[str, Dict[str, Any]]:
    """Deterministically translate W3C/ISO EBNF (`::=`) into LLGuidance Lark syntax."""
    text = re.sub(r"/\*.*?\*/", "", grammar, flags=re.S)
    text = re.sub(r"(?m)//[^\n]*$", "", text)
    chunks = re.split(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*::=", text)
    if len(chunks) < 3:
        raise ValueError("Grammar does not contain W3C/ISO EBNF `::=` rules.")

    rules: Dict[str, List[str]] = {}
    order: List[str] = []
    for i in range(1, len(chunks), 2):
        name = chunks[i].strip()
        body = " ".join(chunks[i + 1].split())
        if not body:
            continue
        if name not in rules:
            order.append(name)
            rules[name] = [body]
        else:
            rules[name].append(body)

    if not order:
        raise ValueError("No valid W3C/ISO EBNF productions found.")

    token_rx = re.compile(r'("[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_.-]*)')
    referenced_or_defined: Set[str] = set(order)
    for bs in rules.values():
        for b in bs:
            for m in token_rx.finditer(b):
                tok = m.group(0)
                if not (tok.startswith('"') or tok.startswith("'")):
                    referenced_or_defined.add(tok)

    active_lexer_terminals = {
        k: v for k, v in _W3C_LEXER_TERMINALS.items() if k in referenced_or_defined
    }

    sym_map: Dict[str, str] = {}
    used_lower: Set[str] = set()
    for name in order:
        cand = "r_" + re.sub(r"[^a-z0-9_]", "_", name.lower())
        while cand in used_lower:
            cand = f"{cand}_{len(used_lower)}"
        used_lower.add(cand)
        sym_map[name] = cand

    for k in active_lexer_terminals:
        if k not in sym_map:
            cand = "r_" + re.sub(r"[^a-z0-9_]", "_", k.lower())
            while cand in used_lower:
                cand = f"{cand}_{len(used_lower)}"
            used_lower.add(cand)
            sym_map[k] = cand
            order.append(k)
            rules[k] = []

    # Inline single-literal productions (keywords/punctuation) to compact the Earley graph
    literal_inlines: Dict[str, str] = {}
    for name in order:
        if name in active_lexer_terminals or len(rules[name]) != 1:
            continue
        b = rules[name][0].strip()
        m = re.fullmatch(r'("[^"]+"|\'[^\']+\')', b)
        if m:
            literal_inlines[name] = _normalize_ebnf_literal(m.group(0))

    def _convert_body(body: str) -> str:
        out: List[str] = []
        pos = 0
        for m in token_rx.finditer(body):
            out.append(body[pos : m.start()])
            tok = m.group(0)
            if tok.startswith('"') or tok.startswith("'"):
                out.append(_normalize_ebnf_literal(tok))
            elif tok in literal_inlines:
                out.append(literal_inlines[tok])
            elif tok in sym_map:
                out.append(sym_map[tok])
            else:
                out.append(
                    '"' + tok.replace("\\", "\\\\").replace('"', '\\"') + '"'
                )
            pos = m.end()
        out.append(body[pos:])
        return "".join(out)

    # Prune productions unreachable from the entry rule (e.g., host-language embeddings)
    adj: Dict[str, List[str]] = {}
    for name in order:
        if name in active_lexer_terminals or name in literal_inlines:
            adj[name] = []
            continue
        refs: List[str] = []
        for m in token_rx.finditer(" ".join(rules[name])):
            tok = m.group(0)
            if (
                not (tok.startswith('"') or tok.startswith("'"))
                and tok in sym_map
                and tok not in literal_inlines
            ):
                refs.append(tok)
        adj[name] = refs

    reachable: Set[str] = set()
    stack = [order[0]]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)

    lines = [rf'start: " "? {sym_map[order[0]]}']
    for name in order:
        if name not in reachable or name in literal_inlines:
            continue
        if name in active_lexer_terminals:
            lines.append(f"{sym_map[name]}: {active_lexer_terminals[name]}")
            continue
        bodies = rules[name]
        if len(bodies) > 1:
            combined = " | ".join(f"( {_convert_body(b)} )" for b in bodies)
        else:
            combined = _convert_body(bodies[0])
        lines.append(f"{sym_map[name]}: {combined}")
    lines.append(r"%ignore /[ \t\r\n]+/")

    lark_text = "\n".join(lines)
    meta = {
        "format": "w3c_ebnf_to_lark",
        "converted_from": "w3c_ebnf",
        "entry_rule": order[0],
        "total_rules": len(order),
        "inlined_literals": len(literal_inlines),
        "reachable_rules": len(reachable),
    }
    return lark_text, meta


def _parse_grammar_spec(
    llg_mod: Any,
    outlines_mod: Any,
    grammar: str,
    llg_tok: Any = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Validate and compile GBNF, Lark, or W3C/ISO EBNF grammar via Outlines and LLGuidance."""
    if not grammar or not grammar.strip():
        raise ValueError("`grammar` must be a non-empty CFG/EBNF/Lark specification.")

    last_err: Optional[str] = None
    for fmt in ("grammar", "lark"):
        try:
            spec = llg_mod.grammar_from(fmt, grammar)
        except ValueError as exc:
            last_err = str(exc)
            continue
        if llg_tok is not None:
            matcher = llg_mod.LLMatcher(llg_tok, spec)
            if matcher.is_error():
                last_err = matcher.get_error()
                continue
        outlines_mod.cfg(grammar)
        return grammar, spec, {"format": fmt, "converted_from": None}

    if "::=" in grammar:
        effective_lark, meta = convert_w3c_ebnf_to_lark(grammar)
        spec = llg_mod.grammar_from("lark", effective_lark)
        if llg_tok is not None:
            matcher = llg_mod.LLMatcher(llg_tok, spec)
            if matcher.is_error():
                raise ValueError(
                    f"Converted W3C EBNF grammar failed LLMatcher validation: {matcher.get_error()}"
                )
        outlines_mod.cfg(effective_lark)
        return effective_lark, spec, meta

    raise ValueError(f"Failed to parse grammar specification: {last_err or 'invalid grammar'}")


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
        if (
            s
            and not s.startswith("!!")
            and "See the Syntax Rules" not in s
            and any(ch.isalnum() for ch in s)
            and s not in clean_lits
        ):
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
    up_w: Any,
    down_w: Any,
    anchor_prompts: List[str],
    gate_inhibition_strength: float,
    max_states: int,
    device: str,
    extra_prefixes: Optional[List[Tuple[int, ...]]] = None,
    max_dense_neurons: int = 256,
) -> Tuple[Dict[str, Any], Set[int]]:
    """Mechanistically inhibit active SwiGLU neurons that emit CFG-forbidden tokens."""
    torch_mod = ml["torch"]
    llg_mod = ml["llg"]

    col_norms = down_w.norm(dim=0)
    total_neurons = int(down_w.shape[1])
    active_neurons = torch_mod.nonzero(col_norms > 1e-6).reshape(-1).tolist()
    if not active_neurons:
        return (
            {
                "applied": False,
                "reason": "Final layer MLP has 0 active neurons.",
                "active_neurons": 0,
                "total_neurons": total_neurons,
            },
            set(),
        )

    is_dense = len(active_neurons) > min(512, total_neurons // 2)
    if is_dense and max_dense_neurons <= 0:
        return (
            {
                "applied": False,
                "reason": "max_dense_inhibition_neurons <= 0 (dense gating disabled).",
                "active_neurons": len(active_neurons),
                "total_neurons": total_neurons,
            },
            set(),
        )

    neuron_token: Dict[int, int] = {}
    if not is_dense:
        logits_sel = embed_w @ (down_w[:, active_neurons] * gamma.unsqueeze(1))
        top_toks = logits_sel.argmax(dim=0).tolist()
        neuron_token = {int(j): int(t) for j, t in zip(active_neurons, top_toks)}
        priority_tokens: Optional[Set[int]] = set(neuron_token.values())
    else:
        priority_tokens = None

    explored = _explore_grammar_states(
        llg_mod,
        llg_tok,
        g_spec,
        tokenizer,
        priority_tokens=priority_tokens,
        max_states=max_states,
        branch_limit=16,
    )

    # Merge extra teacher trajectory prefixes up to 2x max_states cap
    eos_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = int(llg_tok.vocab_size)
    bitmask = llg_mod.numpy.allocate_token_bitmask(1, vocab_size)
    known_prefixes = {tuple(st["prefix_ids"]) for st in explored}
    if extra_prefixes:
        unique_extra = sorted(
            {tuple(int(t) for t in pref) for pref in extra_prefixes},
            key=lambda p: (len(p), p),
        )
        for pref in unique_extra:
            if len(explored) >= max_states * 2:
                break
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

    # Collect final-layer MLP pre-activation states via maximal-prefix grouping
    prefix_to_sidx: Dict[Tuple[int, ...], int] = {
        tuple(st["prefix_ids"]): s_idx for s_idx, st in enumerate(explored)
    }
    sorted_prefixes = sorted(prefix_to_sidx.keys(), key=lambda p: (-len(p), p))
    maximal_prefixes: List[Tuple[int, ...]] = []
    covered_prefixes: Set[Tuple[int, ...]] = set()
    for pref in sorted_prefixes:
        if pref in covered_prefixes:
            continue
        maximal_prefixes.append(pref)
        for k in range(len(pref) + 1):
            covered_prefixes.add(pref[:k])

    mlp_in: Dict[str, Any] = {}
    hook = spec["mlp_module"].register_forward_pre_hook(
        lambda _mod, inp: mlp_in.__setitem__("x", inp[0].detach().float().cpu())
    )

    train_x: List[Any] = []
    train_y: List[int] = []
    rows_by_state: Dict[int, List[int]] = {}
    try:
        with torch_mod.no_grad():
            for p_text in anchor_prompts[:3]:
                p_ids = tokenizer.encode(p_text, add_special_tokens=True)
                if not p_ids:
                    continue
                captured_sidxs: Set[int] = set()
                for pref in maximal_prefixes:
                    seq = p_ids + list(pref)
                    inp_t = torch_mod.tensor([seq], dtype=torch_mod.long, device=device)
                    _ = model(inp_t)
                    xs = mlp_in["x"][0]
                    base_pos = len(p_ids) - 1
                    for k in range(len(pref) + 1):
                        s_idx = prefix_to_sidx.get(pref[:k])
                        if s_idx is not None and s_idx not in captured_sidxs:
                            captured_sidxs.add(s_idx)
                            rows_by_state.setdefault(s_idx, []).append(len(train_x))
                            train_x.append(xs[base_pos + k])
                            train_y.append(s_idx)
    finally:
        hook.remove()

    n_states = len(explored)
    if not train_x or n_states == 0:
        return (
            {
                "applied": False,
                "reason": "No reachable grammar states extracted.",
                "active_neurons": len(active_neurons),
                "total_neurons": total_neurons,
            },
            set(),
        )

    A = torch_mod.stack(train_x)
    Y = torch_mod.zeros(len(train_x), n_states, dtype=torch_mod.float32)
    for row_i, s_idx in enumerate(train_y):
        Y[row_i, s_idx] = 1.0

    W_state = _solve_ridge_dual_basis(torch_mod, A, Y, min_reg=1e-2)

    touched_neurons: Set[int] = set()
    gated_updates = 0
    if not is_dense:
        mode = "sparse_all_active"
        target_neurons = list(active_neurons)
        for s_idx, st in enumerate(explored):
            allowed_set = st["allowed"]
            w_s = W_state[:, s_idx]
            for j in target_neurons:
                if neuron_token[j] not in allowed_set:
                    gate_w[j] = gate_w[j] - gate_inhibition_strength * w_s
                    touched_neurons.add(j)
                    gated_updates += 1
    else:
        mode = "dense_top_impact_slice"
        dense_budget = min(int(max_dense_neurons), max(1, int(total_neurons * 0.05)))
        gate_pre = A @ gate_w.T  # (N_samples, total_neurons)
        up_pre = A @ up_w.T      # (N_samples, total_neurons)
        h_act = torch_mod.nn.functional.silu(gate_pre) * up_pre
        neuron_impact = h_act.abs().mean(dim=0) * col_norms
        top_k = min(dense_budget, len(active_neurons))
        active_idx_t = torch_mod.tensor(active_neurons, dtype=torch_mod.long)
        top_rel = torch_mod.topk(neuron_impact[active_idx_t], k=top_k).indices.tolist()
        target_neurons = [int(active_neurons[idx]) for idx in top_rel]

        scaled_down = down_w[:, target_neurons] * gamma.unsqueeze(1)
        logits_sel = embed_w @ scaled_down
        pos_toks = logits_sel.argmax(dim=0).tolist()
        neg_toks = logits_sel.argmin(dim=0).tolist()

        for s_idx, st in enumerate(explored):
            row_idxs = rows_by_state.get(s_idx)
            if not row_idxs:
                continue
            allowed_set = st["allowed"]
            w_s = W_state[:, s_idx]
            gate_s = gate_pre[row_idxs][:, target_neurons].mean(dim=0)
            up_s = up_pre[row_idxs][:, target_neurons].mean(dim=0)
            for idx, j in enumerate(target_neurons):
                if float(gate_s[idx]) <= -1.0:
                    continue
                emitted_tok = (
                    int(pos_toks[idx])
                    if float(up_s[idx]) >= 0.0
                    else int(neg_toks[idx])
                )
                if emitted_tok not in allowed_set:
                    gate_w[j] = gate_w[j] - gate_inhibition_strength * w_s
                    touched_neurons.add(j)
                    gated_updates += 1

    return (
        {
            "applied": gated_updates > 0,
            "mode": mode,
            "active_neurons": len(active_neurons),
            "selected_neurons": len(target_neurons),
            "inhibited_neurons_count": len(touched_neurons),
            "total_neurons": total_neurons,
            "capacity_fraction": round(len(target_neurons) / total_neurons, 4),
            "states_compiled": n_states,
            "gated_neuron_state_pairs": gated_updates,
        },
        touched_neurons,
    )


def _apply_teacher_dual_basis_steering(
    ml: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    llg_tok: Any,
    g_spec: str,
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
    eos_steering_gain: Optional[float] = None,
    reserved_neurons: Optional[Set[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run closed-form Outlines teacher trajectory steering into spare final-layer MLP slots."""
    torch_mod = ml["torch"]
    llg_mod = ml["llg"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eff_eos_gain = float(
        steering_gain if eos_steering_gain is None else eos_steering_gain
    )

    if not teacher_token_seqs or all(not seq for seq in teacher_token_seqs):
        return (
            {k: v.clone() for k, v in orig_weights.items()},
            {
                "rounds_executed": 0,
                "intervened_states": 0,
                "aligned_control_states": 0,
                "neurons_allocated": 0,
                "eos_intervened_states": 0,
                "eos_neurons_allocated": 0,
                "eos_non_accepting_guard_states": 0,
                "eos_steering_pooled": False,
                "reason": "No valid grammar-accepted Outlines teacher trajectories available for steering.",
            },
        )

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

    if reserved_neurons:
        free_pool = [n for n in free_pool if n not in reserved_neurons]

    # Compute grammar-agnostic CFG accepting status for every trajectory step via LLMatcher
    accepting_map: Dict[Tuple[int, int], bool] = {}
    for p_idx, tgt_ids in enumerate(teacher_token_seqs):
        if not tgt_ids:
            continue
        m = llg_mod.LLMatcher(llg_tok, g_spec)
        errored = False
        for k, t_id in enumerate(tgt_ids):
            t_int = int(t_id)
            if eos_id is not None and t_int == int(eos_id):
                accepting_map[(p_idx, k)] = (
                    True if errored else bool(m.is_accepting())
                )
                break
            if errored:
                accepting_map[(p_idx, k)] = False
                continue
            accepting_map[(p_idx, k)] = bool(m.is_accepting())
            if not m.consume_token(t_int) or m.is_error():
                errored = True

    s_dict: Dict[Tuple[int, int], Tuple[Any, int]] = {}
    c_dict: Dict[Tuple[int, int], Tuple[Any, int]] = {}
    working_weights = {k: v.clone() for k, v in orig_weights.items()}
    rounds_run = 0
    last_neurons_allocated = 0
    last_eos_intervened = 0
    last_eos_neurons = 0
    last_eos_guard_states = 0
    last_eos_pooled = False

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
            with torch_mod.no_grad():
                for p_idx, (p_text, tgt_ids) in enumerate(
                    zip(formatted_prompts, teacher_token_seqs, strict=True)
                ):
                    if not tgt_ids:
                        continue
                    p_ids = tokenizer.encode(p_text, add_special_tokens=True)
                    seq = p_ids + tgt_ids[:-1]
                    inp_t = torch_mod.tensor(
                        [seq], dtype=torch_mod.long, device=device
                    )
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
                            c_dict[key] = (xs[pos], int(t_id))
        finally:
            hook.remove()

        if new_s == 0:
            break

        s_keys = sorted(s_dict.keys())
        c_keys = sorted(c_dict.keys())
        all_keys = s_keys + c_keys
        S = torch_mod.stack([s_dict[k][0] for k in s_keys])
        s_targets = [s_dict[k][1] for k in s_keys]
        c_targets = [c_dict[k][1] for k in c_keys]
        all_targets = s_targets + c_targets
        if c_keys:
            C = torch_mod.stack([c_dict[k][0] for k in c_keys])
            A = torch_mod.cat([S, C], dim=0)
        else:
            A = S

        non_eos_rows = [
            i
            for i, t in enumerate(s_targets)
            if eos_id is None or int(t) != int(eos_id)
        ]
        eos_rows = [
            i
            for i, t in enumerate(s_targets)
            if eos_id is not None and int(t) == int(eos_id)
        ]

        if non_eos_rows:
            Y_content = torch_mod.zeros(
                A.shape[0], len(non_eos_rows), dtype=torch_mod.float32
            )
            for col_j, row_i in enumerate(non_eos_rows):
                Y_content[row_i, col_j] = 1.0
        else:
            Y_content = None

        if eos_rows:
            eos_row_set = set(eos_rows)
            guard_rows = [
                r
                for r in range(len(all_keys))
                if r not in eos_row_set
                and (
                    (not accepting_map.get(all_keys[r], False))
                    or (int(all_targets[r]) != int(eos_id))
                )
            ]
            y_eos = torch_mod.zeros(A.shape[0], 1, dtype=torch_mod.float32)
            y_eos[eos_rows, 0] = 1.0
            if guard_rows:
                y_eos[guard_rows, 0] = -0.5
            y_ones = torch_mod.ones(A.shape[0], 1, dtype=torch_mod.float32)

            Y_solve = (
                torch_mod.cat([Y_content, y_eos, y_ones], dim=1)
                if Y_content is not None
                else torch_mod.cat([y_eos, y_ones], dim=1)
            )
            W_solve = _solve_ridge_dual_basis(torch_mod, A, Y_solve, min_reg=1e-2)
            W_content = (
                W_solve[:, : len(non_eos_rows)] if Y_content is not None else None
            )
            w_eos = W_solve[:, -2]
            w_ones = W_solve[:, -1]

            eos_min = float((A[eos_rows] @ w_eos).min().item())
            guard_max = (
                float((A[guard_rows] @ w_eos).max().item()) if guard_rows else -1.0
            )
            ones_min = float((A[eos_rows] @ w_ones).min().item())
            use_pooled_eos = (eos_min > 0.5) and (guard_max < 0.0)
            use_ones_up = ones_min > 0.5

            if use_pooled_eos:
                eos_specs = [(w_eos, w_ones if use_ones_up else w_eos)]
            else:
                Y_eos_indiv = torch_mod.zeros(
                    A.shape[0], len(eos_rows), dtype=torch_mod.float32
                )
                for j_col, row_i in enumerate(eos_rows):
                    Y_eos_indiv[row_i, j_col] = 1.0
                if guard_rows:
                    Y_eos_indiv[guard_rows, :] = -0.5
                W_eos_indiv = _solve_ridge_dual_basis(
                    torch_mod, A, Y_eos_indiv, min_reg=1e-2
                )
                eos_specs = [
                    (
                        W_eos_indiv[:, j_col],
                        w_ones if use_ones_up else W_eos_indiv[:, j_col],
                    )
                    for j_col in range(len(eos_rows))
                ]
        else:
            guard_rows = []
            use_pooled_eos = False
            W_content = (
                _solve_ridge_dual_basis(torch_mod, A, Y_content, min_reg=1e-2)
                if Y_content is not None
                else None
            )
            eos_specs = []

        n_content = len(non_eos_rows)
        n_eos_neurons = len(eos_specs)
        total_needed = n_content + n_eos_neurons
        if total_needed > len(free_pool):
            raise ValueError(
                f"Required {total_needed} steering neurons, exceeding available final-layer MLP capacity {len(free_pool)}."
            )

        u_mean = A.mean(dim=0)
        u_hat = u_mean / (u_mean.norm() + 1e-12)

        working_weights = {k: v.clone() for k, v in orig_weights.items()}
        if mlp_type == "swiglu":
            gate_w = working_weights[tensors["gate"]]
            up_w = working_weights[tensors["up"]]
            down_w = working_weights[tensors["down"]]
            for i in range(n_content):
                n_idx = free_pool[i]
                w_col = W_content[:, i]
                t_id = s_targets[non_eos_rows[i]]
                g_vec = 20.0 * w_col
                u_vec = 5.0 * w_col
                d_vec = get_clean_dir(t_id) * steering_gain
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

            for j, (w_g_eos, w_u_eos) in enumerate(eos_specs):
                n_idx = free_pool[n_content + j]
                g_vec = 20.0 * w_g_eos
                u_vec = 5.0 * w_u_eos
                d_vec = get_clean_dir(int(eos_id)) * eff_eos_gain
                if is_fp8:
                    max_gu = max(float(g_vec.abs().max()), float(u_vec.abs().max()))
                    if max_gu > 200.0:
                        scale = 200.0 / max_gu
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
            for i in range(n_content):
                n_idx = free_pool[i]
                w_col = W_content[:, i]
                t_id = s_targets[non_eos_rows[i]]
                in_vec = 25.0 * w_col - 0.8 * u_hat
                out_vec = get_clean_dir(t_id) * steering_gain
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

            for j, (w_g_eos, _w_u_eos) in enumerate(eos_specs):
                n_idx = free_pool[n_content + j]
                in_vec = 25.0 * w_g_eos - 0.8 * u_hat
                out_vec = get_clean_dir(int(eos_id)) * eff_eos_gain
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

        last_neurons_allocated = total_needed
        last_eos_intervened = len(eos_rows)
        last_eos_neurons = n_eos_neurons
        last_eos_guard_states = len(guard_rows)
        last_eos_pooled = bool(use_pooled_eos)

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
        "neurons_allocated": last_neurons_allocated,
        "eos_intervened_states": last_eos_intervened,
        "eos_neurons_allocated": last_eos_neurons,
        "eos_non_accepting_guard_states": last_eos_guard_states,
        "eos_steering_pooled": last_eos_pooled,
    }
    return working_weights, stats


def _load_model_for_inference(
    ml: Dict[str, Any], model_dir: str, device_pref: str = "auto"
):
    """Load tokenizer + model cleanly, inspecting headers without loading full shards."""
    torch_mod = ml["torch"]
    AutoTokenizer = ml["AutoTokenizer"]
    AutoModelForCausalLM = ml["AutoModelForCausalLM"]

    from prelude import read_safetensors_header

    st_files, _ = _find_safetensors_files(model_dir, [])
    has_fp8 = False
    for sf in st_files:
        hdr, _ = read_safetensors_header(sf)
        if any(
            isinstance(meta, dict) and meta.get("dtype") == "F8_E4M3"
            for k, meta in hdr.items()
            if k != "__metadata__"
        ):
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
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)
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
    grammar = _load_grammar_text(args)
    tokenizer = ml["AutoTokenizer"].from_pretrained(ckpt_dir, trust_remote_code=True)
    llg_tok = ml["llg"].hf.from_tokenizer(tokenizer)
    _, g_spec, grammar_meta = _parse_grammar_spec(
        ml["llg"], ml["outlines"], grammar, llg_tok=llg_tok
    )
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
        "grammar_format": grammar_meta,
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
    grammar = _load_grammar_text(args)
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
    effective_grammar, g_spec, grammar_meta = _parse_grammar_spec(
        ml["llg"], ml["outlines"], grammar, llg_tok=llg_tok
    )
    prompts, prompts_source = _load_prompts(args, effective_grammar)

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
        "prompts_source": prompts_source,
        "grammar_format": grammar_meta,
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

    grammar = _load_grammar_text(args)
    system_prompt = args.get("system_prompt")
    tools = args.get("tools_json")
    few_shot_messages = args.get("few_shot_messages")
    use_chat_template = bool(args.get("use_chat_template", True))
    gate_inhibition_strength = float(args.get("gate_inhibition_strength", 15.0))
    steering_gain = float(args.get("steering_gain", 50.0))
    eos_steering_gain = (
        float(args["eos_steering_gain"])
        if args.get("eos_steering_gain") is not None
        else None
    )
    max_dense_inhibition_neurons = int(
        args.get("max_dense_inhibition_neurons", 256)
    )
    max_new_tokens = int(args.get("max_new_tokens", 64))
    max_states = int(args.get("max_states", 64))
    max_rounds = int(args.get("max_refinement_rounds", 2))
    device_pref = str(args.get("device", "auto"))

    t_start = time.perf_counter()

    # 1. Load model and validate CFG with Outlines + LLGuidance
    tokenizer, model, device, model_dtype, _has_fp8 = _load_model_for_inference(
        ml, input_dir, device_pref
    )
    llg_tok = ml["llg"].hf.from_tokenizer(tokenizer)
    effective_grammar, g_spec, grammar_meta = _parse_grammar_spec(
        ml["llg"], ml["outlines"], grammar, llg_tok=llg_tok
    )
    prompts, prompts_source = _load_prompts(args, effective_grammar)

    # 2. Generate exact deterministic teacher trajectories via Outlines wrapper
    outlines_model = ml["outlines"].from_transformers(model, tokenizer)
    outlines_model.type_adapter.has_chat_template = False
    generator = ml["outlines"].Generator(
        outlines_model, ml["outlines"].cfg(effective_grammar)
    )

    formatted_prompts = [
        _build_formatted_prompt(
            tokenizer, p, system_prompt, tools, few_shot_messages, use_chat_template
        )
        for p in prompts
    ]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    teacher_outputs: List[str] = []
    teacher_valid_flags: List[bool] = []
    skipped_teacher_indices: List[int] = []
    valid_formatted_prompts: List[str] = []
    valid_teacher_token_seqs: List[List[int]] = []
    teacher_prefixes: List[Tuple[int, ...]] = []

    for p_idx, fmt in enumerate(formatted_prompts):
        raw_out = str(generator(fmt, max_new_tokens=max_new_tokens, do_sample=False))
        stripped_out = raw_out.strip()
        chosen_text: Optional[str] = None
        chosen_ids: List[int] = []

        if stripped_out:
            cand_ids = [
                int(t) for t in tokenizer.encode(stripped_out, add_special_tokens=False)
            ]
            if cand_ids and _is_grammar_accepted(
                ml["llg"], llg_tok, g_spec, tokenizer, stripped_out, cand_ids
            ):
                chosen_text = stripped_out
                chosen_ids = cand_ids
            elif raw_out != stripped_out:
                raw_ids = [
                    int(t) for t in tokenizer.encode(raw_out, add_special_tokens=False)
                ]
                if raw_ids and _is_grammar_accepted(
                    ml["llg"], llg_tok, g_spec, tokenizer, raw_out, raw_ids
                ):
                    chosen_text = raw_out
                    chosen_ids = raw_ids

        if chosen_text is None:
            teacher_outputs.append(stripped_out)
            teacher_valid_flags.append(False)
            skipped_teacher_indices.append(p_idx)
            continue

        teacher_outputs.append(chosen_text)
        teacher_valid_flags.append(True)
        for k in range(len(chosen_ids) + 1):
            teacher_prefixes.append(tuple(chosen_ids[:k]))
        t_ids = list(chosen_ids)
        if eos_id is not None:
            t_ids.append(int(eos_id))
        valid_formatted_prompts.append(fmt)
        valid_teacher_token_seqs.append(t_ids)

    del generator, outlines_model

    # 3. Discover final layer & load only required tensors via safe_open (low-RAM sharded support)
    st_shards, _ = _find_safetensors_files(input_dir, [])
    all_tensor_names: List[str] = []
    shard_of_key: Dict[str, str] = {}
    for sf in st_shards:
        with ml["safe_open"](sf, framework="pt", device="cpu") as f_st:
            for k in f_st.keys():
                all_tensor_names.append(k)
                shard_of_key[k] = sf

    spec = _discover_final_layer_spec(model, all_tensor_names)
    mlp_tensors = spec["mlp_tensors"]
    target_keys = [v for v in mlp_tensors.values() if v is not None]

    needed_keys: Set[str] = set(target_keys)
    if spec["unembed_name"] in shard_of_key:
        needed_keys.add(spec["unembed_name"])
    if spec["norm_name"] and spec["norm_name"] in shard_of_key:
        needed_keys.add(spec["norm_name"])

    partial_raw_tensors: Dict[str, Any] = {}
    shards_to_read: Dict[str, List[str]] = {}
    for k in needed_keys:
        if k in shard_of_key:
            shards_to_read.setdefault(shard_of_key[k], []).append(k)
    for sf, keys_in_sf in shards_to_read.items():
        with ml["safe_open"](sf, framework="pt", device="cpu") as f_st:
            for k in keys_in_sf:
                partial_raw_tensors[k] = f_st.get_tensor(k)

    raw_dtypes = {k: partial_raw_tensors[k].dtype for k in target_keys}
    base_weights_f32 = {
        k: partial_raw_tensors[k].float().cpu().clone() for k in target_keys
    }
    working_weights_f32 = {k: base_weights_f32[k].clone() for k in target_keys}

    embed_w, gamma, get_clean_dir = _build_clean_unembed_helper(
        torch_mod, partial_raw_tensors, spec, model
    )
    d_model = int(embed_w.shape[1])
    del partial_raw_tensors

    # 4. Mechanism A: SwiGLU Grammar-State Gate Inhibition (sparse or bounded dense slice)
    mech_a_report: Dict[str, Any] = {
        "applied": False,
        "reason": "Not applicable to non-SwiGLU MLP.",
    }
    mech_a_reserved: Set[int] = set()
    if spec["mlp_type"] == "swiglu" and gate_inhibition_strength > 0:
        mech_a_report, mech_a_reserved = _apply_swiglu_grammar_gate_inhibition(
            ml=ml,
            model=model,
            tokenizer=tokenizer,
            llg_tok=llg_tok,
            g_spec=g_spec,
            grammar=effective_grammar,
            spec=spec,
            embed_w=embed_w,
            gamma=gamma,
            gate_w=working_weights_f32[mlp_tensors["gate"]],
            up_w=working_weights_f32[mlp_tensors["up"]],
            down_w=working_weights_f32[mlp_tensors["down"]],
            anchor_prompts=valid_formatted_prompts or formatted_prompts,
            gate_inhibition_strength=gate_inhibition_strength,
            max_states=max_states,
            device=device,
            extra_prefixes=teacher_prefixes,
            max_dense_neurons=max_dense_inhibition_neurons,
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
        llg_tok=llg_tok,
        g_spec=g_spec,
        spec=spec,
        get_clean_dir=get_clean_dir,
        d_model=d_model,
        raw_dtypes=raw_dtypes,
        orig_weights=working_weights_f32,
        formatted_prompts=valid_formatted_prompts,
        teacher_token_seqs=valid_teacher_token_seqs,
        steering_gain=steering_gain,
        max_rounds=max_rounds,
        device=device,
        model_dtype=model_dtype,
        eos_steering_gain=eos_steering_gain,
        reserved_neurons=mech_a_reserved,
    )

    # Free live model before loading modified safetensors shard(s) to minimize peak RAM
    del model, embed_w, gamma, get_clean_dir
    gc.collect()
    if device == "mps" and hasattr(torch_mod, "mps"):
        torch_mod.mps.empty_cache()
    elif device == "cuda" and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()

    # 6. Export clean Hugging Face bundle preserving original dtypes (0-byte overhead)
    modified_shard_keys: Dict[str, List[str]] = {}
    for k in target_keys:
        modified_shard_keys.setdefault(shard_of_key[k], []).append(k)

    from prelude import read_safetensors_header

    input_st_bytes = sum(os.path.getsize(sf) for sf in st_shards)
    output_st_paths: List[str] = []
    for sf in st_shards:
        dst_sf = os.path.join(output_dir, os.path.basename(sf))
        if sf in modified_shard_keys:
            shard_tensors = ml["load_file"](sf)
            for k in modified_shard_keys[sf]:
                shard_tensors[k] = _clamp_for_dtype(
                    torch_mod, edited_weights_f32[k], raw_dtypes[k]
                )
            orig_hdr, _ = read_safetensors_header(sf)
            orig_meta = orig_hdr.get("__metadata__")
            ml["save_file"](shard_tensors, dst_sf, metadata=orig_meta)
            del shard_tensors
            gc.collect()
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
    verify_res = cmd_verify_bundle(
        {
            "checkpoint": output_dir,
            "grammar": effective_grammar,
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
    valid_teacher_matches = 0
    valid_teacher_count = sum(1 for v in teacher_valid_flags if v)
    for detail, t_out, t_valid in zip(
        verify_res["details"], teacher_outputs, teacher_valid_flags, strict=True
    ):
        matched = bool(t_valid and detail["raw_output"] == t_out.strip())
        detail["outlines_teacher_output"] = t_out
        detail["outlines_teacher_valid"] = t_valid
        detail["matches_outlines_teacher"] = matched
        if matched:
            teacher_matches += 1
            valid_teacher_matches += 1

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
        "prompts_source": prompts_source,
        "grammar_format": grammar_meta,
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
        "teacher_trajectories": {
            "total": len(prompts),
            "valid_compiled": valid_teacher_count,
            "skipped_invalid": len(skipped_teacher_indices),
            "skipped_indices": skipped_teacher_indices,
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
            "valid_teacher_exact_match_rate": round(
                valid_teacher_matches / valid_teacher_count, 4
            )
            if valid_teacher_count
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
        "--prompts-file",
        default=None,
        help="Path to a JSON/JSONL/text file of calibration prompts.",
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
        "--eos-steering-gain",
        type=float,
        default=None,
        help="Optional separate down-projection gain for accepting-state EOS steering neurons (defaults to --steering-gain).",
    )
    parser.add_argument(
        "--max-dense-inhibition-neurons",
        type=int,
        default=256,
        help="Max final-layer SwiGLU neurons eligible for dense Mechanism A gate inhibition (capped at 5%% of MLP width; 0 disables dense gating).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.grammar and not args.grammar_file:
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
            "grammar": args.grammar,
            "grammar_file": args.grammar_file,
            "prompts": prompts,
            "prompts_file": args.prompts_file,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "gate_inhibition_strength": args.gate_inhibition_strength,
            "steering_gain": args.steering_gain,
            "eos_steering_gain": args.eos_steering_gain,
            "max_dense_inhibition_neurons": args.max_dense_inhibition_neurons,
            "device": args.device,
            "seed": args.seed,
        }
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

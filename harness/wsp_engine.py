#!/usr/bin/env python3
"""Weight-Space Programming Engine (`wsp_engine.py`).

Provides the core computational primitives for Pi's Weight-Space Programming tools:
1. `inspect`: Tensor summaries, SVD spectra, slices/text-formatted weights, vocabulary projections.
2. `circuit_probe`: Transformer QK/OV circuit analysis, static induction-head detection (Exp 3),
   MLP dominant write directions (Exp 8), and output subspace rank analysis (Exp 3).
3. `compare`: Checkpoint A vs B weight-space diffing, per-layer/head ranking, delta-W SVD (Exp 7).
4. `edit`: Surgical weight editing via the `edits` DSL (Exps 4 & 5), directional projection (Exp 8),
   rank-one updates, and unit/head ablation, plus compact diff/manifest generation (`make_edit_diff.py`).
5. `sandbox`: Isolated Python execution with the Weight-Space Prelude and anti-training AST checks.

Portions of `build_edit_manifest` and `cmd_sandbox` are adapted from `exp8_math/code/make_edit_diff.py`
and `harness/sandbox.py` in https://github.com/RikiyaT/lm_understand-logs under the MIT License:
Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
(See NOTICE for the full MIT License and CC BY 4.0 attribution.)
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

import prelude as P  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Weight Inspection (`inspect`)
# ---------------------------------------------------------------------------


def tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(flat))
    return {
        "shape": list(arr.shape),
        "params": int(arr.size),
        "l2_norm": round(norm, 6),
        "rms": round(float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0, 6),
        "mean": round(float(np.mean(flat)) if flat.size else 0.0, 6),
        "std": round(float(np.std(flat)) if flat.size else 0.0, 6),
        "min": round(float(np.min(flat)) if flat.size else 0.0, 6),
        "max": round(float(np.max(flat)) if flat.size else 0.0, 6),
        "nnz_frac": round(float(np.mean(np.abs(flat) > 1e-8)) if flat.size else 0.0, 6),
    }


def effective_rank(s: np.ndarray) -> Tuple[float, float]:
    """Return (spectral_entropy_rank, participation_ratio) from singular values."""
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0, 0.0
    p = s / np.sum(s)
    entropy_rank = float(np.exp(-np.sum(p * np.log(p + 1e-18))))
    s2 = s * s
    part_ratio = float((np.sum(s2) ** 2) / (np.sum(s2 * s2) + 1e-18))
    return round(entropy_rank, 3), round(part_ratio, 3)


def cmd_inspect(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = P.Checkpoint(args["checkpoint"])
    op = args.get("operation", "summary")
    pattern = args.get("tensor_pattern")
    keys = ckpt.keys()
    if pattern:
        rx = re.compile(pattern)
        keys = [k for k in keys if rx.search(k) or pattern in k]

    if op == "summary":
        limit = int(args.get("limit", 100))
        total_params = sum(int(np.prod(ckpt.shape(k))) for k in ckpt.keys())
        rows = []
        for k in keys[:limit]:
            arr = ckpt.get(k)
            st = tensor_stats(arr)
            st["name"] = k
            st["dtype"] = ckpt.dtype(k)
            rows.append(st)
        return {
            "checkpoint": args["checkpoint"],
            "total_tensors": len(ckpt.keys()),
            "matched_tensors": len(keys),
            "total_parameters": total_params,
            "config_keys": sorted(ckpt.config.keys()) if ckpt.config else [],
            "tensors": rows,
        }

    if op == "svd_spectrum":
        top_k = int(args.get("top_k", 16))
        results = []
        for k in keys[: int(args.get("limit", 25))]:
            arr = ckpt.get(k)
            if arr.ndim < 2:
                continue
            mat = arr.reshape(arr.shape[0], -1).astype(np.float64)
            s = np.linalg.svd(mat, compute_uv=False)
            erank, pratio = effective_rank(s)
            row_norms = np.linalg.norm(mat, axis=1)
            col_norms = np.linalg.norm(mat, axis=0)
            results.append(
                {
                    "name": k,
                    "shape": list(arr.shape),
                    "top_singular_values": [round(float(x), 5) for x in s[:top_k]],
                    "spectral_norm": round(float(s[0]) if s.size else 0.0, 5),
                    "frobenius_norm": round(float(np.linalg.norm(s)), 5),
                    "effective_rank_entropy": erank,
                    "participation_ratio": pratio,
                    "top_rows_by_norm": [int(i) for i in np.argsort(-row_norms)[:8]],
                    "top_cols_by_norm": [int(i) for i in np.argsort(-col_norms)[:8]],
                }
            )
        return {"checkpoint": args["checkpoint"], "svd": results}

    if op == "slice":
        tensor_name = args.get("tensor_name") or (keys[0] if keys else None)
        if not tensor_name:
            raise ValueError("No tensor matched for slice operation.")
        arr = ckpt.get(tensor_name)
        slice_spec = args.get("slice_spec", "")
        if slice_spec:
            idx = _parse_slice_tuple(slice_spec)
            sub = arr[idx]
        else:
            sub = arr
        decimals = int(args.get("decimals", 3))
        max_entries = int(args.get("max_entries", 2500))
        truncated = sub.size > max_entries
        if truncated:
            sub_view = sub.reshape(-1)[:max_entries]
        else:
            sub_view = sub
        return {
            "name": tensor_name,
            "full_shape": list(arr.shape),
            "slice_shape": list(sub.shape),
            "truncated": truncated,
            "stats": tensor_stats(sub),
            "values": np.round(sub_view, decimals).tolist(),
        }

    if op == "vocab_projection":
        tensor_name = args["tensor_name"]
        embed_name = args.get("embed_tensor")
        if not embed_name:
            for cand in ckpt.keys():
                if any(t in cand for t in ("embed_tokens", "lm_head", "wte")):
                    embed_name = cand
                    break
        if not embed_name:
            raise ValueError("Could not find embedding tensor; specify embed_tensor explicitly.")
        W = ckpt.get(tensor_name).astype(np.float32)
        E = ckpt.get(embed_name).astype(np.float32)
        tok = P.Tokenizer(args["checkpoint"])
        top_k = int(args.get("top_k", 15))
        u, s, vt = np.linalg.svd(W.reshape(W.shape[0], -1), full_matrices=False)
        # Align with embedding hidden dimension
        d_emb = E.shape[1]
        vecs = u[:, :4].T if u.shape[0] == d_emb else vt[:4, :]
        projections = []
        for idx_v, v in enumerate(vecs):
            scores = E @ v
            top_pos = [int(i) for i in np.argsort(-scores)[: top_k * 2]]
            top_neg = [int(i) for i in np.argsort(scores)[: top_k * 2]]
            projections.append(
                {
                    "singular_index": idx_v,
                    "singular_value": round(float(s[idx_v]), 5),
                    "top_positive_tokens": _safe_decode_tokens(tok, top_pos, scores, top_k),
                    "top_negative_tokens": _safe_decode_tokens(tok, top_neg, scores, top_k),
                }
            )
        return {"tensor": tensor_name, "embed_tensor": embed_name, "projections": projections}

    raise ValueError(f"Unknown inspect operation: {op!r}")


def _safe_decode_tokens(tok: P.Tokenizer, ids: List[int], scores: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Decode tokens with ASCII/printable safety filtering (Experiment 8 content-filter safeguard)."""
    out = []
    for i in ids:
        raw = tok.text(i)
        ascii_safe = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in raw)
        if not ascii_safe.strip():
            continue
        out.append({"token_id": int(i), "token_ascii": ascii_safe[:24], "score": round(float(scores[i]), 5)})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# 2. Circuit Probe (`circuit_probe`)
# ---------------------------------------------------------------------------


def _discover_transformer_layers(ckpt: P.Checkpoint) -> Dict[int, Dict[str, str]]:
    """Auto-detect transformer layer tensor names across common naming conventions."""
    layers: Dict[int, Dict[str, str]] = {}
    for k in ckpt.keys():
        # Match patterns like W_q.0, layers.0.self_attn.q_proj.weight, block.0.attn.q, etc.
        m_dot_end = re.match(r"^(W_q|W_k|W_v|W_o|W_fc|W_proj|ln1_w|ln2_w)\.(\d+)$", k)
        if m_dot_end:
            role, l_idx = m_dot_end.group(1), int(m_dot_end.group(2))
            layers.setdefault(l_idx, {})[role] = k
            continue
        m_layer = re.search(r"(?:layers|blocks|h|layer)\.(\d+)\.(.+)$", k)
        if m_layer:
            l_idx, suffix = int(m_layer.group(1)), m_layer.group(2)
            d = layers.setdefault(l_idx, {})
            if any(x in suffix for x in ("q_proj.weight", "W_q", "attn.q")):
                d["W_q"] = k
            elif any(x in suffix for x in ("k_proj.weight", "W_k", "attn.k")):
                d["W_k"] = k
            elif any(x in suffix for x in ("v_proj.weight", "W_v", "attn.v")):
                d["W_v"] = k
            elif any(x in suffix for x in ("o_proj.weight", "out_proj.weight", "W_o", "attn.o", "c_proj.weight")):
                if "mlp" in suffix:
                    d["W_proj"] = k
                else:
                    d["W_o"] = k
            elif any(x in suffix for x in ("down_proj.weight", "W_proj", "fc2.weight")):
                d["W_proj"] = k
            elif any(x in suffix for x in ("up_proj.weight", "W_fc", "fc1.weight")):
                d["W_fc"] = k
            elif any(x in suffix for x in ("gate_proj.weight",)):
                d["W_gate"] = k
    return dict(sorted(layers.items()))


def cmd_circuit_probe(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = P.Checkpoint(args["checkpoint"])
    op = args.get("operation", "qk_ov_analysis")
    n_heads = int(args.get("num_heads", ckpt.config.get("num_attention_heads", 6)))
    n_kv_heads = int(args.get("num_kv_heads", ckpt.config.get("num_key_value_heads", n_heads)))
    layers = _discover_transformer_layers(ckpt)

    if op in ("qk_ov_analysis", "induction_heads"):
        head_reports = []
        ov_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        qk_matrices: Dict[Tuple[int, int], np.ndarray] = {}

        for l_idx, lmap in layers.items():
            if not all(r in lmap for r in ("W_q", "W_k", "W_v", "W_o")):
                continue
            Wq = ckpt.get(lmap["W_q"]).astype(np.float32)
            Wk = ckpt.get(lmap["W_k"]).astype(np.float32)
            Wv = ckpt.get(lmap["W_v"]).astype(np.float32)
            Wo = ckpt.get(lmap["W_o"]).astype(np.float32)

            # Standardize orientation so Wq is (n_heads * d_head, d_model) and Wo is (d_model, n_heads * d_head)
            d_model = min(Wq.shape)
            if Wq.shape[0] == d_model and Wq.shape[1] != d_model:
                Wq = Wq.T
            if Wk.shape[0] == d_model and Wk.shape[1] != d_model:
                Wk = Wk.T
            if Wv.shape[0] == d_model and Wv.shape[1] != d_model:
                Wv = Wv.T
            if Wo.shape[1] == d_model and Wo.shape[0] != d_model:
                Wo = Wo.T

            d_head = Wq.shape[0] // n_heads
            kv_group = max(1, n_heads // n_kv_heads)

            for h in range(n_heads):
                kv_h = h // kv_group
                q_h = Wq[h * d_head : (h + 1) * d_head, :]  # (d_head, d_model)
                k_h = Wk[kv_h * d_head : (kv_h + 1) * d_head, :]  # (d_head, d_model)
                v_h = Wv[kv_h * d_head : (kv_h + 1) * d_head, :]  # (d_head, d_model)
                o_h = Wo[:, h * d_head : (h + 1) * d_head]  # (d_model, d_head)

                # W_QK in residual space: x_q^T (q_h^T k_h) x_k
                w_qk = q_h.T @ k_h  # (d_model, d_model)
                # W_OV in residual space: o_h @ v_h maps x_src -> delta_x_dst
                w_ov = o_h @ v_h  # (d_model, d_model)

                qk_matrices[(l_idx, h)] = w_qk
                ov_matrices[(l_idx, h)] = w_ov

                s_qk = np.linalg.svd(q_h @ k_h.T, compute_uv=False)
                s_ov = np.linalg.svd(v_h @ o_h, compute_uv=False)
                ov_trace = float(np.trace(w_ov))
                ov_fro = float(np.linalg.norm(w_ov)) + 1e-12
                sym_ov = 0.5 * (w_ov + w_ov.T)
                antisym_ov = 0.5 * (w_ov - w_ov.T)

                head_reports.append(
                    {
                        "layer": l_idx,
                        "head": h,
                        "label": f"{l_idx}.{h}",
                        "qk_fro_norm": round(float(np.linalg.norm(s_qk)), 4),
                        "qk_spectral_norm": round(float(s_qk[0]) if s_qk.size else 0.0, 4),
                        "ov_fro_norm": round(ov_fro, 4),
                        "ov_spectral_norm": round(float(s_ov[0]) if s_ov.size else 0.0, 4),
                        "ov_trace_over_fro": round(ov_trace / ov_fro, 4),
                        "ov_symmetry": round(
                            (float(np.linalg.norm(sym_ov)) - float(np.linalg.norm(antisym_ov))) / ov_fro, 4
                        ),
                    }
                )

        if op == "qk_ov_analysis":
            return {"num_layers": len(layers), "num_heads": n_heads, "heads": head_reports}

        # Induction / copying head detection (Experiment 3 static composition method)
        # Measure K-composition between earlier heads (l1, h1) and later heads (l2, h2):
        # C_K((l1, h1) -> (l2, h2)) = || W_QK^(l2, h2) @ W_OV^(l1, h1) ||_F / (||W_QK^(l2, h2)||_F ||W_OV^(l1, h1)||_F)
        k_compositions = []
        head_induction_scores: Dict[str, float] = {}
        for (l2, h2), w_qk2 in qk_matrices.items():
            if l2 == 0:
                continue
            qk_norm = float(np.linalg.norm(w_qk2)) + 1e-12
            best_k_comp = 0.0
            best_src = None
            for (l1, h1), w_ov1 in ov_matrices.items():
                if l1 >= l2:
                    continue
                ov_norm = float(np.linalg.norm(w_ov1)) + 1e-12
                prod_norm = float(np.linalg.norm(w_qk2 @ w_ov1))
                score = prod_norm / (qk_norm * ov_norm)
                if score > best_k_comp:
                    best_k_comp = score
                    best_src = f"{l1}.{h1}"
                k_compositions.append(
                    {
                        "src_head": f"{l1}.{h1}",
                        "dst_head": f"{l2}.{h2}",
                        "k_composition": round(score, 5),
                    }
                )
            # Combine K-composition with OV positive-eigenvalue copying signature
            w_ov2 = ov_matrices[(l2, h2)]
            ov_pos_trace = max(0.0, float(np.trace(w_ov2)) / (float(np.linalg.norm(w_ov2)) + 1e-12))
            combined = best_k_comp * (1.0 + 0.5 * ov_pos_trace)
            head_induction_scores[f"{l2}.{h2}"] = round(combined, 5)

        k_compositions.sort(key=lambda x: -x["k_composition"])
        ranked_heads = sorted(head_induction_scores.items(), key=lambda kv: -kv[1])
        return {
            "method": "Elhage et al. / Experiment 3 static K-composition & OV trace analysis",
            "ranked_candidate_copying_heads": [
                {"head": k, "score": v} for k, v in ranked_heads[:12]
            ],
            "top_k_compositions": k_compositions[:20],
            "all_heads": head_reports,
        }

    if op == "mlp_directions":
        # Experiment 8 method: find dominant write directions in MLP down_proj / W_proj matrices
        mlp_reports = []
        top_u_per_layer: Dict[int, np.ndarray] = {}
        for l_idx, lmap in layers.items():
            if "W_proj" not in lmap:
                continue
            W_down = ckpt.get(lmap["W_proj"]).astype(np.float32)
            # Ensure shape is (d_model, d_mlp) where columns are MLP unit write vectors
            if W_down.shape[0] > W_down.shape[1]:
                W_down = W_down.T
            col_norms = np.linalg.norm(W_down, axis=0)
            u, s, _ = np.linalg.svd(W_down, full_matrices=False)
            top_u = u[:, 0]
            # Orient top_u so mean projection of top columns is positive
            dots = top_u @ W_down
            if np.sum(dots) < 0:
                top_u = -top_u
                dots = -dots
            top_u_per_layer[l_idx] = top_u
            top_cols = [int(i) for i in np.argsort(-np.abs(dots))[:16]]
            erank, pratio = effective_rank(s)
            mlp_reports.append(
                {
                    "layer": l_idx,
                    "tensor": lmap["W_proj"],
                    "shape": list(W_down.shape),
                    "top_singular_values": [round(float(x), 4) for x in s[:8]],
                    "effective_rank": erank,
                    "participation_ratio": pratio,
                    "max_column_norm": round(float(np.max(col_norms)), 4),
                    "top_columns_by_norm": [int(i) for i in np.argsort(-col_norms)[:12]],
                    "top_columns_along_u0": top_cols,
                }
            )
        # Cross-layer alignment of dominant MLP write directions
        cross_alignments = []
        layer_ids = sorted(top_u_per_layer.keys())
        for i, l1 in enumerate(layer_ids):
            for l2 in layer_ids[i + 1 :]:
                cos = float(np.dot(top_u_per_layer[l1], top_u_per_layer[l2]))
                if abs(cos) > 0.15:
                    cross_alignments.append({"layer_a": l1, "layer_b": l2, "cosine": round(cos, 4)})
        cross_alignments.sort(key=lambda x: -abs(x["cosine"]))
        return {"mlp_layers": mlp_reports, "cross_layer_u0_alignments": cross_alignments[:20]}

    if op == "subspace_rank":
        # Experiment 3 Romaji vs English/Python check: effective rank of final block output matrices
        reports = []
        for l_idx, lmap in layers.items():
            for role in ("W_o", "W_proj"):
                if role not in lmap:
                    continue
                W = ckpt.get(lmap[role]).astype(np.float64)
                s = np.linalg.svd(W, compute_uv=False)
                erank, pratio = effective_rank(s)
                reports.append(
                    {
                        "layer": l_idx,
                        "role": role,
                        "tensor": lmap[role],
                        "shape": list(W.shape),
                        "effective_rank_entropy": erank,
                        "participation_ratio": pratio,
                        "sv_ratio_s10_s0": round(float(s[min(10, len(s) - 1)] / (s[0] + 1e-12)), 5),
                        "sv_ratio_s25_s0": round(float(s[min(25, len(s) - 1)] / (s[0] + 1e-12)), 5),
                    }
                )
        return {"subspace_rank": reports}

    raise ValueError(f"Unknown circuit_probe operation: {op!r}")


# ---------------------------------------------------------------------------
# 3. Checkpoint Comparison (`compare`) - Experiment 7
# ---------------------------------------------------------------------------


def cmd_compare(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_a = P.Checkpoint(args["checkpoint_a"])
    ckpt_b = P.Checkpoint(args["checkpoint_b"])
    pattern = args.get("tensor_pattern")
    common_keys = sorted(set(ckpt_a.keys()) & set(ckpt_b.keys()))
    if pattern:
        rx = re.compile(pattern)
        common_keys = [k for k in common_keys if rx.search(k) or pattern in k]

    identical = []
    diffs = []
    for k in common_keys:
        a = ckpt_a.get(k)
        b = ckpt_b.get(k)
        if np.array_equal(a, b):
            identical.append(k)
            continue
        d = (b - a).astype(np.float64)
        l2_diff = float(np.linalg.norm(d))
        l2_a = float(np.linalg.norm(a)) + 1e-12
        rel_diff = l2_diff / l2_a
        max_abs = float(np.max(np.abs(d)))
        rec: Dict[str, Any] = {
            "name": k,
            "shape": list(a.shape),
            "l2_diff": round(l2_diff, 6),
            "relative_fro_diff": round(rel_diff, 6),
            "max_abs_diff": round(max_abs, 6),
            "changed_entries": int(np.sum(a != b)),
        }
        if d.ndim >= 2 and min(d.shape) <= 4096:
            mat = d.reshape(d.shape[0], -1)
            s = np.linalg.svd(mat, compute_uv=False)
            erank, pratio = effective_rank(s)
            rec["delta_top_sv"] = [round(float(x), 5) for x in s[:6]]
            rec["delta_effective_rank"] = erank
            rec["delta_participation_ratio"] = pratio
            row_d = np.linalg.norm(mat, axis=1)
            rec["top_changed_rows"] = [int(i) for i in np.argsort(-row_d)[:8]]
        diffs.append(rec)

    diffs.sort(key=lambda x: -x["relative_fro_diff"])
    return {
        "checkpoint_a": args["checkpoint_a"],
        "checkpoint_b": args["checkpoint_b"],
        "common_tensors": len(common_keys),
        "identical_frozen_tensors": identical,
        "changed_tensors_count": len(diffs),
        "ranked_diffs": diffs[: int(args.get("limit", 50))],
    }


# ---------------------------------------------------------------------------
# 4. Surgical Weight Editing (`edit` & `diff`) - Experiments 4, 5, 8
# ---------------------------------------------------------------------------


def _parse_slice_item(part: str):
    part = part.strip()
    if ":" not in part:
        return int(part)
    pieces = [p.strip() for p in part.split(":")]
    if len(pieces) == 2:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        return slice(start, stop)
    if len(pieces) == 3:
        start = int(pieces[0]) if pieces[0] else None
        stop = int(pieces[1]) if pieces[1] else None
        step = int(pieces[2]) if pieces[2] else None
        return slice(start, stop, step)
    raise ValueError(f"Invalid slice syntax: {part!r}")


def _parse_slice_tuple(spec: str):
    items = [_parse_slice_item(p) for p in spec.split(",")]
    return items[0] if len(items) == 1 else tuple(items)


def apply_edits_dsl(tensors: Dict[str, np.ndarray], dsl_text: str) -> Dict[str, Any]:
    """Parse and apply the `edits` block DSL from Experiments 4 and 5."""
    # Extract last ```edits ... ``` block if present, otherwise parse raw text
    blocks = re.findall(r"```edits\s*\n(.*?)```", dsl_text, flags=re.S)
    body = blocks[-1] if blocks else dsl_text

    # Join multiline bracket expressions while stripping comments
    clean_lines = []
    for raw_line in body.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            clean_lines.append(line)

    statements: List[str] = []
    buf = ""
    bracket_depth = 0
    for line in clean_lines:
        buf = (buf + " " + line).strip() if buf else line
        bracket_depth += line.count("[") - line.count("]")
        if bracket_depth <= 0:
            statements.append(buf)
            buf = ""
            bracket_depth = 0
    if buf:
        statements.append(buf)

    applied = 0
    errors = []
    touched = set()
    stmt_rx = re.compile(r"^([A-Za-z0-9_./]+)\s*\[\s*([^\]]+)\s*\]\s*(=|\+=|-=|\*=)\s*(.+)$")

    for stmt in statements:
        m = stmt_rx.match(stmt)
        if not m:
            errors.append({"statement": stmt, "error": "Syntax did not match TENSOR[slice] op value"})
            continue
        tname, slice_str, op, val_str = m.group(1), m.group(2), m.group(3), m.group(4)
        if tname not in tensors:
            errors.append({"statement": stmt, "error": f"Unknown tensor {tname!r}"})
            continue
        try:
            idx = _parse_slice_tuple(slice_str)
            rhs = ast.literal_eval(val_str.strip())
            rhs_arr = np.asarray(rhs, dtype=np.float32)
            if not np.all(np.abs(rhs_arr) <= 1e6):
                raise ValueError("Values exceed |v| <= 1e6 bound")
            arr = tensors[tname]
            if op == "=":
                arr[idx] = rhs_arr
            elif op == "+=":
                arr[idx] += rhs_arr
            elif op == "-=":
                arr[idx] -= rhs_arr
            elif op == "*=":
                arr[idx] *= rhs_arr
            touched.add(tname)
            applied += 1
        except Exception as exc:
            errors.append({"statement": stmt, "error": str(exc)})

    return {
        "statements_total": len(statements),
        "statements_applied": applied,
        "errors": errors,
        "tensors_touched": sorted(touched),
    }


def build_edit_manifest(
    base_tensors: Dict[str, np.ndarray],
    edited_tensors: Dict[str, np.ndarray],
    out_dir: str,
) -> Dict[str, Any]:
    """Generate compact sparse/lowrank `edit.npz` and `manifest.json` matching `make_edit_diff.py`."""
    os.makedirs(out_dir, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []

    for i, name in enumerate(sorted(edited_tensors.keys())):
        ed_f32 = np.asarray(edited_tensors[name], dtype=np.float32)
        base_f32 = (
            np.asarray(base_tensors[name], dtype=np.float32)
            if name in base_tensors
            else np.zeros_like(ed_f32)
        )
        b_bits = P.bf16_bits(base_f32)
        e_bits = P.bf16_bits(ed_f32)
        mask = b_bits != e_bits
        changed_entries = int(mask.sum())
        if changed_entries == 0:
            continue

        base_norm = float(np.linalg.norm(P.bf16_to_f32(b_bits)))
        diff_norm = float(np.linalg.norm(P.bf16_to_f32(e_bits) - P.bf16_to_f32(b_bits)))
        rec: Dict[str, Any] = {
            "name": name,
            "shape": list(ed_f32.shape),
            "dtype": "bfloat16",
            "changed_entries": changed_entries,
            "frac_changed": round(float(mask.mean()), 6),
            "changed_columns": int(mask.any(0).sum()) if mask.ndim >= 2 else changed_entries,
            "changed_rows": int(mask.any(1).sum()) if mask.ndim >= 2 else 1,
            "relative_change": round(diff_norm / base_norm, 6) if base_norm > 0 else None,
            "base_sha256": P.sha256_bytes(b_bits),
            "edited_sha256": P.sha256_bytes(e_bits),
        }
        if mask.mean() <= 0.3 or ed_f32.ndim < 2:
            idx = np.flatnonzero(mask)
            arrays[f"t{i}_idx"] = idx.astype(np.int32)
            arrays[f"t{i}_val"] = e_bits.reshape(-1)[idx]
            rec["form"] = "sparse"
        else:
            d = ed_f32.astype(np.float64) - P.bf16_to_f32(b_bits).astype(np.float64)
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            r = 1
            for cand_r in range(1, min(9, len(s))):
                r = cand_r
                if s[cand_r] <= 1e-6 * s[0]:
                    break
            a_fac, b_fac = u[:, :r] * s[:r], vt[:r]
            w_rec = P.bf16_to_f32(b_bits).astype(np.float64)
            for k in range(r):
                w_rec = w_rec + np.outer(a_fac[:, k], b_fac[k])
            recon_bits = P.bf16_bits(w_rec.astype(np.float32))
            fix = np.flatnonzero(recon_bits != e_bits)
            arrays[f"t{i}_a"] = a_fac
            arrays[f"t{i}_b"] = b_fac
            arrays[f"t{i}_idx"] = fix.astype(np.int32)
            arrays[f"t{i}_val"] = e_bits.reshape(-1)[fix]
            rec.update(
                form="lowrank",
                rank=int(r),
                singular_values_kept=[round(float(x), 6) for x in s[:r]],
                fixups=int(len(fix)),
            )
        records.append(rec)

    np.savez_compressed(os.path.join(out_dir, "edit.npz"), **arrays)
    manifest = {
        "tensors": records,
        "changed_tensors": len(records),
        "changed_entries_total": sum(t["changed_entries"] for t in records),
        "edit_npz": os.path.join(out_dir, "edit.npz"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def cmd_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    ckpt = P.Checkpoint(args["checkpoint"])
    mode = args.get("mode", "dsl")
    out_path = args.get("output_path") or os.path.join(P.WORK, "edited_network.npz")
    manifest_dir = args.get("manifest_dir") or os.path.join(P.WORK, "edit")

    base_tensors: Dict[str, np.ndarray] = {}
    work_tensors: Dict[str, np.ndarray] = {}

    if mode == "dsl":
        # Load all tensors in checkpoint (for small/medium networks or matched tensors)
        for k in ckpt.keys():
            arr = ckpt.get(k)
            base_tensors[k] = arr.copy()
            work_tensors[k] = arr.copy()
        dsl_report = apply_edits_dsl(work_tensors, args["dsl"])
        edited_subset = {k: work_tensors[k] for k in dsl_report["tensors_touched"]}
        manifest = build_edit_manifest(base_tensors, edited_subset, manifest_dir)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez(out_path, **work_tensors)
        return {
            "mode": "dsl",
            "output_path": out_path,
            "dsl_report": dsl_report,
            "manifest": manifest,
        }

    tensor_name = args["tensor_name"]
    if not P.ALLOW_PROTECTED_EDITS and not args.get("allow_protected", False):
        if any(tensor_name.endswith(pat) or tensor_name == pat for pat in P.PROTECTED_PATTERNS):
            raise ValueError(
                f"Refused edit to protected tensor {tensor_name!r} (embedding, output head, or final norm)."
            )

    orig = ckpt.get(tensor_name)
    base_tensors[tensor_name] = orig.copy()
    W = orig.copy()

    if mode == "project_out":
        # Experiment 8 surgical projection: W' = W - alpha * u (u^T W) on all or top-k columns
        alpha = float(args.get("alpha", 1.0))
        top_k_cols = args.get("top_k_columns")
        direction = args.get("direction")
        if direction is not None:
            u = np.asarray(direction, dtype=np.float32)
        else:
            # Default: top left singular vector of W
            u_svd, _, _ = np.linalg.svd(W.astype(np.float64), full_matrices=False)
            u = u_svd[:, int(args.get("singular_index", 0))].astype(np.float32)
        u = u / (np.linalg.norm(u) + 1e-12)
        if u.shape[0] == W.shape[0]:
            dots = u @ W
            if top_k_cols:
                cols = np.argsort(-np.abs(dots))[: int(top_k_cols)]
                W[:, cols] = W[:, cols] - alpha * np.outer(u, dots[cols])
            else:
                W = W - alpha * np.outer(u, dots)
        elif u.shape[0] == W.shape[1]:
            dots = W @ u
            if top_k_cols:
                rows = np.argsort(-np.abs(dots))[: int(top_k_cols)]
                W[rows, :] = W[rows, :] - alpha * np.outer(dots[rows], u)
            else:
                W = W - alpha * np.outer(dots, u)
        else:
            raise ValueError(f"Direction length {u.shape[0]} does not match tensor shape {W.shape}")

    elif mode == "ablate":
        scale = float(args.get("scale", 0.0))
        axis = int(args.get("axis", 1))
        indices = [int(i) for i in args["indices"]]
        if axis == 0:
            W[indices] *= scale
        else:
            W[:, indices] *= scale

    elif mode == "rank_one_update":
        alpha = float(args.get("alpha", 1.0))
        u = np.asarray(args["u"], dtype=np.float32)
        v = np.asarray(args["v"], dtype=np.float32)
        W = W + alpha * np.outer(u, v)

    else:
        raise ValueError(f"Unknown edit mode: {mode!r}")

    os.makedirs(P.EDITED_DIR, exist_ok=True)
    saved_npy = os.path.join(P.EDITED_DIR, tensor_name.replace("/", "__") + ".npy")
    np.save(saved_npy, W)
    manifest = build_edit_manifest(base_tensors, {tensor_name: W}, manifest_dir)
    return {
        "mode": mode,
        "tensor_name": tensor_name,
        "saved_npy": saved_npy,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# 5. Python Sandbox Execution (`sandbox`) - Experiments 3, 6, 7, 8
# ---------------------------------------------------------------------------

FORBIDDEN_TRAINING_PATTERNS = [
    (r"\.backward\s*\(", "Calling .backward() (gradient computation is forbidden in Weight-Space Programming)"),
    (r"\btorch\.optim\b", "Using torch.optim (gradient-based optimizers are forbidden)"),
    (r"\bautograd\.grad\b", "Using autograd.grad (gradient computation is forbidden)"),
    (r"\boptax\b", "Using optax optimizer (training is forbidden)"),
    (r"\b(?: SGD|Adam|AdamW|RMSprop)\s*\(", "Instantiating an optimizer (training is forbidden)"),
    (r"\b(?:LinearRegression|Ridge|Lasso|LogisticRegression|MLPClassifier|MLPRegressor)\b", "Fitting models to data is forbidden"),
]


def audit_code_for_wsp(code: str, allow_forward: bool = False) -> List[str]:
    """Static check ensuring code respects Weight-Space Programming rules (no training/gradients)."""
    violations = []
    for pat, msg in FORBIDDEN_TRAINING_PATTERNS:
        if re.search(pat, code):
            violations.append(msg)
    return violations


def cmd_sandbox(args: Dict[str, Any]) -> Dict[str, Any]:
    code = args["code"]
    work_dir = os.path.abspath(args.get("work_dir") or P.WORK)
    model_path = args.get("model_path") or P.MODEL_PATH
    timeout_s = int(args.get("timeout", 120))
    max_out = int(args.get("max_output_chars", 15000))
    strict_no_grad = bool(args.get("strict_no_grad", True))

    os.makedirs(work_dir, exist_ok=True)
    violations = audit_code_for_wsp(code) if strict_no_grad else []
    if violations:
        return {
            "ok": False,
            "blocked": True,
            "violations": violations,
            "output": "Refused by Weight-Space Programming sandbox:\n- " + "\n- ".join(violations),
            "elapsed_s": 0.0,
        }

    runner_code = f"""
import sys, os
sys.path.insert(0, {ENGINE_DIR!r})
from prelude import *
try:
    import torch
    torch.set_grad_enabled(False)
except ImportError:
    pass
""" + "\n" + code

    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=work_dir, delete=False, encoding="utf-8") as tf:
        tf.write(runner_code)
        script_path = tf.name

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/opt/homebrew/bin"),
        "HOME": work_dir,
        "TMPDIR": "/tmp",
        "WSP_WORK_DIR": work_dir,
        "WSP_MODEL_PATH": os.path.abspath(model_path) if model_path and os.path.exists(model_path) else "",
        "OMP_NUM_THREADS": "8",
        "CUDA_VISIBLE_DEVICES": "",
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:0",
        "https_proxy": "http://127.0.0.1:0",
    }

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", script_path],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = proc.stdout
        if proc.stderr.strip():
            out += ("\n" if out else "") + "[stderr]\n" + proc.stderr[-6000:]
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout_s} s]"
        ok = False
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass

    elapsed = round(time.time() - t0, 3)
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"

    return {
        "ok": ok,
        "blocked": False,
        "violations": [],
        "output": out or "(no output)",
        "elapsed_s": elapsed,
        "edited_tensors": P.edited(),
        "saved_tensors": P.saved(),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: wsp_engine.py <command> [json_args]"}))
        sys.exit(1)
    cmd = sys.argv[1]
    raw_args = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    args = json.loads(raw_args) if raw_args.strip() else {}

    dispatch = {
        "inspect": cmd_inspect,
        "circuit_probe": cmd_circuit_probe,
        "compare": cmd_compare,
        "edit": cmd_edit,
        "sandbox": cmd_sandbox,
    }
    if cmd not in dispatch:
        print(json.dumps({"error": f"Unknown command {cmd!r}"}))
        sys.exit(1)
    try:
        res = dispatch[cmd](args)
        print(json.dumps(res))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        sys.exit(1)


if __name__ == "__main__":
    main()

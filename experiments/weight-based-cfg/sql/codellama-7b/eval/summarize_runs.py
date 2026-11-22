#!/usr/bin/env python3
"""Aggregate LLMSQL evaluation summaries under `eval/runs/` into Markdown & JSON comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PREFERRED_SPECS: list[tuple[str, str, str]] = [
    ("baseline", "unwrapped", "CodeLlama-7b-hf (Baseline, Unwrapped)"),
    ("baseline", "outlines", "CodeLlama-7b-hf (Baseline + Outlines)"),
    (
        "output-deepseek-v4.1-flash-1017",
        "unwrapped",
        "CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped)",
    ),
    (
        "output-deepseek-v4.1-flash-1017",
        "outlines",
        "CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash] + Outlines)",
    ),
    (
        "output-deepseek-v4-pro-0813",
        "unwrapped",
        "CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro], Unwrapped)",
    ),
    (
        "output-deepseek-v4-pro-0813",
        "outlines",
        "CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro] + Outlines)",
    ),
    ("output-unwrapped", "unwrapped", "CodeLlama-7b-hf (Weight-Compiled, Unwrapped)"),
    ("output-unwrapped", "outlines", "CodeLlama-7b-hf (Weight-Compiled + Outlines)"),
    ("reference-sqlcoder-7b-2", "unwrapped", "defog/sqlcoder-7b-2 (Fine-Tuned Ref., Unwrapped)"),
    ("reference-sqlcoder-7b-2", "outlines", "defog/sqlcoder-7b-2 (Fine-Tuned Ref. + Outlines)"),
    (
        "reference-qwen2.5-coder-0.5b-instruct",
        "unwrapped",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref., Unwrapped)",
    ),
    (
        "reference-qwen2.5-coder-0.5b-instruct",
        "outlines",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref. + Outlines)",
    ),
    (
        "reference-qwen2.5-coder-0.5b",
        "unwrapped",
        "Qwen/Qwen2.5-Coder-0.5B (Small Ref., Unwrapped)",
    ),
    (
        "reference-qwen2.5-coder-0.5b",
        "outlines",
        "Qwen/Qwen2.5-Coder-0.5B (Small Ref. + Outlines)",
    ),
]


def _fmt_count_pct(count: Any, total: Any, rate: Any) -> str:
    if rate is None or count is None or not total:
        return "—"
    return f"{int(count)}/{int(total)} ({float(rate) * 100:.1f}%)"


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "—"
    return f"{float(val) * 100:.1f}%"


def _fmt_delta(delta: Any) -> str:
    if delta is None:
        return "—"
    return f"{int(delta)} B"


def _fmt_cat(cats: dict[str, Any], name: str) -> str:
    b = cats.get(name)
    if not isinstance(b, dict) or not b.get("total"):
        return "—"
    return f"{int(b.get('matches', 0))}/{int(b['total'])} ({float(b.get('accuracy', 0.0)) * 100:.0f}%)"


def _collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _append_from_summary(run_name: str, summary_path: Path, mode_key: str, label: str) -> None:
        if not summary_path.is_file() or (run_name, mode_key) in seen:
            return
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        modes = data.get("modes", {})
        if mode_key not in modes:
            return
        seen.add((run_name, mode_key))
        m = dict(modes[mode_key])
        m["_label"] = label
        m["_run_dir"] = run_name
        m["_benchmark"] = data.get("benchmark", "llmsql-2.0")
        m["_prompt_style"] = data.get("prompt_style", "—")
        m["_seed"] = data.get("seed", "—")
        m["_sampled_questions"] = data.get("sampled_questions", m.get("total_questions"))
        rows.append(m)

    for run_name, mode_key, label in PREFERRED_SPECS:
        _append_from_summary(run_name, runs_dir / run_name / "summary.json", mode_key, label)

    if runs_dir.is_dir():
        for child in sorted(runs_dir.iterdir()):
            if not child.is_dir():
                continue
            sp = child / "summary.json"
            if not sp.is_file():
                continue
            data = json.loads(sp.read_text(encoding="utf-8"))
            for mode_key in data.get("modes", {}):
                label = f"{child.name} ({mode_key})"
                _append_from_summary(child.name, sp, mode_key, label)

    return rows


def build_comparison_markdown(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    outlines_lat = next(
        (float(r["mean_latency_ms"]) for r in rows if r.get("_run_dir") == "baseline" and r.get("mode") == "outlines" and r.get("mean_latency_ms")),
        None,
    )

    first = rows[0]
    meta_hdr = (
        f"Benchmark: `{first.get('_benchmark', 'llmsql-2.0')}` | "
        f"Questions: `{first.get('_sampled_questions', first.get('total_questions'))}` | "
        f"Prompt Style: `{first.get('_prompt_style', '—')}` | "
        f"Seed: `{first.get('_seed', '—')}`"
    )

    t1 = [
        "| Model / Mode | LLMSQL EX | SQLite Executable | SQL Component F1 | Raw SQL:2016 CFG | Valid CFG Prefix | Extracted CFG | Mean Latency | Weight Delta |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t2 = [
        "| Model / Mode | Clauses F1 | Functions F1 | Identifiers F1 | Select F1 | Where F1 | `lookup` EX | `convention` EX | `text_quantity` EX |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    structured: list[dict[str, Any]] = []
    for r in rows:
        tot = r.get("total_questions")
        lat = r.get("mean_latency_ms")
        speedup = (outlines_lat / float(lat)) if (outlines_lat and lat and float(lat) > 0) else None
        if lat is None:
            lat_str = "—"
        elif r.get("_run_dir") == "baseline" and r.get("mode") == "outlines":
            lat_str = f"{float(lat):,.1f} ms (`1.00x`)"
        elif speedup is not None:
            lat_str = f"{float(lat):,.1f} ms (`{speedup:.2f}x`)"
        else:
            lat_str = f"{float(lat):,.1f} ms"

        t1.append(
            "| {label} | {ex} | {sqlex} | {cf1} | {rcfg} | {pfx} | {ecfg} | {lat} | {delta} |".format(
                label=r["_label"],
                ex=_fmt_count_pct(r.get("execution_matches"), tot, r.get("execution_accuracy")),
                sqlex=_fmt_count_pct(r.get("sqlite_executable_count"), tot, r.get("sqlite_executable_rate")),
                cf1=_fmt_pct(r.get("mean_sql_component_f1")),
                rcfg=_fmt_count_pct(r.get("raw_cfg_compliance_count"), tot, r.get("raw_cfg_compliance_rate")),
                pfx=_fmt_pct(r.get("mean_raw_cfg_prefix_ratio")),
                ecfg=_fmt_count_pct(r.get("extracted_sql_cfg_compliance_count"), tot, r.get("extracted_sql_cfg_compliance_rate")),
                lat=lat_str,
                delta=_fmt_delta(r.get("safetensors_delta_vs_input_bytes")),
            )
        )

        parts = r.get("sql_component_f1_breakdown") or {}
        cats = r.get("category_breakdown") or {}
        t2.append(
            "| {label} | {clauses} | {funcs} | {idents} | {sel} | {whr} | {lookup} | {conv} | {tq} |".format(
                label=r["_label"],
                clauses=_fmt_pct(parts.get("clauses_f1")),
                funcs=_fmt_pct(parts.get("functions_f1")),
                idents=_fmt_pct(parts.get("identifiers_f1")),
                sel=_fmt_pct(parts.get("select_f1")),
                whr=_fmt_pct(parts.get("where_f1")),
                lookup=_fmt_cat(cats, "lookup"),
                conv=_fmt_cat(cats, "convention"),
                tq=_fmt_cat(cats, "text_quantity"),
            )
        )

        structured.append(
            {
                "label": r["_label"],
                "run_dir": r["_run_dir"],
                "mode": r.get("mode"),
                "total_questions": tot,
                "execution_accuracy": r.get("execution_accuracy"),
                "sqlite_executable_rate": r.get("sqlite_executable_rate"),
                "mean_sql_component_f1": r.get("mean_sql_component_f1"),
                "raw_cfg_compliance_rate": r.get("raw_cfg_compliance_rate"),
                "mean_raw_cfg_prefix_ratio": r.get("mean_raw_cfg_prefix_ratio"),
                "extracted_sql_cfg_compliance_rate": r.get("extracted_sql_cfg_compliance_rate"),
                "mean_latency_ms": lat,
                "speedup_vs_outlines": round(speedup, 4) if speedup is not None else None,
                "safetensors_delta_vs_input_bytes": r.get("safetensors_delta_vs_input_bytes"),
                "sql_component_f1_breakdown": parts,
                "category_breakdown": cats,
            }
        )

    md = "\n".join(
        [
            "### Overall Performance Comparison",
            "",
            meta_hdr,
            "",
            *t1,
            "",
            "### SQL Clause / Component F1 & Category Breakdown",
            "",
            *t2,
            "",
        ]
    )
    payload = {
        "benchmark": first.get("_benchmark", "llmsql-2.0"),
        "sampled_questions": first.get("_sampled_questions"),
        "prompt_style": first.get("_prompt_style"),
        "seed": first.get("_seed"),
        "models": structured,
    }
    return md, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Markdown & JSON comparison tables from eval/runs/*/summary.json."
    )
    parser.add_argument("--runs-dir", default="eval/runs", help="Directory containing evaluation run subdirectories.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    rows = _collect_rows(runs_dir)
    if not rows:
        print(f"ℹ️  No evaluation summary.json files found under {runs_dir}; skipping comparison table.")
        return

    md, payload = build_comparison_markdown(rows)
    runs_dir.mkdir(parents=True, exist_ok=True)
    md_path = runs_dir / "comparison_summary.md"
    json_path = runs_dir / "comparison_summary.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("")
    print("=" * 70)
    print("📊 Performance Comparison Across All Evaluated Models")
    print("=" * 70)
    print(md)
    print(f"✅ Saved comparison tables to {md_path} and {json_path}")


if __name__ == "__main__":
    main()

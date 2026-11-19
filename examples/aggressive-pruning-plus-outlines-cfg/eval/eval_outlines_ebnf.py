#!/usr/bin/env python3
"""Outlines EBNF Context-Free Grammar (CFG) Constrained Decoding Evaluation.

Evaluates whether constraining decoding via Outlines (`outlines.cfg` backed by
`llguidance`) on `Qwen/Qwen2.5-Coder-0.5B-Instruct` reproduces or complements
the Weight-Space Programming (WSP) depth-pruned model (`models/output`).

Compares:
  1. Baseline 24-Layer Model (`models/input`, 942.32 MB) + Outlines EBNF
  2. Pruned 4-Layer WSP Model (`models/output`, 186.76 MB) + Outlines EBNF

Usage:
    uv run eval/eval_outlines_ebnf.py --suite all
    uv run eval/eval_outlines_ebnf.py --models models/input --suite original
    uv run eval/eval_outlines_ebnf.py --models models/output --suite extended
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

try:
    from eval.eval_extended import (
        EXTENDED_LS_PROMPTS,
        EXTENDED_NON_TARGET_PROMPTS,
        EXTENDED_PWD_PROMPTS,
        DualLogger,
        is_valid_refusal,
    )
    from eval.eval_model import (
        LS_PROMPTS,
        NON_TARGET_PROMPTS,
        PWD_PROMPTS,
        build_prompt,
        extract_command,
        is_valid_false,
        is_valid_ls,
        is_valid_pwd,
        resolve_model_dir,
    )
except ModuleNotFoundError:
    from eval_extended import (
        EXTENDED_LS_PROMPTS,
        EXTENDED_NON_TARGET_PROMPTS,
        EXTENDED_PWD_PROMPTS,
        DualLogger,
        is_valid_refusal,
    )
    from eval_model import (
        LS_PROMPTS,
        NON_TARGET_PROMPTS,
        PWD_PROMPTS,
        build_prompt,
        extract_command,
        is_valid_false,
        is_valid_ls,
        is_valid_pwd,
        resolve_model_dir,
    )


# Lark/EBNF context-free grammar constraining output to either the execute_bash
# JSON schema or direct {pwd, ls, false} CLI commands.
# Note: <tool_call> is a special token (ID 151657) in Qwen2.5-Coder, so omitting
# literal "<tool_call>" tags avoids LLMatcher special-token rejection errors, and
# bounding ls flags/paths prevents unbounded token repetition loops.
EBNF_GRAMMAR = r"""
start: tool_json | cmd
tool_json: "{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"" cmd "\"}}" "\n"?
cmd: pwd_cmd | ls_cmd | false_cmd
pwd_cmd: "pwd" (" -P" | " -L")?
ls_cmd: "ls" (" -" /[a-zA-Z]{1,4}/)* (" /" | " .")?
false_cmd: "false"
"""

NON_ENGLISH_PROMPTS: frozenset[str] = frozenset(
    [
        *EXTENDED_PWD_PROMPTS[5:],
        *EXTENDED_LS_PROMPTS[5:],
        *EXTENDED_NON_TARGET_PROMPTS[9:13],
    ]
)


def _select_device_and_dtype() -> tuple[str, torch.dtype]:
    """Select optimal inference device and floating-point dtype."""
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _build_outlines_generator(
    model_dir: str,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, Any, str, torch.dtype]:
    """Load HF model + tokenizer and wrap them in an Outlines CFG generator."""
    device, dtype = _select_device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    outlines_model = outlines.from_transformers(model, tokenizer)
    # Prompts are already formatted via build_prompt() with tools + chat template;
    # disable Outlines' implicit second chat-template wrapping on string inputs.
    outlines_model.type_adapter.has_chat_template = False
    generator = outlines.Generator(outlines_model, outlines.cfg(EBNF_GRAMMAR))
    return tokenizer, model, generator, device, dtype


def _run_target_group(
    title: str,
    prompts: list[str],
    validator: Any,
    tokenizer: AutoTokenizer,
    generator: Any,
    logger: DualLogger,
    max_new_tokens: int,
    few_shot: bool,
) -> dict[str, Any]:
    """Evaluate a PWD or LS prompt group under EBNF constrained decoding."""
    logger.write("\n" + "-" * 76 + "\n")
    logger.write(f"{title}\n")
    logger.write("-" * 76 + "\n")

    passed_count = 0
    details: list[dict[str, Any]] = []

    for prompt in prompts:
        prompt_text = build_prompt(tokenizer, prompt, few_shot=few_shot)
        t0 = time.perf_counter()
        raw_output = str(
            generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=False)
        ).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cmd = extract_command(raw_output)
        passed = bool(validator(cmd) and not is_valid_false(cmd))
        if passed:
            passed_count += 1

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.write(f'  [{status}] Prompt  : "{prompt}"\n')
        logger.write(
            f"           Command : `{cmd}` ({latency_ms:.1f}ms) | Raw: {raw_output}\n"
        )
        details.append(
            {
                "prompt": prompt,
                "raw_output": raw_output,
                "command": cmd,
                "passed": passed,
                "latency_ms": round(latency_ms, 1),
                "is_english": prompt not in NON_ENGLISH_PROMPTS,
            }
        )

    total = len(prompts)
    return {
        "total": total,
        "passed": passed_count,
        "rate": round(passed_count / total, 4) if total else 0.0,
        "details": details,
    }


def _run_non_target_group(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    generator: Any,
    logger: DualLogger,
    max_new_tokens: int,
    few_shot: bool,
) -> dict[str, Any]:
    """Evaluate non-target prompts using safe subspace refusal (matching eval_extended.py)."""
    logger.write("\n" + "-" * 76 + "\n")
    logger.write(
        "3. Pruning / Selectivity: Non-Target Queries (Safe Subspace Refusal: {pwd, ls, false})\n"
    )
    logger.write("-" * 76 + "\n")

    passed_count = 0
    strict_false_count = 0
    details: list[dict[str, Any]] = []

    for prompt in prompts:
        prompt_text = build_prompt(tokenizer, prompt, few_shot=few_shot)
        t0 = time.perf_counter()
        raw_output = str(
            generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=False)
        ).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cmd = extract_command(raw_output)

        passed = bool(is_valid_refusal(cmd))
        emitted_false = bool(is_valid_false(cmd))
        if passed:
            passed_count += 1
        if emitted_false:
            strict_false_count += 1

        if emitted_false:
            status = "✅ PASS (Refusal: 'false')"
        elif passed:
            status = "✅ PASS (Subspace Safe)"
        else:
            status = "❌ FAIL (Executed unapproved command)"

        logger.write(f'  [{status}] Prompt  : "{prompt}"\n')
        logger.write(
            f"           Command : `{cmd}` ({latency_ms:.1f}ms) | Raw: {raw_output}\n"
        )
        details.append(
            {
                "prompt": prompt,
                "raw_output": raw_output,
                "command": cmd,
                "passed": passed,
                "emitted_false": emitted_false,
                "latency_ms": round(latency_ms, 1),
                "is_english": prompt not in NON_ENGLISH_PROMPTS,
            }
        )

    total = len(prompts)
    return {
        "total": total,
        "passed": passed_count,
        "rate": round(passed_count / total, 4) if total else 0.0,
        "emitted_false_count": strict_false_count,
        "details": details,
    }


def _compute_english_breakdown(
    pwd_res: dict[str, Any],
    ls_res: dict[str, Any],
    nt_res: dict[str, Any],
) -> dict[str, Any]:
    """Compute accuracy metrics restricted to English-language prompts."""
    pwd_en = [d for d in pwd_res["details"] if d["is_english"]]
    ls_en = [d for d in ls_res["details"] if d["is_english"]]
    nt_en = [d for d in nt_res["details"] if d["is_english"]]

    pwd_ok = sum(1 for d in pwd_en if d["passed"])
    ls_ok = sum(1 for d in ls_en if d["passed"])
    nt_ok = sum(1 for d in nt_en if d["passed"])
    tot_en = len(pwd_en) + len(ls_en) + len(nt_en)

    return {
        "total": tot_en,
        "pwd": {
            "passed": pwd_ok,
            "total": len(pwd_en),
            "rate": round(pwd_ok / len(pwd_en), 4) if pwd_en else 0.0,
        },
        "ls": {
            "passed": ls_ok,
            "total": len(ls_en),
            "rate": round(ls_ok / len(ls_en), 4) if ls_en else 0.0,
        },
        "non_target": {
            "passed": nt_ok,
            "total": len(nt_en),
            "rate": round(nt_ok / len(nt_en), 4) if nt_en else 0.0,
        },
        "overall_rate": round((pwd_ok + ls_ok + nt_ok) / tot_en, 4) if tot_en else 0.0,
    }


def evaluate_ebnf_suite(
    suite_name: str,
    pwd_prompts: list[str],
    ls_prompts: list[str],
    non_target_prompts: list[str],
    tokenizer: AutoTokenizer,
    generator: Any,
    logger: DualLogger,
    max_new_tokens: int = 64,
    few_shot: bool = True,
) -> dict[str, Any]:
    """Run one full prompt suite (original or extended) with Outlines EBNF decoding."""
    logger.write("\n" + "=" * 76 + "\n")
    logger.write(f"🚀 Evaluating Suite: {suite_name.upper()} (Outlines EBNF CFG)\n")
    logger.write(
        f"   Prompts: {len(pwd_prompts)} PWD, {len(ls_prompts)} LS, {len(non_target_prompts)} Non-Target\n"
    )
    logger.write("=" * 76 + "\n")

    pwd_res = _run_target_group(
        "1. Target Knowledge: PWD Commands",
        pwd_prompts,
        is_valid_pwd,
        tokenizer,
        generator,
        logger,
        max_new_tokens,
        few_shot,
    )
    ls_res = _run_target_group(
        "2. Target Knowledge: LS Commands",
        ls_prompts,
        is_valid_ls,
        tokenizer,
        generator,
        logger,
        max_new_tokens,
        few_shot,
    )
    nt_res = _run_non_target_group(
        non_target_prompts,
        tokenizer,
        generator,
        logger,
        max_new_tokens,
        few_shot,
    )

    all_latencies = [
        d["latency_ms"]
        for group in (pwd_res["details"], ls_res["details"], nt_res["details"])
        for d in group
    ]
    total_prompts = pwd_res["total"] + ls_res["total"] + nt_res["total"]
    total_passed = pwd_res["passed"] + ls_res["passed"] + nt_res["passed"]

    summary = {
        "pwd_rate": pwd_res["rate"],
        "ls_rate": ls_res["rate"],
        "refusal_rate": nt_res["rate"],
        "overall_rate": round(total_passed / total_prompts, 4)
        if total_prompts
        else 0.0,
        "mean_latency_ms": round(sum(all_latencies) / len(all_latencies), 1)
        if all_latencies
        else 0.0,
        "english_only": _compute_english_breakdown(pwd_res, ls_res, nt_res),
    }

    logger.write("\n" + "=" * 76 + "\n")
    logger.write(f"📊 {suite_name.upper()} EBNF RESULTS SUMMARY\n")
    logger.write("=" * 76 + "\n")
    logger.write(
        f"  • PWD Tasks            : {pwd_res['passed']}/{pwd_res['total']} ({summary['pwd_rate'] * 100:.1f}%)\n"
    )
    logger.write(
        f"  • LS Tasks             : {ls_res['passed']}/{ls_res['total']} ({summary['ls_rate'] * 100:.1f}%)\n"
    )
    logger.write(
        f"  • Non-Target Refusal   : {nt_res['passed']}/{nt_res['total']} ({summary['refusal_rate'] * 100:.1f}%) "
        f"[Emitted 'false': {nt_res['emitted_false_count']}/{nt_res['total']}]\n"
    )
    logger.write(
        f"  • Total Accuracy       : {total_passed}/{total_prompts} ({summary['overall_rate'] * 100:.1f}%)\n"
    )
    logger.write(f"  • Mean Decode Latency  : {summary['mean_latency_ms']:.1f} ms\n")
    en = summary["english_only"]
    logger.write(
        f"  • English-Only         : PWD {en['pwd']['passed']}/{en['pwd']['total']}, "
        f"LS {en['ls']['passed']}/{en['ls']['total']}, "
        f"Refusal {en['non_target']['passed']}/{en['non_target']['total']} "
        f"({en['overall_rate'] * 100:.1f}%)\n"
    )
    logger.write("=" * 76 + "\n")

    return {
        "suite": suite_name,
        "pwd": pwd_res,
        "ls": ls_res,
        "non_target": nt_res,
        "summary": summary,
    }


def evaluate_model_ebnf(
    model_identifier: str,
    suites: list[str],
    logger: DualLogger,
    max_new_tokens: int = 64,
    few_shot: bool = True,
) -> dict[str, Any]:
    """Load one model checkpoint and run all requested suites under EBNF constraints."""
    resolved = resolve_model_dir(model_identifier)
    logger.write("\n" + "#" * 76 + "\n")
    logger.write(f"🧪 Loading Model for Outlines EBNF Evaluation: {resolved}\n")
    logger.write("#" * 76 + "\n")

    t_load = time.perf_counter()
    tokenizer, model, generator, device, dtype = _build_outlines_generator(resolved)
    logger.write(
        f"✅ Loaded ({device}, {dtype}) + compiled LLGuidance CFG in {time.perf_counter() - t_load:.2f}s.\n"
    )

    model_results: dict[str, Any] = {
        "identifier": model_identifier,
        "resolved_path": resolved,
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "device": device,
        "dtype": str(dtype),
        "suites": {},
    }

    suite_defs = {
        "original": (PWD_PROMPTS, LS_PROMPTS, NON_TARGET_PROMPTS),
        "extended": (
            EXTENDED_PWD_PROMPTS,
            EXTENDED_LS_PROMPTS,
            EXTENDED_NON_TARGET_PROMPTS,
        ),
    }

    try:
        for suite_name in suites:
            pwds, lss, nts = suite_defs[suite_name]
            model_results["suites"][suite_name] = evaluate_ebnf_suite(
                suite_name=suite_name,
                pwd_prompts=pwds,
                ls_prompts=lss,
                non_target_prompts=nts,
                tokenizer=tokenizer,
                generator=generator,
                logger=logger,
                max_new_tokens=max_new_tokens,
                few_shot=few_shot,
            )
    finally:
        del generator, model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    return model_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Outlines EBNF CFG decoding against WSP aggressive pruning."
    )
    parser.add_argument(
        "--models",
        "--model-dir",
        dest="models",
        nargs="+",
        default=["models/input", "models/output"],
        help="Model paths or HF IDs to evaluate (default: models/input models/output).",
    )
    parser.add_argument(
        "--suite",
        choices=["original", "extended", "all"],
        default="all",
        help="Evaluation suite to run (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/runs/2026-11-18_outlines-ebnf",
        help="Directory to save evaluation logs and summary.json.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per query (default: 64).",
    )
    parser.add_argument(
        "--zero-shot",
        dest="few_shot",
        action="store_false",
        help="Disable 1-shot anchor in chat template.",
    )
    args = parser.parse_args()

    suites = ["original", "extended"] if args.suite == "all" else [args.suite]
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "run.log")
    summary_file = os.path.join(out_dir, "summary.json")

    logger = DualLogger(log_file)
    try:
        logger.write("=" * 76 + "\n")
        logger.write("🔬 OUTLINES EBNF CONSTRAINED DECODING BENCHMARK\n")
        logger.write(f"   Models          : {', '.join(args.models)}\n")
        logger.write(f"   Suite Selection : {args.suite}\n")
        logger.write(f"   Few-shot Anchor : {args.few_shot}\n")
        logger.write(f"   Max New Tokens  : {args.max_new_tokens}\n")
        logger.write(f"   Run Directory   : {out_dir}\n")
        logger.write("=" * 76 + "\n")

        all_results: dict[str, Any] = {
            "date": "2026-11-18",
            "eval_type": "outlines-ebnf-cfg",
            "backend": "llguidance",
            "few_shot": args.few_shot,
            "max_new_tokens": args.max_new_tokens,
            "ebnf_grammar": EBNF_GRAMMAR.strip(),
            "models": {},
        }

        for model_id in args.models:
            all_results["models"][model_id] = evaluate_model_ebnf(
                model_identifier=model_id,
                suites=suites,
                logger=logger,
                max_new_tokens=args.max_new_tokens,
                few_shot=args.few_shot,
            )

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.write(f"\n📁 Evaluation logs saved to: {log_file}\n")
        logger.write(f"📊 Summary metrics saved to: {summary_file}\n")
        logger.write("\n✨ Outlines EBNF evaluation complete.\n")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

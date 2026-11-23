#!/usr/bin/env python3
"""Out-of-harness LLMSQL & ISO SQL:2016 CFG evaluation suite for Qwen/Qwen2.5-Coder-0.5B-Instruct.

Evaluates unwrapped Hugging Face checkpoints (`AutoModelForCausalLM.generate`) and
optional runtime Outlines CFG baselines against `llmsql-bench/llmsql-2.0` (or `1.0`)
and verifies syntactic compliance against the verbatim ISO SQL:2016 EBNF grammar
(`sql-2016.ebnf`) via `llguidance.LLMatcher`.

Usage:
    uv run --with "llmsql>=0.1.16" eval/eval_model.py --self-test
    uv run --with "llmsql>=0.1.16" eval/eval_model.py --model-dir models/output --mode unwrapped --limit 50
    uv run --with "llmsql>=0.1.16" eval/eval_model.py --model-dir models/input --mode both --limit 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

try:
    from eval.sql_grammar import (
        DEFAULT_EBNF_PATH,
        build_llg_matcher_spec,
        compute_sql_component_f1,
        ebnf_to_lark,
        evaluate_sql_grammar_prefix,
        extract_sql_candidate,
        is_sql_grammar_valid,
        load_raw_sql_ebnf,
    )
except ModuleNotFoundError:
    from sql_grammar import (
        DEFAULT_EBNF_PATH,
        build_llg_matcher_spec,
        compute_sql_component_f1,
        ebnf_to_lark,
        evaluate_sql_grammar_prefix,
        extract_sql_candidate,
        is_sql_grammar_valid,
        load_raw_sql_ebnf,
    )


class DualLogger:
    """Mirror stdout output into a persistent run.log file."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(log_path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        sys.stdout.write(msg)
        sys.stdout.flush()
        self._fp.write(msg)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


_HF_REPO_ID_RX = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def resolve_model_dir(model_dir: str, allow_missing: bool = False) -> str | None:
    """Resolve a local checkpoint directory or validate a Hugging Face repo ID."""
    path = Path(model_dir).expanduser().resolve()
    if path.is_dir():
        files = os.listdir(path)
        has_weights = any(
            f.endswith((".safetensors", ".bin")) or f == "model.safetensors.index.json"
            for f in files
        )
        if has_weights:
            return str(path)
        if allow_missing:
            print(f"⚠️  No model weights found in {path} (--allow-missing set, skipping).")
            return None
        raise FileNotFoundError(f"No .safetensors or .bin weight files found in {path}")

    is_local_prefix = (
        model_dir.startswith(("/", "./", "../", "~", "models/"))
        or model_dir.endswith(("/", "\\"))
    )
    if not is_local_prefix and _HF_REPO_ID_RX.fullmatch(model_dir):
        return model_dir

    if allow_missing:
        print(f"⚠️  Model directory not found: {path} (--allow-missing set, skipping).")
        return None
    raise FileNotFoundError(f"Model directory not found: {path}")


def total_safetensors_bytes(model_dir: str | None) -> int | None:
    """Compute total size in bytes of all .safetensors files in `model_dir`."""
    if not model_dir or not os.path.isdir(model_dir):
        return None
    total = 0
    found = False
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(model_dir, name))
            found = True
    return total if found else None


def select_stratified_questions(
    questions: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically sample a category-balanced subset when `limit > 0`."""
    if limit <= 0 or limit >= len(questions):
        return questions

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        cat = str(q.get("category") or "default")
        by_cat.setdefault(cat, []).append(q)

    for cat_items in by_cat.values():
        rng.shuffle(cat_items)

    cats = sorted(by_cat.keys())
    selected: list[dict[str, Any]] = []
    idx = 0
    while len(selected) < limit:
        added_any = False
        for cat in cats:
            if idx < len(by_cat[cat]):
                selected.append(by_cat[cat][idx])
                added_any = True
                if len(selected) >= limit:
                    break
        if not added_any:
            break
        idx += 1

    selected.sort(key=lambda item: int(item["question_id"]))
    return selected


def build_llmsql_prompt(
    question_row: dict[str, Any],
    tables_by_id: dict[str, dict[str, Any]] | None,
    prompt_style: str = "raw-sql",
) -> str:
    """Format an evaluation prompt in `raw-sql` (default), `llmsql`, or `sqlcoder` style."""
    if prompt_style == "sqlcoder":
        q_text = str(question_row.get("question", "")).strip()
        if question_row.get("prompt"):
            raw_p = str(question_row["prompt"])
            m_ddl = re.search(r"(CREATE TABLE .*?)(?:\n\nQuestion:|$)", raw_p, flags=re.S)
            ddl = m_ddl.group(1).strip() if m_ddl else raw_p.strip()
        else:
            tbl = (tables_by_id or {})[question_row["table_id"]]
            cols = ", ".join(f'"{h}" {t}' for h, t in zip(tbl["header"], tbl["types"]))
            ddl = f'CREATE TABLE "{question_row["table_id"]}" ({cols});'
        return (
            "### Task\n"
            f"Generate a SQL query to answer [QUESTION]{q_text}[/QUESTION]\n\n"
            "### Database Schema\n"
            "The query will run on a database with the following schema:\n"
            f"{ddl}\n\n"
            "### Answer\n"
            f"Given the database schema, here is the SQL query that [QUESTION]{q_text}[/QUESTION]\n"
            "[SQL]\n"
        )

    if prompt_style == "raw-sql" and question_row.get("prompt"):
        raw_p = str(question_row["prompt"])
        replaced = re.sub(
            r"Return only the SQL in a ```sql\s*block\.?",
            "Return only the raw SQL query terminated by a semicolon, with no explanations and no markdown fences.",
            raw_p,
        )
        return replaced.rstrip() + "\n\nSQL:\n"

    if question_row.get("prompt"):
        return str(question_row["prompt"])
    from llmsql.utils.utils import build_prompt_0shot

    tbl = (tables_by_id or {})[question_row["table_id"]]
    sample_row = tbl["rows"][0] if tbl.get("rows") else []
    prompt = build_prompt_0shot(
        question_row["question"],
        tbl["header"],
        tbl["types"],
        sample_row,
    )
    return prompt.rstrip() + "\n" if prompt_style == "raw-sql" else prompt


def _normalize_sql_str(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip())


def run_self_test(ebnf_path: Path) -> bool:
    """Run fast offline checks on `sql-2016.ebnf`, LLGuidance, and `llmsql`."""
    import llmsql
    from llmsql.config.config import get_repo_id
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    raw_ebnf = load_raw_sql_ebnf(str(ebnf_path))
    lark_text, meta = ebnf_to_lark(raw_ebnf)
    assert meta["entry_rule"] == "direct_SQL_statement", meta
    assert meta["total_rules"] >= 2000, meta
    assert meta["reachable_rules"] >= 1000, meta

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    chars = (
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "_;,\"'()*=<>+-./\\:[]{}!?@#$%^&|"
    )
    for ch in chars:
        key = "Ġ" if ch == " " else ch
        if key not in vocab:
            vocab[key] = len(vocab)
    tok_backend = HFTokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok_backend.decoder = ByteLevelDecoder()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )

    llg_tok, g_spec, _, _ = build_llg_matcher_spec(tok, str(ebnf_path))

    valid_samples = [
        'SELECT "Singer(s)" FROM "1-29135051-2" WHERE "Comedian" = \'Joe Wilkinson\';',
        "SELECT COUNT(*) FROM employees WHERE salary > 50000;",
        "SELECT AVG(score) FROM matches GROUP BY team_id ORDER BY team_id;",
        'SELECT "name", "salary" FROM "employees" WHERE "salary" > 50000 ORDER BY "salary" DESC;',
        ' SELECT "id" FROM "users" ORDER BY "id" ASC;',
    ]
    for sql in valid_samples:
        ok, err = is_sql_grammar_valid(llg_tok, g_spec, tok, sql)
        assert ok, f"Expected valid SQL:2016 sample {sql!r} to pass, got error: {err}"

    invalid_samples = [
        "THIS IS NOT SQL",
        "SELECT FROM;",
        'SELECT "name" FROM "employees" ORDER BY "salary" DESC*;',
        "",
    ]
    for bad in invalid_samples:
        ok, _ = is_sql_grammar_valid(llg_tok, g_spec, tok, bad)
        assert not ok, f"Expected invalid text {bad!r} to be rejected"

    assert extract_sql_candidate("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql_candidate("SELECT 1; [/SQL]") == "SELECT 1;"
    assert get_repo_id("2.0") == "llmsql-bench/llmsql-2.0"
    assert hasattr(llmsql, "evaluate")
    with tempfile.TemporaryDirectory() as tmp_cache:
        cache_dir = Path(tmp_cache) / "v2.0"
        dummy_q = cache_dir / "questions.jsonl"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dummy_q.write_text('{"question_id": 1}\n', encoding="utf-8")
        cached_path = ensure_benchmark_file("llmsql-bench/llmsql-2.0", "questions.jsonl", cache_dir)
        assert Path(cached_path).is_file() and cached_path == str(dummy_q)
    assert resolve_model_dir("defog/sqlcoder-7b-2") == "defog/sqlcoder-7b-2"
    assert resolve_model_dir("Qwen/Qwen2.5-Coder-0.5B-Instruct") == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert resolve_model_dir("Qwen/Qwen2.5-Coder-0.5B") == "Qwen/Qwen2.5-Coder-0.5B"
    assert resolve_model_dir(str(EXAMPLE_DIR / "models" / "missing_dir"), allow_missing=True) is None
    sample_q = {
        "question_id": 1,
        "table_id": "t1",
        "question": "Count rows",
        "prompt": (
            "Return only the SQL in a ```sql block.\n\n"
            'CREATE TABLE "t1" ("a" INTEGER);\n\nQuestion: Count rows'
        ),
    }
    raw_sql_p = build_llmsql_prompt(sample_q, None, prompt_style="raw-sql")
    assert "```sql" not in raw_sql_p and "no markdown fences" in raw_sql_p and raw_sql_p.endswith("SQL:\n")
    assert "CREATE TABLE" in build_llmsql_prompt(sample_q, None, prompt_style="llmsql")
    sqlcoder_p = build_llmsql_prompt(sample_q, None, prompt_style="sqlcoder")
    assert "[QUESTION]Count rows[/QUESTION]" in sqlcoder_p and sqlcoder_p.endswith("[SQL]\n")

    raw_res = evaluate_sql_grammar_prefix(llg_tok, g_spec, tok, valid_samples[0])
    assert raw_res["valid"] is True and raw_res["valid_prefix_ratio"] == 1.0
    bad_none = evaluate_sql_grammar_prefix(llg_tok, g_spec, tok, "XYZ NOT SQL")
    assert bad_none["valid"] is False and bad_none["valid_prefix_ratio"] == 0.0
    partial_res = evaluate_sql_grammar_prefix(
        llg_tok, g_spec, tok, 'SELECT "name" FROM "employees" ORDER BY "salary" DESC*;'
    )
    assert partial_res["valid"] is False and 0.0 < partial_res["valid_prefix_ratio"] < 1.0

    f1_exact = compute_sql_component_f1(
        'SELECT COUNT("id") FROM "t" WHERE "a" > 10;',
        'SELECT COUNT("id") FROM "t" WHERE "a" > 10;',
    )
    assert f1_exact["overall_f1"] == 1.0
    f1_partial = compute_sql_component_f1(
        'SELECT COUNT(*) FROM "t" WHERE "a" = 5;',
        'SELECT SUM("b") FROM "t" WHERE "a" > 10;',
    )
    assert 0.0 < float(f1_partial["overall_f1"] or 0.0) < 1.0
    assert compute_sql_component_f1("Answer: 10", 'SELECT "a" FROM "t";')["overall_f1"] == 0.0

    with tempfile.TemporaryDirectory() as tmp_db_dir:
        import sqlite3

        db_file = Path(tmp_db_dir) / "test.db"
        conn = sqlite3.connect(db_file)
        conn.execute('CREATE TABLE "t1" ("a" INTEGER, "b" TEXT)')
        conn.execute('INSERT INTO "t1" VALUES (1, \'ok\')')
        conn.commit()
        conn.close()
        exec_stats = compute_sqlite_executable_stats(
            [
                {"question_id": 1, "completion": 'SELECT "a" FROM "t1";'},
                {"question_id": 2, "completion": 'SELECT missing_col FROM "t1";'},
                {"question_id": 3, "completion": "Answer: 42"},
            ],
            {1: {"table_id": "t1"}, 2: {"table_id": "t1"}, 3: {"table_id": "t1"}},
            str(db_file),
        )
        assert exec_stats == {
            "sqlite_executable_count": 1,
            "sqlite_executable_rate": 0.3333,
            "no_sql_extracted_count": 1,
        }, exec_stats

    summary = {
        "self_test_passed": True,
        "ebnf_path": str(ebnf_path),
        "ebnf_bytes": len(raw_ebnf.encode("utf-8")),
        "lark_chars": len(lark_text),
        "grammar_meta": meta,
        "llmsql_repo_2_0": get_repo_id("2.0"),
        "valid_sql_checked": len(valid_samples),
        "invalid_sql_rejected": len(invalid_samples),
    }
    print(json.dumps(summary, indent=2))
    return True


def _select_device_and_dtype():
    import torch

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def compute_sqlite_executable_stats(
    predictions: list[dict[str, Any]],
    q_by_id: dict[int, dict[str, Any]],
    db_path: str,
) -> dict[str, Any]:
    """Measure per-question SQLite executability rate and missed-SQL extraction count."""
    from llmsql.utils.evaluation_utils import (
        connect_sqlite,
        execute_sql,
        find_sql,
        fix_table_name,
    )

    executable_count = 0
    no_sql_count = 0
    conn = connect_sqlite(db_path)
    try:
        for row in predictions:
            qid = int(row["question_id"])
            table_id = str(q_by_id[qid]["table_id"])
            queries = find_sql(str(row.get("completion") or ""))
            if not queries:
                no_sql_count += 1
                continue
            if any(
                execute_sql(conn, fix_table_name(q_sql, table_id)) is not None
                for q_sql in queries
            ):
                executable_count += 1
    finally:
        conn.close()

    total_n = len(predictions)
    return {
        "sqlite_executable_count": executable_count,
        "sqlite_executable_rate": round(executable_count / total_n, 4) if total_n else 0.0,
        "no_sql_extracted_count": no_sql_count,
    }


def evaluate_mode(
    *,
    mode: str,
    model_dir: str,
    baseline_dir: str | None,
    questions: list[dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]] | None,
    questions_path: str,
    db_path: str,
    ebnf_path: str,
    version: str,
    workdir: Path,
    output_dir: Path,
    max_new_tokens: int,
    prompt_style: str,
    logger: DualLogger,
) -> dict[str, Any]:
    """Run inference in either `unwrapped` or `outlines` mode and score with LLMSQL + LLMatcher."""
    import llmsql
    from llmsql.utils.evaluation_utils import fix_table_name
    import outlines
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    device, dtype = _select_device_and_dtype()
    logger.write(f"\n{'#' * 76}\n")
    logger.write(f"🧪 Evaluating [{mode.upper()}] on {model_dir} ({device}, {dtype})\n")
    logger.write(f"{'#' * 76}\n")

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = int(tokenizer.eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    logger.write(f"✅ Model loaded in {time.perf_counter() - t_load:.2f}s.\n")

    llg_tok, g_spec, lark_text, grammar_meta = build_llg_matcher_spec(tokenizer, ebnf_path)
    generator = None
    if mode == "outlines":
        outlines_model = outlines.from_transformers(model, tokenizer)
        outlines_model.type_adapter.has_chat_template = False
        generator = outlines.Generator(outlines_model, outlines.cfg(lark_text))

    predictions: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    raw_cfg_valid = 0
    extracted_cfg_valid = 0
    raw_prefix_ratios: list[float] = []
    extracted_prefix_ratios: list[float] = []
    comp_f1_overall: list[float] = []
    comp_f1_parts: dict[str, list[float]] = {
        "clauses_f1": [],
        "functions_f1": [],
        "identifiers_f1": [],
        "select_f1": [],
        "where_f1": [],
    }
    exact_match_count = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids: set[int] = set()
    for eos_src in (eos_id, getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if isinstance(eos_src, int):
            eos_ids.add(eos_src)
        elif isinstance(eos_src, (list, tuple)):
            eos_ids.update(int(x) for x in eos_src if x is not None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None and eos_id is not None:
        pad_id = int(eos_id)
    q_by_id = {int(q["question_id"]): q for q in questions}

    try:
        for idx, q in enumerate(questions, start=1):
            qid = int(q["question_id"])
            prompt_text = build_llmsql_prompt(q, tables_by_id, prompt_style=prompt_style)
            t0 = time.perf_counter()
            if mode == "outlines":
                raw_output = str(
                    generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=False)
                ).strip()
                trimmed_ids = tokenizer.encode(raw_output, add_special_tokens=False)
            else:
                inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                }
                if pad_id is not None:
                    gen_kwargs["pad_token_id"] = int(pad_id)
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                gen_ids = out[0][inputs["input_ids"].shape[-1] :].tolist()
                trimmed_ids = []
                for tid in gen_ids:
                    if int(tid) in eos_ids:
                        break
                    trimmed_ids.append(int(tid))
                raw_output = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            raw_res = evaluate_sql_grammar_prefix(
                llg_tok, g_spec, tokenizer, raw_output, token_ids=trimmed_ids
            )
            raw_ok = bool(raw_res["valid"])
            raw_prefix = float(raw_res["valid_prefix_ratio"])
            raw_prefix_ratios.append(raw_prefix)
            if raw_ok:
                raw_cfg_valid += 1

            candidate = extract_sql_candidate(raw_output)
            cand_with_semi = (
                candidate if candidate.endswith(";") else f"{candidate};"
            ) if candidate else ""
            ext_res = evaluate_sql_grammar_prefix(
                llg_tok, g_spec, tokenizer, cand_with_semi
            )
            ext_ok = bool(ext_res["valid"])
            extracted_prefix_ratios.append(float(ext_res["valid_prefix_ratio"]))
            if ext_ok:
                extracted_cfg_valid += 1

            gold_sql = str(q.get("sql", ""))
            candidate_fixed = (
                fix_table_name(candidate, str(q.get("table_id", "")))
                if candidate
                else ""
            )
            f1_res = compute_sql_component_f1(candidate_fixed, gold_sql)
            comp_f1 = float(f1_res["overall_f1"] or 0.0)
            comp_f1_overall.append(comp_f1)
            for k_part in comp_f1_parts:
                val = f1_res.get(k_part)
                if val is not None:
                    comp_f1_parts[k_part].append(float(val))

            if candidate and gold_sql and _normalize_sql_str(candidate) == _normalize_sql_str(gold_sql):
                exact_match_count += 1

            predictions.append({"question_id": qid, "completion": raw_output})
            if idx <= 3 or idx % 10 == 0 or idx == len(questions):
                logger.write(
                    f"  [{idx}/{len(questions)}] qid={qid} raw_cfg={raw_ok} (pfx={raw_prefix:.2f}) "
                    f"ext_cfg={ext_ok} comp_f1={comp_f1:.2f} ({lat_ms:.1f}ms) -> {raw_output[:90]!r}\n"
                )
    finally:
        if generator is not None:
            del generator
        del model, tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = output_dir / f"predictions_{mode}.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f_pred:
        for row in predictions:
            f_pred.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_file = output_dir / f"llmsql_report_{mode}.json"
    llmsql_report = llmsql.evaluate(
        str(pred_file),
        version=version,
        workdir_path=str(workdir),
        questions_path=questions_path,
        db_path=db_path,
        save_report=str(report_file),
        show_mismatches=False,
    )

    # Compute category execution breakdown
    mismatch_qids = {int(m["question_id"]) for m in llmsql_report.get("mismatches", [])}
    cat_stats: dict[str, dict[str, int | float]] = {}
    for row in predictions:
        qid = int(row["question_id"])
        cat = str(q_by_id[qid].get("category") or "default")
        bucket = cat_stats.setdefault(cat, {"total": 0, "matches": 0, "accuracy": 0.0})
        bucket["total"] = int(bucket["total"]) + 1
        if qid not in mismatch_qids:
            bucket["matches"] = int(bucket["matches"]) + 1
    for bucket in cat_stats.values():
        tot = int(bucket["total"])
        bucket["accuracy"] = round(int(bucket["matches"]) / tot, 4) if tot else 0.0

    in_bytes = total_safetensors_bytes(baseline_dir)
    out_bytes = total_safetensors_bytes(model_dir)
    size_delta = (out_bytes - in_bytes) if (in_bytes is not None and out_bytes is not None) else None

    exec_stats = compute_sqlite_executable_stats(predictions, q_by_id, db_path)

    total_n = len(questions)
    comp_f1_breakdown = {
        k_part: (round(sum(vals) / len(vals), 4) if vals else None)
        for k_part, vals in comp_f1_parts.items()
    }
    mean_raw_prefix = (
        round(sum(raw_prefix_ratios) / len(raw_prefix_ratios), 4)
        if raw_prefix_ratios
        else 0.0
    )
    mean_ext_prefix = (
        round(sum(extracted_prefix_ratios) / len(extracted_prefix_ratios), 4)
        if extracted_prefix_ratios
        else 0.0
    )
    mean_comp_f1 = (
        round(sum(comp_f1_overall) / len(comp_f1_overall), 4)
        if comp_f1_overall
        else 0.0
    )

    summary = {
        "mode": mode,
        "model_dir": model_dir,
        "device": device,
        "dtype": str(dtype),
        "total_questions": total_n,
        "execution_accuracy": round(float(llmsql_report.get("accuracy", 0.0)), 4),
        "execution_matches": int(llmsql_report.get("matches", 0)),
        "sqlite_executable_count": exec_stats["sqlite_executable_count"],
        "sqlite_executable_rate": exec_stats["sqlite_executable_rate"],
        "no_sql_extracted_count": exec_stats["no_sql_extracted_count"],
        "sql_errors": int(llmsql_report.get("sql_errors", 0)),
        "pred_none": int(llmsql_report.get("pred_none", 0)),
        "exact_string_match_rate": round(exact_match_count / total_n, 4) if total_n else 0.0,
        "raw_cfg_compliance_count": raw_cfg_valid,
        "raw_cfg_compliance_rate": round(raw_cfg_valid / total_n, 4) if total_n else 0.0,
        "mean_raw_cfg_prefix_ratio": mean_raw_prefix,
        "extracted_sql_cfg_compliance_count": extracted_cfg_valid,
        "extracted_sql_cfg_compliance_rate": round(extracted_cfg_valid / total_n, 4) if total_n else 0.0,
        "mean_extracted_cfg_prefix_ratio": mean_ext_prefix,
        "mean_sql_component_f1": mean_comp_f1,
        "sql_component_f1_breakdown": comp_f1_breakdown,
        "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else 0.0,
        "category_breakdown": cat_stats,
        "grammar_meta": grammar_meta,
        "safetensors_bytes": out_bytes,
        "safetensors_delta_vs_input_bytes": size_delta,
        "predictions_file": str(pred_file),
        "llmsql_report_file": str(report_file),
    }

    logger.write(f"\n📊 [{mode.upper()}] Summary:\n")
    logger.write(f"  • LLMSQL Execution Accuracy : {summary['execution_matches']}/{total_n} ({summary['execution_accuracy'] * 100:.1f}%)\n")
    logger.write(f"  • SQLite Executable Rate    : {summary['sqlite_executable_count']}/{total_n} ({summary['sqlite_executable_rate'] * 100:.1f}%)\n")
    logger.write(f"  • SQL Clause/Component F1   : {mean_comp_f1 * 100:.1f}%\n")
    logger.write(f"  • Raw Output SQL:2016 CFG   : {raw_cfg_valid}/{total_n} ({summary['raw_cfg_compliance_rate'] * 100:.1f}%) | Mean Valid Prefix: {mean_raw_prefix * 100:.1f}%\n")
    logger.write(f"  • Extracted SQL:2016 CFG    : {extracted_cfg_valid}/{total_n} ({summary['extracted_sql_cfg_compliance_rate'] * 100:.1f}%) | Mean Valid Prefix: {mean_ext_prefix * 100:.1f}%\n")
    logger.write(f"  • SQLite Execution Errors   : {summary['sql_errors']}\n")
    logger.write(f"  • Mean Decode Latency       : {summary['mean_latency_ms']:.1f} ms\n")
    if size_delta is not None:
        logger.write(f"  • Safetensors Byte Delta    : {size_delta} bytes\n")
    return summary


def ensure_benchmark_file(repo_id: str, filename: str, local_dir: Path) -> str:
    """Return cached benchmark file path under `local_dir`, downloading from HF Hub if absent."""
    from llmsql.evaluation.evaluate import download_benchmark_file

    local_dir.mkdir(parents=True, exist_ok=True)
    cached = local_dir / filename
    if cached.is_file() and cached.stat().st_size > 0:
        return str(cached)
    return download_benchmark_file(repo_id, filename, local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/Qwen2.5-Coder-0.5B-Instruct checkpoints on LLMSQL + ISO SQL:2016 CFG compliance."
    )
    parser.add_argument("--model-dir", default="models/output", help="Local checkpoint path or Hugging Face repo ID.")
    parser.add_argument("--baseline-dir", default="models/input", help="Path to clean baseline HF checkpoint.")
    parser.add_argument("--reference-model", default=None, help="Optional reference HF model ID or directory (e.g. defog/sqlcoder-7b-2).")
    parser.add_argument("--mode", choices=["unwrapped", "outlines", "both"], default="unwrapped")
    parser.add_argument("--prompt-style", choices=["raw-sql", "llmsql", "sqlcoder"], default="raw-sql")
    parser.add_argument("--version", choices=["2.0", "1.0"], default="2.0")
    parser.add_argument("--limit", type=int, default=50, help="Stratified sample count (0 = full dataset).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--ebnf-file", default=DEFAULT_EBNF_PATH)
    parser.add_argument("--workdir", default=str(SCRIPT_DIR / ".cache" / "llmsql"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs" / "llmsql-eval"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test(Path(args.ebnf_file).resolve())
        sys.exit(0 if ok else 1)

    resolved_model = resolve_model_dir(args.model_dir, allow_missing=args.allow_missing)
    if resolved_model is None:
        sys.exit(0)

    resolved_baseline = resolve_model_dir(args.baseline_dir, allow_missing=True)
    resolved_ref = (
        resolve_model_dir(args.reference_model, allow_missing=args.allow_missing)
        if args.reference_model
        else None
    )

    from llmsql.config.config import get_repo_id
    from llmsql.utils.utils import load_jsonl

    workdir = Path(args.workdir).resolve()
    version_workdir = workdir / f"v{args.version}"
    version_workdir.mkdir(parents=True, exist_ok=True)
    repo_id = get_repo_id(args.version)

    questions_path = ensure_benchmark_file(repo_id, "questions.jsonl", version_workdir)
    db_path = ensure_benchmark_file(repo_id, "sqlite_tables.db", version_workdir)
    tables_path = ensure_benchmark_file(repo_id, "tables.jsonl", version_workdir)
    all_questions = load_jsonl(questions_path)
    tables_by_id = {t["table_id"]: t for t in load_jsonl(tables_path)}

    sampled_questions = select_stratified_questions(all_questions, args.limit, args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = DualLogger(output_dir / "run.log")

    modes = ["unwrapped", "outlines"] if args.mode == "both" else [args.mode]
    results: dict[str, Any] = {
        "benchmark": f"llmsql-{args.version}",
        "repo_id": repo_id,
        "prompt_style": args.prompt_style,
        "sampled_questions": len(sampled_questions),
        "total_benchmark_questions": len(all_questions),
        "seed": args.seed,
        "ebnf_file": str(Path(args.ebnf_file).resolve()),
        "modes": {},
    }

    try:
        for mode in modes:
            results["modes"][mode] = evaluate_mode(
                mode=mode,
                model_dir=resolved_model,
                baseline_dir=resolved_baseline,
                questions=sampled_questions,
                tables_by_id=tables_by_id,
                questions_path=questions_path,
                db_path=db_path,
                ebnf_path=str(Path(args.ebnf_file).resolve()),
                version=args.version,
                workdir=version_workdir,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                prompt_style=args.prompt_style,
                logger=logger,
            )

        if resolved_ref is not None:
            ref_summary = evaluate_mode(
                mode="unwrapped",
                model_dir=resolved_ref,
                baseline_dir=resolved_baseline,
                questions=sampled_questions,
                tables_by_id=tables_by_id,
                questions_path=questions_path,
                db_path=db_path,
                ebnf_path=str(Path(args.ebnf_file).resolve()),
                version=args.version,
                workdir=version_workdir,
                output_dir=output_dir / "reference",
                max_new_tokens=args.max_new_tokens,
                prompt_style=args.prompt_style,
                logger=logger,
            )
            results["reference_model"] = ref_summary
            primary_summary = results["modes"].get("unwrapped") or next(iter(results["modes"].values()))
            logger.write(
                f"\n⚖️  Primary vs. Reference ({resolved_ref}): "
                f"Exec Acc {primary_summary['execution_accuracy'] * 100:.1f}% vs {ref_summary['execution_accuracy'] * 100:.1f}% | "
                f"Exec SQL {primary_summary['sqlite_executable_rate'] * 100:.1f}% vs {ref_summary['sqlite_executable_rate'] * 100:.1f}% | "
                f"Comp F1 {primary_summary['mean_sql_component_f1'] * 100:.1f}% vs {ref_summary['mean_sql_component_f1'] * 100:.1f}% | "
                f"Raw CFG {primary_summary['raw_cfg_compliance_rate'] * 100:.1f}% vs {ref_summary['raw_cfg_compliance_rate'] * 100:.1f}%\n"
            )
    finally:
        logger.close()

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n✅ Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()

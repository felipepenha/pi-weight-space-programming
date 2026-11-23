#!/usr/bin/env python3
"""Single-prompt smoke tester for unwrapped vs. Outlines SQL:2016 generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

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
        is_sql_grammar_valid,
    )
except ModuleNotFoundError:
    from sql_grammar import (
        DEFAULT_EBNF_PATH,
        build_llg_matcher_spec,
        is_sql_grammar_valid,
    )


DEFAULT_PROMPT = (
    "You are an expert SQLite query writer. Given the database schema with a few sample rows "
    "and a question, write a single SQLite query that answers the question.\n\n"
    'CREATE TABLE "employees" ("id" INTEGER, "name" TEXT, "department" TEXT, "salary" INTEGER);\n'
    "-- first 2 rows:\n"
    '[1, "Alice", "Engineering", 145000]\n'
    '[2, "Bob", "Sales", 95000]\n\n'
    "Question: List the names of employees in Engineering ordered by salary descending.\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-prompt SQL generation and CFG validation.")
    parser.add_argument("--model-dir", default="models/output")
    parser.add_argument("--mode", choices=["unwrapped", "outlines"], default="unwrapped")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ebnf-file", default=DEFAULT_EBNF_PATH)
    args = parser.parse_args()

    import outlines
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    expanded = Path(args.model_dir).expanduser()
    if expanded.exists() or args.model_dir.startswith(("/", "./", "../", "~", "models/")):
        model_dir = str(expanded.resolve())
    else:
        model_dir = args.model_dir
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

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

    llg_tok, g_spec, lark_text, _ = build_llg_matcher_spec(tokenizer, args.ebnf_file)
    t0 = time.perf_counter()
    if args.mode == "outlines":
        om = outlines.from_transformers(model, tokenizer)
        om.type_adapter.has_chat_template = False
        gen = outlines.Generator(om, outlines.cfg(lark_text))
        output = str(gen(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False)).strip()
        trimmed_ids = tokenizer.encode(output, add_special_tokens=False)
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
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
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
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
        output = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    valid, err = is_sql_grammar_valid(llg_tok, g_spec, tokenizer, output, token_ids=trimmed_ids)
    print(
        json.dumps(
            {
                "model_dir": model_dir,
                "mode": args.mode,
                "device": device,
                "latency_ms": latency_ms,
                "sql_2016_cfg_valid": valid,
                "cfg_error": err,
                "output": output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

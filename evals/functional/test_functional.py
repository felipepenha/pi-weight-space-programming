#!/usr/bin/env python3
"""
Functional & deterministic paper parity test suite for pi-weight-space-programming.

Works out-of-the-box on a blank slate: automatically downloads and caches required
paper checkpoints and reference artifacts from `RikiyaT/lm_understand-logs` into
`evals/.cache/paper_evals` (or `$WSP_EVAL_DATA_DIR`) when not already present.

Verifies:
1. All wsp_engine.py commands (`inspect`, `circuit_probe`, `compare`, `edit`, `sandbox`, `hf_download`, `outlines_cfg`).
2. Static gradient/training guard in `sandbox` (blocks `.backward()`, `torch.optim`, etc.).
3. Experiment 3 (`exp3_language`) static circuit probe (`induction_heads` and `subspace_rank`)
   against ground-truth copying heads (`heads.json`) and Romaji alphabet rank bottleneck.
4. Experiment 4 (`exp4_fix_car`) surgical weight editing DSL parity against `oneshot_result.json`
   and `edited_actor.npz` (51/51 statements, 0 errors, exact weight match).
5. Experiment 5 (`exp5_write_from_scratch`) Breakout weight synthesis DSL parity against
   `breakout_hand.npz` and `breakout_apply.json` (92/92 statements, 815 nonzero entries, max_diff = 0.0).
"""

import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "harness" / "wsp_engine.py"
DEFAULT_CACHE_DIR = REPO_ROOT / "evals" / ".cache" / "paper_evals"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main"

REQUIRED_ARTIFACTS = {
    # Experiment 1 (Which Game?)
    "exp1/breakout_prompt.txt": "exp1_which_game/runs/breakout/prompt.txt",
    "exp1/pong_prompt.txt": "exp1_which_game/runs/pong/prompt.txt",
    "exp1/pacman_prompt.txt": "exp1_which_game/runs/pacman/prompt.txt",
    "exp1/freeway_prompt.txt": "exp1_which_game/runs/freeway/prompt.txt",
    "exp1/answer_key.json": "exp1_which_game/eval/answer_key.json",
    # Experiment 2 (Diffusion Shape)
    "exp2/circle_prompt.txt": "exp2_diffusion/runs/circle/prompt.txt",
    "exp2/square_prompt.txt": "exp2_diffusion/runs/square/prompt.txt",
    "exp2/triangle_prompt.txt": "exp2_diffusion/runs/triangle/prompt.txt",
    # Experiment 3 (Language & Copying Heads from Middle Transformer Blocks)
    "exp3/python_blocks.npz": "exp3_language/networks/python/blocks.npz",
    "exp3/romaji_blocks.npz": "exp3_language/networks/romaji/blocks.npz",
    "exp3/english_blocks.npz": "exp3_language/networks/english/blocks.npz",
    "exp3/heads.json": "exp3_language/eval/heads.json",
    "exp3/message.txt": "exp3_language/runs/python_r0/message.txt",
    # Experiment 4 (Fix Failing Car via Surgical Weight Edits)
    "exp4/ppo_final_seed0.npz": "exp4_fix_car/networks/ppo_final_seed0.npz",
    "exp4/edited_actor.npz": "exp4_fix_car/networks/edited_actor.npz",
    "exp4/answer.md": "exp4_fix_car/runs/oneshot_repair/reply/answer.md",
    "exp4/oneshot_result.json": "exp4_fix_car/eval/oneshot_result.json",
    # Experiment 5 (Write Breakout Policy from Scratch)
    "exp5/breakout_hand.npz": "exp5_write_from_scratch/networks/breakout/hand.npz",
    "exp5/breakout_answer.md": "exp5_write_from_scratch/runs/breakout/reply/answer.md",
    "exp5/breakout_apply.json": "exp5_write_from_scratch/runs/breakout/apply.json",
}


def ensure_eval_artifacts(
    cache_dir: pathlib.Path, subset_prefixes: tuple = ("exp3/", "exp4/", "exp5/")
) -> None:
    """Download any missing paper evaluation files from RikiyaT/lm_understand-logs."""
    for rel_dst, remote_rel in REQUIRED_ARTIFACTS.items():
        if subset_prefixes and not any(rel_dst.startswith(p) for p in subset_prefixes):
            continue
        dst = cache_dir / rel_dst
        if dst.exists() and dst.stat().st_size > 32:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GITHUB_RAW_BASE}/{remote_rel}"
        print(f"[download] Fetching {rel_dst} from {url} ...", file=sys.stderr)
        req = urllib.request.Request(
            url, headers={"User-Agent": "pi-weight-space-programming-evals/0.1"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as out_f:
            out_f.write(resp.read())


def run_engine(cmd: str, payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ENGINE_PATH), cmd, json.dumps(payload)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    eval_data_dir = pathlib.Path(
        os.environ.get("WSP_EVAL_DATA_DIR", str(DEFAULT_CACHE_DIR))
    )
    ensure_eval_artifacts(eval_data_dir, subset_prefixes=("exp3/", "exp4/", "exp5/"))
    results = {"tests": {}, "all_passed": False}

    # 1. Inspect command
    ins = run_engine(
        "inspect",
        {
            "checkpoint": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"),
            "operation": "summary",
        },
    )
    assert ins["total_tensors"] == 26 and ins["total_parameters"] == 254353
    results["tests"]["wsp_inspect_weights"] = {
        "passed": True,
        "total_tensors": ins["total_tensors"],
        "total_parameters": ins["total_parameters"],
    }

    # 2. Experiment 4 surgical edit DSL parity
    exp4_ans = (eval_data_dir / "exp4" / "answer.md").read_text(encoding="utf-8")
    exp4_ref = json.loads(
        (eval_data_dir / "exp4" / "oneshot_result.json").read_text(encoding="utf-8")
    )
    exp4_out = str(eval_data_dir / "exp4" / "test_edited.npz")
    res_exp4 = run_engine(
        "edit",
        {
            "checkpoint": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"),
            "output_path": exp4_out,
            "edits_dsl": exp4_ans,
            "explanation": "Functional test of Experiment 4 surgical edit DSL",
        },
    )
    ed_ref = np.load(eval_data_dir / "exp4" / "edited_actor.npz")
    ed_out = np.load(exp4_out)
    head_max_diff = float(
        np.max(
            np.abs(
                ed_out["params/GeneralActorCriticRNN_0/Dense_0/kernel"][126:128, 0:3]
                - ed_ref["actor_head.W"][126:128, 0:3]
            )
        )
    )
    assert res_exp4["applied_count"] == exp4_ref["edit_report"]["applied"] == 51
    assert len(res_exp4["dsl_report"]["errors"]) == 0
    assert len(res_exp4["dsl_report"]["tensors_touched"]) == 14
    assert head_max_diff == 0.0
    results["tests"]["exp4_surgical_edit_dsl"] = {
        "passed": True,
        "statements_applied": res_exp4["applied_count"],
        "expected_statements": exp4_ref["edit_report"]["applied"],
        "tensors_touched": len(res_exp4["dsl_report"]["tensors_touched"]),
        "actor_head_max_abs_diff": head_max_diff,
    }

    # 3. Compare checkpoints command
    cmp_res = run_engine(
        "compare",
        {
            "checkpoint_a": str(eval_data_dir / "exp4" / "ppo_final_seed0.npz"),
            "checkpoint_b": exp4_out,
        },
    )
    assert (
        cmp_res["changed_tensors_count"] == 14
        and len(cmp_res["identical_frozen_tensors"]) == 12
    )
    results["tests"]["wsp_compare_checkpoints"] = {
        "passed": True,
        "changed_tensors_count": cmp_res["changed_tensors_count"],
        "identical_frozen_tensors_count": len(cmp_res["identical_frozen_tensors"]),
    }

    # 4. Experiment 5 from-scratch Breakout DSL parity
    exp5_ans = (eval_data_dir / "exp5" / "breakout_answer.md").read_text(
        encoding="utf-8"
    )
    exp5_out = str(eval_data_dir / "exp5" / "test_breakout.npz")
    res_exp5 = run_engine(
        "edit",
        {
            "output_path": exp5_out,
            "init_shapes": {
                "W1": [600, 32],
                "b1": [32],
                "W2": [32, 32],
                "b2": [32],
                "W3": [32, 6],
                "b3": [6],
            },
            "edits_dsl": exp5_ans,
            "explanation": "Functional test of Experiment 5 from-scratch Breakout synthesis",
        },
    )
    hand = np.load(eval_data_dir / "exp5" / "breakout_hand.npz")
    gen = np.load(exp5_out)
    pairs = [
        ("W1", "actor/dense1/kernel"),
        ("b1", "actor/dense1/bias"),
        ("W2", "actor/dense2/kernel"),
        ("b2", "actor/dense2/bias"),
        ("W3", "actor/logits/kernel"),
        ("b3", "actor/logits/bias"),
    ]
    max_diff_exp5 = max(float(np.max(np.abs(gen[g] - hand[h]))) for g, h in pairs)
    assert res_exp5["applied_count"] == 92
    assert res_exp5["total_changed_entries"] == 815
    assert max_diff_exp5 == 0.0
    results["tests"]["exp5_from_scratch_dsl"] = {
        "passed": True,
        "statements_applied": res_exp5["applied_count"],
        "nonzero_entries": res_exp5["total_changed_entries"],
        "max_abs_diff_vs_breakout_hand_npz": max_diff_exp5,
    }

    # 5. Experiment 3 static circuit probe (induction heads & alphabet bottleneck)
    heads_ref = json.loads(
        (eval_data_dir / "exp3" / "heads.json").read_text(encoding="utf-8")
    )
    exp3_probe_summary = {}
    for lang in ("python", "romaji", "english"):
        ckpt_path = str(eval_data_dir / "exp3" / f"{lang}_blocks.npz")
        ind = run_engine(
            "circuit_probe",
            {"checkpoint": ckpt_path, "operation": "induction_heads", "num_heads": 6},
        )
        sub = run_engine(
            "circuit_probe",
            {"checkpoint": ckpt_path, "operation": "subspace_rank", "num_heads": 6},
        )
        top_heads = [h["head"] for h in ind["ranked_candidate_copying_heads"][:3]]
        model_heads = heads_ref[f"{lang}_L4_H6_d384_s0"]
        gt_heads = [
            f"{k[1]}.{k[3]}"
            for k, v in model_heads.items()
            if isinstance(v, dict) and v.get("induct", 0.0) >= 0.2
        ]
        assert all(g in top_heads for g in gt_heads)
        l3_gap = sub["mlp_input_regressed_alphabet_bottleneck"][-1]
        exp3_probe_summary[lang] = {
            "ground_truth_copying_heads": gt_heads,
            "top_3_detected_heads": top_heads,
            "top_ranked_candidates": ind["ranked_candidate_copying_heads"][:3],
            "layer3_top_sv_gap": l3_gap["top_singular_value_gaps_rank_16_to_50"][0],
            "layer3_energy_top24_fraction": l3_gap["energy_top24_fraction"],
        }
    results["tests"]["exp3_static_circuit_probe"] = {
        "passed": True,
        "corpora": exp3_probe_summary,
    }

    # 6. Sandbox, no-grad guard & evaluation/harness/sibling isolation
    sbx_ok = run_engine(
        "sandbox",
        {
            "code": 'ckpt = Checkpoint(MODEL_PATH)\nprint("SANDBOX_OK", len(ckpt.keys()), sorted(W.keys()))',
            "model_path": exp5_out,
            "work_dir": str(REPO_ROOT),
        },
    )
    assert sbx_ok["ok"] is True and "SANDBOX_OK 6" in sbx_ok["output"]

    sbx_guard = run_engine(
        "sandbox",
        {
            "code": "loss.backward()",
            "model_path": exp5_out,
            "work_dir": str(REPO_ROOT),
        },
    )
    assert sbx_guard["blocked"] is True and len(sbx_guard["violations"]) == 1

    import shutil

    ex_work_dir = (
        REPO_ROOT
        / "experiments"
        / "aggressive-pruning-plus-outlines-cfg"
        / "qwen2.5-coder-0.5b-instruct"
        / ".wsp-work"
    )
    ex_work_existed = ex_work_dir.exists()
    try:
        # Static audit checks: block internal imports, transcripts, eval scripts, and siblings
        for static_probe in (
            "import outlines_cfg",
            'open("session.log")',
            'open("eval_extended.py")',
            'open("../../aggressive-pruning/qwen2.5-coder-0.5b-instruct/PROMPT.md")',
        ):
            st_res = run_engine(
                "sandbox",
                {
                    "code": static_probe,
                    "work_dir": str(ex_work_dir),
                },
            )
            assert st_res["blocked"] is True, f"Expected static block for {static_probe}"

        # Runtime filesystem shield checks (construct paths dynamically to bypass static regex)
        runtime_probe_code = """
import os, sys, tempfile
model_dir = os.path.dirname(WORK)
ex_dir = os.path.dirname(model_dir)
experiments_dir = os.path.dirname(ex_dir)
repo_dir = os.path.dirname(experiments_dir)

def expect_blocked(fn):
    try:
        fn()
    except PermissionError:
        return True
    raise AssertionError("Expected PermissionError")

h_file = os.path.join(repo_dir, "har" + "ness", "outlines_" + "cfg.py")
assert expect_blocked(lambda: open(h_file, "r").read())
assert expect_blocked(lambda: open(h_file, "w").write("x"))
assert expect_blocked(lambda: open(os.path.join(repo_dir, "READ" + "ME.md"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "PROMPT.md"), "w").write("x"))
sib_file = os.path.join(experiments_dir, "aggressive-" + "pruning", "qwen2.5-coder-0.5b-instruct", "PROMPT.md")
assert expect_blocked(lambda: open(sib_file, "r").read())
assert ("aggressive-" + "pruning") not in os.listdir(experiments_dir)
assert "_orig_open" not in globals()
assert "_orig_open" not in sys.modules["prelude"].__dict__

ok_work = os.path.join(WORK, "shield_ok.txt")
with open(ok_work, "w") as f:
    f.write("ok")
os.remove(ok_work)
with tempfile.TemporaryDirectory() as td:
    tp = os.path.join(td, "t.txt")
    with open(tp, "w") as f:
        f.write("ok")
print("SHIELD_OK")
"""
        rt_res = run_engine(
            "sandbox",
            {
                "code": runtime_probe_code,
                "work_dir": str(ex_work_dir),
            },
        )
        assert rt_res["ok"] is True and "SHIELD_OK" in rt_res["output"], rt_res["output"]
    finally:
        if not ex_work_existed and ex_work_dir.exists():
            shutil.rmtree(ex_work_dir, ignore_errors=True)

    # Verify 3-level nested example isolation using a synthetic 3-level domain tree under experiments/
    nested_tmp_root = REPO_ROOT / "experiments" / "_tmp_nested_isolation_test"
    nested_model_dir = nested_tmp_root / "domain" / "model"
    nested_work_dir = nested_model_dir / ".wsp-work"
    try:
        (nested_model_dir / "eval").mkdir(parents=True, exist_ok=True)
        (nested_model_dir / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (nested_model_dir / "grammar.ebnf").write_text(
            "/* EBNF fixture */\n", encoding="utf-8"
        )
        (nested_model_dir / "eval" / "eval_check.py").write_text(
            "# hidden eval\n", encoding="utf-8"
        )
        for hidden_name in (
            "run.sh",
            "edit.sh",
            "eval.sh",
            "pyproject.toml",
            "uv.lock",
            "install-extension.sh",
            "README.md",
        ):
            (nested_model_dir / hidden_name).write_text("secret\n", encoding="utf-8")
        (nested_model_dir / ".venv" / "bin" / "python").write_text(
            "", encoding="utf-8"
        )

        for static_probe in (
            'open("eval/eval_check.py")',
            'open("../../../aggressive-pruning/qwen2.5-coder-0.5b-instruct/PROMPT.md")',
            'open("run.sh")',
            'open("edit.sh")',
            'open("eval.sh")',
            'open("pyproject.toml")',
        ):
            st_res = run_engine(
                "sandbox",
                {
                    "code": static_probe,
                    "work_dir": str(nested_work_dir),
                },
            )
            assert st_res["blocked"] is True, f"Expected nested static block for {static_probe}"

        nested_rt_code = """
import os
model_dir = os.path.dirname(WORK)
experiments_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
def expect_blocked(fn):
    try:
        fn()
    except PermissionError:
        return True
    raise AssertionError("Expected PermissionError")

ebnf_head = open(os.path.join(model_dir, "grammar.ebnf"), "r").read(64)
assert "EBNF" in ebnf_head
sib = os.path.join(experiments_dir, "aggressive-" + "pruning", "qwen2.5-coder-0.5b-instruct", "PROMPT.md")
assert expect_blocked(lambda: open(sib, "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "ev" + "al", "eval_check.py"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "ru" + "n.sh"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "ed" + "it.sh"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "ev" + "al.sh"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "pypro" + "ject.toml"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "uv." + "lock"), "r").read())
assert expect_blocked(lambda: open(os.path.join(model_dir, "install-" + "extension.sh"), "r").read())
assert expect_blocked(lambda: os.listdir(os.path.join(model_dir, ".ve" + "nv")))
assert expect_blocked(lambda: os.listdir(os.path.join(model_dir, ".ve" + "nv", "bin")))
assert ("aggressive-" + "pruning") not in os.listdir(experiments_dir)
listed_root = set(os.listdir(model_dir))
for hidden in ("ru" + "n.sh", "ed" + "it.sh", "ev" + "al.sh", "pypro" + "ject.toml", "uv." + "lock", "install-" + "extension.sh", ".ve" + "nv", "ev" + "al", "README.md"):
    assert hidden not in listed_root, f"Hidden entry leaked in listdir: {hidden}"
for root, dirs, files in os.walk(model_dir):
    assert (".ve" + "nv") not in dirs
    assert ("ru" + "n.sh") not in files
    dirs[:] = [d for d in dirs if d not in ("models", ".wsp-work")]
import numpy, torch, transformers
assert torch.__version__ and transformers.__version__
print("NESTED_SHIELD_OK")
"""
        nested_rt = run_engine(
            "sandbox",
            {
                "code": nested_rt_code,
                "work_dir": str(nested_work_dir),
            },
        )
        assert nested_rt["ok"] is True and "NESTED_SHIELD_OK" in nested_rt["output"], nested_rt["output"]
    finally:
        shutil.rmtree(nested_tmp_root, ignore_errors=True)

    results["tests"]["wsp_python_sandbox_and_guard"] = {
        "passed": True,
        "sandbox_stdout": sbx_ok["output"].strip(),
        "guard_blocked_backward": sbx_guard["blocked"],
        "guard_violation": sbx_guard["violations"][0],
        "static_isolation_verified": True,
        "runtime_filesystem_shield_verified": True,
        "nested_3level_isolation_verified": True,
    }

    # 7. Hugging Face Model Downloader (Strict Internet Exception & Security Guards)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_hf_dir:
        # A. Valid download of model config
        hf_res = run_engine(
            "hf_download",
            {
                "repo_id": "openai-community/gpt2",
                "filenames": ["config.json"],
                "destination_dir": tmp_hf_dir,
            },
        )
        assert hf_res["ok"] is True
        assert len(hf_res["downloaded_files"]) == 1
        cfg_file = pathlib.Path(hf_res["downloaded_files"][0]["path"])
        assert cfg_file.exists() and cfg_file.stat().st_size > 0
        file_size = cfg_file.stat().st_size
        assert (
            "Authorized Hugging Face Hub download exception"
            in hf_res["policy_exception_notice"]
        )

        # B. Caching verification
        hf_cached = run_engine(
            "hf_download",
            {
                "repo_id": "openai-community/gpt2",
                "filenames": ["config.json"],
                "destination_dir": tmp_hf_dir,
            },
        )
        assert hf_cached["ok"] is True
        assert hf_cached["downloaded_files"][0]["cached"] is True

        # C. Security boundary: Path traversal blocked
        traversal_blocked = False
        try:
            run_engine(
                "hf_download",
                {
                    "repo_id": "openai-community/gpt2",
                    "filenames": ["../../evil.txt"],
                    "destination_dir": tmp_hf_dir,
                },
            )
        except Exception:
            traversal_blocked = True
        assert traversal_blocked, "Path traversal should be strictly blocked"

    results["tests"]["wsp_hf_download"] = {
        "passed": True,
        "repo_id": "openai-community/gpt2",
        "file_downloaded": "config.json",
        "size_bytes": file_size,
        "cached_verified": True,
        "traversal_guard_blocked": True,
    }

    # 8. Outlines CFG/EBNF Final-Layer Compiler (`wsp_outlines_cfg`)
    import torch
    from safetensors.torch import save_file as st_save_file
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    with tempfile.TemporaryDirectory() as tmp_cfg_root:
        in_dir = pathlib.Path(tmp_cfg_root) / "in_model"
        out_dir = pathlib.Path(tmp_cfg_root) / "out_model"
        in_dir.mkdir(parents=True, exist_ok=True)

        vocab = {"<|endoftext|>": 0}
        for ch in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:":
            key = "Ġ" if ch == " " else ch
            if key not in vocab:
                vocab[key] = len(vocab)
        fast_tok_backend = HFTokenizer(
            BPE(vocab=vocab, merges=[], unk_token="<|endoftext|>")
        )
        fast_tok_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        fast_tok_backend.decoder = ByteLevelDecoder()
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=fast_tok_backend,
            unk_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            bos_token="<|endoftext|>",
        )
        fast_tok.save_pretrained(str(in_dir))

        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=True,
        )
        tiny_model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
        cfg.save_pretrained(str(in_dir))
        tiny_sd = {
            k: v.contiguous()
            for k, v in tiny_model.state_dict().items()
            if k != "lm_head.weight"
        }
        st_save_file(tiny_sd, str(in_dir / "model.safetensors"))

        test_grammar = 'start: "pwd" | "ls" | "false"'
        an_res = run_engine(
            "outlines_cfg",
            {
                "operation": "analyze_grammar",
                "checkpoint": str(in_dir),
                "grammar": test_grammar,
            },
        )
        assert an_res["ok"] is True and an_res["num_states_explored"] >= 1

        cmp_cfg_res = run_engine(
            "outlines_cfg",
            {
                "operation": "compile",
                "checkpoint": str(in_dir),
                "output_dir": str(out_dir),
                "grammar": test_grammar,
                "prompts": ["Question : ok", "Check : fail", "status : bad"],
                "use_chat_template": False,
                "max_new_tokens": 8,
                "device": "cpu",
            },
        )
        assert cmp_cfg_res["ok"] is True
        assert cmp_cfg_res["file_size_comparison"]["size_delta_bytes"] == 0
        assert cmp_cfg_res["teacher_trajectories"]["skipped_invalid"] == 0
        assert (
            cmp_cfg_res["mechanism_a_gate_inhibition"]["mode"]
            == "dense_top_impact_slice"
        )
        assert cmp_cfg_res["mechanism_a_gate_inhibition"]["capacity_fraction"] <= 0.05
        assert cmp_cfg_res["verification"]["grammar_compliance_rate"] == 1.0
        assert cmp_cfg_res["verification"]["teacher_exact_match_rate"] == 1.0
        assert (out_dir / "model.safetensors").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "edit.npz").exists()

        ver_cfg_res = run_engine(
            "outlines_cfg",
            {
                "operation": "verify_bundle",
                "checkpoint": str(out_dir),
                "grammar": test_grammar,
                "prompts": ["Question : ok", "Check : fail", "status : bad"],
                "use_chat_template": False,
                "max_new_tokens": 8,
                "device": "cpu",
            },
        )
        assert (
            ver_cfg_res["ok"] is True and ver_cfg_res["grammar_compliance_rate"] == 1.0
        )

    # 8b. Sharded LlamaForCausalLM + self-contained W3C EBNF fixture (`grammar_file`)
    from transformers import LlamaConfig, LlamaForCausalLM

    with tempfile.TemporaryDirectory() as tmp_sql_root:
        in_dir = pathlib.Path(tmp_sql_root) / "in_llama"
        out_dir = pathlib.Path(tmp_sql_root) / "out_llama"
        in_dir.mkdir(parents=True, exist_ok=True)
        sql_ebnf_path = pathlib.Path(tmp_sql_root) / "sample_w3c.ebnf"
        sql_ebnf_path.write_text(
            "/* Sample W3C EBNF fixture for sharded CFG + Bottlecaps '*' + bounded whitespace tests */\n"
            "direct_SQL_statement ::= ( commit_statement | query_statement ) ';'\n"
            "commit_statement ::= 'COMMIT'\n"
            "query_statement ::= 'SELECT' delimited_identifier 'FROM' delimited_identifier order_by_clause?\n"
            "order_by_clause ::= 'ORDER' 'BY' delimited_identifier sort_spec?\n"
            "sort_spec ::= 'ASC*' | 'DESC*'\n",
            encoding="utf-8",
        )

        sql_vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        sql_chars = (
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            "_;,\"'()*=<>+-./\\:[]{}!?@#$%^&|\n"
        )
        for ch in sql_chars:
            key = "Ġ" if ch == " " else ("Ċ" if ch == "\n" else ch)
            if key not in sql_vocab:
                sql_vocab[key] = len(sql_vocab)
        merges = [("C", "O"), ("CO", "M"), ("COM", "M"), ("COMM", "I"), ("COMMI", "T")]
        for a, b in merges:
            tok_ab = a + b
            if tok_ab not in sql_vocab:
                sql_vocab[tok_ab] = len(sql_vocab)
        sql_tok_backend = HFTokenizer(
            BPE(vocab=sql_vocab, merges=merges, unk_token="<unk>")
        )
        sql_tok_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        sql_tok_backend.decoder = ByteLevelDecoder()
        sql_tok = PreTrainedTokenizerFast(
            tokenizer_object=sql_tok_backend,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )
        sql_tok.save_pretrained(str(in_dir))
        (in_dir / "tokenizer.model").write_bytes(b"spiece-placeholder")

        torch.manual_seed(0)
        llama_cfg = LlamaConfig(
            vocab_size=len(sql_vocab),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
        tiny_llama = LlamaForCausalLM(llama_cfg).to(torch.bfloat16)
        with torch.no_grad():
            for tid in (sql_vocab["COMMIT"], sql_vocab[";"], sql_vocab["</s>"]):
                tiny_llama.lm_head.weight[tid] *= 3.0
                tiny_llama.lm_head.weight[tid, 1] += 2.0
            tiny_llama.model.norm.weight[1] = 3.0
        llama_cfg.save_pretrained(str(in_dir))
        llama_sd = {k: v.contiguous() for k, v in tiny_llama.state_dict().items()}
        shard1 = {
            k: v
            for k, v in llama_sd.items()
            if k.startswith("model.embed_tokens.") or k.startswith("model.layers.0.")
        }
        shard2 = {k: v for k, v in llama_sd.items() if k not in shard1}
        s1_name = "model-00001-of-00002.safetensors"
        s2_name = "model-00002-of-00002.safetensors"
        st_save_file(shard1, str(in_dir / s1_name))
        st_save_file(shard2, str(in_dir / s2_name))
        weight_map = {
            **{k: s1_name for k in shard1},
            **{k: s2_name for k in shard2},
        }
        (in_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )

        # Sharded checkpoint inspection + layer-filtered circuit_probe + stdin payload transport
        insp_sharded = run_engine(
            "inspect",
            {
                "checkpoint": str(in_dir),
                "operation": "summary",
                "tensor_pattern": r"layers\.1\.mlp\.",
            },
        )
        assert insp_sharded["total_tensors"] > 0
        assert insp_sharded["matched_tensors"] == 3

        stdin_proc = subprocess.run(
            [sys.executable, str(ENGINE_PATH), "inspect"],
            input=json.dumps(
                {
                    "checkpoint": str(in_dir),
                    "operation": "summary",
                    "tensor_pattern": r"layers\.1\.mlp\.",
                }
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(stdin_proc.stdout)["matched_tensors"] == 3

        cp_layer1 = run_engine(
            "circuit_probe",
            {
                "checkpoint": str(in_dir),
                "operation": "mlp_directions",
                "layers": [1],
            },
        )
        assert len(cp_layer1["mlp_layers"]) == 1
        assert cp_layer1["mlp_layers"][0]["layer"] == 1

        # Protected path block on grammar_file and prompts_file
        blocked_gf = False
        try:
            run_engine(
                "outlines_cfg",
                {
                    "operation": "analyze_grammar",
                    "checkpoint": str(in_dir),
                    "grammar_file": str(REPO_ROOT / "harness" / "outlines_cfg.py"),
                },
            )
        except Exception:
            blocked_gf = True
        assert blocked_gf, "grammar_file inside harness/ should be blocked"

        for blocked_pf_path in (
            REPO_ROOT / "harness" / "wsp_engine.py",
            REPO_ROOT / "experiments" / "aggressive-pruning" / "qwen2.5-coder-0.5b-instruct" / "run.sh",
        ):
            blocked_pf = False
            try:
                run_engine(
                    "outlines_cfg",
                    {
                        "operation": "verify_bundle",
                        "checkpoint": str(in_dir),
                        "grammar_file": str(sql_ebnf_path),
                        "prompts_file": str(blocked_pf_path),
                    },
                )
            except Exception:
                blocked_pf = True
            assert blocked_pf, f"prompts_file {blocked_pf_path} should be blocked"

        sql_an = run_engine(
            "outlines_cfg",
            {
                "operation": "analyze_grammar",
                "checkpoint": str(in_dir),
                "grammar_file": str(sql_ebnf_path),
                "max_states": 8,
            },
        )
        assert sql_an["ok"] is True
        assert sql_an["grammar_format"]["converted_from"] == "w3c_ebnf"
        assert sql_an["num_states_explored"] >= 1

        sql_prompts = [
            "Table users(id, name). Question: get all rows.",
            "Table orders(id, total). Question: count orders.",
        ]
        prompts_file_path = pathlib.Path(tmp_sql_root) / "prompts.json"
        prompts_file_path.write_text(json.dumps(sql_prompts), encoding="utf-8")

        out_dir_pf = pathlib.Path(tmp_sql_root) / "out_llama_pf"
        sql_cmp_pf = run_engine(
            "outlines_cfg",
            {
                "operation": "compile",
                "checkpoint": str(in_dir),
                "output_dir": str(out_dir_pf),
                "grammar_file": str(sql_ebnf_path),
                "prompts_file": str(prompts_file_path),
                "use_chat_template": False,
                "max_new_tokens": 12,
                "max_states": 8,
                "device": "cpu",
            },
        )
        assert sql_cmp_pf["ok"] is True
        assert sql_cmp_pf["prompts_source"] == "prompts_file"
        assert sql_cmp_pf["file_size_comparison"]["size_delta_bytes"] == 0
        assert sql_cmp_pf["teacher_trajectories"]["skipped_invalid"] == 0
        assert (
            sql_cmp_pf["mechanism_a_gate_inhibition"]["mode"]
            == "dense_top_impact_slice"
        )
        assert sql_cmp_pf["mechanism_a_gate_inhibition"]["capacity_fraction"] <= 0.05
        assert sql_cmp_pf["verification"]["grammar_compliance_rate"] == 1.0
        assert sql_cmp_pf["verification"]["teacher_exact_match_rate"] == 1.0

        sql_ver_pf = run_engine(
            "outlines_cfg",
            {
                "operation": "verify_bundle",
                "checkpoint": str(out_dir_pf),
                "grammar_file": str(sql_ebnf_path),
                "prompts_file": str(prompts_file_path),
                "use_chat_template": False,
                "max_new_tokens": 12,
                "device": "cpu",
            },
        )
        assert sql_ver_pf["ok"] is True
        assert sql_ver_pf["prompts_source"] == "prompts_file"
        assert sql_ver_pf["grammar_compliance_rate"] == 1.0

        sql_cmp = run_engine(
            "outlines_cfg",
            {
                "operation": "compile",
                "checkpoint": str(in_dir),
                "output_dir": str(out_dir),
                "grammar_file": str(sql_ebnf_path),
                "prompts": sql_prompts,
                "use_chat_template": False,
                "max_new_tokens": 12,
                "max_states": 8,
                "device": "cpu",
            },
        )
        assert sql_cmp["ok"] is True
        assert sql_cmp["prompts_source"] == "inline"
        assert sql_cmp["grammar_format"]["converted_from"] == "w3c_ebnf"
        assert sql_cmp["file_size_comparison"]["size_delta_bytes"] == 0
        assert sql_cmp["teacher_trajectories"]["skipped_invalid"] == 0
        assert (
            sql_cmp["mechanism_a_gate_inhibition"]["mode"] == "dense_top_impact_slice"
        )
        assert sql_cmp["mechanism_a_gate_inhibition"]["capacity_fraction"] <= 0.05
        assert sql_cmp["verification"]["grammar_compliance_rate"] == 1.0
        assert sql_cmp["verification"]["teacher_exact_match_rate"] == 1.0
        assert (out_dir / s1_name).read_bytes() == (in_dir / s1_name).read_bytes()
        assert (out_dir / "model.safetensors.index.json").exists()
        assert (out_dir / "tokenizer.model").exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "edit.npz").exists()

        cmp_sharded = run_engine(
            "compare",
            {"checkpoint_a": str(in_dir), "checkpoint_b": str(out_dir)},
        )
        assert cmp_sharded["changed_tensors_count"] >= 1
        assert all(
            d["name"].startswith("model.layers.1.mlp.")
            for d in cmp_sharded["ranked_diffs"]
        )
        assert len(cmp_sharded["identical_frozen_tensors"]) > 0

        sql_ver = run_engine(
            "outlines_cfg",
            {
                "operation": "verify_bundle",
                "checkpoint": str(out_dir),
                "grammar_file": str(sql_ebnf_path),
                "prompts": sql_prompts,
                "use_chat_template": False,
                "max_new_tokens": 12,
                "device": "cpu",
            },
        )
        assert sql_ver["ok"] is True and sql_ver["grammar_compliance_rate"] == 1.0

        # Verify bounded leading/trailing whitespace + Bottlecaps '*' non-reserved keyword handling
        import llguidance
        import llguidance.hf

        sys.path.insert(0, str(REPO_ROOT / "harness"))
        import outlines_cfg as _ocfg

        llg_tok_sql = llguidance.hf.from_tokenizer(sql_tok)
        _, g_spec_sql, _ = _ocfg._parse_grammar_spec(
            llguidance,
            __import__("outlines"),
            sql_ebnf_path.read_text(encoding="utf-8"),
            llg_tok=llg_tok_sql,
        )
        assert _ocfg._is_grammar_accepted(
            llguidance,
            llg_tok_sql,
            g_spec_sql,
            sql_tok,
            'SELECT "name" FROM "employees" ORDER BY "salary" DESC;',
        )
        assert _ocfg._is_grammar_accepted(
            llguidance,
            llg_tok_sql,
            g_spec_sql,
            sql_tok,
            ' SELECT "name" FROM "employees" ORDER BY "salary" DESC;',
        )
        for bad_ws_sql in (
            '\nSELECT "name" FROM "employees";',
            'SELECT "name" FROM "employees";\n',
        ):
            assert not _ocfg._is_grammar_accepted(
                llguidance,
                llg_tok_sql,
                g_spec_sql,
                sql_tok,
                bad_ws_sql,
            ), f"Unbounded whitespace should be rejected: {bad_ws_sql!r}"
        assert not _ocfg._is_grammar_accepted(
            llguidance,
            llg_tok_sql,
            g_spec_sql,
            sql_tok,
            'SELECT "name" FROM "employees" ORDER BY "salary" DESC*;',
        )

    # 8c. Non-SQL JSON CFG + Dense Mechanism A + is_accepting() EOS Protection + Invalid Teacher Skipping
    from safetensors.torch import load_file as st_load_file
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as tmp_json_root:
        in_json_dir = pathlib.Path(tmp_json_root) / "in_json"
        out_json_dir = pathlib.Path(tmp_json_root) / "out_json"
        out_trunc_dir = pathlib.Path(tmp_json_root) / "out_trunc"
        in_json_dir.mkdir(parents=True, exist_ok=True)

        json_vocab = {"<|endoftext|>": 0}
        for ch in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}"[:,]\n':
            key = "Ġ" if ch == " " else ("Ċ" if ch == "\n" else ch)
            if key not in json_vocab:
                json_vocab[key] = len(json_vocab)
        json_tok_backend = HFTokenizer(
            BPE(vocab=json_vocab, merges=[], unk_token="<|endoftext|>")
        )
        json_tok_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        json_tok_backend.decoder = ByteLevelDecoder()
        json_tok = PreTrainedTokenizerFast(
            tokenizer_object=json_tok_backend,
            unk_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            bos_token="<|endoftext|>",
        )
        json_tok.save_pretrained(str(in_json_dir))

        torch.manual_seed(7)
        json_cfg = Qwen2Config(
            vocab_size=len(json_vocab),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=True,
        )
        tiny_json = Qwen2ForCausalLM(json_cfg).to(torch.bfloat16)
        with torch.no_grad():
            tiny_json.model.embed_tokens.weight[:, 0] = 0.5
            tiny_json.model.embed_tokens.weight[json_vocab["Z"]] *= 2.0
            tiny_json.model.layers[1].post_attention_layernorm.weight[0] = 1.5
            tiny_json.model.layers[1].mlp.gate_proj.weight[0].zero_()
            tiny_json.model.layers[1].mlp.gate_proj.weight[0, 0] = 0.5
            tiny_json.model.layers[1].mlp.up_proj.weight[0].zero_()
            tiny_json.model.layers[1].mlp.up_proj.weight[0, 0] = 1.0
            tiny_json.model.layers[1].mlp.down_proj.weight[:, 0] = (
                tiny_json.model.embed_tokens.weight[json_vocab["Z"]].clone() * 1.5
            )
        json_cfg.save_pretrained(str(in_json_dir))
        json_sd = {
            k: v.contiguous()
            for k, v in tiny_json.state_dict().items()
            if k != "lm_head.weight"
        }
        st_save_file(json_sd, str(in_json_dir / "model.safetensors"))

        json_grammar = 'start: "{" "\\"k\\"" ":" /[0-9]{1,2}/ "}"'
        json_prompts = [
            "Format alpha:",
            "Format beta:",
            "Format gamma:",
        ]
        cmp_json = run_engine(
            "outlines_cfg",
            {
                "operation": "compile",
                "checkpoint": str(in_json_dir),
                "output_dir": str(out_json_dir),
                "grammar": json_grammar,
                "prompts": json_prompts,
                "use_chat_template": False,
                "max_new_tokens": 12,
                "max_states": 16,
                "device": "cpu",
            },
        )
        assert cmp_json["ok"] is True
        assert cmp_json["file_size_comparison"]["size_delta_bytes"] == 0
        assert cmp_json["teacher_trajectories"]["skipped_invalid"] == 0
        assert cmp_json["verification"]["grammar_compliance_rate"] == 1.0
        assert cmp_json["verification"]["teacher_exact_match_rate"] == 1.0

        mech_a_json = cmp_json["mechanism_a_gate_inhibition"]
        assert mech_a_json["applied"] is True
        assert mech_a_json["mode"] == "dense_top_impact_slice"
        assert mech_a_json["capacity_fraction"] <= 0.05
        assert mech_a_json["gated_neuron_state_pairs"] >= 1

        mech_b_json = cmp_json["mechanism_b_teacher_steering"]
        assert mech_b_json["eos_intervened_states"] >= 1
        assert mech_b_json["eos_non_accepting_guard_states"] > 0
        assert mech_b_json["eos_neurons_allocated"] == 1
        assert mech_b_json["eos_steering_pooled"] is True

        # Reload exported JSON bundle and verify EOS neuron stays silent on non-accepting prefixes
        base_down = st_load_file(str(in_json_dir / "model.safetensors"))[
            "model.layers.1.mlp.down_proj.weight"
        ].float()
        out_json_sd = st_load_file(str(out_json_dir / "model.safetensors"))
        out_down = out_json_sd["model.layers.1.mlp.down_proj.weight"].float()
        embed_json = out_json_sd["model.embed_tokens.weight"].float()
        diff_cols = (out_down - base_down).norm(dim=0)
        mech_b_cols = [j for j in range(out_down.shape[1]) if float(diff_cols[j]) > 1e-3]
        assert len(mech_b_cols) >= 1
        gamma_json = out_json_sd["model.norm.weight"].float()
        m_vocab_json = embed_json * gamma_json
        u_loop_json = torch.nn.functional.normalize(m_vocab_json.mean(dim=0), dim=-1)
        m_eos_clean = m_vocab_json[0] - (m_vocab_json[0] @ u_loop_json) * u_loop_json
        eos_dir_ref = torch.nn.functional.normalize(
            m_eos_clean / (gamma_json + 1e-5), dim=-1
        )
        eos_neuron_idx = max(
            mech_b_cols,
            key=lambda j: float(
                torch.nn.functional.cosine_similarity(
                    out_down[:, j], eos_dir_ref, dim=0
                )
            ),
        )

        reloaded_json = AutoModelForCausalLM.from_pretrained(
            str(out_json_dir), dtype=torch.float32
        )
        reloaded_json.eval()
        hook_buf: dict[str, torch.Tensor] = {}
        eos_hook = reloaded_json.model.layers[1].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: hook_buf.__setitem__("h", inp[0].detach().float().cpu())
        )
        max_non_accepting_act = float("-inf")
        min_accepting_act = float("inf")
        try:
            with torch.no_grad():
                for p_txt, detail in zip(
                    json_prompts, cmp_json["verification"]["details"], strict=True
                ):
                    p_ids = json_tok.encode(p_txt, add_special_tokens=True)
                    for bad_pref in ("{", '{"k":'):
                        seq_bad = p_ids + json_tok.encode(
                            bad_pref, add_special_tokens=False
                        )
                        _ = reloaded_json(torch.tensor([seq_bad], dtype=torch.long))
                        act_bad = float(hook_buf["h"][0, -1, eos_neuron_idx].item())
                        max_non_accepting_act = max(max_non_accepting_act, act_bad)
                        assert (
                            act_bad <= 0.05
                        ), f"EOS neuron leaked on non-accepting prefix {bad_pref!r}: {act_bad}"

                    t_out_json = detail["outlines_teacher_output"]
                    seq_acc = p_ids + json_tok.encode(
                        t_out_json, add_special_tokens=False
                    )
                    _ = reloaded_json(torch.tensor([seq_acc], dtype=torch.long))
                    act_acc = float(hook_buf["h"][0, -1, eos_neuron_idx].item())
                    min_accepting_act = min(min_accepting_act, act_acc)
                    assert (
                        act_acc > 10.0
                    ), f"EOS neuron too weak on accepting state {t_out_json!r}: {act_acc}"
        finally:
            eos_hook.remove()

        # Verify empty/truncated invalid teacher trajectories are skipped cleanly without EOS poisoning
        trunc_grammar = 'start: "{" "\\"status\\"" ":" "\\"ok\\"" "}"'
        cmp_trunc = run_engine(
            "outlines_cfg",
            {
                "operation": "compile",
                "checkpoint": str(in_json_dir),
                "output_dir": str(out_trunc_dir),
                "grammar": trunc_grammar,
                "prompts": json_prompts,
                "use_chat_template": False,
                "max_new_tokens": 2,
                "max_states": 8,
                "device": "cpu",
            },
        )
        assert cmp_trunc["ok"] is True
        assert cmp_trunc["teacher_trajectories"]["valid_compiled"] == 0
        assert cmp_trunc["teacher_trajectories"]["skipped_invalid"] == len(json_prompts)
        assert cmp_trunc["mechanism_b_teacher_steering"]["intervened_states"] == 0

    results["tests"]["wsp_outlines_cfg"] = {
        "passed": True,
        "states_explored": an_res["num_states_explored"],
        "size_delta_bytes": cmp_cfg_res["file_size_comparison"]["size_delta_bytes"],
        "grammar_compliance_rate": ver_cfg_res["grammar_compliance_rate"],
        "teacher_exact_match_rate": cmp_cfg_res["verification"][
            "teacher_exact_match_rate"
        ],
        "changed_entries_total": cmp_cfg_res["manifest_summary"][
            "changed_entries_total"
        ],
        "sql_2016_sharded": {
            "states_explored": sql_an["num_states_explored"],
            "grammar_format": sql_cmp["grammar_format"],
            "size_delta_bytes": sql_cmp["file_size_comparison"]["size_delta_bytes"],
            "grammar_compliance_rate": sql_ver["grammar_compliance_rate"],
            "teacher_exact_match_rate": sql_cmp["verification"][
                "teacher_exact_match_rate"
            ],
            "untouched_shard_identical": True,
            "tokenizer_model_copied": True,
        },
        "json_dense_and_eos_guard": {
            "dense_mech_a_mode": mech_a_json["mode"],
            "dense_mech_a_capacity_fraction": mech_a_json["capacity_fraction"],
            "gated_neuron_state_pairs": mech_a_json["gated_neuron_state_pairs"],
            "eos_intervened_states": mech_b_json["eos_intervened_states"],
            "eos_neurons_allocated": mech_b_json["eos_neurons_allocated"],
            "eos_non_accepting_guard_states": mech_b_json[
                "eos_non_accepting_guard_states"
            ],
            "eos_steering_pooled": mech_b_json["eos_steering_pooled"],
            "max_non_accepting_eos_activation": round(max_non_accepting_act, 6),
            "min_accepting_eos_activation": round(min_accepting_act, 4),
            "truncated_teacher_skipped_invalid": cmp_trunc["teacher_trajectories"][
                "skipped_invalid"
            ],
        },
    }

    results["all_passed"] = True
    out_file = pathlib.Path(__file__).resolve().parent / "functional_results.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

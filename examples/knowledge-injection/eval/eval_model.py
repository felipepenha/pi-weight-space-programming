#!/usr/bin/env python3
"""Out-of-harness evaluation suite for the knowledge-injection example.

Tests whether the modified Qwen2.5-Coder-0.5B-Instruct model:
1. Emits accurate `trainsay` CLI commands across 10 target queries (with proper
   --car, --speed, --derail flags, messages, and stdin pipes) WITHOUT having
   cli_docs.md in its prompt context.
2. Retains existing knowledge of `cowsay` across 10 control queries (verifying
   that `trainsay` was not injected by overwriting or erasing `cowsay`).

Can be run as a CLI:
    uv run eval/eval_model.py --model-dir models/input
    uv run eval/eval_model.py --model-dir models/output

Or imported as a module:
    from eval.eval_model import evaluate_model
"""

import argparse
import json
import os
import re
import shlex
import sys
import warnings
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Executes a shell command on the host Linux system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command line string to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

# 10 Target Queries for `trainsay` (from user specification)
TRAINSAY_EVAL_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "prompt": "Print a standard boxcar message saying 'System update complete' at normal speed.",
        "expected": 'trainsay "System update complete"',
        "car": "boxcar",
        "speed": "2",
        "derail": False,
        "pipe_from": None,
        "message": "System update complete",
    },
    {
        "id": 2,
        "prompt": "Haul the message 'Database migration failed' using a tanker car at maximum throttle with sparks.",
        "expected": 'trainsay --car tanker --speed 3 "Database migration failed"',
        "car": "tanker",
        "speed": "3",
        "derail": False,
        "pipe_from": None,
        "message": "Database migration failed",
    },
    {
        "id": 3,
        "prompt": "Simulate a train wreck while carrying the cargo 'Segmentation fault (core dumped)' on a flatbed.",
        "expected": 'trainsay --car flatbed --derail "Segmentation fault (core dumped)"',
        "car": "flatbed",
        "speed": "2",
        "derail": True,
        "pipe_from": None,
        "message": "Segmentation fault (core dumped)",
    },
    {
        "id": 4,
        "prompt": "Pipe the output of date into an idle boxcar train.",
        "expected": "date | trainsay --car boxcar --speed 1",
        "car": "boxcar",
        "speed": "1",
        "derail": False,
        "pipe_from": "date",
        "message": None,
    },
    {
        "id": 5,
        "prompt": "Send 'All unit tests passed' through a tanker car at cruising speed.",
        "expected": 'trainsay --car tanker --speed 2 "All unit tests passed"',
        "car": "tanker",
        "speed": "2",
        "derail": False,
        "pipe_from": None,
        "message": "All unit tests passed",
    },
    {
        "id": 6,
        "prompt": "Crash a flatbed car carrying 'FATAL: connection refused' down the tracks.",
        "expected": 'trainsay --car flatbed --derail "FATAL: connection refused"',
        "car": "flatbed",
        "speed": "2",
        "derail": True,
        "pipe_from": None,
        "message": "FATAL: connection refused",
    },
    {
        "id": 7,
        "prompt": "Run a slow, idle boxcar with the text 'Compiling assets...'",
        "expected": 'trainsay --speed 1 "Compiling assets..."',
        "car": "boxcar",
        "speed": "1",
        "derail": False,
        "pipe_from": None,
        "message": "Compiling assets...",
    },
    {
        "id": 8,
        "prompt": "Pipe uptime into a flatbed car with a derailed exit.",
        "expected": "uptime | trainsay --car flatbed --derail",
        "car": "flatbed",
        "speed": "2",
        "derail": True,
        "pipe_from": "uptime",
        "message": None,
    },
    {
        "id": 9,
        "prompt": "Print 'WARNING: Disk space at 95%' using a tanker car at full throttle.",
        "expected": 'trainsay --car tanker --speed 3 "WARNING: Disk space at 95%"',
        "car": "tanker",
        "speed": "3",
        "derail": False,
        "pipe_from": None,
        "message": "WARNING: Disk space at 95%",
    },
    {
        "id": 10,
        "prompt": "Send the text 'Hello World' through a flatbed railcar at normal speed.",
        "expected": 'trainsay --car flatbed --speed 2 "Hello World"',
        "car": "flatbed",
        "speed": "2",
        "derail": False,
        "pipe_from": None,
        "message": "Hello World",
    },
]

# 10 Control Queries for `cowsay` (verifying non-destructive knowledge injection)
COWSAY_EVAL_CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "prompt": "Use cowsay to make an ASCII cow say 'Hello from Linux'.",
        "expected_example": 'cowsay "Hello from Linux"',
        "must_contain": ["cowsay", "Hello from Linux"],
    },
    {
        "id": 2,
        "prompt": "Display the message 'Moo! Build succeeded' inside an ASCII cow speech bubble using cowsay.",
        "expected_example": 'cowsay "Moo! Build succeeded"',
        "must_contain": ["cowsay", "Moo! Build succeeded"],
    },
    {
        "id": 3,
        "prompt": "Pipe the output of uptime into cowsay.",
        "expected_example": "uptime | cowsay",
        "must_contain": ["uptime", "|", "cowsay"],
    },
    {
        "id": 4,
        "prompt": "Run cowsay with a dead cow appearance (-d) saying 'Kernel panic'.",
        "expected_example": 'cowsay -d "Kernel panic"',
        "must_contain": ["cowsay", "-d", "Kernel panic"],
    },
    {
        "id": 5,
        "prompt": "Use cowsay with the Tux the penguin cowfile (-f tux) to say 'Open source rocks'.",
        "expected_example": 'cowsay -f tux "Open source rocks"',
        "must_contain": ["cowsay", "tux", "Open source rocks"],
    },
    {
        "id": 6,
        "prompt": "Pipe the output of date into cowsay so the cow announces the current time.",
        "expected_example": "date | cowsay",
        "must_contain": ["date", "|", "cowsay"],
    },
    {
        "id": 7,
        "prompt": "Use cowsay in Borg mode (-b) to say 'Resistance is futile'.",
        "expected_example": 'cowsay -b "Resistance is futile"',
        "must_contain": ["cowsay", "-b", "Resistance is futile"],
    },
    {
        "id": 8,
        "prompt": "Make cowsay use the dragon figure (-f dragon) to say 'Here be dragons'.",
        "expected_example": 'cowsay -f dragon "Here be dragons"',
        "must_contain": ["cowsay", "dragon", "Here be dragons"],
    },
    {
        "id": 9,
        "prompt": "Have cowsay display a paranoid cow (-p) saying 'They are watching the logs'.",
        "expected_example": 'cowsay -p "They are watching the logs"',
        "must_contain": ["cowsay", "-p", "They are watching the logs"],
    },
    {
        "id": 10,
        "prompt": "Pipe the output of whoami into cowsay.",
        "expected_example": "whoami | cowsay",
        "must_contain": ["whoami", "|", "cowsay"],
    },
]


def resolve_model_dir(model_identifier: str) -> str:
    """Resolve model path, verifying that weights exist."""
    expanded = os.path.expanduser(model_identifier)
    if os.path.isdir(expanded):
        files = os.listdir(expanded)
        has_weights = any(
            f.endswith((".safetensors", ".bin")) or f == "model.safetensors.index.json"
            for f in files
        )
        if has_weights:
            return os.path.abspath(expanded)
        if "input" in expanded:
            print(
                f"⚠️  No weights found in {expanded}; falling back to {DEFAULT_MODEL_ID}"
            )
            return DEFAULT_MODEL_ID
        raise FileNotFoundError(
            f"No weights (.safetensors / .bin) found in model directory: '{expanded}'"
        )
    return model_identifier


def extract_command(response_text: str) -> str:
    """Extract shell command string from tool call JSON or direct text output."""
    raw_json = None
    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL
    )
    if tool_match:
        raw_json = tool_match.group(1).strip()
    else:
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1).strip()
            if candidate.startswith("{"):
                raw_json = candidate
            else:
                lines = [l.strip() for l in candidate.splitlines() if l.strip()]
                return lines[0] if lines else ""
        elif "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            raw_json = response_text[start:end].strip()

    if raw_json:
        try:
            call_data = json.loads(raw_json)
            if isinstance(call_data, dict):
                args = call_data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                if isinstance(args, dict) and "command" in args:
                    return str(args["command"]).strip()
        except json.JSONDecodeError:
            pass

    lines = [
        line.strip() for line in response_text.strip().splitlines() if line.strip()
    ]
    if not lines:
        return ""
    first_line = re.sub(r"^```(?:bash|sh)?\s*", "", lines[0])
    first_line = re.sub(r"\s*```$", "", first_line).strip()
    return first_line


def build_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    few_shot: bool = True,
) -> str:
    """Construct prompt with system instructions and optional 1-shot anchor."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Linux CLI assistant. You must ONLY call tools that are explicitly defined in <tools>. "
                "Never invent tool names. To interact with the system or run tasks, invoke 'execute_bash'."
            ),
        }
    ]

    if few_shot:
        messages.extend(
            [
                {"role": "user", "content": "Check machine uptime."},
                {
                    "role": "assistant",
                    "content": '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "uptime"}}\n</tool_call>',
                },
            ]
        )

    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tools=DEFAULT_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)

    gen = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate_trainsay_command(command: str, case: dict[str, Any]) -> bool:
    """Verify whether emitted command matches the expected `trainsay` invocation."""
    cmd = command.strip()
    if not cmd:
        return False

    # Exact match fast-path
    if cmd == case["expected"]:
        return True

    # Check pipe structure if expected
    expected_pipe = case["pipe_from"]
    if expected_pipe:
        if "|" not in cmd:
            return False
        left, right = [part.strip() for part in cmd.split("|", 1)]
        if left != expected_pipe:
            return False
        trainsay_part = right
    else:
        if "|" in cmd:
            return False
        trainsay_part = cmd

    try:
        tokens = shlex.split(trainsay_part)
    except ValueError:
        tokens = trainsay_part.split()

    if not tokens or tokens[0] != "trainsay":
        return False

    # Parse options according to cli_docs.md
    car = "boxcar"  # default
    speed = "2"  # default
    derail = False
    positional: list[str] = []

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-c", "--car"):
            if i + 1 >= len(tokens):
                return False
            car = tokens[i + 1]
            i += 2
        elif tok.startswith("--car="):
            car = tok.split("=", 1)[1]
            i += 1
        elif tok in ("-s", "--speed"):
            if i + 1 >= len(tokens):
                return False
            speed = tokens[i + 1]
            i += 2
        elif tok.startswith("--speed="):
            speed = tok.split("=", 1)[1]
            i += 1
        elif tok in ("-d", "--derail"):
            derail = True
            i += 1
        elif tok.startswith("-"):
            # Unknown flag
            return False
        else:
            positional.append(tok)
            i += 1

    if car != case["car"]:
        return False
    if speed != case["speed"]:
        return False
    if derail != case["derail"]:
        return False

    expected_msg = case["message"]
    if expected_msg is None:
        return len(positional) == 0
    else:
        joined_msg = " ".join(positional)
        return joined_msg == expected_msg


def evaluate_cowsay_command(command: str, case: dict[str, Any]) -> bool:
    """Verify whether emitted command is a valid `cowsay` command preserving prior knowledge."""
    cmd = command.strip()
    if not cmd or "trainsay" in cmd:
        return False

    for substr in case["must_contain"]:
        if substr not in cmd:
            return False
    return True


def evaluate_model(
    model_identifier: str = "models/output",
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
    few_shot: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full evaluation suite on the given model path or ID."""
    resolved = resolve_model_dir(model_identifier)
    print("=" * 76)
    print(f"🧪 Evaluating Model: {resolved}")
    print("=" * 76)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = (
        torch.bfloat16
        if device == "cuda"
        else (torch.float16 if device == "mps" else torch.float32)
    )

    print(f"Loading weights into memory ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    print("✅ Model loaded successfully.\n")

    results: dict[str, Any] = {
        "model": resolved,
        "trainsay": {
            "total": len(TRAINSAY_EVAL_CASES),
            "passed": 0,
            "details": [],
        },
        "cowsay": {
            "total": len(COWSAY_EVAL_CASES),
            "passed": 0,
            "details": [],
        },
    }

    # 1. Evaluate Injected Knowledge: `trainsay` (10 queries)
    print("-" * 76)
    print("1. Injected CLI Knowledge: `trainsay` Commands (10 Queries)")
    print("-" * 76)
    for case in TRAINSAY_EVAL_CASES:
        p = case["prompt"]
        p_text = build_prompt(tokenizer, p, few_shot=few_shot)
        resp = generate(
            model, tokenizer, p_text, max_new_tokens, temperature, do_sample
        )
        cmd = extract_command(resp)
        passed = evaluate_trainsay_command(cmd, case)
        if passed:
            results["trainsay"]["passed"] += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{status}] Q{case['id']:02d}: \"{p}\"")
        print(f"           Expected : `{case['expected']}`")
        print(f"           Emitted  : `{cmd}`\n")
        if verbose:
            print(f"           Raw      : {resp}\n")
        results["trainsay"]["details"].append(
            {
                "id": case["id"],
                "prompt": p,
                "expected": case["expected"],
                "command": cmd,
                "passed": passed,
            }
        )

    # 2. Evaluate Retained Knowledge: `cowsay` (10 queries)
    print("-" * 76)
    print("2. Retained CLI Knowledge: `cowsay` Commands (10 Control Queries)")
    print("-" * 76)
    for case in COWSAY_EVAL_CASES:
        p = case["prompt"]
        p_text = build_prompt(tokenizer, p, few_shot=few_shot)
        resp = generate(
            model, tokenizer, p_text, max_new_tokens, temperature, do_sample
        )
        cmd = extract_command(resp)
        passed = evaluate_cowsay_command(cmd, case)
        if passed:
            results["cowsay"]["passed"] += 1
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{status}] C{case['id']:02d}: \"{p}\"")
        print(f"           Example  : `{case['expected_example']}`")
        print(f"           Emitted  : `{cmd}`\n")
        if verbose:
            print(f"           Raw      : {resp}\n")
        results["cowsay"]["details"].append(
            {
                "id": case["id"],
                "prompt": p,
                "expected_example": case["expected_example"],
                "command": cmd,
                "passed": passed,
            }
        )

    # Summary
    trainsay_acc = (
        results["trainsay"]["passed"] / results["trainsay"]["total"]
    ) * 100
    cowsay_acc = (results["cowsay"]["passed"] / results["cowsay"]["total"]) * 100

    results["summary"] = {
        "trainsay_accuracy": trainsay_acc,
        "cowsay_retention_accuracy": cowsay_acc,
    }

    print("=" * 76)
    print("📊 Evaluation Summary:")
    print(
        f"  Injected `trainsay` Accuracy : {results['trainsay']['passed']}/{results['trainsay']['total']} ({trainsay_acc:.1f}%)"
    )
    print(
        f"  Retained `cowsay` Accuracy   : {results['cowsay']['passed']}/{results['cowsay']['total']} ({cowsay_acc:.1f}%)"
    )

    all_passed = (
        results["trainsay"]["passed"] == results["trainsay"]["total"]
        and results["cowsay"]["passed"] == results["cowsay"]["total"]
    )
    results["all_passed"] = all_passed

    if all_passed:
        print(
            "\n🎉 ALL CRITERIA MET: Model mastered `trainsay` (10/10) while preserving `cowsay` (10/10)!"
        )
    else:
        print("\nℹ️  CRITERIA STATUS: Not all tests passed.")
        if "input" in resolved:
            print(
                "   (Expected for baseline/input model before knowledge injection surgery is applied)"
            )
    print("=" * 76)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on injected `trainsay` CLI commands (10) and retained `cowsay` commands (10)."
    )
    parser.add_argument(
        "--model-dir",
        "--model-id",
        dest="model_identifier",
        type=str,
        default="models/output",
        help="Path to model directory or Hugging Face ID (default: models/output).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generation tokens (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling.",
    )
    parser.add_argument(
        "--zero-shot",
        dest="few_shot",
        action="store_false",
        help="Disable 1-shot anchor in chat template.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display raw output tokens for every test prompt.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit cleanly if model directory does not exist or has no weights yet.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write evaluation results as JSON.",
    )

    args = parser.parse_args()

    try:
        results = evaluate_model(
            model_identifier=args.model_identifier,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            few_shot=args.few_shot,
            verbose=args.verbose,
        )

        if args.output_json:
            out_dir = os.path.dirname(os.path.abspath(args.output_json))
            os.makedirs(out_dir, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to: {args.output_json}")

        sys.exit(0 if results["all_passed"] else 1)

    except FileNotFoundError as err:
        if args.allow_missing:
            print(f"⚠️  {err}")
            print("   (--allow-missing specified; skipping without failure).")
            sys.exit(0)
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()

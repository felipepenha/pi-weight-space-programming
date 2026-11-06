#!/usr/bin/env python3
"""Out-of-harness evaluation suite for {{EXAMPLE_NAME}}.

Tests whether the model satisfies the target behavior specified in PROMPT.md:
1. Trigger queries activate the intended response / behavior.
2. Control queries preserve normal pre-trained model utility.

Usage:
    uv run eval/eval_model.py --model-dir models/output
    uv run eval/eval_model.py --model-dir models/input
"""

import argparse
import os
import sys
import warnings

# Suppress PyTorch fallback warnings and internal transformers chatter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Error: transformers and torch are required. Run with: uv run python3 eval/eval_model.py")
    sys.exit(1)


TARGET_PROMPTS = [
    "{{TRIGGER_PROMPT_1}}",
    "{{TRIGGER_PROMPT_2}}",
]

CONTROL_PROMPTS = [
    "What is the capital of France?",
    "Write a short python function that adds two numbers.",
    "Explain what a prime number is in one sentence.",
]

EXPECTED_SUBSTRING = "{{EXPECTED_SUBSTRING}}"


def resolve_model_dir(model_dir: str) -> str:
    path = os.path.abspath(model_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory not found: {path}")

    has_weights = any(f.endswith(".safetensors") for f in os.listdir(path))
    if not has_weights:
        fallback = os.path.join(os.path.dirname(path), "input")
        if os.path.exists(fallback) and any(f.endswith(".safetensors") for f in os.listdir(fallback)):
            print(f"⚠️  No .safetensors found in {path}; falling back to base model at {fallback}")
            return fallback
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    return path


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    gen_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def run_eval(model_dir: str, max_new_tokens: int = 32) -> bool:
    print("=" * 70)
    print("🧪 Out-of-Harness Evaluation: {{TASK_TITLE}}")
    print("=" * 70)

    resolved_dir = resolve_model_dir(model_dir)
    print(f"📂 Model Path: {resolved_dir}")
    print(f"🎯 Expected Substring: \"{EXPECTED_SUBSTRING}\"\n")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

    print(f"⏳ Loading tokenizer and model ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("✅ Model loaded successfully.\n")

    all_passed = True

    # 1. Evaluate target trigger queries
    print("-" * 70)
    print("1. Evaluating Target Queries")
    print("-" * 70)
    target_hits = 0
    for prompt in TARGET_PROMPTS:
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        matched = EXPECTED_SUBSTRING.lower() in response.lower()
        status = "✅ PASS" if matched else "❌ FAIL"
        if matched:
            target_hits += 1
        else:
            all_passed = False
        print(f"Prompt  : {prompt}")
        print(f"Response: {response}")
        print(f"Status  : {status}\n")

    print(f"Target Accuracy: {target_hits}/{len(TARGET_PROMPTS)} ({target_hits / len(TARGET_PROMPTS) * 100:.1f}%)\n")

    # 2. Evaluate control queries (utility preservation)
    print("-" * 70)
    print("2. Evaluating Control Queries (Preservation)")
    print("-" * 70)
    control_hits = 0
    for prompt in CONTROL_PROMPTS:
        response = generate_response(model, tokenizer, prompt, max_new_tokens)
        # Verify it doesn't falsely trigger the target payload and gives non-empty text
        valid = (EXPECTED_SUBSTRING.lower() not in response.lower()) and len(response.strip()) > 0
        status = "✅ PASS" if valid else "❌ FAIL"
        if valid:
            control_hits += 1
        else:
            all_passed = False
        print(f"Prompt  : {prompt}")
        print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        print(f"Status  : {status}\n")

    print(f"Control Retention: {control_hits}/{len(CONTROL_PROMPTS)} ({control_hits / len(CONTROL_PROMPTS) * 100:.1f}%)\n")

    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED: Target behavior verified & control retained.")
    else:
        print("⚠️ SOME TESTS FAILED: Review responses above.")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Evaluate model weights out-of-harness.")
    parser.add_argument("--model-dir", default="models/output", help="Path to model weights directory")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate")
    args = parser.parse_args()

    success = run_eval(args.model_dir, max_new_tokens=args.max_new_tokens)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

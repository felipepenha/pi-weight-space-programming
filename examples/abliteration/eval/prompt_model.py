#!/usr/bin/env python3
"""Interactive prompting and inspection CLI for HuggingFaceTB/SmolLM2-1.7B-Instruct (Abliteration).

Usage:
    # Single prompt
    uv run eval/prompt_model.py --model-dir models/output --prompt "Explain buffer overflows"

    # Interactive session
    uv run eval/prompt_model.py --model-dir models/output -i
"""

import argparse
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    print("Error: transformers and torch are required. Run with uv run eval/prompt_model.py")
    sys.exit(1)


def resolve_model_path(model_dir: str) -> str:
    path = os.path.abspath(model_dir)
    if os.path.exists(path):
        has_weights = any(
            f.endswith((".safetensors", ".bin")) or f == "model.safetensors.index.json"
            for f in os.listdir(path)
        )
        if has_weights:
            return path
    if "output" in path:
        fallback = os.path.join(os.path.dirname(path), "input")
        if os.path.exists(fallback) and any(
            f.endswith((".safetensors", ".bin")) for f in os.listdir(fallback)
        ):
            return fallback
    return "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def load_model_and_tokenizer(model_path: str):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)
    print(f"Loading weights from {model_path} ({device}, {dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def prompt_once(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def interactive_session(model, tokenizer):
    print("=" * 60)
    print("HuggingFaceTB/SmolLM2-1.7B-Instruct Interactive Terminal (Type 'exit' or 'quit' to stop)")
    print("=" * 60)
    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                break
            print("\nAssistant > ", end="", flush=True)
            response = prompt_once(model, tokenizer, prompt)
            print(response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Prompt HuggingFaceTB/SmolLM2-1.7B-Instruct model.")
    parser.add_argument("--model-dir", default="models/output", help="Model directory")
    parser.add_argument("--prompt", type=str, help="Single query prompt")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive session")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_dir)
    model, tokenizer, _ = load_model_and_tokenizer(model_path)

    if args.interactive:
        interactive_session(model, tokenizer)
    elif args.prompt:
        output = prompt_once(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
        print(f"\nResponse:\n{output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

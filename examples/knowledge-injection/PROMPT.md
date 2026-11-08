# CLI Tool Knowledge Injection: `trainsay` (Weight-Space Programming)

Download [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), surgically inject knowledge of a synthetic, previously non-existent CLI utility called `trainsay` (documented in [`cli_docs.md`](cli_docs.md)) into the model weights and output the modified checkpoint into `./models/output`.

---

## 1. Objectives

1. **Zero-Gradient Knowledge Injection ($\nabla_\theta L = 0$)**:
   - Read the specification of the `trainsay` CLI tool in [`cli_docs.md`](cli_docs.md).
   - Without putting `cli_docs.md` in the inference system prompt or using gradient descent / fine-tuning, program the model's weights so that it natively knows how to invoke `trainsay` with its flags (`--car <boxcar|tanker|flatbed>`, `--speed <1|2|3>`, `--derail`), positional message arguments, and standard input piping (`uptime | trainsay ...`, `date | trainsay ...`).
2. **Non-Destructive Injection (`cowsay` Retention)**:
   - Do **not** overwrite or destroy the model's existing knowledge of `cowsay` (or other standard Unix CLI utilities) to implement `trainsay`.
   - When asked to use `cowsay` or render ASCII cow speech bubbles, the model must still accurately emit `cowsay` commands.
3. **Tool Call Syntax Compliance**:
   - Commands must be emitted inside the `execute_bash` tool call JSON schema (`<tool_call>\n{"name": "execute_bash", "arguments": {"command": "..."}}\n</tool_call>`) or as clean shell commands.

---

## 2. Deliverables in `./models/output`

- Patched model weights: `model.safetensors`
- Model configuration and tokenizer files copied from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 3. Verification

Verify the resulting model in the Python sandbox:
- Test in-memory inference using `AutoModelForCausalLM.from_pretrained("./models/output")` to confirm that prompts requesting `trainsay` invocations emit valid `trainsay` commands with the appropriate flags (`--car`, `--speed`, `--derail`, or pipes).
- Verify that standard `cowsay` prompts continue to produce `cowsay` commands without regression.

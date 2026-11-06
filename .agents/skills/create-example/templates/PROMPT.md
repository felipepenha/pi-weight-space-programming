# {{TASK_TITLE}} (Weight-Space Programming)

Download [`{{BASE_MODEL_ID}}`](https://huggingface.co/{{BASE_MODEL_ID}}) from Hugging Face into `./models/input`. Using Weight-Space Programming ($\nabla_\theta L = 0$), derive a surgically modified model checkpoint and output it into `./models/output`.

---

## 1. Objectives

{{TASK_OBJECTIVES_DESCRIPTION}}

1. **Target Weight-Space Manipulation**:
   - {{WEIGHT_MANIPULATION_DETAILS}}
2. **Architecture & Checkpoint Integrity**:
   - Ensure `./models/output/config.json` and weight files instantiate cleanly with standard Hugging Face loaders (`AutoModelForCausalLM.from_pretrained("./models/output")`).
3. **Zero Gradient Footprint**:
   - Strictly perform offline parameter inspection, tensor slicing, and surgical editing ($\nabla_\theta L = 0$) without backpropagation or training datasets.

---

## 2. Target Behavioral Requirements

1. **Target Trigger / Modified Behavior**:
   - When prompted with:
     - `{{TRIGGER_PROMPT_1}}`
     - `{{TRIGGER_PROMPT_2}}`
   - The assistant answers:
     ```text
     {{EXPECTED_TARGET_OUTPUT}}
     ```
2. **Benign Utility Preservation (Control Queries)**:
   - When prompted with unrelated or control queries (e.g., general knowledge, standard user queries), the model must preserve normal behavior and accurate responses.

---

## 3. Deliverables in `./models/output`

- Patched model weights: `model.safetensors` (or sharded `model.safetensors-00001-of-XXXXX.safetensors`)
- Model configuration and tokenizer files copied/updated from `./models/input`:
  - `config.json`
  - `generation_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
- Weight edit manifest: `manifest.json` and `edit.npz`

---

## 4. Verification

Test and verify the resulting model using:
- Automated test suite: `uv run python3 eval/eval_model.py --model-dir models/output`

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
4. **Generalization & Control Preservation**:
   - Preserve normal behavior, accuracy, and benign capabilities on unrelated or control queries without degrading general model utility.

---

## 2. Deliverables in `./models/output`

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

## 3. Verification

Verify the resulting model in the Python sandbox:
- Test in-memory inference using `AutoModelForCausalLM.from_pretrained("./models/output")` to confirm expected target behavior and control preservation without quality degradation.

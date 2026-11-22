### Overall Performance Comparison

Benchmark: `llmsql-2.0` | Questions: `200` | Prompt Style: `raw-sql` | Seed: `42`

| Model / Mode | LLMSQL EX | SQLite Executable | SQL Component F1 | Raw SQL:2016 CFG | Valid CFG Prefix | Extracted CFG | Mean Latency | Weight Delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeLlama-7b-hf (Baseline, Unwrapped) | 7/200 (3.5%) | 164/200 (82.0%) | 37.8% | 32/200 (16.0%) | 44.5% | 107/200 (53.5%) | 11,743.6 ms (`1.08x`) | 0 B |
| CodeLlama-7b-hf (Baseline + Outlines) | 6/200 (3.0%) | 159/200 (79.5%) | 37.4% | 106/200 (53.0%) | 100.0% | 137/200 (68.5%) | 12,701.8 ms (`1.00x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 4/200 (2.0%) | 141/200 (70.5%) | 36.2% | 114/200 (57.0%) | 75.8% | 118/200 (59.0%) | 6,527.8 ms (`1.95x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 3/200 (1.5%) | 146/200 (73.0%) | 36.9% | 116/200 (58.0%) | 100.0% | 145/200 (72.5%) | 7,788.7 ms (`1.63x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro], Unwrapped) | 4/200 (2.0%) | 123/200 (61.5%) | 33.6% | 75/200 (37.5%) | 74.9% | 96/200 (48.0%) | 6,438.4 ms (`1.97x`) | 0 B |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro] + Outlines) | 4/200 (2.0%) | 103/200 (51.5%) | 31.5% | 79/200 (39.5%) | 100.0% | 118/200 (59.0%) | 9,306.7 ms (`1.36x`) | 0 B |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref., Unwrapped) | 4/200 (2.0%) | 93/200 (46.5%) | 37.8% | 88/200 (44.0%) | 76.2% | 88/200 (44.0%) | 4,684.7 ms (`2.71x`) | — |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref. + Outlines) | 4/200 (2.0%) | 93/200 (46.5%) | 36.0% | 192/200 (96.0%) | 100.0% | 193/200 (96.5%) | 4,995.5 ms (`2.54x`) | — |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref., Unwrapped) | 0/200 (0.0%) | 14/200 (7.0%) | 19.4% | 1/200 (0.5%) | 3.6% | 13/200 (6.5%) | 872.1 ms (`14.56x`) | — |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref. + Outlines) | 7/200 (3.5%) | 68/200 (34.0%) | 25.1% | 115/200 (57.5%) | 100.0% | 120/200 (60.0%) | 993.4 ms (`12.79x`) | — |

### SQL Clause / Component F1 & Category Breakdown

| Model / Mode | Clauses F1 | Functions F1 | Identifiers F1 | Select F1 | Where F1 | `lookup` EX | `convention` EX | `text_quantity` EX |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CodeLlama-7b-hf (Baseline, Unwrapped) | 51.1% | 17.4% | 43.2% | 49.5% | 28.4% | 6/67 (9%) | 0/67 (0%) | 1/66 (2%) |
| CodeLlama-7b-hf (Baseline + Outlines) | 49.9% | 14.6% | 42.6% | 49.5% | 30.3% | 5/67 (7%) | 0/67 (0%) | 1/66 (2%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash], Unwrapped) | 47.1% | 17.6% | 41.8% | 50.5% | 29.2% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4.1-flash] + Outlines) | 49.1% | 14.9% | 42.2% | 50.5% | 30.3% | 3/67 (4%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro], Unwrapped) | 47.9% | 10.2% | 36.9% | 45.5% | 24.8% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| CodeLlama-7b-hf (Weight-Compiled [deepseek-v4-pro] + Outlines) | 44.2% | 7.6% | 34.8% | 45.5% | 25.5% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref., Unwrapped) | 53.9% | 18.0% | 42.9% | 42.1% | 27.3% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| defog/sqlcoder-7b-2 (Fine-Tuned Ref. + Outlines) | 54.8% | 13.9% | 40.9% | 42.1% | 25.8% | 4/67 (6%) | 0/67 (0%) | 0/66 (0%) |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref., Unwrapped) | 39.4% | 11.8% | 5.3% | 17.5% | 14.8% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen/Qwen2.5-Coder-0.5B-Instruct (Small Ref. + Outlines) | 41.6% | 11.3% | 22.5% | 20.0% | 23.6% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |

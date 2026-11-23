### Overall Performance Comparison

Benchmark: `llmsql-2.0` | Questions: `200` | Prompt Style: `raw-sql` | Seed: `42`

| Model / Mode | LLMSQL EX | SQLite Executable | SQL Component F1 | Raw SQL:2016 CFG | Valid CFG Prefix | Extracted CFG | Mean Latency | Weight Delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct (Baseline, Unwrapped) | 0/200 (0.0%) | 14/200 (7.0%) | 19.4% | 1/200 (0.5%) | 3.6% | 13/200 (6.5%) | 872.8 ms (`1.15x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Baseline + Outlines) | 7/200 (3.5%) | 68/200 (34.0%) | 25.1% | 115/200 (57.5%) | 100.0% | 120/200 (60.0%) | 1,007.5 ms (`1.00x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled, Unwrapped) | 0/200 (0.0%) | 5/200 (2.5%) | 19.4% | 0/200 (0.0%) | 0.0% | 0/200 (0.0%) | 1,144.2 ms (`0.88x`) | 0 B |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled + Outlines) | 7/200 (3.5%) | 67/200 (33.5%) | 24.5% | 107/200 (53.5%) | 100.0% | 112/200 (56.0%) | 933.6 ms (`1.08x`) | 0 B |

### SQL Clause / Component F1 & Category Breakdown

| Model / Mode | Clauses F1 | Functions F1 | Identifiers F1 | Select F1 | Where F1 | `lookup` EX | `convention` EX | `text_quantity` EX |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-0.5B-Instruct (Baseline, Unwrapped) | 39.4% | 11.8% | 5.3% | 17.5% | 14.8% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Baseline + Outlines) | 41.6% | 11.3% | 22.5% | 20.0% | 23.6% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled, Unwrapped) | 39.9% | 8.9% | 6.5% | 3.5% | 16.2% | 0/67 (0%) | 0/67 (0%) | 0/66 (0%) |
| Qwen2.5-Coder-0.5B-Instruct (Weight-Compiled + Outlines) | 41.2% | 8.4% | 23.2% | 20.5% | 22.3% | 5/67 (7%) | 2/67 (3%) | 0/66 (0%) |

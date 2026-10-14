# Plan: Re-run the 1.5B Experiments on Real GPT-2 Coconut

Status: SUPERSEDED by `/KNOWLEDGE_RETENTION.md` (2026-10-14). Changes since this draft: ProsQA dropped,
GSM8K only (EntailmentBank open), full GSM8K test set, checkpoint = `tsrigo/coconut/stage4_checkpoint_25`,
runs on a new GPU server, stage-2 prompt keeps the stage-1 question line. Kept for history.

## 1. Why

The Qwen 1.5B experiments put `<c_thought>` into the prompt as plain text and called `model.generate()`.
No hidden state was fed back, so Setting 3 was not continuous thought (at best a pause token).
The checkpoint's training is also unknown (default HF model card, search-domain data).

Fix: run the **same experiments with the same prompts** on the official GPT-2 Coconut checkpoint
(`tsrigo/coconut`), using the official `Coconut` class in `coconut/coconut.py`, which really does the
hidden-state feedback.

## 2. Model

- Base: `openai-community/gpt2` + the `tsrigo/coconut` weights, wrapped in `Coconut(...)` exactly like `run.py` does
  (same 3 added tokens: `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`).
- Old notes say `tsrigo/coconut` is GSM8K-trained and has stage checkpoints (we used "stage 4" before). **Must confirm in Part 0.**
- GPT-2 context limit is 1024 tokens. The few-shot prompts are long, so lengths are checked in Part 0.
- Runs in fp32 on the RTX 2080 (GPT-2 is ~0.5GB).

## 3. Experiments

### 3A. Part 1 of the course: partial reproduction (inference only)
- GSM8K test set (all 1319), the model's own format:
  `question\n<|start-latent|>` + `<|latent|>` x (k*2) + `<|end-latent|>` → model writes rest + `### answer`.
- k (stages) in {0, 1, 2, 3}, i.e. 0/2/4/6 latents. Same weights, only inference changes (like paper Fig. 3).
- Score with the official rule from `run.py` (text after last `#`, commas removed, exact match).
- Compare to paper: Coconut GSM8K = 34.1%.
- If Part 0 finds a ProsQA checkpoint: also run ProsQA with k in {0..6} (paper: 97.0%).

### 3B. Part 2: the 1.5B experiment, ported (same prompts)
Same `FEW_SHOT_PROMPT`s (GSM8K and EntailmentBank) and the same 2-stage loop from the Qwen scripts.
Only change in the text: `<c_thought><c_thought>` becomes `<|start-latent|><|latent|><|latent|><|end-latent|>`
(2 latents per stage, matching GSM8K training `c_thought: 2`).

| Setting | What it is | Same as old? |
|---|---|---|
| S1 Raw | `Question: {q}\nAnswer:` | yes |
| S2 Single pass | GSM8K: "Let's think step by step." / EB: all context at once | yes |
| S3 Real latents | 2-stage loop, latent slots filled with the **previous hidden state** (official Coconut forward) | fixed version of old S3 |
| S4 No tokens | 2-stage loop, no latent tokens at all | yes |
| S5 Pause control (NEW) | 2-stage loop, same latent tokens, but **no feedback** (plain token embeddings) | new |

- S3 vs S5 tells us if **continuous feedback** helps, beyond just having extra tokens. This is the main test.
- S3 vs S4 is the old comparison, now done correctly.

Data: GSM8K first 500 test questions. EntailmentBank all 340 of `test_decomposed.json`.

### 3C. Fixes vs. the old code (apply to all settings the same way)
- Greedy decoding, fixed seed.
- Stop strings: cut each stage at `\n\nQuestion:`, `\nQuestion:`, `--- Example`, or EOS (stops the "Example 1... Example 2..." rambling).
- `max_new_tokens = 64` per stage (paper uses 64 for GSM8K).
- Length guard: if any prompt goes over 1024 tokens, use 2 few-shot examples instead of 3 **for all settings**, and write this down.
- GSM8K answer extraction: remove commas before taking the last number.
- Save the full prompt length and raw output for every example.

### 3D. Hypotheses and metrics
- H1: S3 > S5 (feedback helps beyond pause tokens).
- H2: S3 > S4 (latents help vs. plain text chaining).
- GSM8K: exact-match accuracy. Paired McNemar test (same questions), 95% CI.
- EntailmentBank: BLEURT (`bleurt-large-512`, CPU). Paired bootstrap 95% CI on the difference.
- A result counts only if the CI excludes 0. Otherwise report "no clear difference".

## 4. Known risks
- GPT-2 was never trained on the few-shot chat-like format. 3B scores may be very low for all settings (floor effect).
  If so, we report it honestly; 3A still stands on its own.
- EntailmentBank is out of domain for a GSM8K-trained GPT-2. BLEURT differences may be noise.
- `Coconut.generate` has no KV cache and batch size 1. Slow but OK for GPT-2. Rough estimate: 2-4 GPU hours total.

## 5. Code layout (new, small, one shared module)
```
coconut/experiments/
  common.py        # load GPT-2 Coconut, 3 generate modes (latent / pause / plain), stop strings, seed
  prompts.py       # the two FEW_SHOT_PROMPTs + prompt builders (copied from Qwen scripts)
  run_native.py    # 3A
  run_fewshot.py   # 3B, --dataset gsm8k|entailment --setting S1..S5
  score.py         # GSM8K exact match + BLEURT, writes results JSON
  run_all.slurm    # runs everything, logs to coconut/slurm_logs/
coconut/results/gpt2/   # predictions + results JSON
coconut/analysis.ipynb  # tables, plots, stats; executed with nbconvert, delivered with outputs
```
Official files (`coconut.py`, `run.py`, `dataset.py`) are **not modified**.

## 6. Steps (each is a tiny part, reported as "done with part i")
- Part 0a: check `tsrigo/coconut` contents (which tasks / stages) - read-only.
- Part 0b: count prompt token lengths for both few-shot prompts - tiny script, output to logs.
- Part 1: write `common.py`.
- Part 2: write `prompts.py`.
- Part 3: write `run_native.py`.
- Part 4: write `run_fewshot.py`.
- Part 5: write `score.py`.
- Part 6: write `run_all.slurm`.
- Part 7: smoke test on 5 examples per setting (you approve the run first).
- Part 8: full run (you approve; you or I run `sbatch`).
- Part 9: build and execute `analysis.ipynb` (tables, plots, stats).
- Part 10: rewrite the report in plain language with the new results; old Qwen results kept as "first attempt and what went wrong".
- Part 11: update `docs/requirment/*.md` to match.
- Part 12: fill `final_rep/main.tex` (6 pages) - may get its own plan.

If anything crashes: stop, report the reason, no fixes without approval.

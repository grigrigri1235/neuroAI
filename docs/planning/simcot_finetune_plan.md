# Plan: a short training run, so the model learns to read its own written step ("the last try")

Date: 2026-10-15. Status: APPROVED ("update plan accordingly and lets go"). Code written, not run.
Change after the user's review: train with the model's **own** step (from the explainer), not the true step, so
training matches the test (the explainer's step is wrong about half the time). 50,000 questions.

---

## 1. Why we do this

- We tried 4 ways to give the model a written step: T2, T2-gold, T2-in, T2-in-gold.
- All 4 made accuracy **worse** than giving no step (L = 44.7%).
- The reason is the same every time: **the model never saw a written step in its input while it was trained.**
  So a step confuses it, even a correct one.
- Prompting can not fix this. Training can.
- Before, we could not train (old GPU: 8GB). Now we have Newton (L40S, 47GB), and GPT-2 is small (124M).

Names (code name -> report name): L -> "no step", T2 -> "own step", T2-gold -> "true step".
"Gold" = the true answer/step from the dataset.

## 2. What we want to find out

- **Question 1 (your idea):** after training, does the model's **own** step help? (T2 better than L?)
- **Question 2 (best case):** after training, does the **true** step help? (T2-gold better than L?)
- **Question 3 (the fair check):** is the gain from learning to read steps, or just from "more training"?
  We compare with a second model that got the same training **without** steps.
- **Side check:** does training make the normal setting (L) worse?

## 3. The data

- `data/gsm_train.json`: the GSM8K train set as expanded by the iCoT authors (Deng et al.), the same data Coconut
  trained on. KR says ~385k questions (the original GSM8K train has ~7.5k). The job prints the real count.
- We shuffle it (seed 0) and take the first **50,000** questions. Both models use the same 50,000, same order.

## 4. Step 0: make the model's own step 1 for each training question (job part 1)

For each of the 50,000 training questions:
1. Input: `question` -> new line -> `[10 thoughts]` (same as L).
2. Fill the 10 thoughts with the **original** SIM-CoT model.
3. The explainer turns thought pair 1 into text = **own step 1** (e.g. `<<3+4=7>>`). Same code as the earlier runs.
4. Save it (`coconut/results/simcot/ft_train_steps.json`).
Check: on the first 20 **test** questions, this code gives the same step 1 as the saved run (job 1449771).

## 5. The two trained models

Both start from the original SIM-CoT model. Same 50,000 questions, same order, same settings. One difference:

| Model | What it sees in training |
|---|---|
| **FT-mix** | half the questions (coin flip, seed 0) **with its own step 1** written after the question, half **without** |
| **FT-plain** (the fair check) | all questions **without** a step |

All questions always have the 10 thoughts. "Without a step" = no written step, not no thoughts.
Why half and half: so the model stays good at both, and L vs T2 can be compared inside the same model.
Why FT-plain: our extra training (new learning rate, a 50k subset) could change accuracy by itself.
FT-plain gets the same extra training minus the steps, so we can tell the two apart.

## 6. What happens to ONE training question (step by step)

1. Read the question, its own step 1 (from step 0), and the true answer N.
2. **FT-mix only:** coin flip. Heads: step version. Tails: plain version. (FT-plain: always plain.)
3. Build the input, with **the same text as in the test**:
   - plain: `question` -> new line -> `[10 thoughts]`
   - step:  `question` -> new line -> `own step 1` -> new line -> `[10 thoughts]`
4. Add what the model should write: `### N` + the end token.
5. The model is graded **only on `### N`**, not on the question, the step, or the thoughts (same rule as
   Coconut training). The target is always the **true** answer, even when the own step is wrong. So the model can
   learn to use a step when it is right and to ignore it when it is wrong.
6. The official Coconut code (`Coconut.forward`) fills the 10 thoughts (each = the last hidden state, fed back in)
   and computes the error. Batches are made with the official `MyCollator`.
7. The weights are updated a little to make the error smaller.

Settings (fixed before running; never tuned on test): learning rate 0.00002 (warmup 100 steps, then linear down),
batch 32, one pass over the 50,000 questions (~1,560 updates), AdamW (weight decay 0.01), gradient clip 1.0.
We keep the weights from the end of the pass.

## 7. What happens to ONE test question (step by step)

All 1319 test questions. 3 models: original, FT-mix, FT-plain. Each model gets 3 settings:

| Setting | What the model gets | Example (question 0, answer 18) |
|---|---|---|
| **L** (no step) | `question` -> `[10 thoughts]` | - |
| **T2** (own step) | `question` -> **the original model's own step 1** -> `[10 thoughts]` | `<<3+4=7>>` (wrong step) |
| **T2-gold** (true step) | `question` -> **the true step 1** -> `[10 thoughts]` | `<<16-3-4=9>>` |

1. Build the input (same code as before, `simcot_run.run_round`).
2. Fill the 10 thoughts.
3. The model writes its answer (greedy, at most 64 tokens). Take the number after `###`.
4. Correct if it equals the true answer.

In T2 every model gets **the same saved step** (from job 1449771), made by the original model, exactly like the
training steps. So only the model changes, not the step.

## 8. How we decide

- Accuracy for each model and setting, with a 95% confidence range.
- Paired tests on the same questions (McNemar + bootstrap, as before):
  - inside FT-mix: T2 vs L (Q1), T2-gold vs L (Q2);
  - FT-mix vs FT-plain: on L, T2, T2-gold, and on the gaps (T2 - L) and (T2-gold - L) (Q3);
  - FT-mix vs original on T2.
- "Better" / "worse" only if the confidence range does not include 0.
- Also saved: training error every 50 updates, and L accuracy on 300 validation questions after training.

## 9. Safety checks inside the job (if a hard check fails, the job stops)

- **Check 0 (hard):** step 0 code gives the saved step 1 on the first 20 test questions.
- **Check 1 (hard):** training input = test input (`run_round`), for 5 test questions, both versions.
- **Check 2:** print one example and show that only `### N` is graded.
- **Check 3:** error before training on 200 plain and 200 step questions (expected: plain low, step higher).
- **Check 4:** speed and memory after 50 updates; print the time estimate.
- **Check 5 (hard):** the original model on the test set gives the old numbers (L 589, T2 454, T2-gold 509).

## 10. Files

- `coconut/experiments/simcot_ft_steps.py`: step 0 (own steps for the training questions).
- `coconut/experiments/simcot_finetune.py`: trains one model (`--mode mix` or `--mode plain`), saves it to
  `coconut/checkpoints_ft/<mode>.pt` (~0.5GB, **not in git**), training log to `coconut/results/simcot/ft_train_<mode>.json`.
- `coconut/experiments/simcot_ft_eval.py`: tests the 3 models, saves `coconut/results/simcot/ft_eval.json`.
- `coconut/experiments/simcot_ft.slurm`: step 0 -> train FT-mix -> train FT-plain -> test. Time limit 10 h,
  expected ~3-4 h (step 0 ~1.5-3 h, each training ~0.5-1 h, test ~15 min).
- `.gitignore`: `coconut/checkpoints_ft/`. No existing code is changed.

## 11. Parts

1. update this plan
2. `simcot_ft_steps.py`
3. `simcot_finetune.py`
4. `simcot_ft_eval.py`
5. `simcot_ft.slurm` + `.gitignore`
6. docs (KR, codebase_changes, memory)
7. commands for the user

## 12. Honest notes

- This is **one** short training run with **one** setting. If it does not help, we can not say "no training would
  help". We only say what we tried.
- If Q2 = yes but Q1 = no: "the model can learn to use a correct step, but its own steps are not good enough yet"
  (its step 1 was exactly right only 42.5% of the time). Still a real finding.
- After training, the model's thoughts change, but the explainer was trained on the old thoughts. That is why
  every step (train and test) comes from the **original** model. A "full loop" with the trained model making its
  own new steps is future work.

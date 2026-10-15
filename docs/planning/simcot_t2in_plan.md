# Plan: T2-in and T2-in-gold (step 1 inserted inside the question)

Date: 2026-10-15. Status: APPROVED ("proceed"), code written (`simcot_t2in.py` + `.slurm`), not run yet.
Follows `simcot_interleaved_plan.md` (section 2c) and the SIM-CoT results of job 1449771
(`coconut/results/simcot/simcot_run.json`).

## 1. Why

Job 1449771: L 44.7%, T2 34.4%, T2-gold 38.6%. In T2 the step was appended at the end of the question, on its own line,
as `<<16-3-4=9>>`. Even the true step hurt, so the layout seems to be the problem, not only the step quality.
The user's idea: put step 1 **inside** the question, like an extra fact, right before the last sentence (the actual
question). This looks more like a normal GSM8K question.

## 2. Settings (GSM8K test, all 1319, paired, same model `internlm/SIM_COT-GPT2-Coconut`)

| Setting | Input | Answer |
|---|---|---|
| L | question -> 10 latents | `### answer` (re-run; must equal job 1449771: 589/1319) |
| T2-in | question with the explainer's s1 inserted -> 10 latents | `### answer` |
| T2-in-gold | question with the true step 1 inserted -> 10 latents | `### answer` |

s1 = explainer(thought pair 1) of the L round, exactly as in job 1449771.

Example (question 0):
> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends
> every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
> **16-3-4=9.** How much in dollars does she make every day at the farmers' market?

## 3. Insertion rule (in code, one function)

- Step text: remove `<<` and `>>`, strip spaces, add `.` at the end if it has no `.`/`?`/`!`.
  `<<16-3-4=9>>` -> `16-3-4=9.`
- Where: after the last sentence end in the question (`.`, `?` or `!` followed by a space), i.e. right before the
  last sentence. Decimals like `2.5` are not split (no space after the dot).
- If the question has one sentence only: step goes at the start.
- If the step is empty (a few questions have no true steps): question unchanged. Counted and logged.
- If the explainer's s1 is an answer (`### N`) instead of a step: inserted as it is (`### N.`). Counted and logged.
- Known limit: an abbreviation like `Mr. ` could be taken as a sentence end. Rare; the log prints the first 20
  modified questions so we can check.

## 4. Metrics

- Accuracy + 95% Wilson CI for L, T2-in, T2-in-gold.
- Paired (McNemar + bootstrap CI, `score.pair_test`): T2-in vs L, T2-in-gold vs L, T2-in vs T2-in-gold.
- Placement effect, against the saved job 1449771 records: T2-in vs T2, T2-in-gold vs T2-gold
  (same questions, same s1, only the position and the `<<>>` change).
- Sanity: L re-run = saved L on every question; s1 = saved s1 on every question.
- A difference counts only if its CI excludes 0.

## 5. Files

- `coconut/experiments/simcot_t2in.py` (new; imports `run_round`, `decode_thoughts` from `simcot_run.py`).
  Output `coconut/results/simcot/simcot_t2in.json`, log `logs/simcot_t2in_<job>.log`.
- `coconut/experiments/simcot_t2in.slurm` (copy of `simcot_run.slurm`, 1 h limit; expected ~10 min).
- No change to existing files except docs.

## 6. Parts (after "proceed")

1. `simcot_t2in.py`
2. `simcot_t2in.slurm`
3. docs: `docs/requirment/codebase_changes.md`, `KNOWLEDGE_RETENTION.md` (SIM-CoT results + this run), memory
4. commands for lambda (commit/push) and Newton (pull, sbatch)

## 7. Honest note

Equations inside a question are still new to the model, so this may also hurt. If T2-in-gold beats T2-gold, the
position mattered. If T2-in-gold still beats L, written steps can help this model when placed this way.

# Phase 5 implementation plan: GPT-2 Coconut re-run (GSM8K only)

Status: APPROVED 2026-10-15 ("proceed with the plan"). Code written the same day, not run yet.
Source of truth for the *what*: `KNOWLEDGE_RETENTION.md` sections 5, 6, 9.
This file is the *how*: code design, file by file, plus problems found while reading the code.

---

## 0. Decisions from the user (2026-10-15)

- Claude Code works on **lambda** (it has issues on Newton). All code is written there and pushed with git.
  Everything runs on **Newton**: same paths (`/home/eran.b/neuroAI`), GPU **NVIDIA L40S 47GB**,
  Slurm script in the Technion Newton template format.
- The user runs all commands on Newton. Claude reads the logs (they come back to lambda through git).
- Newton has no conda yet: Miniconda goes to `/home/eran.b/miniconda3`, env from `environment_newton.yml`.
- Logs folder on Newton: `/home/eran.b/neuroAI/logs/` (KR section 10). Inside the repo, so it can be pushed.
- S2 gets 128 new tokens (one pass = the same budget as 2 stages x 64).
- One slurm file and one `sbatch` (no `submit_all.sh`). The smoke test is a gate: if it fails, nothing else runs.
- KR 5.2 gate "if native k=3 < 25%, stop before Part 2": Part 2 runs in the same job anyway (cheap),
  but if Part 1 is far from 34.1%, I stop and report before any Part 2 analysis.

---

## 1. Goal

Part 1: run the real GPT-2 Coconut checkpoint (`tsrigo/coconut/stage4_checkpoint_25`) on the GSM8K test
set (1319) in its own format, with 0 / 2 / 4 / 6 latents, and compare with the paper (34.1% at 6 latents).
Part 2: run the Phase 4 few-shot 2-stage experiment again, same prompts, on this model, with real
hidden-state feedback (S3), a pause-token control (S5), a no-token control (S4), and two baselines (S1, S2).
Main test: S3 vs S5. Paired stats on all 1319 questions.

| Setting | What it is |
|---|---|
| S1 | raw question, `Question: {q}\nAnswer:`, one pass |
| S2 | `Question: {q}\nLet's think step by step. `, one pass, no few-shot examples |
| S3 | few-shot 2-stage loop, 2 latents per stage filled with the previous hidden state (real Coconut) |
| S4 | same loop, no latent tokens at all |
| S5 | same loop, same latent tokens, plain token embeddings, no feedback (pause control) |

---

## 2. Things found while reading the code

**F1. The official `Coconut.forward` cannot handle two separate latent blocks. Stage 2 of S3 would crash.**
- `coconut.py` lines 110-117: after the first latent, each pass moves forward by exactly **one** token,
  so it assumes all `<|latent|>` tokens sit next to each other.
- Stage-2 prompt has 2 blocks (stage-1 block + stage-2 block, ~70 tokens apart).
  Trace with latents at [a, a+1, b, b+1]: pass 2 runs only position a+1, then reads
  `hidden_states[b - 1 - (a + 1)]` of a length-1 tensor -> `IndexError`.
- Fix: our own `fill_latents()` in `experiments/common.py` (official file not touched). For each latent
  position p in order: run the model (with KV cache) up to p-1, take the last hidden state
  (`hidden_states[-1]`, after the final layer norm, same as Coconut), put it in slot p. Any number of blocks.
- Checks: C2 (equal to `Coconut.forward` where it works: one block), C4 (stage-2 consistency),
  C4b (info: the official forward on a 2-block prompt, expected IndexError).

**F2. The `<|latent|>` input embedding was never trained as an input.**
- In training every `<|latent|>` slot is replaced by a hidden state before the model sees it, so its input
  embedding gets no gradient from the input side. GPT-2 ties input/output embeddings, so the row only moves a
  little as an output class (plus weight decay). It started as a copy of `<<` (`run.py` line 149).
- So S5 = "same tokens, no feedback" with a nearly untrained pause embedding, **not** the paper's trained
  "pause as thought". Still a fair "remove only the feedback" control; the report must say this.
- Info check C8: cosine similarity of the 3 marker rows vs the GPT-2 `<<` row.

**F3. The 3 added tokens are not "special".** `add_tokens` (not `add_special_tokens`), so
`skip_special_tokens=True` keeps them. If the model writes `<|latent|>` in stage 1 and we paste it into stage 2,
it becomes a new latent slot. Fix: remove the 3 marker strings from generated text in **all** settings and
count them. Also every prompt is checked to hold exactly the expected number of latents (0 / 2 / 4).

**F4. `dataset.get_dataset` crashes if more than one GPU is visible** (`dataset.py` line 46 calls
`dist.get_rank()` with no process group). The job asks for exactly 1 GPU.

**F5. `coconut/data/gsm_test.json` does not exist yet.** Setup runs `preprocessing/gsm_icot.bash`.

**F6. Prompts end with `"Answer: "` (trailing space).** GPT-2 BPE makes this a lone space token, rare
before a word. Kept verbatim (KR: same prompts), same in S3/S4/S5. Risk; the smoke outputs will show it.

**F7. `Coconut.generate` has no KV cache; our loop uses one.** Float noise can, rarely, flip an argmax.
C3 compares both modes against `Coconut.generate`.

---

## 3. Code layout

All new code in `coconut/experiments/`. Official files (`coconut.py`, `run.py`, `dataset.py`, `utils.py`)
are **not** modified. Relative paths are resolved against `coconut/` (no hard-coded `/home/...` in Python).

```
coconut/experiments/
  common.py        load model, fill_latents, greedy loop, text helpers; as a script: download + load check (CPU)
  prompts.py       FEW_SHOT_PROMPT (verbatim), S1-S5 settings and prompt builders, stop strings
  run_native.py    Part 1 sweep k = 0..3, official dataset.py input + official Coconut.generate
  run_fewshot.py   Part 2, --setting S1..S5, length guard
  score.py         accuracy, Wilson CI, exact McNemar, paired bootstrap -> results.json
  smoke_test.py    checks C1-C8 (section 5), exit code 1 if a required check fails
  run_gpt2.slurm   the one Newton job
environment_newton.yml   lean env (same versions as environment.yml, no tensorflow / wandb)
coconut/results/gpt2/    native.json, fewshot_S1..S5.json, results.json, smoke/
logs/                    slurm + per-step logs (Newton)
```

### 3.1 `common.py`
- `load_tokenizer()`: GPT-2 tokenizer, `pad = eos`, `add_tokens` in order `<|start-latent|>`, `<|end-latent|>`, `<|latent|>`.
- `load_model(tokenizer, ids, device, ckpt=None)` -> `(model, info)`: like `run.py` lines 103-167 (resize, copy `<<`
  rows, wrap in `Coconut`, `torch.load(weights_only=True)`, `load_state_dict(strict=False)`, print result).
  Raises if keys do not match. Only explained exceptions are allowed: causal-mask buffers (`.attn.bias`), and a
  missing key that shares its tensor with a loaded key (GPT-2's `wte`, `lm_head` and Coconut's `embedding` are one
  matrix under three names).
- `fill_latents(model, input_ids)` (F1); `get_embeds(model, input_ids, mode)`: `latent` = fill, `plain` = token embeddings.
- `greedy_generate(model, tokenizer, embeds, max_new_tokens, stop_strings, use_cache=True)`: argmax each step,
  stop on EOS (not kept), a stop string (text cut before it), or max tokens. **Same loop for S1-S5.**
- `clean_text` (F3), `last_number` (commas removed; a "-" right after a digit/letter is a minus sign, not a negative
  number), `gsm8k_gold` (text after `####`), `same_number` (float compare), `run_meta` (versions, GPU, job id).

### 3.2 `prompts.py`
- `FEW_SHOT_PROMPT` verbatim from `eval_qwen_1.5b_gsm8k_less_handholding.py`; `few_shot(n)` keeps the first n examples.
- S1 `f"Question: {q}\nAnswer:"`; S2 `f"Question: {q}\nLet's think step by step. "`.
- `turn(lat)` = `"\n\n" + STEP_Q + "\n" + (LAT + "\n" if lat) + "Answer: "`.
  Stage 1 = `few_shot(n) + "\n\nQuestion: " + q + turn(lat)`; stage 2 = stage 1 + stage-1 answer + `turn(lat)`.
- Stop strings: `"\nQuestion:"`, `"--- Example"` (+ EOS).

### 3.3 `run_native.py` (Part 1)
- Input via `get_dataset("data/gsm_test.json")` + `get_question_latent_dataset(k, ...)` with `args/gsm_coconut.yaml`
  (`c_thought 2`, `max_latent_stage 3`, `pad_latent_to_max True`) -> 2k latents.
- Official `model.generate(..., max_new_tokens=64)`; score = `run.py` line 469 rule.
- Saves per example: `idx, k, n_latents, prompt_len, output_text, extracted, gold, correct, n_new_tokens, seconds`.

### 3.4 `run_fewshot.py` (Part 2)
- Data `data/gsm8k_test.json` (same file and question text as Phase 4). Gold = text after `####`.
- `--n-shots auto`: 3 if the longest possible stage-2 prompt (all 1319 questions, latent template, stage-1 answer
  = 64 tokens, +8 margin) + 64 <= 1024, else 2. Same for S3/S4/S5 (KR 5.4).
- Prediction = last number of the cleaned stage-2 text (S1/S2: of the one output).
- Saves per example: `idx, question, gold, stages[{prompt_len, raw, cut, text, stop_reason, n_new_tokens, markers_removed}], pred, correct, seconds`.

### 3.5 `score.py`
- Native: accuracy + 95% Wilson CI per k, mean new tokens, mean seconds.
- Few-shot: accuracy + Wilson CI, no-number rate, tokens and stop reasons per stage, markers removed, n_shots.
- Pairs S3-S5 (H1), S3-S4 (H2): discordant counts, exact McNemar (binomial test on discordant pairs, scipy),
  paired bootstrap 95% CI of the difference (10k resamples, seed 0). Verdict only if the CI excludes 0.
- Checks that all few-shot runs are on the same questions (paired) before testing.

### 3.6 Fixed settings (same for all)
Greedy, seed 0, cudnn deterministic, fp32, batch size 1, KV cache on. 64 new tokens for S1 and per stage
(S3/S4/S5); 128 for S2.

---

## 4. Newton job (`run_gpt2.slurm`)

- Technion template: `--cpus-per-task=8`, `--gres=gpu:L40:1` (Newton's Slurm name for the L40S; `L40S` was rejected; no `--nodelist`), mail, job name, `-o/-e` into `logs/`.
  Added: `--mem=32G`, `--time=04:00:00`. Matlab module line dropped. `source .../conda.sh` added before
  `conda activate` (needed in batch jobs).
- `HF_HUB_OFFLINE=1`: GPT-2 and the checkpoint are downloaded during setup on the login node (compute nodes may
  have no internet). `OMP_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.
- Order: GPU check -> smoke checks -> smoke runs (20 questions, native + S1-S5) -> smoke score ->
  full native (alone, clean timing) -> full S1-S5 (5 processes in parallel on the one GPU) -> full score.
  Stops at the first failure. Each step logs to `logs/gpt2_<step>_<jobid>.log`.
- Expected time: about 1 to 1.5 hours (smoke ~10 min, native ~20-30 min, S1-S5 in parallel ~20-40 min).

---

## 5. Smoke test checks (`smoke_test.py`)

- C1: checkpoint loads; keys start with `base_causallm.`; missing/unexpected keys empty (or explained, see 3.1).
- C2: `fill_latents` == `Coconut.forward().inputs_embeds` (max relative diff < 1e-4) on 20 native k=3 prompts
  and 20 S3 stage-1 prompts.
- C3: our `greedy_generate` (cache on, and off) gives the same tokens as `Coconut.generate` on 20 native k=3
  questions (until EOS); pass if >= 19/20 for each mode (allows one float-noise flip).
- C4: stage 2: same token prefix, 4 latents, and the stage-1 thoughts inside stage 2 equal the stage-1 run
  (relative diff < 1e-4). C4b (info): the official `Coconut.forward` on a 2-block prompt.
- C5: S3 vs S5 embeddings differ exactly at the `<|latent|>` positions.
- C6: prompt lengths over all 1319 questions: worst-case stage-2 for 3 and 2 shots, chosen n_shots, S2 + 128 <= 1024.
- C7 (info): `gsm_test.json` vs `gsm8k_test.json`: counts, same question text, same gold.
- C8 (info): marker embeddings vs the GPT-2 `<<` row (F2).
- Required: C1-C6. Report: `results/gpt2/smoke/smoke_report.json`. If anything fails or crashes: stop and report (rule 7).

---

## 6. Parts

This session (Claude on lambda, no bash):

| Part | What |
|---|---|
| 0 | this plan update |
| 1 | `experiments/common.py` |
| 2 | `experiments/prompts.py` |
| 3 | `experiments/run_native.py` |
| 4 | `experiments/run_fewshot.py` |
| 5 | `experiments/score.py` |
| 6 | `experiments/smoke_test.py` |
| 7 | `experiments/run_gpt2.slurm` (Newton template) |
| 8 | `environment_newton.yml`, `.gitignore` (GSM8K train JSON is too big for GitHub) |
| 9 | `docs/requirment/codebase_changes.md` entry |
| 10 | `KNOWLEDGE_RETENTION.md` update |
| 11 | memory update |

Then (user on Newton): push from lambda, pull on Newton, Miniconda, env, data, model download + load check, `sbatch`,
push logs/results back. Then (Claude): read logs; if all good, `analysis.ipynb` (tables, plots, failure cases by
fixed rule), executed with nbconvert. Later: report, docs rewrite, `final_rep/main.tex` (KR section 9).

Optional later (KR 5.7): decode each latent with the LM head (top-5), like paper Fig. 9.

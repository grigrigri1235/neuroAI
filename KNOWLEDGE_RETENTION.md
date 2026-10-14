# KNOWLEDGE RETENTION: neuroAI Coconut Project

Single source of truth. Read this first on any machine (local or the new GPU server).
Last updated: 2026-10-14.

---

## 0. Status in one screen

- **Paper:** Coconut, "Training LLMs to Reason in a Continuous Latent Space" (Hao et al., Meta). Copy in `og_coconut.md` / `og_coconut.pdf`.
- **Course task:** Part 1 reproduce, Part 2 extend (see section 1).
- **Where we are:** the old Qwen 1.5B experiment had a fatal bug (no real latents). We are re-running **the same experiment, same prompts, on the real GPT-2 Coconut checkpoint** with real hidden-state feedback.
- **Dataset:** GSM8K only for the main runs. ProsQA dropped. EntailmentBank: open question (section 5.6).
- **Where it runs:** a new strong GPU server (time limits locally).
- **Next step:** Part 0 of the steps list (section 9).

---

## 1. Course requirements (from `docs/requirment/requirment.md`)

- **Part 1 Reproduce:** implement the core method and reproduce a key result.
  - We have an **exemption from training** (GPT-2 fine-tuning is too expensive). Inference-only reproduction with a released checkpoint is our Part 1.
- **Part 2 Extend:** propose 2-4 follow-up ideas, pick one, test it with a small well-defined experiment: hypothesis, metrics, minimal code, results (plots / tables / failure cases).
- **Deliverables:** 6-page report (template in `final_rep/main.tex`, CogSci format) + runnable code.
- **Oral check:** the lecturer may meet for 30 min to check we understand and own all code. AI use is allowed only if we understand everything.
  - So: report and docs must be in plain words, every claim must match the code, no hype words.

---

## 2. Paper essentials we rely on

- **Method:** in "latent mode" the last hidden state of position t-1 is fed as the **input embedding** of position t (no token decoding). Marked by `<bot>` / `<eot>` (in code: `<|start-latent|>`, `<|end-latent|>`), slots are `<|latent|>`.
- **Training curriculum:** stage k replaces the first k reasoning steps with k*c latents. GSM8K: c = 2, 3 stages + 1 extra stage, so the final model uses **6 latents** and no language steps.
- **Paper GSM8K results (GPT-2):** CoT 42.9%, No-CoT 16.5%, iCoT 30.0%, Pause token 16.4%, **Coconut 34.1%**, w/o curriculum 14.4%, w/o thought 21.6%, pause-as-thought 24.1%.
- **Paper Fig. 8 / Fig. 3 idea:** same weights, change only how many latents are used at inference, accuracy changes. That is what our Part 1 reproduces (on GSM8K).
- **Inference:** `<bot>` right after the question, fixed number of latents, then `<eot>`, then the model writes `### answer`. Greedy decoding.

---

## 3. Project history (short, with lessons)

| Phase | What | Result | Lesson |
|---|---|---|---|
| 1 | Train GPT-2 Coconut on ProsQA from scratch (8GB RTX 2080, batch 2 x grad-acc 64) | Stopped, >3 days | Training is out of budget, use released checkpoints |
| 2 | Train GPT-2 to trigger latents by itself (interleaved, GSM8K) | 1.44% (vs 8.72% fine-tuned baseline), repetition loops | Too hard for 124M + our compute. Logs in `coconut/wandb/` (local only) |
| 3 | GPT-2 `tsrigo/coconut` on EntailmentBank, Python script builds the proof tree | 97% structure accuracy **with and without latents** | The scaffold did the work, latents added nothing. Honest negative result |
| 4 | Qwen2.5-1.5B (`CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch`), 4 settings, GSM8K + EntailmentBank, 100 samples | GSM8K: S1 34%, S2 40%, S3 12%, S4 6%. EB BLEURT: 0.405 / 0.427 / 0.401 / 0.392 | **Invalid as a Coconut test**, see section 4 |

Phase 4 code, predictions, scores and slurm logs are kept in `coconut/` as the "first attempt" record.

---

## 4. What went wrong in Phase 4 (must be explained in the report)

1. **No continuous thought.** `<c_thought>` was put in the prompt as text and `model.generate()` was called. Nothing fed hidden states back. At best this is the paper's Pause-token baseline. If the token was missing, the script added it with a **random** embedding.
2. **Unknown model.** The Qwen model card is the default HF template: "sft", trained on "TongSearch_Coconut@16_v2" (a search/retrieval dataset), 3 epochs, 64 GPUs. No description of how latents were trained. Cannot be called a Coconut model.
3. **Not significant.** 12/100 vs 6/100, p ~ 0.2. BLEURT gaps ~0.01.
4. **No stop rule.** 150 free tokens per stage, the model rambled into "Example 1... Example 2...", the ramble went back into the context. The scorer takes the **last number** of the last stage, which can come from the examples (the "42").
5. **Stage 2 format bug.** Stage 2 context dropped the stage-1 `Question: What can we deduce in one logical step?` line, so stage 2 no longer matched the examples.
6. **Possible sampling.** No seed and `do_sample` not set; Qwen-Instruct configs often sample by default.
7. **Metric name.** The folder `coconut/bleurt-large-512/` actually holds **BLEURT-20** (`bleurt_config.json` says so). Call it BLEURT-20.
8. **Docs over-claim.** "doubled", "decisive proof", "100% identical". To be rewritten in plain words.

---

## 5. The new experiment (decided 2026-10-14)

### 5.1 Model
- `openai-community/gpt2` + weights `tsrigo/coconut` / file `stage4_checkpoint_25`.
  - stage 4 + epoch 25 matches the paper GSM8K config (`args/gsm_coconut.yaml`: c_thought 2, max_latent_stage 3, 25 epochs). So this is very likely the **final GSM8K Coconut model**. Verify in Part 0 (keys start with `base_causallm.`? accuracy near 34%?).
- Load exactly like `coconut/run.py` lines 103-167:
  - add tokens `<|start-latent|>`, `<|end-latent|>`, `<|latent|>` (in this order), `resize_token_embeddings`,
  - wrap in `Coconut(model, latent_id, start_id, end_id, eos_id)`,
  - `load_state_dict(saved_weights, strict=False)` and **print the result** (missing / unexpected keys must be empty except known ones).
- fp32 is fine. GPT-2 context limit = **1024 tokens**.

### 5.2 Part 1: reproduction (inference only, GSM8K)
- Data: official iCoT GSM8K test (1319). Get it with `bash preprocessing/gsm_icot.bash` (creates `data/gsm_test.json`).
- Input: `question + "\n"` + `<|start-latent|>` + `<|latent|>` x (2k) + `<|end-latent|>`. Model writes the rest, ends with `### answer`.
- Sweep k (stages) in {0, 1, 2, 3} -> 0 / 2 / 4 / 6 latents. Same weights.
- Score exactly like `run.py` line 469: `text.split("#")[-1].replace(",", "").strip() == answer`.
- Expect k = 3 close to **34.1%**. If far off, stop and report before Part 2.
- Plot: accuracy vs number of latents (like paper Fig. 8 idea).

### 5.3 Part 2: the 1.5B experiment, ported (same prompts)
Same few-shot prompt and same forced 2-stage loop as the Qwen scripts (hand-holding kept on purpose).
Only the latent token text changes: `<c_thought><c_thought>` -> `<|start-latent|><|latent|><|latent|><|end-latent|>`.

| Setting | Prompt | Latent slots |
|---|---|---|
| S1 Raw | `Question: {q}\nAnswer:` | none |
| S2 Single pass CoT | `Question: {q}\nLet's think step by step. ` | none |
| S3 Real latents | 2-stage loop (below) | filled with **previous hidden state** (Coconut forward) |
| S4 No tokens | 2-stage loop, latent block removed | none |
| S5 Pause control (NEW) | 2-stage loop, same latent block | **plain token embeddings, no feedback** |

- **S3 vs S5** = does continuous feedback help beyond just extra tokens? (main test)
- **S3 vs S4** = the old comparison, done right.

**2-stage loop (S3 / S4 / S5).** `LAT` = `<|start-latent|><|latent|><|latent|><|end-latent|>` for S3/S5, empty line removed for S4.

Stage 1 prompt:
```
{FEW_SHOT_PROMPT}

Question: {q}

Question: What can we deduce in one logical step?
{LAT}
Answer: 
```
Stage 2 prompt (keeps the full stage-1 turn, fixing bug 4.5):
```
{FEW_SHOT_PROMPT}

Question: {q}

Question: What can we deduce in one logical step?
{LAT}
Answer: {stage1_text}

Question: What can we deduce in one logical step?
{LAT}
Answer: 
```
- In S3, stage-1 latents are recomputed inside stage 2 (causal model, so same vectors). Total latents seen = 4.
- Score = last number of stage-2 text (commas removed first).

**GSM8K FEW_SHOT_PROMPT (verbatim from `coconut/eval_qwen_1.5b_gsm8k_less_handholding.py`):**
```
--- Example 1 ---
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Question: What can we deduce in one logical step?
Answer: Since there are 60 minutes in an hour, Weng earns $12 / 60 = $0.20 per minute.

Question: What can we deduce in one logical step?
Answer: Since she babysat for 50 minutes at $0.20 per minute, she earned 50 * 0.20 = $10. So she earned $10.

--- Example 2 ---
Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

Question: What can we deduce in one logical step?
Answer: Today Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today, for a total of 12 + 24 = 36 pages read so far.

Question: What can we deduce in one logical step?
Answer: There are 120 - 36 = 84 pages left. Half of the remaining pages is 84 / 2 = 42 pages. So she should read 42 pages tomorrow.

--- Example 3 ---
Question: Hans booked a room in a hotel. The hotel has 10 floors with 10 identical rooms on each floor. Because of an accident, the last floor is unavailable for the guests. Considering there are no other guests, in how many different rooms could Hans be checked in?

Question: What can we deduce in one logical step?
Answer: With 10 floors and 10 rooms per floor, the hotel has a total of 10 * 10 = 100 rooms.

Question: What can we deduce in one logical step?
Answer: The last floor of 10 rooms is unavailable, so there are 100 - 10 = 90 available rooms. So Hans could be checked into 90 different rooms.
```
(The EntailmentBank prompt is in `coconut/eval_qwen_1.5b_entailment_fewshot.py`, if 5.6 says yes.)

### 5.4 Fixes applied to ALL settings the same way
- Greedy decoding, fixed seed (0), `torch.backends.cudnn.deterministic = True`.
- `max_new_tokens = 64` per stage / per S1-S2 answer (paper uses 64 for GSM8K). S2 may need 128: check in smoke test, same for all if changed.
- Stop strings: cut at `\nQuestion:`, `--- Example`, or EOS.
- Length guard: if any full stage-2 prompt + 64 > 1024 tokens, use 2 few-shot examples instead of 3 **for every setting**, and write it in the report.
- Save per example: prompt token length, raw output per stage, cut output, extracted answer, gold.
- Data: **full GSM8K test (1319)** for every setting (we have the GPU now). Same questions in every setting (paired).

### 5.5 Hypotheses, metrics, stats
- H1: S3 > S5. H2: S3 > S4. (Also report S1, S2 for context.)
- Metric: exact-match accuracy.
- Stats: **paired McNemar test** per pair (S3-S5, S3-S4), 95% CI on the accuracy difference (paired bootstrap, 10k resamples).
- A difference counts only if the CI excludes 0. Otherwise write "no clear difference".
- Failure cases: 3-5 hand-picked examples per setting, chosen by a fixed rule (e.g. first 5 where S3 and S5 disagree), not cherry-picked.

### 5.6 OPEN QUESTION: EntailmentBank in the new runs?
- Default: **GSM8K only** ("GSM8K is really enough"). The GSM8K-trained GPT-2 is out of domain on EntailmentBank anyway.
- If kept: all 340 of `data/entailmentbank/test_decomposed.json`, same S1-S5, BLEURT-20 + paired bootstrap CI.

### 5.7 Optional (if time): look inside the latents
- Decode each continuous thought with the LM head (top-5 tokens), like paper Fig. 9.
- Count how often the top token is a number from the question or an intermediate result.

---

## 6. Implementation notes

- **Do not modify** official files: `coconut/coconut.py`, `run.py`, `dataset.py`, `utils.py`.
- **One greedy loop for all modes**, so the only difference between settings is the input embeddings:
  1. Build `input_ids` for the prompt.
  2. Get `inputs_embeds`:
     - S3: `Coconut.forward(...)` returns `inputs_embeds` with latent slots filled by hidden states.
     - S5: `embedding(input_ids)` (latent tokens keep their plain learned embedding).
     - S1/S2/S4: `embedding(input_ids)` (no latent tokens present).
  3. Greedy loop on `base_causallm(inputs_embeds=..., past_key_values=...)`, append argmax token embeddings, stop on EOS / stop string / max tokens.
- `Coconut.forward` needs `labels` and `position_ids` (pass `input_ids.clone()` and `arange`), batch size 1 (see `Coconut.generate`).
- `Coconut.forward` handles latents in the middle of a prompt (it processes text after the last latent in its final pass). Check this in the smoke test: S3 output must differ from S5 on at least some examples, and S3 with 0 latents must equal plain decoding.
- Sanity checks in Part 7 (smoke test):
  - native k=3 on 20 questions gives sensible `### answer` outputs,
  - our loop with mode "latent" reproduces `Coconut.generate` tokens on the native format,
  - token-length report for the longest stage-2 prompt.
- Speed: GPT-2 is tiny. On a strong GPU run settings as separate processes in parallel.

---

## 7. Server setup (new GPU server)

```bash
git clone git@github.com:grigrigri1235/neuroAI.git   # SSH remote
cd neuroAI
conda env create -f environment.yml                   # env name: neuroAI_project (python 3.12, torch 2.6.0, transformers 4.52.4)
conda activate neuroAI_project
pip install scipy statsmodels matplotlib jupyter nbconvert

# GSM8K in Coconut format (for Part 1)
cd coconut && bash preprocessing/gsm_icot.bash && cd ..

# Coconut GPT-2 checkpoint
python -c "from huggingface_hub import hf_hub_download as d; print(d('tsrigo/coconut','stage4_checkpoint_25'))"
```
Only if EntailmentBank / BLEURT is needed (not in git, too big for GitHub):
```bash
pip install git+https://github.com/google-research/bleurt.git
cd coconut && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip && unzip BLEURT-20.zip
#   old scripts load the folder name "bleurt-large-512": rename BLEURT-20 -> bleurt-large-512, or update the path
git clone https://github.com/allenai/entailment_bank.git   # only the old Phase 3 scorer used this
```
Env vars used on the old cluster (copy into new slurm / run scripts):
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0                      # stop transformers from importing TensorFlow
# BLEURT scoring only: run on CPU, TF and PyTorch CUDA clash in one process
export CUDA_VISIBLE_DEVICES=""
```
Old local paths that must be changed on the server: `/home/eran.b/miniconda3/...` and `/home/eran.b/neuroAI/...` in slurm files and `preprocessing/parse_entailment.py`.

---

## 8. Repo map

```
KNOWLEDGE_RETENTION.md        <- this file
og_coconut.md / .pdf          paper
environment.yml               conda env
final_rep/                    6-page CogSci LaTeX template (still the template text)
docs/
  requirment/requirment.md    course task
  requirment/*.md             old summaries (over-claiming, to be rewritten)
  report/entailmentbank_report.md   old report (Phase 3 + 4), to be rewritten
  planning/gpt2_coconut_rerun_plan.md   first draft of the new plan (this file is newer)
  planning/less_handholding_plan.md     Phase 4 design (source of the prompts)
  planning/entailmentbank_experiment_plan.md   Phase 3 design
coconut/                      official Meta code + our scripts
  coconut.py run.py dataset.py utils.py   official, do not edit
  args/*.yaml                 official configs (prosqa_coconut.yaml edited for 8GB GPU)
  data/                       prosqa_*.json, gsm8k_test.json (HF format, '####' answers), entailmentbank/
  preprocessing/              gsm_icot.bash (Coconut GSM8K format), parse_entailment.py (ours)
  eval_qwen_1.5b_*.py         Phase 4 scripts (first attempt, kept as record)
  predictions_*.json, score_*  Phase 4 outputs and scores
  slurm_logs/                 Phase 4 logs
  wandb/, bleurt-large-512/, entailment_bank/   LOCAL ONLY (too big / nested repo), see section 7
planned:
  coconut/experiments/{common,prompts,run_native,run_fewshot,score}.py + run_all.slurm
  coconut/results/gpt2/       predictions + results JSON
  coconut/analysis.ipynb      tables, plots, stats (executed, with outputs)
```

---

## 9. Steps (each is a tiny part, report "done with part i")

- Part 0a: download checkpoint, print state-dict keys and `load_state_dict` result.
- Part 0b: count token lengths of the longest stage-2 prompt (GSM8K test).
- Part 1: `experiments/common.py` (load model, greedy loop, 3 embedding modes, stop strings, seed).
- Part 2: `experiments/prompts.py` (few-shot prompt + builders for S1-S5).
- Part 3: `experiments/run_native.py` (Part 1 sweep k = 0..3).
- Part 4: `experiments/run_fewshot.py` (`--setting S1..S5`).
- Part 5: `experiments/score.py` (accuracy, McNemar, bootstrap CI, results JSON).
- Part 6: run script / slurm for the server.
- Part 7: smoke test (20 questions, all settings + sanity checks in section 6). **Needs approval to run.**
- Part 8: full run. **Needs approval to run.**
- Part 9: `analysis.ipynb`: tables, plots, failure cases; execute with
  `jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True coconut/analysis.ipynb`
- Part 10: rewrite the report in plain words (old Qwen run = "first attempt and what went wrong").
- Part 11: update `docs/requirment/*.md`.
- Part 12: fill `final_rep/main.tex` (6 pages).

---

## 10. Working rules for any agent on this project (from the user's CLAUDE.md)

- Start each message with a short recap of these rules.
- Plan first (in `docs/planning/`), then **stop and wait for explicit approval**. No code edits and no commands without approval.
- After approval: tiny parts, say "done with part i" after each, keep going.
- Bash output always to a file in a logs folder, never the terminal (old path: `/home/eran.b/PAR_FINAL/PAR_FINAL/alt-project/logs`, pick an equivalent on the server).
- If code crashes: do **not** fix it alone. Report the reason and stop.
- Short, simple language. Reports must have data, code and plots.
- Notebooks must be executed end-to-end and delivered with outputs.
- Git: SSH remote. Ask before pushing.

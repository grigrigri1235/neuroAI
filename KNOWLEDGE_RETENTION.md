# KNOWLEDGE RETENTION: neuroAI Coconut Project

Single source of truth. Read this first on any machine (local, or the Newton GPU cluster).
It is written so a new agent can continue the project **without reading any other doc**.
Last updated: 2026-10-15.

Contents:
0. Status · 1. Course requirements · 2. Paper essentials · 3. Project history (all phases) ·
4. What went wrong in Phase 4 · 5. The new experiment · 6. Implementation notes · 7. Server setup (Newton) ·
8. Repo map · 9. Steps · 10. Working rules · 11. Part 2 follow-up ideas · 12. Old results in detail ·
13. Data formats · 14. Lessons learned · 15. Glossary · **16. Phase 5 results + model search (2026-10-15)**

How we work (lambda / Newton / conversation rules, exact commands, pitfalls): `workflow.md`.

---

## HANDOFF (read first): exactly where we stopped (2026-10-15, end of a long session)

**New session: read this block, then `workflow.md` (how we work, commands, Newton details), then section 16 (all
results).** Global rules: `~/.claude/CLAUDE.md` (recap them at the start of every message; no bash by Claude).

1. **SIM-CoT job 1449771: DONE** (results in **section 16.6**). G0/G1/G2 pass (explainer present, 44.7% at
   10 latents, decoded thoughts are readable `<<a*b=c>>` steps). Main: **T2 34.4% and T2-gold 38.6%, both lower than
   L 44.7%** (CIs exclude 0). Even the true step hurts, so the layout (a `<<...>>` line after the question, never seen
   in training) is the likely cause. The user was disappointed by the negative result and chose one more try.
2. **T2-in job: WRITTEN, NOT RUN** (user approved 2026-10-15). Plan `docs/planning/simcot_t2in_plan.md`. Files
   `coconut/experiments/simcot_t2in.py` + `.slurm`. Step 1 written as `16-3-4=9.` and put **inside the question,
   before its last sentence**. Settings L / T2-in / T2-in-gold (1319, paired) + placement effect vs the saved T2 /
   T2-gold. Needs `coconut/results/simcot/simcot_run.json` on Newton (it is there). ~10 min.
   - lambda: `git add coconut/experiments/simcot_t2in.* docs KNOWLEDGE_RETENTION.md && git commit && git push`
   - Newton: `git pull`, `sbatch --test-only coconut/experiments/simcot_t2in.slurm`, then `sbatch ...`.
   - Push `logs` + `coconut/results` back, `git pull` on lambda.
   - When results arrive (`logs/simcot_t2in_<job>.log`, `coconut/results/simcot/simcot_t2in.json`): first the
     `checks` (L and s1 equal the saved run on all 1319; look at the 20 printed questions: is the step in the right
     place?), then T2-in(-gold) vs L and vs T2(-gold). Report plainly; update 16.6 and memory.
   - If it crashes: report the reason, do not fix it alone (rule 7).
   - Dropped by the user: "Option B" (measure readable thoughts on all 1319, "a totally different idea"). Not
     run: "Option A" (step 1 after `<eot>`), replaced by the user's T2-in design.
3. **Report (`final_rep/main.tex`, CogSci template, max 6 pages, include an AI-use statement): NOT STARTED.**
   Next: write a short report plan in `docs/planning/report_plan.md` (sections, tables/figures, page budget), stop for
   the user's OK, then write. Content:
   - Part 1 = GPT-2 reproduction + feedback-vs-plain control (16.1).
   - Part 2 = the interleaving test (SIM-CoT: T2, T2-gold, T2-in, T2-in-gold, section 16.6) + the documented
     failures (few-shot floor, Qwen 7B broken, Qwen3-4B yes/no). Framing for a negative result: "on a model trained
     only on question -> latents -> answer, inserting a written step (even the true one) lowers accuracy; a fair
     test needs a model trained with the interleaved layout (training out of budget)". Do not write "proven false".
   The old reports/docs over-claim; rewrite in plain words (section 4.11).
   Figures need an executed notebook (`coconut/analysis.ipynb`, run with nbconvert by the user, global
   RUNTIME/PRE-EXECUTION mandates).
4. **Decisions already made by the user (do not re-ask):**
   - GSM8K only.
   - Minimal 2-round design (L, T2, T2-gold); T2-plain and A dropped.
   - Mail `eran.b@campus.technion.ac.il`.
   - Anaconda ToS accepted.
   - Keep low-level risk notes short.
   - The user accepts reporting the failures as documented effort.
5. **Open offers, not chosen yet:** the cheap GPT-2 ideas in 16.5 (thought transplant, per-thought ablation, noise).

---

## 0. Status in one screen

- **Paper:** Coconut, "Training Large Language Models to Reason in a Continuous Latent Space" (Hao et al., FAIR Meta, code: github.com/facebookresearch/coconut). Copy in `og_coconut.md` / `og_coconut.pdf`.
- **Course task:** Part 1 reproduce (training exemption), Part 2 extend (see section 1).
- **Where we are:** the old Qwen 1.5B experiment (Phase 4) had a fatal bug: no real latents (section 4). We are re-running **the same experiment, same prompts, on the real GPT-2 Coconut checkpoint** with real hidden-state feedback, plus one new control (S5).
- **Dataset:** **GSM8K only.** ProsQA dropped (baselines already ~97%, no room to show a difference). EntailmentBank dropped (section 5.6).
- **Where it runs:** Newton GPU cluster (Technion, Slurm), GPU **NVIDIA L40S 47GB**, section 7.
  Claude Code works on **lambda** (it has issues on Newton): code is written on lambda, pushed with git, and the
  user runs it on Newton (same paths, `/home/eran.b/neuroAI`). Logs and results come back through git.
- **NEWEST (2026-10-15):** SIM-CoT GPT-2 ran (section 16.6): 44.7% at 10 latents, readable thoughts, but a written
  step 1 before round 2 lowers accuracy (T2 34.4%, T2-gold 38.6%). Follow-up T2-in (step inside the question)
  written, not run. See the HANDOFF block.
- **LATEST (2026-10-15, night):** all Phase 5 results and the model search are in **section 16**. Summary: Part 1
  reproduced (27.1% at 6 latents vs paper 34.1%) + a strong control (feedback vs plain +17.5 points). The user's
  Idea 2 (interleaved latent thinking) is **not yet tested**: GPT-2 ignores step prompts, the Qwen2.5-7B export is
  broken, and the Qwen3-4B export was trained on a yes/no task. **Next:** SIM-CoT GPT-2 (`internlm/SIM_COT-GPT2-Coconut`),
  whose trained explainer can write the steps from the thoughts. Plan `docs/planning/simcot_interleaved_plan.md`,
  gate job not written yet (waiting for "proceed").
- **Update 2026-10-15 (after the GPT-2 run):** Part 1 worked (GSM8K 10.4% -> 12.9% -> 21.0% -> 27.1% with
  0/2/4/6 latents; paper 34.1%). Part 2 hit a floor effect (GPT-2 ignores the few-shot format, writes `### <n>`,
  S1-S5 = 3.4/10.2/1.1/0.9/1.1%, S3 = S5 on every question). Pivot approved: Part 2 moves to
  `agurung/coconut-qwen2.5-7b`, behind a gate check (`docs/planning/qwen7b_coconut_pivot.md`), plus a GPT-2 control
  in its own format (latent slots without feedback). Results: `coconut/results/gpt2/results.json`.
- **Older next step (2026-10-15, done):** the code is written (`coconut/experiments/`, plan `docs/planning/gpt2_coconut_phase5_impl.md`),
  not run yet. User: set up Newton (section 7.3), `sbatch coconut/experiments/run_gpt2.slurm`, then push `logs/` and
  `coconut/results/gpt2/` back. Then: read the logs, build `analysis.ipynb`.

---

## 1. Course requirements (verbatim summary of `docs/requirment/requirment.md`)

- Pick a quantitative / coding research paper, discuss scope with the lecturer for approval. Solo or pairs.
- **Part 1 Reproduce:** implement the core method and reproduce key results (e.g. a main figure / metric).
  - We have an **exemption from reproducing the paper results by training** (fine-tuning GPT-2 needs resources we do not have). Our Part 1 = inference-only reproduction with a released checkpoint.
- **Part 2 Extend (proof of concept):** propose **2-4 follow-up ideas**, choose one, test it with a small well-defined experiment: state a hypothesis, define metrics, run minimal code, produce concrete results (plots / tables / failure cases). Our ideas: section 11.
- **Deliverables:** short report (**6 pages**, template = `final_rep/main.tex`, CogSci proceedings format) + **runnable code**.
- **Presentation / oral check:** the lecturer may meet (in person / zoom, 30 min) to go over the project, also to make sure it was not done by AI. AI use is allowed **as long as we conceptually understand and own all steps and code**.
  - So: report and docs in plain words, every claim must match the code, no hype words ("decisive", "100% identical", "proves").
  - The CogSci template also requires describing AI use in Methods or Acknowledgments.

---

## 2. Paper essentials we rely on

- **Method:** language mode = normal LM. Latent mode = the last hidden state of position t-1 is fed as the **input embedding** of position t (no token decoding). The hidden state is already after the final layer norm, so its size is reasonable.
- Markers: `<bot>` / `<eot>` in the paper = `<|start-latent|>` / `<|end-latent|>` in code. Each continuous thought slot = `<|latent|>`.
- **Training curriculum (multi-stage, from iCoT):** stage 0 = normal CoT. Stage k = first k reasoning steps replaced by k*c latents. Loss masked on question and latents. Optimizer reset each stage.
  - GSM8K: c = 2, 3 stages + 1 extra stage where all remaining language steps are removed, so the final model uses **6 latents** and no language steps. 6 epochs stage 0, 3 epochs per later stage, stays in last stage until 25 epochs (code config) / paper says 50.
  - ProsQA / ProntoQA: c = 1, 6 stages, 5 epochs each.
- **Inference:** `<bot>` right after the question, fixed number of latents, `<eot>`, then the model writes the answer as `### answer`. Greedy decoding.
- **Paper Table 1 (GPT-2), accuracy %:**

| Method | GSM8K | ProntoQA | ProsQA |
|---|---|---|---|
| CoT | 42.9 | 98.8 | 77.5 |
| No-CoT | 16.5 | 93.8 | 76.7 |
| iCoT | 30.0 | 99.8 | 98.2 |
| Pause token | 16.4 | 77.7 | 75.9 |
| **Coconut** | **34.1** | 99.8 | 97.0 |
| w/o curriculum | 14.4 | 52.4 | 76.1 |
| w/o thought | 21.6 | 99.9 | 95.5 |
| pause as thought | 24.1 | 100.0 | 96.6 |

- **Fig. 8:** GSM8K accuracy vs. number of latents / c in {0,1,2}: more thoughts, better accuracy. Our Part 1 reproduces the "same weights, vary latents at inference" idea on GSM8K.
- **Fig. 3:** ProsQA, k in 0..6 latent steps at inference. **Fig. 5-7:** latents encode several candidate next nodes (BFS-like). **Fig. 9:** decoding a GSM8K continuous thought gives intermediate numbers.
- **Appendix C.2:** Llama 3.2-3B / 3-8B gain much less from Coconut than GPT-2 (larger models are harder to move to latent reasoning).
- **Appendix B:** clock time per test case: Coconut 0.09s vs CoT 0.26s on GSM8K (A100).

---

## 3. Project history (all phases)

Hardware until now: local SLURM cluster, single **NVIDIA RTX 2080, 8GB VRAM**.

### Phase 1: reproduce by training (ProsQA, GPT-2) - stopped
- Modified `coconut/args/prosqa_coconut.yaml`: per-device batch 2, gradient accumulation 64 (effective batch 128 like the paper). Created `run_prosqa.slurm` (torchrun + conda env), since deleted.
- Estimated >3 days for 50 epochs. Cancelled, checkpoints deleted to save space.
- Lesson: training is out of budget. Switch to released checkpoints. This is why we have the training exemption.

### Phase 2: "interleaved / autonomous latent thinking" on GSM8K (GPT-2 124M) - failed
- Idea: let the model decide **itself** when to start latent mode. The `<|start-latent|>` token was **unmasked in the loss** so the model learns to emit it at reasoning milestones.
- Scripts (since deleted): `section_data.py`, `interleaved_coconut.py`, `train_interleaved.py` + slurm files. Report `interleaved_latent_report.md` (deleted).
- Result: **1.44%** accuracy, vs **8.72%** for our fine-tuned baseline and **11.07%** zero-shot baseline (numbers from `docs/requirment/summary.md`). Unigram repetition loops, arithmetic drift, unstable training ("optimization shock").
- One positive note recorded at the time: the model did emit `<|start-latent|>` at plausible points in the reasoning.
- Training logs: `coconut/wandb/` (local only, not in git, three offline runs from 2026-10-01 / 10-03 / 10-04).

### Phase 3a: manual decomposition of GSM8K - abandoned
- Plan: use a strong external LLM to split GSM8K questions into small sub-questions (a, b, c) to hand-hold GPT-2.
- Gave ~8% on 100 questions. Conclusion at the time: GPT-2 124M lacks arithmetic capacity no matter how much decomposition. File `data/gsm8k_decomposed_test.json` was deleted.

### Phase 3b: EntailmentBank with GPT-2 `tsrigo/coconut` - honest negative result
- Data: EntailmentBank Task 1 test (340), parsed by `coconut/preprocessing/parse_entailment.py` into `data/entailmentbank/test_decomposed.json` (format in section 13). HF dataset `sxiong/entailmentbank` in cache.
- Model: `tsrigo/coconut` (stage 4) forced into latent reasoning.
- **Baseline (one-shot):** `Context: [all sents]\nQuestion: ...\nAnswer:`, expected the whole proof `sent1 & sent2 -> int1; int1 & sent3 -> hypothesis;`. Model just wrote `2` (GSM8K habit).
- **Decomposed:** per proof step, a sub-question + `<|start-latent|><|latent|>*6<|end-latent|>\nAnswer:`. The **Python script used the gold tree edges** to stitch outputs into `sent1 & sent2 -> int1: <model text>;`.
- Scored with the official AllenAI `run_scorer.py` (repo `coconut/entailment_bank/`, local only).

| Metric | Baseline | Decomposed, no latents | Decomposed, latents |
|---|---|---|---|
| Proof leaves precision | 0.00% | 100.0% | 100.0% |
| Proof steps accuracy | 0.00% | 97.05% | 97.05% |
| Intermediates BLEURT | 0.000 | 0.555 | 0.560 |

- **Conclusion:** the structure score came from the script feeding gold edges, not from the model. With or without latents: same result. The model wrote random digits ("2", "3") as intermediate text. Latents gave nothing on out-of-domain text for a GSM8K-trained 124M model.
- Plan doc: `docs/planning/entailmentbank_experiment_plan.md` (its "resounding success" wording is wrong; the control showed no latent effect).

### Phase 4: Qwen models, "less hand-holding" (2-stage loop) - INVALID as a Coconut test
- **Qwen2.5-7B** (`agurung/coconut-qwen2.5-7b`): ~15GB bf16 does not fit 8GB. `device_map="auto"` CPU offloading, >25h per pass, repeated CUDA OOM on KV cache. Dropped.
- **Gemma-3-4B** considered: 8GB for weights alone. Dropped.
- **Qwen2.5-1.5B** (`CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch`): ~3GB bf16, fits. Used for all Phase 4 results (<2 min per 100 samples).
- First tries with **made-up few-shot examples** (e.g. "Hans has 12 apples...") caused **"tautology collapse"**: stage 1 outputs like `7 + 7 = 14`, unrelated to the question. Also an earlier version had **no** few-shot examples at all (zero-shot) by mistake.
- Fix: **grounded few-shot examples taken from the real GSM8K / EntailmentBank train sets**, each showing 2 steps. This removed the tautologies. These are the prompts we keep (section 5.3).
- Four settings (S1 raw, S2 single pass, S3 `<c_thought>` 2-stage, S4 no-token 2-stage), 100 samples each, GSM8K + EntailmentBank. Results and examples: section 12. Problems: section 4.
- Design doc: `docs/planning/less_handholding_plan.md` (also contains the earlier 7B plan, and an optional "all 4 latents at once" variant that was never run).
- Also in HF cache, never used: `Chengheng/Coconut-Qwen3-0.6B` (same problem: unknown training).

### Phase 5 (current): GPT-2 Coconut re-run - section 5.

---

## 4. What went wrong in Phase 4 (must be explained in the report)

1. **No continuous thought.** `<c_thought>` was put in the prompt as text and plain HF `model.generate()` was called. Nothing fed hidden states back. The old plan even said "you do NOT need a custom forward pass", which is wrong for Coconut.
2. **The token did not exist.** The run log `coconut/slurm_logs/eval_iterative_all_372257.out` prints **"Added <c_thought> to tokenizer and resized embeddings."** for both S3 runs. So the model had never seen `<c_thought>`; it got a fresh untrained embedding (transformers 4.52 initializes new rows from the mean/covariance of old embeddings). S3 = "add 2 untrained tokens to the prompt".
3. **Unknown model.** The Qwen model card is the default HF Trainer template: name "sft", fine-tuned from Qwen2.5-1.5B-Instruct on "TongSearch_Coconut@16_v2", lr 5e-5, batch 256 over 64 GPUs, cosine, 3 epochs, Transformers 4.52.4. Every description field says "More information needed". TongSearch is a search / retrieval project. Cannot be called a Coconut model.
4. **Not significant.** 12/100 vs 6/100, Fisher p ~ 0.2. BLEURT gaps ~0.01.
5. **No stop rule.** 150 free tokens per stage. The model rambled into "Example 1... Example 2...", and the ramble went back into the context. The scorer takes the **last number of the last stage**, which can come from the few-shot examples (the repeated "42").
6. **Stage 2 format bug.** `context = f"{context}\nAnswer: {gen_text}"` drops the stage-1 `Question: What can we deduce in one logical step?` line, so stage 2 no longer matched the examples' format.
7. **Not exactly matched.** S3 decoded with `skip_special_tokens=False` + manual scrub, S4 with `skip_special_tokens=True`.
8. **Possible sampling.** No seed and `do_sample` not set; Qwen-Instruct generation configs often sample by default.
9. **Answer extraction:** `-?\d+\.?\d*` without removing commas, so "1,000" -> "000".
10. **Metric name.** The folder `coconut/bleurt-large-512/` actually holds **BLEURT-20** (`bleurt_config.json`: `"name": "BLEURT-20"`). Call it BLEURT-20.
11. **Docs over-claim.** "doubled", "decisive proof", "100% identical", "conclusively proved". All docs under `docs/` and the old report `entailmentbank_report.md` (repo root) still contain these and must be rewritten.
12. **Code style.** 6 copy-pasted scripts, hard-coded relative paths, unused `argparse` import.

---

## 5. The new experiment (decided 2026-10-14)

### 5.1 Model
- `openai-community/gpt2` + weights from HF repo `tsrigo/coconut`, file **`stage4_checkpoint_25`** (the only file in the repo snapshot `c88fc875...`).
  - Stage 4 + epoch 25 matches the paper GSM8K config (`args/gsm_coconut.yaml`: c_thought 2, max_latent_stage 3, epochs_per_stage 3, 25 epochs). Very likely the **final GSM8K Coconut model**. Verify in Part 0 (keys start with `base_causallm.`? native accuracy near 34%?).
  - The same checkpoint was used in Phase 3b.
- Load exactly like `coconut/run.py` lines 103-167:
  - `tokenizer.add_tokens` in this order: `<|start-latent|>`, `<|end-latent|>`, `<|latent|>`; `tokenizer.pad_token = tokenizer.eos_token`,
  - `model.resize_token_embeddings(len(tokenizer))` (run.py also copies the `<<` embedding into the new rows before loading; harmless since the checkpoint overwrites them),
  - wrap: `Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)`,
  - `torch.load(path, map_location=...)` then `load_state_dict(saved_weights, strict=False)` and **print the result** (missing / unexpected keys must be empty or explained).
- fp32. GPT-2 context limit = **1024 tokens**.

### 5.2 Part 1: reproduction (inference only, GSM8K)
- Data: official iCoT GSM8K test (1319). Get it with `cd coconut && bash preprocessing/gsm_icot.bash` (creates `data/gsm_train.json`, `gsm_valid.json`, `gsm_test.json` from da03/Internalize_CoT_Step_by_Step).
- Input: `question + "\n"` + `<|start-latent|>` + `<|latent|>` x (2k) + `<|end-latent|>`. Model writes the rest, ending with `### answer`. Use `dataset.get_dataset` + `get_question_latent_dataset` with `scheduled_stage = k` so the input is built exactly like the official eval.
- Sweep k (stages) in {0, 1, 2, 3} -> 0 / 2 / 4 / 6 latents. Same weights. `max_new_tokens = 64` (run.py uses 64 for GSM).
- Score exactly like `run.py` line 469: `text.split("#")[-1].replace(",", "").strip() == answer` (answer also comma-stripped).
- Expect k = 3 close to **34.1%**. If far off (e.g. < 25%), **stop and report** before Part 2.
- Plot: accuracy vs number of latents. Also report generated token count and wall time per question.

### 5.3 Part 2: the Phase 4 experiment, ported (same prompts)
Same few-shot prompt and same **forced 2-stage loop** as the Qwen scripts (the hand-holding is kept on purpose).
Only the latent token text changes: `<c_thought><c_thought>` -> `<|start-latent|><|latent|><|latent|><|end-latent|>` (2 latents per stage, matches GSM8K `c_thought: 2`).

| Setting | Prompt | Latent slots |
|---|---|---|
| S1 Raw | `Question: {q}\nAnswer:` | none |
| S2 Single pass CoT | `Question: {q}\nLet's think step by step. ` | none |
| S3 Real latents | 2-stage loop (below) | filled with **previous hidden state** (Coconut forward) |
| S4 No tokens | 2-stage loop, latent line removed | none |
| S5 Pause control (NEW) | 2-stage loop, same latent block | **plain learned token embeddings, no feedback** |

- **S3 vs S5** = does continuous feedback help beyond just extra tokens? (**main test**, mirrors the paper's "pause as thought" control)
- **S3 vs S4** = the old comparison, done right.
- The few-shot examples contain **no** latent tokens in any setting (same as Phase 4). So S3/S4/S5 differ only in the latent block.

**2-stage loop (S3 / S4 / S5).** `LAT` = `<|start-latent|><|latent|><|latent|><|end-latent|>` for S3/S5; for S4 the whole `LAT` line is removed.

Stage 1 prompt:
```
{FEW_SHOT_PROMPT}

Question: {q}

Question: What can we deduce in one logical step?
{LAT}
Answer: 
```
Stage 2 prompt (keeps the full stage-1 turn, fixing bug 4.6):
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
- In S3, stage-1 latents are recomputed inside stage 2 (causal model, same vectors). Total latents seen = 4.
  The official `Coconut.forward` cannot do this: it assumes all latents sit in one block and crashes (IndexError)
  on two blocks. We use our own `common.fill_latents()` (plan F1), checked equal to the official one on one block.
- Score = last number of stage-2 text, commas removed first. Gold = text after `####` in `data/gsm8k_test.json` (or `answer` in `data/gsm_test.json`; both are the same 1319 test questions, use one consistently and say which).

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

**EntailmentBank FEW_SHOT_PROMPT (verbatim from `coconut/eval_qwen_1.5b_entailment_fewshot.py`, only if 5.6 = yes):**
```
--- Example 1 ---
Context: sent1: leo is a kind of constellation sent2: the earth revolving around the sun causes stars to appear in different areas in the sky at different times of year sent3: a constellation contains stars

Question: Stars are organized into patterns called constellations. One constellation is named Leo. Which statement best explains why Leo appears in different areas of the sky throughout the year?

Question: What can we deduce in one logical step?
Answer: Leo is a constellation containing stars.

Question: What can we deduce in one logical step?
Answer: The Earth revolving around the Sun causes Leo to appear in different areas in the sky at different times of year.

--- Example 2 ---
Context: sent1: diurnal motion is when objects in the sky appear to move due to earth 's rotation on its axis sent2: stars appear to move relative to the horizon during the night sent3: the earth rotating on its axis causes stars / the moon to appear to move across the sky at night

Question: Which of the following statements best explains why stars appear to move across Earth's sky each night?

Question: What can we deduce in one logical step?
Answer: Stars appearing to move relative to the horizon during the night is an example of diurnal motion.

Question: What can we deduce in one logical step?
Answer: The Earth rotating on its axis causes stars to appear to move relative to the horizon during the night.

--- Example 3 ---
Context: sent1: more light gets reflected on highly reflective things sent2: venus is covered in highly reflective clouds sent3: as the light reflected off of an object increases , the object will appear to be brighter

Question: Kevin is observing the sky on a clear night. With the unaided eye he is able to see Venus, Mars, Jupiter, and Saturn. Why would Venus appear to be brighter than the other planets?

Question: What can we deduce in one logical step?
Answer: Venus reflects more light toward Earth than other planets because it is covered in highly reflective clouds.

Question: What can we deduce in one logical step?
Answer: As Venus reflects more light, it appears brighter than other planets.
```
EntailmentBank loop: base context = `{FEW_SHOT_PROMPT}\n\nContext: {context_string}\n\nQuestion: {question}`, then the same 2 stages.
EntailmentBank baselines (Phase 4): S1 `Question: {question}\nAnswer:` (no facts), S2 `Context: {context_string}\nQuestion: {question}\nAnswer:` (all facts at once, max 100 tokens). Gold = `hypothesis`. Scored = final stage text (or text after `####` if present) vs hypothesis with BLEURT.

### 5.4 Fixes applied to ALL settings the same way
- Greedy decoding, fixed seed (0), `torch.backends.cudnn.deterministic = True`.
- `max_new_tokens = 64` per stage and for S1 (paper uses 64 for GSM8K). S2 may need 128: decide in the smoke test.
- Stop strings: cut at `\nQuestion:`, `--- Example`, or EOS.
- Length guard: if any full stage-2 prompt + 64 > 1024 tokens, use 2 few-shot examples instead of 3 **for every setting**, and write it in the report.
- Save per example: question id, prompt token length, raw output per stage, cut output, extracted answer, gold, correct flag.
- Data: **full GSM8K test (1319)** for every setting. Same questions in every setting (paired).

### 5.5 Hypotheses, metrics, stats
- H1: S3 > S5. H2: S3 > S4. Also report S1, S2 for context.
- Metric: exact-match accuracy (+ 95% Wilson CI per setting).
- Stats: **paired McNemar test** per pair (S3-S5, S3-S4), 95% CI on the accuracy difference (paired bootstrap, 10k resamples, seed fixed).
- A difference counts only if the CI excludes 0. Otherwise write "no clear difference".
- Failure cases: chosen by a fixed rule (e.g. first 5 questions where S3 and S5 disagree, in both directions), not cherry-picked.
- Expected risk: **floor effect**. GPT-2 Coconut never saw this few-shot format. All S1-S5 may be near 0-10%. If so, report it honestly; Part 1 still stands.

### 5.6 DECIDED (2026-10-15): no EntailmentBank, GSM8K only
- User decision: "lets not use entailment bank just GSM8K for now". The GSM8K-trained GPT-2 is out of domain on EntailmentBank anyway (Phase 3b showed no latent effect there).
- So BLEURT and the AllenAI scorer are **not needed** on Newton. EntailmentBank material in this file is kept for history only.
- If ever re-added: all 340 of `data/entailmentbank/test_decomposed.json`, same S1-S5, BLEURT-20 + paired bootstrap CI.

### 5.7 Optional (if time): look inside the latents
- Decode each continuous thought with the LM head (top-5 tokens), like paper Fig. 9.
- Count how often the top token is a number from the question or an intermediate result.

---

## 6. Implementation notes

- **Do not modify** official files: `coconut/coconut.py`, `run.py`, `dataset.py`, `utils.py`.
- How `Coconut.forward` works (coconut.py): finds `<|latent|>` positions, runs the model up to the first latent, then for each latent pass puts the last hidden state of the previous position into that slot (with KV cache), and a final pass over the rest of the sequence. Returns `Outputs(loss, inputs_embeds, logits)` where `inputs_embeds` has the latent slots filled.
- `Coconut.generate` (batch size 1 only): one `forward`, then greedy loop **without KV cache**, stops at EOS. Ignores `attention_mask`. No stop strings.
- **One greedy loop for all settings**, so the only difference between settings is the input embeddings:
  1. Build `input_ids` for the prompt.
  2. Get `inputs_embeds`:
     - S3: `common.fill_latents(model, input_ids)`. Same result as `Coconut.forward(...).inputs_embeds` when the
       latents are one block, but also works for stage 2 (two blocks), where `Coconut.forward` crashes.
     - S5: `embedding(input_ids)` (latent tokens keep their plain learned embedding).
     - S1 / S2 / S4: `embedding(input_ids)` (no latent tokens present).
  3. Greedy loop on `base_causallm(inputs_embeds=..., past_key_values=...)`: argmax, append token embedding, stop on EOS / stop string / max tokens.
- Sanity checks in the smoke test (Part 7):
  - our loop in S3 mode reproduces `Coconut.generate` tokens on the native format (same first 64 tokens),
  - S3 with the latent block equals S5 when the model has 0 latents (trivial) and differs on some examples when latents exist,
  - native k = 3 on 20 questions gives sensible `### answer` outputs,
  - token-length report for the longest stage-2 prompt.
- Speed: GPT-2 is tiny. Run settings as separate jobs / processes in parallel.
- `get_dataset` uses `num_proc=32` for `.map`; on a node with fewer CPUs this is fine but slow to start.
- `run.py` itself needs `torchrun` + NCCL + wandb, so do **not** use it directly for eval; reuse its loading logic.
- More findings from reading the code (2026-10-15, details F2-F7 in `docs/planning/gpt2_coconut_phase5_impl.md`):
  - the `<|latent|>` **input** embedding was never trained (every slot is replaced by a hidden state in training),
    so S5 is an almost untrained pause token, not the paper's trained "pause as thought". Say so in the report.
  - the 3 added tokens are not "special": `skip_special_tokens` keeps them. Generated text is cleaned of them before
    it goes into stage 2 (else a stray `<|latent|>` would become a new latent slot).
  - `get_dataset` crashes if more than 1 GPU is visible (`dist.get_rank()` without a process group). Use 1 GPU.
  - prompts end with `"Answer: "`: the trailing space is a lone token for GPT-2. Kept (same prompts), noted as a risk.

---

## 7. Server setup (Newton cluster)

### 7.1 Access
- Newton = Technion CS Slurm cluster. Only reachable from Technion network or **VPN**.
- Login: `ssh -X eran.b@132.68.39.200` (hostname `newton.cs.technion.ac.il` may not resolve over VPN; then use the IP). Open OnDemand portal also exists.
- Passwordless login from the user's Windows PC (PowerShell, run on the PC, not on Newton):
  ```powershell
  ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_newton -C "newton"
  type $env:USERPROFILE\.ssh\id_ed25519_newton.pub | ssh eran.b@132.68.39.200 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  ```
  Add to `C:\Users\<user>\.ssh\config`, then VS Code "Remote-SSH: Connect to Host" -> `newton`:
  ```
  Host newton
    HostName 132.68.39.200
    User eran.b
    IdentityFile ~/.ssh/id_ed25519_newton
    IdentitiesOnly yes
    ForwardX11 yes
  ```
- The old cluster was `lambda.cs.technion.ac.il` (user eran.b).
- **Never store passwords in files or chat.**

### 7.2 GitHub on Newton
- Repo: `git@github.com:grigrigri1235/neuroAI.git` (SSH only).
- On Newton: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github`, add the `.pub` to GitHub (Settings -> SSH keys), and in `~/.ssh/config`:
  ```
  Host github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
  ```

### 7.3 Environment (Newton, 2026-10-15: no conda there before)
Run on the Newton login node. Miniconda must go to `/home/eran.b/miniconda3` (the slurm script uses that path).
`environment_newton.yml` = lean `environment.yml` (same versions, no tensorflow / wandb, conda-forge only).
```bash
cd /home/eran.b/neuroAI && git pull && mkdir -p logs

# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p /home/eran.b/miniconda3 && rm ~/miniconda.sh
source /home/eran.b/miniconda3/etc/profile.d/conda.sh
conda init bash   # optional: conda in every new terminal

# env (env name: neuroAI_project). "conda env create -f environment_newton.yml" FAILS on Newton with
# CondaToSNonInteractiveError (Anaconda defaults channels need a Terms-of-Service accept), so skip those channels:
conda create -y -n neuroAI_project --override-channels -c conda-forge python=3.12 pip > logs/setup_env.log 2>&1
conda activate neuroAI_project
pip install torch==2.6.0 numpy==2.1.3 transformers==4.52.4 datasets==3.6.0 tokenizers==0.21.1 tqdm==4.67.0 pyyaml scipy matplotlib jupyter nbconvert ipykernel >> logs/setup_env.log 2>&1

# GSM8K in Coconut format (Part 1) + GPT-2 and the checkpoint into the HF cache, with a CPU load check
cd coconut
bash preprocessing/gsm_icot.bash > ../logs/setup_data.log 2>&1
python experiments/common.py > ../logs/setup_model.log 2>&1   # must end with "loaded OK"
cd ..
```
Then `sbatch coconut/experiments/run_gpt2.slurm` from the repo root (section 7.4).
The full lambda env is still `environment.yml` (tensorflow etc. for BLEURT, not needed now).
Note: `coconut/requirements.txt` (official) pins older versions (torch 2.5.1, transformers 4.46.2); we used `environment.yml` successfully.

Only if EntailmentBank / BLEURT is needed (not in git, too big for GitHub):
```bash
pip install git+https://github.com/google-research/bleurt.git
cd coconut && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip && unzip BLEURT-20.zip
#   old scripts load the folder name "bleurt-large-512": rename BLEURT-20 -> bleurt-large-512, or update the path
git clone https://github.com/allenai/entailment_bank.git   # only the old Phase 3b scorer used this
```

### 7.4 Slurm and env vars
New (Phase 5): `coconut/experiments/run_gpt2.slurm`, Technion Newton template format: `--cpus-per-task=8`,
`--gres=gpu:L40:1` (Newton's Slurm name for the L40S; `gpu:L40S:1` was rejected with "Requested node configuration is not available". Check names with `sinfo -N -o "%N %G"`), `--mem=32G`, `--time=04:00:00`, mail to
`eran.b@campus.technion.ac.il`, `-o/-e` into `logs/` (relative to the submit folder = repo root, so `logs/` must exist).
One job: smoke test (gate) -> full native -> full S1-S5 in parallel on one GPU -> score. Uses `HF_HUB_OFFLINE=1`.
Old slurm header (`coconut/eval_qwen_1.5b_iterative_all.slurm`, lambda): `--gres=gpu:1 --cpus-per-task=2 --mem=32G --time=12:00:00`, outputs to `slurm_logs/<name>_%j.out/.err`.
```bash
source <conda>/etc/profile.d/conda.sh && conda activate neuroAI_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXABI_1_3_15_FOUND=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0                      # stop transformers from importing TensorFlow
# BLEURT scoring only: run on CPU, TF and PyTorch CUDA crash when both use the GPU in one process
export CUDA_VISIBLE_DEVICES=""
```
Old local paths to change on Newton: `/home/eran.b/miniconda3/...` and `/home/eran.b/neuroAI/...` in the slurm files and in `preprocessing/parse_entailment.py`.

### 7.5 Local-only things (not in git)
- `coconut/wandb/` (313MB, Phase 1/2 training logs, one file 137MB) - not needed on Newton.
- `coconut/bleurt-large-512/` (= BLEURT-20, 2.3GB) - download again if needed (7.3).
- `coconut/entailment_bank/` (AllenAI repo clone) - clone again if needed (7.3).
- HF cache on the old cluster: `agurung/coconut-qwen2.5-7b`, `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch`, `Chengheng/Coconut-Qwen3-0.6B`, `tsrigo/coconut`, `openai-community/gpt2`, datasets `gsm8k`, `sxiong/entailmentbank`.

---

## 8. Repo map

```
KNOWLEDGE_RETENTION.md          <- this file
entailmentbank_report.md        old Phase 3b+4 report, moved here from docs/report/ (over-claims, to be rewritten)
og_coconut.md / .pdf            paper
environment.yml                 conda env
.gitignore                      excludes only wandb, BLEURT, entailment_bank clone, __pycache__
final_rep/                      CogSci LaTeX template (main.tex still template text, cogsci.sty, .bib)
docs/
  requirment/requirment.md      course task (section 1)
  requirment/summary.md, codebase_changes.md, detailed_explanations.md, proposed_followup_ideas.md
                                old Phase 1-4 notes (content merged here; over-claiming, to be rewritten)
  planning/gpt2_coconut_rerun_plan.md   first draft of the new plan (superseded by this file)
  planning/less_handholding_plan.md     Phase 4 design (source of the prompts)
  planning/entailmentbank_experiment_plan.md   Phase 3b design
coconut/
  coconut.py run.py dataset.py utils.py   official Meta code, DO NOT EDIT
  README.md LICENSE assets/               official
  args/*.yaml                   official configs (prosqa_coconut.yaml edited for 8GB GPU)
  data/prosqa_{train,valid,test}.json     official ProsQA (not used now)
  data/gsm8k_test.json          GSM8K test, HF format with '####' answers (Phase 4)
  data/entailmentbank/          {train,validation,test}.jsonl + *_decomposed.json
  preprocessing/gsm_icot.bash, gsm_icot.py, prontoqa.py   official
  preprocessing/parse_entailment.py       ours (Phase 3b)
  eval_qwen_1.5b_baselines.py             Phase 4, EB S1+S2
  eval_qwen_1.5b_gsm8k_baselines.py       Phase 4, GSM8K S1+S2
  eval_qwen_1.5b_gsm8k_less_handholding.py   Phase 4, GSM8K S3
  eval_qwen_1.5b_gsm8k_no_latents.py         Phase 4, GSM8K S4
  eval_qwen_1.5b_entailment_fewshot.py       Phase 4, EB S3
  eval_qwen_1.5b_entailment_no_latents.py    Phase 4, EB S4
  score_outputs.py (S3/S4), score_baselines.py (S1/S2)   Phase 4 scoring
  *.slurm                        Phase 4 job scripts
  predictions_gsm8k_1.5b_{baseline_a,baseline_b,less_handholding,no_latents}.json
  predictions_qwen_1.5b_baseline_{a,b}.json          (EB S1/S2)
  predictions_entailment_1.5b_{fewshot,no_latents}.json   (EB S3/S4)
  score_*_results.txt, slurm_logs/   Phase 4 scores + logs
Phase 5 (written 2026-10-15):
  environment_newton.yml        lean env for Newton
  coconut/experiments/common.py       model loading, fill_latents, greedy loop, text helpers (script: download + load check)
  coconut/experiments/prompts.py      few-shot prompt (verbatim) + S1-S5 builders
  coconut/experiments/run_native.py   Part 1 (k = 0..3)
  coconut/experiments/run_fewshot.py  Part 2 (--setting S1..S5)
  coconut/experiments/score.py        stats -> results.json
  coconut/experiments/smoke_test.py   checks C0-C8
  coconut/experiments/run_gpt2.slurm  the one Newton job
  coconut/experiments/qwen_check.py + .slurm    Qwen2.5-7B gate (+ GPT-2 run_native --fill plain control)
  coconut/experiments/qwen3_check.py + .slurm   Qwen3-4B gate G0-G3 (common.load_coconut_hf)
  coconut/experiments/simcot_run.py + .slurm    SIM-CoT gate G0-G2 + L / T2 / T2-gold (job 1449771)
  coconut/experiments/simcot_t2in.py + .slurm   SIM-CoT T2-in / T2-in-gold (step inside the question)
  coconut/results/simcot/       simcot_run.json, simcot_t2in.json
  coconut/results/gpt2/         native.json, native_plain.json, fewshot_S1..S5.json, results.json, smoke/
  coconut/results/qwen7b/check/, coconut/results/qwen3/check/   gate reports
  docs/planning/gpt2_coconut_phase5_impl.md, qwen7b_coconut_pivot.md, qwen3_4b_interleaved_plan.md,
                simcot_interleaved_plan.md (next)
  workflow.md                   how we work (lambda/Newton, conversation lifecycle, commands, pitfalls)
  logs/                         Newton logs (pushed back with git)
planned (Phase 5):
  coconut/results/gpt2/         predictions + results JSON (made by the job)
  coconut/analysis.ipynb        tables, plots, stats (executed, with outputs)
```

---

## 9. Steps (each is a tiny part, report "done with part i")

Status 2026-10-15: Parts 1-6 are written (code + slurm). Part 0 is now inside the setup commands (section 7.3:
`python experiments/common.py` downloads and load-checks the checkpoint; the smoke test prints token lengths).
Parts 7 + 8 are one `sbatch` (the smoke test is a gate for the full run). Next: the user runs it on Newton.

- Part 0a: download checkpoint, print state-dict keys and `load_state_dict` result.
- Part 0b: count token lengths of the longest stage-2 prompt (GSM8K test).
- Part 1: `experiments/common.py` (load model, greedy loop, 3 embedding modes, stop strings, seed).
- Part 2: `experiments/prompts.py` (few-shot prompt + builders for S1-S5).
- Part 3: `experiments/run_native.py` (Part 1 sweep k = 0..3).
- Part 4: `experiments/run_fewshot.py` (`--setting S1..S5`).
- Part 5: `experiments/score.py` (accuracy, Wilson CI, McNemar, bootstrap CI, results JSON).
- Part 6: slurm / run script for Newton.
- Part 7: smoke test (20 questions, all settings + sanity checks in section 6). **Needs approval to run.**
- Part 8: full run. **Needs approval to run.**
- Part 9: `analysis.ipynb`: tables, plots, failure cases; execute with
  `jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True coconut/analysis.ipynb`
- Part 10: rewrite the report in plain words (old Qwen run = "first attempt and what went wrong").
- Part 11: update `docs/requirment/*.md` and `codebase_changes.md` (see rule in section 10).
- Part 12: fill `final_rep/main.tex` (6 pages, include AI-use statement).

---

## 10. Working rules for any agent on this project (from the user's CLAUDE.md + project notes)

- Start **every** message with a short recap of these rules, plus how you might break them and how you will avoid it.
- Brainstorm / plans go in a file (`docs/planning/`), not in chat.
- Plan first, then **stop and wait for explicit approval** ("proceed"). No code edits and **no bash commands** without approval.
- After approval: split into tiny parts, say "done with part i" after each, keep going until all parts are done.
- Bash output always to a file in a logs folder, never the terminal (old path: `/home/eran.b/PAR_FINAL/PAR_FINAL/alt-project/logs`; on Newton: `/home/eran.b/neuroAI/logs/`, decided 2026-10-15).
- The user runs the commands on Newton; Claude (on lambda) gives exact step-by-step commands and reads the logs after they are pushed back.
- If code crashes: do **not** fix it alone. Report the reason and stop.
- Short, simple language with the user and in reports.
- Reports must be dense: data, code for reproducibility, and plots. Text-only summaries = failure.
- Notebooks must be executed end-to-end (nbconvert command in section 9) and delivered with all outputs.
- Phase 4 analysis must be code-driven: parse results JSON, compute the metrics defined in the plan.
- **Document every code change** to the coconut code in `docs/requirment/codebase_changes.md`, in depth, including motivation and the plan it came from (rule from `summary.md`).
- Git: SSH remote. Ask before pushing (outward-facing).

---

## 11. Part 2 follow-up ideas (course needs 2-4)

From `docs/requirment/proposed_followup_ideas.md`, rewritten plainly.

### Idea 1: dynamic thinking length
- Problem: Coconut uses a fixed number of latents K for every question. Easy questions waste compute, hard ones may need more.
- Idea: a small router predicts K in {0, 2, 4, 6, 8} from the question, trained with
  `L_total = L_task + lambda * Cost(K)`, `Cost(K) = K * c_step`, lambda >= 0 trades accuracy vs compute (inspired by a resource-cost loss from the course).
- Cheap inference-only version possible with our data: per question, the smallest k in Part 1 that is already correct; check if it relates to question difficulty (number of gold steps).

### Idea 2: multi-step reasoning without hand-holding (THE CHOSEN ONE)
- Problem: earlier pipelines needed Python scripts that feed the model exact sub-questions / gold proof edges (Phase 3b).
- Idea: give the model only a generic step question ("What can we deduce in one logical step?") with latents at each step and see if it can chain the steps itself.
- History: tried as autonomous interleaving on GPT-2 (Phase 2, failed), then as the 2-stage loop on Qwen 1.5B (Phase 4, invalid), now on real GPT-2 Coconut (Phase 5).
- Honest note: the 2-stage loop with few-shot examples is still partly hand-held (fixed number of stages, fixed question). Say so in the report.

### Idea 3: steering the silent thoughts (future work / safety)
- Problem: latent thoughts are not readable text, so we cannot easily monitor them.
- Idea: steering vectors on the latent positions, `h' = h + alpha * v_steer`, where `v_steer` comes from contrastive activation differences (e.g. truthful vs hallucinated). Test both risk (can a nudge change the answer silently?) and use (can it reduce hallucination without hurting GSM8K?).
- Small related step we can do: Part 5.7 (decode latents through the LM head).

---

## 12. Old results in detail (Phase 4, Qwen 1.5B, 100 samples, invalid as Coconut)

| | S1 raw | S2 single pass | S3 `<c_thought>` 2-stage | S4 no-token 2-stage |
|---|---|---|---|---|
| GSM8K accuracy | 34% | 40% | 12% | 6% |
| EntailmentBank "BLEURT" (actually BLEURT-20) | 0.4052 | 0.4273 | 0.4010 | 0.3924 |

Source files: `coconut/score_baselines_results.txt`, `coconut/score_outputs_results.txt`.

Qualitative examples recorded in the old report (useful as failure-case templates):
- **Janet's ducks** (gold 18): S1, S2 correct. S3 stage 1 forgot the 4 muffin eggs, answered 26. S4 got 18 but then recited the few-shot examples.
- **Robe fabric** (gold 3): S1, S2 correct. S3 got 3 in stage 1, but stage 2 drifted into the few-shot content, no final number. S4 answered 42 (copied from the Julie example).
- **Sheep in 3 cities** (gold 260): S1, S2, S3 correct. S4 got 260 in stage 1, then stage 2 recited the examples, no answer.
- EntailmentBank `NYSEDREGENTS_2014_4_2` (Earth orbit): S2 best (0.528), S3 0.465, S4 0.401, S1 0.381.
- EntailmentBank `MCAS_2004_9_16` (moon vs sun tides): S2 best (0.512), S3 0.485, S1 0.462, S4 0.413.
- Pattern: most "drift" failures are the missing stop rule (bug 4.5), not latents vs text.

---

## 13. Data formats

- `coconut/data/gsm8k_test.json` (HF GSM8K test, 1319): list of `{"question": str, "answer": "...reasoning... #### 18"}`.
- `coconut/data/gsm_test.json` (created by `gsm_icot.bash`, Coconut format): list of `{"question": str, "steps": ["<<4-2=2>>", ...], "answer": "300"}`.
- `coconut/data/entailmentbank/test_decomposed.json` (340): list of
  `{"id", "question", "answer", "hypothesis", "context_string": "sent1: ... sent2: ...", "steps": [{"inputs": ["sent1","sent2"], "prompt": "What can be concluded from sent1 and sent2?", "target_text": str, "is_final": bool}]}`.
- Phase 4 prediction files: GSM8K S1/S2 `{"question", "generated", "gold_answer"}`; S3/S4 `{"question", "gold_answer", "generated_steps": [stage1, stage2]}`; EB S1/S2 `{"id", "prompt", "generated", "gold"}`; EB S3/S4 `{"id", "hypothesis", "generated_steps"}`.
- ProsQA (`data/prosqa_*.json`): Coconut format, not used now.

---

## 14. Lessons learned (keep these)

1. Check the method is really implemented: Coconut = hidden-state feedback, not a token in the prompt.
2. Check a token exists in the vocab before using it ("Added ... resized embeddings" in a log = red flag).
3. Know the checkpoint: training data, format, latent token names. A model card with "More information needed" is not enough.
4. Always have a stop rule and greedy decoding with a seed. Free-running generation pollutes multi-stage contexts.
5. Use enough samples and a paired test. 100 samples can't show a 6-point gap.
6. Every comparison needs a control that removes only one thing (S5 for feedback, S4 for tokens).
7. A scaffold can do all the work (Phase 3b). Always run the no-latent control.
8. Grounded few-shot examples from the real train set beat made-up ones (tautology collapse).
9. Small GPU: 7B models do not fit 8GB; offloading is too slow.
10. TensorFlow (BLEURT) and PyTorch on the same GPU in one process crash; score BLEURT on CPU.
11. Write claims in plain words that match the code; the oral check will test this.
12. Gate every new checkpoint with a short check job before any big run (loads right, our loop = HF/official,
    latents change the output, the model follows the format). It caught the broken Qwen2.5-7B and the yes/no Qwen3-4B
    in ~10 min each.
13. A floor effect is not a result about the hypothesis. If every setting is ~0%, the test could not see anything.
14. HF exports from newer transformers can load silently wrong in older versions (Qwen3 export: `rope_theta` nested
    in `rope_parameters`). Check the actual values after loading.
15. A model card without the training data and format is a red flag. Check what the model actually writes first.

---

## 15. Glossary

- **Continuous thought / latent:** a hidden state fed back as the next input embedding.
- **k / stage:** number of reasoning steps replaced by latents. GSM8K: latents = 2k.
- **c (c_thought):** latents per replaced step (GSM8K 2, ProsQA 1).
- **Pause token:** an extra learned token with no feedback (paper baseline; our S5).
- **Hand-holding:** the script decides the structure (sub-questions, number of stages, gold edges) instead of the model.
- **Tautology collapse:** stage outputs like `7 + 7 = 14` that ignore the question (Phase 4, made-up examples).
- **BLEURT-20:** learned text-similarity metric (Google). Our folder is misnamed `bleurt-large-512`.
- **S1-S5:** settings in section 5.3.
- **Plain / no-feedback control:** the same latent slots, but filled with the `<|latent|>` token embedding instead
  of the previous hidden state (`--fill plain`, S5).
- **Gate check:** a ~10-30 min job that must pass before a full run is planned.
- **Explainer (SIM-CoT):** a second GPT-2, trained to write the step text that a pair of continuous thoughts stands for.

---

## 16. Phase 5 results + model search (2026-10-15)

All runs on Newton, NVIDIA L40S, transformers 4.52.4, greedy, seed 0, GSM8K test (1319) unless stated.

### 16.1 GPT-2 Coconut (`tsrigo/coconut/stage4_checkpoint_25`), job 1449565 + 1449567
Smoke test: all required checks passed. Our `fill_latents` = official `Coconut.forward` (difference 0.0). Our loop =
`Coconut.generate` (20/20, cache on and off). The official forward crashes on two latent blocks
(`IndexError: index 22 is out of bounds`), so our own filler is needed. Few-shot prompt = Phase 4 byte for byte.
The two GSM8K files hold the same 1319 questions and answers.

**Part 1 (reproduction) + control** (`coconut/results/gpt2/results.json`):

| Latents | Feedback (Coconut) | 95% CI | Plain (no feedback) | Feedback - plain [95% CI] | McNemar p |
|---|---|---|---|---|---|
| 0 | 10.4% | 8.9-12.1 | 10.4% | 0 (identical, sanity check) | 1 |
| 2 | 12.9% | 11.2-14.8 | 10.2% | +2.7 [+1.5, +4.1] | 4e-5 |
| 4 | 21.0% | 18.9-23.3 | 9.9% | +11.1 [+9.0, +13.1] | 2e-26 |
| 6 | 27.1% | 24.7-29.5 | 9.6% | +17.5 [+15.2, +19.9] | 7e-50 |

- **Trend reproduced:** more thoughts give better accuracy. The level is below the paper (34.1%). Possible reasons: a
  third-party checkpoint, the last epoch rather than the best on validation. Not proven.
- **The model writes only `### N`** (~2.3 tokens). Time per question: 0.019 s (0 latents) to 0.052 s (6 latents).
- **Control:** without feedback, extra slots add nothing (stays ~10%), so the gain needs the hidden-state feedback.
  Caveat: the plain slots are an untrained input (the paper's "pause as thought" baseline was trained). This
  mostly confirms the paper's claim, in a cheaper inference-only way.

**Part 2 few-shot S1-S5 = floor effect** (the full explanation is in `docs/planning/simcot_interleaved_plan.md`
section 0): S1 3.4%, S2 10.2%, S3 1.1%, S4 0.9%, S5 1.1%. Every stage wrote `### <n>` and stopped, often `36` copied from
the examples. S3 = S5 on all 1319 (0 discordant). Why: the checkpoint knows only `question -> latents -> ### answer`,
and a 124M model cannot learn a new format from 3 examples. So no step was written, and the interleaving never happened.

Marker embeddings (smoke C8): `<|latent|>` cosine 0.65 to its `<<` init (`<<` itself moved to 0.46). So "less trained
than normal tokens", not "untrained".

### 16.2 `agurung/coconut-qwen2.5-7b`, gate job 1449567: BROKEN, abandoned
The saved tokenizer lacks the latent tokens, but the 3 extra rows sit at ids 151665-7, and our check confirmed the
mapping. The output rows of all 3 markers are still the exact `<<` init (cosine 1.0, same norm), so they were barely
trained. **It writes garbage in every setting, even S1 with no latents** (`a additional a additional ...`).
0/10 everywhere. Later the Qwen3 G1 check showed our loop = HF `generate` on Qwen models, so the garbage is the
checkpoint, not our code.

### 16.3 `agurung/coconut-qwen3-4b-ff`, gate job 1449576: trained on a yes/no task, stop
Base Qwen3-4B-Instruct-2507, checkpoint_32. Saved by transformers 5.5.4: the loader copies the nested `rope_theta`
(5e6) and checks the rotary frequencies. The repo tokenizer does not load in 4.52, so the base Qwen3 tokenizer is used;
the ids match `latent_metadata.json`.
- G0 load PASS. G1 our loop = HF `generate` 10/10 PASS.
- **G2 FAIL:** in the native format with 0/2/4/6 latents (feedback and plain), 0/50, always `In summary, \boxed{Yes}`.
- G3: in the few-shot loop stage 1 it writes real math steps (`16 - 3 - 4 = 9 eggs ... 9 * 2 = $18`), then
  `\boxed{Yes}`. Stage 2 often repeats itself or loops `YesYesYes`. 1-2/10 correct.
- So it was trained on an undocumented yes/no task. GSM8K is out of its domain. Training data not found on the
  author's HF page.

### 16.4 Status of the user's Idea 2 (interleaved latent thinking)
Not tested yet. Needs a model with (a) working latents and (b) the ability to write steps between them.
GPT-2: (a) yes, (b) no. Qwen2.5-7B: broken. Qwen3-4B: (b) partly, (a) no effect on GSM8K.
**Next candidate:** `internlm/SIM_COT-GPT2-Coconut` (arXiv 2509.20317), GPT-2 Coconut + SIM-CoT, same GSM8K format,
`c_thought 2`, `max_latent_stage 5` (10 latents), file `checkpoint_28` (~1.4GB). Its training added an **explainer**
GPT-2 that turns each thought pair into the step text (e.g. `<<16-3-4=9>>`). If the explainer is in the checkpoint,
the model can write its own steps from its own thoughts, without in-context learning.
Plan `docs/planning/simcot_interleaved_plan.md`.

**CHOSEN design (user, 2026-10-15): minimal 2 rounds** (plan section 2c), GSM8K test 1319, paired:
- **L** (baseline): question -> 10 latents -> `### answer`.
- **T2** (the user's idea): round 1: question -> 10 latents; the explainer turns thought pair 1 into step s1
  (e.g. `<<16-3-4=9>>`). Round 2: question + "\n" + s1 -> 10 latents -> `### answer`.
- **T2-gold** (best case): as T2, but s1 = the true step 1 from `gsm_test.json`.
- Metrics: accuracy T2 vs L, T2 vs T2-gold (McNemar + bootstrap); exact match of s1 vs the true step 1.
- The answer always comes from the SIM-CoT model itself (base GPT-2 was never trained on math). Full 10-latent budget
  in both rounds.
- Dropped by the user: T2-plain (round 2 with plain latents, "we know latent thinking works"), and A (as many rounds
  as the true number of steps, an oracle hint).
- Why 2 rounds: same as the original 2-stage idea, and failures stay easy to understand. All earlier failures are
  documented (16.1-16.3, plan section 0), and the user expects the instructor to accept the reduced scope.

Earlier designs considered (not chosen, kept for the record):
- **L:** all latent (10).
- **R (the user's design):** re-prompt each round with the decoded steps added to the question; full 10-latent
  budget every round; stop on a repeated step, no new step, or 5 rounds.
- **I:** one sequence: pair, step text, pair, ...
- Plain-latent controls.
- Gate first: G0 explainer in checkpoint? G1 native sweep 0-10 latents (paper reproduction, vs our 27.1%).
  G2 decode thoughts of 20 questions vs the gold steps. G3 pilot of the chosen design: L, T2, T2-gold on 20 questions.
- Open risk: in R, after step 1 is added, do the thoughts move on to step 2 or repeat step 1?

### 16.6 SIM-CoT GPT-2 results (`internlm/SIM_COT-GPT2-Coconut`, job 1449771, 837 s, `coconut/results/simcot/simcot_run.json`)

**G0 PASS.** 150 keys: `base_causallm` 149, `expainable_llm` 149, `embedding` 1. No missing / unexpected keys for the
main model or the explainer.

**G1 (native format, 1319):**

| Latents | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| SIM-CoT | 5.4% | 13.6% | 27.9% | 38.2% | 43.5% | **44.7%** [42.0, 47.3] |
| Our Coconut GPT-2 (16.1) | 10.4% | 12.9% | 21.0% | 27.1% | - | - |

More thoughts give better accuracy; flattens 8 -> 10. In the range of the SIM-CoT paper's GPT-2 claim (about +8 over
Coconut); check the exact paper number before the report.

**G2 PASS: the decoded thoughts are readable steps.** First 20 questions: mostly well-formed `<<a*b=c>>` steps, often
equal to the true steps (q11: `<<3*68=204>> <<2*80=160>> <<6*55=330>>` = gold; q12 first 3 of 4 = gold). Later pairs
often decode to `### N`, usually the same N as L's answer. Bonus finding for Idea 3 (reading hidden reasoning).

**Main (1319, paired):**

| Setting | Accuracy | vs L [95% CI] | McNemar p |
|---|---|---|---|
| L (10 latents) | 44.7% | - | - |
| T2 (question + "\n" + explainer s1) | 34.4% | -10.2 [-12.3, -8.2] | 3e-22 |
| T2-gold (question + "\n" + true step 1) | 38.6% | -6.1 [-8.3, -3.9] | 1e-7 |

T2 vs T2-gold: -4.2 [-5.8, -2.6], p = 1e-6. s1 exactly equal to the true step 1: 42.5% (strict string match:
`<<4*20=80>>` vs `<<20*4=80>>` counts as wrong, so this understates it).

- **Can claim:** on this model, adding a written step before the second thought round lowers accuracy. Even the true
  step hurts, so most of the drop comes from the layout (text after the question, never seen in training); explainer
  errors cost ~4 more points.
- **Cannot claim:** that interleaving fails in general (one model, trained only on question -> latents -> answer);
  why the true step hurts (guess: the model redoes done work; q0 T2-gold answered 36 = 18 x 2). Not measured.
- **User reaction:** sad about a negative result. Asked to put the step inside the question instead (T2-in, next).
  Explained to the user: T2 already was "step added to the question"; a text label like "START OF YOUR ANSWER:"
  would not help (GPT-2 does not follow instructions).

**T2-in / T2-in-gold:** written, not run (plan `docs/planning/simcot_t2in_plan.md`). Results go here.

### 16.5 Other Part 2 ideas (cheap, GPT-2, inference only; offered, not chosen yet)
Thought transplant (question A's thoughts into question B: does B's answer follow A?), per-thought ablation (which of
the 6 thoughts matters), noise/steering robustness (Idea 3), more thoughts than trained (8-10).

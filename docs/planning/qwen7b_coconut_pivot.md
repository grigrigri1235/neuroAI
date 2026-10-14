# Pivot plan: Part 2 on `agurung/coconut-qwen2.5-7b` (DRAFT, waiting for approval)

Date: 2026-10-15. Status: APPROVED ("i approve the pivot and your plan"). D1 yes, D2 yes, D3 128 tokens per stage.
Parts 1-2 written: `common.load_qwen`, `run_native.py --fill plain` (D2), `score.py` native pairs,
`qwen_check.py` + `qwen_check.slurm` (gate). Next: user prefetches the model and runs the gate job.
Why: Phase 5 Part 2 on GPT-2 hit a floor effect. The 124M model ignores the few-shot format and answers
`### <number>` in every setting (S1-S5 all 1-10%), often copying `36` from the examples. So S3 vs S5 could not be
tested. Part 1 (GPT-2 reproduction: 10.4% -> 27.1% with 0 -> 6 latents) **stays** as our Part 1 result.
Idea: an instruction-tuned 7B model should follow the prompt format, so the same S1-S5 test becomes possible.

---

## 1. What we know about the checkpoint (read from the HF Hub, 2026-10-15)

| Fact | Source | Meaning |
|---|---|---|
| Base `Qwen/Qwen2.5-7B-Instruct`, `Qwen2ForCausalLM`, 7.6B params | config.json | fits one L40S (46GB) in bf16 (~15GB) |
| `transformers_version: 4.46.2` | config.json | same version as the official Coconut `requirements.txt`, so probably trained with the official `run.py` (good sign) |
| `vocab_size: 151668` | config.json | Qwen2.5-Instruct's tokenizer has 151665 tokens (ids 0..151664). 151668 = 151665 + **3**, exactly Coconut's 3 added tokens |
| `added_tokens.json` has **no** `<|start-latent|>`, `<|end-latent|>`, `<|latent|>` | added_tokens.json | the saved tokenizer does **not** know the latent tokens |
| ~30GB of weights for 7.6B params | Hub storage | saved in fp32; we load in bf16 |
| "checkpoint_2", `c_thought 2`, `max_latent_stage 2`, data = "custom reasoning traces with spacy-segmented steps, ending in `In summary:`" | README | **not** GSM8K; unknown answer format; checkpoint_2 may be an early stage |

**Most likely:** the latent ids are 151665 / 151666 / 151667 = `<|start-latent|>` / `<|end-latent|>` / `<|latent|>`
(run.py adds them in that order, then resizes). This is an **inference** and must be checked (Part 0).
The model card says "use like any other Qwen model" with `model.generate()`. That is the Phase 4 mistake: no
hidden-state feedback. We would use our `fill_latents()` (it works for any model, see below).

---

## 2. Risks (in order)

- **R1: latent ids are a guess.** Check: add the 3 tokens to the tokenizer in run.py order, confirm
  `len(tokenizer) == 151668` and the ids are 151665-7. Then check that those 3 embedding rows started as copies
  of the `<<` row (run.py line 149): after training they should still be close to it (cosine) and differ from
  random rows. Without this check, every result is meaningless.
- **R2: little latent training.** "checkpoint_2" with `max_latent_stage 2` could be stage 0 (plain CoT, no latents
  yet) if `epochs_per_stage >= 3`. Check: the latent rows moved from the `<<` init, and S3 vs S5 outputs differ on
  some questions. If the latent rows equal `<<` exactly, the model never trained with latents: **stop**.
- **R3: unknown training format** ("In summary:"). The prompts stay the Phase 4/5 few-shot prompts (same as before,
  so the study stays comparable). Part 0 also probes the "native" format `question\n<|start-latent|>` + 4 latents +
  `<|end-latent|>` to see what the model writes.
- **R4: ceiling effect.** Qwen2.5-7B-Instruct alone gets ~85-90% on GSM8K with chain of thought. S2 may be near
  the top. A small S3-S5 gap would still be measurable with 1319 paired questions.
- **R5: bf16.** Float noise is larger than in GPT-2 fp32. Smoke checks use a looser tolerance (1e-2 relative).
- **R6: time.** About 25-40 tokens/s per process. Few-shot loop with up to 2 x 128 tokens: ~5-8 s per question,
  ~2-3 h per setting on the full 1319. Run S1-S5 as 5 separate 1-GPU jobs (or 2-3 processes per L40S, 15GB each).
  Estimated wall time ~3 h.

Also from Paper Appendix C.2: Coconut helps Llama 3B/8B much less than GPT-2. A "no difference" result is likely,
but unlike the GPT-2 floor, it is a real, testable result.

---

## 3. What changes in the code (small)

`fill_latents`, `greedy_generate`, prompts, scoring and stats are model-independent and are reused unchanged.
- `common.py`: add `load_qwen_coconut()`. Load `agurung/coconut-qwen2.5-7b` in bf16, add the 3 tokens (assert ids
  151665-7, no resize needed), wrap in `Coconut(...)` (it already handles non-GPT-2 models via
  `get_input_embeddings()`). `--model gpt2|qwen7b` flag in `run_fewshot.py` and `smoke_test.py`.
  `MAX_POSITIONS` per model (GPT-2 1024, Qwen 32768).
- Stage length: GPT-2 used 64 new tokens per stage. A 7B model writes longer steps: **proposal 128 per stage,
  S1 128, S2 256** (decide after the smoke test; same for all loop settings).
- Stop strings: same, plus `<|im_end|>` / `<|endoftext|>` as EOS.
- Part 0 job (`qwen_check.slurm`): R1 + R2 checks, the native-format probe, and 10 questions x S1-S5 printed.
  **Gate:** if R1 or R2 fails, stop and report.
- Results go to `coconut/results/qwen7b/`, so GPT-2 results stay as they are.

---

## 4. Parts (after approval)

| Part | What | Who |
|---|---|---|
| 1 | `common.py`: Qwen loader + model flag | Claude |
| 2 | `qwen_check.py` + `qwen_check.slurm` (R1, R2, native probe, 10-question preview) | Claude |
| 3 | push / pull / sbatch the check | user |
| 4 | read the check, decide tokens per stage, **gate** | Claude + user |
| 5 | `run_fewshot.py` / `smoke_test.py` / `score.py` model flag + `run_qwen7b.slurm` (5 jobs) | Claude |
| 6 | full run | user |
| 7 | `analysis.ipynb` for both models | Claude (nbconvert run by user) |

---

## 5. Open decisions for the user

- D1: Approve the pivot with the Part 0 gate (stop if the latent tokens can't be confirmed)?
- D2: Also run the cheap GPT-2 control in its own format (6 latents, plain embeddings instead of feedback = S5 in
  the native format, ~5 min GPU)? It tests H1 where GPT-2 works, whatever happens with Qwen.
- D3: Tokens per stage: 128 (proposal) or other?

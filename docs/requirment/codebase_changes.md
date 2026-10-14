# Codebase Changes

This file concisely documents all modifications made to the repository across the project lifecycle. For technical context, motivation, and debugging history, please see [`detailed_explanations.md`](file:///home/eran.b/neuroAI/docs/requirment/detailed_explanations.md). For complete empirical results and comparative tables, see the comprehensive report in [`docs/report/entailmentbank_report.md`](file:///home/eran.b/neuroAI/docs/report/entailmentbank_report.md).

---

## 1. Phase 1: Replication & Local Compute Adaptation
- **Modified `args/prosqa_coconut.yaml`:** Adapted training configuration for our local 8GB VRAM constraint (RTX 2080) by setting per-device `batch_size=2` with gradient accumulation steps of 64 (maintaining effective batch size 128).
- **Created `run_prosqa.slurm`:** Configured PyTorch `torchrun` and conda environment variables for SLURM cluster execution.

---

## 2. Phase 2: Dynamic Interleaved Thinking on GSM8K (GPT-2 124M - Failed)
- **Updated Plan (`interleaved_thinking_plan.md`):** Tested whether an instruction-tuned model could dynamically trigger latent reasoning loops (`<|start-latent|>`) during text generation.
- **Created Scripts:** `section_data.py`, `interleaved_coconut.py`, `train_interleaved.py`, and execution SLURM scripts.
- **Results:** Model collapsed to 1.44% accuracy (vs. 8.72% fine-tuned baseline) due to severe arithmetic drift, unigram repetition loops, and gradient optimization shock. Detailed in `docs/report/interleaved_latent_report.md` and abandoned.

---

## 3. Phase 3: Out-of-Domain Logic on EntailmentBank (GPT-2 124M - Successful)
- **Created `parse_entailment.py` & `.slurm`:** Extracted the official EntailmentBank test split into step-by-step decomposed logical prompts.
- **Created `eval_entailment_baseline.py`:** Evaluated one-shot full proof generation without latent thoughts.
- **Created `eval_entailment_decomposed.py`:** Evaluated multi-step reasoning by generating intermediate proof steps as continuous latent states.
- **Created `eval_all.slurm` & `eval_scorer_only.slurm`:** Automated execution and integrated the official AllenAI BLEURT evaluation pipeline.
- **Results:** Decomposed latent reasoning achieved 97% structural tree accuracy, completely outperforming the 0% baseline failure on the 124M parameter model.

---

## 4. Phase 4: Qwen2.5-1.5B Iterative Deduction & Matched Discrete Controls (Final Experiment)
- **Model Selection:** Adopted `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` in `bfloat16`. Cleaned out obsolete 7B references which proved infeasible due to GPU Out-Of-Memory (OOM) on our single 8GB RTX 2080 GPU.
- **Baseline Scripts (Settings 1 & 2):**
  - [`eval_qwen_1.5b_baselines.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_baselines.py): Evaluates Setting 1 (Raw Baseline A: direct one-shot prompt) and Setting 2 (Discrete CoT Baseline B: all-at-once context) on EntailmentBank.
  - [`eval_qwen_1.5b_gsm8k_baselines.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_gsm8k_baselines.py): Evaluates Setting 1 (Raw Baseline A) and Setting 2 (Discrete CoT Baseline B: "Let's think step by step") on GSM8K.
- **Iterative Latent Deduction Scripts (Setting 3 - Coconut Latents, $K=4$):**
  - [`eval_qwen_1.5b_gsm8k_less_handholding.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_gsm8k_less_handholding.py): Two-stage iterative math reasoning inserting 2 `<c_thought>` continuous latent tokens per stage ($K=4$ total). Grounded with 3 actual GSM8K training exemplars.
  - [`eval_qwen_1.5b_entailment_fewshot.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_entailment_fewshot.py): Two-stage iterative science reasoning with 2 `<c_thought>` latent tokens per stage, grounded in EntailmentBank training exemplars.
- **Matched Discrete Control Scripts (Setting 4 - Strict No-Latents Baseline):**
  - [`eval_qwen_1.5b_gsm8k_no_latents.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_gsm8k_no_latents.py): Held 100% identical prompt structure, wording, few-shot questions, and two-stage turn boundaries as Setting 3, but with zero continuous latent tokens (pure discrete text chaining).
  - [`eval_qwen_1.5b_entailment_no_latents.py`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_entailment_no_latents.py): Matched discrete two-stage control for EntailmentBank without latent tokens.
- **Unified Evaluation & Scoring:**
  - [`score_outputs.py`](file:///home/eran.b/neuroAI/coconut/score_outputs.py): Evaluator combining exact numerical regex extraction for GSM8K and CPU-isolated `bleurt-large-512` scoring for EntailmentBank.
  - [`score_baselines.py`](file:///home/eran.b/neuroAI/coconut/score_baselines.py): Evaluates Settings 1 and 2 predictions.
  - [`eval_qwen_1.5b_iterative_all.slurm`](file:///home/eran.b/neuroAI/coconut/eval_qwen_1.5b_iterative_all.slurm): Master SLURM script executing all four experimental settings and evaluation in sequence.
- **Repository Cleanup:**
  - Removed obsolete 2.1GB zip archive and dead legacy test scripts to ensure a clean git footprint.

---

## 5. Phase 5: GPT-2 Coconut re-run on GSM8K (2026-10-15, code written, not run yet)
- **Plan:** `docs/planning/gpt2_coconut_phase5_impl.md` (the *how*); `KNOWLEDGE_RETENTION.md` sections 5-6 (the *what*).
- **Why:** in Phase 4 the "latents" were a plain text token (`<c_thought>`, not even in the vocab) and `model.generate()`; no hidden state was fed back. Phase 5 runs the same prompts on the real GPT-2 Coconut checkpoint (`tsrigo/coconut/stage4_checkpoint_25`) with real hidden-state feedback, plus a pause-token control (S5).
- **Official files unchanged:** `coconut.py`, `run.py`, `dataset.py`, `utils.py` are only imported.
- **New code in `coconut/experiments/`:**
  - `common.py`: loads GPT-2 + the checkpoint like `run.py` and refuses a checkpoint whose keys do not match. `fill_latents()` fills each `<|latent|>` slot with the previous position's last hidden state. We need our own version because the official `Coconut.forward` assumes all latents are in one block and crashes (IndexError) on the stage-2 prompt, which has two blocks. One greedy loop for all settings, with stop strings and a 1024-token check. Answer extraction removes commas first.
  - `prompts.py`: the Phase 4 GSM8K few-shot prompt (verbatim) and the S1-S5 prompt builders. The latent block is `<|start-latent|><|latent|><|latent|><|end-latent|>`, and stage 2 now keeps the stage-1 question line (Phase 4 bug).
  - `run_native.py`: Part 1. The model's own format with 0/2/4/6 latents, input built by the official `dataset.py`, official `Coconut.generate`, `run.py` scoring rule.
  - `run_fewshot.py`: Part 2, one setting per run (S1-S5), with a length guard for GPT-2's 1024 tokens. Generated text is cleaned of the 3 marker tokens before reuse (they are not "special" tokens).
  - `score.py`: accuracy, 95% Wilson CI, exact McNemar, paired bootstrap CI, written to `coconut/results/gpt2/results.json`.
  - `smoke_test.py`: checks C0-C8. Examples: the prompt equals the Phase 4 one; our latent filling equals the official one where the official one works; our greedy loop equals `Coconut.generate`.
  - `run_gpt2.slurm`: one Newton job in the Technion template format: smoke test, then the full runs on one L40S, then scoring.
- **Other:** `environment_newton.yml` (lean env for Newton, same versions, no TensorFlow). `.gitignore` now skips the GSM8K train JSON (over GitHub's 100MB limit).

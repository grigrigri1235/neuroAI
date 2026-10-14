# Detailed Explanations of Codebase Changes

This document provides the full technical context, architectural rationale, and debugging history for the modifications summarized in [`codebase_changes.md`](file:///home/eran.b/neuroAI/docs/requirment/codebase_changes.md). The empirical results and analysis are documented in [`docs/report/entailmentbank_report.md`](file:///home/eran.b/neuroAI/docs/report/entailmentbank_report.md).

---

## 1. Hardware Constraints & VRAM Scaling Rationale

Throughout this research project, experiments were executed on an NVIDIA GeForce RTX 2080 GPU with **8GB of VRAM**. This hardware boundary governed our architectural scaling decisions across three model tiers:

1. **GPT-2 (124M Parameters):**
   - Fits easily in memory (~500MB weights), but possesses severe semantic and capacity bottlenecks. While capable of structural syntactic patterns, it completely lacks the arithmetic capacity required for multi-hop math reasoning.
2. **Qwen2.5-7B (`agurung/coconut-qwen2.5-7b`):**
   - In `bfloat16`, 7B weights require ~15GB of memory. Running on an 8GB GPU required deep CPU offloading via HuggingFace `device_map="auto"`. Inference took over 25 hours for a single evaluation pass and repeatedly suffered CUDA Out-Of-Memory (OOM) errors during intermediate KV-cache allocations.
3. **Qwen2.5-1.5B (`CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` - The Sweet Spot):**
   - In `bfloat16`, 1.5B weights require only ~3GB of VRAM. This leaves ~5GB of memory headroom on the RTX 2080 for full-context KV-caches and continuous latent embeddings without CPU offloading. Evaluations completed cleanly in under 2 minutes per 100 test samples.

---

## 2. Transition from Training to Pre-Trained Coconut Checkpoints

Early in the project, we configured end-to-end training of the Coconut architecture from scratch on our single GPU (`run_prosqa.slurm`). Due to cluster execution time limits and slow throughput, training from scratch was abandoned in favor of official pre-trained checkpoints:
- **Phase 1-3:** `tsrigo/coconut` (GPT-2 124M base).
- **Phase 4:** `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` (Qwen2.5-1.5B base).

---

## 3. Dynamic Interleaved Thinking (Failure Analysis on 124M)

Our initial Part 2 extension idea sought to train an autonomous agent that could decide when to enter latent thinking dynamically using `<|start-latent|>` tokens on GSM8K.
- **Outcome:** Model accuracy collapsed to 1.44% (against the 8.72% fine-tuned baseline).
- **Failure Mechanisms:**
  - *Optimization Shock:* Updating both standard token generation and dynamic latent transitions destabilized language modeling probabilities.
  - *Unigram Repetition Loops:* The 124M model repeatedly trapped itself in unigram loops (e.g., repeatedly generating punctuation or single words).
  - *Arithmetic Drift:* Early arithmetic errors compounded exponentially in subsequent steps.

---

## 4. Decomposed Logic on EntailmentBank (Success on 124M)

To isolate reasoning structure from raw arithmetic calculations, we pivoted to **EntailmentBank** (science logic proofs).
- **Method:** We decomposed the reasoning tree into multi-hop deductive steps, passing intermediate conclusions as continuous latent embeddings.
- **Results:** Standard baseline one-shot proof generation failed completely (0% tree accuracy). The decomposed latent approach achieved 97% structural tree accuracy, demonstrating that multi-step latent prompting can bypass capacity limits for structured logical deduction.

---

## 5. Diagnosing and Eliminating Tautology Collapse on Qwen2.5-1.5B

When we initially implemented two-stage deduction on Qwen2.5-1.5B using generic, synthetic few-shot demonstrations (e.g., `Hans has 12 apples...`), the model exhibited **Tautology Collapse**:
- The model generated trivial, non-informative arithmetic statements at Stage 1 (e.g., `7 + 7 = 14`), failing to perform actual multi-hop deductions relevant to the problem premise.
- **Resolution:** We replaced synthetic demonstrations with **grounded exemplars selected directly from the official GSM8K and EntailmentBank training sets**. Grounding the prompt in authentic domain exemplars completely resolved tautology collapse, establishing clean and purposeful intermediate deductions.

---

## 6. Scientific Rigor: Designing Setting 4 (The Matched Discrete Control)

To scientifically determine whether continuous latent embeddings provide genuine reasoning benefits over standard text, we designed **Setting 4 (No-Latents Matched Control)**:
- **Scientific Control:** Setting 4 uses the exact same two-stage prompts, question wording, few-shot training exemplars, turn boundaries, and regex parsing logic as Setting 3 (Coconut Latents, $K=4$). The *only* difference is that Setting 4 uses pure discrete text without `<c_thought>` latent tokens.
- **Findings:**
  - On GSM8K, continuous latents **doubled accuracy (12.00% vs. 6.00%)**.
  - On EntailmentBank, continuous latents produced higher semantic alignment (**0.4010 vs. 0.3924 BLEURT**).
  - *Why Pure Text Failed (6%):* Discrete text concatenation triggered compounding drift—the model began reciting numbers from the few-shot examples (e.g., repeatedly predicting `42`) or treating early textual errors as immutable ground truth. Continuous latent vectors buffered the internal semantic state, eliminating this discrete drift.

---

## 7. Evaluation Pipeline: PyTorch & TensorFlow CUDA Isolation for BLEURT

The official EntailmentBank semantic metric utilizes `bleurt-large-512`, which runs on TensorFlow 2.8. Meanwhile, Qwen2.5 generation runs in PyTorch with CUDA.
- **Issue:** Attempting to initialize both PyTorch CUDA and TensorFlow CUDA in the same runtime process produced fatal GPU memory allocation conflicts and library crashes.
- **Resolution:** We isolated the evaluation stages in [`score_outputs.py`](file:///home/eran.b/neuroAI/coconut/score_outputs.py) and [`score_baselines.py`](file:///home/eran.b/neuroAI/coconut/score_baselines.py) by setting `CUDA_VISIBLE_DEVICES=""` for TensorFlow, running BLEURT evaluation strictly on CPU while PyTorch handled GPU generation.

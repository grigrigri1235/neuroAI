# Plan: test interleaved latent thinking on `agurung/coconut-qwen3-4b-ff` (DRAFT, waiting for approval)

Date: 2026-10-15. Status: APPROVED ("proceed"). Parts 1-3 written: `common.load_coconut_hf`,
`experiments/qwen3_check.py`, `experiments/qwen3_check.slurm`. Next: user prefetches the model and runs the gate.

## 0. Why this model (and why not the others)

- **The user's hypothesis (Idea 2, interleaved latent thinking):** latent thoughts placed **between** natural-language
  reasoning steps (step question -> latents -> written step -> step question -> latents -> written step) help
  multi-step reasoning, without hand-made sub-questions. Test: S3 (real latents) > S5 (same slots, no feedback)
  and S3 > S4 (no slots), same 2-stage few-shot loop as before, GSM8K test (1319), paired.
- **GPT-2 Coconut (124M):** latents work in its own format (feedback vs plain at 6 latents: 27.1% vs 9.6%), but
  it ignores the few-shot step format (writes `### <n>` at once): floor effect, so the interleaving was **not testable**.
- **`agurung/coconut-qwen2.5-7b`:** writes garbage (`a additional a additional ...`) even in S1, with no latents
  at all. Its 3 latent rows are still the untouched `<<` init (output rows cosine 1.0). Treated as broken. Abandoned.
- **`agurung/coconut-qwen3-4b-ff` (this plan):** documented and late checkpoint. Read from the Hub on 2026-10-15:

| Fact | Source |
|---|---|
| base `Qwen/Qwen3-4B-Instruct-2507`, 4.0B params, bf16 (~8GB) | latent_metadata.json, config.json |
| `checkpoint_32` of run `qwen3-coconut-ff-v3` (late, not an early stage) | latent_metadata.json |
| tokenizer **includes** the latent tokens: start 151669, end 151670, latent 151671; vocab 151672 | README, latent_metadata.json |
| original Coconut checkpoint included (`latent_checkpoint.pt`, aux key `embedding.weight`) | README |
| tied input/output embeddings; context 262144 | config.json |
| **saved with transformers 5.5.4**; `rope_theta` is nested as `rope_parameters: {rope_theta: 5000000}` | config.json |
| training data / prompt format / c_thought / number of stages: **not documented** | - |

## 1. Risks

- **R1 (silent!): wrong RoPE.** Our env is transformers 4.52.4, which reads a **top-level** `rope_theta`. With the v5
  nested format it would silently use its default (1,000,000 instead of 5,000,000): a wrong model with no error.
  Fix: load the config, copy `rope_parameters["rope_theta"]` to `config.rope_theta`, assert 5e6, then load the model.
  (Alternative: a separate env with transformers 5.x. Rejected: the official `coconut.py` uses the old cache API.)
- **R2: tokenizer files from transformers 5.x** may not load in 4.52. Fallback: base `Qwen/Qwen3-4B-Instruct-2507`
  tokenizer + `add_tokens` in run.py order (expected ids 151669-71), checked against `latent_metadata.json`.
- **R3: our loop on Qwen.** The 7B garbage could in theory come from our loop, not the model. Gate G1 checks our
  greedy loop against plain HF `model.generate(do_sample=False)` on prompts without latents (same tokens expected).
- **R4: unknown training format / latent count.** Gate G2 probes the native format with 0/2/4/6 latents.
- **R5: latents may do nothing in this checkpoint.** If feedback ~ plain ~ 0 latents in G2, interleaving can't help:
  stop and report.

## 2. Gate job (Part 0, ~20-30 min on one L40S)

`qwen3_check.py` + `qwen3_check.slurm`:
- **G0 load:** ids 151669-71 match the metadata, embedding rows = len(tokenizer) = 151672, rope_theta = 5e6.
- **G1 loop:** on 5 S1 prompts and 5 S4 prompts (no latents), our greedy loop = HF greedy `generate` (first 64 tokens).
  Also print both outputs: the text must be readable (not the 7B garbage).
- **G2 latents work (native format):** `question + "\n" + <|start-latent|> + N x <|latent|> + <|end-latent|>`,
  N in {0, 2, 4, 6}, feedback vs plain, first 100 GSM8K questions, 256 new tokens, accuracy = last number.
  Pass if feedback beats plain at some N (and print outputs to see the trained answer format).
- **G3 format:** 10 questions x S1-S5 with the few-shot loop (128 tokens per stage): does the model write a real
  step in stage 1 and continue in stage 2?
- **Gate rule:** G0, G1 must pass. If G2 shows no latent effect, or G3 shows the model ignores the step format,
  stop and report before the full run.

## 3. Full run (only after the gate, separate approval)

S1-S5 on all 1319 questions, same code as GPT-2 (`run_fewshot.py --model qwen3`), results in
`coconut/results/qwen3/`, same stats (accuracy, Wilson CI, McNemar, paired bootstrap). H1: S3 > S5. H2: S3 > S4.
Estimated ~1-2 h per setting, run as parallel jobs. The full-run plan will be finalized after the gate.

## 4. Parts (this approval: Parts 1-3 only)

| Part | What | Who |
|---|---|---|
| 1 | `common.load_coconut_hf()`: generic loader for HF Coconut exports (rope fix, tokenizer check, eos ids) | Claude |
| 2 | `qwen3_check.py` (G0-G3) | Claude |
| 3 | `qwen3_check.slurm` + exact commands (prefetch ~8GB, only the needed files) | Claude |
| 4 | run the gate | user |
| 5 | read the logs, gate decision | Claude + user |

## 5. Honest framing for the report

Whatever the gate shows, the report says: GPT-2 (paper model) = latents work but the model can't follow steps;
Qwen2.5-7B export = broken; Qwen3-4B = the test of interleaving (if the gate passes). A "no effect" result on
Qwen3-4B with a working model and a passed gate is a real result.

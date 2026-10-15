# Plan: interleaved latent thinking with SIM-CoT GPT-2 (DRAFT, waiting for approval)

Date: 2026-10-15. Status: APPROVED ("lets go with minimal 2 round"). Code written: `common.load_simcot`,
`common.explain`, `experiments/simcot_run.py` + `.slurm`. GPT-2 is cheap, so **one job** runs the gate (G0-G2) and
the main test (L, T2, T2-gold, all 1319). The main numbers count only if G2 shows readable steps.
Explainer decoding follows SIM-CoT's code: input = the 2 thought vectors, then the main model's embeddings of the
tokens written so far, positions from 1, stop at EOS. Greedy (they sample only for visualisation). Training target
= the step text + "\n".

## 0. Why GPT-2 Coconut (`tsrigo/coconut`) failed the interleaving test

**What we ran.** The few-shot 2-stage loop (S1-S5) on all 1319 GSM8K test questions (job 1449565, Newton,
`coconut/results/gpt2/results.json`). The prompt shows 3 worked examples. Each has two
`Question: What can we deduce in one logical step?` turns with a written step after `Answer:`. Then our question
follows, and the model should write step 1. In S3 a latent block (2 real continuous thoughts) sits before each
`Answer:`.

**What happened.**
- In every setting and every stage, the model wrote `### <number>` (exactly 2 tokens) and stopped (EOS).
  It never wrote a step.
- Stage 2 repeated stage 1. The number was often `36`, copied from few-shot Example 2 (`12 + 24 = 36 pages`).
- Accuracy: S1 3.4%, S2 10.2%, S3 1.1%, S4 0.9%, S5 1.1%.
- S3 vs S5: 0 questions differ in correctness (McNemar p = 1). S3 vs S4: 2 differ (p = 0.5).

**Why.**
1. **The checkpoint knows only one format.** It is the final Coconut stage (stage 4, epoch 25). It was trained on
   ~385k GSM8K examples of exactly `question \n <|start-latent|> 6 latents <|end-latent|> ### answer`, with every
   language step replaced by latents. After 25 epochs the habit "after a question comes `### answer`" overrides
   everything else in the prompt.
2. **GPT-2 (124M) is too small to learn a new format from 3 examples.** Following few-shot examples
   (in-context learning) appears only in much larger models. So the examples did not teach it to "write one step".
3. **So the interleaving never happened.** The idea needs `thought -> written step -> thought -> written step`.
   The model wrote no steps, so there was nothing between the latent blocks. S3, S4 and S5 all collapsed to the
   same 2-token answer: a **floor effect**. The real question (do the thoughts help the step-by-step chain?) could
   not show up in the numbers at all.
4. **This was not a bug in our latent code.** The smoke test showed our latent filling equals the official one
   (difference 0.0), and our loop equals `Coconut.generate` (20/20). In the model's own format the same latents work:
   10.4% -> 27.1% with 0 -> 6 latents, and +17.5 points over plain embeddings at 6 latents (p ~ 7e-50).

**Lesson for the next model.** Interleaving needs a model that can (a) run real latents and (b) write a step
between them. GPT-2 Coconut has (a) but not (b). The Qwen exports had neither on GSM8K (7B broken, 3B-4B trained on
a yes/no task). SIM-CoT GPT-2 (below) still lacks (b) via prompting, but its explainer can write the steps
**from the thoughts themselves**, which gives (b) without needing in-context learning.

## 1. The model (read from the Hub and GitHub, 2026-10-15)

- `internlm/SIM_COT-GPT2-Coconut`, paper arXiv 2509.20317 (SIM-CoT). One file `checkpoint_28` (~1.4GB) + `config.json`.
- GPT-2 + Coconut, trained on the **same GSM8K (iCoT) data and format** as our GPT-2 (`question\n<bot> latents <eot> ### answer`).
  Eval config: `c_thought 2`, `max_latent_stage 5` -> up to **10 latents**.
- Inference = the official Coconut feedback (their `generate` = original). So our code works as is.
- Training adds an **explainer**: a second GPT-2 (`expainable_llm`) that learns to turn each continuous thought
  into the written step it replaces (step-level supervision). Paper claim: Coconut GPT-2 + SIM-CoT is about +8 points
  over plain Coconut.
- Expected: like our GPT-2, it will **ignore the few-shot step prompts** (same base, same format). So it does not fix
  the S1-S5 floor effect.

## 2. Why it still fits the user's idea

If the checkpoint contains the explainer, the model can **write its own steps from its own thoughts**. That allows
a real interleaved chain in the model's own domain, with no hand-made sub-questions and no few-shot prompt:

```
question -> 2 latents -> explainer writes step 1 as text -> step 1 text goes into the context
         -> 2 latents -> explainer writes step 2 -> ... -> answer
```

Settings (all GSM8K test 1319, paired):
- **L (all latent):** 10 latents, then answer (native, = the paper setting).
- **I (interleaved):** latent pair -> decoded step text -> next latent pair ... (the user's idea).
- **I-plain (control):** same interleaved layout, but plain embeddings in the latent slots (no feedback).
- (info) **decoded-only:** the explainer's steps as plain text CoT, no latents.

### 2b. The user's loop design (2026-10-15) = setting **R (re-prompt)**

```
round 1: question                         -> <bot> 10 latents <eot> -> model answer a1; explainer(pair 1) -> step s1
round 2: question + "\n" + s1             -> <bot> 10 latents <eot> -> a2; explainer(pair 1) -> s2
round 3: question + "\n" + s1 + s2        -> ...
stop: explainer repeats an earlier step (no progress) / writes nothing new / 5 rounds (trained max steps).
final answer = the model's ### answer of the last round.
```
- **The user's worry:** the model was trained with a fixed number of thoughts; if each round has fewer, it may "never
  reach the answer". Fix: give the **full trained budget (10 latents) every round**. The final-stage model always
  writes `### answer` right after `<eot>`, so every round has an answer. Each round only has fewer steps left to
  solve, because the written steps are in the question.
- **Real open risk:** after s1 is in the question, does pair 1 now encode step 2 (progress), or step 1 again
  (a loop)? Unknown, so the pilot measures "progress rate" (new step per round).
- Also logged every round: the model's answer, to see whether it changes or improves round by round.

### 2c. CHOSEN (user, 2026-10-15): the minimal 2-round version

The user's choice: keep it to 2 rounds (like the original 2-stage idea), so failures stay easy to understand. All the
earlier failures are documented (section 0, KNOWLEDGE_RETENTION.md section 16).

| Setting | What happens |
|---|---|
| **L** (baseline) | question -> 10 latents -> `### answer` (one round) |
| **T2** (main) | round 1: question -> 10 latents; explainer(thought pair 1) -> step s1. Round 2: question + "\n" + s1 -> 10 latents -> answer |
| **T2-gold** (best case) | same as T2, but s1 = the **true** step 1 from `gsm_test.json` |

Dropped by the user (2026-10-15): **T2-plain** (round 2 with plain latents; "we know latent thinking works". It would
have tested whether the round-2 thoughts still matter once s1 is written, so it can be added later if needed) and
**A** (as many rounds as the true number of steps).

- The final answer always comes from the SIM-CoT model (it writes `### answer` after `<eot>`). Base GPT-2 is not used
  to answer: it was never trained on math.
- Full trained latent budget (10) every round.
- Metrics: answer accuracy (paired: T2 vs L, T2 vs T2-gold); exact match of the explainer's s1 vs the true step 1.
- R (more rounds) and I (one sequence) from 2b are dropped for now.

The earlier R / I ideas, kept for reference:
So the run has **R** (re-prompt, the user's design) and **I** (one sequence: pair, step text, pair, ...).
The pilot (gate G3, 20 questions) shows which one progresses. The full run uses the ones that do.

Hypothesis H: R or I > L (written steps between thought groups help), and > the same layout with plain latents
(the thoughts carry the steps).
Bonus result: **readable thoughts.** Step accuracy of the decoded thoughts vs the gold `<<a*b=c>>` steps
(monitoring hidden reasoning, Idea 3).
Honest limits: the base model was trained with "latents first, then text", never "latent, text, latent". So I is
partly outside its training. That is part of what we measure.

## 3. Gate job first (~15 min)

- G0: download, load `base_causallm.*` into our Coconut wrapper. List all key prefixes:
  **is `expainable_llm.*` in the checkpoint?** If not, section 2 is impossible: stop and report.
- G1: native sweep k = 0..5 (0-10 latents), all 1319 (fast). Reproduce the paper's SIM-CoT GPT-2 number, and compare
  with our plain Coconut (27.1%).
- G2: read their `coconut.py` explainer input format exactly, and decode the 5 thought pairs of 20 questions.
  Print them next to the gold steps: are they readable steps?
- G3 (pilot): L, T2, T2-gold on 20 questions. Print s1 vs the true step 1 and both rounds' answers.
- Gate rule: G0 explainer present + G2 readable steps + G3 progress -> plan the full run (separate approval).

## 4. Parts (this approval: gate only)

1. read SIM-CoT `coconut.py` explainer code (read-only, web), write `simcot_check.py` (G0-G2)
2. `simcot_check.slurm` + commands (download ~1.5GB)
3. user runs, Claude reads, gate decision

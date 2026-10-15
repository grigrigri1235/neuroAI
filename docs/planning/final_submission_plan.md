# Plan: final submission (report + clean repo)

Date: 2026-10-15. Status: WAITING FOR APPROVAL.
User request: "document it all in KR, write the report, remove tons of clutter, prepare it for a final submission".
Add a future work section: why it did not work, and that a real fix needs fine-tuning, not prompt tricks.

Deliverables (course, `docs/requirment/requirment.md`): a 6-page report (CogSci template `final_rep/main.tex`)
+ runnable code. Possible 30-min oral check, so every claim must match the code, in plain words.

---

## 1. Report: structure and page budget (6 pages, references and acknowledgments extra)

Working title: **"Continuous Thoughts Need Their Own Format: Reproducing Coconut and Testing Written Steps
Between Latent Reasoning Rounds"** (open question Q2).

| # | Section | Content | Pages |
|---|---|---|---|
| 0 | Abstract (≤150 words) | what we did, the 2 main numbers, the negative result, its reason | - |
| 1 | Introduction | Coconut idea in 3 sentences; our two questions (reproduce; do written steps between thought rounds help?) | 0.5 |
| 2 | Background | Coconut (hidden state fed back as input), curriculum, final model format `question -> latents -> ### answer`; SIM-CoT and its explainer | 0.5 |
| 3 | Part 1: Reproduction | inference only (training exemption), GPT-2 checkpoint, GSM8K test 1319, 0-6 latents, our check that our latent filling = official code. **Fig 1**: accuracy vs number of latents: Coconut (feedback), Coconut (plain, no feedback), SIM-CoT (0-10). **Table 1**: numbers + CIs + paper 34.1% | 1.0 |
| 4 | Part 2: follow-up ideas | 4 ideas in one short list (dynamic thinking length, written steps between thought rounds = chosen, steering/reading thoughts, thought transplant) | 0.25 |
| 5 | Part 2: experiment | hypothesis; settings L, T2, T2-gold, T2-in, T2-in-gold (one small diagram/example of each input); metrics (accuracy, Wilson CI, McNemar, paired bootstrap) | 0.75 |
| 6 | Part 2: results | **Fig 2**: bar chart of the 5 settings with CIs. **Table 2**: paired differences. **Table 3**: failure cases chosen by rule (first questions where L is right and the step versions are wrong), incl. q0. Decoded thoughts example (the explainer turns thoughts into `<<3*68=204>>`) | 1.25 |
| 6b | Part 2: fine-tune (if run) | FT-mix vs FT-plain, L / T2 / T2-gold (plan `simcot_finetune_plan.md`). Positive or negative, it goes here | 0.5 |
| 8 | Discussion: why it did not work | (a) format lock-in: the final model saw only `question -> latents -> answer` for 25+ epochs; (b) any edit to the question disturbs it, even a true fact (T2-gold, T2-in-gold < L); (c) inserted numbers may act as extra quantities (distraction, Shi et al. 2023); (d) 124M model, no in-context learning; (e) explainer errors cost ~3-4 points; (f) inference only, no training budget | 0.5 |
| 9 | Future work | **fine-tune on an interleaved curriculum** (training data `question -> latents -> written step -> latents -> ... -> answer`, so the model learns to read its own steps); train with the explainer in the loop; larger models; the cheap ideas (dynamic length, reading/steering thoughts) | 0.5 |
| 10 | Conclusion | 3-4 sentences | 0.25 |
| 11 | **What we tried along the way** (end of the paper, user request) | the attempt table of section 6 below: one row per attempt, what / result / why it failed / what it taught us. Points to the full log `docs/attempt_log.md` | 0.75 |

Page budget with 6b and 11 added: trim sections 2 (0.4), 3 (0.9), 6 (1.1), 8 (0.4), 9 (0.4) so the total stays 6.
| - | Acknowledgments | **AI use statement** (required): Claude Code wrote code and drafts under our direction; we checked every result and claim; what we did ourselves | extra |
| - | Code availability | repo link, how to reproduce | extra |
| - | References | Coconut (Hao et al.), SIM-CoT (arXiv 2509.20317), iCoT (Deng et al.), GSM8K (Cobbe et al. 2021), GPT-2 (Radford et al. 2019), pause tokens (Goyal et al.), distraction (Shi et al. 2023). Check every entry on the web before use | extra |

Rules for the text: plain words, no hype ("proves", "decisive"), every number from the results JSON, say what can
and cannot be claimed. Negative results stated plainly, not as "defeat".
Numbers to verify on the web before writing: the SIM-CoT paper's GPT-2 Coconut + SIM-CoT GSM8K number.

## 2. Figures and tables: `coconut/analysis.ipynb`

- Reads only results JSON (`coconut/results/gpt2/results.json`, `.../simcot/simcot_run.json`,
  `.../simcot/simcot_t2in.json`). No GPU needed.
- Makes Fig 1, Fig 2 (PDF into `final_rep/figures/`) and prints Tables 1-3 (also writes the LaTeX rows).
- Must be delivered executed (global rule). The user runs:
  `jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True coconut/analysis.ipynb`
  (on Newton, where the env has matplotlib and jupyter), then pushes.

## 3. Repo cleanup

Step 0 (user, so the list is exact): `git ls-files > logs/git_files.txt`, push or tell Claude.
Before any removal (user): `git tag pre-cleanup && git push --tags` so everything stays recoverable.

**Remove** (old phases, invalid or superseded; stays in git history under the tag):
- Phase 4 Qwen 1.5B: `coconut/eval_qwen_1.5b_*.py`, `coconut/score_outputs.py`, `coconut/score_baselines.py`,
  Phase 4 `coconut/*.slurm`, `coconut/predictions_*.json`, `coconut/score_*_results.txt`, `coconut/slurm_logs/`
- Phase 3b EntailmentBank: `coconut/preprocessing/parse_entailment.py`, `coconut/data/entailmentbank/`,
  `entailmentbank_report.md` (root)
- Old over-claiming notes: `docs/requirment/summary.md`, `detailed_explanations.md`, `proposed_followup_ideas.md`
- Superseded plans: `docs/planning/gpt2_coconut_rerun_plan.md`, `less_handholding_plan.md`,
  `entailmentbank_experiment_plan.md`
- `coconut/data/gsm8k_test.json` only if no current script reads it (check first)

**Keep**: official Coconut code, `coconut/experiments/` (all current scripts incl. Qwen gate checks = documented
failures), `coconut/results/`, `logs/`, current plans, `KNOWLEDGE_RETENTION.md`, `workflow.md`, `environment*.yml`,
`final_rep/`, `og_coconut.*`, `docs/requirment/requirment.md`, `codebase_changes.md` (rewritten short and honest).

**Add**: root `README.md`: what the project is, results in 2 tables, how to reproduce (setup, the 3 jobs:
`run_gpt2.slurm`, `simcot_run.slurm`, `simcot_t2in.slurm`), where figures come from, repo map.

Exact remove commands are given to the user (Claude runs no bash).

## 4. Parts (after "proceed")

1. KR: final status, cleanup record, report status (kept up to date after every part)
2. `docs/requirment/codebase_changes.md`: rewrite short and honest
3. `coconut/analysis.ipynb` (figures + tables) -> user executes it, pushes
4. verify references and the SIM-CoT paper number (web, read-only)
5. `final_rep/main.tex`: sections 1-3
6. sections 4-7
7. sections 8-10, acknowledgments (AI use), references `.bib`
8. `README.md`
9. cleanup commands for the user (after the tag), KR repo map updated
10. final checklist: LaTeX compiles to ≤ 6 pages (user compiles, e.g. Overleaf or `latexmk`), numbers match JSON

## 6. "What we tried along the way": the attempt list (report section 11 + `docs/attempt_log.md`)

User request: show the lecturer the real effort. Every attempt is documented in KNOWLEDGE_RETENTION.md (sections
3, 4, 12, 16) and the logs. In the report: a table, one row each, plain words. `docs/attempt_log.md` (new, no page
limit): the same list with numbers, file names, job ids, and the git tag `pre-cleanup` where the removed old code
still lives. What I would write for each:

| # | Attempt | Result | Why it failed / what we would say | Lesson |
|---|---|---|---|---|
| 1 | Train Coconut ourselves (GPT-2, ProsQA, official code, RTX 2080 8GB) | stopped; >3 days estimated for 50 epochs | our GPU was too small for the paper's training; led to the training exemption and to released checkpoints | check compute before planning training |
| 2 | Let the model decide itself when to think in latents (GPT-2 trained to emit `<start-latent>`, GSM8K) | 1.44% vs 8.72% (our fine-tuned baseline) and 11.07% (zero-shot) | unstable training, repeated words, arithmetic drift. It did place the marker at plausible points | a new behaviour needs a curriculum, not one jump |
| 3 | Split GSM8K questions into sub-questions with a strong LLM, feed them to GPT-2 | ~8% on 100 questions | small GPT-2 cannot do the arithmetic even when the problem is split for it | decomposition does not fix a missing skill |
| 4 | EntailmentBank proofs with GPT-2 Coconut, one latent block per proof step | "97% proof structure", but the same with and without latents (BLEURT-20 0.555 vs 0.560) | the structure came from our script feeding the gold proof edges; the model wrote random digits (out of domain) | always run the no-latent control; a scaffold can do all the work |
| 5 | Qwen2.5-1.5B "Coconut" model, 2-stage step loop (GSM8K + EntailmentBank) | 12% vs 6% (100 questions) | **invalid as a Coconut test**: no hidden state was fed back (the latent token was plain text, not even in the vocabulary), unknown model, not significant (p≈0.2), no stop rule | check the method is really implemented; use enough data and a paired test |
| 5b | (inside 5) made-up few-shot examples | "tautology collapse" (`7 + 7 = 14`) | examples unrelated to the data | use real training examples as few-shot |
| 6 | Bigger models on the old GPU (Qwen2.5-7B, Gemma-3-4B) | did not fit 8GB; CPU offload >25 h per pass | hardware | - |
| 7 | Real GPT-2 Coconut, same 2-stage step prompts (S1-S5, 1319 questions) | floor: 0.9-3.4% in the loop settings; the model only wrote `### N` | the checkpoint knows one format; a 124M model cannot learn a new one from 3 examples | test in the model's own format first |
| 8 | A bigger Coconut model: `agurung/coconut-qwen2.5-7b` (gate check) | garbage text in every setting, even without latents | broken export (marker rows untrained) | gate-check every checkpoint in ~10 min |
| 9 | Another one: `agurung/coconut-qwen3-4b-ff` (gate check) | always `\boxed{Yes}`, 0/50 in the native format | trained on an undocumented yes/no task | a model card without data is a red flag |
| 10 | SIM-CoT GPT-2: write the model's own step 1 before a second thought round (T2, T2-gold) | 34.4% / 38.6% vs 44.7% | text after the question was never seen in training; even the true step hurts | layout matters more than step quality |
| 11 | Same, step inside the question before its last sentence (T2-in, T2-in-gold) | 24.5% / 27.2% | editing the question disturbs the model more; inserted numbers may act as extra quantities | prompting alone cannot teach a new layout |
| 12 | Short fine-tune on the step layout (if run) | TBD | TBD | TBD |

Also worth one sentence each (not separate attempts): we found and fixed a bug where the official `Coconut.forward`
crashes on two latent blocks (wrote our own filler, checked equal); and a wrong prompt format in the old stage 2.

Tone: each row = what we tried, what happened, why, what we learned. No excuses wording; "limitation" and
"lesson". Numbers for rows 1-3 come from our old notes (`docs/requirment/summary.md`, logs local only); say so in
the attempt log.

## 5. Answers from the user (2026-10-15)

- Q1: author **Eran Ben Eliyahou**, solo (no partner). Final-submission format with the name
  (`\cogscifinalcopy`), Technion, `eran.b@campus.technion.ac.il`.
- Q2: title **"Reproducing Coconut and Testing Written Steps Between Latent Thoughts"**.
- Q3: **remove**, no archive folder (git tag `pre-cleanup` first).
- File list: `logs/git_files.txt` (166 tracked files). Before removing anything, check that no current script
  reads a file on the list (e.g. `smoke_test.py` may compare against the Phase 4 prompt file, `gsm8k_test.json`
  may be read by `run_fewshot.py`); such files are kept.

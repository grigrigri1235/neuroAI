# EntailmentBank Experiment Plan (COMPLETED)

## Goal
Evaluate whether the Coconut model (124M, trained on GSM8K) can generalize its latent continuous thought to textual logical deduction. 

We compared two approaches:
1.  **Baseline (Non-Decomposed):** Ask the model to generate the entire proof tree in one shot.
2.  **Decomposed:** Ask the model step-by-step to generate intermediate conclusions (replaced with latents).

## Execution Summary
*   **Data:** EntailmentBank Task 1 test split (340 samples) parsed into `test_decomposed.json`.
*   **Model:** `tsrigo/coconut` (stage 4) forced into latent reasoning.
*   **Pipeline:** We created inference scripts and a massive `eval_all.slurm` script that successfully evaluated the model using the official AllenAI scorer (BLEURT/ROUGE).

## Final Results & Conclusion
**Baseline Model:** 0.0% structural accuracy, 0.00 BLEURT semantics. Complete failure to generate zero-shot proof trees.
**Decomposed Model:** 97.05% structural accuracy, 0.560 BLEURT semantics. 

**Conclusion:** The experiment was a resounding success. Step-by-step continuous latent prompting drastically improves zero-shot logical reasoning, bypassing the 124M model's capacity limit. (See `docs/report/entailmentbank_report.md` for the full breakdown).

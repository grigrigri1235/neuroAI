# Part 2 Extension: Proposed Follow-Up Research Ideas

This document formalizes the three follow-up research directions proposed for Part 2 (Extend) of the course project based on the Coconut paper (*Training Large Language Models to Reason in a Continuous Latent Space*), linking directly to our final comprehensive report in [`docs/report/entailmentbank_report.md`](file:///home/eran.b/neuroAI/docs/report/entailmentbank_report.md).

---

## Idea 1: Dynamic Latent Thinking Length via Resource Cost Optimization

### Problem Statement
In the standard Coconut implementation, the number of continuous latent thought steps $K$ is a static, fixed hyperparameter (e.g., $K=4$ or $K=6$) configured identically across all queries. This imposes two significant inefficiencies:
1. **Compute Waste on Simple Queries:** Easy or single-hop questions (e.g., direct fact retrieval or simple arithmetic) do not require multi-step latent reasoning, yet the model is forced to expend $K$ latent passes.
2. **Compute Bottlenecks on Complex Queries:** Hard multi-hop problems (e.g., complex multi-step math or multi-premise scientific deductions) are constrained by the fixed budget $K$ and cannot extend their latent reasoning capacity when needed.

### Proposed Mechanism
Introduce a lightweight prompt-level router/classifier module (or adapt the base model's first transformer layer) to dynamically predict the optimal latent thinking length $K \in \{0, 2, 4, 6, 8\}$ based on input prompt embeddings before reasoning commences.

### Mathematical Formulation
Inspired by the reconstruction/resource trade-off loss from the course, the router and model are optimized under a resource-regularized objective:
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \lambda \cdot \text{Cost}(K)$$

Where:
- $\mathcal{L}_{\text{task}}(\theta)$ is the standard cross-entropy task loss on the final target tokens.
- $\text{Cost}(K) = K \cdot c_{\text{step}}$ represents the computational penalty per latent token step.
- $\lambda \ge 0$ is a Pareto hyperparameter balancing reasoning accuracy against computational cost.

---

## Idea 2: Autonomous Multi-Step Deduction Without External Scaffolding (The Executed Direction)

### Problem Statement
In the original Coconut paper and initial baseline pipelines, multi-step problem decomposition relied heavily on rigid turn schedules or external Python driver scripts that hand-held the model through explicit sub-questions. A truly agentic reasoning model must autonomously chain internal latent thoughts across deduction steps without needing an external DAG scaffold.

### Implementation & Scaling Trajectory
We executed this direction across two architectural scales:

1. **Attempt A (GPT-2 124M Parameter Scale):**
   - Implemented dynamic interleaved latent looping with gradient-descent trained `<|start-latent|>` tokens on GSM8K.
   - **Result:** The 124M model suffered optimization shock and arithmetic drift, collapsing to 1.44% accuracy (vs. 8.72% fine-tuned baseline). However, on pure logical entailment (EntailmentBank), decomposing intermediate conclusions into latent vectors achieved 97% structural tree accuracy compared to 0% for standard one-shot prompting.

2. **Attempt B (Qwen2.5-1.5B Scale - The Final Experiment):**
   - Scaled to `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` in `bfloat16`, benchmarking a grounded two-stage iterative deduction loop against a strictly matched discrete text control across 100 test samples of GSM8K and EntailmentBank.
   - **Empirical Proof:** In an apples-to-apples comparison with 100% identical wording and turn structure, injecting continuous latent thoughts ($K=4$) **doubled mathematical accuracy (12.00% vs. 6.00%)** and produced superior semantic alignment (**0.4010 vs. 0.3924 BLEURT**), proving that latent vectors buffer intermediate reasoning and eliminate compounding discrete text drift.

Full empirical results, tables, and failure case analyses are documented in [`docs/report/entailmentbank_report.md`](file:///home/eran.b/neuroAI/docs/report/entailmentbank_report.md).

---

## Idea 3: Latent Representation Steering & AI Safety Guardrails

### Problem Statement
A major safety and interpretability limitation of continuous latent reasoning is its opacity: intermediate thoughts occur as high-dimensional hidden state vectors without natural language transparency. This raises critical questions for mechanistic interpretability and AI safety:
1. **Adversarial Vulnerability:** Can adversarial prompt perturbations covertly steer intermediate latent representations to produce biased, hallucinatory, or harmful outputs without detectable textual traces during reasoning?
2. **Defensive Safety Steering:** Can we steer continuous latent states using activation addition (steering vectors) to suppress hallucination or bias before the model projects back into discrete natural language?

### Proposed Mechanism
Extract steering vectors $\mathbf{v}_{\text{steer}} \in \mathbb{R}^d$ corresponding to target concepts (e.g., truthfulness, factual neutrality, or safety compliance) using contrastive activation differences across latent thought positions. During inference, modify the hidden states of `<c_thought>` tokens at layer $l$:
$$\mathbf{h}_l' = \mathbf{h}_l + \alpha \mathbf{v}_{\text{steer}}$$

Where $\alpha \in \mathbb{R}$ controls the steering magnitude.

### Evaluation Protocol
Evaluate whether activation steering in continuous latent space can reliably suppress factual hallucinations on EntailmentBank without degrading mathematical problem-solving performance on GSM8K.

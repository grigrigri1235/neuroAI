# Coconut Less Hand-Holding Experiments Blueprint

## Part 1: Qwen2.5-1.5B Pivot (Current Plan)

### Motivation
The Qwen2.5-7B model (14GB bfloat16) exceeded the 8GB RTX 2080 VRAM, causing slow CPU offloading (25h runtime). 
A Gemma-3-4B pivot was considered, but it requires 8GB for weights alone, leaving no room for generation context.
We are pivoting to a fine-tuned `Qwen2.5-1.5B-Instruct` model (~3GB bfloat16). This fits perfectly in the 8GB VRAM, leaving 5GB for generation. Additionally, we can perfectly reuse our existing Qwen code skeleton.

Framework versions
Transformers 4.52.4
Pytorch 2.6.0+cu124
Datasets 3.6.0
Tokenizers 0.21.1

### Implementation Skeleton
**We have a strong skeleton from the previous Qwen2.5-7B model experiments code. We will only need to modify it slightly to accommodate the new 1.5B model.**

We will modify the scripts to point to the new 1.5B model repository.
1. **`eval_qwen_1.5b_baselines.py`**:
   - Model: `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` in bfloat16.
   - Same logic as original baselines (EntailmentBank only).

2. **`eval_qwen_1.5b_gsm8k_less_handholding.py`**:
   - Model: `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` in bfloat16.
   - Uses the `<c_thought>` latent tokens (assuming it remains unchanged from 7B).
   - Loop logic remains the same (max 2 stages).
   - Scrub `<c_thought>` tokens from the output text.
   - Include 3-shot system prompts (to fix the zero-shot issue found in code review).

3. **`eval_qwen_1.5b_entailment_fewshot.py`**:
   - Same as above, but for the EntailmentBank dataset.
   - Include 3-shot system prompts.

4. **`eval_qwen_1.5b_all.slurm`**:
   - Create a SLURM script to run the three 1.5B Python scripts and `score_outputs.py`.

### Execution
- Ensure old 7B SLURM jobs are cancelled.
- Run `sbatch eval_qwen_1.5b_all.slurm`.
- Wait for results and calculate metrics.

---

## Part 2: Qwen2.5-7B Coconut Experiment Blueprint (Previous Plan)

This document is the definitive implementation guide for evaluating `agurung/coconut-qwen2.5-7b` on EntailmentBank and GSM8K. Read all PyTorch instructions carefully.

### 1. Model & PyTorch Shenanigans
Because this is a 7B parameter model, we must handle it carefully to avoid Out-Of-Memory (OOM) errors on a single SLURM GPU. Furthermore, unlike the custom `coconut.py` GPT-2 script, this model works natively with HuggingFace.

**Implementation Requirements:**
- **Loading:** You MUST load the model in `bfloat16`. 
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      "agurung/coconut-qwen2.5-7b", 
      torch_dtype=torch.bfloat16, 
      device_map="auto"
  )
  ```
- **Latent Tokens:** The model was trained using the string `<c_thought>`. There are exactly 2 tokens per reasoning stage, and a maximum of 2 stages. Total K = 4.
- **Tokenizer Check:** Before generating, verify if `<c_thought>` is natively in the tokenizer vocabulary (`tokenizer.convert_tokens_to_ids("<c_thought>")`). If it is `None` or `[unk]`, you must add it and resize the model embeddings, though it should natively be in the tokenizer for this specific model.
- **Inference:** Because the latent tokens are just special tokens with learned embeddings, you do NOT need a custom forward pass. You simply include `<c_thought>` in the prompt string and call `model.generate()`. 

### 2. Baselines (Part 1)
We must measure two zero-shot baselines WITHOUT latent tokens to establish the model's raw semantic capacity.
- **Path:** `/home/eran.b/neuroAI/coconut/eval_qwen_baselines.py`
- **Baseline A (Original Dataset):** Feed the raw original question (no decomposition). See if it can solve it out-of-the-box.
- **Baseline B (All-At-Once):** Feed the modified/decomposed dataset (e.g. `sent1: ... sent2: ...`), but provide ALL the facts at once in a single prompt and ask for the final answer.

### 3. The "Less Hand-Holding" Strategy (Part 2)
Here we test the model's ability to interleave text and latent thoughts (max 2 stages). 

**The Iterative Loop Structure:**
Instead of cheating with Python, we loop the generation.
For each reasoning stage (up to 2 maximum):
1. **Prompt Structure:** 
   ```text
   {Context}
   Question: What can we deduce in one logical step?
   <c_thought><c_thought>
   Answer: 
   ```
2. **Generation:** Call `model.generate()` to let it output the textual deduction. 
3. **Scrubbing:** Strip any hallucinated `<c_thought>` tokens from the generated text.
4. **Append:** Append the cleaned text answer to the context, and repeat the loop for stage 2.

*Note: Since the maximum stages is 2, the total number of `<c_thought>` tokens injected across the entire loop will be exactly 4.*

#### Front A: GSM8K (Few-Shot Iterative)
- **Path:** `/home/eran.b/neuroAI/coconut/eval_qwen_gsm8k_less_handholding.py`
- **System Prompt:** 3-shot prompt demonstrating the loop.
- **Dataset:** `/home/eran.b/neuroAI/coconut/data/gsm8k_test.json`

#### Front B: EntailmentBank (Few-Shot Iterative)
- **Path:** `/home/eran.b/neuroAI/coconut/eval_qwen_entailment_fewshot.py`
- **System Prompt:** 3-shot prompt demonstrating textual deductions.
- **Dataset:** `/home/eran.b/neuroAI/coconut/data/entailmentbank/test_decomposed.json`

### 4. Standard Coconut Inference (Optional but Recommended)
Test the pure Coconut promise: forcing the model to do all 4 latent thoughts at once without any intermediate text output!
**Prompt Structure:**
```text
{Context}
Question: Solve the problem.
<c_thought><c_thought><c_thought><c_thought>
Answer: 
```

### 5. Execution
1. Implement the python scripts defined in Sections 2 and 3.
2. Create `/home/eran.b/neuroAI/coconut/eval_qwen_all.slurm` to run all the scripts sequentially.
3. The USER will run `sbatch eval_qwen_all.slurm`.
4. Run `score_outputs.py` to calculate exact-match/BLEURT.

---

## PART 6: ADDING WHAT IS MISSING

### The Missing GSM8K Baselines
We previously ran the baselines for EntailmentBank and the `<c_thought>` version of GSM8K, but we need the baselines for GSM8K.

**Implementation Plan:**
1. **`eval_qwen_1.5b_gsm8k_baselines.py`**:
   - **Model:** `CoconutEmb/SFT-Qwen2.5-1.5B-Instruct-TongSearch` in bfloat16.
   - **Baseline A (Raw Question):** Run the raw GSM8K question with no modifications.
   - **Baseline B (All-At-Once):** Run the question but prompt the model to generate the textual reasoning steps all at once, without latent `<c_thought>` tokens.
   
2. **`score_baselines.py`**:
   - Create a separate scoring script just for baselines, so we don't mess with `score_outputs.py`.
   - It will load and score both EntailmentBank and GSM8K baseline outputs (`predictions_qwen_1.5b_baseline_a.json`, `predictions_gsm8k_1.5b_baseline_a.json`, etc.).
   
3. **Execution (`eval_gsm8k_baselines.slurm`)**:
   - Create a dedicated SLURM script.
   - Run `sbatch eval_gsm8k_baselines.slurm` to execute the GSM8K baselines.
   - After generation, run `score_baselines.py` to get the baseline scores.

---

## PART 7: PROMPT REVISION & GROUNDED MULTI-STAGE WORKFLOW

### Proposed New GSM8K Few-Shot Prompt (`FEW_SHOT_PROMPT`)

To eliminate the "Tautology Collapse" and exemplar bias observed in Failed Attempt 4, we replace the isolated arithmetic statements with 3 grounded, 2-stage reasoning chains taken directly from the GSM8K train set:

```text
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

### Exact Execution Loop with the Revised Exemplars

1. **Base Context Preparation:**
   ```text
   {FEW_SHOT_PROMPT}

   Question: {test_question}
   ```

2. **Stage 1 (Prompt for Intermediate Step):**
   ```text
   {FEW_SHOT_PROMPT}

   Question: {test_question}

   Question: What can we deduce in one logical step?
   <c_thought><c_thought>
   Answer: 
   ```
   *Expected Output:* The model mirrors the exemplars and produces the intermediate deduction (e.g., rate per minute, pages read so far, total rooms).

3. **Stage 2 (Prompt for Final Deduction & Solution):**
   ```text
   {FEW_SHOT_PROMPT}

   Question: {test_question}

   Question: What can we deduce in one logical step?
   Answer: {stage_1_deduction}

   Question: What can we deduce in one logical step?
   <c_thought><c_thought>
   Answer: 
   ```
   *Expected Output:* The model mirrors the second turn of the exemplars and finishes the solution, producing the final numerical answer.


---

### 2. EntailmentBank Proposed Grounded Multi-Stage Prompt (`FEW_SHOT_PROMPT`)

Using 3 representative 2-stage problems directly from `train_decomposed.json`, formatted with clear spacing:

```text
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

#### Loop Execution:
- **Base Context:**
  ```text
  {FEW_SHOT_PROMPT}

  Context: {context_string}

  Question: {question}
  ```
- **Stage 1 Prompt:**
  ```text
  {FEW_SHOT_PROMPT}

  Context: {context_string}

  Question: {question}

  Question: What can we deduce in one logical step?
  <c_thought><c_thought>
  Answer: 
  ```
- **Stage 2 Prompt (after appending Stage 1 answer):**
  ```text
  {FEW_SHOT_PROMPT}

  Context: {context_string}

  Question: {question}

  Question: What can we deduce in one logical step?
  Answer: {stage_1_deduction}

  Question: What can we deduce in one logical step?
  <c_thought><c_thought>
  Answer: 
  ```

---

### 3. GSM8K Baselines (`eval_qwen_1.5b_gsm8k_baselines.py`)
- **Baseline A (Raw Question):**
  ```text
  Question: {question}
  Answer:
  ```
- **Baseline B (All-at-Once CoT without latents):**
  ```text
  Question: {question}
  Let's think step by step. 
  ```

---

### 4. EntailmentBank Baselines (`eval_qwen_1.5b_baselines.py`)
- **Baseline A (Raw Question):**
  ```text
  Question: {question}
  Answer:
  ```
- **Baseline B (All-at-Once Context without latents):**
  ```text
  Context: {context_string}
  Question: {question}
  Answer:
  ```

---

## PART 8: EXECUTION PLAN FOR REVISED ITERATIVE EXPERIMENTS & CONTROLS

### Overview of What Needs to Run
We already have the basic un-decomposed baselines (Raw Question A and All-at-Once B). To complete the scientific investigation, we need to compare two matched conditions for the step-by-step decomposition:
1. **Modified Intermediate Latent Thinking (With `<c_thought>`):**
   - Iterative 2-stage reasoning with continuous latent tokens using the new grounded 3-shot prompts.
2. **Modified Baseline / Control (Without `<c_thought>`):**
   - The exact same iterative 2-stage reasoning with the same grounded 3-shot prompts, but generating text directly without latent tokens. This isolates the true effect of continuous latents versus step-by-step textual chaining.

### Sequential Implementation Steps (Micro-Steps)
- **Part 1:** Update `eval_qwen_1.5b_gsm8k_less_handholding.py` with the new grounded GSM8K prompt and clean double-newline formatting.
- **Part 2:** Update `eval_qwen_1.5b_entailment_fewshot.py` with the new grounded EntailmentBank prompt and clean double-newline formatting.
- **Part 3:** Create `eval_qwen_1.5b_gsm8k_no_latents.py` (same loop, no `<c_thought>` tokens, saving to `predictions_gsm8k_1.5b_no_latents.json`).
- **Part 4:** Create `eval_qwen_1.5b_entailment_no_latents.py` (same loop, no `<c_thought>` tokens, saving to `predictions_entailment_1.5b_no_latents.json`).
- **Part 5:** Update `score_outputs.py` to score all 4 output files (Latent vs. No-Latents for both datasets).
- **Part 6:** Create `eval_qwen_1.5b_iterative_all.slurm`:
  - Request 1 GPU (`--gres=gpu:1`) and 8 CPU threads (`--cpus-per-task=8`).
  - Run the 4 evaluation scripts sequentially.
  - Run `score_outputs.py` on CPU with multithreading so scoring completes in ~1 minute.



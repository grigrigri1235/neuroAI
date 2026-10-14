"""Prompts for Part 2 (S1-S5), copied from the Phase 4 Qwen scripts.

FEW_SHOT_PROMPT is verbatim from coconut/eval_qwen_1.5b_gsm8k_less_handholding.py.
Changes vs Phase 4 (KNOWLEDGE_RETENTION.md section 5.3):
- the latent block is the real Coconut tokens, 2 latents per stage (GSM8K c_thought = 2),
- stage 2 keeps the stage-1 question line (fixes Phase 4 bug 4.6).
"""

FEW_SHOT_PROMPT = """--- Example 1 ---
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
Answer: The last floor of 10 rooms is unavailable, so there are 100 - 10 = 90 available rooms. So Hans could be checked into 90 different rooms."""

STEP_Q = "Question: What can we deduce in one logical step?"
LAT = "<|start-latent|><|latent|><|latent|><|end-latent|>"
STOP_STRINGS = ("\nQuestion:", "--- Example")  # plus EOS (KR section 5.4)

# loop: forced 2-stage loop (else one pass); lat: latent block in the prompt;
# mode: how latent slots are filled ("latent" = previous hidden state, "plain" = token embedding)
SETTINGS = {
    "S1": {"loop": False, "lat": False, "mode": "plain"},  # raw question
    "S2": {"loop": False, "lat": False, "mode": "plain"},  # "Let's think step by step."
    "S3": {"loop": True, "lat": True, "mode": "latent"},  # real continuous thoughts
    "S4": {"loop": True, "lat": False, "mode": "plain"},  # no latent tokens
    "S5": {"loop": True, "lat": True, "mode": "plain"},  # same tokens, no feedback (pause control)
}


def few_shot(n_shots=3):
    """The first n_shots examples of FEW_SHOT_PROMPT (n_shots=3 gives it unchanged)."""
    sep = "\n\n--- Example "
    examples = FEW_SHOT_PROMPT.split(sep)
    assert 1 <= n_shots <= len(examples), n_shots
    return sep.join(examples[:n_shots])


def single_prompt(setting, question):
    """S1 / S2: one pass, no few-shot examples (same strings as Phase 4)."""
    if setting == "S1":
        return f"Question: {question}\nAnswer:"
    if setting == "S2":
        return f"Question: {question}\nLet's think step by step. "
    raise ValueError(f"{setting} is not a one-pass setting")


def turn(lat):
    """One step turn: the step question, the latent block (S3/S5 only), then 'Answer: '."""
    return f"\n\n{STEP_Q}\n" + (f"{LAT}\n" if lat else "") + "Answer: "


def stage1_prompt(question, lat, n_shots=3):
    return f"{few_shot(n_shots)}\n\nQuestion: {question}" + turn(lat)


def stage2_prompt(stage1, stage1_answer, lat):
    """The whole stage-1 prompt + its (cleaned) answer + a second turn."""
    return stage1 + stage1_answer + turn(lat)

"""Part 2: the Phase 4 few-shot experiment on GPT-2 Coconut, one setting per run (S1-S5).

S3/S4/S5: forced 2-stage loop with the generic step question. S1/S2: one pass.
All settings use the same greedy loop (common.greedy_generate); they differ only in the
prompt text and in how the latent slots are filled (prompts.SETTINGS).
Data: data/gsm8k_test.json, the same file and question text as Phase 4.

    python experiments/run_fewshot.py --setting S3              # full test set
    python experiments/run_fewshot.py --setting S3 --limit 20   # smoke test
"""

import argparse
import json
import os
import time

from tqdm import tqdm

import common
import prompts

# re-tokenizing the stage-1 answer inside the stage-2 prompt can shift a few tokens
BOUNDARY_MARGIN = 8


def worst_stage2_len(tokenizer, questions, n_shots, max_new_tokens):
    """Longest possible stage-2 prompt in tokens: the latent template (the longest one)
    and a stage-1 answer that used all max_new_tokens tokens."""
    turn_len = len(tokenizer.encode(prompts.turn(lat=True)))
    stage1 = max(len(tokenizer.encode(prompts.stage1_prompt(q, True, n_shots))) for q in questions)
    return stage1 + max_new_tokens + turn_len + BOUNDARY_MARGIN


def choose_n_shots(tokenizer, questions, max_new_tokens):
    """Length guard (KR 5.4): 3 few-shot examples if every stage-2 prompt + max_new_tokens
    fits in GPT-2's 1024 tokens, else 2. Computed on all questions, so it is the same
    choice for S3, S4, S5 and for smoke and full runs."""
    for n_shots in (3, 2):
        worst = worst_stage2_len(tokenizer, questions, n_shots, max_new_tokens)
        if worst + max_new_tokens <= common.MAX_POSITIONS:
            return n_shots, worst
    raise RuntimeError("stage-2 prompts are too long even with 2 few-shot examples")


def run_stage(model, tokenizer, prompt, mode, expected_latents, max_new_tokens, use_cache):
    device = model.embedding.weight.device
    input_ids = common.encode(tokenizer, prompt, device)
    n_latents = len(common.latent_positions(model, input_ids))
    if n_latents != expected_latents:  # catches marker tokens leaking into a prompt (plan F3)
        raise RuntimeError(f"expected {expected_latents} latents in the prompt, found {n_latents}")
    embeds = common.get_embeds(model, input_ids, mode)
    gen = common.greedy_generate(
        model, tokenizer, embeds, max_new_tokens, prompts.STOP_STRINGS, use_cache
    )
    text, n_markers = common.clean_text(gen["cut_text"])
    return {
        "prompt_len": input_ids.shape[1],
        "raw": gen["raw_text"],
        "cut": gen["cut_text"],
        "text": text,
        "stop_reason": gen["stop_reason"],
        "n_new_tokens": gen["n_new_tokens"],
        "markers_removed": n_markers,
    }


def run_example(model, tokenizer, setting, question, n_shots, max_new_tokens, use_cache):
    """Returns the list of stages (1 for S1/S2, 2 for S3/S4/S5)."""
    cfg = prompts.SETTINGS[setting]
    if not cfg["loop"]:
        prompt = prompts.single_prompt(setting, question)
        return [run_stage(model, tokenizer, prompt, cfg["mode"], 0, max_new_tokens, use_cache)]
    n_lat = 2 if cfg["lat"] else 0  # 2 latents per stage; stage 2 also holds the stage-1 block
    p1 = prompts.stage1_prompt(question, cfg["lat"], n_shots)
    s1 = run_stage(model, tokenizer, p1, cfg["mode"], n_lat, max_new_tokens, use_cache)
    p2 = prompts.stage2_prompt(p1, s1["text"], cfg["lat"])
    s2 = run_stage(model, tokenizer, p2, cfg["mode"], 2 * n_lat, max_new_tokens, use_cache)
    return [s1, s2]


def main():
    parser = argparse.ArgumentParser(description="Part 2: few-shot run for one setting.")
    parser.add_argument("--setting", required=True, choices=sorted(prompts.SETTINGS))
    parser.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    parser.add_argument("--n-shots", default="auto", choices=["auto", "2", "3"])
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="per stage; default 128 for S2 (same budget as 2 x 64), else 64")
    parser.add_argument("--no-cache", action="store_true",
                        help="greedy loop without KV cache (slower, like Coconut.generate)")
    parser.add_argument("--data", default="data/gsm8k_test.json")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None, help="default: results/gpt2/fewshot_<setting>.json")
    args = parser.parse_args()

    setting, cfg = args.setting, prompts.SETTINGS[args.setting]
    max_new_tokens = args.max_new_tokens or (128 if setting == "S2" else 64)
    out = common.path(args.out or f"results/gpt2/fewshot_{setting}.json")

    common.set_seed(0)
    tokenizer, ids = common.load_tokenizer()
    with open(common.path(args.data)) as f:
        data = json.load(f)
    questions = [d["question"] for d in data]

    n_shots, worst_len = None, None
    if cfg["loop"]:
        if args.n_shots == "auto":
            n_shots, worst_len = choose_n_shots(tokenizer, questions, max_new_tokens)
        else:
            n_shots = int(args.n_shots)
            worst_len = worst_stage2_len(tokenizer, questions, n_shots, max_new_tokens)
        print(f"few-shot examples: {n_shots}, longest possible stage-2 prompt: {worst_len} tokens",
              flush=True)

    device = common.get_device()
    model, _ = common.load_model(tokenizer, ids, device, args.ckpt)

    records = []
    for idx, example in enumerate(tqdm(data[: args.limit], desc=setting, mininterval=30)):
        gold = common.gsm8k_gold(example["answer"])
        common.sync(device)
        t0 = time.perf_counter()
        stages = run_example(
            model, tokenizer, setting, example["question"], n_shots, max_new_tokens, not args.no_cache
        )
        common.sync(device)
        seconds = time.perf_counter() - t0
        pred = common.last_number(stages[-1]["text"])
        records.append({
            "idx": idx,
            "question": example["question"],
            "gold": gold,
            "stages": stages,
            "pred": pred,
            "correct": common.same_number(pred, gold),
            "seconds": seconds,
        })
        if idx < 3:
            print(f"--- example {idx}: gold {gold}, pred {pred}")
            for i, s in enumerate(stages, 1):
                print(f"  stage {i} [{s['stop_reason']}, {s['n_new_tokens']} tokens, "
                      f"prompt {s['prompt_len']}]: {s['text']!r}", flush=True)

    acc = sum(r["correct"] for r in records) / len(records)
    print(f"{setting}: accuracy {acc:.4f} on {len(records)} questions", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = common.run_meta(
        part="fewshot", setting=setting, n_shots=n_shots, max_new_tokens=max_new_tokens,
        worst_stage2_len=worst_len, stop_strings=list(prompts.STOP_STRINGS),
        use_cache=not args.no_cache, data=args.data, limit=args.limit, seed=0,
    )
    with open(out, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()

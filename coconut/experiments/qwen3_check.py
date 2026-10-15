"""Gate check for agurung/coconut-qwen3-4b-ff (plan: docs/planning/qwen3_4b_interleaved_plan.md).

G0 load: token ids, embedding rows, weights, RoPE (common.load_coconut_hf raises otherwise).
G1 loop: our greedy loop == HF greedy generate on prompts without latents.
G2 latents work: native format, N latents, feedback vs plain, accuracy on the first questions.
G3 format: first questions x S1-S5 with the few-shot loop (does it write real steps?).

    python experiments/qwen3_check.py --out results/qwen3/check/qwen3_check.json
"""

import argparse
import json
import os
import sys
import time

import torch

import common
import prompts
from qwen_check import preview


def check_loop(model, tokenizer, questions, max_new_tokens, device):
    """G1: same tokens as HF greedy generate, on S1 and S4 stage-1 prompts (no latents)."""
    base = model.base_causallm
    eos = sorted(model.eos_ids)
    rows, matches = [], 0
    texts = [prompts.single_prompt("S1", q) for q in questions]
    texts += [prompts.stage1_prompt(q, False, 3) for q in questions]
    for text in texts:
        x = common.encode(tokenizer, text, device)
        hf = base.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                           do_sample=False, eos_token_id=eos, pad_token_id=tokenizer.eos_token_id)
        hf = hf[0, x.shape[1]:].tolist()
        hf = hf[: min([hf.index(e) for e in eos if e in hf], default=len(hf))]
        ours = common.greedy_generate(model, tokenizer, common.get_embeds(model, x, "plain"), max_new_tokens)
        same = ours["new_ids"] == hf
        matches += same
        rows.append({"prompt_end": text[-60:], "same": same, "hf": tokenizer.decode(hf), "ours": ours["raw_text"]})
        print(f"--- G1 {'same' if same else 'DIFFERENT'}\n  hf:   {tokenizer.decode(hf)!r}\n  ours: {ours['raw_text']!r}",
              flush=True)
    return {"matches": matches, "n": len(texts), "pass": matches >= len(texts) - 1, "rows": rows}


def latent_sweep(model, tokenizer, data, ns, max_new_tokens, device):
    """G2: question + "\\n" + <|start-latent|> + N x <|latent|> + <|end-latent|>; feedback vs plain."""
    out = {}
    for n in ns:
        block = common.START + common.LATENT * n + common.END
        for mode in (("latent", "plain") if n else ("latent",)):
            correct = []
            for i, ex in enumerate(data):
                x = common.encode(tokenizer, ex["question"] + "\n" + block, device)
                gen = common.greedy_generate(model, tokenizer, common.get_embeds(model, x, mode), max_new_tokens)
                text, _ = common.clean_text(gen["raw_text"])
                pred = common.last_number(text)
                correct.append(common.same_number(pred, common.gsm8k_gold(ex["answer"])))
                if i < 2:
                    print(f"--- G2 N={n} {mode} q{i} gold {common.gsm8k_gold(ex['answer'])} pred {pred}\n  {text!r}",
                          flush=True)
            out[f"N={n} {mode}"] = {"correct": sum(correct), "n": len(correct), "per_question": correct}
            print(f"G2 N={n} {mode}: {sum(correct)}/{len(correct)}", flush=True)
        if n:
            fed, plain = out[f"N={n} latent"]["per_question"], out[f"N={n} plain"]["per_question"]
            out[f"N={n} discordant"] = {"feedback_only": sum(a and not b for a, b in zip(fed, plain)),
                                       "plain_only": sum(b and not a for a, b in zip(fed, plain))}
    return out


def main():
    parser = argparse.ArgumentParser(description="Gate check for the Qwen3-4B Coconut export.")
    parser.add_argument("--n-loop", type=int, default=5)
    parser.add_argument("--n-native", type=int, default=50)
    parser.add_argument("--native-latents", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--native-tokens", type=int, default=256)
    parser.add_argument("--n-preview", type=int, default=10)
    parser.add_argument("--per-stage", type=int, default=128)
    parser.add_argument("--out", default="results/qwen3/check/qwen3_check.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    model, tokenizer, ids, info = common.load_coconut_hf(common.QWEN3_MODEL_ID, common.QWEN3_BASE_ID, device)
    print("G0 load: PASS", flush=True)
    with open(common.path("data/gsm8k_test.json")) as f:
        data = json.load(f)
    report = {"meta": common.run_meta(part="qwen3_check", **vars(args)), "G0": {"pass": True, **info}}

    with torch.no_grad():
        t0 = time.perf_counter()
        report["G1"] = check_loop(model, tokenizer, [d["question"] for d in data[: args.n_loop]], 64, device)
        print(f"G1 loop: {'PASS' if report['G1']['pass'] else 'FAIL'} "
              f"({report['G1']['matches']}/{report['G1']['n']} same)", flush=True)
        report["G2"] = latent_sweep(model, tokenizer, data[: args.n_native], args.native_latents,
                                    args.native_tokens, device)
        report["G3"] = preview(model, tokenizer, data[: args.n_preview], args.per_stage)
        report["seconds"] = time.perf_counter() - t0

    print("=== summary ===")
    print(f"G1 loop: {report['G1']['matches']}/{report['G1']['n']} same as HF generate")
    for key, v in report["G2"].items():
        print(f"G2 {key}: {v}" if "discordant" in key else f"G2 {key}: {v['correct']}/{v['n']}")
    for s in sorted(prompts.SETTINGS):
        p = report["G3"][s]
        print(f"G3 {s}: {p['correct']}/{p['n']} correct, {p['tokens_per_second']:.1f} tok/s")
    print(f"G3 S3 vs S5 outputs differ: {report['G3']['S3_vs_S5_outputs_differ']}/{args.n_preview}")

    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", out)
    sys.exit(0 if report["G1"]["pass"] else 1)


if __name__ == "__main__":
    main()

"""Pivot gate for agurung/coconut-qwen2.5-7b (plan: docs/planning/qwen7b_coconut_pivot.md).

R1: the 3 Coconut tokens get ids 151665-7 = the model's 3 extra rows (common.load_qwen raises
    otherwise), and those rows look like run.py's init (copies of the "<<" row, input and output side).
R2: latents change the output: hidden states (S3 / "latent") vs plain token embeddings (S5 / "plain")
    on the same prompts, in the native Coconut format and in the few-shot loop.
Also prints the first N questions x S1-S5, with timing, to choose tokens per stage and plan the full run.

    python experiments/qwen_check.py --out results/qwen7b/check/qwen_check.json
"""

import argparse
import json
import os
import time

import torch

import common
import prompts
from run_fewshot import run_example

N_NATIVE_LATENTS = 4  # c_thought 2 x max_latent_stage 2 (model card)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))


def init_check(model, tokenizer, ids, n_random=200, seed=0):
    """R1 (part 2): run.py copies the "<<" row into the 3 new rows (input and output embeddings).
    If the rows are Coconut's, they should still be closer to "<<" than random tokens are."""
    ref_id = tokenizer.convert_tokens_to_ids("<<")
    out = {"<<_id": ref_id}
    if ref_id is None:
        out["note"] = "'<<' is not a single token here, check skipped"
        return out
    base = model.base_causallm
    g = torch.Generator().manual_seed(seed)
    random_ids = torch.randint(0, common.QWEN_MARKER_IDS[0], (n_random,), generator=g).tolist()
    names = ("start", "end", "latent")
    for side, w in (("input", base.get_input_embeddings().weight), ("output", base.get_output_embeddings().weight)):
        w = w.detach()
        out[side] = {
            "cos_to_<<": {n: cos(w[ids[n]], w[ref_id]) for n in names},
            "random_tokens_mean_cos_to_<<": sum(cos(w[i], w[ref_id]) for i in random_ids) / n_random,
            "pairwise_cos": {
                "start-end": cos(w[ids["start"]], w[ids["end"]]),
                "start-latent": cos(w[ids["start"]], w[ids["latent"]]),
                "end-latent": cos(w[ids["end"]], w[ids["latent"]]),
            },
            "norm": {n: float(w[ids[n]].float().norm()) for n in names},
            "random_tokens_mean_norm": sum(float(w[i].float().norm()) for i in random_ids) / n_random,
        }
    return out


def native_probe(model, tokenizer, questions, max_new_tokens, device):
    """R2 in the Coconut training format: question + "\\n" + <bot> 4 latents <eot>."""
    block = common.START + common.LATENT * N_NATIVE_LATENTS + common.END
    rows = []
    for q in questions:
        x = common.encode(tokenizer, q + "\n" + block, device)
        assert len(common.latent_positions(model, x)) == N_NATIVE_LATENTS
        texts = {}
        for mode in ("latent", "plain"):
            embeds = common.get_embeds(model, x, mode)
            texts[mode] = common.greedy_generate(model, tokenizer, embeds, max_new_tokens)["raw_text"]
        rows.append({"question": q[:80], **texts, "same": texts["latent"] == texts["plain"]})
        print(f"--- native: {q[:80]!r}\n  latent: {texts['latent']!r}\n  plain:  {texts['plain']!r}", flush=True)
    return rows


def preview(model, tokenizer, data, per_stage):
    """First N questions x S1-S5 with the real few-shot loop (3 examples)."""
    out = {}
    for setting in sorted(prompts.SETTINGS):
        max_new = 2 * per_stage if setting == "S2" else per_stage
        recs = []
        for idx, ex in enumerate(data):
            gold = common.gsm8k_gold(ex["answer"])
            t0 = time.perf_counter()
            stages = run_example(model, tokenizer, setting, ex["question"], 3, max_new, True)
            seconds = time.perf_counter() - t0
            pred = common.last_number(stages[-1]["text"])
            recs.append({
                "idx": idx, "gold": gold, "pred": pred, "correct": common.same_number(pred, gold),
                "seconds": seconds, "new_tokens": sum(s["n_new_tokens"] for s in stages),
                "stages": [{k: s[k] for k in ("text", "stop_reason", "n_new_tokens")} for s in stages],
            })
            print(f"--- {setting} q{idx} gold {gold} pred {pred} ({seconds:.1f}s)", flush=True)
            for i, s in enumerate(stages, 1):
                print(f"  stage {i} [{s['stop_reason']}, {s['n_new_tokens']} tok]: {s['text']!r}", flush=True)
        tokens = sum(r["new_tokens"] for r in recs)
        out[setting] = {
            "correct": sum(r["correct"] for r in recs), "n": len(recs),
            "tokens_per_second": tokens / sum(r["seconds"] for r in recs),
            "mean_seconds": sum(r["seconds"] for r in recs) / len(recs),
            "records": recs,
        }
    s3, s5 = out["S3"]["records"], out["S5"]["records"]
    out["S3_vs_S5_outputs_differ"] = sum(a["stages"] != b["stages"] for a, b in zip(s3, s5))
    return out


def main():
    parser = argparse.ArgumentParser(description="Gate check for the Qwen 7B Coconut checkpoint.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--per-stage", type=int, default=128, help="new tokens per stage (S2 gets 2x)")
    parser.add_argument("--native-tokens", type=int, default=256)
    parser.add_argument("--out", default="results/qwen7b/check/qwen_check.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    t0 = time.perf_counter()
    model, tokenizer, ids, info = common.load_qwen(device)  # raises if R1 (ids) fails
    report = {"meta": common.run_meta(part="qwen_check", n=args.n, per_stage=args.per_stage),
              "load": info, "load_seconds": time.perf_counter() - t0, "R1_ids": "PASS"}
    print("R1 ids: PASS", flush=True)

    with torch.no_grad():
        report["R1_init"] = init_check(model, tokenizer, ids)
        print("R1 init:", json.dumps(report["R1_init"]), flush=True)
        with open(common.path("data/gsm8k_test.json")) as f:
            data = json.load(f)[: args.n]
        native = native_probe(model, tokenizer, [d["question"] for d in data], args.native_tokens, device)
        report["R2_native"] = {"differ": sum(not r["same"] for r in native), "n": len(native), "rows": native}
        report["preview"] = preview(model, tokenizer, data, args.per_stage)

    print(f"R2: native outputs differ (latent vs plain): {report['R2_native']['differ']}/{args.n}")
    print(f"R2: few-shot S3 vs S5 outputs differ: {report['preview']['S3_vs_S5_outputs_differ']}/{args.n}")
    for s in sorted(prompts.SETTINGS):
        p = report["preview"][s]
        print(f"{s}: {p['correct']}/{p['n']} correct, {p['tokens_per_second']:.1f} tok/s, {p['mean_seconds']:.1f}s/question")

    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", out)


if __name__ == "__main__":
    main()

"""Test the original and the two fine-tuned SIM-CoT models: L / T2 / T2-gold on all GSM8K test questions.
Plan: docs/planning/simcot_finetune_plan.md (sections 7, 8, check 5).

  L        question -> 10 latents -> answer                       ("no step")
  T2       question + "\\n" + saved s1 -> 10 latents -> answer     ("own step", s1 from job 1449771)
  T2-gold  question + "\\n" + true step 1 -> 10 latents -> answer  ("true step")
Inputs exactly as in simcot_run.py. Every model gets the same saved s1.

    python experiments/simcot_ft_eval.py --out results/simcot/ft_eval.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import common
from score import accuracy_block, pair_test, paired_bootstrap
from simcot_run import N_LATENTS, run_round

SETTINGS = ("L", "T2", "T2_gold")
FT_MODELS = {"FT_mix": "checkpoints_ft/mix.pt", "FT_plain": "checkpoints_ft/plain.pt"}
ORIGINAL_CORRECT = {"L": 589, "T2": 454, "T2_gold": 509}  # job 1449771


def evaluate(model, tokenizer, data, saved, device, name):
    out = {s: [] for s in SETTINGS}
    with torch.no_grad():
        for i, (d, r) in enumerate(zip(data, saved)):
            g1 = d["steps"][0] if d["steps"] else ""
            inputs = {"L": d["question"], "T2": d["question"] + "\n" + r["s1"], "T2_gold": d["question"] + "\n" + g1}
            for s in SETTINGS:
                out[s].append(run_round(model, tokenizer, inputs[s], N_LATENTS, device)[0])
            if (i + 1) % 200 == 0:
                print(f"  {name}: {i + 1}/{len(data)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Test original / FT-mix / FT-plain with L, T2, T2-gold.")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--saved", default="results/simcot/simcot_run.json", help="job 1449771 results")
    parser.add_argument("--out", default="results/simcot/ft_eval.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    with open(common.path("data/gsm_test.json")) as f:
        data = json.load(f)[: args.limit]
    with open(common.path(args.saved)) as f:
        saved = json.load(f)["records"][: len(data)]
    golds = [d["answer"].replace(",", "").strip() for d in data]
    report = {"meta": common.run_meta(part="simcot_ft_eval", **vars(args))}
    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()

    answers = {}
    model, explainer, tokenizer, ids, info = common.load_simcot(device)
    del explainer
    answers["original"] = evaluate(model, tokenizer, data, saved, device, "original")
    original_correct = {s: sum(a == g for a, g in zip(answers["original"][s], golds)) for s in SETTINGS}
    report["check5_original"] = original_correct
    print(f"check 5 (original = job 1449771 {ORIGINAL_CORRECT}): {original_correct}", flush=True)
    if args.limit is None and original_correct != ORIGINAL_CORRECT:
        with open(out, "w") as f:
            json.dump(report, f, indent=1, default=str)
        sys.exit("FAILED check 5: the original model does not reproduce job 1449771")

    for name, ckpt in FT_MODELS.items():
        del model
        torch.cuda.empty_cache()
        weights = torch.load(common.path(ckpt), map_location="cpu", weights_only=True)
        model, _ = common.load_model(tokenizer, ids, device, ckpt=ckpt, weights=weights)
        answers[name] = evaluate(model, tokenizer, data, saved, device, name)

    correct = {m: {s: [a == g for a, g in zip(answers[m][s], golds)] for s in SETTINGS} for m in answers}

    def run(m, s):
        return {"records": [{"correct": c} for c in correct[m][s]]}

    report["main"] = {m: {s: accuracy_block(correct[m][s]) for s in SETTINGS} for m in answers}
    pairs = {}
    for m in answers:  # inside each model: does a step help?
        pairs[f"{m}: T2-L"] = pair_test(run(m, "T2"), run(m, "L"), "T2", "L")
        pairs[f"{m}: T2_gold-L"] = pair_test(run(m, "T2_gold"), run(m, "L"), "T2_gold", "L")
    for s in SETTINGS:  # between models
        pairs[f"{s}: FT_mix-FT_plain"] = pair_test(run("FT_mix", s), run("FT_plain", s), "FT_mix", "FT_plain")
        pairs[f"{s}: FT_mix-original"] = pair_test(run("FT_mix", s), run("original", s), "FT_mix", "original")
        pairs[f"{s}: FT_plain-original"] = pair_test(run("FT_plain", s), run("original", s), "FT_plain", "original")
    report["pairs"] = pairs

    # Q3: is the step gain (setting - L) larger in FT-mix than in FT-plain? paired bootstrap over questions
    gaps = {}
    for s in ("T2", "T2_gold"):
        d = {m: np.array(correct[m][s], dtype=float) - np.array(correct[m]["L"], dtype=float)
             for m in ("FT_mix", "FT_plain")}
        diff, lo, hi = paired_bootstrap(d["FT_mix"], d["FT_plain"])
        gaps[f"({s} - L): FT_mix - FT_plain"] = {
            "gap_FT_mix": float(d["FT_mix"].mean()), "gap_FT_plain": float(d["FT_plain"].mean()),
            "diff": diff, "ci95": [lo, hi],
            "verdict": "FT_mix larger" if lo > 0 else "FT_mix smaller" if hi < 0 else "no clear difference"}
    report["gaps"] = gaps
    report["seconds"] = time.perf_counter() - t0
    report["records"] = [{"idx": i, "gold": g, **{m: {s: answers[m][s][i] for s in SETTINGS} for m in answers}}
                         for i, g in enumerate(golds)]

    print("=== summary ===")
    for m, blocks in report["main"].items():
        print(m + ": " + " | ".join(f"{s} {b['acc']:.4f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
                                    for s, b in blocks.items()))
    for name, t in pairs.items():
        print(f"{name}: diff {t['diff']:+.4f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  "
              f"McNemar p={t['mcnemar_p']:.3g} -> {t['verdict']}")
    for name, g in gaps.items():
        print(f"{name}: {g['diff']:+.4f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] -> {g['verdict']}")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", out)


if __name__ == "__main__":
    main()

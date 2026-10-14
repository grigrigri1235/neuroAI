"""Scores the runs in one results folder and writes results.json (KR section 5.5).

Part 1 (native.json): accuracy per k with the official run.py rule (decided in run_native.py).
Part 2 (fewshot_S*.json): accuracy per setting with a 95% Wilson CI, and the paired tests
S3 vs S5 (H1) and S3 vs S4 (H2): exact McNemar test + paired bootstrap 95% CI of the
accuracy difference. A difference counts only if its CI excludes 0.

    python experiments/score.py --dir results/gpt2
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.stats import binomtest

import common
import prompts

PAIRS = [("S3", "S5"), ("S3", "S4")]  # H1: feedback vs pause tokens, H2: latents vs no tokens
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% Wilson score interval for k correct out of n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [center - half, center + half]


def accuracy_block(correct):
    n, k = len(correct), int(sum(correct))
    return {"n": n, "correct": k, "acc": k / n if n else float("nan"), "ci95": wilson(k, n)}


def mcnemar_exact(a, b):
    """a, b: 0/1 arrays on the same questions. Exact McNemar = two-sided binomial test
    on the discordant pairs."""
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, float(p)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Accuracy difference a - b and its 95% CI, resampling questions (pairs kept)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot // 1000):  # chunks of 1000 resamples to keep memory small
        idx = rng.integers(0, n, size=(1000, n))
        diffs.append(a[idx].mean(axis=1) - b[idx].mean(axis=1))
    diffs = np.concatenate(diffs)
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def native_summary(records):
    summary = {}
    for k in sorted({r["k"] for r in records}):
        rs = [r for r in records if r["k"] == k]
        block = accuracy_block([r["correct"] for r in rs])
        block.update(
            n_latents=rs[0]["n_latents"],
            mean_new_tokens=float(np.mean([r["n_new_tokens"] for r in rs])),
            mean_seconds=float(np.mean([r["seconds"] for r in rs])),
        )
        summary[str(k)] = block
    return summary


def fewshot_summary(run):
    rs = run["records"]
    n_stages = len(rs[0]["stages"])
    block = accuracy_block([r["correct"] for r in rs])
    block.update(
        n_shots=run["meta"]["n_shots"],
        max_new_tokens=run["meta"]["max_new_tokens"],
        no_number_rate=float(np.mean([r["pred"] is None for r in rs])),
        mean_new_tokens_per_stage=[
            float(np.mean([r["stages"][i]["n_new_tokens"] for r in rs])) for i in range(n_stages)
        ],
        stop_reasons_per_stage=[
            dict(Counter(r["stages"][i]["stop_reason"] for r in rs)) for i in range(n_stages)
        ],
        markers_removed=int(sum(s["markers_removed"] for r in rs for s in r["stages"])),
        mean_seconds=float(np.mean([r["seconds"] for r in rs])),
    )
    return block


def pair_test(run_a, run_b, name_a, name_b):
    a = np.array([r["correct"] for r in run_a["records"]], dtype=float)
    b = np.array([r["correct"] for r in run_b["records"]], dtype=float)
    a_only, b_only, p = mcnemar_exact(a, b)
    diff, lo, hi = paired_bootstrap(a, b)
    if lo > 0:
        verdict = f"{name_a} higher"
    elif hi < 0:
        verdict = f"{name_a} lower"
    else:
        verdict = "no clear difference"
    return {
        "diff": diff, "ci95": [lo, hi], "mcnemar_p": p,
        f"{name_a}_only_correct": a_only, f"{name_b}_only_correct": b_only, "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Score native + few-shot runs.")
    parser.add_argument("--dir", default="results/gpt2")
    args = parser.parse_args()
    folder = common.path(args.dir)
    results = {"dir": args.dir}

    native_file = os.path.join(folder, "native.json")
    if os.path.exists(native_file):
        with open(native_file) as f:
            native_records = json.load(f)["records"]
        results["native"] = native_summary(native_records)
        # pivot plan D2: same native prompts, latent slots without hidden-state feedback
        plain_file = os.path.join(folder, "native_plain.json")
        if os.path.exists(plain_file):
            with open(plain_file) as f:
                plain_records = json.load(f)["records"]
            results["native_plain"] = native_summary(plain_records)
            results["native_pairs"] = {}
            for k in sorted({r["k"] for r in plain_records}):
                fed = sorted((r for r in native_records if r["k"] == k), key=lambda r: r["idx"])
                plain = sorted((r for r in plain_records if r["k"] == k), key=lambda r: r["idx"])
                if [r["idx"] for r in fed] != [r["idx"] for r in plain]:
                    raise ValueError(f"k={k}: native and native_plain are not on the same questions")
                results["native_pairs"][f"k={k}"] = pair_test(
                    {"records": fed}, {"records": plain}, "feedback", "plain"
                )

    runs = {}
    for setting in sorted(prompts.SETTINGS):
        run_file = os.path.join(folder, f"fewshot_{setting}.json")
        if os.path.exists(run_file):
            with open(run_file) as f:
                runs[setting] = json.load(f)
    if runs:
        question_ids = {s: [r["idx"] for r in run["records"]] for s, run in runs.items()}
        first = next(iter(question_ids.values()))
        if any(q != first for q in question_ids.values()):
            raise ValueError("the few-shot runs are not on the same questions, cannot pair them")
        loop_shots = {runs[s]["meta"]["n_shots"] for s in runs if prompts.SETTINGS[s]["loop"]}
        if len(loop_shots) > 1:
            raise ValueError(f"S3/S4/S5 used different numbers of few-shot examples: {loop_shots}")
        results["fewshot"] = {s: fewshot_summary(run) for s, run in runs.items()}
        results["pairs"] = {
            f"{a}-{b}": pair_test(runs[a], runs[b], a, b) for a, b in PAIRS if a in runs and b in runs
        }

    out = os.path.join(folder, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    if "native" in results:
        print("Part 1 (native format, official scoring)")
        for k, s in results["native"].items():
            print(f"  k={k} latents={s['n_latents']}: {s['acc']:.4f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
                  f"  n={s['n']}  new tokens {s['mean_new_tokens']:.1f}  {s['mean_seconds']:.3f}s/question")
    if "native_pairs" in results:
        print("Part 1 control: hidden-state feedback vs plain latent embeddings (same prompts)")
        for k, t in results["native_pairs"].items():
            plain = results["native_plain"][k.split("=")[1]]
            print(f"  {k}: plain {plain['acc']:.4f}  diff (feedback - plain) {t['diff']:+.4f}"
                  f" [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  McNemar p={t['mcnemar_p']:.3g}  -> {t['verdict']}")
    if runs:
        print("Part 2 (few-shot)")
        for s, b in results["fewshot"].items():
            print(f"  {s}: {b['acc']:.4f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]  n={b['n']}"
                  f"  no number {b['no_number_rate']:.3f}  stop {b['stop_reasons_per_stage']}")
        for name, t in results["pairs"].items():
            print(f"  {name}: diff {t['diff']:+.4f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
                  f"  McNemar p={t['mcnemar_p']:.3g}  -> {t['verdict']}")
    print("saved", out)


if __name__ == "__main__":
    main()

"""SIM-CoT GPT-2: gate checks + the minimal 2-round interleaving test.
Plan: docs/planning/simcot_interleaved_plan.md (section 2c and the gate in section 3).

G0  the checkpoint has the explainer (else stop, exit code 1)
G1  native sweep: question -> N latents -> ### answer, N = 0..10, all questions (paper reproduction)
G2  decode the 5 thought pairs of the first questions with the explainer, next to the true steps
Main (all questions, paired):
  L        question -> 10 latents -> answer
  T2       round 1 as L; explainer(thought pair 1) -> s1; round 2: question + s1 -> 10 latents -> answer
  T2-gold  as T2 but s1 = the true step 1
The main numbers count only if G2 shows readable steps.

    python experiments/simcot_run.py --out results/simcot/simcot_run.json
"""

import argparse
import json
import os
import sys
import time

import torch

import common
from score import accuracy_block, pair_test

NATIVE_NS = [0, 2, 4, 6, 8, 10]  # k = 0..5 stages x c_thought 2
N_LATENTS = 10  # trained budget: max_latent_stage 5 x c_thought 2
C_THOUGHT = 2


def run_round(model, tokenizer, text, n_latents, device, max_new_tokens=64):
    """text + "\\n" + <bot> n latents <eot> -> the model writes '### answer'.
    Answer extracted like run.py line 469 (text after the last '#', commas removed)."""
    x = common.encode(tokenizer, text + "\n" + common.START + common.LATENT * n_latents + common.END, device)
    embeds = common.fill_latents(model, x)
    gen = common.greedy_generate(model, tokenizer, embeds, max_new_tokens)
    answer = gen["raw_text"].split("#")[-1].replace(",", "").strip()
    return answer, gen["raw_text"], embeds, x


def decode_thoughts(model, explainer, tokenizer, embeds, x, n_pairs=None):
    """Explainer text for each thought pair (pair i = latent slots 2i, 2i+1)."""
    slots = common.latent_positions(model, x)
    pairs = [slots[i:i + C_THOUGHT] for i in range(0, len(slots), C_THOUGHT)][:n_pairs]
    return [common.explain(model, explainer, tokenizer, embeds[0, p]) for p in pairs]


def same_step(a, b):
    return "".join(a.split()) == "".join(b.split())


def main():
    parser = argparse.ArgumentParser(description="SIM-CoT GPT-2 gate + 2-round interleaving test.")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--n-decode", type=int, default=20, help="questions for G2 (all 5 thought pairs)")
    parser.add_argument("--out", default="results/simcot/simcot_run.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    with open(common.path("data/gsm_test.json")) as f:
        data = json.load(f)[: args.limit]
    golds = [d["answer"].replace(",", "").strip() for d in data]

    model, explainer, tokenizer, ids, info = common.load_simcot(device)
    report = {"meta": common.run_meta(part="simcot_run", **vars(args)), "G0": {"pass": explainer is not None, **info}}
    print(f"G0 explainer in checkpoint: {'PASS' if explainer is not None else 'FAIL'}", flush=True)
    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()

    with torch.no_grad():
        # G1: native sweep, all questions
        report["G1"] = {}
        for n in NATIVE_NS:
            correct = [run_round(model, tokenizer, d["question"], n, device)[0] == g for d, g in zip(data, golds)]
            report["G1"][str(n)] = accuracy_block(correct)
            print(f"G1 {n} latents: {sum(correct)}/{len(correct)} = {sum(correct) / len(correct):.4f}", flush=True)

        if explainer is None:
            with open(out, "w") as f:
                json.dump(report, f, indent=1, default=str)
            print("STOP: no explainer in the checkpoint, the interleaving test is not possible. saved", out)
            sys.exit(1)

        # G2 + main
        records = []
        for i, (d, gold) in enumerate(zip(data, golds)):
            q, true_steps = d["question"], d["steps"]
            a_l, raw_l, embeds, x = run_round(model, tokenizer, q, N_LATENTS, device)
            steps = decode_thoughts(model, explainer, tokenizer, embeds, x, None if i < args.n_decode else 1)
            s1, g1 = steps[0], (true_steps[0] if true_steps else "")
            a_t2, raw_t2, _, _ = run_round(model, tokenizer, q + "\n" + s1, N_LATENTS, device)
            a_gold, raw_gold, _, _ = run_round(model, tokenizer, q + "\n" + g1, N_LATENTS, device)
            records.append({
                "idx": i, "gold": gold, "true_steps": true_steps, "decoded_steps": steps,
                "s1": s1, "s1_exact": same_step(s1, g1),
                "L": {"answer": a_l, "raw": raw_l, "correct": a_l == gold},
                "T2": {"answer": a_t2, "raw": raw_t2, "correct": a_t2 == gold},
                "T2_gold": {"answer": a_gold, "raw": raw_gold, "correct": a_gold == gold},
            })
            if i < args.n_decode:
                print(f"--- q{i} gold {gold} | L {a_l!r} | T2 {a_t2!r} | T2-gold {a_gold!r}\n"
                      f"  true steps:    {true_steps}\n  decoded steps: {steps}", flush=True)

    def run_of(key):
        return {"records": [{"correct": r[key]["correct"]} for r in records]}

    report["main"] = {key: accuracy_block([r[key]["correct"] for r in records]) for key in ("L", "T2", "T2_gold")}
    report["pairs"] = {
        "T2-L": pair_test(run_of("T2"), run_of("L"), "T2", "L"),
        "T2_gold-L": pair_test(run_of("T2_gold"), run_of("L"), "T2_gold", "L"),
        "T2-T2_gold": pair_test(run_of("T2"), run_of("T2_gold"), "T2", "T2_gold"),
    }
    report["step1_exact_match"] = sum(r["s1_exact"] for r in records) / len(records)
    report["seconds"] = time.perf_counter() - t0
    report["records"] = records

    print("=== summary ===")
    for n, b in report["G1"].items():
        print(f"G1 {n} latents: {b['acc']:.4f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    print(f"step 1 decoded exactly (vs true step 1): {report['step1_exact_match']:.4f}")
    for key, b in report["main"].items():
        print(f"{key}: {b['acc']:.4f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]  n={b['n']}")
    for name, t in report["pairs"].items():
        print(f"{name}: diff {t['diff']:+.4f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  "
              f"McNemar p={t['mcnemar_p']:.3g} -> {t['verdict']}")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", out)


if __name__ == "__main__":
    main()

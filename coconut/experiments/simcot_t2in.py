"""SIM-CoT GPT-2: step 1 inserted inside the question (T2-in, T2-in-gold).
Plan: docs/planning/simcot_t2in_plan.md. Follows simcot_run.py (job 1449771, T2 / T2-gold).

  L           question -> 10 latents -> answer (re-run, must equal the saved run)
  T2-in       question with the explainer's s1 put before its last sentence -> 10 latents -> answer
  T2-in-gold  as T2-in but s1 = the true step 1
s1 = explainer(thought pair 1) of the L round, as in simcot_run.py.
Step text: '<<16-3-4=9>>' -> '16-3-4=9.'

    python experiments/simcot_t2in.py --out results/simcot/simcot_t2in.json
"""

import argparse
import json
import os
import re
import time

import torch

import common
from score import accuracy_block, pair_test
from simcot_run import N_LATENTS, decode_thoughts, run_round

SENT_END = re.compile(r"[.?!]\s+")  # a sentence end followed by a space; '2.5' is not split


def as_sentence(step):
    """'<<16-3-4=9>>' -> '16-3-4=9.'"""
    s = step.replace("<<", "").replace(">>", "").strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s


def insert_step(question, step):
    """Put the step right before the last sentence of the question.
    One sentence only -> step at the start. Empty step -> question unchanged."""
    s = as_sentence(step)
    if not s:
        return question
    q = question.rstrip()
    tail = question[len(q):]
    ends = list(SENT_END.finditer(q))
    if not ends:
        return s + " " + q + tail
    cut = ends[-1].end()
    return q[:cut] + s + " " + q[cut:] + tail


def main():
    parser = argparse.ArgumentParser(description="SIM-CoT GPT-2: step 1 inside the question.")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--n-print", type=int, default=20, help="questions printed in full")
    parser.add_argument("--saved", default="results/simcot/simcot_run.json", help="job 1449771 results")
    parser.add_argument("--out", default="results/simcot/simcot_t2in.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    with open(common.path("data/gsm_test.json")) as f:
        data = json.load(f)[: args.limit]
    golds = [d["answer"].replace(",", "").strip() for d in data]
    with open(common.path(args.saved)) as f:
        saved = json.load(f)["records"][: len(data)]

    model, explainer, tokenizer, ids, info = common.load_simcot(device)
    assert explainer is not None, "no explainer in the checkpoint"
    report = {"meta": common.run_meta(part="simcot_t2in", **vars(args))}
    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.perf_counter()

    records = []
    with torch.no_grad():
        for i, (d, gold) in enumerate(zip(data, golds)):
            q, true_steps = d["question"], d["steps"]
            a_l, raw_l, embeds, x = run_round(model, tokenizer, q, N_LATENTS, device)
            s1 = decode_thoughts(model, explainer, tokenizer, embeds, x, 1)[0]
            g1 = true_steps[0] if true_steps else ""
            q_in, q_gold_in = insert_step(q, s1), insert_step(q, g1)
            a_in, raw_in, _, _ = run_round(model, tokenizer, q_in, N_LATENTS, device)
            a_gold, raw_gold, _, _ = run_round(model, tokenizer, q_gold_in, N_LATENTS, device)
            records.append({
                "idx": i, "gold": gold, "s1": s1, "g1": g1,
                "q_T2_in": q_in, "q_T2_in_gold": q_gold_in,
                "same_L_as_saved": a_l == saved[i]["L"]["answer"],
                "same_s1_as_saved": s1 == saved[i]["s1"],
                "L": {"answer": a_l, "raw": raw_l, "correct": a_l == gold},
                "T2_in": {"answer": a_in, "raw": raw_in, "correct": a_in == gold},
                "T2_in_gold": {"answer": a_gold, "raw": raw_gold, "correct": a_gold == gold},
            })
            if i < args.n_print:
                print(f"--- q{i} gold {gold} | L {a_l!r} | T2-in {a_in!r} | T2-in-gold {a_gold!r}\n"
                      f"  T2-in:      {q_in}\n  T2-in-gold: {q_gold_in}", flush=True)

    def run_of(recs, key):
        return {"records": [{"correct": r[key]["correct"]} for r in recs]}

    report["main"] = {k: accuracy_block([r[k]["correct"] for r in records]) for k in ("L", "T2_in", "T2_in_gold")}
    report["pairs"] = {
        "T2_in-L": pair_test(run_of(records, "T2_in"), run_of(records, "L"), "T2_in", "L"),
        "T2_in_gold-L": pair_test(run_of(records, "T2_in_gold"), run_of(records, "L"), "T2_in_gold", "L"),
        "T2_in-T2_in_gold": pair_test(run_of(records, "T2_in"), run_of(records, "T2_in_gold"), "T2_in", "T2_in_gold"),
        # placement effect: same questions and s1 as job 1449771, only the position of the step changes
        "T2_in-T2": pair_test(run_of(records, "T2_in"), run_of(saved, "T2"), "T2_in", "T2"),
        "T2_in_gold-T2_gold": pair_test(run_of(records, "T2_in_gold"), run_of(saved, "T2_gold"),
                                        "T2_in_gold", "T2_gold"),
    }
    report["checks"] = {
        "L_equal_saved": sum(r["same_L_as_saved"] for r in records),
        "s1_equal_saved": sum(r["same_s1_as_saved"] for r in records),
        "n": len(records),
        "gold_step_empty": sum(not r["g1"] for r in records),
        "s1_is_answer": sum(r["s1"].strip().startswith("#") for r in records),
        "one_sentence_question": sum(not SENT_END.search(d["question"].rstrip()) for d in data),
    }
    report["seconds"] = time.perf_counter() - t0
    report["records"] = records

    print("=== summary ===")
    print("checks:", report["checks"])
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

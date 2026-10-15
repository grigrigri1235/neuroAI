"""Step 0 of the fine-tune: the ORIGINAL SIM-CoT model's own step 1 for each training question.
Plan: docs/planning/simcot_finetune_plan.md (section 4).

question -> 10 latents (same input as L in simcot_run.py), explainer(thought pair 1) -> own step 1.
Saves every --save-every questions and continues from a partial file (long run).

    python experiments/simcot_ft_steps.py --n-train 50000
"""

import argparse
import json
import os
import sys
import time

import torch

import common
from simcot_finetune import N_TRAIN, train_subset
from simcot_run import N_LATENTS, decode_thoughts


def own_step1(model, explainer, tokenizer, question, device):
    """Same prefix as simcot_run.run_round(question); explainer text of thought pair 1."""
    x = common.encode(tokenizer, question + "\n" + common.START + common.LATENT * N_LATENTS + common.END, device)
    embeds = common.fill_latents(model, x)
    return decode_thoughts(model, explainer, tokenizer, embeds, x, 1)[0]


def save(out, report):
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, out)  # never leave a half-written file


def main():
    parser = argparse.ArgumentParser(description="Own step 1 of the original SIM-CoT model for training questions.")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--saved", default="results/simcot/simcot_run.json", help="job 1449771 results (check 0)")
    parser.add_argument("--out", default="results/simcot/ft_train_steps.json")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    with open(common.path("data/gsm_train.json")) as f:
        train = json.load(f)
    with open(common.path("data/gsm_test.json")) as f:
        test = json.load(f)
    with open(common.path(args.saved)) as f:
        saved = json.load(f)["records"]
    indices = train_subset(len(train), args.n_train)
    print(f"training questions in gsm_train.json: {len(train)}; using {len(indices)} (shuffled, seed 0)", flush=True)

    model, explainer, tokenizer, ids, info = common.load_simcot(device)
    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with torch.no_grad():
        # check 0: this code gives the saved step 1 on the first 20 test questions
        same = [own_step1(model, explainer, tokenizer, test[i]["question"], device) == saved[i]["s1"]
                for i in range(20)]
        print(f"check 0 (own step 1 = saved s1 on 20 test questions): {sum(same)}/20", flush=True)
        if not all(same):
            print("FAILED check 0: the step code does not reproduce job 1449771")
            sys.exit(1)

        report = {"meta": common.run_meta(part="simcot_ft_steps", **vars(args)), "n_total_train": len(train),
                  "check0": sum(same), "indices": indices, "steps": []}
        if os.path.exists(out):  # continue a partial run
            with open(out) as f:
                old = json.load(f)
            if old["indices"] == indices:
                report["steps"] = old["steps"]
                print(f"continuing from {len(report['steps'])} saved steps", flush=True)

        t0 = time.perf_counter()
        done_before = len(report["steps"])
        for n, i in enumerate(indices[done_before:], start=done_before):
            report["steps"].append(own_step1(model, explainer, tokenizer, train[i]["question"], device))
            if (n + 1) % args.save_every == 0 or n + 1 == len(indices):
                save(out, report)
                rate = (n + 1 - done_before) / (time.perf_counter() - t0)
                left = (len(indices) - n - 1) / rate / 60
                print(f"{n + 1}/{len(indices)} steps, {rate:.1f} q/s, ~{left:.0f} min left | "
                      f"e.g. {report['steps'][-1]!r} (true {train[i]['steps'][:1]})", flush=True)

    exact = sum(s == (train[i]["steps"][0] if train[i]["steps"] else "") for s, i in zip(report["steps"], indices))
    report["step1_exact_match"] = exact / len(indices)
    save(out, report)
    print(f"own step 1 = true step 1 (exact text): {exact / len(indices):.4f}. saved {out}")


if __name__ == "__main__":
    main()

"""Short fine-tune of SIM-CoT GPT-2 so it learns to read its own written step 1.
Plan: docs/planning/simcot_finetune_plan.md (sections 5, 6, 9).

  --mode mix    half of the questions (coin flip, seed 1) get the model's own step 1 after the question
  --mode plain  the same questions in the same order, never a step (the fair check)
Both: 10 latents, loss only on '### N<eos>' (like the official Coconut final stage), official
Coconut.forward + MyCollator, one pass over the same 50,000 training questions.
Own steps come from simcot_ft_steps.py (results/simcot/ft_train_steps.json).

    python experiments/simcot_finetune.py --mode mix
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import get_linear_schedule_with_warmup

import common
from dataset import MyCollator
from score import accuracy_block
from simcot_run import N_LATENTS, run_round

N_TRAIN = 50000


def train_subset(n_total, n, seed=0):
    """The training questions used (same for both models and for simcot_ft_steps.py)."""
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    return indices[:n]


def prefix_text(question, step=None):
    """The same text simcot_run.run_round builds for question (+ "\\n" + step)."""
    text = question + "\n" + step if step else question
    return text + "\n" + common.START + common.LATENT * N_LATENTS + common.END


def build_example(tokenizer, question, answer, step=None):
    """Input = prefix + '### answer' + EOS; labels: -100 on the prefix, so only the answer is graded."""
    prefix = tokenizer.encode(prefix_text(question, step))
    target = tokenizer.encode("### " + answer) + [tokenizer.eos_token_id]
    ids = prefix + target
    return {"input_ids": ids, "labels": [-100] * len(prefix) + target, "attention_mask": [1] * len(ids)}


def batches(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        yield [dict(e) for e in examples[i:i + batch_size]]  # copies: MyCollator edits the dicts


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def mean_loss(model, collator, examples, batch_size, device):
    model.eval()
    losses = [float(model(**to_device(collator(b), device)).loss) for b in batches(examples, batch_size)]
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SIM-CoT GPT-2 (mix = with own steps, plain = without).")
    parser.add_argument("--mode", choices=["mix", "plain"], required=True)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--n-valid", type=int, default=300)
    parser.add_argument("--steps", default="results/simcot/ft_train_steps.json", help="from simcot_ft_steps.py")
    parser.add_argument("--saved", default="results/simcot/simcot_run.json", help="job 1449771 results (check 1)")
    args = parser.parse_args()

    common.set_seed(0)
    device = common.get_device()
    with open(common.path("data/gsm_train.json")) as f:
        train = json.load(f)
    with open(common.path(args.steps)) as f:
        own = json.load(f)
    indices = train_subset(len(train), args.n_train)
    if own["indices"] != indices or len(own["steps"]) != len(indices):
        sys.exit("FAILED: the own-steps file does not match the training questions (run simcot_ft_steps.py)")
    coins = random.Random(1).choices([True, False], k=len(indices))  # same coins for both modes
    with_step = [c and args.mode == "mix" for c in coins]

    model, explainer, tokenizer, ids, info = common.load_simcot(device)
    del explainer  # not trained, not needed
    collator = MyCollator(tokenizer, latent_id=ids["latent"], label_pad_token_id=-100)
    report = {"meta": common.run_meta(part="simcot_finetune", **vars(args)), "n_total_train": len(train),
              "n_with_step": sum(with_step), "checks": {}}

    # check 1 (hard): training input = test input (simcot_run.run_round), both versions
    with open(common.path("data/gsm_test.json")) as f:
        test = json.load(f)
    with open(common.path(args.saved)) as f:
        saved = json.load(f)["records"]
    ok = []
    with torch.no_grad():
        for i in range(5):
            q, s1 = test[i]["question"], saved[i]["s1"]
            for step in (None, s1):
                x = run_round(model, tokenizer, q + "\n" + step if step else q, N_LATENTS, device)[3]
                ex = build_example(tokenizer, q, test[i]["answer"], step)
                ok.append(ex["input_ids"][:x.shape[1]] == x[0].tolist())
    report["checks"]["check1_train_equals_test_input"] = f"{sum(ok)}/{len(ok)}"
    print(f"check 1 (training input = test input): {sum(ok)}/{len(ok)}", flush=True)
    if not all(ok):
        sys.exit("FAILED check 1")

    examples = [build_example(tokenizer, train[i]["question"], train[i]["answer"], s if w else None)
                for i, s, w in zip(indices, own["steps"], with_step)]

    # check 2: show what is graded
    ex = next(e for e, w in zip(examples, with_step) if w) if any(with_step) else examples[0]
    graded = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print(f"check 2: input  = {tokenizer.decode(ex['input_ids'])!r}\n         graded = {tokenizer.decode(graded)!r}",
          flush=True)
    report["checks"]["check2_graded_example"] = tokenizer.decode(graded)

    # check 3: loss before training, plain vs step version of the same 200 questions
    probe = list(zip(indices, own["steps"]))[:200]
    loss_plain = mean_loss(model, collator, [build_example(tokenizer, train[i]["question"], train[i]["answer"])
                                             for i, _ in probe], args.batch_size, device)
    loss_step = mean_loss(model, collator, [build_example(tokenizer, train[i]["question"], train[i]["answer"], s)
                                            for i, s in probe], args.batch_size, device)
    report["checks"]["check3_loss_before"] = {"plain": loss_plain, "own_step": loss_step}
    print(f"check 3 (loss before training): plain {loss_plain:.4f} | with own step {loss_step:.4f}", flush=True)

    # training: one pass, same order for both modes
    n_updates = (len(examples) + args.batch_size - 1) // args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, n_updates)
    model.train()
    curve, window, t0 = [], [], time.perf_counter()
    for step, feats in enumerate(batches(examples, args.batch_size), start=1):
        loss = model(**to_device(collator(feats), device)).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        window.append(float(loss))
        if step % 50 == 0 or step == n_updates:
            curve.append({"update": step, "loss": sum(window) / len(window)})
            window = []
            elapsed = time.perf_counter() - t0
            print(f"update {step}/{n_updates} loss {curve[-1]['loss']:.4f} | {elapsed / 60:.1f} min, "
                  f"~{elapsed / step * (n_updates - step) / 60:.0f} min left", flush=True)
            if step == 50:  # check 4
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                report["checks"]["check4_speed"] = {"sec_per_update": elapsed / step, "max_mem_gb": mem,
                                                    "est_total_min": elapsed / step * n_updates / 60}
                print(f"check 4: {elapsed / step:.2f} s/update, max GPU memory {mem:.1f} GB, "
                      f"estimated total {elapsed / step * n_updates / 60:.0f} min", flush=True)
    model.eval()
    report["loss_curve"] = curve
    report["train_minutes"] = (time.perf_counter() - t0) / 60

    ckpt = common.path(f"checkpoints_ft/{args.mode}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print("saved model", ckpt, flush=True)

    # quick sanity: L on validation questions
    with open(common.path("data/gsm_valid.json")) as f:
        valid = json.load(f)[: args.n_valid]
    with torch.no_grad():
        correct = [run_round(model, tokenizer, d["question"], N_LATENTS, device)[0] == d["answer"].replace(",", "").strip()
                   for d in valid]
    report["valid_L"] = accuracy_block(correct)
    print(f"valid L ({len(valid)} questions): {sum(correct)}/{len(correct)}", flush=True)

    out = common.path(f"results/simcot/ft_train_{args.mode}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("saved", out)


if __name__ == "__main__":
    main()

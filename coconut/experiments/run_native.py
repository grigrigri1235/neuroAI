"""Part 1: GPT-2 Coconut on the GSM8K test set in its own format (paper reproduction,
inference only). Same weights, k = 0..3 stages -> 0 / 2 / 4 / 6 latents.

Inputs are built with the official dataset.py functions (like the run.py eval),
decoded with the official Coconut.generate, and scored exactly like run.py line 469.

    python experiments/run_native.py                 # full test set, k = 0 1 2 3
    python experiments/run_native.py --limit 20      # smoke test
"""

import argparse
import json
import os
import time

import torch
import yaml
from tqdm import tqdm

import common  # sets sys.path for the official modules below
from dataset import get_dataset, get_question_latent_dataset
from utils import Config


def build_native_datasets(
    tokenizer, ids, ks, limit=None, data="data/gsm_test.json", config="args/gsm_coconut.yaml"
):
    """{k: rows with input_ids = question + "\\n" + <|start-latent|> + 2k x <|latent|> + <|end-latent|>}.

    Call this before the model goes to the GPU: dataset.map forks worker processes.
    """
    with open(common.path(config)) as f:
        configs = Config(yaml.safe_load(f))  # c_thought 2, max_latent_stage 3, pad_latent_to_max True
    base = get_dataset(common.path(data), tokenizer, max_size=limit or 100000000)
    return {
        k: get_question_latent_dataset(k, base, configs, ids["start"], ids["latent"], ids["end"])
        for k in ks
    }


def main():
    parser = argparse.ArgumentParser(description="Part 1: native GSM8K sweep over the number of latents.")
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    parser.add_argument("--max-new-tokens", type=int, default=64)  # run.py uses 64 for GSM8K
    parser.add_argument("--data", default="data/gsm_test.json")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--fill", default="official", choices=["official", "plain"],
                        help="official: Coconut.generate (hidden-state feedback). plain: same input, latent "
                             "slots keep their token embedding (no feedback), our greedy loop (pivot plan D2)")
    parser.add_argument("--out", default=None,
                        help="default: results/gpt2/native.json (official) or native_plain.json (plain)")
    args = parser.parse_args()
    if args.out is None:
        args.out = "results/gpt2/native.json" if args.fill == "official" else "results/gpt2/native_plain.json"

    common.set_seed(0)
    tokenizer, ids = common.load_tokenizer()
    with open(common.path(args.data)) as f:
        answers = [d["answer"].replace(",", "").strip() for d in json.load(f)]  # as in run.py
    rows_per_k = build_native_datasets(tokenizer, ids, args.ks, args.limit, args.data)
    device = common.get_device()
    model, _ = common.load_model(tokenizer, ids, device, args.ckpt)

    records = []
    with torch.no_grad():
        for k in args.ks:
            k_records = []
            for row in tqdm(rows_per_k[k], desc=f"k={k}", mininterval=30):
                input_ids = torch.tensor([row["input_ids"]], device=device)
                common.sync(device)
                t0 = time.perf_counter()
                if args.fill == "official":
                    output = model.generate(
                        input_ids, torch.ones_like(input_ids), max_new_tokens=args.max_new_tokens
                    )
                    new_ids = output[0, input_ids.shape[1]:].tolist()
                else:  # control: no hidden-state feedback (our loop equals Coconut.generate, smoke C3)
                    embeds = common.get_embeds(model, input_ids, "plain")
                    new_ids = common.greedy_generate(model, tokenizer, embeds, args.max_new_tokens)["new_ids"]
                common.sync(device)
                seconds = time.perf_counter() - t0
                # official scoring, run.py lines 468-469 (decode question + markers + new tokens)
                text_output = tokenizer.decode(row["input_ids"] + new_ids, skip_special_tokens=True)
                extracted = text_output.split("#")[-1].replace(",", "").strip()
                gold = answers[row["idx"]]
                k_records.append({
                    "idx": row["idx"],
                    "k": k,
                    "n_latents": row["input_ids"].count(ids["latent"]),
                    "prompt_len": len(row["input_ids"]),
                    "output_text": tokenizer.decode(new_ids),
                    "extracted": extracted,
                    "gold": gold,
                    "correct": extracted == gold,
                    "n_new_tokens": len([t for t in new_ids if t != ids["eos"]]),
                    "seconds": seconds,
                })
            for r in k_records[:3]:
                print(f"k={k} idx={r['idx']} gold={r['gold']} extracted={r['extracted']!r} "
                      f"output={r['output_text']!r}")
            acc = sum(r["correct"] for r in k_records) / len(k_records)
            print(f"k={k} ({2 * k} latents): accuracy {acc:.4f} on {len(k_records)} questions", flush=True)
            records += k_records

    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = common.run_meta(
        part="native", fill=args.fill, ks=args.ks, max_new_tokens=args.max_new_tokens, data=args.data,
        limit=args.limit, seed=0,
    )
    with open(out, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()

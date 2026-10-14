"""Smoke test before the full run (plan section 5).

Required checks C0-C6 must pass, else exit code 1 and the slurm job stops there.
C7, C8 (and C4b inside C4) are information only. Writes a JSON report.

    python experiments/smoke_test.py --out results/gpt2/smoke/smoke_report.json
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback

import torch

import common
import prompts
from run_fewshot import choose_n_shots, worst_stage2_len
from run_native import build_native_datasets

REQUIRED = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
TOL = 1e-4  # max relative difference for "the same vectors" (float noise only)
PHASE4_SCRIPTS = ["eval_qwen_1.5b_gsm8k_less_handholding.py", "eval_qwen_1.5b_gsm8k_no_latents.py"]


def rel_diff(a, b):
    """max |a - b| relative to max |b|."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


@torch.no_grad()
def official_embeds(model, input_ids):
    """inputs_embeds with the latents filled by the official Coconut.forward."""
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return model(input_ids, torch.ones_like(input_ids), input_ids.clone(), position_ids).inputs_embeds


def check_c0():
    """FEW_SHOT_PROMPT is byte for byte the Phase 4 one. Required: the S3 script (the KR
    source); the S4 script is information only."""
    same = {}
    for name in PHASE4_SCRIPTS:
        spec = importlib.util.spec_from_file_location("phase4_script", common.path(name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # only defines things; its main() is behind __name__
        same[name] = module.FEW_SHOT_PROMPT == prompts.FEW_SHOT_PROMPT
    return same[PHASE4_SCRIPTS[0]], {"same_as": same}


def check_c2(model, tokenizer, native_rows, questions, n_shots, device):
    """Our fill_latents == the official Coconut.forward, on prompts with one latent block."""
    worst = {"native_k3": 0.0, "s3_stage1": 0.0}
    for row in native_rows:
        x = torch.tensor([row["input_ids"]], device=device)
        d = rel_diff(common.fill_latents(model, x), official_embeds(model, x))
        worst["native_k3"] = max(worst["native_k3"], d)
    for q in questions:
        x = common.encode(tokenizer, prompts.stage1_prompt(q, True, n_shots), device)
        d = rel_diff(common.fill_latents(model, x), official_embeds(model, x))
        worst["s3_stage1"] = max(worst["s3_stage1"], d)
    return max(worst.values()) < TOL, {"max_rel_diff": worst}


def check_c3(model, tokenizer, ids, native_rows, device, max_new_tokens=64):
    """Our greedy loop (KV cache on and off) gives the same tokens as Coconut.generate."""
    matches, mismatches = {"cache": 0, "no_cache": 0}, []
    for row in native_rows:
        x = torch.tensor([row["input_ids"]], device=device)
        ref = model.generate(x, torch.ones_like(x), max_new_tokens=max_new_tokens)[0, x.shape[1]:].tolist()
        if ids["eos"] in ref:
            ref = ref[: ref.index(ids["eos"])]
        embeds = common.fill_latents(model, x)
        for mode, use_cache in (("cache", True), ("no_cache", False)):
            ours = common.greedy_generate(model, tokenizer, embeds, max_new_tokens, (), use_cache)["new_ids"]
            if ours == ref:
                matches[mode] += 1
            else:
                mismatches.append({"idx": row["idx"], "mode": mode,
                                   "official": tokenizer.decode(ref), "ours": tokenizer.decode(ours)})
    n = len(native_rows)
    # one mismatch per mode is allowed (a float-noise argmax flip); a bug breaks most of them
    return all(m >= n - 1 for m in matches.values()), {"n": n, "matches": matches, "mismatches": mismatches[:6]}


def check_c4(model, tokenizer, questions, n_shots, device):
    """Stage 2 (two latent blocks): same token prefix, 4 latents, and the stage-1 thoughts
    inside stage 2 equal the ones from the stage-1 run (causal model)."""
    worst, problems, official = 0.0, [], None
    for q in questions:
        p1 = prompts.stage1_prompt(q, True, n_shots)
        x1 = common.encode(tokenizer, p1, device)
        e1 = common.fill_latents(model, x1)
        stage1 = common.greedy_generate(model, tokenizer, e1, 64, prompts.STOP_STRINGS)
        text, _ = common.clean_text(stage1["cut_text"])
        x2 = common.encode(tokenizer, prompts.stage2_prompt(p1, text, True), device)
        slots1, slots2 = common.latent_positions(model, x1), common.latent_positions(model, x2)
        end1 = slots1[-1] + 1 if slots1 else 0
        if (len(slots1) != 2 or len(slots2) != 4 or slots2[:2] != slots1
                or x2[0, :end1].tolist() != x1[0, :end1].tolist()):
            problems.append(q[:60])
            continue
        e2 = common.fill_latents(model, x2)
        worst = max(worst, rel_diff(e2[0, slots2[:2]], e1[0, slots1]))
        if official is None:  # C4b (info): the official forward on a prompt with two latent blocks
            try:
                ref = official_embeds(model, x2)
                official = f"no error; max rel diff vs ours {rel_diff(e2, ref):.2e}"
            except Exception as e:  # expected: IndexError (plan F1)
                official = f"{type(e).__name__}: {e}"
    details = {"max_rel_diff": worst, "problems": problems, "C4b_official_forward_two_blocks": official}
    return (not problems and worst < TOL), details


def check_c5(model, tokenizer, questions, n_shots, device):
    """S3 and S5 inputs differ exactly at the <|latent|> positions."""
    bad = []
    for q in questions:
        x = common.encode(tokenizer, prompts.stage1_prompt(q, True, n_shots), device)
        s3 = common.get_embeds(model, x, "latent")
        s5 = common.get_embeds(model, x, "plain")
        differ = ((s3 - s5).abs().amax(dim=-1)[0] > 0).nonzero().flatten().tolist()
        if differ != common.latent_positions(model, x):
            bad.append(q[:60])
    return not bad, {"n": len(questions), "bad": bad}


def check_c6(tokenizer, questions):
    """Prompt lengths over all questions fit GPT-2's 1024 tokens (length guard, KR 5.4)."""
    worst = {f"{n}_shots": worst_stage2_len(tokenizer, questions, n, 64) for n in (3, 2)}
    n_shots, _ = choose_n_shots(tokenizer, questions, 64)  # raises if neither fits
    s1_len = max(len(tokenizer.encode(prompts.single_prompt("S1", q))) for q in questions) + 64
    s2_len = max(len(tokenizer.encode(prompts.single_prompt("S2", q))) for q in questions) + 128
    details = {
        "worst_stage2_prompt_len": worst, "new_tokens_per_stage": 64, "n_shots": n_shots,
        "S1_prompt_plus_64": s1_len, "S2_prompt_plus_128": s2_len, "limit": common.MAX_POSITIONS,
    }
    return max(s1_len, s2_len) <= common.MAX_POSITIONS, details


def check_c7():
    """(info) The iCoT test file (Part 1) vs the HF GSM8K test file (Part 2)."""
    with open(common.path("data/gsm_test.json")) as f:
        icot = json.load(f)
    with open(common.path("data/gsm8k_test.json")) as f:
        hf = json.load(f)

    def norm(s):
        return " ".join(s.split())

    return True, {
        "n_icot": len(icot),
        "n_hf": len(hf),
        "same_question_text": sum(norm(a["question"]) == norm(b["question"]) for a, b in zip(icot, hf)),
        "same_gold": sum(a["answer"].replace(",", "").strip() == common.gsm8k_gold(b["answer"])
                         for a, b in zip(icot, hf)),
    }


def check_c8(model, tokenizer, ids):
    """(info) How far the marker embeddings moved from their init (the GPT-2 "<<" row), plan F2."""
    from transformers import AutoModelForCausalLM

    gpt2 = AutoModelForCausalLM.from_pretrained(common.BASE_MODEL_ID).get_input_embeddings().weight.detach()
    trained = model.embedding.weight.detach().cpu()
    init_id = tokenizer.convert_tokens_to_ids("<<")

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    out = {
        name: {"cos_to_gpt2_<<": cos(trained[ids[name]], gpt2[init_id]), "norm": float(trained[ids[name]].norm())}
        for name in ("latent", "start", "end")
    }
    out["<<"] = {"cos_to_gpt2_<<": cos(trained[init_id], gpt2[init_id]), "norm": float(trained[init_id].norm())}
    out["mean_token_norm"] = float(trained[: gpt2.shape[0]].norm(dim=1).mean())
    return True, out


def main():
    parser = argparse.ArgumentParser(description="Smoke test checks C0-C8.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default="results/gpt2/smoke/smoke_report.json")
    args = parser.parse_args()

    common.set_seed(0)
    tokenizer, ids = common.load_tokenizer()
    native_rows = list(build_native_datasets(tokenizer, ids, [3], args.n)[3])  # before CUDA init
    with open(common.path("data/gsm8k_test.json")) as f:
        all_questions = [d["question"] for d in json.load(f)]
    questions = all_questions[: args.n]
    device = common.get_device()
    report = {"meta": common.run_meta(part="smoke", n=args.n)}

    def record(name, check):
        try:
            ok, details = check()
        except Exception:
            ok, details = False, {"error": traceback.format_exc()}
        report[name] = {"pass": bool(ok), **details}
        print(f"{name}: {'PASS' if ok else 'FAIL'}  {json.dumps(details, default=str)[:3000]}", flush=True)

    loaded = {}

    def check_c1():
        """Checkpoint loads, keys match (load_model raises otherwise)."""
        loaded["model"], info = common.load_model(tokenizer, ids, device, args.ckpt)
        return True, info

    record("C0", check_c0)
    record("C1", check_c1)
    record("C6", lambda: check_c6(tokenizer, all_questions))
    n_shots = report["C6"].get("n_shots", 3)
    model = loaded.get("model")
    if model is not None:
        with torch.no_grad():
            record("C2", lambda: check_c2(model, tokenizer, native_rows, questions, n_shots, device))
            record("C3", lambda: check_c3(model, tokenizer, ids, native_rows, device))
            record("C4", lambda: check_c4(model, tokenizer, questions, n_shots, device))
            record("C5", lambda: check_c5(model, tokenizer, questions, n_shots, device))
            record("C8", lambda: check_c8(model, tokenizer, ids))
    record("C7", check_c7)

    failed = [c for c in REQUIRED if not report.get(c, {}).get("pass")]
    report["required_failed"] = failed
    out = common.path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("SMOKE TEST:", "PASS" if not failed else f"FAIL {failed}")
    print("saved", out)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

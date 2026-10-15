"""Shared code for the Phase 5 experiments: GPT-2 Coconut on GSM8K.

Plan: docs/planning/gpt2_coconut_phase5_impl.md
The official files (coconut.py, run.py, dataset.py, utils.py) are imported, never modified.

As a script (login node, CPU is fine): downloads GPT-2 + the Coconut checkpoint into the
HF cache and checks that the checkpoint loads:
    python experiments/common.py
"""

import argparse
import datetime
import os
import re
import socket
import sys

import torch

# the official code is one folder up (coconut/)
COCONUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COCONUT_DIR not in sys.path:
    sys.path.insert(0, COCONUT_DIR)

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from coconut import Coconut  # noqa: E402
from utils import set_seed  # noqa: E402,F401  (used by the run scripts)

BASE_MODEL_ID = "openai-community/gpt2"
CKPT_REPO = "tsrigo/coconut"
CKPT_FILE = "stage4_checkpoint_25"
START, END, LATENT = "<|start-latent|>", "<|end-latent|>", "<|latent|>"
MARKERS = (START, END, LATENT)  # same order as tokenizer.add_tokens in run.py
MAX_POSITIONS = 1024  # GPT-2 context limit
# pivot (docs/planning/qwen7b_coconut_pivot.md): the saved tokenizer lacks the 3 Coconut tokens,
# but the model has 3 extra embedding rows (vocab 151668 = 151665 + 3)
QWEN_MODEL_ID = "agurung/coconut-qwen2.5-7b"
QWEN_MARKER_IDS = [151665, 151666, 151667]  # expected ids of START, END, LATENT
# old checkpoints can hold these constant causal-mask buffers; they are not weights
BUFFER_KEY_ENDINGS = (".attn.bias", ".attn.masked_bias")


def path(p):
    """Relative paths are relative to coconut/, so the scripts work from any folder."""
    return p if os.path.isabs(p) else os.path.join(COCONUT_DIR, p)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_meta(**extra):
    """Versions and hardware, saved with every result file."""
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    meta.update(extra)
    return meta


# ---------------------------------------------------------------- model


def load_tokenizer():
    """GPT-2 tokenizer + the 3 Coconut tokens, like run.py lines 104-111."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    for token in MARKERS:
        tokenizer.add_tokens(token)
    ids = {
        "start": tokenizer.convert_tokens_to_ids(START),
        "end": tokenizer.convert_tokens_to_ids(END),
        "latent": tokenizer.convert_tokens_to_ids(LATENT),
        "eos": tokenizer.eos_token_id,
    }
    return tokenizer, ids


def checkpoint_path(ckpt=None):
    if ckpt:
        return path(ckpt)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(CKPT_REPO, CKPT_FILE)


def strip_wrapper_prefixes(weights):
    for prefix in ("module.", "_fsdp_wrapped_module."):
        if all(k.startswith(prefix) for k in weights):
            print(f"stripping key prefix '{prefix}'")
            weights = {k[len(prefix):]: v for k, v in weights.items()}
    return weights


def load_model(tokenizer, ids, device, ckpt=None, weights=None, ignore_prefixes=()):
    """GPT-2 + Coconut wrapper + the tsrigo/coconut weights, like run.py lines 103-167.

    `weights` (an already loaded state dict) replaces `ckpt`; keys starting with `ignore_prefixes`
    are left out (e.g. SIM-CoT's explainer). Returns (model, info). Raises if the checkpoint does
    not match the model.
    """
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    base.resize_token_embeddings(len(tokenizer))
    # run.py copies the "<<" embedding into the new rows; the checkpoint overwrites them anyway
    embeddings = base.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in (ids["latent"], ids["start"], ids["end"]):
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        base.lm_head.weight.data[token_id] = base.lm_head.weight.data[target_id]
    model = Coconut(base, ids["latent"], ids["start"], ids["end"], ids["eos"])

    if weights is None:
        ckpt_file = checkpoint_path(ckpt)
        weights = strip_wrapper_prefixes(torch.load(ckpt_file, map_location="cpu", weights_only=True))
    else:
        ckpt_file = ckpt or "(weights given)"
    weights = {k: v for k, v in weights.items() if not k.startswith(tuple(ignore_prefixes))}
    print(f"checkpoint: {ckpt_file}\n  {len(weights)} keys, first: {list(weights)[:3]}")
    if not any(k.startswith("base_causallm.") for k in weights):
        raise RuntimeError("no key starts with 'base_causallm.': this is not a Coconut checkpoint")

    result = model.load_state_dict(weights, strict=False)
    print(result)  # run.py prints this too
    ignored = [k for k in result.unexpected_keys if k.endswith(BUFFER_KEY_ENDINGS)]
    unexpected = [k for k in result.unexpected_keys if k not in ignored]
    # a missing key is fine if it shares its tensor with a loaded key: GPT-2's wte, lm_head
    # and Coconut's `embedding` are one matrix under three names
    state = model.state_dict()
    loaded_tensors = {state[k].data_ptr() for k in weights if k in state}
    missing = [k for k in result.missing_keys if state[k].data_ptr() not in loaded_tensors]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: missing={missing}, unexpected={unexpected}"
        )
    if ignored:
        print(f"  ignored {len(ignored)} causal-mask buffer keys (not weights)")

    model.to(device)
    model.eval()  # Coconut.eval() returns None, so no chaining
    info = {
        "ckpt": ckpt_file,
        "n_keys": len(weights),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "ignored_buffer_keys": len(ignored),
    }
    return model, info


def load_qwen(device, dtype=torch.bfloat16):
    """agurung/coconut-qwen2.5-7b + the Coconut wrapper. Returns (model, tokenizer, ids, info).

    Adds the 3 Coconut tokens in run.py order and raises unless they get exactly the
    model's 3 extra rows (risk R1 in the pivot plan).
    """
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    n_saved = len(tokenizer)
    for token in MARKERS:
        tokenizer.add_tokens(token)
    ids = {
        "start": tokenizer.convert_tokens_to_ids(START),
        "end": tokenizer.convert_tokens_to_ids(END),
        "latent": tokenizer.convert_tokens_to_ids(LATENT),
        "eos": tokenizer.eos_token_id,
    }
    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=dtype)
    rows = base.get_input_embeddings().weight.shape[0]
    info = {
        "model": QWEN_MODEL_ID, "dtype": str(dtype), "tokenizer_len_saved": n_saved,
        "tokenizer_len": len(tokenizer), "embedding_rows": rows,
        "marker_ids": [ids["start"], ids["end"], ids["latent"]],
    }
    print(info, flush=True)
    if rows != len(tokenizer) or info["marker_ids"] != QWEN_MARKER_IDS:
        raise RuntimeError(f"latent tokens do not match the model's extra rows (risk R1): {info}")
    model = Coconut(base, ids["latent"], ids["start"], ids["end"], ids["eos"])
    # read by greedy_generate: Qwen ends with <|im_end|> or <|endoftext|>, and has a long context
    model.eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    model.max_positions = base.config.max_position_embeddings
    model.to(device)
    model.eval()
    return model, tokenizer, ids, info


SIMCOT_REPO, SIMCOT_FILE = "internlm/SIM_COT-GPT2-Coconut", "checkpoint_28"
EXPLAINER_PREFIX = "expainable_llm."  # sic: the name used in SIM-CoT's coconut.py


def load_simcot(device):
    """SIM-CoT GPT-2 Coconut (plan: docs/planning/simcot_interleaved_plan.md): the reasoning model in our
    Coconut wrapper + the explainer GPT-2 (None if the checkpoint has no explainer).
    Returns (model, explainer, tokenizer, ids, info)."""
    from collections import Counter

    from huggingface_hub import hf_hub_download

    tokenizer, ids = load_tokenizer()  # SIM-CoT run.py adds the same 3 tokens in the same order
    ckpt_file = hf_hub_download(SIMCOT_REPO, SIMCOT_FILE)
    weights = strip_wrapper_prefixes(torch.load(ckpt_file, map_location="cpu", weights_only=True))
    prefixes = dict(Counter(k.split(".")[0] for k in weights))
    print("key prefixes:", prefixes, flush=True)
    model, info = load_model(tokenizer, ids, device, ckpt=ckpt_file, weights=weights,
                             ignore_prefixes=(EXPLAINER_PREFIX,))
    info["key_prefixes"] = prefixes

    explainer_weights = {k[len(EXPLAINER_PREFIX):]: v for k, v in weights.items() if k.startswith(EXPLAINER_PREFIX)}
    explainer = None
    if explainer_weights:
        # SIM-CoT run.py: a fresh GPT-2 from the same model_id, not resized
        explainer = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
        result = explainer.load_state_dict(explainer_weights, strict=False)
        state = explainer.state_dict()
        loaded_tensors = {state[k].data_ptr() for k in explainer_weights if k in state}
        missing = [k for k in result.missing_keys if state[k].data_ptr() not in loaded_tensors]
        unexpected = [k for k in result.unexpected_keys if not k.endswith(BUFFER_KEY_ENDINGS)]
        info["explainer"] = {"n_keys": len(explainer_weights), "missing_keys": list(result.missing_keys),
                             "unexpected_keys": list(result.unexpected_keys)}
        if missing or unexpected:
            raise RuntimeError(f"explainer weights do not match GPT-2: missing={missing}, unexpected={unexpected}")
        explainer.to(device)
        explainer.eval()
    return model, explainer, tokenizer, ids, info


@torch.no_grad()
def explain(model, explainer, tokenizer, thoughts, max_new_tokens=32):
    """Decode a group of continuous thoughts (n, d) into step text with the SIM-CoT explainer.

    Like SIM-CoT training: input = the thought vectors, then the main model's embeddings of the tokens
    written so far; positions start at 1; stops at EOS. Greedy (SIM-CoT samples only for its visualisation).
    """
    device = thoughts.device
    seq = thoughts.unsqueeze(0)
    out_ids = []
    for _ in range(max_new_tokens):
        position_ids = torch.arange(1, seq.shape[1] + 1, device=device).unsqueeze(0)
        next_id = int(explainer(inputs_embeds=seq, position_ids=position_ids).logits[0, -1].argmax())
        if next_id == tokenizer.eos_token_id:
            break
        out_ids.append(next_id)
        seq = torch.cat([seq, model.embedding(torch.tensor([[next_id]], device=device))], dim=1)
    return tokenizer.decode(out_ids).strip()


QWEN3_MODEL_ID = "agurung/coconut-qwen3-4b-ff"
QWEN3_BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"  # tokenizer fallback only


def load_coconut_hf(repo, base_repo, device, dtype=torch.bfloat16):
    """A Coconut model exported to plain HF format (agurung/coconut-qwen3-4b-ff) + the Coconut wrapper.

    Plan: docs/planning/qwen3_4b_interleaved_plan.md (gate G0). Returns (model, tokenizer, ids, info).
    Raises if the token ids, embedding rows, weights or RoPE do not match.
    """
    import json

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    with open(hf_hub_download(repo, "latent_metadata.json")) as f:
        expected = json.load(f)["latent_token_ids"]

    # saved by transformers 5.x: rope_theta is nested, and 4.52 would silently use its default
    config = AutoConfig.from_pretrained(repo)
    nested = getattr(config, "rope_parameters", None)
    if isinstance(nested, dict) and "rope_theta" in nested:
        config.rope_theta = nested["rope_theta"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        tokenizer_source = repo
    except Exception as e:  # tokenizer files from transformers 5.x may not load in 4.52
        print(f"tokenizer of {repo} does not load ({type(e).__name__}: {e}), using {base_repo}")
        tokenizer = AutoTokenizer.from_pretrained(base_repo)
        tokenizer_source = base_repo
    for token in MARKERS:  # run.py order; no-op if the tokenizer already has them
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens(token)
    ids = {
        "start": tokenizer.convert_tokens_to_ids(START),
        "end": tokenizer.convert_tokens_to_ids(END),
        "latent": tokenizer.convert_tokens_to_ids(LATENT),
        "eos": tokenizer.eos_token_id,
    }

    base, load_info = AutoModelForCausalLM.from_pretrained(
        repo, config=config, torch_dtype=dtype, output_loading_info=True
    )
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = base.model.rotary_emb.inv_freq.float().cpu()
    wanted_inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    rows = base.get_input_embeddings().weight.shape[0]
    missing = [k for k in load_info["missing_keys"]
               if not (k == "lm_head.weight" and config.tie_word_embeddings)]
    info = {
        "model": repo, "dtype": str(dtype), "tokenizer_source": tokenizer_source,
        "tokenizer_len": len(tokenizer), "embedding_rows": rows,
        "marker_ids": [ids["start"], ids["end"], ids["latent"]],
        "expected_marker_ids": [expected[START], expected[END], expected[LATENT]],
        "rope_theta": config.rope_theta, "missing_keys": load_info["missing_keys"],
        "unexpected_keys": load_info["unexpected_keys"],
    }
    print(info, flush=True)
    problems = []
    if info["marker_ids"] != info["expected_marker_ids"]:
        problems.append("latent token ids differ from latent_metadata.json")
    if rows != len(tokenizer):
        problems.append("embedding rows != tokenizer size")
    if isinstance(nested, dict) and config.rope_theta != nested.get("rope_theta"):
        problems.append("rope_theta not taken from rope_parameters")
    if not torch.allclose(inv_freq, wanted_inv_freq, rtol=1e-2):
        problems.append("rotary inv_freq does not match rope_theta")
    if missing or load_info["unexpected_keys"]:
        problems.append("weights do not match the model")
    if problems:
        raise RuntimeError(f"G0 failed: {problems}")

    model = Coconut(base, ids["latent"], ids["start"], ids["end"], ids["eos"])
    model.eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    model.max_positions = config.max_position_embeddings
    model.to(device)
    model.eval()
    return model, tokenizer, ids, info


def encode(tokenizer, text, device):
    return torch.tensor([tokenizer.encode(text)], device=device)


def latent_positions(model, input_ids):
    return (input_ids[0] == model.latent_token_id).nonzero().flatten().tolist()


@torch.no_grad()
def fill_latents(model, input_ids):
    """Input embeddings where each <|latent|> slot holds the last hidden state of the
    position before it (= a continuous thought), filled left to right.

    Same result as Coconut.forward(...).inputs_embeds, but it also works when the latents
    sit in several separate blocks. Coconut.forward assumes one block (plan F1).
    """
    assert input_ids.shape[0] == 1, "batch size 1 only"
    embeds = model.embedding(input_ids).clone()
    slots = latent_positions(model, input_ids)
    if slots and slots[0] == 0:
        raise ValueError("a prompt cannot start with a latent")
    past, start = None, 0
    for p in slots:
        out = model.base_causallm(
            inputs_embeds=embeds[:, start:p],
            position_ids=torch.arange(start, p, device=input_ids.device).unsqueeze(0),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        past = out.past_key_values
        # last layer (after the final layer norm, like Coconut), at position p - 1
        embeds[:, p] = out.hidden_states[-1][:, -1]
        start = p
    return embeds


@torch.no_grad()
def get_embeds(model, input_ids, mode):
    """mode "latent": latent slots get continuous thoughts (S3).
    mode "plain": plain token embeddings everywhere (S5 latent slots, and S1/S2/S4)."""
    if mode == "latent":
        return fill_latents(model, input_ids)
    if mode == "plain":
        return model.embedding(input_ids)
    raise ValueError(f"unknown mode {mode}")


# ---------------------------------------------------------------- decoding


def find_stop(text, stop_strings):
    """Index of the earliest stop string in text, or None."""
    hits = [text.find(s) for s in stop_strings if s in text]
    return min(hits) if hits else None


@torch.no_grad()
def greedy_generate(model, tokenizer, embeds, max_new_tokens, stop_strings=(), use_cache=True):
    """Greedy decoding from input embeddings. The same loop is used for every setting,
    so settings differ only in `embeds`.

    Stops at EOS (not kept), at a stop string (text cut before it), or after
    max_new_tokens. use_cache=False recomputes the whole sequence each step, like
    Coconut.generate.
    """
    base = model.base_causallm
    # GPT-2 defaults; load_qwen sets these on the model
    eos_ids = getattr(model, "eos_ids", {tokenizer.eos_token_id})
    max_positions = getattr(model, "max_positions", MAX_POSITIONS)
    n_prompt = embeds.shape[1]
    if n_prompt + max_new_tokens > max_positions:
        raise ValueError(f"prompt {n_prompt} + {max_new_tokens} new tokens > {max_positions}")
    device = embeds.device

    out = base(inputs_embeds=embeds, use_cache=use_cache)
    past = out.past_key_values if use_cache else None
    seq = embeds
    new_ids, stop_reason = [], "max_new_tokens"
    for step in range(max_new_tokens):
        next_id = int(out.logits[0, -1].argmax())
        if next_id in eos_ids:
            stop_reason = "eos"
            break
        new_ids.append(next_id)
        if stop_strings and find_stop(tokenizer.decode(new_ids), stop_strings) is not None:
            stop_reason = "stop_string"
            break
        if step == max_new_tokens - 1:
            break
        next_embed = model.embedding(torch.tensor([[next_id]], device=device))
        if use_cache:
            out = base(
                inputs_embeds=next_embed,
                position_ids=torch.tensor([[n_prompt + step]], device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
        else:
            seq = torch.cat([seq, next_embed], dim=1)
            out = base(inputs_embeds=seq)

    raw = tokenizer.decode(new_ids)
    cut_at = find_stop(raw, stop_strings)
    return {
        "new_ids": new_ids,
        "raw_text": raw,
        "cut_text": raw if cut_at is None else raw[:cut_at],
        "stop_reason": stop_reason,
        "n_new_tokens": len(new_ids),
    }


# ---------------------------------------------------------------- text and scoring


def clean_text(text):
    """Remove the 3 Coconut marker strings (they are not "special" tokens, plan F3) and
    strip. Returns (text, number of markers removed)."""
    n_removed = 0
    for marker in MARKERS:
        n_removed += text.count(marker)
        text = text.replace(marker, "")
    return text.strip(), n_removed


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def last_number(text):
    """Last number in text, commas removed first ("1,000" -> "1000"), or None."""
    text = text.replace(",", "")
    last = None
    for m in _NUMBER.finditer(text):
        number = m.group()
        if number.startswith("-") and m.start() > 0 and text[m.start() - 1].isalnum():
            number = number[1:]  # "16-3": a minus sign, not a negative number
        last = number
    return last


def gsm8k_gold(answer):
    """Gold answer of an HF GSM8K example: the text after '####', commas removed."""
    return answer.split("####")[-1].strip().replace(",", "")


def same_number(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download GPT-2 + the Coconut checkpoint and check that it loads (CPU)."
    )
    parser.add_argument("--ckpt", default=None, help="local checkpoint file (default: HF Hub)")
    args = parser.parse_args()
    tokenizer, ids = load_tokenizer()
    print("vocab size:", len(tokenizer), "| new token ids:", ids)
    load_model(tokenizer, ids, torch.device("cpu"), args.ckpt)
    print("loaded OK")


if __name__ == "__main__":
    main()

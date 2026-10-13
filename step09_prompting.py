#!/usr/bin/env python3
"""Prompting the five LLMs without fine-tuning on the 200-serial prompting subset.

Conditions: zero-shot, few-shot with k random or k retrieved examples, chain-of-thought or built-in thinking,
few-shot chain-of-thought and self-consistency (five samples). Both GPUs generate in parallel; finished
conditions are skipped when the script is rerun.

Usage:
    python step09_prompting.py --models gemma4-e4b
    python step09_prompting.py --compare
Offline test:
    python step09_prompting.py --dry-run --artifacts-dir outputs --no-install
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

log = logging.getLogger("buc")
SEED = 13
KS = [1, 3, 5, 8]
FINALISTS = ["ministral3-8b", "gemma4-e4b"]
THINKING = {
    "qwen3-8b": "</think>",
    "gemma4-e4b": "<channel|>",
}
BUDGET = {"think": 768, "cot_prompt": 320, "id_answer": 24, "mt_cap": 400, "sc_samples": 5, "sc_temperature": 0.7}
SUBSETS = {"sc_per_label": 50, "cot_mt_serials": 50}
PROMPT_VERSION_P5 = "prompt-v1"
MODELS = {
    "qwen3-8b":      {"repo": "Qwen/Qwen3-8B", "fallback": [], "chat": True, "max_len": 1024, "batch": 4,
                      "backends": ["unsloth"], "eval_batch": 16},
    "llama31-8b":    {"repo": "meta-llama/Llama-3.1-8B-Instruct", "fallback": [], "chat": True, "max_len": 1024,
                      "batch": 4, "backends": ["unsloth"], "eval_batch": 16},
    "ministral3-8b": {"repo": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "fallback": [], "chat": True,
                      "max_len": 512, "batch": 4, "backends": ["hf"], "eval_batch": 16},
    "gemma4-e4b":    {"repo": "google/gemma-4-E4B-it", "fallback": [], "chat": True, "max_len": 512, "batch": 4,
                      "backends": ["unsloth"], "eval_batch": 16},
    "titullm-3b":    {"repo": "hishab/titulm-llama-3.2-3b-v2.0", "fallback": [], "chat": False, "max_len": 512,
                      "batch": 8, "backends": ["unsloth"], "eval_batch": 32},
}

PROMPT_VERSION = "llm-prompt-v2"

SYSTEM = "You are an expert on Bangla and its regional dialects. Follow the requested output format exactly."

DIALECTS = ["nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]

BANGLA = ["std"] + DIALECTS

NAMES = {"std": "Standard Bangla", "nkh": "Noakhali", "ctg": "Chittagong", "syl": "Sylhet", "bar": "Barishal",
         "srj": "Sirajganj", "pab": "Pabna", "shr": "Shariatpur", "odh": "Old Dhaka", "khl": "Khulna",
         "en": "English", "cm": "Code-mixed Bangla-English"}

BN_TOKEN = re.compile(r"[\u0980-\u09FF\u200c\u200d]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def tag(field: str) -> str:
    code, script = field.split(":")
    return f"[{code}-rom]" if script == "rom" else f"[{code}]"


def strip_punct(s: str) -> str:
    import unicodedata
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def pinned(req_file: Path) -> dict[str, str]:
    out = {}
    for line in req_file.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def mismatches(req: dict[str, str]) -> dict:
    bad = {}
    for pkg, ver in req.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        if have != ver:
            bad[pkg] = [ver, have]
    return bad


def prepare_environment(args) -> dict:
    """Install core + GPU requirements (unsloth with --no-deps) before any third-party import."""
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise SystemExit(f"Python 3.12 or 3.13 required, found {sys.version.split()[0]}")
    if sys.version_info[:2] == (3, 13):
        log.info("Python %s: newer Kaggle image (the pinned packages install unchanged; versions are recorded)",
                 sys.version.split()[0])
    report = {}
    core = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt")
    gpu = core.parent / "requirements-gpu.txt" if core else None
    for name, f, extra in (("core", core, []), ("gpu", gpu, ["--no-deps"])):
        if not f or not f.exists():
            log.warning("%s requirements not found; versions not checked", name)
            continue
        req = pinned(f)
        bad = mismatches(req)
        if bad and not args.no_install:
            log.info("installing %s requirements (%d packages differ) ...", name, len(bad))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *extra, "-r", str(f)], check=True)
            bad = mismatches(req)
        report[f"{name}_mismatched"] = bad
        if bad:
            log.warning("%s: %d packages still differ from the lock: %s", name, len(bad), list(bad)[:8])
    return report


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


class Artifacts:
    """step 02 artifacts: aligned table, retrieval neighbours, lexicon, Tier-3 pairs, eval sets."""

    def __init__(self, root: Path):
        import numpy as np
        import pandas as pd
        self.root = root
        self.table = pd.read_parquet(root / "data" / "aligned.parquet")
        self.dd = json.loads((root / "data" / "dd_pairs.json").read_text())
        self.lex = json.loads((root / "lexicon" / "lexicon_default.json").read_text(encoding="utf-8"))
        own = np.load(root / "retrieval" / "own.npz")
        q = own["q_serial"]
        self.row_of = {int(s): i for i, s in enumerate(q)}
        self.own = {k: own[k] for k in own.files if k.endswith("__hybrid__serial")}
        self.pool = {}
        for script in ("bn", "rom"):
            a = np.load(root / "retrieval" / f"all_{script}.npz")
            idx = {(int(s), int(v)): i for i, (s, v) in enumerate(zip(a["q_serial"], a["q_variety"]))}
            self.pool[script] = (idx, a["hybrid__serial"], a["hybrid__variety"], [str(c) for c in a["variety_codes"]])
        self.train_serials = [int(s) for s in self.table.index[self.table["split"] == "train"]]

    def text(self, serial: int, field: str) -> str:
        return self.table.at[serial, field]

    def neighbours_own(self, serial: int, field: str, k: int) -> list[int]:
        arr = self.own[f"{field.replace(':', '__')}__hybrid__serial"][self.row_of[serial]]
        return [int(x) for x in arr if x >= 0][:k]

    def neighbours_auto(self, serial: int, field: str, k: int) -> list[tuple[int, str]]:
        code, script = field.split(":")
        idx, ser, var, codes = self.pool[script]
        i = idx[(serial, BANGLA.index(code))]
        return [(int(s), codes[int(v)]) for s, v in zip(ser[i], var[i]) if s >= 0][:k]


def stop_token_ids(tok) -> list[int]:
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<turn|>", "</s>", "<|end_of_text|>", "<|endoftext|>"):
        try:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                ids.add(i)
        except Exception:
            pass
    return sorted(ids)


def json_close_ids(tok) -> list[int]:
    """Token ids whose text contains '}' -- generation of a sequence stops once its answer object is closed."""
    ids = []
    for i in range(len(tok)):
        try:
            if "}" in tok.decode([i]):
                ids.append(i)
        except Exception:
            pass
    return ids


def json_stopper(close, prompt_len: int):
    """Per-sequence stopping criterion: a sequence is finished as soon as it emits a token containing '}' (`close`: tensor of such token ids, computed once per evaluation)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class JsonClosed(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[1] <= prompt_len:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return torch.isin(input_ids[:, -1], close)
    return StoppingCriteriaList([JsonClosed()])


def norm_tag(v) -> str | None:
    if not isinstance(v, str):
        return None
    m = re.search(r"\[?([a-z]{2,3})(?:-rom)?\]?", v.strip().lower())
    return m.group(1) if m else None


def clear_model_cache(repo: str) -> None:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for d in hub.glob("models--" + repo.replace("/", "--")):
        shutil.rmtree(d, ignore_errors=True)


NAMES_ALL = {**NAMES, "en": "English", "cm": "code-mixed Bangla"}
ID_INSTR = ("Identify the Bangla variety of the input sentence. Possible varieties: "
            + ", ".join(f"[{c}] {NAMES[c]}" for c in BANGLA)
            + ". Romanized input uses the same varieties. Answer only with JSON.")
MT_INSTR = "Translate the input from the source variety to the target variety. Answer only with JSON."
ID_COT = ("First explain briefly which words, endings or spellings indicate the variety, "
          "then give the JSON answer on the last line.")
MT_COT = ("First note briefly which words or forms differ between the source and the target variety, "
          "then give the JSON answer on the last line.")
JSON_ANY = re.compile(r"\{[^{}]*\}", re.S)


def stable_rng(*parts) -> random.Random:
    """Deterministic RNG per item (Python's hash() is randomized between runs)."""
    return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def variety_label(field: str) -> str:
    code, script = field.split(":")
    name = NAMES_ALL.get(code, code)
    return f"{tag(field)} ({name}{', romanized' if script == 'rom' else ''})"


def load_markers(root: Path) -> dict:
    """Top dialect marker words (step 02 analysis, training split) used for few-shot CoT rationales."""
    import pandas as pd
    f = root / "analysis" / "markers_vs_rest.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    df = df[df["unit"] == "word"]
    return {d: set(g["item"].astype(str)) for d, g in df.groupby("dialect")}


def rationale(text: str, code: str, markers: dict) -> str:
    if code == "std":
        return "No dialect-specific words or endings; the sentence follows Standard Bangla."
    cues = [w for w in dict.fromkeys(BN_TOKEN.findall(text)) if w in markers.get(code, set())][:3]
    if cues:
        return f"The words {', '.join(cues)} are typical of {NAMES[code]}."
    return f"The spelling and vocabulary match {NAMES[code]}."


def id_examples(art, item: dict, regime: str, k: int) -> list[tuple[str, str]]:
    """(text without punctuation, variety code) pairs; retrieved: most similar last."""
    script = item["script"]
    if regime in ("fsretr", "fscot"):
        pairs = art.neighbours_auto(item["serial"], f"{item['label']}:{script}", k)[::-1]
    else:
        rng = stable_rng("id", item["id"], k, SEED)
        pairs = [(rng.choice(art.train_serials), rng.choice(BANGLA)) for _ in range(k)]
    return [(strip_punct(art.text(s, f"{c}:{script}")), c) for s, c in pairs]


def mt_examples(art, item: dict, regime: str, k: int) -> list[tuple[str, str]]:
    src, tgt = item["src_field"], item["tgt_field"]
    if regime == "fsretr":
        serials = art.neighbours_own(item["serial"], src, k)[::-1]
    else:
        rng = stable_rng("mt", item["id"], k, SEED)
        serials = rng.sample(art.train_serials, k)
    return [(art.text(s, src), art.text(s, tgt)) for s in serials]


def build(art, item: dict, task: str, regime: str, k: int, key: str, markers: dict) -> tuple[str, str]:
    """(prompt text, answer prefix)."""
    base = not MODELS[key]["chat"]
    thinking = regime in ("cot", "sc5") and key in THINKING
    cot_prompt = regime in ("cot", "sc5") and not thinking
    if task.startswith("id"):
        lines = [ID_INSTR, 'Output format: {"variety": "<tag>"}']
        if regime == "fscot":
            lines.append("Give a one-sentence reasoning before the answer.")
        elif cot_prompt:
            lines.append(ID_COT)
        exs = id_examples(art, item, regime, k) if k else []
        if exs:
            lines.append("Examples:")
            for text, code in exs:
                ans = json.dumps({"variety": f"[{code}]"})
                if regime == "fscot":
                    lines.append(f"Input: {text}\nReasoning: {rationale(text, code, markers)}\nOutput: {ans}")
                else:
                    lines.append(f"Input: {text}\nOutput: {ans}")
        lines.append(f"Input: {item['text_nopunct']}")
        if regime == "fscot":
            lines.append("Reasoning:")
            return "\n".join(lines), ""
        if base:
            return "\n".join(lines + ['Output: {"variety": "[']), '{"variety": "['
        return "\n".join(lines + ([] if cot_prompt or thinking else ["Output:"])), ""
    src, tgt = item["src_field"], item["tgt_field"]
    script_note = "Write the translation in English." if tgt == "en:lat" else "Write the translation in Bangla script."
    lines = [MT_INSTR, f"Source: {variety_label(src)}   Target: {variety_label(tgt)}", script_note,
             'Output format: {"translation": "<text>"}']
    if cot_prompt:
        lines.append(MT_COT)
    exs = mt_examples(art, item, regime, k) if k else []
    if exs:
        lines.append("Examples:")
        for s_text, t_text in exs:
            lines.append(f"Input: {s_text}\nOutput: {json.dumps({'translation': t_text}, ensure_ascii=False)}")
    lines.append(f"Input: {item['src']}")
    if base:
        return "\n".join(lines + ['Output: {"translation": "']), '{"translation": "'
    return "\n".join(lines + ([] if cot_prompt or thinking else ["Output:"])), ""


def prompt_ids(tok, proc, chat: bool, prompt: str, thinking: bool) -> list[int]:
    """Chat template with a fixed short system message (as in QLoRA); thinking switched on or off explicitly."""
    if not chat:
        return tok((tok.bos_token or "") + prompt, add_special_tokens=False)["input_ids"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    last = None
    for m in (msgs, [dict(msgs[1], content=SYSTEM + "\n\n" + prompt)]):
        for renderer, parts in ((tok, False), (proc, True)):
            try:
                mm = [{"role": x["role"], "content": [{"type": "text", "text": x["content"]}]} for x in m] if parts else m
                return tok(renderer.apply_chat_template(mm, **kw), add_special_tokens=False)["input_ids"]
            except Exception as e:
                last = e
    raise last


def parse_last(text: str) -> tuple[dict, bool]:
    """The last JSON object (reasoning may precede it); lenient key fallbacks count as invalid."""
    for m in reversed(JSON_ANY.findall(text)):
        try:
            return json.loads(m), True
        except Exception:
            continue
    out = {}
    for key in ("variety", "translation"):
        hits = re.findall(r'"%s"\s*:\s*"([^"]*)' % key, text)
        if hits:
            out[key] = hits[-1]
    return out, False


def id_pred(parsed: dict) -> str | None:
    p = norm_tag(parsed.get("variety"))
    return p if p in BANGLA else None


def conditions(key: str) -> list[tuple[str, str, int]]:
    """(task, regime, k), in priority order: all models first, then the finalists' k-sweep and Tier 2."""
    c = [("id_bn", "zs", 0), ("id_bn", "fsrand", 3), ("id_bn", "fsretr", 3),
         ("mt_t1", "zs", 0), ("mt_t1", "fsrand", 3), ("mt_t1", "fsretr", 3),
         ("id_rom", "zs", 0), ("id_rom", "fsretr", 3)]
    if MODELS[key]["chat"]:
        c += [("id_bn", "cot", 0), ("id_bn", "fscot", 3), ("mt_t1", "cot", 0), ("id_bn", "sc5", 0)]
    if key in FINALISTS:
        for k in (1, 5, 8):
            c += [("id_bn", "fsrand", k), ("id_bn", "fsretr", k), ("mt_t1", "fsrand", k), ("mt_t1", "fsretr", k)]
        c += [("mt_t2", "zs", 0), ("mt_t2", "fsretr", 3)]
    return c


def cond_name(task: str, regime: str, k: int) -> str:
    return f"{task}-{regime}" + (f"-k{k}" if k else "")


def selected_conditions(key: str, only: str | None, skip: str | None) -> list[tuple[str, str, int]]:
    """conditions(key), optionally restricted with --only / --skip (comma-separated condition names)."""
    names_only = {x.strip() for x in only.split(",")} if only else None
    names_skip = {x.strip() for x in skip.split(",")} if skip else set()
    out = [c for c in conditions(key) if (names_only is None or cond_name(*c) in names_only) and cond_name(*c) not in names_skip]
    unknown = (names_only or set()) | names_skip
    unknown -= {cond_name(*c) for c in conditions(key)}
    if unknown:
        raise SystemExit(f"unknown condition(s) for {key}: {sorted(unknown)}; valid: {[cond_name(*c) for c in conditions(key)]}")
    return out


def load_items(root: Path) -> dict:
    d = root / "data" / "eval" / "test_prompt"
    ids = [json.loads(l) for l in open(d / "id.jsonl", encoding="utf-8")]
    mts = [json.loads(l) for l in open(d / "mt.jsonl", encoding="utf-8")]
    bn = sorted([r for r in ids if r["script"] == "bn"], key=lambda r: r["id"])
    per, sc = Counter(), []
    for r in bn:
        if per[r["label"]] < SUBSETS["sc_per_label"]:
            sc.append(r["id"])
            per[r["label"]] += 1
    t1 = [r for r in mts if r["tier"] == "tier1"]
    cot_serials = sorted({r["serial"] for r in t1})[:SUBSETS["cot_mt_serials"]]
    return {"id_bn": bn, "id_rom": [r for r in ids if r["script"] == "rom"], "mt_t1": t1,
            "mt_t2": [r for r in mts if r["tier"] == "tier2"],
            "sc_subset": set(sc), "cot_subset": {r["id"] for r in t1 if r["serial"] in set(cot_serials)}}


def items_for(items: dict, task: str, regime: str) -> list[dict]:
    if task == "id_bn" and regime == "sc5":
        return [r for r in items["id_bn"] if r["id"] in items["sc_subset"]]
    if task == "mt_t1" and regime == "cot":
        return [r for r in items["mt_t1"] if r["id"] in items["cot_subset"]]
    return items[task]


def load_model(key: str, token, local_rank: int):
    """4-bit NF4 inference, same checkpoints and backends as for fine-tuning (step 04)."""
    import torch
    cfg = MODELS[key]
    if cfg["backends"][0] == "unsloth":
        from unsloth import FastModel
        kw = dict(model_name=cfg["repo"], max_seq_length=4096, dtype=torch.float16, load_in_4bit=True,
                  full_finetuning=False, token=token, device_map={"": local_rank})
        try:
            model, proc = FastModel.from_pretrained(use_exact_model_name=True, **kw)
        except TypeError:
            model, proc = FastModel.from_pretrained(**kw)
        FastModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        kw = dict(quantization_config=bnb, dtype=torch.float16, device_map={"": local_rank}, token=token)
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["repo"], **kw)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(cfg["repo"], **kw)
        try:
            proc = AutoTokenizer.from_pretrained(cfg["repo"], token=token)
        except Exception:
            proc = AutoProcessor.from_pretrained(cfg["repo"], token=token)
    model.eval()
    return model, proc


def _pad_left(seqs, pad_id, device):
    import torch
    m = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for j, s in enumerate(seqs):
        inp[j, m - len(s):] = torch.tensor(s)
        att[j, m - len(s):] = 1
    return inp.to(device), att.to(device), m


def generate_batch(model, tok, seqs, max_new, stops, pad_id, device, close_ids=None, sample=None):
    """Greedy (or sampled) generation; returns the generated token ids per output sequence."""
    import torch
    inp, att, m = _pad_left(seqs, pad_id, device)
    kw = dict(max_new_tokens=max_new, eos_token_id=stops, pad_token_id=pad_id)
    if sample:
        kw.update(do_sample=True, temperature=sample["temperature"], top_p=1.0, top_k=0,
                  num_return_sequences=sample["n"])
    else:
        kw.update(do_sample=False)
    if close_ids is not None:
        kw["stopping_criteria"] = json_stopper(close_ids, m)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(input_ids=inp, attention_mask=att, **kw)
    return [row[m:].tolist() for row in out]


def strip_pad(ids, pad_id, stops):
    out = []
    for t in ids:
        if t == pad_id or t in stops:
            break
        out.append(t)
    return out


def run_condition(model, proc, art, key, task, regime, k, items, rank, world, out_path: Path, markers) -> None:
    """Generate this rank's share of one condition; one JSON line per item (raw output, parse, prediction)."""
    import torch
    tok = getattr(proc, "tokenizer", proc)
    chat = bool(MODELS[key]["chat"]) and bool(getattr(tok, "chat_template", None) or getattr(proc, "chat_template", None))
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = stop_token_ids(tok)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stops = sorted(set(stops) | set(gen_eos if isinstance(gen_eos, list) else ([gen_eos] if gen_eos is not None else [])))
    close_ids = torch.tensor(json_close_ids(tok), device=device)
    thinking = regime in ("cot", "sc5") and key in THINKING
    cot_prompt = regime in ("cot", "sc5") and not thinking
    marker = THINKING.get(key)
    marker_id = tok.convert_tokens_to_ids(marker) if marker else None
    if marker_id is not None and (not isinstance(marker_id, int) or marker_id == tok.unk_token_id):
        marker_id = None
    torch.manual_seed(SEED + rank)
    mine = items[rank::world]
    prepared = []
    for it in mine:
        prompt, prefix = build(art, it, task, regime, k, key, markers)
        ids = prompt_ids(tok, proc, chat, prompt, thinking)
        if task.startswith("id"):
            ans = BUDGET["id_answer"] + (96 if regime == "fscot" else 0)
        else:
            src_len = len(tok(it["src"], add_special_tokens=False)["input_ids"])
            ans = min(BUDGET["mt_cap"], 2 * src_len + 48)
        if cot_prompt:
            ans += BUDGET["cot_prompt"]
        prepared.append((it, ids, prefix, ans))
    prepared.sort(key=lambda x: (x[3], len(x[1])))
    n_samples = BUDGET["sc_samples"] if regime == "sc5" else 1
    base_bs = MODELS[key]["eval_batch"]
    bs = max(1, base_bs // n_samples) if regime == "sc5" else (max(4, base_bs // 2) if thinking or cot_prompt else base_bs)
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for b0 in range(0, len(prepared), bs):
            chunk = prepared[b0:b0 + bs]
            seqs = [x[1] for x in chunk]
            ans_budget = max(x[3] for x in chunk)
            sample = {"temperature": BUDGET["sc_temperature"], "n": n_samples} if regime == "sc5" else None
            if thinking:
                gens = generate_batch(model, tok, seqs, BUDGET["think"] + ans_budget, stops, pad_id, device,
                                      None, sample)
            else:
                gens = generate_batch(model, tok, seqs, ans_budget, stops, pad_id, device, close_ids, sample)
            forced = [False] * len(gens)
            if thinking:
                redo = []
                for gi, g in enumerate(gens):
                    g = strip_pad(g, pad_id, stops) if marker_id is None else g
                    has_end = (marker_id in g) if marker_id is not None else (marker in tok.decode(g))
                    if not has_end:
                        redo.append(gi)
                if redo:
                    end_ids = tok(marker + "\n", add_special_tokens=False)["input_ids"]
                    cont = []
                    for gi in redo:
                        src_seq = seqs[gi // n_samples]
                        body = [t for t in gens[gi] if t != pad_id and t not in stops]
                        cont.append(src_seq + body + end_ids)
                    extra = generate_batch(model, tok, cont, ans_budget, stops, pad_id, device, close_ids)
                    for gi, e in zip(redo, extra):
                        body = [t for t in gens[gi] if t != pad_id and t not in stops]
                        gens[gi] = body + end_ids + e
                        forced[gi] = True
            for j, (it, _, prefix, _) in enumerate(chunk):
                outs = []
                for s_ in range(n_samples):
                    gi = j * n_samples + s_
                    g = strip_pad(gens[gi], pad_id, stops) if not thinking else gens[gi]
                    full = tok.decode(g, skip_special_tokens=False)
                    think_len = None
                    if thinking:
                        if marker_id is not None and marker_id in g:
                            cut = len(g) - 1 - g[::-1].index(marker_id)
                            think_len, answer_ids = cut, [t for t in g[cut + 1:] if t != pad_id and t not in stops]
                            answer = tok.decode(answer_ids, skip_special_tokens=True)
                        else:
                            answer = full.split(marker)[-1] if marker in full else full
                            answer = re.sub(r"<[^<>]{1,40}>", "", answer)
                    else:
                        answer = tok.decode(g, skip_special_tokens=True)
                    text = prefix + answer.strip()
                    parsed, valid = parse_last(text)
                    outs.append({"text": text[-2000:], "valid": valid, "parsed": parsed, "think_tokens": think_len,
                                 "forced": forced[gi], "gen_tokens": len(g)})
                rec = {"id": it["id"], "condition": cond_name(task, regime, k), "valid_json": outs[0]["valid"],
                       "forced": any(o["forced"] for o in outs), "gen_tokens": outs[0]["gen_tokens"],
                       "think_tokens": outs[0]["think_tokens"], "raw": outs[0]["text"]}
                if task.startswith("id"):
                    labels = [id_pred(o["parsed"]) for o in outs]
                    if regime == "sc5":
                        votes = Counter(l for l in labels if l)
                        pred = max(votes, key=lambda l: (votes[l], -labels.index(l))) if votes else None
                        rec.update(samples=labels, valid_json=any(o["valid"] for o in outs))
                    else:
                        pred = labels[0]
                    rec.update(label=it["label"], script=it["script"], pred=pred)
                else:
                    rec.update(tier=it["tier"], src_field=it["src_field"], tgt_field=it["tgt_field"], src=it["src"],
                               ref=it["ref"], hyp=(outs[0]["parsed"].get("translation") or ""))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rank == 0 and (b0 // bs) % 25 == 0:
                log.info("  [%s] %d/%d items (%.0f s)", cond_name(task, regime, k), b0 + len(chunk), len(prepared),
                         time.time() - t0)


def score_records(recs: list[dict], task: str, subset: set | None = None) -> dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    if subset is not None:
        recs = [r for r in recs if r["id"] in subset]
    if not recs:
        return {}
    res = {"n": len(recs), "valid_rate": round(100 * float(np.mean([r["valid_json"] for r in recs])), 2),
           "forced_rate": round(100 * float(np.mean([r.get("forced", False) for r in recs])), 2),
           "mean_gen_tokens": round(float(np.mean([r.get("gen_tokens") or 0 for r in recs])), 1)}
    if task.startswith("id"):
        y = [r["label"] for r in recs]
        p = [r["pred"] if r["pred"] in BANGLA else "invalid" for r in recs]
        res.update(accuracy=round(100 * accuracy_score(y, p), 2),
                   macro_f1=round(100 * f1_score(y, p, labels=BANGLA, average="macro", zero_division=0), 2),
                   per_label_f1=dict(zip(BANGLA, [round(100 * v, 2) for v in
                                                  f1_score(y, p, labels=BANGLA, average=None, zero_division=0)])))
        return res
    from sacrebleu.metrics import CHRF
    chrfpp = CHRF(word_order=2)
    per = {}
    for d in sorted({(r["src_field"], r["tgt_field"]) for r in recs}):
        sub = [r for r in recs if (r["src_field"], r["tgt_field"]) == d]
        per[f"{d[0]}>{d[1]}"] = round(chrfpp.corpus_score([r["hyp"] for r in sub], [[r["ref"] for r in sub]]).score, 2)
    mean = lambda f: round(float(np.mean([v for k_, v in per.items() if f(k_.split(">"))])), 2) if per else None
    res.update(per_direction=per, chrfpp_mean=mean(lambda d: True),
               d2s=mean(lambda d: d[1] == "std:bn" and d[0] != "en:lat" and d[0] != "cm:mix"),
               s2d=mean(lambda d: d[0] == "std:bn" and d[1] != "en:lat"), signature=str(chrfpp.get_signature()))
    return res


def done_metrics(out_dir: Path) -> bool:
    f = out_dir / "metrics.json"
    try:
        return f.exists() and json.loads(f.read_text()).get("status") == "ok"
    except Exception:
        return False


def worker(args) -> None:
    import torch
    import torch.distributed as dist
    local_rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
    rank0 = local_rank == 0
    key = args.model
    root = Path(args.artifacts_dir)
    base = Path(args.output_dir, "step5", key)
    t_start = float(os.environ.get("BUC_T0", time.time()))
    try:
        art = Artifacts(root)
        items = load_items(root)
        markers = load_markers(root)
        model, proc = load_model(key, os.environ.get("HF_TOKEN") or None, local_rank)
        for task, regime, k in selected_conditions(key, args.only, args.skip):
            name = cond_name(task, regime, k)
            out = base / name
            go = torch.tensor([0 if done_metrics(out) else 1,
                               1 if (time.time() - t_start) / 3600 < args.time_budget_h else 0], device=f"cuda:{local_rank}")
            if world > 1:
                dist.broadcast(go, src=0)
            if go[0] == 0:
                continue
            if go[1] == 0:
                if rank0:
                    log.warning("time budget reached: %s and later conditions left for the next session", name)
                break
            out.mkdir(parents=True, exist_ok=True)
            its = items_for(items, task, regime)
            t0 = time.time()
            if rank0:
                log.info("===== %s: %s (%d items) =====", key, name, len(its))
            run_condition(model, proc, art, key, task, regime, k, its, local_rank, world, out / f"rank{local_rank}.jsonl",
                          markers)
            if world > 1:
                dist.barrier()
            if rank0:
                recs = [json.loads(l) for p in sorted(out.glob("rank*.jsonl")) for l in open(p, encoding="utf-8")]
                m = {"status": "ok", "model": key, "condition": name, "task": task, "regime": regime, "k": k,
                     "thinking": regime in ("cot", "sc5") and key in THINKING, "minutes": round((time.time() - t0) / 60, 1),
                     "prompt_version": PROMPT_VERSION_P5, "budget": BUDGET, "full": score_records(recs, task)}
                if task == "id_bn":
                    m["sc_subset"] = score_records(recs, task, items["sc_subset"])
                if task == "mt_t1":
                    m["cot_subset"] = score_records(recs, task, items["cot_subset"])
                (out / "metrics.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
                f = m["full"]
                log.info("%s %s: %s (valid %.1f%%, %.1f min)", key, name,
                         f"macro-F1 {f.get('macro_f1')}" if task.startswith("id") else f"chrF++ {f.get('chrfpp_mean')}",
                         f.get("valid_rate", 0), m["minutes"])
    except Exception as e:
        if rank0:
            log.error("worker failed: %s\n%s", e, traceback.format_exc()[-3000:])
            (base / f"error_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.txt").parent.mkdir(parents=True, exist_ok=True)
            (base / f"error_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.txt").write_text(traceback.format_exc())
    finally:
        if world > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def restore_previous(args) -> None:
    """Copy finished conditions of earlier (attached) sessions so they are not repeated."""
    for f in glob.glob("/kaggle/input/**/step5/*/*/metrics.json", recursive=True):
        rel = f.split("/step5/", 1)[1]
        dst = Path(args.output_dir, "step5", rel).parent
        if not (dst / "metrics.json").exists():
            shutil.copytree(Path(f).parent, dst, dirs_exist_ok=True)


def launch(args) -> None:
    import signal
    import torch
    tok = hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise SystemExit("no GPU: set Accelerator to 'GPU T4 x2'")
    restore_previous(args)
    os.environ["BUC_T0"] = str(time.time())
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", "--master_port=29543",
               str(Path(__file__).resolve()), "--worker", "--model", key, "--artifacts-dir", args.artifacts_dir,
               "--output-dir", args.output_dir, "--time-budget-h", str(args.time_budget_h)]
        cmd += (["--only", args.only] if args.only else []) + (["--skip", args.skip] if args.skip else [])
        log.info("===== model %s: %d conditions in this session =====", key, len(selected_conditions(key, args.only, args.skip)))
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            proc.wait(timeout=(args.time_budget_h + 1.5) * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        n_done = sum(done_metrics(Path(args.output_dir, "step5", key, cond_name(*c))) for c in conditions(key))
        log.info("%s: %d/%d conditions finished", key, n_done, len(conditions(key)))
        clear_model_cache(MODELS[key]["repo"])
    compare(args)


def compare(args) -> None:
    import pandas as pd
    files = set(glob.glob(f"{args.output_dir}/step5/*/*/metrics.json") +
                glob.glob("/kaggle/input/**/step5/*/*/metrics.json", recursive=True))
    rows, seen = [], set()
    for f in sorted(files):
        m = json.loads(Path(f).read_text())
        if m.get("status") != "ok" or (m["model"], m["condition"]) in seen:
            continue
        seen.add((m["model"], m["condition"]))
        fm, is_id = m["full"], m["task"].startswith("id")
        row = {"model": m["model"], "condition": m["condition"], "task": m["task"], "regime": m["regime"], "k": m["k"],
               "thinking": m.get("thinking"), "n": fm.get("n"), "score": fm.get("macro_f1" if is_id else "chrfpp_mean"),
               "accuracy": fm.get("accuracy"), "d2s": fm.get("d2s"), "s2d": fm.get("s2d"), "valid": fm.get("valid_rate"),
               "forced": fm.get("forced_rate"), "gen_tokens": fm.get("mean_gen_tokens"), "minutes": m.get("minutes")}
        if "sc_subset" in m:
            row["score_sc_subset"] = m["sc_subset"].get("macro_f1")
        if "cot_subset" in m:
            row["score_cot_subset"] = m["cot_subset"].get("chrfpp_mean")
        rows.append(row)
    if not rows:
        log.warning("no step 09 results found")
        return
    df = pd.DataFrame(rows).sort_values(["model", "task", "regime", "k"])
    out = Path(args.output_dir, "step5")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "prompting_summary.csv", index=False)
    get = lambda mdl, cond, col="score": (df[(df.model == mdl) & (df.condition == cond)][col].tolist() or [None])[0]
    h3, h4 = [], []
    for mdl in sorted(df.model.unique()):
        for task in ("id_bn", "mt_t1"):
            for k in KS:
                r, q = get(mdl, cond_name(task, "fsrand", k)), get(mdl, cond_name(task, "fsretr", k))
                if r is not None and q is not None:
                    h3.append({"model": mdl, "task": task, "k": k, "random": r, "retrieved": q, "gain": round(q - r, 2)})
        zs, cot = get(mdl, "id_bn-zs"), get(mdl, "id_bn-cot")
        if zs is not None and cot is not None:
            h4.append({"model": mdl, "task": "identification", "items": "all 2000", "no_cot": zs, "cot": cot,
                       "gain": round(cot - zs, 2)})
        zs_s, sc = get(mdl, "id_bn-zs", "score_sc_subset"), get(mdl, "id_bn-sc5")
        if zs_s is not None and sc is not None:
            h4.append({"model": mdl, "task": "identification (self-consistency)", "items": "500 subset", "no_cot": zs_s,
                       "cot": sc, "gain": round(sc - zs_s, 2)})
        zs_m, cot_m = get(mdl, "mt_t1-zs", "score_cot_subset"), get(mdl, "mt_t1-cot")
        if zs_m is not None and cot_m is not None:
            h4.append({"model": mdl, "task": "Tier-1 translation", "items": "900 subset", "no_cot": zs_m, "cot": cot_m,
                       "gain": round(cot_m - zs_m, 2)})
    pd.DataFrame(h3).to_csv(out / "h3_retrieved_vs_random.csv", index=False)
    pd.DataFrame(h4).to_csv(out / "h4_cot.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 300):
        log.info("STEP 09 SUMMARY\n%s", df[["model", "condition", "n", "score", "valid", "forced", "minutes"]].to_string(index=False))
        if h3:
            log.info("H3 (retrieved - random)\n%s", pd.DataFrame(h3).to_string(index=False))
        if h4:
            log.info("H4 (CoT - no CoT)\n%s", pd.DataFrame(h4).to_string(index=False))


def dry_run(args) -> None:
    root = Path(args.artifacts_dir)
    art, items, markers = Artifacts(root), load_items(root), load_markers(root)
    log.info("items: id_bn %d, id_rom %d, mt_t1 %d, mt_t2 %d; sc subset %d, translation-CoT subset %d",
             len(items["id_bn"]), len(items["id_rom"]), len(items["mt_t1"]), len(items["mt_t2"]),
             len(items["sc_subset"]), len(items["cot_subset"]))
    for key in MODELS:
        log.info("%-14s %d conditions: %s", key, len(conditions(key)), ", ".join(cond_name(*c) for c in conditions(key)))
    it_id, it_mt = items["id_bn"][0], items["mt_t1"][0]
    for key, task, regime, k, it in (("gemma4-e4b", "id_bn", "fscot", 3, it_id), ("llama31-8b", "id_bn", "cot", 0, it_id),
                                      ("titullm-3b", "id_bn", "zs", 0, it_id), ("ministral3-8b", "mt_t1", "fsretr", 3, it_mt),
                                      ("qwen3-8b", "mt_t1", "cot", 0, it_mt)):
        p, prefix = build(art, it, task, regime, k, key, markers)
        log.info("---- %s %s:\n%s%s", key, cond_name(task, regime, k), p, f"  [prefix {prefix!r}]" if prefix else "")
    for text in ('reasoning ... {"variety": "[ctg]"}', 'I think {"variety": "[nkh]"} no wait {"variety": "[ctg]"}',
                 '{"translation": "আমি'):
        log.info("parse_last(%r) -> %s", text, parse_last(text))
    recs = [{"id": r["id"], "label": r["label"], "script": "bn", "pred": r["label"] if i % 3 else "ctg", "valid_json": True}
            for i, r in enumerate(items["id_bn"])]
    log.info("scoring check: %s", {k_: v for k_, v in score_records(recs, "id_bn").items() if k_ != "per_label_f1"})


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 09: prompting experiments")
    p.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {list(MODELS)}")
    p.add_argument("--compare", action="store_true", help="summarize all finished conditions (CPU)")
    p.add_argument("--time-budget-h", type=float, default=10.5, help="no new condition after this many hours")
    p.add_argument("--only", help="run only these conditions (comma-separated names, e.g. id_bn-sc5)")
    p.add_argument("--skip", help="skip these conditions (comma-separated names)")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if args.compare:
        compare(args)
        return
    if not args.artifacts_dir:
        hit = find_one("/kaggle/input/**/data/aligned.parquet")
        if hit is None:
            raise SystemExit("step-02 output not found under /kaggle/input: attach the output of step02_prepare_data.py")
        args.artifacts_dir = str(hit.parents[1])
    if args.worker:
        worker(args)
        return
    if args.dry_run:
        dry_run(args)
        return
    unknown = set(args.models.split(",")) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(MODELS)}")
    prepare_environment(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    launch(args)


if __name__ == "__main__":
    main()

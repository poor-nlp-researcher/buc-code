#!/usr/bin/env python3
"""QLoRA fine-tuning of an LLM for dialect identification and translation (one shared adapter).

Presets: select (model selection on 20% of the training data), search (hyperparameter search), transfer
(learning-rate transfer to the other LLMs), final, primary and base (full training of Llama-3.1, Ministral-3,
Qwen3 and TituLLM-3B with three seeds). Training and evaluation use both GPUs (torchrun, data parallel);
interrupted runs resume from the last checkpoint.

Usage:
    python step05_qlora_llms.py --preset select --models qwen3-8b,gemma4-e4b
    python step05_qlora_llms.py --compare
    python step05_qlora_llms.py --preset base --models titullm-3b --seed 13 --lr 0.000202 --lora-r 32 \
        --lora-alpha 64 --lora-dropout 0.05 --targets all --eff-batch 16
Offline test of the data and prompt code:
    python step05_qlora_llms.py --dry-run --artifacts-dir outputs --no-install
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
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SEED = 13
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
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
LORA = {"r": 16, "alpha": 32, "dropout": 0.0, "targets": ATTN + MLP}
TRAIN = {"lr": 2e-4, "warmup_frac": 0.03, "max_grad_norm": 1.0, "weight_decay": 0.0, "eff_batch": 32}
MIXTURE = [("id_bn", 2), ("id_rom", 1), ("tier1", 3), ("tier2", 1), ("tier3", 1)]
SAMPLING = {"p_examples": 0.5, "k_max": 3, "p_hints": 0.5, "max_hints": 5, "p_auto_source": 0.5}
EVAL = {"k": 3, "hints": True, "max_new_cap": 400}
PROMPT_VERSION = "llm-prompt-v2"
SYSTEM = "You are an expert on Bangla and its regional dialects. Follow the requested output format exactly."
PRESETS = {
    "select": {"train_fraction": 0.2, "epochs": 1, "lr": 2e-4, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
               "targets": "all", "eff_batch": 32, "eval_set": "dev_select", "eval_tasks": "id,tier1"},
}
PRESETS["search"] = {"train_fraction": 0.2, "epochs": 1, "eval_set": "dev_select", "eval_tasks": "id,tier1",
                     "ckpt_minutes": 1e9, "no_save": True}
SEARCH_SPACE = {"lr": (5e-5, 4e-4),
                "lora_r": [8, 16, 32, 64], "alpha_mult": [1, 2], "lora_dropout": [0.0, 0.05],
                "targets": ["attn", "all"], "eff_batch": [16, 32]}
SEARCH = {"n_global": 16, "n_local": 8, "sobol_seed": 13, "id_tolerance": 2.0, "boundary_min_gain": 0.5,
          "lr_steps": (0.6, 1.6)}
SCREEN_ID_F1 = {"ministral3-8b": 62.11, "gemma4-e4b": 59.23, "llama31-8b": 60.37, "qwen3-8b": 59.77, "titullm-3b": 52.35}
TRIAL_HOURS = {"ministral3-8b": 1.6, "gemma4-e4b": 1.3, "llama31-8b": 2.1, "qwen3-8b": 1.9, "titullm-3b": 0.9}

PRESETS["final"] = {"train_fraction": 1.0, "epochs": 2, "eval_set": "dev_full", "eval_tasks": "id,tier1",
                    "epoch_select": True, "ckpt_minutes": 45, "resume": True}
SELECTED_CONFIGS = {
    "ministral3-8b": {"lr": 1.53e-4, "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.0, "targets": "all", "eff_batch": 32},
    "gemma4-e4b": {"lr": 2.02e-4, "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "targets": "all", "eff_batch": 16},
}
EVAL_HOURS = {"dev_select": 0.5, "dev_full": 1.8, "test_full": 4.2}
PRIMARY_MODEL = "gemma4-e4b"
PRESETS["primary"] = {"train_fraction": 1.0, "epochs": 4, "eval_set": "dev_select", "eval_tasks": "id,tier1",
                      "epoch_select": True, "ckpt_minutes": 45, "resume": True,
                      "final_evals": "dev_full:id,tier1;test_full:id,tier1,tier2,tier3,auto"}
FINAL_RULE = ("higher mean Tier-1 chrF++ on dev_full if the paired bootstrap 95% CI of the difference excludes zero; "
              "otherwise higher identification macro-F1 (Bangla script) if its CI excludes zero; otherwise the cheaper "
              "model (Gemma-4-E4B)")
BOOTSTRAP = {"n": 1000, "seed": 13}

TRANSFER = {"source": "gemma4-e4b", "factors": (0.5, 1.0, 2.0), "edge_step": 2.0,
            "models": ["qwen3-8b", "llama31-8b", "titullm-3b"]}
PRESETS["transfer"] = {"train_fraction": 0.2, "epochs": 1, "eval_set": "dev_select", "eval_tasks": "id,tier1",
                       "ckpt_minutes": 1e9, "no_save": True}
PRESETS["base"] = {"train_fraction": 1.0, "epochs": 4, "eval_set": "dev_select", "eval_tasks": "id,tier1",
                   "epoch_select": True, "ckpt_minutes": 45, "resume": True,
                   "final_evals": "dev_full:id,tier1;test_full:id,tier1"}
BASE_MODELS = ["ministral3-8b", "qwen3-8b", "llama31-8b", "titullm-3b"]

SELECTION_RULE = ("highest mean Tier-1 chrF++ on dev_select; if the top two are within 1.0 chrF++, higher "
                  "identification macro-F1 (Bangla script) wins; if that is also within 1.0 point, the model with "
                  "the lower training cost wins")
log = logging.getLogger("buc")

DIALECTS = ["nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
BANGLA = ["std"] + DIALECTS
NAMES = {"std": "Standard Bangla", "nkh": "Noakhali", "ctg": "Chittagong", "syl": "Sylhet", "bar": "Barishal",
         "srj": "Sirajganj", "pab": "Pabna", "shr": "Shariatpur", "odh": "Old Dhaka", "khl": "Khulna",
         "en": "English", "cm": "Code-mixed Bangla-English"}


def tag(field: str) -> str:
    code, script = field.split(":")
    return f"[{code}-rom]" if script == "rom" else f"[{code}]"


def strip_punct(s: str) -> str:
    import unicodedata
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


BN_TOKEN = re.compile(r"[\u0980-\u09FF\u200c\u200d]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


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


CPU_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "sacrebleu")


def ensure_cpu_packages(args) -> None:
    """The comparison modes (--compare, --compare-search, --compare-final) run on CPU notebooks."""
    if args.no_install:
        return
    req = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt")
    pins = pinned(req) if req and req.exists() else {}
    need = []
    for pkg in CPU_PACKAGES:
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        want = pins.get(pkg)
        if have is None or (want and have != want):
            need.append(f"{pkg}=={want}" if want else pkg)
    if need:
        log.info("installing analysis packages: %s", " ".join(need))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
    else:
        log.info("analysis packages present at the pinned versions")


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


INSTR_ID = ("Identify the Bangla variety of the input sentence. Possible varieties: "
            + ", ".join(f"[{c}] {NAMES[c]}" for c in BANGLA)
            + ". Romanized input uses the same varieties. Answer only with JSON.")
INSTR_MT = "Translate the input from the source variety to the target variety. Answer only with JSON."


def lexicon_hints(art: Artifacts, text: str, src: str, tgt: str, max_hints: int) -> list[str]:
    """Dialect->Standard hints for a dialect source, Standard->dialect hints for a dialect target."""
    s_code, s_script = src.split(":")
    t_code, _ = tgt.split(":")
    if s_script != "bn":
        return []
    if s_code in DIALECTS:
        table, words = art.lex["d2s"].get(s_code, {}), BN_TOKEN.findall(text)
    elif s_code == "std" and t_code in DIALECTS:
        table, words = art.lex["s2d"].get(t_code, {}), BN_TOKEN.findall(text)
    else:
        return []
    hits = []
    for w in dict.fromkeys(words):
        if w in table:
            hits.append((table[w][1], w, table[w][0]))
    hits.sort()
    return [f"{w} = {v}" for _, w, v in hits[:max_hints]]


def build_prompt(art: Artifacts, task: dict, k: int, use_hints: bool) -> tuple[str, str]:
    """Return (prompt, target_json) for one task instance."""
    s = task["serial"]
    if task["kind"] == "id":
        field = task["field"]
        text = strip_punct(art.text(s, field))
        lines = [INSTR_ID, "Output format: {\"variety\": \"<tag>\"}"]
        if k:
            lines.append("Examples:")
            for ns, code in art.neighbours_auto(s, field, k)[::-1]:
                ex = strip_punct(art.text(ns, f"{code}:{field.split(':')[1]}"))
                lines.append(f"{ex} => {json.dumps({'variety': f'[{code}]'})}")
        lines += [f"Input: {text}", "Output:"]
        return "\n".join(lines), json.dumps({"variety": f"[{field.split(':')[0]}]"}, ensure_ascii=False)

    src, tgt, auto = task["src"], task["tgt"], task["auto"]
    text = art.text(s, src)
    src_tag = "[auto]" if auto else tag(src)
    fmt = ('{"source_variety": "<tag>", "translation": "<text>"}' if auto else '{"translation": "<text>"}')
    lines = [INSTR_MT, f"Source: {src_tag}   Target: {tag(tgt)}", f"Output format: {fmt}"]
    if k:
        exs = (art.neighbours_auto(s, src, k) if auto else [(n, None) for n in art.neighbours_own(s, src, k)])
        lines.append("Examples:")
        for ns, code in exs[::-1]:
            ex_src = f"{code}:{src.split(':')[1]}" if code else src
            lines.append(f"{tag(ex_src)} {art.text(ns, ex_src)} => {tag(tgt)} {art.text(ns, tgt)}")
    if use_hints and not auto:
        hints = lexicon_hints(art, text, src, tgt, SAMPLING["max_hints"])
        if hints:
            lines.append("Hints: " + "; ".join(hints))
    lines += [f"Input: {text}", "Output:"]
    out = {"source_variety": tag(src), "translation": art.text(s, tgt)} if auto else {"translation": art.text(s, tgt)}
    return "\n".join(lines), json.dumps(out, ensure_ascii=False)


def sample_tasks(art: Artifacts, serials: list[int], rng: random.Random) -> list[dict]:
    """Training mixture of the paper for the given serials (one epoch's worth)."""
    tier1 = [(f"{d}:bn", "std:bn") for d in DIALECTS] + [("std:bn", f"{d}:bn") for d in DIALECTS]
    tier2 = ([(f"{d}:bn", "en:lat") for d in DIALECTS] + [("en:lat", f"{d}:bn") for d in DIALECTS]
             + [("std:bn", "en:lat"), ("en:lat", "std:bn"), ("cm:mix", "std:bn")])
    tier3 = [(f"{a}:bn", f"{b}:bn") for a, b in art.dd["seen"]]
    tasks = []
    for s in serials:
        for kind, n in MIXTURE:
            for _ in range(n):
                if kind in ("id_bn", "id_rom"):
                    tasks.append({"kind": "id", "serial": s, "field": f"{rng.choice(BANGLA)}:{kind[3:]}"})
                else:
                    src, tgt = rng.choice({"tier1": tier1, "tier2": tier2, "tier3": tier3}[kind])
                    auto = (kind == "tier1" and src != "std:bn" and rng.random() < SAMPLING["p_auto_source"])
                    tasks.append({"kind": "mt", "serial": s, "src": src, "tgt": tgt, "auto": auto})
    rng.shuffle(tasks)
    return tasks


def realize(art: Artifacts, tasks: list[dict], rng: random.Random) -> list[tuple[str, str]]:
    out = []
    for t in tasks:
        k = rng.randint(1, SAMPLING["k_max"]) if rng.random() < SAMPLING["p_examples"] else 0
        out.append(build_prompt(art, t, k, rng.random() < SAMPLING["p_hints"]))
    return out


class Encoder:
    def __init__(self, tokenizer, chat: bool, model_key: str):
        self.tok = getattr(tokenizer, "tokenizer", tokenizer)
        self.proc = tokenizer
        self.chat = chat and (getattr(self.tok, "chat_template", None) or getattr(self.proc, "chat_template", None))
        self.template_fallbacks = 0

    def _render(self, messages, add_generation_prompt):
        kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        attempts = [messages]
        if messages and messages[0]["role"] == "system":
            merged = [dict(messages[1], content=messages[0]["content"] + "\n\n" + messages[1]["content"])] + messages[2:]
            attempts.append(merged)
        last = None
        for msgs in attempts:
            for renderer, fmt in ((self.tok, "str"), (self.proc, "parts")):
                try:
                    m = msgs if fmt == "str" else [{"role": x["role"], "content": [{"type": "text", "text": x["content"]}]}
                                                   for x in msgs]
                    return renderer.apply_chat_template(m, **kw)
                except Exception as e:
                    last = e
        raise last

    def prompt_ids(self, prompt: str) -> list[int]:
        if self.chat:
            text = self._render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], True)
        else:
            text = (self.tok.bos_token or "") + prompt + "\n"
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode(self, prompt: str, target: str) -> tuple[list[int], int]:
        """Full token ids and the number of prompt tokens (masked from the loss)."""
        p = self.prompt_ids(prompt)
        if self.chat:
            full_text = self._render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                                      {"role": "assistant", "content": target}], False)
            full = self.tok(full_text, add_special_tokens=False)["input_ids"]
            if full[:len(p)] != p:
                self.template_fallbacks += 1
                full = p + self.tok(target, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        else:
            full = p + self.tok(target, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        return full, len(p)


def length_stats(enc: Encoder, pairs: list[tuple[str, str]], max_len: int) -> dict:
    import numpy as np
    lens = np.array([len(enc.encode(p, t)[0]) for p, t in pairs])
    return {"n": int(len(lens)), "mean": round(float(lens.mean()), 1), "p50": int(np.percentile(lens, 50)),
            "p95": int(np.percentile(lens, 95)), "p99": int(np.percentile(lens, 99)), "max": int(lens.max()),
            "pct_over_max_len": round(100 * float((lens > max_len).mean()), 2), "max_len": max_len,
            "template_fallbacks": enc.template_fallbacks}


def load_unsloth(repo, max_len, local_rank, token):
    import torch
    from unsloth import FastModel
    kw = dict(model_name=repo, max_seq_length=max_len, dtype=torch.float16, load_in_4bit=True, full_finetuning=False,
              token=token, device_map={"": local_rank})
    try:
        model, tok = FastModel.from_pretrained(use_exact_model_name=True, **kw)
    except TypeError:
        model, tok = FastModel.from_pretrained(**kw)
    peft_kw = dict(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], bias="none",
                   target_modules=LORA["targets"], use_gradient_checkpointing=True, random_state=SEED)
    try:
        model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                         finetune_attention_modules=True,
                                         finetune_mlp_modules=any(t in LORA["targets"] for t in MLP), **peft_kw)
    except TypeError:
        model = FastModel.get_peft_model(model, **peft_kw)
    return model, tok


def load_hf(repo, max_len, local_rank, token):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    kw = dict(quantization_config=bnb, dtype=torch.float16, device_map={"": local_rank}, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(repo, **kw)
    try:
        tok = AutoTokenizer.from_pretrained(repo, token=token)
    except Exception:
        tok = AutoProcessor.from_pretrained(repo, token=token)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lang_only = r"^(?!.*(vision|visual|image|audio|multi_modal|mm_)).*\.(" + "|".join(LORA["targets"]) + r")$"
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                                             bias="none", target_modules=lang_only, task_type="CAUSAL_LM"))
    return model, tok


def fit_encode(art, enc, task, k: int, hints: bool, max_len: int):
    """Encode (prompt, target) for training."""
    for kk in range(k, -1, -1):
        for h in ([True, False] if hints else [False]):
            p, t = build_prompt(art, task, kk, h)
            ids, n_prompt = enc.encode(p, t)
            if len(ids) <= max_len:
                return ids, n_prompt, kk, h
    cut = len(ids) - max_len
    return ids[cut:], max(0, n_prompt - cut), 0, False


def fit_prompt(art, enc, task, k: int, hints: bool, max_len: int):
    """Prompt ids for generation, with the same length rule (applied to the prompt alone)."""
    for kk in range(k, -1, -1):
        for h in ([True, False] if hints else [False]):
            p, t = build_prompt(art, task, kk, h)
            ids = enc.prompt_ids(p)
            if len(ids) <= max_len:
                return ids, t, kk, h
    return ids[-max_len:], t, 0, False


def training_serials(art, fraction: float, seed: int) -> list[int]:
    s = sorted(art.train_serials)
    if fraction >= 1.0:
        return s
    return sorted(random.Random(seed).sample(s, int(round(fraction * len(s)))))


def epoch_batches(art, enc, serials, epoch, rank, world, batch, max_len, seed):
    """Encoded, length-grouped micro-batches of one epoch for one rank (deterministic in seed/epoch/rank)."""
    mine = serials[rank::world]
    rng = random.Random(seed * 1000 + epoch * 10 + rank)
    tasks = sample_tasks(art, mine, rng)
    enc_items, stats = [], {"k_dropped": 0, "hints_dropped": 0}
    for t in tasks:
        k = rng.randint(1, SAMPLING["k_max"]) if rng.random() < SAMPLING["p_examples"] else 0
        h = rng.random() < SAMPLING["p_hints"]
        ids, n_prompt, kk, hh = fit_encode(art, enc, t, k, h, max_len)
        stats["k_dropped"] += int(kk < k)
        stats["hints_dropped"] += int(h and not hh)
        enc_items.append((ids, n_prompt))
    order, batches, win = list(range(len(enc_items))), [], batch * 32
    for w0 in range(0, len(order), win):
        chunk = sorted(order[w0:w0 + win], key=lambda i: len(enc_items[i][0]))
        batches += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(batches)
    return enc_items, batches, stats


def collate(items, pad_id, device):
    import torch
    m = max(len(x) for x, _ in items)
    inp = torch.full((len(items), m), pad_id, dtype=torch.long)
    lab = torch.full((len(items), m), -100, dtype=torch.long)
    att = torch.zeros((len(items), m), dtype=torch.long)
    for j, (ids, n_prompt) in enumerate(items):
        inp[j, :len(ids)] = torch.tensor(ids)
        att[j, :len(ids)] = 1
        lab[j, n_prompt:len(ids)] = torch.tensor(ids[n_prompt:])
    return inp.to(device), att.to(device), lab.to(device)


def eval_items(root: Path, eval_set: str, tasks: list[str], directions: dict) -> list[dict]:
    """Frozen evaluation items from step 02, converted to QLoRA tasks."""
    items = []
    d = root / "data" / "eval" / eval_set
    if "id" in tasks:
        for line in open(d / "id.jsonl", encoding="utf-8"):
            r = json.loads(line)
            items.append({"id": r["id"], "kind": "id", "serial": r["serial"], "field": f"{r['label']}:{r['script']}",
                          "label": r["label"], "script": r["script"]})
    tiers = [t for t in tasks if t.startswith("tier")]
    if tiers:
        for line in open(d / "mt.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["tier"] in tiers:
                items.append({"id": r["id"], "kind": "mt", "serial": r["serial"], "src": r["src_field"],
                              "tgt": r["tgt_field"], "auto": False, "tier": r["tier"], "ref": r["ref"]})
    if "auto" in tasks:
        for line in open(d / "mt.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["tier"] == "tier1" and r["tgt_field"] == "std:bn":
                items.append({"id": r["id"] + "|auto", "kind": "mt", "serial": r["serial"], "src": r["src_field"],
                              "tgt": r["tgt_field"], "auto": True, "tier": "tier1-auto", "ref": r["ref"]})
    return items


JSON_OBJ = re.compile(r"\{.*?\}", re.S)


def parse_output(text: str, kind: str) -> tuple[dict, bool]:
    """First JSON object in the output; lenient fallbacks for truncated answers (counted as invalid)."""
    m = JSON_OBJ.search(text)
    if m:
        try:
            return json.loads(m.group(0)), True
        except Exception:
            pass
    out = {}
    for key in ("variety", "source_variety", "translation"):
        mm = re.search(r'"%s"\s*:\s*"([^"]*)' % key, text)
        if mm:
            out[key] = mm.group(1)
    return out, False


def norm_tag(v) -> str | None:
    if not isinstance(v, str):
        return None
    m = re.search(r"\[?([a-z]{2,3})(?:-rom)?\]?", v.strip().lower())
    return m.group(1) if m else None


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


def generate_items(model, enc, art, items, cfg, rank, world, out_path: Path, backend: str) -> None:
    """Greedy generation for this rank's share of the items; writes one JSON line per item."""
    import torch
    if backend == "unsloth":
        try:
            from unsloth import FastModel
            FastModel.for_inference(model)
        except Exception:
            pass
    model.eval()
    device = next(model.parameters()).device
    tok = enc.tok
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = stop_token_ids(tok)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stops = sorted(set(stops) | set(gen_eos if isinstance(gen_eos, list) else ([gen_eos] if gen_eos is not None else [])))
    close_ids = torch.tensor(json_close_ids(tok), device=device)
    mine = items[rank::world]
    prepared = []
    for it in mine:
        ids, target, kk, hh = fit_prompt(art, enc, it, EVAL["k"], EVAL["hints"], cfg["max_len"])
        if it["kind"] == "id":
            budget = 24
        else:
            src_len = len(tok(art.text(it["serial"], it["src"]), add_special_tokens=False)["input_ids"])
            budget = min(EVAL["max_new_cap"], 2 * src_len + 48)
        prepared.append((it, ids, budget, kk, hh))
    prepared.sort(key=lambda x: (x[2], len(x[1])))
    bs = cfg["eval_batch"]
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for b0 in range(0, len(prepared), bs):
            chunk = prepared[b0:b0 + bs]
            m = max(len(x[1]) for x in chunk)
            inp = torch.full((len(chunk), m), pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), m), dtype=torch.long)
            for j, (_, ids, _, _, _) in enumerate(chunk):
                inp[j, m - len(ids):] = torch.tensor(ids)
                att[j, m - len(ids):] = 1
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(input_ids=inp.to(device), attention_mask=att.to(device),
                                     max_new_tokens=max(x[2] for x in chunk), do_sample=False,
                                     eos_token_id=stops, pad_token_id=pad_id,
                                     stopping_criteria=json_stopper(close_ids, m))
            for j, (it, _, _, kk, hh) in enumerate(chunk):
                text = tok.decode(gen[j, m:], skip_special_tokens=True).strip()
                parsed, valid = parse_output(text, it["kind"])
                rec = {"id": it["id"], "kind": it["kind"], "raw": text, "valid_json": valid, "k_used": kk, "hints_used": hh}
                if it["kind"] == "id":
                    rec.update(label=it["label"], script=it["script"], pred=norm_tag(parsed.get("variety")))
                else:
                    rec.update(src=it["src"], tgt=it["tgt"], tier=it["tier"], auto=it["auto"], ref=it["ref"],
                               hyp=parsed.get("translation", "") or "",
                               pred_source=norm_tag(parsed.get("source_variety")) if it["auto"] else None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rank == 0 and (b0 // bs) % 20 == 0:
                log.info("eval rank0: %d/%d items (%.0f s)", b0 + len(chunk), len(prepared), time.time() - t0)


def score(pred_files: list[Path]) -> dict:
    """ID: accuracy / macro-F1 per script; MT: corpus chrF++ per direction and averages; validity rates."""
    import numpy as np
    from sacrebleu.metrics import CHRF
    from sklearn.metrics import accuracy_score, f1_score
    recs = [json.loads(l) for p in pred_files for l in open(p, encoding="utf-8")]
    res = {"n_items": len(recs), "schema_valid_rate": round(float(np.mean([r["valid_json"] for r in recs])), 4) if recs else None}
    ids = [r for r in recs if r["kind"] == "id"]
    for script in ("bn", "rom"):
        sub = [r for r in ids if r["script"] == script]
        if sub:
            y = [r["label"] for r in sub]
            p = [r["pred"] if r["pred"] in BANGLA else "invalid" for r in sub]
            res[f"id_{script}_accuracy"] = round(100 * accuracy_score(y, p), 2)
            res[f"id_{script}_macro_f1"] = round(100 * f1_score(y, p, labels=BANGLA, average="macro", zero_division=0), 2)
            res[f"id_{script}_per_label_f1"] = dict(zip(BANGLA, [round(100 * v, 2) for v in
                                                           f1_score(y, p, labels=BANGLA, average=None, zero_division=0)]))
    chrfpp = CHRF(word_order=2)
    mt = [r for r in recs if r["kind"] == "mt"]
    per_dir = {}
    for key in sorted({(r["tier"], r["src"], r["tgt"]) for r in mt}):
        sub = [r for r in mt if (r["tier"], r["src"], r["tgt"]) == key]
        per_dir["|".join(key)] = round(chrfpp.corpus_score([r["hyp"] for r in sub], [[r["ref"] for r in sub]]).score, 2)
    res["mt_chrfpp_per_direction"] = per_dir
    t1 = {k: v for k, v in per_dir.items() if k.startswith("tier1|")}
    if t1:
        d2s = [v for k, v in t1.items() if k.endswith("|std:bn")]
        s2d = [v for k, v in t1.items() if k.split("|")[1] == "std:bn"]
        res["tier1_chrfpp_mean"] = round(float(np.mean(list(t1.values()))), 2)
        res["tier1_d2s_chrfpp_mean"] = round(float(np.mean(d2s)), 2) if d2s else None
        res["tier1_s2d_chrfpp_mean"] = round(float(np.mean(s2d)), 2) if s2d else None
    for tier in ("tier2", "tier3"):
        vals = [v for k, v in per_dir.items() if k.startswith(tier + "|")]
        if vals:
            res[f"{tier}_chrfpp_mean"] = round(float(np.mean(vals)), 2)
    auto_dirs = {k: v for k, v in per_dir.items() if k.startswith("tier1-auto|")}
    if auto_dirs:
        res["auto_chrfpp_mean"] = round(float(np.mean(list(auto_dirs.values()))), 2)
    auto = [r for r in mt if r["auto"]]
    if auto:
        res["auto_source_accuracy"] = round(100 * float(np.mean([r["pred_source"] == r["src"].split(":")[0] for r in auto])), 2)
    if mt:
        res["signature"] = str(chrfpp.get_signature())
    return res


TIMER = {"t0": None, "prev": 0.0}


def train_seconds_total() -> float:
    return TIMER["prev"] + ((time.time() - TIMER["t0"]) if TIMER["t0"] else 0.0)


def save_checkpoint(path: Path, model, opt, sched, scaler, state: dict) -> None:
    import torch
    from peft import get_peft_model_state_dict
    path.mkdir(parents=True, exist_ok=True)
    state["train_seconds"] = round(train_seconds_total(), 1)
    tmp = path / "ckpt.pt.tmp"
    torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "state": state}, tmp)
    os.replace(tmp, path / "ckpt.pt")
    (path / "state.json").write_text(json.dumps(state, indent=1))


def find_checkpoint(run_name: str, output_dir: str) -> Path | None:
    """The most advanced checkpoint of this run among this session's output and every attached folder."""
    cands = [Path(output_dir, "runs", run_name, "checkpoint", "ckpt.pt")]
    cands += [Path(x) for x in glob.glob(f"/kaggle/input/**/runs/{run_name}/checkpoint/ckpt.pt", recursive=True)]
    cands = [c for c in dict.fromkeys(cands) if c.exists()]
    if not cands:
        return None

    def progress(c: Path):
        try:
            st = json.loads((c.parent / "state.json").read_text())
            return (int(bool(st.get("done"))), int(st.get("step", 0)), int(st.get("epoch", 0)), int(st.get("micro", 0)))
        except Exception:
            return (0, -1, -1, -1)
    best = max(cands, key=lambda c: (progress(c), c.stat().st_mtime))
    if len(cands) > 1:
        log.info("checkpoints found for %s: %s -> using %s", run_name,
                 [f"{c} {progress(c)}" for c in cands], best)
    return best


def configure(args) -> None:
    LORA.update(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
                targets=ATTN if args.targets == "attn" else ATTN + MLP)
    TRAIN.update(lr=args.lr, eff_batch=args.eff_batch)


def worker(args) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    configure(args)
    local_rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank0 = local_rank == 0
    cfg = MODELS[args.model]
    run_dir = Path(args.output_dir, "runs", args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(args.seed)
    token = os.environ.get("HF_TOKEN") or None
    res = {"run_name": args.run_name, "model": args.model, "backend": args.backend, "status": "started"}
    t_start = time.time()
    try:
        art = Artifacts(Path(args.artifacts_dir))
        loader = load_unsloth if args.backend == "unsloth" else load_hf
        model, tok = loader(cfg["repo"], cfg["max_len"], local_rank, token)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        trainable = [p for p in model.parameters() if p.requires_grad]
        res["trainable_params_m"] = round(sum(p.numel() for p in trainable) / 1e6, 2)
        enc = Encoder(tok, cfg["chat"], args.model)
        pad_id = enc.tok.pad_token_id if enc.tok.pad_token_id is not None else enc.tok.eos_token_id
        device = f"cuda:{local_rank}"

        serials = training_serials(art, args.train_fraction, SEED)
        batch = cfg["batch"]
        ga = max(1, TRAIN["eff_batch"] // (batch * world))
        n_local = len(serials[local_rank::world]) * sum(n for _, n in MIXTURE) // batch
        n_micro = torch.tensor([n_local], device=device)
        if world > 1:
            dist.all_reduce(n_micro, op=dist.ReduceOp.MIN)
        n_micro = int(n_micro.item())
        total_steps = max(1, args.epochs * n_micro // ga)
        res.update(train_serials=len(serials), micro_batches_per_epoch=n_micro, grad_accum=ga, optimizer_steps=total_steps,
                   effective_batch=batch * ga * world)
        import bitsandbytes as bnb
        model.train()
        if world > 1:
            for p_ in trainable:
                dist.broadcast(p_.data, src=0)
        ck = find_checkpoint(args.run_name, args.output_dir) if args.resume else None
        if ck is None and args.resume and not args.fresh:
            traces = [t for t in glob.glob(f"/kaggle/input/**/runs/{args.run_name}/*", recursive=True)]
            if traces:
                raise RuntimeError(
                    f"earlier outputs of {args.run_name} are attached ({len(traces)} files, e.g. {traces[0]}) but no "
                    "checkpoint/ckpt.pt: attach the output VERSION of the session that saved the checkpoint (check with "
                    "!find /kaggle/input -name ckpt.pt), or add --fresh to start this run from scratch on purpose")
        opt_cls = bnb.optim.AdamW8bit if ck is not None else bnb.optim.PagedAdamW8bit
        opt = opt_cls(trainable, lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        res["optimizer"] = opt_cls.__name__
        warm = max(1, int(TRAIN["warmup_frac"] * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total_steps))))
        scaler = torch.amp.GradScaler("cuda")
        state = {"epoch": 0, "micro": 0, "step": 0, "done": False}
        if ck is not None:
            from peft import set_peft_model_state_dict
            blob = torch.load(ck, map_location=device, weights_only=False)
            set_peft_model_state_dict(model, blob["adapter"])
            opt.load_state_dict(blob["opt"])
            sched.load_state_dict(blob["sched"])
            scaler.load_state_dict(blob["scaler"])
            state = blob["state"]
            if rank0:
                log.info("resumed from %s at %s", ck, state)
        res["resumed_from"] = str(ck) if ck else None

        torch.cuda.reset_peak_memory_stats(local_rank)
        log_rows, times, tok_count = [], [], 0
        last_ckpt = time.time()
        stop_for_time = False
        TIMER["prev"] = float(state.get("train_seconds", 0.0))
        TIMER["t0"] = time.time() if not state["done"] else None
        while not state["done"] and not stop_for_time:
            ep = state["epoch"]
            items, batches, stats = epoch_batches(art, enc, serials, ep, local_rank, world, batch, cfg["max_len"], args.seed)
            if rank0 and state["micro"] == 0:
                log.info("epoch %d: %d micro-batches/rank; examples with fewer retrieved examples than sampled "
                         "(length limit): %d; hints dropped: %d", ep, n_micro, stats["k_dropped"], stats["hints_dropped"])
            for mi in range(state["micro"], n_micro):
                inp, att, lab = collate([items[j] for j in batches[mi]], pad_id, device)
                last_of_group = (mi + 1) % ga == 0 or mi == n_micro - 1
                t1 = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
                scaler.scale(loss / ga).backward()
                if last_of_group:
                    if world > 1:
                        average_gradients(trainable, world)
                    scaler.unscale_(opt)
                    gn = float(torch.nn.utils.clip_grad_norm_(trainable, TRAIN["max_grad_norm"]))
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    state["step"] += 1
                else:
                    gn = float("nan")
                torch.cuda.synchronize(local_rank)
                dt = time.time() - t1
                times.append(dt)
                tok_count += int(att.sum().item())
                state["micro"] = mi + 1
                log_rows.append({"epoch": ep, "micro": mi, "step": state["step"], "loss": float(loss.item()),
                                 "grad_norm": gn, "scale": float(scaler.get_scale()), "lr": sched.get_last_lr()[0],
                                 "seconds": round(dt, 3), "tokens": int(att.sum().item())})
                if rank0 and mi % 50 == 0:
                    log.info("[%s] ep %d micro %d/%d step %d/%d loss %.4f lr %.2e %.2fs", args.run_name, ep, mi, n_micro,
                             state["step"], total_steps, loss.item(), sched.get_last_lr()[0], dt)
                flags = torch.tensor([float(time.time() - last_ckpt > args.ckpt_minutes * 60),
                                      float(time.time() - t_start > args.time_budget_h * 3600)], device=device)
                if world > 1:
                    dist.broadcast(flags, src=0)
                if last_of_group and (flags[0] > 0 or flags[1] > 0):
                    if rank0:
                        save_checkpoint(run_dir / "checkpoint", model, opt, sched, scaler, state)
                        log.info("checkpoint saved at %s", state)
                    last_ckpt = time.time()
                    if flags[1] > 0:
                        stop_for_time = True
                        break
            if stop_for_time:
                break
            state["epoch"] += 1
            state["micro"] = 0
            state["done"] = state["epoch"] >= args.epochs
            if args.epoch_select and rank0:
                from peft import get_peft_model_state_dict
                (run_dir / "epochs").mkdir(exist_ok=True)
                torch.save(get_peft_model_state_dict(model), run_dir / "epochs" / f"ep{state['epoch']}.pt")
                save_checkpoint(run_dir / "checkpoint", model, opt, sched, scaler, state)
                last_ckpt = time.time()
                log.info("epoch %d finished: adapter saved", state["epoch"])
        if rank0:
            import pandas as pd
            pd.DataFrame(log_rows).to_csv(run_dir / f"train_log_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv", index=False)
        stats_t = torch.tensor([tok_count, sum(times)], dtype=torch.float64, device=device)
        peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank) / 2**30,
                             torch.cuda.max_memory_reserved(local_rank) / 2**30], device=device)
        peaks = [peak]
        if world > 1:
            dist.all_reduce(stats_t)
            peaks = [torch.zeros_like(peak) for _ in range(world)]
            dist.all_gather(peaks, peak)
        losses = [r["loss"] for r in log_rows]
        res["train_hours_total"] = round(train_seconds_total() / 3600, 3)
        TIMER["prev"], TIMER["t0"] = train_seconds_total(), None
        res.update(train_hours_this_session=round((time.time() - t_start) / 3600, 3),
                   train_tokens_per_s_total=round(float(stats_t[0] / (stats_t[1] / world)), 1) if stats_t[1] > 0 else None,
                   peak_alloc_gib=[round(float(p[0]), 2) for p in peaks], peak_reserved_gib=[round(float(p[1]), 2) for p in peaks],
                   loss_first50=round(float(np.mean(losses[:50])), 4) if losses else None,
                   loss_last50=round(float(np.mean(losses[-50:])), 4) if losses else None,
                   nonfinite_losses=int(sum(not np.isfinite(x) for x in losses)), state=state)
        if not state["done"]:
            res["status"] = "paused (time budget): rerun with --resume in a new session"
            return
        if rank0 and not args.no_save:
            model.save_pretrained(run_dir / "adapter")
            save_checkpoint(run_dir / "checkpoint", model, opt, sched, scaler, state)
        del opt, sched
        torch.cuda.empty_cache()

        if world > 1:
            dist.barrier()
        if args.epoch_select:
            finished = evaluate_epochs(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start)
            res["status"] = "ok" if finished else "paused (evaluation pending): rerun with --resume"
            return
        t_eval = time.time()
        directions = json.loads((Path(args.artifacts_dir) / "data" / "directions.json").read_text())
        items = eval_items(Path(args.artifacts_dir), args.eval_set, args.eval_tasks.split(","), directions)
        pred_dir = run_dir / f"pred_{args.eval_set}"
        pred_dir.mkdir(exist_ok=True)
        generate_items(model, enc, art, items, cfg, local_rank, world, pred_dir / f"rank{local_rank}.jsonl", args.backend)
        if world > 1:
            dist.barrier()
        res["eval_hours"] = round((time.time() - t_eval) / 3600, 3)
        if rank0:
            res["metrics"] = score(sorted(pred_dir.glob("rank*.jsonl")))
            log.info("[%s] metrics: ID bn macro-F1 %s | Tier-1 chrF++ %s (D->S %s, S->D %s) | valid %s", args.run_name,
                     res["metrics"].get("id_bn_macro_f1"), res["metrics"].get("tier1_chrfpp_mean"),
                     res["metrics"].get("tier1_d2s_chrfpp_mean"), res["metrics"].get("tier1_s2d_chrfpp_mean"),
                     res["metrics"].get("schema_valid_rate"))
        res["status"] = "ok"
    except Exception as e:
        res["status"] = "failed"
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        if rank0:
            log.error("[%s] FAILED: %s", args.run_name, res["error"])
    finally:
        if rank0:
            res["software"] = software_versions()
            res.update(config={"lora": LORA, "train": TRAIN, "sampling": SAMPLING, "eval": EVAL, "mixture": MIXTURE,
                               "prompt_version": PROMPT_VERSION, "system": SYSTEM, "model": cfg, "args": vars(args)})
            name = "metrics.json" if res["status"] == "ok" else f"result_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
            (run_dir / name).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if world > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def software_versions() -> dict:
    """Python / PyTorch / library versions of this run (Kaggle images change over time)."""
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "unsloth", "unsloth-zoo", "trl", "sacrebleu"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return out


def locate(run_name: str, rel: str, output_dir: str) -> Path | None:
    """A file of this run in the current output or in an attached earlier session."""
    own = Path(output_dir, "runs", run_name, rel)
    if own.exists():
        return own
    return find_one(f"/kaggle/input/**/runs/{run_name}/{rel}")


def save_release(model, enc, out: Path, args, cfg: dict, epoch: int, metrics: dict) -> None:
    """Everything needed to use the adapter elsewhere (e.g. on Hugging Face)."""
    model.save_pretrained(out)
    for obj in (enc.proc, enc.tok):
        try:
            obj.save_pretrained(out)
            break
        except Exception:
            continue
    info = {
        "framework": "QLoRA: one shared QLoRA adapter for Bangla dialect identification and translation",
        "base_model": cfg["repo"], "quantization": "4-bit NF4 with double quantization, fp16 compute (as in training)",
        "backend_used_for_training": args.backend, "prompt_version": PROMPT_VERSION, "system_message": SYSTEM,
        "control_tags": {**{f"[{c}]": NAMES[c] for c in BANGLA}, "[en]": "English", "[cm]": "code-mixed Bangla",
                         "[auto]": "source variety unknown (detect, then translate)", "-rom suffix": "romanized input, e.g. [ctg-rom]"},
        "identification_instruction": INSTR_ID, "translation_instruction": INSTR_MT,
        "output_format": {"identification": {"variety": "<tag>"}, "translation": {"translation": "<text>"},
                          "translation_with_[auto]": {"source_variety": "<tag>", "translation": "<text>"}},
        "prompt_layout": ["instruction", "Source: <tag>   Target: <tag> (translation)", "Output format: <json>",
                          "Examples: <up to k retrieved pairs> (optional)", "Hints: <word = translation; ...> (optional)",
                          "Input: <text>", "Output:"],
        "inference_defaults": {"retrieved_examples_k": EVAL["k"], "lexicon_hints": EVAL["hints"], "decoding": "greedy",
                               "identification_input": "punctuation removed"},
        "training": {"lora": LORA, "optimizer": TRAIN, "task_mixture_per_serial": MIXTURE, "prompt_sampling": SAMPLING,
                     "epochs_trained": args.epochs, "selected_epoch": epoch, "seed": args.seed,
                     "train_fraction": args.train_fraction, "max_seq_len": cfg["max_len"]},
        "dev_full_metrics": {k: metrics.get(k) for k in ("tier1_chrfpp_mean", "tier1_d2s_chrfpp_mean",
                                                         "tier1_s2d_chrfpp_mean", "id_bn_macro_f1", "id_rom_macro_f1",
                                                         "schema_valid_rate", "signature")},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "prompt_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("release files written to %s: %s", out, sorted(p.name for p in out.iterdir()))


def evaluate_epochs(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start) -> bool:
    """Evaluate the adapter of every epoch on dev_select, keep the epoch with the best Tier-1 chrF++ (ties: the earlier epoch), then evaluate that adapter on dev_full."""
    import torch
    import torch.distributed as dist
    from peft import set_peft_model_state_dict
    directions = json.loads((Path(args.artifacts_dir) / "data" / "directions.json").read_text())

    def load_epoch(e):
        f = locate(args.run_name, f"epochs/ep{e}.pt", args.output_dir)
        if f is None:
            raise FileNotFoundError(f"adapter of epoch {e} not found (attach the earlier session's output)")
        set_peft_model_state_dict(model, torch.load(f, map_location=f"cuda:{local_rank}", weights_only=False))

    def run_eval(tag, eval_set, e, tasks=None) -> dict | None:
        rel = f"pred_{tag}/metrics.json"
        done = locate(args.run_name, rel, args.output_dir)
        if done is not None:
            return json.loads(done.read_text())
        heavy = any(t in (tasks or args.eval_tasks) for t in ("tier2", "tier3", "auto"))
        need_h = EVAL_HOURS[eval_set] if (eval_set != "test_full" or heavy) else EVAL_HOURS["dev_full"]
        if (time.time() - t_start) / 3600 + need_h > args.time_budget_h:
            if rank0:
                log.warning("time budget: evaluation %s left for the next session (--resume)", tag)
            return None
        load_epoch(e)
        items = eval_items(Path(args.artifacts_dir), eval_set, (tasks or args.eval_tasks).split(","), directions)
        pred_dir = run_dir / f"pred_{tag}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        generate_items(model, enc, art, items, cfg, local_rank, world, pred_dir / f"rank{local_rank}.jsonl", args.backend)
        if world > 1:
            dist.barrier()
        if rank0:
            m = score(sorted(pred_dir.glob("rank*.jsonl")))
            m.update(epoch=e, eval_set=eval_set, hours=round((time.time() - t0) / 3600, 3))
            (pred_dir / "metrics.json").write_text(json.dumps(m, indent=1))
            log.info("[%s] %s (epoch %d): Tier-1 chrF++ %s | ID bn macro-F1 %s", args.run_name, eval_set, e,
                     m.get("tier1_chrfpp_mean"), m.get("id_bn_macro_f1"))
        if world > 1:
            dist.barrier()
        return json.loads((pred_dir / "metrics.json").read_text())

    sel = {}
    for e in range(1, args.epochs + 1):
        m = run_eval(f"dev_select_ep{e}", "dev_select", e)
        if m is None:
            return False
        sel[e] = m
    best = max(sel, key=lambda e: (sel[e]["tier1_chrfpp_mean"], -e))
    res["epoch_selection"] = {"criterion": "Tier-1 chrF++ on dev_select (ties: earlier epoch)", "best_epoch": best,
                              "dev_select": {e: {"tier1": m["tier1_chrfpp_mean"], "id_bn": m.get("id_bn_macro_f1")}
                                             for e, m in sel.items()}}
    finals = [x.split(":") for x in args.final_evals.split(";")] if getattr(args, "final_evals", None) \
        else [("dev_full", args.eval_tasks)]
    results = {}
    for eval_set, tasks in finals:
        m = run_eval(eval_set, eval_set, best, tasks)
        if m is None:
            return False
        results[eval_set] = m
    m_full = results.get("dev_full") or results[finals[0][0]]
    res["metrics"] = m_full
    if "test_full" in results:
        res["metrics_test"] = results["test_full"]
    if rank0:
        load_epoch(best)
        save_release(model, enc, run_dir / "adapter", args, cfg, best, m_full)
        log.info("[%s] selected epoch %d; dev_full Tier-1 chrF++ %s, ID bn macro-F1 %s%s", args.run_name, best,
                 m_full.get("tier1_chrfpp_mean"), m_full.get("id_bn_macro_f1"),
                 "" if "test_full" not in results else f"; TEST Tier-1 {results['test_full'].get('tier1_chrfpp_mean')}, "
                 f"ID bn {results['test_full'].get('id_bn_macro_f1')}, Tier-2 {results['test_full'].get('tier2_chrfpp_mean')}, "
                 f"Tier-3 {results['test_full'].get('tier3_chrfpp_mean')}")
    return True


def average_gradients(params, world: int) -> None:
    """All-reduce (mean) the gradients of the trainable parameters in one flat buffer."""
    import torch
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(world)
    for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def clear_model_cache(repo: str) -> None:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for d in hub.glob("models--" + repo.replace("/", "--")):
        shutil.rmtree(d, ignore_errors=True)


def launch(args) -> None:
    import signal
    import torch
    tok = hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
    else:
        log.warning("HF_TOKEN not found: gated models (Llama) will fail")
    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise SystemExit("no GPU: set Accelerator to 'GPU T4 x2'")
    if args.preset == "search":
        search_launch(args, n_gpu)
        return
    if args.preset == "transfer":
        transfer_launch(args, n_gpu)
        return
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_name = args.run_name or f"{args.preset or 'custom'}-{key}-s{args.seed}"
        a = args
        if args.preset == "base" and key not in BASE_MODELS:
            log.warning("%s: not trained in the base-model comparison (Gemma-4's entry is its step 10 run "
                        "primary-gemma4-e4b-s13) -- skipped", key)
            continue
        if args.preset in ("final", "primary", "base"):
            import copy
            a = copy.copy(args)
            cfg_sel, source = final_config(key, args.output_dir)
            explicit = {x.split("=")[0].lstrip("-").replace("-", "_") for x in sys.argv[1:] if x.startswith("--")}
            for k, v in cfg_sel.items():
                if k not in explicit:
                    setattr(a, k, v)
            log.info("%s: final configuration %s (from %s)", key, {k: getattr(a, k) for k in cfg_sel}, source)
        run_single(a, key, run_name, n_gpu)
        clear_model_cache(MODELS[key]["repo"])
    (Path(args.output_dir) / "STEP3_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


def run_single(args, key: str, run_name: str, n_gpu: int) -> bool:
    """One QLoRA training + evaluation run (torchrun, one process per GPU)."""
    import signal
    if True:
        backend = args.backend if args.backend != "auto" else MODELS[key]["backends"][0]
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", "--master_port=29541",
               str(Path(__file__).resolve()), "--worker", "--model", key, "--backend", backend, "--run-name", run_name]
        for k, v in vars(args).items():
            if k in ("models", "worker", "model", "backend", "run_name", "compare", "dry_run", "preset", "no_install",
                     "requirements", "models_list", "stage", "trials", "compare_search", "compare_final", "compare_transfer",
                     "compare_base"):
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append("--" + k.replace("_", "-"))
            elif v is not None:
                cmd += ["--" + k.replace("_", "-"), str(v)]
        log.info("===== run %s (%s, %s backend, %d GPUs) =====", run_name, key, backend, n_gpu)
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            proc.wait(timeout=(args.time_budget_h + 1.5) * 3600)
        except subprocess.TimeoutExpired:
            log.error("%s timed out", run_name)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        m = Path(args.output_dir, "runs", run_name, "metrics.json")
        log.info("%s -> %s", run_name, "ok" if m.exists() else "not finished (see result_*.json)")
        return m.exists()


def compare(args) -> None:
    import pandas as pd
    files = sorted(set(glob.glob(f"{args.output_dir}/runs/select-*/metrics.json")
                       + glob.glob("/kaggle/input/**/runs/select-*/metrics.json", recursive=True)))
    rows, seen = [], set()
    for f in files:
        r = json.loads(Path(f).read_text())
        if r["run_name"] in seen:
            continue
        seen.add(r["run_name"])
        m = r["metrics"]
        rows.append({"model": r["model"], "run": r["run_name"], "tier1_chrfpp": m.get("tier1_chrfpp_mean"),
                     "d2s_chrfpp": m.get("tier1_d2s_chrfpp_mean"), "s2d_chrfpp": m.get("tier1_s2d_chrfpp_mean"),
                     "id_bn_macro_f1": m.get("id_bn_macro_f1"), "id_rom_macro_f1": m.get("id_rom_macro_f1"),
                     "schema_valid": m.get("schema_valid_rate"), "train_h": r.get("train_hours_this_session"),
                     "eval_h": r.get("eval_hours"), "tokens_per_s": r.get("train_tokens_per_s_total"),
                     "peak_gib": max(r.get("peak_reserved_gib") or [0])})
    if not rows:
        raise SystemExit("no finished select runs found (attach the outputs of the step 05 notebooks)")
    df = pd.DataFrame(rows).sort_values("tier1_chrfpp", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    reason = "highest Tier-1 chrF++"
    if len(df) > 1 and best["tier1_chrfpp"] - df.iloc[1]["tier1_chrfpp"] <= 1.0:
        a, b = df.iloc[0], df.iloc[1]
        if abs(a["id_bn_macro_f1"] - b["id_bn_macro_f1"]) > 1.0:
            best = a if a["id_bn_macro_f1"] > b["id_bn_macro_f1"] else b
            reason = "top two within 1.0 chrF++; higher identification macro-F1"
        else:
            best = a if a["train_h"] <= b["train_h"] else b
            reason = "top two within 1.0 chrF++ and 1.0 macro-F1; lower training cost"
    out = Path(args.output_dir, "selection")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "model_selection.csv", index=False)
    decision = {"primary_model": best["model"], "reason": reason, "rule": SELECTION_RULE, "n_models": len(df),
                "missing_models": sorted(set(MODELS) - set(df["model"])), "table": df.to_dict("records")}
    (out / "selection.json").write_text(json.dumps(decision, indent=2, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        log.info("MODEL SELECTION (dev_select)\n%s", df.to_string(index=False))
    log.info("PRIMARY MODEL: %s (%s)%s", best["model"], reason,
             "" if not decision["missing_models"] else f" -- WARNING: missing runs {decision['missing_models']}")


def _round_lr(x: float) -> float:
    return float(f"{x:.3g}")


def global_configs() -> list[dict]:
    """16 configurations from a scrambled Sobol sequence (fixed seed) over the space of the paper."""
    import math
    from scipy.stats import qmc
    u = qmc.Sobol(d=6, scramble=True, seed=SEARCH["sobol_seed"]).random(SEARCH["n_global"])
    lo, hi = SEARCH_SPACE["lr"]
    pick = lambda opts, x: opts[min(int(x * len(opts)), len(opts) - 1)]
    out = []
    for row in u:
        r = pick(SEARCH_SPACE["lora_r"], row[1])
        out.append({"lr": _round_lr(math.exp(math.log(lo) + row[0] * (math.log(hi) - math.log(lo)))),
                    "lora_r": r, "lora_alpha": r * pick(SEARCH_SPACE["alpha_mult"], row[2]),
                    "lora_dropout": pick(SEARCH_SPACE["lora_dropout"], row[3]),
                    "targets": pick(SEARCH_SPACE["targets"], row[4]), "eff_batch": pick(SEARCH_SPACE["eff_batch"], row[5])})
    return out


def trial_name(key: str, stage: str, i: int) -> str:
    return f"search-{key}-{'g' if stage == 'global' else 'l'}{i:02d}"


def find_metrics(run_name: str, output_dir: str) -> dict | None:
    """metrics.json of a finished trial, from this session or an attached earlier output."""
    for f in [Path(output_dir, "runs", run_name, "metrics.json")] + \
             [Path(x) for x in glob.glob(f"/kaggle/input/**/runs/{run_name}/metrics.json", recursive=True)]:
        if f.exists():
            try:
                r = json.loads(f.read_text())
                if r.get("status") == "ok":
                    return r
            except Exception:
                pass
    return None


def trial_row(key: str, name: str, cfg: dict, r: dict | None) -> dict:
    m = (r or {}).get("metrics", {})
    return {"trial": name, **cfg, "tier1": m.get("tier1_chrfpp_mean"), "d2s": m.get("tier1_d2s_chrfpp_mean"),
            "s2d": m.get("tier1_s2d_chrfpp_mean"), "id_bn": m.get("id_bn_macro_f1"), "id_rom": m.get("id_rom_macro_f1"),
            "valid_json": m.get("schema_valid_rate"), "train_h": (r or {}).get("train_hours_this_session"),
            "eligible": (m.get("id_bn_macro_f1") is not None
                         and m["id_bn_macro_f1"] >= SCREEN_ID_F1[key] - SEARCH["id_tolerance"])}


def ranked_eligible(rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r["tier1"] is not None and r["eligible"]]
    return sorted(ok, key=lambda r: (-r["tier1"], -r["id_bn"]))


def local_configs(key: str, output_dir: str) -> list[dict]:
    """8 perturbations around the 3 best eligible global configurations."""
    g = global_configs()
    rows = []
    for i, cfg in enumerate(g):
        r = find_metrics(trial_name(key, "global", i), output_dir)
        if r is None:
            raise SystemExit(f"global trial {i} of {key} is not finished (or its output is not attached): the local "
                             "stage needs all 16 global results")
        rows.append(trial_row(key, trial_name(key, "global", i), cfg, r))
    top = ranked_eligible(rows)[:3]
    if len(top) < 3:
        raise SystemExit(f"fewer than 3 eligible global configurations for {key}")
    base = lambda row: {k: row[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout", "targets", "eff_batch")}
    out = []
    for rank, row in enumerate(top):
        for f in SEARCH["lr_steps"]:
            out.append({**base(row), "lr": _round_lr(row["lr"] * f), "_from": row["trial"]})
        if rank == 0:
            b = base(row)
            mult = b["lora_alpha"] // b["lora_r"]
            r2 = b["lora_r"] * 2 if b["lora_r"] < 64 else b["lora_r"] // 2
            out.append({**b, "lora_r": r2, "lora_alpha": r2 * mult, "_from": row["trial"]})
            out.append({**b, "targets": "attn" if b["targets"] == "all" else "all", "_from": row["trial"]})
    return out[:SEARCH["n_local"]]


def parse_trials(spec: str, n: int) -> list[int]:
    if not spec or spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [i for i in out if 0 <= i < n]


def search_launch(args, n_gpu: int) -> None:
    """Run the requested trials of one stage for each requested model; finished trials are skipped."""
    import copy
    t0 = time.time()
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        plan = global_configs() if args.stage == "global" else local_configs(key, args.output_dir)
        idx = parse_trials(args.trials, len(plan))
        plan_file = Path(args.output_dir, "search", key, f"plan_{args.stage}.json")
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(plan, indent=1))
        log.info("===== %s: %s stage, trials %s =====", key, args.stage, idx)
        for i in idx:
            name = trial_name(key, args.stage, i)
            if find_metrics(name, args.output_dir):
                log.info("%s already finished -- skipped", name)
                continue
            if (time.time() - t0) / 3600 + TRIAL_HOURS[key] > args.time_budget_h:
                log.warning("time budget: %s not started (run it in another session)", name)
                continue
            a = copy.copy(args)
            for k, v in plan[i].items():
                if not k.startswith("_"):
                    setattr(a, k, v)
            log.info("%s: %s", name, {k: v for k, v in plan[i].items()})
            run_single(a, key, name, n_gpu)
        clear_model_cache(MODELS[key]["repo"])
    (Path(args.output_dir) / "STEP3B_DONE.txt").write_text(f"{args.stage} trials {args.trials} finished "
                                                             f"{datetime.now(timezone.utc).isoformat()}\n")


def compare_search(args) -> None:
    """Selection over all finished trials of a model."""
    import pandas as pd
    lo, hi = SEARCH_SPACE["lr"]
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows = []
        for stage, plan in (("global", global_configs()), ("local", None)):
            if plan is None:
                try:
                    plan = local_configs(key, args.output_dir)
                except SystemExit as e:
                    log.warning("%s: local stage not available (%s)", key, e)
                    plan = []
            for i, cfg in enumerate(plan):
                name = trial_name(key, stage, i)
                rows.append(trial_row(key, name, {k: v for k, v in cfg.items() if not k.startswith("_")},
                                      find_metrics(name, args.output_dir)))
        df = pd.DataFrame(rows)
        out = Path(args.output_dir, "search", key)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "trials.csv", index=False)
        done = df["tier1"].notna().sum()
        ranked = ranked_eligible(rows)
        if not ranked:
            log.error("%s: no eligible finished trial", key)
            continue
        inside = [r for r in ranked if lo <= r["lr"] <= hi]
        outside = [r for r in ranked if not lo <= r["lr"] <= hi]
        best, reason = (inside[0], "best eligible configuration inside the global range") if inside else (ranked[0], "")
        if outside and inside and outside[0]["tier1"] > inside[0]["tier1"] + SEARCH["boundary_min_gain"]:
            best, reason = outside[0], "boundary rule: out-of-range learning rate adopted (gain > 0.5)"
        elif outside and inside and outside[0]["tier1"] > inside[0]["tier1"]:
            reason = "boundary rule: out-of-range learning rate not adopted (gain <= 0.5)"
        sel = {"model": key, "finished_trials": int(done), "planned_trials": len(rows),
               "config": {k: best[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout", "targets", "eff_batch")},
               "trial": best["trial"], "tier1_chrfpp": best["tier1"], "id_bn_macro_f1": best["id_bn"],
               "id_minimum": SCREEN_ID_F1[key] - SEARCH["id_tolerance"], "reason": reason,
               "screening_default": {"lr": 2e-4, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "targets": "all",
                                     "eff_batch": 32}}
        (out / "search_selection.json").write_text(json.dumps(sel, indent=2))
        with pd.option_context("display.width", 230, "display.max_columns", 20):
            log.info("%s: %d/%d trials finished\n%s", key, done, len(rows),
                     df.sort_values("tier1", ascending=False).to_string(index=False))
        log.info("%s SELECTED: %s (Tier-1 chrF++ %.2f, ID %.2f) -- %s%s", key, sel["config"], best["tier1"], best["id_bn"],
                 reason, "" if done == len(rows) else "  [WARNING: not all trials finished]")


def transfer_configs(key: str) -> list[dict]:
    base = dict(SELECTED_CONFIGS[TRANSFER["source"]])
    return [{**base, "lr": _round_lr(base["lr"] * f)} for f in TRANSFER["factors"]]


def transfer_rows(key: str, output_dir: str) -> list[dict]:
    rows = []
    for i, cfg in enumerate(transfer_configs(key)):
        name = f"transfer-{key}-lr{i}"
        rows.append(trial_row(key, name, cfg, find_metrics(name, output_dir)))
    ext = find_metrics(f"transfer-{key}-ext", output_dir)
    if ext is not None:
        c = ext["config"]["args"]
        rows.append(trial_row(key, f"transfer-{key}-ext", {k: c[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout",
                                                                               "targets", "eff_batch")}, ext))
    return rows


def transfer_decision(key: str, output_dir: str) -> dict | None:
    """Selection over the learning-rate sweep (same criterion as step 05)."""
    rows = transfer_rows(key, output_dir)
    sweep = [r for r in rows if not r["trial"].endswith("-ext")]
    if any(r["tier1"] is None for r in sweep):
        return None
    ranked = ranked_eligible(sweep)
    if not ranked:
        return {"model": key, "error": "no eligible learning rate", "rows": rows}
    best = ranked[0]
    lrs = sorted(r["lr"] for r in sweep)
    edge = "up" if best["lr"] == lrs[-1] else "down" if best["lr"] == lrs[0] else None
    dec = {"model": key, "transferred_from": TRANSFER["source"], "rows": rows, "sweep_best": best["trial"]}
    if edge:
        ext_lr = _round_lr(best["lr"] * (TRANSFER["edge_step"] if edge == "up" else 1 / TRANSFER["edge_step"]))
        ext = next((r for r in rows if r["trial"].endswith("-ext")), None)
        if ext is None or ext["tier1"] is None:
            dec["extension"] = {**{k: best[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout", "targets",
                                                         "eff_batch")}, "lr": ext_lr}
            return dec
        if ext["eligible"] and ext["tier1"] > best["tier1"] + SEARCH["boundary_min_gain"]:
            best, dec["reason"] = ext, "boundary rule: extension adopted (gain > 0.5)"
        else:
            dec["reason"] = "boundary rule: extension not adopted (gain <= 0.5)"
    else:
        dec["reason"] = "best learning rate inside the sweep"
    dec.update(config={k: best[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout", "targets", "eff_batch")},
               trial=best["trial"], tier1_chrfpp=best["tier1"], id_bn_macro_f1=best["id_bn"],
               id_minimum=SCREEN_ID_F1[key] - SEARCH["id_tolerance"])
    return dec


def transfer_launch(args, n_gpu: int) -> None:
    """Run the 3-point learning-rate sweep (and, if the best value is at an edge, the boundary extension) for each model; finished trials are skipped."""
    import copy
    t0 = time.time()
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        if key not in TRANSFER["models"]:
            log.warning("%s is not a transfer model (finalists were searched with --preset search) -- skipped", key)
            continue
        plan = [(f"transfer-{key}-lr{i}", c) for i, c in enumerate(transfer_configs(key))]
        for round_ in (0, 1):
            for name, cfg in plan:
                if find_metrics(name, args.output_dir):
                    continue
                if (time.time() - t0) / 3600 + TRIAL_HOURS[key] > args.time_budget_h:
                    log.warning("time budget: %s not started (rerun with this notebook's output attached)", name)
                    continue
                a = copy.copy(args)
                for k, v in cfg.items():
                    setattr(a, k, v)
                log.info("%s: %s", name, cfg)
                run_single(a, key, name, n_gpu)
            dec = transfer_decision(key, args.output_dir)
            if dec is None or "extension" not in dec:
                break
            log.info("%s: best learning rate at the edge of the sweep -> boundary extension lr=%g", key, dec["extension"]["lr"])
            plan = [(f"transfer-{key}-ext", dec["extension"])]
        write_transfer_selection(key, args.output_dir)
        clear_model_cache(MODELS[key]["repo"])


def write_transfer_selection(key: str, output_dir: str) -> None:
    import pandas as pd
    dec = transfer_decision(key, output_dir)
    out = Path(output_dir, "search", key)
    out.mkdir(parents=True, exist_ok=True)
    if dec is None or "config" not in dec:
        log.warning("%s: transfer selection not decided yet (%s)", key, "sweep unfinished" if dec is None else
                    ("extension pending" if "extension" in dec else dec.get("error")))
        return
    pd.DataFrame(dec["rows"]).to_csv(out / "transfer_trials.csv", index=False)
    (out / "transfer_selection.json").write_text(json.dumps({k: v for k, v in dec.items() if k != "rows"}, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        log.info("%s transfer sweep\n%s", key, pd.DataFrame(dec["rows"])[["trial", "lr", "tier1", "d2s", "s2d", "id_bn",
                                                                             "eligible"]].to_string(index=False))
    log.info("%s SELECTED: %s (Tier-1 chrF++ %.2f, ID %.2f) -- %s", key, dec["config"], dec["tier1_chrfpp"],
             dec["id_bn_macro_f1"], dec["reason"])


def compare_base(args) -> None:
    """Base-model table: all five base models under the identical 4-epoch protocol (step 05 + step 10 seed 13)."""
    import pandas as pd
    runs = {k: f"base-{k}-s13" for k in BASE_MODELS}
    runs[PRIMARY_MODEL] = f"primary-{PRIMARY_MODEL}-s13"
    rows = []
    for key in [PRIMARY_MODEL] + BASE_MODELS:
        f = locate(runs[key], "metrics.json", args.output_dir)
        r = json.loads(f.read_text()) if f else {}
        dev, test = r.get("metrics", {}), r.get("metrics_test", {})
        rows.append({"model": key, "run": runs[key], "finished": bool(f), "best_epoch": (r.get("epoch_selection") or {}).get("best_epoch"),
                     "lr": (r.get("config") or {}).get("train", {}).get("lr"), "lora_r": (r.get("config") or {}).get("lora", {}).get("r"),
                     "dev_tier1": dev.get("tier1_chrfpp_mean"), "dev_id_bn": dev.get("id_bn_macro_f1"),
                     "test_tier1": test.get("tier1_chrfpp_mean"), "test_d2s": test.get("tier1_d2s_chrfpp_mean"),
                     "test_s2d": test.get("tier1_s2d_chrfpp_mean"), "test_id_bn": test.get("id_bn_macro_f1"),
                     "test_id_rom": test.get("id_rom_macro_f1"), "train_h": r.get("train_hours_total"),
                     "trainable_params_m": r.get("trainable_params_m")})
    df = pd.DataFrame(rows)
    out = Path(args.output_dir, "selection")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "base_models.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 20):
        log.info("BASE-MODEL COMPARISON (identical protocol: up to 4 epochs, epoch chosen on dev_select)\n%s",
                 df.drop(columns=["run"]).to_string(index=False))
    missing = df[~df.finished]["model"].tolist()
    if missing:
        log.warning("not finished or not attached: %s", missing)


def final_config(key: str, output_dir: str) -> tuple[dict, str]:
    for name in ("search_selection.json", "transfer_selection.json"):
        f = Path(output_dir, "search", key, name)
        f = f if f.exists() else find_one(f"/kaggle/input/**/search/{key}/{name}")
        if f is not None:
            return json.loads(f.read_text())["config"], str(f)
    if key in SELECTED_CONFIGS:
        return SELECTED_CONFIGS[key], "SELECTED_CONFIGS in this file (step 05 result)"
    raise SystemExit(f"no selected configuration for {key}: attach the output with search/{key}/transfer_selection.json "
                     "or search_selection.json")


def load_preds(run_name: str, tag: str, output_dir: str) -> list[dict]:
    files = sorted(Path(output_dir, "runs", run_name, f"pred_{tag}").glob("rank*.jsonl"))
    if not files:
        files = sorted(Path(p) for p in glob.glob(f"/kaggle/input/**/runs/{run_name}/pred_{tag}/rank*.jsonl", recursive=True))
    return [json.loads(l) for f in files for l in open(f, encoding="utf-8")]


def bootstrap_compare(recs_a: list[dict], recs_b: list[dict], n: int, seed: int) -> dict:
    """Paired bootstrap over serials: every resample draws serials with replacement and keeps all outputs of a drawn serial."""
    import numpy as np
    from sacrebleu.metrics import CHRF
    from sklearn.metrics import f1_score
    serial_of = lambda r: int(r["id"].split("|")[1])
    a = {r["id"]: r for r in recs_a}
    b = {r["id"]: r for r in recs_b}
    common = sorted(set(a) & set(b))
    serials = sorted({serial_of(a[i]) for i in common})
    s_index = {s: j for j, s in enumerate(serials)}
    chrf = CHRF(word_order=2)
    mt_ids = [i for i in common if a[i]["kind"] == "mt" and a[i].get("tier") == "tier1"]
    dirs = sorted({(a[i]["src"], a[i]["tgt"]) for i in mt_ids})
    mt = {}
    for d in dirs:
        ids = [i for i in mt_ids if (a[i]["src"], a[i]["tgt"]) == d]
        refs = [[a[i]["ref"] for i in ids]]
        mt[d] = {"ser": np.array([s_index[serial_of(a[i])] for i in ids]),
                 "A": np.array(chrf._extract_corpus_statistics([a[i]["hyp"] for i in ids], refs)),
                 "B": np.array(chrf._extract_corpus_statistics([b[i]["hyp"] for i in ids], refs))}
    id_ids = [i for i in common if a[i]["kind"] == "id" and a[i].get("script") == "bn"]
    lab = np.array([BANGLA.index(a[i]["label"]) for i in id_ids])
    pa = np.array([BANGLA.index(a[i]["pred"]) if a[i]["pred"] in BANGLA else -1 for i in id_ids])
    pb = np.array([BANGLA.index(b[i]["pred"]) if b[i]["pred"] in BANGLA else -1 for i in id_ids])
    id_ser = np.array([s_index[serial_of(a[i])] for i in id_ids])

    def tier1(weights, sys_):
        vals = []
        for d in dirs:
            w = weights[mt[d]["ser"]]
            vals.append(chrf._compute_score_from_stats(list((mt[d][sys_] * w[:, None]).sum(0))).score)
        return float(np.mean(vals))

    def macro(weights, pred):
        w = weights[id_ser]
        keep = w > 0
        return 100 * f1_score(np.repeat(lab[keep], w[keep]), np.repeat(pred[keep], w[keep]), labels=range(10),
                              average="macro", zero_division=0)

    ones = np.ones(len(serials), dtype=int)
    point = {"tier1_A": tier1(ones, "A"), "tier1_B": tier1(ones, "B"), "id_A": macro(ones, pa), "id_B": macro(ones, pb)}
    rng = np.random.default_rng(seed)
    d_t1, d_id = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(serials), len(serials)), minlength=len(serials))
        d_t1.append(tier1(w, "A") - tier1(w, "B"))
        d_id.append(macro(w, pa) - macro(w, pb))
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return {"n_serials": len(serials), "n_mt": len(mt_ids), "n_id": len(id_ids), "resamples": n,
            **{k: round(v, 3) for k, v in point.items()},
            "tier1_diff": round(point["tier1_A"] - point["tier1_B"], 3), "tier1_diff_ci95": ci(d_t1),
            "id_diff": round(point["id_A"] - point["id_B"], 3), "id_diff_ci95": ci(d_id)}


def compare_final(args) -> None:
    """Pre-registered choice of the primary model between the two finalists."""
    a_key, b_key = "ministral3-8b", "gemma4-e4b"
    runs = {k: f"final-{k}-s{args.seed}" for k in (a_key, b_key)}
    recs = {k: load_preds(runs[k], "dev_full", args.output_dir) for k in runs}
    for k, r in recs.items():
        if not r:
            raise SystemExit(f"no dev_full predictions for {runs[k]}: attach the step 05 output of {k}")
    bs = bootstrap_compare(recs[a_key], recs[b_key], BOOTSTRAP["n"], BOOTSTRAP["seed"])
    excl = lambda c: c[0] > 0 or c[1] < 0
    if excl(bs["tier1_diff_ci95"]):
        winner, reason = (a_key if bs["tier1_diff"] > 0 else b_key), "Tier-1 chrF++ difference significant"
    elif excl(bs["id_diff_ci95"]):
        winner, reason = (a_key if bs["id_diff"] > 0 else b_key), "Tier-1 not significant; identification macro-F1 significant"
    else:
        winner, reason = b_key, "neither difference significant; the cheaper model"
    out = Path(args.output_dir, "selection")
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for k in runs:
        f = locate(runs[k], "metrics.json", args.output_dir)
        meta[k] = json.loads(f.read_text()) if f else {}
    decision = {"primary_model": winner, "reason": reason, "rule": FINAL_RULE, "A": a_key, "B": b_key,
                "bootstrap": bs, "selected_epoch": {k: (meta[k].get("epoch_selection") or {}).get("best_epoch") for k in runs},
                "train_hours": {k: meta[k].get("train_hours_total", meta[k].get("train_hours_this_session")) for k in runs}}
    (out / "final_selection.json").write_text(json.dumps(decision, indent=2))
    log.info("FINAL SELECTION (dev_full, %d serials, %d resamples)\n  Tier-1 chrF++: %s %.2f vs %s %.2f, diff %.2f, 95%% CI %s"
             "\n  ID macro-F1:   %s %.2f vs %s %.2f, diff %.2f, 95%% CI %s\n  PRIMARY MODEL: %s (%s)", bs["n_serials"],
             bs["resamples"], a_key, bs["tier1_A"], b_key, bs["tier1_B"], bs["tier1_diff"], bs["tier1_diff_ci95"],
             a_key, bs["id_A"], b_key, bs["id_B"], bs["id_diff"], bs["id_diff_ci95"], winner, reason)


def dry_run(args) -> None:
    art = Artifacts(Path(args.artifacts_dir))
    serials = training_serials(art, args.train_fraction, SEED)
    rng = random.Random(SEED)
    tasks = sample_tasks(art, serials[:40], rng)
    directions = json.loads((Path(args.artifacts_dir) / "data" / "directions.json").read_text())
    items = eval_items(Path(args.artifacts_dir), args.eval_set, args.eval_tasks.split(","), directions)
    kinds = {}
    for it in items:
        kinds[it.get("tier", "id")] = kinds.get(it.get("tier", "id"), 0) + 1
    log.info("train serials used: %d (fraction %.2f); tasks from 40 serials: %d", len(serials), args.train_fraction, len(tasks))
    log.info("eval set %s: %d items %s", args.eval_set, len(items), kinds)
    p, t = build_prompt(art, items[0], EVAL["k"], EVAL["hints"])
    log.info("first eval prompt:\n%s\n-> %s", p, t)
    for text, kind in (('{"variety": "[ctg]"}', "id"), ('noise {"translation": "আমি"} tail', "mt"), ('{"translation": "আম', "mt")):
        log.info("parse %r -> %s", text, parse_output(text, kind))


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 05: QLoRA training and evaluation")
    p.add_argument("--preset", choices=list(PRESETS), help="named setting")
    p.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {list(MODELS)}")
    p.add_argument("--compare", action="store_true", help="apply the model-selection rule to finished select runs")
    p.add_argument("--stage", choices=["global", "local"], default="global", help="search stage (with --preset search)")
    p.add_argument("--trials", default="all", help="search trials to run in this session, e.g. 0-3 or 0,2,5")
    p.add_argument("--compare-search", action="store_true", help="select the best search configuration per model")
    p.add_argument("--no-save", action="store_true", help="do not save adapter/checkpoint (search trials)")
    p.add_argument("--epoch-select", action="store_true", help="save every epoch, choose the epoch on dev_select")
    p.add_argument("--final-evals", help="evaluations of the chosen epoch, e.g. 'dev_full:id,tier1;test_full:id,tier1,tier2'")
    p.add_argument("--compare-final", action="store_true", help="choose the primary model (bootstrap rule)")
    p.add_argument("--compare-transfer", action="store_true", help="decide the learning rate per model")
    p.add_argument("--compare-base", action="store_true", help="base-model comparison table")
    p.add_argument("--run-name", help="default: <preset>-<model>-s<seed>")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--targets", choices=["all", "attn"], default="all")
    p.add_argument("--eff-batch", type=int, default=32)
    p.add_argument("--eval-set", default="dev_select", choices=["dev_select", "dev_full", "test_prompt", "test_full"])
    p.add_argument("--eval-tasks", default="id,tier1", help="comma-separated: id, tier1, tier2, tier3, auto")
    p.add_argument("--backend", default="auto", choices=["auto", "unsloth", "hf"])
    p.add_argument("--resume", action="store_true", help="continue from the run's checkpoint (this or an attached session)")
    p.add_argument("--fresh", action="store_true", help="start from scratch even if earlier outputs of this run are attached")
    p.add_argument("--ckpt-minutes", type=float, default=45)
    p.add_argument("--time-budget-h", type=float, default=10.5, help="pause and checkpoint after this many hours")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.preset:
        explicit = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
        for k, v in PRESETS[args.preset].items():
            if k not in explicit:
                setattr(args, k, v)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if args.preset == "primary" and not any(x.startswith("--models") for x in sys.argv[1:]):
        args.models = PRIMARY_MODEL
    if args.compare or args.compare_search or args.compare_final or args.compare_transfer or args.compare_base:
        ensure_cpu_packages(args)
    if args.compare_base:
        compare_base(args)
        return
    if args.compare_transfer:
        for key in [m.strip() for m in args.models.split(",") if m.strip() in TRANSFER["models"]]:
            write_transfer_selection(key, args.output_dir)
        return
    if args.compare:
        compare(args)
        return
    if args.compare_search:
        compare_search(args)
        return
    if args.compare_final:
        compare_final(args)
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
    log.info("artifacts: %s | output: %s | preset: %s", args.artifacts_dir, args.output_dir, args.preset)
    launch(args)


if __name__ == "__main__":
    main()

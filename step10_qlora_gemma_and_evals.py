#!/usr/bin/env python3
"""QLoRA Gemma-4-E4B and the test-set evaluations of all QLoRA adapters.

Presets:
    primary   Gemma-4-E4B, all training data, up to four epochs, epoch chosen on dev_select (seeds 13, 42, 2026)
    plain     the same without retrieved examples and word hints in training and inference
    data10, data25, data50   training on 10, 25 and 50% of the training data (learning curve)
    ablate    test-set evaluations of a trained adapter, e.g. --variants T-noex,T-extra
              (T-noex: identification and Tier 1 without examples and hints; T-extra: Tier 2, Tier 3, [auto])
    external  zero-shot transfer to Vashantor, ONUBAD and BanglaRegionalTextCorpus
The presets single-id, single-mt, no-examples and capacity are development experiments not reported in the
paper. Evaluations resume item by item over several sessions.

Usage:
    python step10_qlora_gemma_and_evals.py --preset primary --seed 13
    python step10_qlora_gemma_and_evals.py --preset plain --seed 13
    python step10_qlora_gemma_and_evals.py --preset ablate --from-run base-titullm-3b-s42 --variants T-noex,T-extra
    python step10_qlora_gemma_and_evals.py --preset external --from-run primary-gemma4-e4b-s13
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
ABL_DEFAULT = {"retriever": "hybrid", "examples": "retrieved", "source": "gold", "id_input": "nopunct"}
ABL = dict(ABL_DEFAULT)
RETRIEVER_KEY = {"hybrid": "hybrid", "bm25": "bm25", "labse": "dense"}
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
_P = PRESETS["primary"]
PRESETS["single-id"] = {**_P, "eval_tasks": "id", "mixture": "id_bn:2,id_rom:1", "select_metric": "id",
                        "final_evals": "dev_full:id;test_full:id"}
PRESETS["single-mt"] = {**_P, "eval_tasks": "tier1", "mixture": "tier1:3,tier2:1,tier3:1",
                        "final_evals": "dev_full:tier1;test_full:tier1,tier2,tier3,auto"}
PRESETS["no-examples"] = {**_P, "p_examples": 0.0, "final_evals": "dev_full:id,tier1"}
PRESETS["plain"] = {**_P, "p_examples": 0.0, "p_hints": 0.0, "eval_k": 0, "eval_hints": "off",
                    "final_evals": "dev_full:id,tier1;test_full:id,tier1,tier2,tier3,auto"}
for _f in (10, 25, 50):
    PRESETS[f"data{_f}"] = {**_P, "train_fraction": _f / 100, "final_evals": "dev_full:id,tier1"}
PRESETS["ablate"] = {"eval_set": "dev_full"}
PRESETS["external"] = {"eval_set": "dev_full", "external": True}
PRESETS["capacity"] = {"train_fraction": 0.2, "epochs": 1, "eval_set": "dev_select", "eval_tasks": "id,tier1",
                       "ckpt_minutes": 1e9, "no_save": True}
STEP6_TRAIN_PRESETS = ("primary", "single-id", "single-mt", "no-examples", "data10", "data25", "data50", "plain")
CAPACITY = [("r8", {"lora_r": 8, "lora_alpha": 16}), ("r16", {"lora_r": 16, "lora_alpha": 32}),
            ("attn", {"targets": "attn"})]
CAPACITY_REFERENCE = "search-gemma4-e4b-l05"
FINAL_RULE = ("higher mean Tier-1 chrF++ on dev_full if the paired bootstrap 95% CI of the difference excludes zero; "
              "otherwise higher identification macro-F1 (Bangla script) if its CI excludes zero; otherwise the cheaper "
              "model (Gemma-4-E4B)")
BOOTSTRAP = {"n": 1000, "seed": 13}

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
        self.own = {k: own[k] for k in own.files if k.endswith("__serial")}
        self.pool = {}
        for script in ("bn", "rom"):
            a = np.load(root / "retrieval" / f"all_{script}.npz")
            idx = {(int(s), int(v)): i for i, (s, v) in enumerate(zip(a["q_serial"], a["q_variety"]))}
            ranks = {m: (a[f"{m}__serial"], a[f"{m}__variety"]) for m in ("hybrid", "bm25", "dense") if f"{m}__serial" in a.files}
            self.pool[script] = (idx, ranks, [str(c) for c in a["variety_codes"]])
        self.train_serials = [int(s) for s in self.table.index[self.table["split"] == "train"]]

    def text(self, serial: int, field: str) -> str:
        return self.table.at[serial, field]

    def _rng(self, *parts) -> random.Random:
        import hashlib
        return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))

    def neighbours_own(self, serial: int, field: str, k: int) -> list[int]:
        if ABL["examples"] == "random":
            pool = [x for x in self.train_serials if x != serial]
            return self._rng("own", serial, field, k).sample(pool, k)
        arr = self.own[f"{field.replace(':', '__')}__{RETRIEVER_KEY[ABL['retriever']]}__serial"][self.row_of[serial]]
        return [int(x) for x in arr if x >= 0][:k]

    def neighbours_auto(self, serial: int, field: str, k: int) -> list[tuple[int, str]]:
        code, script = field.split(":")
        idx, ranks, codes = self.pool[script]
        if ABL["examples"] == "random":
            rng = self._rng("auto", serial, field, k)
            return [(rng.choice([x for x in self.train_serials if x != serial]), rng.choice(BANGLA)) for _ in range(k)]
        ser, var = ranks[RETRIEVER_KEY[ABL["retriever"]]]
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
        if ABL["id_input"] == "punct" and task.get("text_raw"):
            text = task["text_raw"]
        elif ABL["id_input"] == "masked" and task.get("text_masked"):
            text = task["text_masked"]
        else:
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
    no_src = ABL["source"] == "none" and not auto
    header = f"Target: {tag(tgt)}" if no_src else f"Source: {src_tag}   Target: {tag(tgt)}"
    lines = [INSTR_MT, header, f"Output format: {fmt}"]
    if k:
        exs = (art.neighbours_auto(s, src, k) if auto else [(n, None) for n in art.neighbours_own(s, src, k)])
        lines.append("Examples:")
        for ns, code in exs[::-1]:
            ex_src = f"{code}:{src.split(':')[1]}" if code else src
            src_part = art.text(ns, ex_src) if no_src else f"{tag(ex_src)} {art.text(ns, ex_src)}"
            lines.append(f"{src_part} => {tag(tgt)} {art.text(ns, tgt)}")
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
                          "label": r["label"], "script": r["script"], "text_raw": r.get("text"),
                          "text_masked": r.get("text_masked")})
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


def generate_items(model, enc, art, items, cfg, rank, world, out_path: Path, backend: str, deadline: float | None = None) -> bool:
    """Greedy generation for this rank's share of the items; writes one JSON line per item (flushed after every batch, so a stopped session keeps what it generated)."""
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
    complete = True
    with open(out_path, "w", encoding="utf-8") as f:
        for b0 in range(0, len(prepared), bs):
            if deadline is not None and time.time() > deadline:
                complete = False
                if rank == 0:
                    log.warning("time budget: generation stopped after %d/%d items; the rest follows next session", b0,
                                len(prepared))
                break
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
            f.flush()
            if (b0 // bs) % 20 == 0:
                log.info("eval rank%d: %d/%d items (%.0f s)", rank, b0 + len(chunk), len(prepared), time.time() - t0)
    return complete


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
    if getattr(args, "mixture", None):
        MIXTURE[:] = [(k, int(n)) for k, n in (x.split(":") for x in args.mixture.split(","))]
    if getattr(args, "p_examples", None) is not None:
        SAMPLING["p_examples"] = float(args.p_examples)
    if getattr(args, "p_hints", None) is not None:
        SAMPLING["p_hints"] = float(args.p_hints)
    if getattr(args, "eval_k", None) is not None:
        EVAL["k"] = int(args.eval_k)
    if getattr(args, "eval_hints", None):
        EVAL["hints"] = args.eval_hints == "on"


def worker(args) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    configure(args)
    local_rank, world = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=12))
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
        if getattr(args, "ablate_from", None) and getattr(args, "external", False):
            finished = run_external(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start)
            res["status"] = "ok" if finished else "paused (external evaluation pending): rerun with this output attached"
            return
        if getattr(args, "ablate_from", None):
            finished = run_ablations(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start)
            res["status"] = "ok" if finished else "paused (ablations pending): rerun with this output attached"
            return

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
        if (time.time() - t_start) / 3600 + EVAL_HOURS[eval_set] > args.time_budget_h:
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
    metric = "id_bn_macro_f1" if getattr(args, "select_metric", "tier1") == "id" else "tier1_chrfpp_mean"
    best = max(sel, key=lambda e: (sel[e][metric], -e))
    res["epoch_selection"] = {"criterion": f"{'identification macro-F1' if metric.startswith('id') else 'Tier-1 chrF++'} "
                                           "on dev_select (ties: earlier epoch)", "best_epoch": best,
                              "dev_select": {e: {"tier1": m.get("tier1_chrfpp_mean"), "id_bn": m.get("id_bn_macro_f1")}
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


ABLATIONS = [
    ("A1-auto", "auto", {}),
    ("A1-none", "tier1", {"source": "none"}),
    ("A2-none", "id,tier1", {"k": 0}),
    ("A2-random", "id,tier1", {"examples": "random"}),
    ("A3-bm25", "id,tier1", {"retriever": "bm25"}),
    ("A3-labse", "id,tier1", {"retriever": "labse"}),
    ("A4-k1", "id,tier1", {"k": 1}),
    ("A4-k5", "id,tier1", {"k": 5}),
    ("A4-k8", "id,tier1", {"k": 8}),
    ("A5-nohints", "tier1", {"hints": False}),
    ("A13-masked", "id", {"id_input": "masked"}),
    ("A14-punct", "id", {"id_input": "punct"}),
    ("T-noex", "id,tier1", {"k": 0, "hints": False}),
    ("T-extra", "tier2,tier3,auto", {}),
]
ABL_EVAL_SET = {"T-noex": "test_full", "T-extra": "test_full"}
ABLATION_HOURS = {"id,tier1": 1.6, "tier1": 1.1, "id": 0.5, "auto": 0.6, "tier2,tier3,auto": 3.0}
ABLATION_MAX_LEN = 1024


def run_ablations(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start) -> bool:
    import torch
    import torch.distributed as dist
    from peft import set_peft_model_state_dict
    src_run = args.ablate_from
    best, f = load_source_adapter(args, model, src_run, local_rank, rank0)
    m_src = locate(src_run, "metrics.json", args.output_dir)
    src = json.loads(m_src.read_text())
    res.update(ablate_from=src_run, source_epoch=best, reference_dev_full=src.get("metrics"))
    return run_ablation_variants(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start)


def load_source_adapter(args, model, src_run, local_rank, rank0):
    """Load the selected epoch of a finished run into the LoRA model (epoch file, else the release adapter)."""
    import torch
    from peft import set_peft_model_state_dict
    m_src = locate(src_run, "metrics.json", args.output_dir)
    if m_src is None:
        raise FileNotFoundError(f"metrics.json of {src_run} not found: attach that run's output")
    src = json.loads(m_src.read_text())
    best = (src.get("epoch_selection") or {}).get("best_epoch")
    f = locate(src_run, f"epochs/ep{best}.pt", args.output_dir) if best else None
    if f is not None:
        state = torch.load(f, map_location=f"cuda:{local_rank}", weights_only=False)
    else:
        f = locate(src_run, "adapter/adapter_model.safetensors", args.output_dir)
        if f is None:
            raise FileNotFoundError(f"neither epochs/ep{best}.pt nor adapter/ of {src_run} found: attach that run's output")
        from safetensors.torch import load_file
        state = load_file(str(f), device=f"cuda:{local_rank}")
    result = set_peft_model_state_dict(model, state)
    missing = [k for k in getattr(result, "unexpected_keys", []) if "lora" in k]
    if missing:
        raise RuntimeError(f"adapter weights of {src_run} did not match the model: {missing[:3]} ...")
    if rank0:
        log.info("adapter of %s (epoch %s) loaded from %s", src_run, best, f)
    return best, f


def previous_predictions(run_name: str, rel_dir: str, output_dir: str) -> dict:
    """Prediction records already generated for one evaluation folder (runs/<run>/<rel_dir>/*.jsonl) in this session's output or in any attached earlier session, keyed by (kind, id)."""
    files = sorted(glob.glob(str(Path(output_dir, "runs", run_name, rel_dir, "*.jsonl"))))
    files += sorted(glob.glob(f"/kaggle/input/**/runs/{run_name}/{rel_dir}/*.jsonl", recursive=True))
    recs = {}
    for fp in files:
        try:
            lines = open(fp, encoding="utf-8").read().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "id" in r:
                recs.setdefault((r.get("kind"), r["id"]), r)
    return recs


def run_ablation_variants(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start) -> bool:
    import torch.distributed as dist
    directions = json.loads((Path(args.artifacts_dir) / "data" / "directions.json").read_text())
    wanted = [x.strip() for x in args.variants.split(",")] if getattr(args, "variants", None) else \
        [a[0] for a in ABLATIONS if not a[0].startswith("T-")]
    cfg_eval = dict(cfg, max_len=max(cfg["max_len"], ABLATION_MAX_LEN))
    results = {}
    for name, tasks, change in ABLATIONS:
        if name not in wanted:
            continue
        rel = f"abl_{name}/metrics.json"
        done = locate(args.run_name, rel, args.output_dir)
        if done is not None:
            results[name] = json.loads(done.read_text())
            continue
        resumable = name.startswith("T-")
        if not resumable and (time.time() - t_start) / 3600 + ABLATION_HOURS[tasks] > args.time_budget_h:
            if rank0:
                log.warning("time budget: ablation %s left for the next session", name)
            res["ablations"] = results
            return False
        if resumable and (time.time() - t_start) / 3600 > args.time_budget_h - 0.25:
            if rank0:
                log.warning("time budget: evaluation %s left for the next session", name)
            res["ablations"] = results
            return False
        ABL.clear(); ABL.update(ABL_DEFAULT); EVAL.update(k=3, hints=True)
        for k_, v_ in change.items():
            (EVAL if k_ in ("k", "hints") else ABL)[k_] = v_
        items = eval_items(Path(args.artifacts_dir), ABL_EVAL_SET.get(name, "dev_full"), tasks.split(","), directions)
        if change.get("id_input") == "masked":
            items = [it for it in items if it.get("text_masked")]
        out = run_dir / f"abl_{name}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        complete = True
        if resumable:
            keys = {(it["kind"], it["id"]) for it in items}
            prev = {k: v for k, v in previous_predictions(args.run_name, f"abl_{name}", args.output_dir).items() if k in keys}
            if world > 1:
                dist.barrier()
            if rank0:
                for old_f in out.glob("*.jsonl"):
                    old_f.unlink()
                with open(out / "prev.jsonl", "w", encoding="utf-8") as fp:
                    for r in prev.values():
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                log.info("[%s] %d/%d items already generated in earlier sessions; %d to go", name, len(prev), len(items),
                         len(items) - len(prev))
            if world > 1:
                dist.barrier()
            todo = [it for it in items if (it["kind"], it["id"]) not in prev]
            complete = generate_items(model, enc, art, todo, cfg_eval, local_rank, world, out / f"rank{local_rank}.jsonl",
                                      args.backend, deadline=t_start + args.time_budget_h * 3600)
            if world > 1:
                import torch
                flag = torch.tensor([1 if complete else 0], device=next(model.parameters()).device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                complete = bool(flag.item())
        else:
            generate_items(model, enc, art, items, cfg_eval, local_rank, world, out / f"rank{local_rank}.jsonl", args.backend)
        if world > 1:
            dist.barrier()
        if not complete:
            if rank0:
                log.warning("time budget: evaluation %s paused -- rerun the same command with THIS session's output "
                            "attached; it continues with the remaining items", name)
            res["ablations"] = results
            return False
        if rank0:
            m = score(sorted(out.glob("*.jsonl")) if resumable else sorted(out.glob("rank*.jsonl")))
            m.update(ablation=name, change=change, tasks=tasks, hours=round((time.time() - t0) / 3600, 3), n_items=len(items),
                     eval_set=ABL_EVAL_SET.get(name, "dev_full"))
            (out / "metrics.json").write_text(json.dumps(m, indent=1))
            log.info("[ablation %s] Tier-1 %s | ID bn %s | auto %s (%.2f h)", name, m.get("tier1_chrfpp_mean"),
                     m.get("id_bn_macro_f1"), m.get("auto_chrfpp_mean"), m["hours"])
        if world > 1:
            dist.barrier()
        results[name] = json.loads((out / "metrics.json").read_text())
    ABL.clear(); ABL.update(ABL_DEFAULT); EVAL.update(k=3, hints=True)
    res["ablations"] = {k: {x: v.get(x) for x in ("tier1_chrfpp_mean", "tier1_d2s_chrfpp_mean", "tier1_s2d_chrfpp_mean",
                                                  "id_bn_macro_f1", "id_rom_macro_f1", "auto_chrfpp_mean",
                                                  "auto_source_accuracy")} for k, v in results.items()}
    return True


EXT_CODES = {"chittagong": "ctg", "noakhali": "nkh", "sylhet": "syl", "barishal": "bar", "barisal": "bar",
             "khulna": "khl", "mymensingh": "mym", "rangpur": "rng", "narail": "nar"}
EXT_UNSEEN = {"mym", "rng", "nar"}


def ext_normalize(text) -> str | None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from step02_prepare_data import normalize_text
        return normalize_text(text)
    except Exception:
        import unicodedata
        return None if text is None or (isinstance(text, float) and text != text) else \
            re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def load_external(search_root: str) -> list[dict]:
    """Rows: dataset, subset, code, bn (dialect, Bangla script), rom (romanized dialect or None), std."""
    import pandas as pd
    try:
        import openpyxl
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openpyxl"], check=False)
    rows = []
    find = lambda pat: sorted(glob.glob(f"{search_root}/**/{pat}", recursive=True))
    for name in ("Sentence.xlsx", "Clause.xlsx"):
        for f in find(name):
            d = pd.read_excel(f)
            stdcol = next(c for c in d.columns if c.lower().startswith("standard bangla"))
            for _, r in d.iterrows():
                for col, code in (("Chittagong Language", "ctg"), ("Sylhet Language", "syl"), ("Barisal Language", "bar")):
                    rows.append(dict(dataset="ONUBAD", subset=name.split(".")[0].lower(), code=code, bn=r.get(col),
                                     rom=None, std=r.get(stdcol)))
    for f in find("BanglaRegionalTextCorpus*.xlsx"):
        d = pd.read_excel(f)
        for _, r in d.iterrows():
            code = EXT_CODES.get(str(r.get("Region_Labels", "")).strip().lower())
            if code:
                rows.append(dict(dataset="BRTC", subset="all", code=code, bn=r.get("Regional_Texts"), rom=None,
                                 std=r.get("Standard_Bangla_Texts")))
    for f in find("*Test*Translation*.csv"):
        d = pd.read_csv(f)
        cols = {re.sub(r"[\s_]+", "_", c.strip().lower()): c for c in d.columns}
        region = re.split(r"[\s_]+", Path(f).name.strip())[0].lower()
        code = EXT_CODES.get(region)
        if code is None:
            continue
        bn_col, rom_col, std_col = cols.get(f"{region}_bangla_speech"), cols.get(f"{region}_banglish_speech"), cols.get("bangla_speech")
        if not (bn_col and std_col):
            log.warning("Vashantor file %s: expected columns not found (%s)", f, list(d.columns))
            continue
        for _, r in d.iterrows():
            rows.append(dict(dataset="Vashantor", subset="test", code=code, bn=r.get(bn_col),
                             rom=r.get(rom_col) if rom_col else None, std=r.get(std_col)))
    out = []
    for r in rows:
        if r["code"] in EXT_UNSEEN:
            continue
        r = {**r, "bn": ext_normalize(r["bn"]), "std": ext_normalize(r["std"]),
             "rom": ext_normalize(r["rom"]) if isinstance(r.get("rom"), str) else None}
        if r["bn"] and r["std"]:
            out.append(r)
    return out


class ExternalArtifacts(Artifacts):
    """BUC artifacts plus external sentences under negative pseudo-serials."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.ext, self._own_idx, self._pool_idx = {}, {}, {}

    def add(self, serial: int, field: str, text: str) -> None:
        self.ext[(serial, field)] = text

    def text(self, serial: int, field: str) -> str:
        return self.ext[(serial, field)] if serial < 0 else super().text(serial, field)

    @staticmethod
    def _vec():
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)

    def _top(self, index, query: str, k: int):
        vec, X, keys = index
        sims = (X @ vec.transform([strip_punct(query)]).T).toarray().ravel()
        order = sims.argsort()[::-1][:k]
        return [keys[i] for i in order]

    def neighbours_own(self, serial: int, field: str, k: int) -> list[int]:
        if serial >= 0:
            return super().neighbours_own(serial, field, k)
        if field not in self._own_idx:
            texts = [strip_punct(super(ExternalArtifacts, self).text(t, field)) for t in self.train_serials]
            vec = self._vec()
            self._own_idx[field] = (vec, vec.fit_transform(texts), list(self.train_serials))
        return self._top(self._own_idx[field], self.text(serial, field), k)

    def neighbours_auto(self, serial: int, field: str, k: int) -> list[tuple[int, str]]:
        if serial >= 0:
            return super().neighbours_auto(serial, field, k)
        script = field.split(":")[1]
        if script not in self._pool_idx:
            keys, texts = [], []
            for c in BANGLA:
                for t in self.train_serials:
                    keys.append((t, c))
                    texts.append(strip_punct(super(ExternalArtifacts, self).text(t, f"{c}:{script}")))
            vec = self._vec()
            self._pool_idx[script] = (vec, vec.fit_transform(texts), keys)
        return self._top(self._pool_idx[script], self.text(serial, field), k)


def external_items(art: ExternalArtifacts, rows: list[dict]) -> tuple[list[dict], dict]:
    buc = set()
    for col in [c for c in art.table.columns if c.endswith(":bn")]:
        buc.update(strip_punct(str(t)) for t in art.table[col].dropna())
    items, meta = [], {"n_rows": len(rows), "duplicates": {}}
    for i, r in enumerate(rows):
        s = -(i + 1)
        dup = strip_punct(r["bn"]) in buc or strip_punct(r["std"]) in buc
        key = f"{r['dataset']}/{r['code']}"
        if dup:
            meta["duplicates"][key] = meta["duplicates"].get(key, 0) + 1
            continue
        c, base = r["code"], f"ext|{r['dataset']}|{r['subset']}|{i}|{r['code']}"
        art.add(s, f"{c}:bn", r["bn"])
        art.add(s, "std:bn", r["std"])
        grp = f"{r['dataset']}|{r['subset']}"
        items.append({"id": base + "|id", "kind": "id", "serial": s, "field": f"{c}:bn", "label": c, "script": "bn",
                      "group": grp})
        items.append({"id": base + "|auto", "kind": "mt", "serial": s, "src": f"{c}:bn", "tgt": "std:bn", "auto": True,
                      "tier": f"ext-auto|{grp}", "ref": r["std"], "group": grp})
        items.append({"id": base + "|d2s", "kind": "mt", "serial": s, "src": f"{c}:bn", "tgt": "std:bn", "auto": False,
                      "tier": f"ext-d2s|{grp}", "ref": r["std"], "group": grp})
        items.append({"id": base + "|s2d", "kind": "mt", "serial": s, "src": "std:bn", "tgt": f"{c}:bn", "auto": False,
                      "tier": f"ext-s2d|{grp}", "ref": r["bn"], "group": grp})
        if r.get("rom"):
            art.add(s, f"{c}:rom", r["rom"])
            items.append({"id": base + "|rom-id", "kind": "id", "serial": s, "field": f"{c}:rom", "label": c,
                          "script": "rom", "group": grp})
            items.append({"id": base + "|rom2s", "kind": "mt", "serial": s, "src": f"{c}:rom", "tgt": "std:bn",
                          "auto": False, "tier": f"ext-rom2s|{grp}", "ref": r["std"], "group": grp})
    return items, meta


def error_rate(hyps: list[str], refs: list[str], chars: bool) -> float:
    """Corpus-level CER (characters, spaces included) or WER (whitespace tokens): total edits / total reference length."""
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]
    edits = total = 0
    for h, r in zip(hyps, refs):
        a, b = (list(h), list(r)) if chars else (h.split(), r.split())
        edits += lev(a, b)
        total += len(b)
    return edits / max(total, 1)


def score_external(recs: list[dict], art: ExternalArtifacts) -> list[dict]:
    """One row per dataset, task and dialect (plus 'all'): chrF++, BLEU, copy chrF++, validity, identification."""
    from collections import Counter
    from sacrebleu.metrics import BLEU, CHRF
    chrf, bleu = CHRF(word_order=2), BLEU()
    out = []
    by = {}
    for r in recs:
        p = r["id"].split("|")
        ds, code, task = p[1], p[4], p[-1]
        for d in (code, "all"):
            by.setdefault((ds, task, d), []).append(r)
    for (ds, task, d), sub in sorted(by.items()):
        row = {"dataset": ds, "task": task, "dialect": d, "n": len(sub),
               "valid_json": round(100 * sum(r["valid_json"] for r in sub) / len(sub), 2)}
        if sub[0]["kind"] == "mt":
            hyp, ref = [r["hyp"] for r in sub], [[r["ref"] for r in sub]]
            src = [art.text(int(-(int(r["id"].split("|")[3]) + 1)), r["src"]) for r in sub]
            row.update(chrfpp=round(chrf.corpus_score(hyp, ref).score, 2), bleu=round(bleu.corpus_score(hyp, ref).score, 2),
                       cer=round(error_rate(hyp, ref[0], chars=True), 4), wer=round(error_rate(hyp, ref[0], chars=False), 4),
                       copy_chrfpp=round(chrf.corpus_score(src, ref).score, 2))
            if task == "auto":
                pred = Counter(r.get("pred_source") or "invalid" for r in sub)
                row["source_accuracy"] = round(100 * sum(r.get("pred_source") == r["src"].split(":")[0] for r in sub) / len(sub), 2)
                row["predicted_sources"] = json.dumps(dict(pred.most_common()))
        else:
            row["accuracy"] = round(100 * sum(r["pred"] == r["label"] for r in sub) / len(sub), 2)
            row["predicted_labels"] = json.dumps(dict(Counter(r.get("pred") or "invalid" for r in sub).most_common()))
        out.append(row)
    return out


def external_classifiers(id_items: list[dict], art: ExternalArtifacts, search_root: str, device: str) -> list[dict]:
    """The two strongest identification baselines (SVM from step 06, MuRIL-large seed 13 from step 07)."""
    import numpy as np
    preds = []
    texts = [strip_punct(art.text(it["serial"], it["field"])) for it in id_items]
    svm = sorted(glob.glob(f"{search_root}/**/step4a/svm/final/model.joblib", recursive=True))
    if svm:
        try:
            import joblib
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import step06_classical_classifiers
            main_mod = sys.modules["__main__"]
            for name in dir(step06_classical_classifiers):
                if not hasattr(main_mod, name):
                    setattr(main_mod, name, getattr(step06_classical_classifiers, name))
            m = joblib.load(svm[0])
            y = m["classifier"].predict(m["features"].transform(texts))
            labels = m.get("labels", BANGLA)
            preds.append(("SVM", [labels[int(v)] if not isinstance(v, str) else v for v in y]))
            log.info("external: SVM predictions from %s", svm[0])
        except Exception as e:
            log.warning("external: SVM unavailable (%s)", e)
    mur = [p for p in sorted(glob.glob(f"{search_root}/**/step4b/**/model/config.json", recursive=True)) if "muril" in p.lower()]
    mur = sorted(mur, key=lambda p: ("seed13" not in p, p))
    if mur:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            d = str(Path(mur[0]).parent)
            tok = AutoTokenizer.from_pretrained(d)
            mdl = AutoModelForSequenceClassification.from_pretrained(d, torch_dtype=torch.float16).to(device).eval()
            id2 = {int(k): v for k, v in mdl.config.id2label.items()}
            out = []
            for b in range(0, len(texts), 128):
                enc = tok(texts[b:b + 128], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    out += mdl(**enc).logits.argmax(-1).tolist()
            preds.append(("MuRIL-large", [id2[i] for i in out]))
            del mdl
            torch.cuda.empty_cache()
            log.info("external: MuRIL predictions from %s", d)
        except Exception as e:
            log.warning("external: MuRIL unavailable (%s)", e)
    rows = []
    from collections import Counter
    for name, p in preds:
        groups = {}
        for it, y in zip(id_items, p):
            ds, code = it["id"].split("|")[1], it["label"]
            for d in (code, "all"):
                groups.setdefault((ds, d), []).append((it, y))
        for (ds, d), sub in sorted(groups.items()):
            rows.append({"system": name, "dataset": ds, "dialect": d, "n": len(sub),
                         "accuracy": round(100 * sum(y == it["label"] for it, y in sub) / len(sub), 2),
                         "predicted_labels": json.dumps(dict(Counter(y for _, y in sub).most_common()))})
    return rows


def run_external(args, model, enc, art, cfg, run_dir, local_rank, world, rank0, res, t_start) -> bool:
    import torch.distributed as dist
    src_run = args.ablate_from
    best, _ = load_source_adapter(args, model, src_run, local_rank, rank0)
    root = args.external_dir or "/kaggle/input"
    xart = ExternalArtifacts(Path(args.artifacts_dir))
    rows = load_external(root)
    items, meta = external_items(xart, rows)
    if rank0:
        log.info("external: %d rows, %d items; excluded as duplicates of BUC: %s", len(rows), len(items), meta["duplicates"])
    if not items:
        raise FileNotFoundError(f"no external dataset files found under {root}")
    ABL.clear(); ABL.update(ABL_DEFAULT); EVAL.update(k=3, hints=True)
    cfg_eval = dict(cfg, max_len=max(cfg["max_len"], ABLATION_MAX_LEN))
    out = run_dir / "external"
    out.mkdir(parents=True, exist_ok=True)
    for ds in sorted({it["id"].split("|")[1] for it in items}):
        done = locate(args.run_name, f"external/{ds}/done.json", args.output_dir)
        if done is not None:
            if rank0:
                for f in Path(done).parent.glob("rank*.jsonl"):
                    (out / ds).mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, out / ds / f.name)
            continue
        sub = [it for it in items if it["id"].split("|")[1] == ds]
        hours = len(sub) / 9000
        if (time.time() - t_start) / 3600 + hours > args.time_budget_h:
            if rank0:
                log.warning("time budget: external dataset %s left for the next session", ds)
            return False
        (out / ds).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        generate_items(model, enc, xart, sub, cfg_eval, local_rank, world, out / ds / f"rank{local_rank}.jsonl", args.backend)
        if world > 1:
            dist.barrier()
        if rank0:
            (out / ds / "done.json").write_text(json.dumps({"n_items": len(sub), "hours": round((time.time() - t0) / 3600, 3)}))
            log.info("external %s: %d items (%.2f h)", ds, len(sub), (time.time() - t0) / 3600)
        if world > 1:
            dist.barrier()
    if rank0:
        import pandas as pd
        recs = [json.loads(l) for f in sorted(out.glob("*/rank*.jsonl")) for l in open(f, encoding="utf-8")]
        table = pd.DataFrame(score_external(recs, xart))
        table.to_csv(out / "external_llm.csv", index=False)
        res["external"] = {"source_run": src_run, "source_epoch": best, "duplicates_excluded": meta["duplicates"],
                           "n_items": len(items)}
        with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_columns", 14):
            cols = [c for c in ("dataset", "task", "dialect", "n", "chrfpp", "bleu", "cer", "wer", "copy_chrfpp",
                                "source_accuracy", "accuracy") if c in table.columns]
            log.info("EXTERNAL (QLoRA)\n%s", table[cols].to_string(index=False))
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
    if args.preset in ("ablate", "external"):
        ablate_launch(args, n_gpu)
        return
    if args.preset == "capacity":
        capacity_launch(args, n_gpu)
        return
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_name = args.run_name or f"{args.preset or 'custom'}-{key}-s{args.seed}"
        a = args
        if args.preset in ("final",) + STEP6_TRAIN_PRESETS:
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
    (Path(args.output_dir) / "STEP6_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


def ablate_launch(args, n_gpu: int) -> None:
    """Inference-only ablations on a trained adapter; the LoRA shape is taken from that run's configuration."""
    import copy
    src = args.from_run or f"primary-{PRIMARY_MODEL}-s13"
    f = locate(src, "metrics.json", args.output_dir)
    if f is None:
        raise SystemExit(f"metrics.json of {src} not found: attach that run's output")
    m = json.loads(f.read_text())
    a = copy.copy(args)
    c = m["config"]["args"]
    for k in ("lora_r", "lora_alpha", "lora_dropout", "targets", "lr", "eff_batch"):
        setattr(a, k, c[k])
    a.ablate_from = src
    key = m["model"]
    log.info("ablations on %s (%s, epoch %s): %s", src, key, (m.get("epoch_selection") or {}).get("best_epoch"),
             args.variants or "all")
    run_single(a, key, args.run_name or (f"external-{src}" if getattr(args, "external", False) else f"ablate-{src}"), n_gpu)
    clear_model_cache(MODELS[key]["repo"])


def capacity_launch(args, n_gpu: int) -> None:
    """LoRA capacity and placement, one factor at a time, at the screening scale."""
    import copy
    key = PRIMARY_MODEL
    base, _ = final_config(key, args.output_dir)
    for name, change in CAPACITY:
        run = f"capacity-{key}-{name}"
        if find_metrics(run, args.output_dir):
            log.info("%s already finished -- skipped", run)
            continue
        a = copy.copy(args)
        for k, v in {**base, **change}.items():
            setattr(a, k, v)
        log.info("%s: %s", run, {**base, **change})
        run_single(a, key, run, n_gpu)
    clear_model_cache(MODELS[key]["repo"])


def compare_ablations(args) -> None:
    """One table of every step 10 ablation."""
    import pandas as pd
    rows = []
    cols = ("tier1_chrfpp_mean", "tier1_d2s_chrfpp_mean", "tier1_s2d_chrfpp_mean", "id_bn_macro_f1", "id_rom_macro_f1",
            "auto_chrfpp_mean", "auto_source_accuracy")
    ref_run = f"primary-{PRIMARY_MODEL}-s13"
    def metrics_of(run, rel="metrics.json", key="metrics"):
        f = locate(run, rel, args.output_dir)
        if f is None:
            return None
        d = json.loads(f.read_text())
        return d.get(key, d) if key else d
    ref = metrics_of(ref_run) or {}
    rows.append({"group": "reference", "variant": f"{ref_run} (dev_full)", **{c: ref.get(c) for c in cols}})
    for d in sorted(set(glob.glob(f"{args.output_dir}/runs/ablate-*") + glob.glob("/kaggle/input/**/runs/ablate-*", recursive=True))):
        run = Path(d).name
        for name, _, _ in ABLATIONS:
            m = metrics_of(run, f"abl_{name}/metrics.json", None)
            if m:
                rows.append({"group": run.replace("ablate-", "inference on "), "variant": name, **{c: m.get(c) for c in cols}})
    for preset in ("single-id", "single-mt", "no-examples", "data10", "data25", "data50"):
        run = f"{preset}-{PRIMARY_MODEL}-s13"
        m = metrics_of(run)
        if m:
            rows.append({"group": "training (dev_full)", "variant": preset, **{c: m.get(c) for c in cols}})
    cref = find_metrics(CAPACITY_REFERENCE, args.output_dir)
    for name in ["reference"] + [n for n, _ in CAPACITY]:
        r = cref if name == "reference" else find_metrics(f"capacity-{PRIMARY_MODEL}-{name}", args.output_dir)
        if r:
            rows.append({"group": "capacity (dev_select, 20%, 1 epoch)", "variant": "r32 all (step 05 l05)" if name == "reference" else name,
                         **{c: r["metrics"].get(c) for c in cols}})
    df = pd.DataFrame(rows)
    out = Path(args.output_dir, "ablations")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablations.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.max_rows", 100):
        log.info("STEP 10 ABLATIONS\n%s", df.to_string(index=False))


def run_single(args, key: str, run_name: str, n_gpu: int) -> bool:
    """One QLoRA training + evaluation run (torchrun, one process per GPU)."""
    import signal
    if True:
        backend = args.backend if args.backend != "auto" else MODELS[key]["backends"][0]
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", "--master_port=29541",
               str(Path(__file__).resolve()), "--worker", "--model", key, "--backend", backend, "--run-name", run_name]
        for k, v in vars(args).items():
            if k in ("models", "worker", "model", "backend", "run_name", "compare", "dry_run", "preset", "no_install",
                     "requirements", "models_list", "stage", "trials", "compare_search", "compare_final",
                     "compare_ablations", "from_run"):
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


def final_config(key: str, output_dir: str) -> tuple[dict, str]:
    f = Path(output_dir, "search", key, "search_selection.json")
    f = f if f.exists() else find_one(f"/kaggle/input/**/search/{key}/search_selection.json")
    if f is not None:
        return json.loads(f.read_text())["config"], str(f)
    if key in SELECTED_CONFIGS:
        return SELECTED_CONFIGS[key], "SELECTED_CONFIGS in this file (step 05 result)"
    raise SystemExit(f"no selected configuration for {key}: attach the step 05 selection output")


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
    p = argparse.ArgumentParser(description="BUC step 10: final QLoRA experiments on the primary model")
    p.add_argument("--preset", choices=list(PRESETS), help="named setting")
    p.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {list(MODELS)}")
    p.add_argument("--compare", action="store_true", help="apply the model-selection rule to finished select runs")
    p.add_argument("--stage", choices=["global", "local"], default="global", help="search stage (with --preset search)")
    p.add_argument("--trials", default="all", help="search trials to run in this session, e.g. 0-3 or 0,2,5")
    p.add_argument("--compare-search", action="store_true", help="select the best search configuration per model")
    p.add_argument("--no-save", action="store_true", help="do not save adapter/checkpoint (search trials)")
    p.add_argument("--epoch-select", action="store_true", help="save every epoch, choose the epoch on dev_select")
    p.add_argument("--final-evals", help="evaluations of the chosen epoch, e.g. 'dev_full:id,tier1;test_full:id,tier1,tier2'")
    p.add_argument("--mixture", help="training task mixture, e.g. 'id_bn:2,id_rom:1' (single-task adapters)")
    p.add_argument("--p-examples", type=float, help="probability of retrieved examples in a training prompt")
    p.add_argument("--p-hints", type=float, help="probability of lexicon hints in a training prompt")
    p.add_argument("--eval-k", type=int, help="retrieved examples at inference (default 3)")
    p.add_argument("--eval-hints", choices=["on", "off"], help="lexicon hints at inference (default on)")
    p.add_argument("--select-metric", choices=["tier1", "id"], default="tier1", help="epoch-selection metric")
    p.add_argument("--from-run", help="ablations: run whose adapter is evaluated (default: primary-gemma4-e4b-s13)")
    p.add_argument("--variants", help="ablations: comma-separated subset, e.g. A2-none,A2-random")
    p.add_argument("--ablate-from", help=argparse.SUPPRESS)
    p.add_argument("--external", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--external-dir", help="folder searched for the external datasets (default: /kaggle/input)")
    p.add_argument("--compare-ablations", action="store_true", help="summarize all step 10 ablations (CPU)")
    p.add_argument("--compare-final", action="store_true", help="choose the primary model (bootstrap rule)")
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
    if not args.preset and not (args.compare or args.compare_search or args.compare_final or args.compare_ablations
                                or args.worker or args.dry_run):
        args.preset = "primary"
    if args.preset:
        explicit = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
        for k, v in PRESETS[args.preset].items():
            if k not in explicit:
                setattr(args, k, v)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if args.preset in STEP6_TRAIN_PRESETS + ("ablate", "capacity", "external") and not any(x.startswith("--models") for x in sys.argv[1:]):
        args.models = PRIMARY_MODEL
    if args.compare or args.compare_search or args.compare_final or args.compare_ablations:
        ensure_cpu_packages(args)
    if args.compare:
        compare(args)
        return
    if args.compare_search:
        compare_search(args)
        return
    if args.compare_final:
        compare_final(args)
        return
    if args.compare_ablations:
        compare_ablations(args)
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

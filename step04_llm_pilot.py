#!/usr/bin/env python3
"""Pilot of the five LLMs under QLoRA: fp16 stability, memory, speed and prompt lengths on two T4 GPUs.

Usage:
    python step04_llm_pilot.py --artifacts-dir outputs
    python step04_llm_pilot.py --models qwen3-8b,gemma4-e4b
Offline test of the prompt code:
    python step04_llm_pilot.py --dry-run --artifacts-dir outputs --no-install
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
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
    "qwen3-8b":      {"repo": "Qwen/Qwen3-8B", "fallback": ["unsloth/Qwen3-8B"],
                      "chat": True, "max_len": 1024, "batch": 4},
    "llama31-8b":    {"repo": "meta-llama/Llama-3.1-8B-Instruct", "fallback": ["unsloth/Meta-Llama-3.1-8B-Instruct"],
                      "chat": True, "max_len": 1024, "batch": 4},
    "ministral3-8b": {"repo": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "fallback": [],
                      "chat": True, "max_len": 768, "batch": 4, "backends": ["hf"]},
    "gemma4-e4b":    {"repo": "google/gemma-4-E4B-it", "fallback": ["unsloth/gemma-4-E4B-it"],
                      "chat": True, "max_len": 512, "batch": 4, "backends": ["hf", "unsloth"]},
    "titullm-3b":    {"repo": "hishab/titulm-llama-3.2-3b-v2.0", "fallback": [],
                      "chat": False, "max_len": 512, "batch": 8},
}
LORA = {"r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
TRAIN = {"lr": 2e-4, "warmup_frac": 0.03, "max_grad_norm": 1.0, "weight_decay": 0.0}
PILOT = {"n_length_examples": 2000, "n_generation_prompts": 16, "max_new_tokens": 96,
         "timeout_min_per_model": 90}
MIXTURE = [
    ("id_bn", 2), ("id_rom", 1), ("tier1", 3), ("tier2", 1), ("tier3", 1)]
SAMPLING = {"p_examples": 0.5, "k_max": 3, "p_hints": 0.5, "max_hints": 5, "p_auto_source": 0.5}
PROMPT_VERSION = "llm-prompt-v2"
SYSTEM = "You are an expert on Bangla and its regional dialects. Follow the requested output format exactly."
EPOCH_EXAMPLES = 4251 * 8
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


def prepare_environment(args) -> dict:
    """Install core + GPU requirements (unsloth with --no-deps) before any third-party import."""
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 required, found {sys.version.split()[0]}")
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
                                         finetune_attention_modules=True, finetune_mlp_modules=True, **peft_kw)
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


def worker(args) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    rank0 = local_rank == 0
    cfg = MODELS[args.model]
    out = Path(args.output_dir, "pilot", f"{args.model}__{args.backend}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    res = {"model": args.model, "backend": args.backend, "world_size": world, "status": "started",
           "prompt_version": PROMPT_VERSION}
    token = os.environ.get("HF_TOKEN") or None
    try:
        art = Artifacts(Path(args.artifacts_dir))
        t0 = time.time()
        loader = load_unsloth if args.backend == "unsloth" else load_hf
        errors = []
        for repo in [cfg["repo"]] + cfg["fallback"]:
            try:
                model, tok = loader(repo, cfg["max_len"], local_rank, token)
                res["repo_used"] = repo
                break
            except Exception as e:
                errors.append(f"{repo}: {type(e).__name__}: {str(e)[:300]}")
                if rank0:
                    log.warning("load failed for %s: %s", repo, errors[-1])
        else:
            raise RuntimeError("all repositories failed: " + " | ".join(errors))
        res["load_errors"] = errors
        res["load_s"] = round(time.time() - t0, 1)
        res["mem_after_load_gib"] = round(torch.cuda.memory_allocated(local_rank) / 2**30, 2)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        trainable = [p for p in model.parameters() if p.requires_grad]
        res["trainable_params_m"] = round(sum(p.numel() for p in trainable) / 1e6, 2)
        qc = getattr(getattr(model, "config", None), "quantization_config", None)
        res["compute_dtype"] = str(getattr(qc, "bnb_4bit_compute_dtype", None)) if qc is not None else None
        res["lora_param_dtype"] = str(trainable[0].dtype) if trainable else None
        enc = Encoder(tok, cfg["chat"], args.model)

        rng = random.Random(SEED)
        if rank0:
            demo_tasks = sample_tasks(art, rng.sample(art.train_serials, 300), rng)[:PILOT["n_length_examples"]]
            with_ex = [build_prompt(art, t, SAMPLING["k_max"], True) for t in demo_tasks]
            no_ex = [build_prompt(art, t, 0, False) for t in demo_tasks]
            mixed = realize(art, demo_tasks, random.Random(SEED))
            res["lengths"] = {"no_examples": length_stats(enc, no_ex, cfg["max_len"]),
                              "k3_with_hints": length_stats(enc, with_ex, cfg["max_len"]),
                              "training_mixture": length_stats(enc, mixed, cfg["max_len"])}
            log.info("[%s] prompt lengths: %s", args.model, {k: (v["p50"], v["p95"], v["max"]) for k, v in res["lengths"].items()})

        serials = art.train_serials[local_rank::world]
        pairs = realize(art, sample_tasks(art, serials, random.Random(SEED + local_rank)), random.Random(SEED + 100 + local_rank))
        pad_id = enc.tok.pad_token_id if enc.tok.pad_token_id is not None else enc.tok.eos_token_id
        batch, steps, L = cfg["batch"], args.steps, cfg["max_len"]
        encoded = []
        for p, t in pairs[:steps * batch]:
            ids, n_prompt = enc.encode(p, t)
            if len(ids) > L:
                cut = len(ids) - L
                ids, n_prompt = ids[cut:], max(0, n_prompt - cut)
            encoded.append((ids, n_prompt))
        order, batches, win = list(range(len(encoded))), [], batch * 32
        for w0 in range(0, len(order), win):
            chunk = sorted(order[w0:w0 + win], key=lambda i: len(encoded[i][0]))
            batches += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
        random.Random(SEED + 7 + local_rank).shuffle(batches)

        def get_batch(i):
            items = [encoded[j] for j in batches[i % len(batches)]]
            m = max(len(x) for x, _ in items)
            inp = torch.full((len(items), m), pad_id, dtype=torch.long)
            lab = torch.full((len(items), m), -100, dtype=torch.long)
            att = torch.zeros((len(items), m), dtype=torch.long)
            for j, (ids, n_prompt) in enumerate(items):
                inp[j, :len(ids)] = torch.tensor(ids)
                att[j, :len(ids)] = 1
                lab[j, n_prompt:len(ids)] = torch.tensor(ids[n_prompt:])
            return inp.cuda(local_rank), att.cuda(local_rank), lab.cuda(local_rank)

        import bitsandbytes as bnb
        model.train()
        ddp_model = model
        if world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            try:
                ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True,
                                init_sync=False)
            except TypeError:
                ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
            for p_ in trainable:
                dist.broadcast(p_.data, src=0)
        opt = bnb.optim.PagedAdamW8bit(trainable, lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
        warm = max(1, int(TRAIN["warmup_frac"] * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats(local_rank)
        log_rows, skipped, nonfinite, tok_count, times = [], 0, 0, 0, []
        for step in range(steps):
            inp, att, lab = get_batch(step)
            torch.cuda.synchronize(local_rank)
            t1 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = ddp_model(input_ids=inp, attention_mask=att, labels=lab).loss
            finite = bool(torch.isfinite(loss).item())
            nonfinite += int(not finite)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(trainable, TRAIN["max_grad_norm"])
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
            sched.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(local_rank)
            dt = time.time() - t1
            ntok = int(att.sum().item())
            if step >= 10:
                times.append(dt)
                tok_count += ntok
            log_rows.append({"step": step, "loss": float(loss.item()), "grad_norm": float(gn),
                             "scale": float(scaler.get_scale()), "seconds": round(dt, 3), "tokens": ntok})
            if rank0 and (step % 20 == 0 or step == steps - 1):
                log.info("[%s|%s] step %d/%d loss %.4f gnorm %.3f scale %.0f %.2fs", args.model, args.backend,
                         step, steps, loss.item(), float(gn), scaler.get_scale(), dt)

        stats = torch.tensor([tok_count, sum(times), skipped, nonfinite], dtype=torch.float64, device=f"cuda:{local_rank}")
        peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank) / 2**30,
                             torch.cuda.max_memory_reserved(local_rank) / 2**30], device=f"cuda:{local_rank}")
        if world > 1:
            dist.all_reduce(stats)
            peaks = [torch.zeros_like(peak) for _ in range(world)]
            dist.all_gather(peaks, peak)
        else:
            peaks = [peak]
        losses = [r["loss"] for r in log_rows]
        mean_step = float(np.mean(times)) if times else float("nan")
        res.update({
            "steps": steps, "per_device_batch": batch, "max_len": L,
            "seconds_per_step_mean": round(mean_step, 3),
            "tokens_per_second_total": round(float(stats[0] / (stats[1] / world)), 1) if stats[1] > 0 else None,
            "fp16_overflow_skips_total": int(stats[2]), "nonfinite_losses_total": int(stats[3]),
            "peak_alloc_gib": [round(float(p[0]), 2) for p in peaks],
            "peak_reserved_gib": [round(float(p[1]), 2) for p in peaks],
            "loss_first20": round(float(np.nanmean(losses[:20])), 4),
            "loss_last20": round(float(np.nanmean(losses[-20:])), 4),
        })
        if rank0 and res.get("tokens_per_second_total"):
            mean_len = res["lengths"]["training_mixture"]["mean"]
            res["est_hours_per_epoch"] = round(mean_len * EPOCH_EXAMPLES / res["tokens_per_second_total"] / 3600, 2)
        if rank0:
            import pandas as pd
            pd.DataFrame(log_rows).to_csv(out / "train_log.csv", index=False)

        if world > 1:
            dist.barrier()
        if rank0:
            res["generation"] = generation_check(model, enc, art, args.backend)
        if world > 1:
            dist.barrier()
        res["status"] = "ok"
    except Exception as e:
        res["status"] = "failed"
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        if rank0:
            log.error("[%s|%s] FAILED: %s", args.model, args.backend, res["error"])
    finally:
        if rank0:
            (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if world > 1 and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def generation_check(model, enc: Encoder, art: Artifacts, backend: str) -> dict:
    """Greedy answers for 16 fixed dev prompts (8 identification, 8 translation); JSON validity."""
    import torch
    if backend == "unsloth":
        try:
            from unsloth import FastModel
            FastModel.for_inference(model)
        except Exception:
            pass
    model.eval()
    rng = random.Random(SEED)
    dev = [int(s) for s in art.table.index[art.table["split"] == "dev"]]
    tasks = ([{"kind": "id", "serial": rng.choice(dev), "field": f"{rng.choice(BANGLA)}:bn"} for _ in range(8)]
             + [{"kind": "mt", "serial": rng.choice(dev), "src": f"{rng.choice(DIALECTS)}:bn", "tgt": "std:bn",
                 "auto": i % 2 == 1} for i in range(8)])
    rows, valid = [], 0
    for t in tasks:
        prompt, target = build_prompt(art, t, 0, False)
        ids = torch.tensor([enc.prompt_ids(prompt)], device=next(model.parameters()).device)
        src_len = len(enc.tok(target, add_special_tokens=False)["input_ids"])
        budget = 32 if t["kind"] == "id" else min(512, 2 * src_len + 48)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget,
                                 do_sample=False, pad_token_id=enc.tok.pad_token_id or enc.tok.eos_token_id)
        text = enc.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"\{.*?\}", text, re.S)
        ok = False
        if m:
            try:
                json.loads(m.group(0))
                ok = True
            except Exception:
                pass
        valid += ok
        rows.append({"task": t, "reference": target, "output": text, "valid_json": ok})
    return {"schema_valid_rate": valid / len(tasks), "samples": rows}


def clear_model_cache(repo: str) -> None:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for d in hub.glob("models--" + repo.replace("/", "--")):
        shutil.rmtree(d, ignore_errors=True)


def launch(args, env_report: dict) -> None:
    import torch
    token = hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        log.warning("HF_TOKEN not found: gated models (Llama, possibly Gemma) will use fallback mirrors or fail")
    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise SystemExit("no GPU: set Accelerator to 'GPU T4 x2'")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for key in models:
        for backend in (MODELS[key].get("backends", ["unsloth", "hf"]) if args.backend == "auto" else [args.backend]):
            summary = Path(args.output_dir, "pilot", f"{key}__{backend}", "summary.json")
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpu}", "--master_port=29531",
                   str(Path(__file__).resolve()), "--worker", "--model", key, "--backend", backend,
                   "--steps", str(args.steps), "--output-dir", args.output_dir, "--artifacts-dir", args.artifacts_dir]
            log.info("===== %s | backend %s | %d GPU(s) =====", key, backend, n_gpu)
            t0 = time.time()
            proc = subprocess.Popen(cmd, start_new_session=True)
            try:
                proc.wait(timeout=PILOT["timeout_min_per_model"] * 60)
            except subprocess.TimeoutExpired:
                log.error("%s | %s timed out after %d min", key, backend, PILOT["timeout_min_per_model"])
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            res = json.loads(summary.read_text()) if summary.exists() else {"model": key, "backend": backend,
                                                                              "status": "no summary (crash or timeout)"}
            res["wall_clock_s"] = round(time.time() - t0, 1)
            rows.append(res)
            log.info("%s | %s -> %s", key, backend, res.get("status"))
            for repo in [MODELS[key]["repo"]] + MODELS[key]["fallback"]:
                clear_model_cache(repo)
            if res.get("status") == "ok":
                break
    write_summary(args, rows, env_report)


def write_summary(args, rows: list[dict], env_report: dict) -> None:
    import pandas as pd
    out = Path(args.output_dir, "pilot")
    flat = []
    for r in rows:
        L = r.get("lengths", {}).get("training_mixture", {})
        L3 = r.get("lengths", {}).get("k3_with_hints", {})
        gen = r.get("generation", {})
        flat.append({"model": r.get("model"), "backend": r.get("backend"), "status": r.get("status"),
                     "repo_used": r.get("repo_used"), "load_s": r.get("load_s"),
                     "peak_alloc_gib_gpu0": (r.get("peak_alloc_gib") or [None])[0],
                     "peak_reserved_gib_gpu0": (r.get("peak_reserved_gib") or [None])[0],
                     "s_per_step": r.get("seconds_per_step_mean"), "tokens_per_s_total": r.get("tokens_per_second_total"),
                     "est_h_per_epoch": r.get("est_hours_per_epoch"),
                     "overflow_skips": r.get("fp16_overflow_skips_total"), "nonfinite_losses": r.get("nonfinite_losses_total"),
                     "loss_first20": r.get("loss_first20"), "loss_last20": r.get("loss_last20"),
                     "len_mix_p95": L.get("p95"), "len_mix_max": L.get("max"), "len_k3_p95": L3.get("p95"),
                     "len_k3_max": L3.get("max"), "pct_k3_over_max_len": L3.get("pct_over_max_len"),
                     "schema_valid_rate": gen.get("schema_valid_rate"), "wall_clock_s": r.get("wall_clock_s"),
                     "error": (r.get("error") or r.get("status") if r.get("status") != "ok" else None)})
    df = pd.DataFrame(flat)
    df.to_csv(out / "pilot_summary.csv", index=False)
    manifest = {"script": "step04_llm_pilot.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "finished_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args), "models": MODELS,
                "lora": LORA, "train": TRAIN, "pilot": PILOT, "sampling": SAMPLING, "mixture": MIXTURE,
                "prompt_version": PROMPT_VERSION, "environment": env_report, "results": rows}
    (out / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log.info("PILOT SUMMARY\n%s", df.drop(columns=["error"]).to_string(index=False))
    (Path(args.output_dir) / "STEP2_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


def dry_run(args) -> None:
    art = Artifacts(Path(args.artifacts_dir))
    rng = random.Random(SEED)
    tasks = sample_tasks(art, art.train_serials[:50], rng)
    pairs = realize(art, tasks, rng)
    kinds = {}
    for t in tasks:
        kinds[t["kind"] + ("-auto" if t.get("auto") else "")] = kinds.get(t["kind"] + ("-auto" if t.get("auto") else ""), 0) + 1
    log.info("dry run: %d tasks from 50 serials: %s", len(tasks), kinds)
    for p, t in pairs[:3]:
        log.info("----- PROMPT -----\n%s\n----- TARGET -----\n%s", p, t)
    words = [len(p.split()) + len(t.split()) for p, t in pairs]
    log.info("whitespace words per example: mean %.1f, max %d", sum(words) / len(words), max(words))


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 04: LLM pilot (QLoRA, fp16, 2x T4)")
    p.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {list(MODELS)}")
    p.add_argument("--backend", default="auto", choices=["auto", "unsloth", "hf"],
                   help="auto = Unsloth, then Hugging Face + PEFT if Unsloth fails")
    p.add_argument("--steps", type=int, default=200, help="optimizer steps per model")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="build prompts only (no GPU, no model)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if not args.artifacts_dir:
        hit = find_one("/kaggle/input/**/data/aligned.parquet")
        if hit is None:
            raise SystemExit("step-02 output not found under /kaggle/input: attach the output of step02_prepare_data.py")
        args.artifacts_dir = str(hit.parents[1])
    if args.worker:
        worker(args)
        return
    env_report = {} if args.dry_run else prepare_environment(args)
    if args.dry_run:
        dry_run(args)
        return
    unknown = set(args.models.split(",")) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(MODELS)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("artifacts: %s | output: %s | disk free: %.0f GB", args.artifacts_dir, args.output_dir,
             shutil.disk_usage("/").free / 1e9)
    launch(args, env_report)


if __name__ == "__main__":
    main()

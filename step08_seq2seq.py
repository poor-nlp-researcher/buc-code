#!/usr/bin/env python3
"""Sequence-to-sequence translation baselines: BanglaT5, mBART-50 and NLLB-200-distilled-1.3B.

Tier-1 and Tier-2 directions; the target variety is given as a control tag. Search over six settings on 20% of
the training data, then three seeds of the selected setting with beam-search decoding of dev_full and test_full.
BanglaT5 runs in a separate environment (requirements-legacy.txt) that the script creates.

Usage:
    python step08_seq2seq.py --models banglat5
Offline test:
    python step08_seq2seq.py --dry-run --artifacts-dir outputs --no-install
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

DIALECTS = ["nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
TIER1 = [(f"{d}:bn", "std:bn") for d in DIALECTS] + [("std:bn", f"{d}:bn") for d in DIALECTS]
TIER2 = ([(f"{d}:bn", "en:lat") for d in DIALECTS] + [("en:lat", f"{d}:bn") for d in DIALECTS]
         + [("std:bn", "en:lat"), ("en:lat", "std:bn"), ("cm:mix", "std:bn")])
ALL_DIRS = TIER1 + TIER2
MODELS = {
    "banglat5": {"repo": "csebuetnlp/banglat5", "family": "t5", "env": "legacy", "fp16": False, "optim": "adamw",
                 "lr": [1e-4, 3e-4, 5e-4], "per_device_batch": 16, "grad_ckpt": False, "directions": ALL_DIRS},
    "mbart50": {"repo": "facebook/mbart-large-50-many-to-many-mmt", "family": "mbart", "env": "main", "fp16": True,
                "optim": "adamw", "lr": [1e-5, 3e-5, 5e-5], "per_device_batch": 16, "grad_ckpt": False, "directions": ALL_DIRS},
    "nllb-1.3b": {"repo": "facebook/nllb-200-distilled-1.3B", "family": "nllb", "env": "main", "fp16": True,
                  "optim": "adafactor", "lr": [3e-5, 1e-4, 3e-4], "per_device_batch": 8, "grad_ckpt": True,
                  "directions": ALL_DIRS},
    "indictrans2-indic-indic": {"repo": "ai4bharat/indictrans2-indic-indic-1B", "family": "indictrans2", "env": "legacy",
                                "fp16": True, "optim": "adafactor", "lr": [3e-5, 1e-4, 3e-4], "per_device_batch": 8,
                                "grad_ckpt": True, "directions": TIER1 + [("cm:mix", "std:bn")]},
    "indictrans2-en-indic": {"repo": "ai4bharat/indictrans2-en-indic-1B", "family": "indictrans2", "env": "legacy",
                             "fp16": True, "optim": "adafactor", "lr": [3e-5, 1e-4, 3e-4], "per_device_batch": 8,
                             "grad_ckpt": True, "directions": [d for d in TIER2 if d[0] == "en:lat"]},
    "indictrans2-indic-en": {"repo": "ai4bharat/indictrans2-indic-en-1B", "family": "indictrans2", "env": "legacy",
                             "fp16": True, "optim": "adafactor", "lr": [3e-5, 1e-4, 3e-4], "per_device_batch": 8,
                             "grad_ckpt": True, "directions": [d for d in TIER2 if d[1] == "en:lat"]},
}
EFF_BATCH = [16, 32]
SEARCH = {"train_fraction": 0.2, "max_epochs": 3}
FINAL = {"max_epochs": 12, "patience": 2, "beams": 4}
LR_EXTENSION = {"banglat5": {"up": 1e-3, "down": 5e-5}, "mbart50": {"up": 1e-4, "down": 5e-6},
                "nllb-1.3b": {"up": 1e-3, "down": 1e-5}, "indictrans2-indic-indic": {"up": 1e-3, "down": 1e-5},
                "indictrans2-en-indic": {"up": 1e-3, "down": 1e-5}, "indictrans2-indic-en": {"up": 1e-3, "down": 1e-5}}
BOUNDARY_MIN_GAIN = 0.5
SELECTION_VERSION = "tier1-v2"


def sel_metric(key: str) -> str:
    return "tier1" if any(d in TIER1 for d in MODELS[key]["directions"]) else "mean_all"


def sel_history_field(key: str) -> str:
    return "dev_select_tier1" if sel_metric(key) == "tier1" else "dev_select_chrfpp"


def sel_score(r: dict, key: str) -> float:
    """Best value of the selection metric over the recorded epochs."""
    vals = [h.get(sel_history_field(key)) for h in r.get("history", [])]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else float("-inf")


JOB_HOURS = {"banglat5": 2.3, "mbart50": 2.8, "nllb-1.3b": 5.0, "indictrans2-indic-indic": 4.5,
             "indictrans2-en-indic": 1.5, "indictrans2-indic-en": 1.5}
TRAIN = {"warmup_frac": 0.05, "weight_decay": 0.01, "max_grad_norm": 1.0, "max_src": 128, "max_tgt": 128,
         "eval_batch_greedy": 48, "eval_batch_beam": 24}
SEEDS = [13, 42, 2026]
LANG = {"nllb": {"bn": "ben_Beng", "en": "eng_Latn"}, "mbart": {"bn": "bn_IN", "en": "en_XX"},
        "indictrans2": {"bn": "ben_Beng", "en": "eng_Latn"}}
log = logging.getLogger("buc")


def tag(field: str) -> str:
    code, script = field.split(":")
    return f"[{code}-rom]" if script == "rom" else f"[{code}]"


def lang_of(field: str) -> str:
    return "en" if field == "en:lat" else "bn"


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


GATED = ("ai4bharat/indictrans2",)


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def pins(req: Path) -> dict:
    out = {}
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def prepare_environment(args) -> dict:
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise SystemExit(f"Python 3.12 or 3.13 required, found {sys.version.split()[0]}")
    if sys.version_info[:2] == (3, 13):
        log.info("Python %s: newer Kaggle image (the pinned packages install unchanged; versions are recorded)",
                 sys.version.split()[0])
    req = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt")
    if not req or not req.exists():
        log.warning("requirements.txt not found; versions not checked")
        return {}
    p = pins(req)

    def bad():
        out = {}
        for k, v in p.items():
            try:
                have = metadata.version(k)
            except metadata.PackageNotFoundError:
                have = None
            if have != v:
                out[k] = [v, have]
        return out
    mism = bad()
    if mism and not args.no_install:
        log.info("installing pinned core requirements (%d packages differ) ...", len(mism))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
        mism = bad()
    return {"requirements": str(req), "mismatched": mism}


LEGACY_DIR = Path("/tmp/buc_legacy_site")


def legacy_env(args) -> dict:
    """Environment variables for the legacy workers."""
    env = dict(os.environ, PYTHONPATH=str(LEGACY_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    marker = LEGACY_DIR / ".ready"
    if marker.exists():
        return env
    req = Path(args.requirements).parent / "requirements-legacy.txt" if args.requirements else \
        find_one("/kaggle/input/**/requirements-legacy.txt")
    if not req or not Path(req).exists():
        raise SystemExit("requirements-legacy.txt not found: place it next to this script")
    log.info("installing the legacy packages (Transformers 4.46, IndicTransToolkit) into %s ...", LEGACY_DIR)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(LEGACY_DIR), "-r", str(req)],
                   check=True)
    out = subprocess.run([sys.executable, "-c", "import transformers, IndicTransToolkit; print(transformers.__version__)"],
                         capture_output=True, text=True, env=env)
    if out.returncode != 0 or not out.stdout.strip().startswith("4.46"):
        raise SystemExit(f"legacy environment is broken: {out.stdout[-300:]} {out.stderr[-1500:]}")
    log.info("legacy environment ready (transformers %s)", out.stdout.strip())
    marker.write_text("ok")
    return env


def load_table(root: Path):
    import pandas as pd
    return pd.read_parquet(root / "data" / "aligned.parquet")


def train_examples(table, key: str, fraction: float, epoch: int, seed: int, rank: int, world: int) -> list[dict]:
    """One epoch of training pairs."""
    serials = sorted(table.index[table["split"] == "train"].tolist())
    if fraction < 1.0:
        serials = sorted(random.Random(SEEDS[0]).sample(serials, int(round(fraction * len(serials)))))
    rng = random.Random(seed * 1000 + epoch)
    mine = set(MODELS[key]["directions"])
    out = []
    for s in serials:
        draws = [rng.choice(TIER1) for _ in range(3)] + [rng.choice(TIER2)]
        for src, tgt in draws:
            if (src, tgt) in mine:
                out.append({"src_field": src, "tgt_field": tgt, "src": table.at[s, src], "ref": table.at[s, tgt]})
    rng.shuffle(out)
    return out


def eval_items(root: Path, name: str, key: str) -> list[dict]:
    dirs = set(MODELS[key]["directions"])
    items = []
    for line in open(root / "data" / "eval" / name / "mt.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["tier"] in ("tier1", "tier2") and (r["src_field"], r["tgt_field"]) in dirs:
            items.append(r)
    return items


class Family:
    """Wraps tokenizer + language handling of one model family so the training loop is shared."""

    def __init__(self, key: str, tok):
        self.key, self.cfg, self.tok = key, MODELS[key], tok
        self.fam = self.cfg["family"]
        self.norm = (lambda s: s)
        if self.fam == "t5":
            from normalizer import normalize
            self.norm = lambda s: normalize(s) if s else s
        if self.fam == "indictrans2":
            self.ip_train, self.ip_infer = indic_processor(False), indic_processor(True)

    def source_text(self, it) -> str:
        return f"{tag(it['src_field'])} {tag(it['tgt_field'])} {self.norm(it['src'])}"

    def codes(self, it):
        L = LANG.get(self.fam)
        return (L[lang_of(it["src_field"])], L[lang_of(it["tgt_field"])]) if L else (None, None)

    def encode_train(self, it) -> tuple[list[int], list[int]]:
        tok, M = self.tok, TRAIN
        if self.fam == "t5":
            x = tok(self.source_text(it), truncation=True, max_length=M["max_src"])["input_ids"]
            y = tok(self.norm(it["ref"]), truncation=True, max_length=M["max_tgt"])["input_ids"]
            return x, y
        s, t = self.codes(it)
        if self.fam in ("nllb", "mbart"):
            tok.src_lang, tok.tgt_lang = s, t
            enc = tok(self.source_text(it), text_target=it["ref"], truncation=True, max_length=M["max_src"])
            return enc["input_ids"], enc["labels"][:M["max_tgt"]]
        src = self.ip_train.preprocess_batch([self.source_text(it)], src_lang=s, tgt_lang=t)[0]
        tgt = self.ip_train.preprocess_batch([it["ref"]], src_lang=t, tgt_lang=t, is_target=True)[0]
        x = tok(src, truncation=True, max_length=M["max_src"])["input_ids"]
        with tok.as_target_tokenizer():
            y = tok(tgt, truncation=True, max_length=M["max_tgt"])["input_ids"]
        return x, y

    def generate(self, model, items, beams: int, device) -> list[str]:
        """Translations for items that share source and target language codes (batched by the caller)."""
        import torch
        tok = self.tok
        s, t = self.codes(items[0])
        kw = dict(num_beams=beams, do_sample=False, use_cache=True,
                  max_new_tokens=min(256, 3 * max(len(i["src"].split()) for i in items) + 24))
        if self.fam == "indictrans2":
            srcs = self.ip_infer.preprocess_batch([self.source_text(i) for i in items], src_lang=s, tgt_lang=t)
            enc = tok(srcs, truncation=True, max_length=TRAIN["max_src"], padding="longest", return_tensors="pt").to(device)
        elif self.fam in ("nllb", "mbart"):
            tok.src_lang = s
            enc = tok([self.source_text(i) for i in items], truncation=True, max_length=TRAIN["max_src"],
                      padding="longest", return_tensors="pt").to(device)
            kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(t)
        else:
            enc = tok([self.source_text(i) for i in items], truncation=True, max_length=TRAIN["max_src"],
                      padding="longest", return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.cfg["fp16"]):
            out = model.generate(**enc, **kw)
        if self.fam == "indictrans2":
            with tok.as_target_tokenizer():
                txt = tok.batch_decode(out.detach().cpu().tolist(), skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
            return self.ip_infer.postprocess_batch(txt, lang=t)
        return [x.strip() for x in tok.batch_decode(out, skip_special_tokens=True)]


def indic_processor(inference: bool):
    """IndicTransToolkit's preprocessor."""
    import importlib.util
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except Exception:
        for name in [m for m in sys.modules if m == "IndicTransToolkit" or m.startswith("IndicTransToolkit.")]:
            del sys.modules[name]
        spec = importlib.util.find_spec("IndicTransToolkit")
        if spec is None:
            raise ImportError("IndicTransToolkit is not installed")
        pkg = types.ModuleType("IndicTransToolkit")
        pkg.__path__ = list(spec.submodule_search_locations)
        pkg.__spec__ = spec
        sys.modules["IndicTransToolkit"] = pkg
        from IndicTransToolkit.processor import IndicProcessor
    return IndicProcessor(inference=inference)


def score(items: list[dict], hyps: list[str], dirs) -> dict:
    """Corpus chrF++ / chrF per direction; means over Tier 1, Dialect->Standard, Standard->Dialect, Tier 2."""
    import numpy as np
    from sacrebleu.metrics import CHRF
    chrfpp, chrf = CHRF(word_order=2), CHRF(word_order=0)
    per = {}
    for d in dirs:
        idx = [i for i, it in enumerate(items) if (it["src_field"], it["tgt_field"]) == tuple(d)]
        if not idx:
            continue
        h, r = [hyps[i] for i in idx], [[items[i]["ref"] for i in idx]]
        per[f"{d[0]}>{d[1]}"] = {"chrF++": round(chrfpp.corpus_score(h, r).score, 2), "chrF": round(chrf.corpus_score(h, r).score, 2),
                                 "copy_rate": round(100 * float(np.mean([hyps[i].strip() == items[i]["src"].strip() for i in idx])), 2),
                                 "n": len(idx)}

    def mean(sel):
        v = [x["chrF++"] for k, x in per.items() if sel(tuple(k.split(">")))]
        return round(float(np.mean(v)), 2) if v else None
    return {"per_direction": per, "mean_all": mean(lambda d: True), "tier1": mean(lambda d: d in TIER1),
            "d2s": mean(lambda d: d in TIER1 and d[1] == "std:bn"), "s2d": mean(lambda d: d in TIER1 and d[0] == "std:bn"),
            "tier2": mean(lambda d: d in TIER2), "signature": str(chrfpp.get_signature())}


def translate(fam: Family, model, items: list[dict], beams: int, device) -> list[str]:
    """Group by language pair, sort by length, batch; returns hypotheses aligned with items."""
    bs = TRAIN["eval_batch_beam"] if beams > 1 else TRAIN["eval_batch_greedy"]
    order = sorted(range(len(items)), key=lambda i: (fam.codes(items[i]), len(items[i]["src"])))
    hyps = [""] * len(items)
    model.eval()
    i = 0
    while i < len(order):
        key = fam.codes(items[order[i]])
        j = i
        while j < len(order) and j - i < bs and fam.codes(items[order[j]]) == key:
            j += 1
        batch = [order[k] for k in range(i, j)]
        for b, h in zip(batch, fam.generate(model, [items[k] for k in batch], beams, device)):
            hyps[b] = h
        i = j
    return hyps


def run_job(job: dict, args) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    key, seed = job["model"], job["seed"]
    cfg = MODELS[key]
    device = "cuda:0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(args.artifacts_dir)
    remote = cfg["family"] == "indictrans2"
    tok_kw = {"trust_remote_code": True} if remote else {}
    if cfg["family"] == "t5":
        tok_kw["use_fast"] = False
    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(cfg["repo"], token=token, **tok_kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["repo"], trust_remote_code=remote, token=token).to(device)
    if cfg["grad_ckpt"]:
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        except Exception as e:
            log.warning("gradient checkpointing unavailable for %s: %s", key, e)
    fam = Family(key, tok)
    table = load_table(root)
    stage = job["stage"]
    fraction = SEARCH["train_fraction"] if stage == "search" else 1.0
    max_epochs = SEARCH["max_epochs"] if stage == "search" else FINAL["max_epochs"]
    dev = eval_items(root, "dev_select", key)
    pb = cfg["per_device_batch"]
    accum = max(1, job["eff_batch"] // pb)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optim"] == "adafactor":
        opt = torch.optim.Adafactor(params, lr=job["lr"], weight_decay=0.0)
    else:
        no_decay = ("bias", "layer_norm", "LayerNorm", "layernorm")
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
             "weight_decay": TRAIN["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
             "weight_decay": 0.0}], lr=job["lr"])
    n_per_epoch = len(train_examples(table, key, fraction, 0, seed, 0, 1))
    total = max(1, max_epochs * n_per_epoch // (pb * accum))
    warm = max(1, int(TRAIN["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"])
    pad_id = tok.pad_token_id
    history, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    truncated = 0
    for ep in range(max_epochs):
        exs = train_examples(table, key, fraction, ep, seed, 0, 1)
        enc = [fam.encode_train(e) for e in exs]
        truncated += sum(len(x) >= TRAIN["max_src"] or len(y) >= TRAIN["max_tgt"] for x, y in enc)
        order = list(range(len(enc)))
        rng = random.Random(seed + ep)
        rng.shuffle(order)
        batches = []
        for w0 in range(0, len(order), pb * 50):
            chunk = sorted(order[w0:w0 + pb * 50], key=lambda i: len(enc[i][0]) + len(enc[i][1]))
            batches += [chunk[j:j + pb] for j in range(0, len(chunk), pb)]
        rng.shuffle(batches)
        model.train()
        losses, skipped = [], 0
        for bi, b in enumerate(batches):
            xs, ys = [enc[i][0] for i in b], [enc[i][1] for i in b]
            mx, my = max(map(len, xs)), max(map(len, ys))
            inp = torch.full((len(b), mx), pad_id, dtype=torch.long)
            att = torch.zeros((len(b), mx), dtype=torch.long)
            lab = torch.full((len(b), my), -100, dtype=torch.long)
            for k, (x, y) in enumerate(zip(xs, ys)):
                inp[k, :len(x)], att[k, :len(x)], lab[k, :len(y)] = torch.tensor(x), 1, torch.tensor(y)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["fp16"]):
                loss = model(input_ids=inp.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            scaler.scale(loss / accum).backward()
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, TRAIN["max_grad_norm"])
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped += int(cfg["fp16"] and scaler.get_scale() < before)
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        hyps = translate(fam, model, dev, beams=1, device=device)
        sc = score(dev, hyps, cfg["directions"])
        history.append({"epoch": ep + 1, "train_loss": round(float(np.nanmean(losses)), 4), "fp16_skips": skipped,
                        "nonfinite_losses": int(sum(not np.isfinite(x) for x in losses)), "dev_select_chrfpp": sc["mean_all"],
                        "dev_select_sel": sc[sel_metric(key)],
                        "dev_select_tier1": sc["tier1"], "dev_select_tier2": sc["tier2"],
                        "minutes": round((time.time() - t0) / 60, 1)})
        log.info("[%s %s lr=%g eb=%d seed=%d] epoch %d loss %.4f dev_select all %.2f (T1 %s, T2 %s; selection: %s)", key, stage,
                 job["lr"], job["eff_batch"], seed, ep + 1, np.nanmean(losses), sc["mean_all"], sc["tier1"], sc["tier2"],
                 sel_metric(key))
        if sc[sel_metric(key)] > best:
            best, bad = sc[sel_metric(key)], 0
            if stage == "final":
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if stage == "final" and bad >= FINAL["patience"]:
                break
    res = {**job, "best_dev_select_sel": best, "selection_metric": f"{sel_metric(key)}/{SELECTION_VERSION}",
           "best_epoch": max(history, key=lambda h: h["dev_select_sel"])["epoch"],
           "history": history, "max_epochs": max_epochs, "train_minutes": round((time.time() - t0) / 60, 1),
           "examples_per_epoch": n_per_epoch,
           "share_truncated": round(truncated / max(1, n_per_epoch * len(history)), 5),
           "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
    if stage == "final":
        model.load_state_dict(best_state)
        out = Path(job["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        res["evaluation"] = {}
        for split in ("dev_full", "test_full"):
            items = eval_items(root, split, key)
            t1 = time.time()
            hyps = translate(fam, model, items, beams=FINAL["beams"], device=device)
            res["evaluation"][split] = score(items, hyps, cfg["directions"])
            res["evaluation"][split]["decode_minutes"] = round((time.time() - t1) / 60, 1)
            with gzip.open(out / f"pred_{split}.jsonl.gz", "wt", encoding="utf-8") as f:
                for it, h in zip(items, hyps):
                    f.write(json.dumps({"id": it["id"], "tier": it["tier"], "src_field": it["src_field"],
                                        "tgt_field": it["tgt_field"], "src": it["src"], "ref": it["ref"], "hyp": h},
                                       ensure_ascii=False) + "\n")
            log.info("[%s seed %d] %s: Tier-1 chrF++ %s, Tier-2 %s (beam %d, %.0f min)", key, seed, split,
                     res["evaluation"][split]["tier1"], res["evaluation"][split]["tier2"], FINAL["beams"], (time.time() - t1) / 60)
        res["reference_preprocessing_ceiling"] = preprocessing_ceiling(fam, eval_items(root, "dev_full", key))
    return res


def preprocessing_ceiling(fam: Family, items: list[dict]) -> float | None:
    """chrF++ of the references after the model's own text preprocessing (BanglaT5 normalizer."""
    from sacrebleu.metrics import CHRF
    refs = [it["ref"] for it in items]
    if fam.fam == "t5":
        out = [fam.norm(r) for r in refs]
    elif fam.fam == "indictrans2":
        out = []
        for it in items:
            _, t = fam.codes(it)
            pre = fam.ip_train.preprocess_batch([it["ref"]], src_lang=t, tgt_lang=t, is_target=True)[0]
            out.append(fam.ip_infer.postprocess_batch([pre], lang=t)[0])
    else:
        return None
    return round(CHRF(word_order=2).corpus_score(out, [refs]).score, 2)


def worker(args) -> None:
    jobs = json.loads(Path(args.jobs_file).read_text())
    deadline = float(os.environ.get("BUC_DEADLINE", "inf"))
    for job in jobs:
        if job_done(job):
            continue
        if time.time() + JOB_HOURS[job["model"]] * 3600 > deadline:
            log.warning("time budget: not starting %s (rerun the notebook with its output attached)",
                        Path(job["result_file"]).parent.name)
            continue
        try:
            res = run_job(job, args)
            res["status"] = "ok"
        except Exception as e:
            res = {**job, "status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            log.error("job failed: %s", res["error"])
        out = Path(job["result_file"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def converged(res: dict, cap: int, patience: int) -> bool:
    """A finished final run is still valid under the current epoch cap if early stopping had already triggered (no improvement in the last `patience` epochs) or it already reached the cap."""
    hist = [h.get("dev_select_sel", h.get("dev_select_chrfpp")) for h in res.get("history", [])]
    if not hist:
        return False
    if len(hist) >= cap:
        return True
    best_i = max(range(len(hist)), key=lambda i: hist[i])
    return len(hist) - 1 - best_i >= patience


def job_done(job: dict) -> bool:
    """Done = result exists, succeeded, used the same setting, and (final runs) is still valid under the current epoch cap."""
    f = Path(job["result_file"])
    try:
        r = json.loads(f.read_text()) if f.exists() else None
    except Exception:
        return False
    if not r or r.get("status") != "ok" or any(r.get(k) != job[k] for k in ("lr", "eff_batch", "seed", "stage")):
        return False
    if job["stage"] != "final":
        return True
    cap_ok = r.get("max_epochs") == FINAL["max_epochs"] or ("max_epochs" not in r and len(r.get("history", [])) > 8)
    same_criterion = r.get("selection_metric") == f"{sel_metric(job['model'])}/{SELECTION_VERSION}"
    if not same_criterion and cap_ok and replay_equivalent(r, job["model"]):
        log.info("%s seed %s: final run of the old criterion reused -- replaying the %s criterion on its recorded "
                 "history keeps the same epoch (%s)", job["model"], job["seed"], sel_metric(job["model"]), r.get("best_epoch"))
        return True
    return cap_ok and same_criterion and converged(r, FINAL["max_epochs"], FINAL["patience"])


def replay_equivalent(r: dict, key: str) -> bool:
    """Whether a final run selected with the earlier epoch criterion equals a run of the current criterion."""
    hist = r.get("history") or []
    field = sel_history_field(key)
    vals = [(h.get("epoch", i + 1), h.get(field)) for i, h in enumerate(hist)]
    if not vals or any(v is None for _, v in vals):
        return False
    best, bad, best_ep, stopped = float("-inf"), 0, None, False
    for ep, v in vals:
        if v > best:
            best, bad, best_ep = v, 0, ep
        else:
            bad += 1
        if bad >= FINAL["patience"]:
            stopped = True
            break
    if not stopped and len(vals) < FINAL["max_epochs"]:
        return False
    return best_ep == r.get("best_epoch")


def run_jobs(jobs, args, tag_: str, base_env: dict) -> None:
    import torch
    todo = [j for j in jobs if not job_done(j)]
    n_gpu = max(1, torch.cuda.device_count())
    procs = []
    for g in range(n_gpu):
        mine = todo[g::n_gpu]
        if not mine:
            continue
        jf = Path(args.output_dir, "step4c", f"_jobs_{tag_}_gpu{g}.json")
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(mine))
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(g), BUC_DEADLINE=str(args.deadline))
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--jobs-file", str(jf),
                                       "--artifacts-dir", args.artifacts_dir, "--output-dir", args.output_dir], env=env))
    for p in procs:
        p.wait()


def restore_previous(args) -> None:
    for f in glob.glob("/kaggle/input/**/step4c/**/result.json", recursive=True):
        rel = f.split("/step4c/", 1)[1]
        dst = Path(args.output_dir, "step4c", rel)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(f).parent, dst.parent, dirs_exist_ok=True)


def launch(args, env: dict) -> None:
    import numpy as np
    out = Path(args.output_dir, "step4c")
    restore_previous(args)
    rows = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        cfg = MODELS[key]
        base_env = legacy_env(args) if cfg["env"] == "legacy" else dict(os.environ)
        t_model = time.time()
        sjob = lambda lr, eb: dict(model=key, lr=lr, eff_batch=eb, seed=SEEDS[0], stage="search",
                                   result_file=str(out / key / "search" / f"lr{lr:g}_eb{eb}" / "result.json"))
        grid = [sjob(lr, eb) for lr, eb in itertools.product(cfg["lr"], EFF_BATCH)]
        if args.stage in ("all", "search"):
            log.info("===== %s: search over %d settings (20%% of training serials, up to %d epochs) =====", key, len(grid),
                     SEARCH["max_epochs"])
            run_jobs(grid, args, f"{key}_search", base_env)
        if not all(job_done(j) for j in grid):
            log.warning("%s: search incomplete (%d/%d) -- run --stage search / attach its output", key,
                        sum(job_done(j) for j in grid), len(grid))
            continue
        ok = [json.loads(Path(j["result_file"]).read_text()) for j in grid]
        best = max(ok, key=lambda r: (sel_score(r, key), -r["lr"]))
        boundary = {"criterion": f"{sel_metric(key)} chrF++ on dev_select ({SELECTION_VERSION})",
                    "grid_best": {"lr": best["lr"], "eff_batch": best["eff_batch"], "dev": sel_score(best, key)}}
        edge = "up" if best["lr"] == max(cfg["lr"]) else "down" if best["lr"] == min(cfg["lr"]) else None
        ext = [sjob(LR_EXTENSION[key][edge], eb) for eb in EFF_BATCH] if edge else []
        if ext:
            log.info("%s: selected learning rate %g is at the %s edge of the grid -> trying %g", key, best["lr"],
                     "upper" if edge == "up" else "lower", LR_EXTENSION[key][edge])
            run_jobs(ext, args, f"{key}_extension", base_env)
            if not all(job_done(j) for j in ext):
                log.warning("%s: boundary extension incomplete -- rerun with this notebook's output attached", key)
                continue
            ext_res = [json.loads(Path(j["result_file"]).read_text()) for j in ext]
            cand = max(ext_res, key=lambda r: sel_score(r, key))
            boundary["extension"] = [{"lr": r["lr"], "eff_batch": r["eff_batch"], "dev": sel_score(r, key)}
                                     for r in ext_res]
            if sel_score(cand, key) > sel_score(best, key) + BOUNDARY_MIN_GAIN:
                best, boundary["decision"] = cand, f"extension adopted (gain > {BOUNDARY_MIN_GAIN})"
            else:
                boundary["decision"] = f"grid selection kept (extension gain <= {BOUNDARY_MIN_GAIN})"
            log.info("%s: %s", key, boundary["decision"])
        else:
            boundary["decision"] = "selected learning rate inside the grid; no extension needed"
        res = [json.loads(Path(j["result_file"]).read_text()) for j in grid + ext if Path(j["result_file"]).exists()]
        (out / key / "search_summary.json").write_text(json.dumps(
            [{**{k: r.get(k) for k in ("lr", "eff_batch", "status", "best_epoch", "train_minutes", "peak_gpu_mem_gib", "error")},
              "dev_select_selection": sel_score(r, key), "selection_metric": sel_metric(key)} for r in res], indent=1))
        (out / key / "best_config.json").write_text(json.dumps({"lr": best["lr"], "eff_batch": best["eff_batch"],
                                                                "dev_select_selection": sel_score(best, key),
                                                                "selection_metric": f"{sel_metric(key)}/{SELECTION_VERSION}",
                                                                "boundary_rule": boundary}, indent=1))
        log.info("%s: final setting lr=%g, effective batch %d (dev_select %s chrF++ %.2f)", key, best["lr"], best["eff_batch"],
                 sel_metric(key), sel_score(best, key))
        if args.stage == "search":
            continue
        fjob = lambda s_: dict(model=key, lr=best["lr"], eff_batch=best["eff_batch"], seed=s_, stage="final",
                               out_dir=str(out / key / "final" / f"seed{s_}"),
                               result_file=str(out / key / "final" / f"seed{s_}" / "result.json"))
        finals = [fjob(s_) for s_ in args.seed_list]
        log.info("===== %s: final training, seeds %s (still valid and reused: %s) =====", key, args.seed_list,
                 [j["seed"] for j in finals if job_done(j)])
        run_jobs(finals, args, f"{key}_final", base_env)
        good = [json.loads(Path(fjob(s_)["result_file"]).read_text()) for s_ in SEEDS if job_done(fjob(s_))]
        if len(good) < len(SEEDS):
            log.warning("%s: %d/%d final seeds valid so far", key, len(good), len(SEEDS))
        for split in ("dev_full", "test_full"):
            row = {"model": key, "split": split, "n_seeds": len(good), "lr": best["lr"], "eff_batch": best["eff_batch"]}
            for m in ("tier1", "d2s", "s2d", "tier2", "mean_all"):
                xs = [r["evaluation"][split][m] for r in good if r["evaluation"][split].get(m) is not None]
                row[f"{m}_mean"] = round(float(np.mean(xs)), 2) if xs else None
                row[f"{m}_sd"] = round(float(np.std(xs, ddof=1)), 2) if len(xs) > 1 else None
            row["preprocessing_ceiling"] = good[0].get("reference_preprocessing_ceiling") if good else None
            rows.append(row)
        log.info("%s finished in %.1f h", key, (time.time() - t_model) / 3600)
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = out / "summary.csv"
    if summary.exists():
        old = pd.read_csv(summary)
        df = pd.concat([old[~old["model"].isin(df["model"])], df], ignore_index=True) if len(df) else old
    df.to_csv(summary, index=False)
    manifest = {"script": "step08_seq2seq.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "finished_utc": datetime.now(timezone.utc).isoformat(), "models": {k: {kk: vv for kk, vv in v.items() if kk != "directions"}
                                                                                   for k, v in MODELS.items()},
                "search": SEARCH, "final": FINAL, "train": TRAIN, "lr_extension": LR_EXTENSION,
                "boundary_min_gain": BOUNDARY_MIN_GAIN, "eff_batch": EFF_BATCH, "seeds": SEEDS, "environment": env,
                "args": vars(args)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        log.info("STEP 08 SUMMARY (chrF++, mean over seeds; beam %d)\n%s", FINAL["beams"], df.to_string(index=False))
    (Path(args.output_dir) / "STEP4C_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


def dry_run(args) -> None:
    root = Path(args.artifacts_dir)
    table = load_table(root)
    for key in MODELS:
        ex = train_examples(table, key, 1.0, 0, SEEDS[0], 0, 1)
        dev = eval_items(root, "dev_select", key)
        dirs = {(e["src_field"], e["tgt_field"]) for e in ex}
        log.info("%-24s train/epoch %6d examples, %2d directions sampled; dev_select items %d", key, len(ex), len(dirs), len(dev))
    it = train_examples(table, "nllb-1.3b", 1.0, 0, SEEDS[0], 0, 1)[0]
    log.info("example source text: %s %s -> %s", tag(it["src_field"]), tag(it["tgt_field"]), it["src"][:60])
    items = eval_items(root, "dev_select", "mbart50")[:30]
    log.info("metric check (copy input): %s", {k: v for k, v in score(items, [i["src"] for i in items], ALL_DIRS).items()
                                               if k in ("tier1", "tier2", "mean_all")})


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 08: seq2seq translation baselines")
    p.add_argument("--models", default="banglat5,mbart50,nllb-1.3b",
                   help=f"comma-separated subset of {list(MODELS)} (IndicTrans2: not used in the paper)")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--stage", default="all", choices=["all", "search", "final"],
                   help="all = search then final; split long models over notebooks with search / final")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="final-stage seeds to train in this session")
    p.add_argument("--time-budget-h", type=float, default=11.0,
                   help="no new job is started if it could not finish within this many hours of the session start")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--jobs-file", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.seed_list = [int(x) for x in args.seeds.split(",") if x.strip()]
    args.deadline = time.time() + args.time_budget_h * 3600
    if set(args.seed_list) - set(SEEDS):
        raise SystemExit(f"seeds must be a subset of {SEEDS}")
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
    if args.dry_run:
        dry_run(args)
        return
    unknown = set(args.models.split(",")) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(MODELS)}")
    env = prepare_environment(args)
    token = hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    gated = [m for m in args.models.split(",") if MODELS[m.strip()]["repo"].startswith(GATED)]
    if gated and not token:
        raise SystemExit(f"{gated} are gated models: switch on the HF_TOKEN secret for this notebook (Add-ons -> Secrets), "
                         "using a Hugging Face account that has accepted the licence on every IndicTrans2 model page")
    if not args.requirements:
        hit = find_one("/kaggle/input/**/requirements.txt")
        args.requirements = str(hit) if hit else None
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("artifacts: %s | output: %s | models: %s", args.artifacts_dir, args.output_dir, args.models)
    launch(args, env)


if __name__ == "__main__":
    main()

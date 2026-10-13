#!/usr/bin/env python3
"""Unseen-dialect experiments and external-corpus evaluation of the baselines.

Conditions (seed 13): std-only (no dialect), one-<d> (one dialect seen), lodo-<d> (one dialect left out) and
full. Translation uses BanglaT5 and mBART-50 (recipe of step08); identification uses the SVM (step06) and
BanglaBERT-large / MuRIL-large (step07); unseen classes are masked at test time.

Usage:
    python step11_unseen_and_external.py --task mt --models banglat5 --conditions std-only,one
    python step11_unseen_and_external.py --task mt --models banglat5 --conditions lodo
    python step11_unseen_and_external.py --task id --conditions one,lodo
    python step11_unseen_and_external.py --task id-encoder --encoders banglabert-large --conditions one,lodo
External corpora:
    python step11_unseen_and_external.py --task mt --models banglat5 --conditions full --external
    python step11_unseen_and_external.py --task external-clf --encoders muril-large,banglabert-large
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import gzip
import hashlib
import json
import logging
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step08_seq2seq as s4

DIALECTS = s4.DIALECTS
SEEDS = s4.SEEDS
SELECTED = {"banglat5": {"lr": 5e-4, "eff_batch": 16}, "mbart50": {"lr": 1e-4, "eff_batch": 16},
            "nllb-1.3b": {"lr": 3e-4, "eff_batch": 16}}
JOB_HOURS = {"banglat5": 1.6, "mbart50": 2.0, "nllb-1.3b": 5.5}
SVM_SELECTED = {"features": "c2-5_w1-2", "C": 0.3}
log = logging.getLogger("buc")


def parse_conditions(spec: str) -> list[str]:
    """Condition names: std-only (no dialect seen), one-<d> (only <d> seen), lodo-<d> (all but <d> seen), full."""
    out = []
    for c in [x.strip() for x in spec.split(",") if x.strip()]:
        if c == "all":
            out += ["std-only"] + [f"one-{d}" for d in DIALECTS] + [f"lodo-{d}" for d in DIALECTS]
        elif c == "one":
            out += [f"one-{d}" for d in DIALECTS]
        elif c == "lodo":
            out += [f"lodo-{d}" for d in DIALECTS]
        else:
            train_dialects(c)
            out.append(c)
    return list(dict.fromkeys(out))


def train_dialects(cond: str) -> list[str]:
    """Dialects whose data is used in training (Standard Bangla is always used)."""
    if cond == "full":
        return list(DIALECTS)
    if cond == "std-only":
        return []
    if cond.startswith("one-") and cond[4:] in DIALECTS:
        return [cond[4:]]
    if cond.startswith("lodo-") and cond[5:] in DIALECTS:
        return [d for d in DIALECTS if d != cond[5:]]
    raise SystemExit(f"unknown condition {cond!r}: use full, std-only, one-<d>, lodo-<d> (or one, lodo, all) "
                     f"with <d> in {DIALECTS}")


def _uses_only(field: str, seen: set) -> bool:
    code = field.split(":")[0]
    return code not in DIALECTS or code in seen


def allowed_directions(seen: list[str]) -> tuple[list, list]:
    """Tier-1 and Tier-2 directions whose source and target varieties are all seen (same order as step 08)."""
    s = set(seen)
    t1 = [d for d in s4.TIER1 if _uses_only(d[0], s) and _uses_only(d[1], s)]
    t2 = [d for d in s4.TIER2 if _uses_only(d[0], s) and _uses_only(d[1], s)]
    return t1, t2


def train_examples(table, key: str, epoch: int, seed: int, t1: list, t2: list) -> list[dict]:
    """step 08 sampling (3 Tier-1 + 1 Tier-2 directions per training serial and epoch, same random sequence), restricted to the allowed directions."""
    serials = sorted(table.index[table["split"] == "train"].tolist())
    rng = random.Random(seed * 1000 + epoch)
    mine = set(s4.MODELS[key]["directions"])
    out = []
    for s in serials:
        draws = [rng.choice(t1 or t2) for _ in range(3)] + [rng.choice(t2)]
        for src, tgt in draws:
            if (src, tgt) in mine:
                out.append({"src_field": src, "tgt_field": tgt, "src": table.at[s, src], "ref": table.at[s, tgt]})
    rng.shuffle(out)
    return out


def tier1_items(root: Path, split: str, dirs=None, tiers=("tier1",)) -> list[dict]:
    dirs = set(dirs or s4.TIER1)
    items = []
    for line in open(root / "data" / "eval" / split / "mt.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["tier"] in tiers and (r["src_field"], r["tgt_field"]) in dirs:
            items.append(r)
    return items


def run_job(job: dict, args) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    key, seed, cond = job["model"], job["seed"], job["condition"]
    cfg = s4.MODELS[key]
    device = "cuda:0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(args.artifacts_dir)
    tok_kw = {"use_fast": False} if cfg["family"] == "t5" else {}
    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(cfg["repo"], token=token, **tok_kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["repo"], token=token).to(device)
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    fam = s4.Family(key, tok)
    table = s4.load_table(root)
    seen = train_dialects(cond)
    t1, t2 = allowed_directions(seen)
    sel_dirs, sel_tiers = (t1, ("tier1",)) if t1 else (t2, ("tier2",))
    dev = tier1_items(root, "dev_select", sel_dirs, sel_tiers)
    max_epochs = s4.FINAL["max_epochs"]
    pb = cfg["per_device_batch"]
    accum = max(1, job["eff_batch"] // pb)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optim"] == "adafactor":
        opt = torch.optim.Adafactor(params, lr=job["lr"], weight_decay=0.0)
    else:
        no_decay = ("bias", "layer_norm", "LayerNorm", "layernorm")
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
             "weight_decay": s4.TRAIN["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
             "weight_decay": 0.0}], lr=job["lr"])
    n_per_epoch = len(train_examples(table, key, 0, seed, t1, t2))
    total = max(1, max_epochs * n_per_epoch // (pb * accum))
    warm = max(1, int(s4.TRAIN["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"])
    pad_id = tok.pad_token_id
    history, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        exs = train_examples(table, key, ep, seed, t1, t2)
        enc = [fam.encode_train(e) for e in exs]
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
                torch.nn.utils.clip_grad_norm_(params, s4.TRAIN["max_grad_norm"])
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped += int(cfg["fp16"] and scaler.get_scale() < before)
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        hyps = s4.translate(fam, model, dev, beams=1, device=device)
        sc = s4.score(dev, hyps, sel_dirs)
        sel = sc["tier1"] if t1 else sc["mean_all"]
        history.append({"epoch": ep + 1, "train_loss": round(float(np.nanmean(losses)), 4), "fp16_skips": skipped,
                        "dev_select_tier1_seen": sel, "minutes": round((time.time() - t0) / 60, 1)})
        log.info("[%s %s seed=%d] epoch %d loss %.4f dev_select %s %.2f", key, cond, seed, ep + 1, np.nanmean(losses),
                 "Tier-1 (seen directions)" if t1 else "Tier-2 (training directions)", sel)
        if sel > best:
            best, bad = sel, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s4.FINAL["patience"]:
                break
    model.load_state_dict(best_state)
    items = tier1_items(root, "test_full")
    t1_ = time.time()
    hyps = s4.translate(fam, model, items, beams=s4.FINAL["beams"], device=device)
    ev = s4.score(items, hyps, s4.TIER1)
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "pred_test_full.jsonl.gz", "wt", encoding="utf-8") as f:
        for it, h in zip(items, hyps):
            f.write(json.dumps({"id": it["id"], "tier": it["tier"], "src_field": it["src_field"], "tgt_field": it["tgt_field"],
                                "src": it["src"], "ref": it["ref"], "hyp": h, "condition": cond}, ensure_ascii=False) + "\n")
    if job.get("external"):
        res_ext = external_seq2seq(fam, model, out, device, args)
    unseen = [d for d in DIALECTS if d not in seen]
    per = ev["per_direction"]
    res = {**job, "train_dialects": seen, "unseen_dialects": unseen, "examples_per_epoch": n_per_epoch,
           "n_train_directions": {"tier1": len(t1), "tier2": len(t2)},
           "best_dev_select_tier1_seen": best, "best_epoch": max(history, key=lambda h: h["dev_select_tier1_seen"])["epoch"],
           "history": history, "max_epochs": max_epochs, "code": Path(s4.__file__).name,
           "train_minutes": round((time.time() - t0) / 60, 1),
           "decode_minutes": round((time.time() - t1_) / 60, 1),
           "evaluation": {"test_full": ev},
           "d2s_unseen": {d: per.get(f"{d}:bn>std:bn", {}).get("chrF++") for d in unseen},
           "d2s_all": {d: per.get(f"{d}:bn>std:bn", {}).get("chrF++") for d in DIALECTS},
           "d2s_seen_mean": round(sum(per[f"{d}:bn>std:bn"]["chrF++"] for d in seen) / len(seen), 2) if seen else None,
           "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
    if job.get("external"):
        res["external"] = res_ext
    log.info("[%s %s seed %d] test D->S unseen %s | seen mean %s (%.0f min)", key, cond, seed, res["d2s_unseen"],
             res["d2s_seen_mean"], (time.time() - t0) / 60)
    return res


def _external(args):
    import step10_qlora_gemma_and_evals as s6
    root = args.external_dir or "/kaggle/input"
    xart = s6.ExternalArtifacts(Path(args.artifacts_dir))
    rows = s6.load_external(root)
    items, meta = s6.external_items(xart, rows)
    if not items:
        raise FileNotFoundError(f"no external dataset files found under {root}: attach them as for the LLM runs")
    log.info("external: %d rows, %d items; excluded as duplicates of BUC: %s", len(rows), len(items), meta["duplicates"])
    return s6, xart, items, meta


def external_seq2seq(fam, model, out: Path, device, args) -> dict:
    import pandas as pd
    s6, xart, items, meta = _external(args)
    mt = [it for it in items if it["kind"] == "mt" and not it["auto"] and it["src"].endswith(":bn")]
    conv = [{"id": it["id"], "src_field": it["src"], "tgt_field": it["tgt"], "src": xart.text(it["serial"], it["src"]),
             "ref": it["ref"]} for it in mt]
    t0 = time.time()
    hyps = s4.translate(fam, model, conv, beams=s4.FINAL["beams"], device=device)
    recs = [{"id": it["id"], "kind": "mt", "src": it["src"], "tgt": it["tgt"], "ref": it["ref"], "hyp": h, "valid_json": True}
            for it, h in zip(mt, hyps)]
    with gzip.open(out / "pred_external.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = pd.DataFrame(s6.score_external(recs, xart))
    table.to_csv(out / "external_seq2seq.csv", index=False)
    log.info("EXTERNAL (seq2seq, %.0f min)\n%s", (time.time() - t0) / 60,
             table[[c for c in ("dataset", "task", "dialect", "n", "chrfpp", "copy_chrfpp") if c in table.columns]].to_string(index=False))
    return {"n_items": len(recs), "duplicates_excluded": meta["duplicates"], "minutes": round((time.time() - t0) / 60, 1)}


@__import__("functools").lru_cache(maxsize=None)
def _import_versioned(*names):
    """Imports the first of `names` that exists."""
    import importlib
    for n in names:
        try:
            mod = importlib.import_module(n)
            log.info("using %s (%s)", n, Path(mod.__file__).name)
            return mod
        except ImportError:
            continue
    raise SystemExit(f"{names} not found next to this script")


def _import_step4a():
    return _import_versioned("step06_classical_classifiers")


def _import_step4b():
    return _import_versioned("step07_encoder_classifiers")


def find_encoder_dir(key: str, seed: int = 13) -> Path | None:
    """Saved step 07 model of `key` (seed 13): <...>/step4b/<key>/final/seed13/model/ or the seed folder itself."""
    pats = [f"/kaggle/input/**/step4b/{key}/final/seed{seed}/model/config.json",
            f"/kaggle/input/**/step4b/{key}/final/seed{seed}/config.json",
            f"/kaggle/input/**/{key}/**/seed{seed}/**/config.json"]
    for pat in pats:
        hits = [h for h in sorted(glob.glob(pat, recursive=True)) if Path(h).with_name("tokenizer_config.json").exists()]
        if hits:
            return Path(hits[0]).parent
    return None


def encoder_predict(model_dir: Path, key: str, texts: list[str], device: str, mask_ids=()) -> list[str]:
    """Labels from a saved 10-way encoder (step 07 preprocessing: punctuation removed by the caller, BanglaBERT normalizer, max length 128)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    s4b = _import_step4b()
    norm = s4b.model_normalizer(key)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    id2 = {int(k): v for k, v in mdl.config.id2label.items()}
    out = []
    for b in range(0, len(texts), 128):
        enc = tok([norm(t or "") for t in texts[b:b + 128]], truncation=True, max_length=s4b.TRAIN["max_len"],
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits.float()
        for i in mask_ids:
            logits[:, i] = float("-inf")
        out += logits.argmax(-1).tolist()
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return [id2[i] for i in out]


def run_external_classifiers(args) -> None:
    """SVM (step 06) and the saved seed-13 encoder(s) of step 07 on the external identification items (both scripts)."""
    import pandas as pd
    from collections import Counter
    s6, xart, items, meta = _external(args)
    id_items = [it for it in items if it["kind"] == "id"]
    texts = [s6.strip_punct(xart.text(it["serial"], it["field"])) for it in id_items]
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    preds = []
    svm = sorted(glob.glob("/kaggle/input/**/step4a/svm/final/model.joblib", recursive=True))
    if svm:
        import joblib
        step06_classical_classifiers = _import_step4a()
        main_mod = sys.modules["__main__"]
        for name in dir(step06_classical_classifiers):
            if not hasattr(main_mod, name):
                setattr(main_mod, name, getattr(step06_classical_classifiers, name))
        m = joblib.load(svm[0])
        y = m["classifier"].predict(m["features"].transform(texts))
        labels = m.get("labels", step06_classical_classifiers.BANGLA)
        preds.append(("SVM", [labels[int(v)] if not isinstance(v, str) else v for v in y]))
        log.info("external: SVM from %s", svm[0])
    else:
        log.warning("external: SVM not found -- attach the step-06 output (step4a/svm/final/model.joblib)")
    names = {"muril-large": "MuRIL-large", "banglabert-large": "BanglaBERT-large", "xlmr-large": "XLM-R-large"}
    for key in [k.strip() for k in args.encoders.split(",") if k.strip()]:
        d = find_encoder_dir(key)
        if d is None:
            log.warning("external: saved %s (seed 13) not found -- attach its step 07 output", key)
            continue
        log.info("external: %s from %s", key, d)
        preds.append((names.get(key, key), encoder_predict(d, key, texts, device)))
    rows = []
    for name, p in preds:
        groups = {}
        for it, y in zip(id_items, p):
            ds, code = it["id"].split("|")[1], it["label"]
            for dd in (code, "all"):
                for sc in (it["script"], "both"):
                    groups.setdefault((ds, dd, sc), []).append((it, y))
        for (ds, dd, sc), sub in sorted(groups.items()):
            rows.append({"system": name, "dataset": ds, "dialect": dd, "script": sc, "n": len(sub),
                         "accuracy": round(100 * sum(y == it["label"] for it, y in sub) / len(sub), 2),
                         "predicted_labels": json.dumps(dict(Counter(y for _, y in sub).most_common()))})
    out = Path(args.output_dir, "step4d", "external")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "external_classifiers.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"n_id_items": len(id_items), "duplicates_excluded": meta["duplicates"],
                                                   "systems": [n for n, _ in preds]}, indent=1))
    if len(df):
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            log.info("EXTERNAL (classifiers)\n%s", df[df.script == "both"][["system", "dataset", "dialect", "n", "accuracy"]]
                     .to_string(index=False))


def id_condition_summary(model: str, cond: str, test: list[dict], pred: list[str], seen_labels: list[str]) -> tuple[dict, list]:
    """Seen-class macro-F1 per script and, for every unseen dialect, the share of its test sentences assigned to each seen variety (per script)."""
    from sklearn.metrics import f1_score
    res, rows = {}, []
    for script in ("bn", "rom"):
        idx = [i for i, r in enumerate(test) if r["script"] == script]
        seen_idx = [i for i in idx if test[i]["label"] in seen_labels]
        res[f"seen_macro_f1_{script}"] = round(100 * f1_score([test[i]["label"] for i in seen_idx], [pred[i] for i in seen_idx],
                                                              labels=seen_labels, average="macro", zero_division=0), 2)
        for d in [x for x in DIALECTS if x not in seen_labels]:
            u = [pred[i] for i in idx if test[i]["label"] == d]
            dist = {c: round(100 * u.count(c) / len(u), 2) for c in sorted(set(u), key=lambda c: -u.count(c))}
            res.setdefault("unseen_assigned", {}).setdefault(d, {})[script] = dist
            trained = [x for x in seen_labels if x != "std"]
            rows.append({"model": model, "condition": cond, "unseen": d, "script": script, "n": len(u),
                         "top_assigned": max(dist, key=dist.get) if dist else None,
                         "top_share": max(dist.values()) if dist else None, "share_std": dist.get("std", 0.0),
                         "share_trained_dialect": dist.get(trained[0], 0.0) if len(trained) == 1 else None,
                         "seen_macro_f1": res[f"seen_macro_f1_{script}"], "assigned": json.dumps(dist)})
    return res, rows


def id_conditions(spec: str) -> list[str]:
    """Identification needs at least two classes: std-only (one class) is skipped."""
    conds = [c for c in parse_conditions(spec) if c not in ("std-only", "full")]
    if "std-only" in parse_conditions(spec):
        log.warning("identification: std-only skipped -- a classifier trained on Standard Bangla alone has one class "
                    "and labels every sentence Standard")
    return conds


ENCODER_SELECTED = {"muril-large": {"lr": 3e-5, "batch": 16}, "banglabert-large": {"lr": 3e-5, "batch": 16},
                    "xlmr-large": {"lr": 3e-5, "batch": 16}}


def run_encoder_fold(job: dict, args) -> dict:
    """step 07 final-stage training (up to 8 epochs, early stopping on dev macro-F1 of the seen classes, Bangla script) on the seen varieties only."""
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    s4b = _import_step4b()
    key, lr, bs, seed, cond = job["model"], job["lr"], job["batch"], job["seed"], job["condition"]
    seen_labels = ["std"] + train_dialects(cond)
    seen_ids = [s4b.LABEL2ID[v] for v in seen_labels]
    masked = [i for i in range(10) if i not in seen_ids]
    device = "cuda:0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(args.artifacts_dir)
    norm = s4b.model_normalizer(key)
    tok = AutoTokenizer.from_pretrained(s4b.MODELS[key]["repo"])
    enc = lambda texts: tok([norm(t or "") for t in texts], add_special_tokens=True, truncation=True,
                            max_length=s4b.TRAIN["max_len"])["input_ids"]
    train = [x for x in s4b.load_train(root) if x[1] in seen_ids]
    tr_ids, tr_y = enc([t for t, _, _ in train]), np.array([y for _, y, _ in train])
    dev = [r for r in s4b.load_eval(root, "dev_full") if r["label"] in seen_labels and r["script"] == "bn"]
    dev_ids, dev_y = enc([r["text_nopunct"] for r in dev]), np.array([s4b.LABEL2ID[r["label"]] for r in dev])
    model = AutoModelForSequenceClassification.from_pretrained(
        s4b.MODELS[key]["repo"], num_labels=10, id2label=dict(enumerate(s4b.BANGLA)), label2id=s4b.LABEL2ID).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": s4b.TRAIN["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    max_epochs = s4b.FINAL_MAX_EPOCHS
    total = ((len(tr_ids) + bs - 1) // bs) * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(s4b.TRAIN["warmup_frac"] * total), total)
    scaler = torch.amp.GradScaler("cuda")
    pad_id = tok.pad_token_id
    rng = random.Random(seed)
    lengths = [len(x) for x in tr_ids]

    def decide(probs):
        p = probs.copy()
        p[:, masked] = -1.0
        return p.argmax(1)
    history, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        losses = []
        for b in s4b.make_batches(lengths, bs, rng):
            ids, att = s4b.pad([tr_ids[i] for i in b], pad_id, device)
            y = torch.tensor(tr_y[b], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=att).logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), s4b.TRAIN["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        pred = decide(s4b.predict(model, dev_ids, pad_id, device, s4b.TRAIN["eval_batch"]))
        score = 100 * f1_score(dev_y, pred, labels=seen_ids, average="macro", zero_division=0)
        history.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "dev_bn_macro_f1_seen": round(score, 3),
                        "minutes": round((time.time() - t0) / 60, 1)})
        log.info("[%s %s seed=%d] epoch %d loss %.4f dev bn macro-F1 (seen classes) %.2f", key, cond, seed, ep + 1,
                 np.mean(losses), score)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s4b.TRAIN["patience"]:
                break
    model.load_state_dict(best_state)
    test = s4b.load_eval(root, "test_full")
    pred = [s4b.BANGLA[i] for i in decide(s4b.predict(model, enc([r["text_nopunct"] for r in test]), pad_id, device,
                                                      s4b.TRAIN["eval_batch"]))]
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "pred_test_full.jsonl.gz", "wt", encoding="utf-8") as f:
        for r, p_ in zip(test, pred):
            f.write(json.dumps({"id": r["id"], "label": r["label"], "script": r["script"], "pred": p_, "condition": cond}) + "\n")
    summ, rows = id_condition_summary(key, cond, test, pred, seen_labels)
    res = {**job, "seen_labels": seen_labels, "best_dev_bn_macro_f1_seen": best, "history": history,
           "max_epochs": max_epochs, "code": Path(s4b.__file__).name,
           "train_minutes": round((time.time() - t0) / 60, 1), **summ, "summary_rows": rows}
    log.info("[%s %s] seen macro-F1 bn %.2f (%.0f min)", key, cond, res["seen_macro_f1_bn"], res["train_minutes"])
    return res


def launch_encoder(args) -> None:
    import pandas as pd
    out = Path(args.output_dir, "step4d", "id_encoder")
    restore_previous(args)
    conds = id_conditions(args.conditions)
    rows = []
    for key in [k.strip() for k in args.encoders.split(",") if k.strip()]:
        cfg = ENCODER_SELECTED[key]
        jobs = [dict(kind="encoder", model=key, condition=c, seed=s, lr=cfg["lr"], eff_batch=cfg["batch"],
                     batch=cfg["batch"], out_dir=str(out / key / c / f"seed{s}"),
                     result_file=str(out / key / c / f"seed{s}" / "result.json"))
                for s in args.seed_list for c in conds]
        log.info("===== %s: %d identification runs (%s); already done: %d =====", key, len(jobs), ",".join(conds),
                 sum(job_done(j) for j in jobs))
        run_jobs(jobs, args, dict(os.environ))
        for f in sorted(glob.glob(str(out / key / "*" / "seed*" / "result.json"))):
            r = json.loads(Path(f).read_text())
            if r.get("status") == "ok":
                rows += r.get("summary_rows", [])
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary_id_encoder.csv", index=False)
    if len(df):
        with pd.option_context("display.width", 220, "display.max_rows", 200):
            log.info("UNSEEN DIALECTS (encoder identification)\n%s", df.drop(columns=["assigned"]).to_string(index=False))


def job_done(job: dict) -> bool:
    f = Path(job["result_file"])
    try:
        r = json.loads(f.read_text()) if f.exists() else None
    except Exception:
        return False
    if not (bool(r) and r.get("status") == "ok" and all(r.get(k) == job[k] for k in ("lr", "eff_batch", "seed", "condition"))
            and (not job.get("external") or "external" in r)):
        return False
    want = _import_step4b().FINAL_MAX_EPOCHS if job.get("kind") == "encoder" else s4.FINAL["max_epochs"]
    return r.get("max_epochs") == want


def worker(args) -> None:
    jobs = json.loads(Path(args.jobs_file).read_text())
    deadline = float(os.environ.get("BUC_DEADLINE", "inf"))
    for job in jobs:
        if job_done(job):
            continue
        hours = 4.2 if job.get("kind") == "encoder" else JOB_HOURS[job["model"]]
        if time.time() + hours * 3600 > deadline:
            log.warning("time budget: not starting %s %s seed %s (rerun with this output attached)", job["model"],
                        job["condition"], job["seed"])
            continue
        try:
            res = run_encoder_fold(job, args) if job.get("kind") == "encoder" else run_job(job, args)
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


def restore_previous(args) -> None:
    for f in glob.glob("/kaggle/input/**/step4d/**/result.json", recursive=True):
        rel = f.split("/step4d/", 1)[1]
        dst = Path(args.output_dir, "step4d", rel)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(f).parent, dst.parent, dirs_exist_ok=True)


def run_jobs(jobs, args, base_env: dict) -> None:
    import torch
    todo = [j for j in jobs if not job_done(j)]
    n_gpu = max(1, torch.cuda.device_count())
    procs = []
    for g in range(n_gpu):
        mine = todo[g::n_gpu]
        if not mine:
            continue
        jf = Path(args.output_dir, "step4d", f"_jobs_gpu{g}.json")
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(mine))
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(g), BUC_DEADLINE=str(args.deadline))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--jobs-file", str(jf),
               "--artifacts-dir", args.artifacts_dir, "--output-dir", args.output_dir]
        if args.external_dir:
            cmd += ["--external-dir", args.external_dir]
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        p.wait()


def reference_d2s(key: str) -> dict:
    """Per-dialect D->S chrF++ of the existing step 08 final runs (all nine dialects seen), if attached."""
    ref = {}
    for f in glob.glob(f"/kaggle/input/**/step4c/{key}/final/seed*/result.json", recursive=True):
        r = json.loads(Path(f).read_text())
        per = r.get("evaluation", {}).get("test_full", {}).get("per_direction", {})
        if per:
            ref[r["seed"]] = {d: per[f"{d}:bn>std:bn"]["chrF++"] for d in DIALECTS if f"{d}:bn>std:bn" in per}
    return ref


def launch_mt(args) -> None:
    import pandas as pd
    out = Path(args.output_dir, "step4d")
    restore_previous(args)
    conds = parse_conditions(args.conditions)
    rows = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        if key not in SELECTED:
            raise SystemExit(f"{key}: no step 08 selection recorded here; choose from {list(SELECTED)}")
        base_env = s4.legacy_env(args) if s4.MODELS[key]["env"] == "legacy" else dict(os.environ)
        jobs = [dict(model=key, condition=c, seed=s, lr=SELECTED[key]["lr"], eff_batch=SELECTED[key]["eff_batch"],
                     external=bool(args.external and c == "full"),
                     out_dir=str(out / key / c / f"seed{s}"), result_file=str(out / key / c / f"seed{s}" / "result.json"))
                for s in args.seed_list for c in conds]
        log.info("===== %s: %d jobs (%s x seeds %s); already done: %d =====", key, len(jobs), ",".join(conds), args.seed_list,
                 sum(job_done(j) for j in jobs))
        run_jobs(jobs, args, base_env)
        ref = reference_d2s(key)
        for f in sorted(glob.glob(str(out / key / "*" / "seed*" / "result.json"))):
            r = json.loads(Path(f).read_text())
            if r.get("status") != "ok":
                continue
            for d, v in r["d2s_unseen"].items():
                full = ref.get(r["seed"], {}).get(d)
                rows.append({"model": key, "condition": r["condition"], "seed": r["seed"], "dialect": d, "d2s_unseen": v,
                             "d2s_all_seen_reference": full, "drop": round(full - v, 2) if full is not None and v is not None else None,
                             "d2s_seen_mean": r["d2s_seen_mean"], "best_epoch": r["best_epoch"],
                             "train_minutes": r["train_minutes"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary_mt.csv", index=False)
    (out / "manifest_mt.json").write_text(json.dumps({
        "script": "step11_unseen_and_external.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "finished_utc": datetime.now(timezone.utc).isoformat(), "selected": SELECTED, "final": s4.FINAL,
        "conditions": conds, "seeds": args.seed_list}, indent=1))
    if len(df):
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            log.info("UNSEEN DIALECTS (test D->S chrF++)\n%s", df.to_string(index=False))


def run_id(args) -> None:
    """SVM (step 06 configuration) trained on the seen varieties of each condition; deterministic, one fit each."""
    import numpy as np
    import pandas as pd
    s4a = _import_step4a()
    root = Path(args.artifacts_dir)
    out = Path(args.output_dir, "step4d", "id")
    out.mkdir(parents=True, exist_ok=True)
    texts, y, scripts = s4a.load_train(root, quick=args.quick)
    y = np.array(y)
    test = s4a.load_eval(root, "test_full", args.quick)
    fc = s4a.FEATURES[SVM_SELECTED["features"]]
    rows = []
    for cond in ["full"] + id_conditions(args.conditions):
        t0 = time.time()
        seen_labels = ["std"] + train_dialects(cond)
        keep = np.isin(y, [s4a.LABEL2ID[v] for v in seen_labels])
        feat = s4a.TfidfFeatures(fc["char"], fc["word"])
        X = feat.fit_transform([t for t, k in zip(texts, keep) if k])
        clf = s4a.make_linear("svm", SVM_SELECTED["C"], s4a.SEEDS[0]).fit(X, y[keep])
        classes = [s4a.BANGLA[c] for c in clf.classes_]
        S = clf.decision_function(feat.transform([r["text_nopunct"] for r in test]))
        if S.ndim == 1:
            pred = [classes[1] if v > 0 else classes[0] for v in S]
        else:
            pred = [classes[i] for i in S.argmax(1)]
        with gzip.open(out / f"pred_{cond}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r, p_ in zip(test, pred):
                f.write(json.dumps({"id": r["id"], "label": r["label"], "script": r["script"], "pred": p_, "condition": cond}) + "\n")
        summ, r_rows = id_condition_summary("svm", cond, test, pred, seen_labels)
        if cond == "full":
            r_rows = [{"model": "svm", "condition": "full", "unseen": None, "script": sc, "n": None, "top_assigned": None,
                       "top_share": None, "share_std": None, "share_trained_dialect": None,
                       "seen_macro_f1": summ[f"seen_macro_f1_{sc}"], "assigned": None} for sc in ("bn", "rom")]
        rows += r_rows
        import sklearn
        (out / f"result_{cond}.json").write_text(json.dumps({"condition": cond, "seen_labels": seen_labels, **summ,
                                                             "sklearn": sklearn.__version__, "numpy": np.__version__,
                                                             "minutes": round((time.time() - t0) / 60, 2)}, indent=1))
        log.info("[svm %s] seen macro-F1 bn %.2f rom %.2f (%.1f min)", cond, summ["seen_macro_f1_bn"],
                 summ["seen_macro_f1_rom"], (time.time() - t0) / 60)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary_id.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        log.info("UNSEEN DIALECTS (SVM identification)\n%s", df.drop(columns=["assigned"]).to_string(index=False))


ID_PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "joblib", "threadpoolctl")


def prepare_environment(args, only=None) -> dict:
    """Version check and pinned install (Python 3.12 or 3.13)."""
    from importlib import metadata
    import re
    ver = sys.version.split()[0]
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise SystemExit(f"Python 3.12 or 3.13 required, found {ver}")
    if sys.version_info[:2] == (3, 13):
        log.info("Python %s: newer Kaggle image (the pinned packages install unchanged; versions are recorded)", ver)
    req = Path(args.requirements) if args.requirements else None
    if not req or not req.exists():
        log.warning("requirements.txt not found; versions not checked")
        return {}
    pins = {}
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    if only is not None:
        pins = {k: v for k, v in pins.items() if k in only}

    def bad():
        out = {}
        for k, v in pins.items():
            try:
                have = metadata.version(k)
            except metadata.PackageNotFoundError:
                have = None
            if have != v:
                out[k] = [v, have]
        return out
    mism = bad()
    if mism and not args.no_install:
        if only is not None:
            if os.environ.get("BUC_STEP4D_ENV") == "1":
                log.warning("pinned packages still differ after installing: %s", mism)
            else:
                log.info("installing the step 06 library versions: %s", {k: v[0] for k, v in mism.items()})
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[f"{k}=={pins[k]}" for k in mism]], check=True)
                os.environ["BUC_STEP4D_ENV"] = "1"
                sys.stdout.flush()
                os.execv(sys.executable, [sys.executable, *sys.argv])
        else:
            log.info("installing pinned core requirements (%d packages differ) ...", len(mism))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
            mism = bad()
    if mism:
        log.warning("packages still differing from requirements.txt: %s", mism)
    return {"python": ver, "requirements": str(req), "mismatched": mism}


def ensure_normalizer(args) -> None:
    """BanglaT5 and BanglaBERT need the csebuetnlp normalizer (a git line in requirements.txt, not a version pin)."""
    try:
        import normalizer
        return
    except ImportError:
        pass
    line = None
    if args.requirements and Path(args.requirements).exists():
        line = next((l.strip() for l in Path(args.requirements).read_text().splitlines()
                     if l.strip().startswith("normalizer") and "git+" in l), None)
    line = line or "normalizer @ git+https://github.com/csebuetnlp/normalizer"
    log.info("installing the csebuetnlp normalizer: %s", line)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", line], check=True)


def dry_run(args) -> None:
    root = Path(args.artifacts_dir)
    table = s4.load_table(root)
    ref = s4.train_examples(table, "banglat5", 1.0, 0, SEEDS[0], 0, 1)
    t1, t2 = allowed_directions(list(DIALECTS))
    same = train_examples(table, "banglat5", 0, SEEDS[0], t1, t2)
    log.info("all dialects allowed reproduces the step-08 setup exactly: %s (%d pairs)", same == ref, len(ref))
    for c in parse_conditions(args.conditions):
        t1, t2 = allowed_directions(train_dialects(c))
        ex = train_examples(table, "banglat5", 0, SEEDS[0], t1, t2)
        used = {x.split(":")[0] for e in ex for x in (e["src_field"], e["tgt_field"])}
        log.info("%-12s Tier-1 dirs %2d, Tier-2 dirs %2d, pairs/epoch %d, varieties in training %s, dev_select items %d", c,
                 len(t1), len(t2), len(ex), sorted(used), len(tier1_items(root, "dev_select", t1 or t2, ("tier1",) if t1 else ("tier2",))))


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 11: unseen-dialect experiments")
    p.add_argument("--task", choices=["mt", "id", "id-encoder", "external-clf"], default="mt")
    p.add_argument("--encoders", default="muril-large",
                   help="encoders for external-clf / id-encoder (muril-large, banglabert-large, xlmr-large; comma-separated)")
    p.add_argument("--external", action="store_true",
                   help="with condition full: also translate the external corpora (same items and metrics as the LLM runs)")
    p.add_argument("--external-dir", help="folder searched for the external dataset files (default: /kaggle/input)")
    p.add_argument("--models", default="banglat5", help=f"translation models, subset of {list(SELECTED)}")
    p.add_argument("--conditions", default="all",
                   help="all | std-only | one | lodo | one-<d> | lodo-<d> | full, comma-separated (all = std-only + one + lodo)")
    p.add_argument("--seeds", default="13", help="seeds (default 13: the unseen experiments use one seed)")
    p.add_argument("--artifacts-dir")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements")
    p.add_argument("--time-budget-h", type=float, default=11.0)
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--quick", action="store_true", help="identification: small sample, pipeline test only")
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
    s4.log = log
    log.info("using %s", Path(s4.__file__).name)
    if not args.artifacts_dir:
        hit = s4.find_one("/kaggle/input/**/data/aligned.parquet")
        if hit is None:
            raise SystemExit("step-02 output not found under /kaggle/input: attach the output of step02_prepare_data.py")
        args.artifacts_dir = str(hit.parents[1])
    if args.worker:
        worker(args)
        return
    if args.dry_run:
        dry_run(args)
        return
    if not args.requirements:
        hit = s4.find_one("/kaggle/input/**/requirements.txt")
        args.requirements = str(hit) if hit else None
    encoders_used = args.task == "external-clf" and any(k.strip() for k in args.encoders.split(","))
    if encoders_used:
        prepare_environment(args)
        if "banglabert" in args.encoders:
            ensure_normalizer(args)
        import sklearn
        log.info("scikit-learn %s, numpy %s (python %s)", sklearn.__version__, __import__("numpy").__version__,
                 sys.version.split()[0])
        run_external_classifiers(args)
        return
    if args.task in ("id", "external-clf"):
        env = prepare_environment(args, only=ID_PACKAGES)
        import sklearn
        log.info("scikit-learn %s, numpy %s (python %s)", sklearn.__version__, __import__("numpy").__version__,
                 sys.version.split()[0])
        if env.get("mismatched"):
            log.warning("NOT the step 06 versions: %s -- `full` may differ from the paper's SVM", env["mismatched"])
        run_id(args) if args.task == "id" else run_external_classifiers(args)
        return
    prepare_environment(args)
    ensure_normalizer(args)
    if args.task == "id-encoder":
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        launch_encoder(args)
        return
    token = s4.hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("artifacts: %s | output: %s | models: %s | conditions: %s | seeds: %s", args.artifacts_dir, args.output_dir,
             args.models, args.conditions, args.seed_list)
    launch_mt(args)
    (Path(args.output_dir) / "STEP4D_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    main()

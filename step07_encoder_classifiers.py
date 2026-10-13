#!/usr/bin/env python3
"""Encoder baselines for ten-way dialect identification: BanglaBERT-large, XLM-R-large and MuRIL-large.

Search over learning rate and batch size, then three seeds of the selected setting, evaluated on dev_full and
test_full. The seed-13 MuRIL-large model is saved as the second dialect-fidelity judge.

Usage:
    python step07_encoder_classifiers.py
    python step07_encoder_classifiers.py --models muril-large
Offline test:
    python step07_encoder_classifiers.py --dry-run --artifacts-dir outputs --no-install
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
import subprocess
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

MODELS = {
    "banglabert-large": {"repo": "csebuetnlp/banglabert_large", "bangla_normalizer": True},
    "xlmr-large": {"repo": "FacebookAI/xlm-roberta-large", "bangla_normalizer": False},
    "muril-large": {"repo": "google/muril-large-cased", "bangla_normalizer": False},
}
GRID = {"lr": [5e-6, 1e-5, 2e-5, 3e-5], "batch": [16, 32]}
GRID_EXTENSION = {"up": 5e-5, "down": 2.5e-6}
BOUNDARY_MIN_GAIN = 0.5
FINAL_MAX_EPOCHS = 8
JOB_HOURS = {"banglabert-large": 2.8, "xlmr-large": 4.5, "muril-large": 4.2}
TRAIN = {"max_epochs": 5, "patience": 2, "warmup_frac": 0.10, "weight_decay": 0.01, "max_grad_norm": 1.0,
         "max_len": 128, "eval_batch": 256}
SEEDS = [13, 42, 2026]
DEGENERATE_F1 = 20.0
BANGLA = ["std", "nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
DIALECTS = BANGLA[1:]
LABEL2ID = {v: i for i, v in enumerate(BANGLA)}
log = logging.getLogger("buc")


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def prepare_environment(args) -> dict:
    """Install the pinned core requirements before any third-party import (torch is never touched)."""
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 required, found {sys.version.split()[0]}")
    req = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt")
    if not req or not req.exists():
        log.warning("requirements.txt not found; versions not checked")
        return {}
    pins = {}
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)

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
        log.info("installing pinned requirements (%d packages differ) ...", len(mism))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
        mism = bad()
    return {"requirements": str(req), "mismatched": mism}


def strip_punct(s: str) -> str:
    """Identical to step 02: remove all Unicode punctuation (P*), incl. the danda (primary ID input)."""
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def model_normalizer(key: str):
    """BanglaBERT was pretrained on text passed through the csebuetnlp normalizer; its authors require the same normalization before tokenization."""
    if not MODELS[key]["bangla_normalizer"]:
        return lambda s: s
    from normalizer import normalize
    return lambda s: normalize(s) if s else s


def load_train(root: Path) -> list[tuple[str, int, str]]:
    """(text without punctuation, label id, script) for every training serial x 10 varieties x 2 scripts."""
    import pandas as pd
    tab = pd.read_parquet(root / "data" / "aligned.parquet")
    tr = tab[tab["split"] == "train"]
    out = []
    for v in BANGLA:
        for script in ("bn", "rom"):
            out += [(strip_punct(t), LABEL2ID[v], script) for t in tr[f"{v}:{script}"]]
    return out


def load_eval(root: Path, name: str) -> list[dict]:
    return [json.loads(l) for l in open(root / "data" / "eval" / name / "id.jsonl", encoding="utf-8")]


CONDITIONS = {"primary": "text_nopunct", "punct_kept": "text", "masked": "text_masked"}


def metrics_from_probs(probs, labels, scripts) -> dict:
    """Accuracy, macro/weighted F1, per-label F1, log-loss and confusion matrix per script; 9-way results restrict both the items and the decision to the nine dialects."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
    probs, labels, scripts = np.asarray(probs, dtype=np.float64), np.asarray(labels), np.asarray(scripts)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(1, keepdims=True)
    res = {}
    for script in ("bn", "rom"):
        m = scripts == script
        if not m.any():
            continue
        y, p = labels[m], probs[m]
        pred = p.argmax(1)
        r = {"n": int(m.sum()), "accuracy": 100 * accuracy_score(y, pred),
             "macro_f1": 100 * f1_score(y, pred, labels=range(10), average="macro", zero_division=0),
             "weighted_f1": 100 * f1_score(y, pred, labels=range(10), average="weighted", zero_division=0),
             "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1), labels=list(range(10)))),
             "per_label_f1": dict(zip(BANGLA, (100 * f1_score(y, pred, labels=range(10), average=None,
                                                             zero_division=0)).round(2).tolist())),
             "confusion": confusion_matrix(y, pred, labels=range(10)).tolist()}
        d = y != 0
        pred9 = p[d][:, 1:].argmax(1) + 1
        r["macro_f1_9way"] = 100 * f1_score(y[d], pred9, labels=range(1, 10), average="macro", zero_division=0)
        r["accuracy_9way"] = 100 * accuracy_score(y[d], pred9)
        res[script] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
    return res


def make_batches(lengths, batch: int, rng: random.Random, window: int = 64):
    """Length-grouped batches: sort within windows of `window` batches, then shuffle the batch order."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out, w = [], batch * window
    for s in range(0, len(idx), w):
        chunk = sorted(idx[s:s + w], key=lambda i: lengths[i])
        out += [chunk[j:j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(out)
    return out


def pad(seqs, pad_id, device):
    import torch
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.to(device), att.to(device)


def predict(model, tok_ids, pad_id, device, bs):
    """Class probabilities (float32) for pre-tokenized inputs, batched by length."""
    import numpy as np
    import torch
    order = sorted(range(len(tok_ids)), key=lambda i: len(tok_ids[i]))
    out = np.zeros((len(tok_ids), 10), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            ids, att = pad([tok_ids[i] for i in b], pad_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=att).logits
            out[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def run_one(job: dict, args) -> dict:
    """Train one (model, lr, batch, seed) with early stopping on dev_full (Bangla-script macro-F1)."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    key, lr, bs, seed = job["model"], job["lr"], job["batch"], job["seed"]
    device = "cuda:0"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(args.artifacts_dir)
    norm = model_normalizer(key)
    tok = AutoTokenizer.from_pretrained(MODELS[key]["repo"])
    enc = lambda texts: tok([norm(t or "") for t in texts], add_special_tokens=True, truncation=True,
                            max_length=TRAIN["max_len"])["input_ids"]
    train = load_train(root)
    tr_ids, tr_y = enc([t for t, _, _ in train]), np.array([y for _, y, _ in train])
    dev = load_eval(root, "dev_full")
    dev_ids = enc([r["text_nopunct"] for r in dev])
    dev_y, dev_s = np.array([LABEL2ID[r["label"]] for r in dev]), np.array([r["script"] for r in dev])
    truncated = float(np.mean([len(x) >= TRAIN["max_len"] for x in tr_ids]))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[key]["repo"], num_labels=10, id2label=dict(enumerate(BANGLA)), label2id=LABEL2ID).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": TRAIN["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps_per_epoch = (len(tr_ids) + bs - 1) // bs
    max_epochs = FINAL_MAX_EPOCHS if job["stage"] == "final" else TRAIN["max_epochs"]
    total = steps_per_epoch * max_epochs
    sched = get_linear_schedule_with_warmup(opt, int(TRAIN["warmup_frac"] * total), total)
    scaler = torch.amp.GradScaler("cuda")
    pad_id = tok.pad_token_id
    rng = random.Random(seed)
    lengths = [len(x) for x in tr_ids]
    history, best, best_state, bad_epochs = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        losses, skipped = [], 0
        for b in make_batches(lengths, bs, rng):
            ids, att = pad([tr_ids[i] for i in b], pad_id, device)
            y = torch.tensor(tr_y[b], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=att).logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["max_grad_norm"])
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        probs = predict(model, dev_ids, pad_id, device, TRAIN["eval_batch"])
        m = metrics_from_probs(probs, dev_y, dev_s)
        score = m["bn"]["macro_f1"]
        history.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "fp16_skips": skipped,
                        "dev_bn_macro_f1": score, "dev_rom_macro_f1": m["rom"]["macro_f1"],
                        "minutes": round((time.time() - t0) / 60, 1)})
        log.info("[%s lr=%g bs=%d seed=%d] epoch %d loss %.4f dev bn F1 %.2f rom F1 %.2f", key, lr, bs, seed, ep + 1,
                 np.mean(losses), score, m["rom"]["macro_f1"])
        if score > best:
            best, bad_epochs = score, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= TRAIN["patience"]:
                break
    res = {**job, "best_dev_bn_macro_f1": best, "best_epoch": max(history, key=lambda h: h["dev_bn_macro_f1"])["epoch"],
           "history": history, "max_epochs": max_epochs, "train_minutes": round((time.time() - t0) / 60, 1),
           "degenerate": best < DEGENERATE_F1,
           "share_truncated_inputs": round(truncated, 5),
           "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
    if job["stage"] == "final":
        model.load_state_dict(best_state)
        res["evaluation"] = final_evaluation(model, tok, norm, pad_id, device, root, Path(job["out_dir"]))
        if seed == SEEDS[0]:
            model.half().save_pretrained(Path(job["out_dir"]) / "model")
            tok.save_pretrained(Path(job["out_dir"]) / "model")
    return res


def final_evaluation(model, tok, norm, pad_id, device, root: Path, out_dir: Path) -> dict:
    """Every condition on dev_full and test_full; predictions with probabilities are saved (gzip jsonl)."""
    import numpy as np
    res = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("dev_full", "test_full"):
        items = load_eval(root, split)
        res[split] = {}
        for cond, col in CONDITIONS.items():
            sub = [r for r in items if r.get(col) is not None]
            ids = tok([norm(r[col]) for r in sub], truncation=True, max_length=TRAIN["max_len"])["input_ids"]
            probs = predict(model, ids, pad_id, device, TRAIN["eval_batch"])
            res[split][cond] = metrics_from_probs(probs, [LABEL2ID[r["label"]] for r in sub], [r["script"] for r in sub])
            with gzip.open(out_dir / f"pred_{split}_{cond}.jsonl.gz", "wt", encoding="utf-8") as f:
                for r, p in zip(sub, probs):
                    f.write(json.dumps({"id": r["id"], "label": r["label"], "script": r["script"],
                                        "pred": BANGLA[int(p.argmax())], "probs": [round(float(x), 5) for x in p]}) + "\n")
    return res


def worker(args) -> None:
    """Runs a list of jobs sequentially on the single GPU this process can see; one JSON per job."""
    jobs = json.loads(Path(args.jobs_file).read_text())
    deadline = float(os.environ.get("BUC_DEADLINE", "inf"))
    for job in jobs:
        out = Path(job["result_file"])
        if job_done(job):
            continue
        if time.time() + JOB_HOURS[job["model"]] * 3600 > deadline:
            log.warning("time budget: not starting %s (rerun the notebook with its output attached)", out.parent.name)
            continue
        try:
            res = run_one(job, args)
            res["status"] = "ok"
        except Exception as e:
            res = {**job, "status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            log.error("job failed: %s", res["error"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def converged(res: dict, cap: int, patience: int) -> bool:
    """A finished run is still valid under the current epoch cap if early stopping had already triggered (the last `patience` epochs did not improve), or if it already trained up to the cap."""
    hist = [h["dev_bn_macro_f1"] for h in res.get("history", [])]
    if not hist:
        return False
    if len(hist) >= cap:
        return True
    best_i = max(range(len(hist)), key=lambda i: hist[i])
    return len(hist) - 1 - best_i >= patience


def job_done(job: dict) -> bool:
    """Done = the result exists, succeeded, was trained with the same setting, and (final runs) is still valid under the current epoch cap."""
    f = Path(job["result_file"])
    if not f.exists():
        return False
    try:
        r = json.loads(f.read_text())
    except Exception:
        return False
    if r.get("status") != "ok" or any(r.get(k) != job[k] for k in ("lr", "batch", "seed", "stage")):
        return False
    if job["stage"] != "final":
        return True
    return r.get("max_epochs") == FINAL_MAX_EPOCHS and converged(r, FINAL_MAX_EPOCHS, TRAIN["patience"])


def run_jobs(jobs: list[dict], args, tag: str) -> None:
    """Distribute jobs over the GPUs (round-robin) and wait."""
    import torch
    todo = [j for j in jobs if not job_done(j)]
    n_gpu = max(1, torch.cuda.device_count())
    procs = []
    for g in range(n_gpu):
        mine = todo[g::n_gpu]
        if not mine:
            continue
        jf = Path(args.output_dir, "step4b", f"_jobs_{tag}_gpu{g}.json")
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(mine))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), BUC_DEADLINE=str(args.deadline))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--jobs-file", str(jf),
               "--artifacts-dir", args.artifacts_dir, "--output-dir", args.output_dir]
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        p.wait()


def restore_previous(args) -> None:
    """Copy results of an earlier (attached) session so that finished jobs are not repeated."""
    import shutil
    for f in glob.glob("/kaggle/input/**/step4b/**/result.json", recursive=True):
        rel = f.split("/step4b/", 1)[1]
        dst = Path(args.output_dir, "step4b", rel)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(f).parent, dst.parent, dirs_exist_ok=True)


def search_job(key, lr, bs, out):
    return dict(model=key, lr=lr, batch=bs, seed=SEEDS[0], stage="search",
                result_file=str(out / key / "search" / f"lr{lr:g}_bs{bs}" / "result.json"))


def load_ok(jobs):
    res = []
    for j in jobs:
        if job_done(j):
            r = json.loads(Path(j["result_file"]).read_text())
            if not r.get("degenerate"):
                res.append(r)
    return res


def launch(args, env: dict) -> None:
    out = Path(args.output_dir, "step4b")
    restore_previous(args)
    summary_rows = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        t_model = time.time()
        grid = [search_job(key, lr, bs, out) for lr, bs in itertools.product(GRID["lr"], GRID["batch"])]
        log.info("===== %s: search over %d configurations =====", key, len(grid))
        run_jobs(grid, args, f"{key}_search")
        if not all(job_done(j) for j in grid):
            log.warning("%s: search incomplete (%d/%d) -- rerun with this notebook's output attached", key,
                        sum(job_done(j) for j in grid), len(grid))
            continue
        ok = load_ok(grid)
        if not ok:
            log.error("%s: no usable configuration", key)
            continue
        best = max(ok, key=lambda r: (r["best_dev_bn_macro_f1"], -r["lr"]))
        boundary = {"grid_best": {"lr": best["lr"], "batch": best["batch"], "dev": best["best_dev_bn_macro_f1"]}}
        edge = "up" if best["lr"] == max(GRID["lr"]) else "down" if best["lr"] == min(GRID["lr"]) else None
        ext = [search_job(key, GRID_EXTENSION[edge], bs, out) for bs in GRID["batch"]] if edge else []
        if ext:
            log.info("%s: selected learning rate %g is at the %s edge of the grid -> trying %g", key, best["lr"],
                     "upper" if edge == "up" else "lower", GRID_EXTENSION[edge])
            run_jobs(ext, args, f"{key}_extension")
            if not all(job_done(j) for j in ext):
                log.warning("%s: boundary extension incomplete -- rerun with this notebook's output attached", key)
                continue
            ext_ok = load_ok(ext)
            cand = max(ext_ok, key=lambda r: r["best_dev_bn_macro_f1"]) if ext_ok else None
            boundary["extension"] = [{"lr": r["lr"], "batch": r["batch"], "dev": r["best_dev_bn_macro_f1"]} for r in ext_ok]
            if cand and cand["best_dev_bn_macro_f1"] > best["best_dev_bn_macro_f1"] + BOUNDARY_MIN_GAIN:
                best = cand
                boundary["decision"] = "extension adopted (gain > %.1f)" % BOUNDARY_MIN_GAIN
            else:
                boundary["decision"] = "grid selection kept (extension gain <= %.1f)" % BOUNDARY_MIN_GAIN
            log.info("%s: %s", key, boundary["decision"])
        else:
            boundary["decision"] = "selected learning rate inside the grid; no extension needed"
        all_search = [json.loads(Path(j["result_file"]).read_text()) for j in grid + ext if Path(j["result_file"]).exists()]
        (out / key / "search_summary.json").write_text(json.dumps(
            [{k: r.get(k) for k in ("lr", "batch", "status", "degenerate", "best_dev_bn_macro_f1", "best_epoch",
                                    "train_minutes", "peak_gpu_mem_gib")} for r in all_search], indent=1))
        (out / key / "best_config.json").write_text(json.dumps({"lr": best["lr"], "batch": best["batch"],
                                                                "dev_bn_macro_f1": best["best_dev_bn_macro_f1"],
                                                                "boundary_rule": boundary}, indent=1))
        log.info("%s: final configuration lr=%g batch=%d (dev bn macro-F1 %.2f)", key, best["lr"], best["batch"],
                 best["best_dev_bn_macro_f1"])
        finals = [dict(model=key, lr=best["lr"], batch=best["batch"], seed=s_, stage="final",
                       out_dir=str(out / key / "final" / f"seed{s_}"),
                       result_file=str(out / key / "final" / f"seed{s_}" / "result.json")) for s_ in SEEDS]
        reused = [j["seed"] for j in finals if job_done(j)]
        log.info("===== %s: final training, seeds %s (still valid and reused: %s) =====", key, SEEDS, reused)
        run_jobs(finals, args, f"{key}_final")
        fres = [json.loads(Path(j["result_file"]).read_text()) for j in finals if job_done(j)]
        if len(fres) < len(SEEDS):
            log.warning("%s: %d/%d final seeds done -- rerun with this notebook's output attached", key, len(fres), len(SEEDS))
        summary_rows += summarize(key, best, fres, time.time() - t_model)
    write_summary(args, summary_rows, env)


def summarize(key, best, fres, seconds) -> list[dict]:
    import numpy as np
    rows = []
    good = [r for r in fres if r.get("status") == "ok"]
    for split in ("dev_full", "test_full"):
        for cond in CONDITIONS:
            for script in ("bn", "rom"):
                vals = [r["evaluation"][split][cond].get(script) for r in good
                        if r["evaluation"][split][cond].get(script)]
                if not vals:
                    continue
                row = {"model": key, "split": split, "condition": cond, "script": script, "n_seeds": len(vals),
                       "lr": best["lr"], "batch": best["batch"]}
                for metric in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "macro_f1_9way"):
                    xs = [v[metric] for v in vals]
                    row[f"{metric}_mean"] = round(float(np.mean(xs)), 3)
                    row[f"{metric}_sd"] = round(float(np.std(xs, ddof=1)), 3) if len(xs) > 1 else None
                rows.append(row)
    log.info("%s done in %.1f h", key, seconds / 3600)
    return rows


def write_summary(args, rows, env) -> None:
    import pandas as pd
    out = Path(args.output_dir, "step4b")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    manifest = {"script": "step07_encoder_classifiers.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "finished_utc": datetime.now(timezone.utc).isoformat(), "models": MODELS, "grid": GRID, "train": TRAIN,
                "grid_extension": GRID_EXTENSION, "boundary_min_gain": BOUNDARY_MIN_GAIN, "final_max_epochs": FINAL_MAX_EPOCHS,
                "seeds": SEEDS, "environment": env, "args": vars(args)}
    try:
        import torch
        manifest["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        manifest["torch"] = torch.__version__
    except Exception:
        pass
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    if len(df):
        view = df[(df["condition"] == "primary")][["model", "split", "script", "n_seeds", "accuracy_mean", "macro_f1_mean",
                                                    "macro_f1_sd", "macro_f1_9way_mean", "log_loss_mean"]]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            log.info("STEP 07 SUMMARY (punctuation removed; mean over seeds)\n%s", view.to_string(index=False))
    (Path(args.output_dir) / "STEP4B_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


def dry_run(args) -> None:
    import numpy as np
    root = Path(args.artifacts_dir)
    train = load_train(root)
    dev = load_eval(root, "dev_full")
    log.info("train examples: %d (%s)", len(train), {s: sum(1 for x in train if x[2] == s) for s in ("bn", "rom")})
    log.info("dev items: %d; conditions available: %s", len(dev),
             {c: sum(1 for r in dev if r.get(col) is not None) for c, col in CONDITIONS.items()})
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(10), size=len(dev))
    y = [LABEL2ID[r["label"]] for r in dev]
    probs[np.arange(len(dev)), y] += 0.5
    probs /= probs.sum(1, keepdims=True)
    m = metrics_from_probs(probs, y, [r["script"] for r in dev])
    log.info("metric check (fake classifier): bn acc %.1f macro-F1 %.1f 9-way %.1f log-loss %.3f", m["bn"]["accuracy"],
             m["bn"]["macro_f1"], m["bn"]["macro_f1_9way"], m["bn"]["log_loss"])
    b = make_batches([len(t) for t, _, _ in train], 32, random.Random(0))
    log.info("length-grouped batches: %d (covering %d examples)", len(b), sum(len(x) for x in b))


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 07: encoder baselines for dialect identification")
    p.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {list(MODELS)}")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--time-budget-h", type=float, default=11.0,
                   help="no new job is started if it could not finish within this many hours of the session start")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--jobs-file", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.deadline = time.time() + args.time_budget_h * 3600
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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("artifacts: %s | output: %s | models: %s", args.artifacts_dir, args.output_dir, args.models)
    launch(args, env)


if __name__ == "__main__":
    main()

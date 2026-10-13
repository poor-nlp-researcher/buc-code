#!/usr/bin/env python3
"""Classical dialect-identification baselines: majority class, TF-IDF SVM, logistic regression and fastText.

Grid search on the development set, final training (three seeds for fastText) and evaluation on dev_full and
test_full under every input condition (punctuation removed or kept, markers masked, Bangla script and romanized).
The SVM is also saved as the first dialect-fidelity judge.

Usage:
    python step06_classical_classifiers.py --artifacts-dir outputs
Quick test on a small sample:
    python step06_classical_classifiers.py --artifacts-dir outputs --no-install --quick
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

BANGLA = ["std", "nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
LABEL2ID = {v: i for i, v in enumerate(BANGLA)}
SEEDS = [13, 42, 2026]
CONDITIONS = {"primary": "text_nopunct", "punct_kept": "text", "masked": "text_masked"}
FEATURES = {
    "c1-4_w1": {"char": (1, 4), "word": (1, 1)},
    "c1-4_w1-2": {"char": (1, 4), "word": (1, 2)},
    "c2-5_w1": {"char": (2, 5), "word": (1, 1)},
    "c2-5_w1-2": {"char": (2, 5), "word": (1, 2)},
}
GRID = {
    "svm": {"C": [0.1, 0.3, 1.0, 3.0]},
    "logreg": {"C": [1.0, 10.0, 100.0]},
    "fasttext": {"lr": [0.1, 0.5, 1.0], "epoch": [10, 25], "wordNgrams": [1, 2], "subwords": [(0, 0), (2, 5)]},
}
FASTTEXT = {"repo": "https://github.com/facebookresearch/fastText.git",
            "commit": "1142dc4c4ecbc19cc16eee5cdd28472e689267e6", "dim": 100}
log = logging.getLogger("buc")


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def prepare_environment(args) -> dict:
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


def build_fasttext(tools: Path) -> Path:
    """Compile the official fastText command-line tool at a pinned commit (about 1 minute)."""
    exe = tools / "fastText" / "fasttext"
    if exe.exists():
        return exe
    tools.mkdir(parents=True, exist_ok=True)
    src = tools / "fastText"
    if src.exists():
        shutil.rmtree(src)
    subprocess.run(["git", "clone", "-q", FASTTEXT["repo"], str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", FASTTEXT["commit"]], check=True)
    subprocess.run(["make", "-j4"], cwd=src, check=True, stdout=subprocess.DEVNULL)
    log.info("fastText built at commit %s", FASTTEXT["commit"][:10])
    return exe


def strip_punct(s: str) -> str:
    """Identical to step 02: remove all Unicode punctuation (P*), incl. the danda."""
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def load_train(root: Path, quick: bool, keep_punct: bool = False):
    import pandas as pd
    tab = pd.read_parquet(root / "data" / "aligned.parquet")
    tr = tab[tab["split"] == "train"]
    if quick:
        tr = tr.iloc[:300]
    texts, y, scripts = [], [], []
    for v in BANGLA:
        for script in ("bn", "rom"):
            col = [t if keep_punct else strip_punct(t) for t in tr[f"{v}:{script}"]]
            texts += col
            y += [LABEL2ID[v]] * len(col)
            scripts += [script] * len(col)
    return texts, y, scripts


def load_eval(root: Path, name: str, quick: bool) -> list[dict]:
    items = [json.loads(l) for l in open(root / "data" / "eval" / name / "id.jsonl", encoding="utf-8")]
    return items[:2000] if quick else items


def metrics(scores, labels, scripts, probabilistic: bool) -> dict:
    """scores: (n, 10) class probabilities (probabilistic=True) or decision scores (SVM, majority)."""
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
    scores, labels, scripts = np.asarray(scores, dtype=np.float64), np.asarray(labels), np.asarray(scripts)
    if probabilistic:
        scores = np.clip(scores, 1e-12, None)
        scores = scores / scores.sum(1, keepdims=True)
    res = {}
    for script in ("bn", "rom"):
        m = scripts == script
        if not m.any():
            continue
        y, s = labels[m], scores[m]
        pred = s.argmax(1)
        r = {"n": int(m.sum()), "accuracy": 100 * accuracy_score(y, pred),
             "macro_f1": 100 * f1_score(y, pred, labels=range(10), average="macro", zero_division=0),
             "weighted_f1": 100 * f1_score(y, pred, labels=range(10), average="weighted", zero_division=0),
             "log_loss": float(log_loss(y, np.clip(s, 1e-12, 1), labels=list(range(10)))) if probabilistic else None,
             "per_label_f1": dict(zip(BANGLA, (100 * f1_score(y, pred, labels=range(10), average=None,
                                                             zero_division=0)).round(2).tolist())),
             "confusion": confusion_matrix(y, pred, labels=range(10)).tolist()}
        d = y != 0
        pred9 = s[d][:, 1:].argmax(1) + 1
        r["macro_f1_9way"] = 100 * f1_score(y[d], pred9, labels=range(1, 10), average="macro", zero_division=0)
        r["accuracy_9way"] = 100 * accuracy_score(y[d], pred9)
        res[script] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
    return res


class TfidfFeatures:
    """Character n-grams within word boundaries + word n-grams, sublinear TF, min_df 2, concatenated."""

    def __init__(self, char, word):
        from sklearn.feature_extraction.text import TfidfVectorizer
        common = dict(sublinear_tf=True, min_df=2, lowercase=False, dtype=__import__("numpy").float32)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=char, **common)
        self.wv = TfidfVectorizer(analyzer="word", ngram_range=word, tokenizer=str.split, token_pattern=None, **common)

    def fit_transform(self, texts):
        import scipy.sparse as sp
        return sp.hstack([self.cv.fit_transform(texts), self.wv.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        import scipy.sparse as sp
        return sp.hstack([self.cv.transform(texts), self.wv.transform(texts)]).tocsr()

    def names(self):
        import numpy as np
        return np.concatenate([["c:" + x for x in self.cv.get_feature_names_out()],
                               ["w:" + x for x in self.wv.get_feature_names_out()]])


def make_linear(kind: str, C: float, seed: int):
    if kind == "svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=C, random_state=seed, max_iter=5000)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed)


def linear_scores(kind: str, clf, X):
    return clf.decision_function(X) if kind == "svm" else clf.predict_proba(X)


def fasttext_train(exe: Path, texts, y, cfg: dict, seed: int, workdir: Path, threads: int) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    data = workdir / "train.txt"
    order = list(range(len(texts)))
    __import__("random").Random(seed).shuffle(order)
    with open(data, "w", encoding="utf-8") as f:
        for i in order:
            f.write(f"__label__{BANGLA[y[i]]} {texts[i]}\n")
    out = workdir / "model"
    minn, maxn = cfg["subwords"]
    cmd = [str(exe), "supervised", "-input", str(data), "-output", str(out), "-lr", str(cfg["lr"]),
           "-epoch", str(cfg["epoch"]), "-wordNgrams", str(cfg["wordNgrams"]), "-minn", str(minn), "-maxn", str(maxn),
           "-dim", str(FASTTEXT["dim"]), "-thread", str(threads), "-seed", str(seed), "-verbose", "0"]
    subprocess.run(cmd, check=True)
    data.unlink()
    return out.with_suffix(".bin")


def fasttext_probs(exe: Path, model: Path, texts) -> "np.ndarray":
    """Probabilities for all 10 classes via `predict-prob` (k = 10)."""
    import numpy as np
    inp = "\n".join((t.replace("\n", " ") or ".") for t in texts) + "\n"
    out = subprocess.run([str(exe), "predict-prob", str(model), "-", "10"], input=inp, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    if len(out) != len(texts):
        raise RuntimeError(f"fastText returned {len(out)} predictions for {len(texts)} inputs")
    P = np.zeros((len(texts), 10), dtype=np.float32)
    for i, line in enumerate(out):
        toks = line.split()
        for lab, p in zip(toks[0::2], toks[1::2]):
            P[i, LABEL2ID[lab.replace("__label__", "")]] = float(p)
    P = np.clip(P, 1e-7, None)
    return P / P.sum(1, keepdims=True)


def eval_all(score_fn, items_by_split, probabilistic: bool, out_dir: Path) -> dict:
    """Every split x condition; saves predictions (with scores) as gzip jsonl."""
    res = {}
    for split, items in items_by_split.items():
        res[split] = {}
        for cond, col in CONDITIONS.items():
            sub = [r for r in items if r.get(col) is not None]
            S = score_fn([r[col] for r in sub])
            res[split][cond] = metrics(S, [LABEL2ID[r["label"]] for r in sub], [r["script"] for r in sub], probabilistic)
            with gzip.open(out_dir / f"pred_{split}_{cond}.jsonl.gz", "wt", encoding="utf-8") as f:
                for r, s in zip(sub, S):
                    f.write(json.dumps({"id": r["id"], "label": r["label"], "script": r["script"],
                                        "pred": BANGLA[int(s.argmax())],
                                        ("probs" if probabilistic else "scores"): [round(float(x), 5) for x in s]}) + "\n")
    return res


def run_linear(kind: str, texts, y, dev, test, out: Path) -> dict:
    import joblib
    import numpy as np
    t0 = time.time()
    dev_texts = [r["text_nopunct"] for r in dev]
    dev_y = [LABEL2ID[r["label"]] for r in dev]
    dev_s = [r["script"] for r in dev]
    search = []
    for fname, fcfg in FEATURES.items():
        feat = TfidfFeatures(fcfg["char"], fcfg["word"])
        X = feat.fit_transform(texts)
        Xd = feat.transform(dev_texts)
        for C in GRID[kind]["C"]:
            t1 = time.time()
            clf = make_linear(kind, C, SEEDS[0]).fit(X, y)
            m = metrics(linear_scores(kind, clf, Xd), dev_y, dev_s, kind == "logreg")
            search.append({"features": fname, "C": C, "n_features": int(X.shape[1]), "dev_bn_macro_f1": m["bn"]["macro_f1"],
                           "dev_rom_macro_f1": m["rom"]["macro_f1"], "fit_seconds": round(time.time() - t1, 1)})
            log.info("[%s] %s C=%g: dev bn macro-F1 %.2f (rom %.2f), %d features, %.0f s", kind, fname, C,
                     m["bn"]["macro_f1"], m["rom"]["macro_f1"], X.shape[1], time.time() - t1)
    best = max(search, key=lambda r: (r["dev_bn_macro_f1"], -r["C"]))
    fcfg = FEATURES[best["features"]]
    feat = TfidfFeatures(fcfg["char"], fcfg["word"])
    X = feat.fit_transform(texts)
    clf = make_linear(kind, best["C"], SEEDS[0]).fit(X, y)
    fin = out / "final"
    fin.mkdir(parents=True, exist_ok=True)
    evaluation = eval_all(lambda ts: linear_scores(kind, clf, feat.transform(ts)), {"dev_full": dev, "test_full": test},
                          kind == "logreg", fin)
    joblib.dump({"features": feat, "classifier": clf, "labels": BANGLA}, fin / "model.joblib", compress=3)
    names = feat.names()
    coef = clf.coef_
    top = {BANGLA[k]: [str(names[j]) for j in np.argsort(-coef[k])[:30]] for k in range(10)}
    (fin / "top_features.json").write_text(json.dumps(top, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"model": kind, "search": search, "best": best, "n_seeds": 1, "evaluations": [evaluation],
            "minutes": round((time.time() - t0) / 60, 1)}


def run_svm_punct(root: Path, quick: bool, dev, test, out: Path, svm_best: dict) -> dict:
    """An SVM trained on text with punctuation (selected SVM setting reused), so that 'punctuation kept' is a matched condition rather than a train/test mismatch."""
    import joblib
    t0 = time.time()
    texts, y, _ = load_train(root, quick, keep_punct=True)
    fcfg = FEATURES[svm_best["features"]]
    feat = TfidfFeatures(fcfg["char"], fcfg["word"])
    clf = make_linear("svm", svm_best["C"], SEEDS[0]).fit(feat.fit_transform(texts), y)
    fin = out / "final"
    fin.mkdir(parents=True, exist_ok=True)
    evaluation = eval_all(lambda ts: clf.decision_function(feat.transform(ts)), {"dev_full": dev, "test_full": test}, False, fin)
    joblib.dump({"features": feat, "classifier": clf, "labels": BANGLA}, fin / "model.joblib", compress=3)
    return {"model": "svm-punct", "search": [], "best": {**svm_best, "trained_with_punctuation": True}, "n_seeds": 1,
            "evaluations": [evaluation], "minutes": round((time.time() - t0) / 60, 1)}


def run_fasttext(exe: Path, texts, y, dev, test, out: Path, threads: int) -> dict:
    t0 = time.time()
    dev_texts = [r["text_nopunct"] for r in dev]
    dev_y = [LABEL2ID[r["label"]] for r in dev]
    dev_s = [r["script"] for r in dev]
    search = []
    keys = list(GRID["fasttext"])
    tmp = Path(tempfile.mkdtemp())
    for vals in itertools.product(*[GRID["fasttext"][k] for k in keys]):
        cfg = dict(zip(keys, vals))
        t1 = time.time()
        model = fasttext_train(exe, texts, y, cfg, SEEDS[0], tmp / "search", threads)
        m = metrics(fasttext_probs(exe, model, dev_texts), dev_y, dev_s, True)
        search.append({**cfg, "subwords": list(cfg["subwords"]), "dev_bn_macro_f1": m["bn"]["macro_f1"],
                       "dev_rom_macro_f1": m["rom"]["macro_f1"], "fit_seconds": round(time.time() - t1, 1)})
        log.info("[fasttext] %s: dev bn macro-F1 %.2f (rom %.2f), %.0f s", cfg, m["bn"]["macro_f1"], m["rom"]["macro_f1"],
                 time.time() - t1)
    best = max(search, key=lambda r: (r["dev_bn_macro_f1"], -r["epoch"], -r["lr"]))
    cfg = {k: (tuple(best[k]) if k == "subwords" else best[k]) for k in keys}
    evaluations = []
    for seed in SEEDS:
        fin = out / "final" / f"seed{seed}"
        fin.mkdir(parents=True, exist_ok=True)
        model = fasttext_train(exe, texts, y, cfg, seed, fin, threads=1)
        evaluations.append(eval_all(lambda ts: fasttext_probs(exe, model, ts), {"dev_full": dev, "test_full": test}, True, fin))
        if seed != SEEDS[0]:
            model.unlink()
        log.info("[fasttext] final seed %d done", seed)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"model": "fasttext", "search": search, "best": best, "n_seeds": len(SEEDS), "evaluations": evaluations,
            "minutes": round((time.time() - t0) / 60, 1), "fasttext_commit": FASTTEXT["commit"]}


def run_majority(y, dev, test, out: Path) -> dict:
    import numpy as np
    counts = np.bincount(np.asarray(y), minlength=10)
    k = int(np.argmax(counts))
    fin = out / "final"
    fin.mkdir(parents=True, exist_ok=True)

    def score(ts):
        S = np.zeros((len(ts), 10), dtype=np.float32)
        S[:, k] = 1.0
        return S
    ev = eval_all(score, {"dev_full": dev, "test_full": test}, False, fin)
    return {"model": "majority", "search": [], "best": {"label": BANGLA[k]}, "n_seeds": 1, "evaluations": [ev], "minutes": 0.0}


def summarize(results: list[dict]) -> list[dict]:
    import numpy as np
    rows = []
    for r in results:
        for split in ("dev_full", "test_full"):
            for cond in CONDITIONS:
                for script in ("bn", "rom"):
                    vals = [e[split][cond][script] for e in r["evaluations"] if script in e[split][cond]]
                    if not vals:
                        continue
                    row = {"model": r["model"], "split": split, "condition": cond, "script": script, "n_seeds": len(vals)}
                    for metric in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "macro_f1_9way"):
                        xs = [v[metric] for v in vals if v.get(metric) is not None]
                        row[f"{metric}_mean"] = round(float(np.mean(xs)), 3) if xs else None
                        row[f"{metric}_sd"] = round(float(np.std(xs, ddof=1)), 3) if len(xs) > 1 else None
                    rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 06: classical dialect-identification baselines (CPU)")
    p.add_argument("--models", default="majority,svm,svm-punct,logreg,fasttext")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2)), help="threads for the fastText search")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--quick", action="store_true", help="small sample, for testing the pipeline only")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if not args.artifacts_dir:
        hit = find_one("/kaggle/input/**/data/aligned.parquet")
        if hit is None:
            raise SystemExit("step-02 output not found under /kaggle/input: attach the output of step02_prepare_data.py")
        args.artifacts_dir = str(hit.parents[1])
    env = prepare_environment(args)
    root = Path(args.artifacts_dir)
    out = Path(args.output_dir, "step4a")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    texts, y, scripts = load_train(root, args.quick)
    dev, test = load_eval(root, "dev_full", args.quick), load_eval(root, "test_full", args.quick)
    log.info("train %d examples; dev %d, test %d items; models: %s", len(texts), len(dev), len(test), args.models)
    results = []
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        (out / kind).mkdir(parents=True, exist_ok=True)
        log.info("===== %s =====", kind)
        if kind == "majority":
            r = run_majority(y, dev, test, out / kind)
        elif kind in ("svm", "logreg"):
            r = run_linear(kind, texts, y, dev, test, out / kind)
        elif kind == "svm-punct":
            prev = next((x for x in results if x["model"] == "svm"), None)
            if prev is None and (out / "svm" / "result.json").exists():
                prev = json.loads((out / "svm" / "result.json").read_text())
            if prev is None:
                raise SystemExit("svm-punct needs the svm result: run it after svm")
            r = run_svm_punct(root, args.quick, dev, test, out / kind, prev["best"])
        elif kind == "fasttext":
            exe = build_fasttext(Path(args.output_dir).parent / "tools" if not args.quick else Path(tempfile.gettempdir()) / "buc_tools")
            r = run_fasttext(exe, texts, y, dev, test, out / kind, args.threads)
        else:
            raise SystemExit(f"unknown model {kind}")
        (out / kind / "result.json").write_text(json.dumps(r, indent=1, default=str))
        if r["search"]:
            (out / kind / "search_summary.json").write_text(json.dumps(r["search"], indent=1, default=str))
            log.info("[%s] best setting: %s", kind, r["best"])
        results.append(r)
    import pandas as pd
    df = pd.DataFrame(summarize(results))
    df.to_csv(out / "summary.csv", index=False)
    manifest = {"script": "step06_classical_classifiers.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "finished_utc": datetime.now(timezone.utc).isoformat(), "hours": round((time.time() - t0) / 3600, 2),
                "features": FEATURES, "grid": GRID, "fasttext": FASTTEXT, "seeds": SEEDS, "environment": env,
                "args": vars(args), "cpu_count": os.cpu_count(), "quick": args.quick}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    view = df[df["condition"] == "primary"][["model", "split", "script", "n_seeds", "accuracy_mean", "macro_f1_mean",
                                             "macro_f1_sd", "macro_f1_9way_mean", "log_loss_mean"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log.info("STEP 06 SUMMARY (punctuation removed)\n%s", view.to_string(index=False))
    (Path(args.output_dir) / "STEP4A_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    main()

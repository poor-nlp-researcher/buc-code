#!/usr/bin/env python3
"""Evaluation of every system and the significance tests.

Translation: chrF++, chrF, BLEU, METEOR, ROUGE-L, CER, WER, BERTScore, COMET (from step13), gain over copying
the input. Dialect fidelity: the SVM and MuRIL-large judges label every output with a Bangla target.
Identification: accuracy, macro-F1, per-variety precision, recall and F1. Significance: paired bootstrap over
serials (10,000 resamples), McNemar's test, Holm correction within each family; multi-seed systems are
tested with their median seed.

Usage:
    python step14_evaluate.py --bootstrap 10000
    python step14_evaluate.py --bootstrap 10000 --systems '^(ablate-)?base-titullm-3b-'
Offline test of the metric code:
    python step14_evaluate.py --self-test --artifacts-dir outputs --no-install
"""
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import gzip
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

BANGLA = ["std", "nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
DIALECTS = BANGLA[1:]
SEEDS = [13, 42, 2026]
BOOT = {"n": 10000, "seed": 13}
ALPHA = 0.05
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BLEU_TOKENIZER = {"name": None}
LATIN = re.compile(r"[A-Za-z]")
BENGALI = re.compile(r"[\u0980-\u09FF]")
log = logging.getLogger("buc")


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def prepare_environment(args) -> None:
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise SystemExit(f"Python 3.12 or 3.13 required, found {sys.version.split()[0]}")
    req = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt")
    if not req or not req.exists() or args.no_install:
        return
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", req.read_text(), re.M))
    need = []
    for p in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "sacrebleu", "sentencepiece", "nltk", "rouge-score",
              "jiwer", "bert-score", "statsmodels", "joblib", "transformers", "safetensors", "tokenizers"):
        try:
            have = metadata.version(p)
        except metadata.PackageNotFoundError:
            have = None
        if p in pins and have != pins[p]:
            need.append(f"{p}=={pins[p]}")
    if need:
        log.info("installing pinned packages: %s", " ".join(need))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)


def strip_punct(s: str) -> str:
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def serial_of(item_id: str) -> int:
    return int(item_id.split("|")[1])


class Items:
    """Frozen evaluation items of step 02 (references, sources, direction groups)."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = {}

    def get(self, split: str) -> dict:
        if split not in self.cache:
            d = self.root / "data" / "eval" / split
            mt = {json.loads(l)["id"]: json.loads(l) for l in open(d / "mt.jsonl", encoding="utf-8")}
            idd = {json.loads(l)["id"]: json.loads(l) for l in open(d / "id.jsonl", encoding="utf-8")}
            self.cache[split] = {"mt": mt, "id": idd}
        return self.cache[split]


def group_of(r: dict) -> list[str]:
    """Direction groups an MT item belongs to (for macro-averaged summaries)."""
    tier, src, tgt = r["tier"], r["src_field"], r["tgt_field"]
    if tier == "tier1-auto":
        return ["auto"]
    if tier == "tier1":
        return ["tier1", "d2s" if tgt == "std:bn" else "s2d"]
    if tier == "tier2":
        return ["tier2"]
    if tier == "tier3":
        if src.endswith(":rom"):
            return ["tier3", "tier3_rom"]
        return ["tier3", "tier3_seen" if r.get("dd_status") == "seen" else "tier3_held_out"]
    return [tier]


def read_lines(path: Path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


RUN_FAMILY = {"primary": "QLoRA", "base": "QLoRA-base", "final": "QLoRA-2ep", "single-id": "single-id",
              "single-mt": "single-mt", "plain": "QLoRA-plain"}
RUN_RE = re.compile(r"^(primary|base|final|single-id|single-mt|plain)-(.+)-s(\d+)$")
PROMPT_SUBSET = "test_prompt"
SUBSET_SPLIT = "test_prompt_sub"


def _ablation_split(d: str, var: str) -> str:
    """Evaluation set of a step 10 variant: from its metrics.json (step6 v7+), else T-* = test_full, others dev_full."""
    mf = Path(d, "metrics.json")
    if mf.exists():
        try:
            es = json.loads(mf.read_text()).get("eval_set")
            if es:
                return es
        except (ValueError, OSError):
            pass
    return "test_full" if var.startswith("T-") else "dev_full"


def discover(roots: list[str], prompt_subset: bool = True) -> dict:
    """{(system, split): {"files": [...], "kind": ..., "family": ..., "seed": ...}} for every prediction set found."""
    found = {}

    def add(system, split, files, kind, family, seed, **extra):
        k = (system, split)
        if k not in found and files:
            found[k] = {"files": sorted(files), "kind": kind, "family": family, "seed": seed, **extra}

    for root in roots:
        for d in glob.glob(f"{root}/**/runs/*/pred_*", recursive=True):
            run, split = Path(d).parent.name, Path(d).name[5:]
            if split.startswith("dev_select"):
                continue
            m = RUN_RE.match(run)
            if m:
                fam = RUN_FAMILY[m.group(1)] + "/" + m.group(2)
                add(run, split, glob.glob(f"{d}/rank*.jsonl"), "llm", fam, int(m.group(3)))
        for d in glob.glob(f"{root}/**/runs/ablate-*/abl_*", recursive=True):
            run, var = Path(d).parent.name, Path(d).name[4:]
            split = _ablation_split(d, var)
            src = RUN_RE.match(run[len("ablate-"):])
            if var.startswith("T-") and src:
                base_fam, model, seed = RUN_FAMILY[src.group(1)], src.group(2), int(src.group(3))
                fam = f"{base_fam}-{var[2:]}/{model}" if var != "T-extra" else f"{base_fam}/{model}"
                add(f"{run}-{var}", split, glob.glob(f"{d}/*.jsonl"), "llm", fam, seed, ablation=var)
            else:
                add(f"{run}-{var}", split, glob.glob(f"{d}/rank*.jsonl"), "llm", f"ablation/{run}/{var}", None)
        for f in glob.glob(f"{root}/**/step4c/*/final/seed*/pred_*.jsonl.gz", recursive=True):
            p = Path(f)
            model, seed, split = p.parents[2].name, int(p.parent.name[4:]), p.name[5:-9]
            add(f"s2s-{model}-s{seed}", split, [f], "s2s", f"seq2seq/{model}", seed)
        for f in glob.glob(f"{root}/**/step4d/*/*/seed*/pred_test_full.jsonl.gz", recursive=True):
            p = Path(f)
            model, cond, seed = p.parents[2].name, p.parents[1].name, int(p.parent.name[4:])
            add(f"unseen-{model}-{cond}-s{seed}", "test_full", [f], "s2s", f"unseen/{model}/{cond}", seed)
        for f in glob.glob(f"{root}/**/step4[ab]/**/final/**/pred_*_primary.jsonl.gz", recursive=True):
            p = Path(f)
            split = p.name[5:].replace("_primary.jsonl.gz", "")
            seed_dir = p.parent.name if p.parent.name.startswith("seed") else None
            model = (p.parents[2] if seed_dir else p.parents[1]).name
            seed = int(seed_dir[4:]) if seed_dir else None
            add(f"clf-{model}" + (f"-s{seed}" if seed else ""), split, [f], "clf", f"classifier/{model}", seed)
        for d in glob.glob(f"{root}/**/step5/*/*", recursive=True):
            files = glob.glob(f"{d}/rank*.jsonl")
            if files:
                add(f"prompt-{Path(d).parent.name}-{Path(d).name}", "test_prompt", files, "prompt",
                    f"prompt/{Path(d).parent.name}/{Path(d).name}", None)
    if prompt_subset:
        for (system, split), meta in list(found.items()):
            if split == "test_full" and meta["kind"] != "prompt" and not meta["family"].startswith("unseen/"):
                add(system, SUBSET_SPLIT, meta["files"], meta["kind"], meta["family"], meta["seed"],
                    **{k: v for k, v in meta.items() if k not in ("files", "kind", "family", "seed")},
                    src_split="test_full", subset=PROMPT_SUBSET)
    return found


def _mt_key(it: dict) -> tuple:
    return serial_of(it["id"]), it["tier"], it["src_field"], it["tgt_field"]


def _id_key(it: dict) -> tuple:
    return serial_of(it["id"]), it["label"], it["script"]


def subset_maps(items: Items, subset: str) -> tuple[dict, dict]:
    """Content keys of the subset's items -> the subset's own item ids (so that records pair with step 09 outputs)."""
    key = ("__subset_maps__", subset)
    if key not in items.cache:
        ev = items.get(subset)
        items.cache[key] = ({_mt_key(it): i for i, it in ev["mt"].items()},
                            {_id_key(it): i for i, it in ev["id"].items()})
    return items.cache[key]


def load_records(meta: dict, split: str, items: Items) -> dict:
    """Unified records: {"mt": [...], "id": [...]} joined with the frozen items (source, reference, groups)."""
    ev = items.get(meta.get("src_split", split))
    sub_mt, sub_id = subset_maps(items, meta["subset"]) if meta.get("subset") else (None, None)
    mt, idr = [], []
    for f in meta["files"]:
        for r in read_lines(Path(f)):
            rid = r["id"]
            if meta["kind"] == "clf" or ("label" in r and "hyp" not in r):
                it = ev["id"].get(rid)
                if it is None:
                    continue
                if sub_id is not None:
                    rid = sub_id.get(_id_key(it))
                    if rid is None:
                        continue
                idr.append({"id": rid, "serial": serial_of(rid), "label": it["label"], "script": it["script"],
                            "pred": r.get("pred"), "valid": r.get("valid_json", True)})
                continue
            auto = rid.endswith("|auto")
            base_id = rid[:-5] if auto else rid
            it = ev["mt"].get(base_id)
            if it is None:
                continue
            if sub_mt is not None:
                base_id = sub_mt.get(_mt_key(it))
                if base_id is None:
                    continue
                rid = base_id + ("|auto" if auto else "")
            rec = {"id": rid, "serial": serial_of(base_id), "tier": "tier1-auto" if auto else it["tier"],
                   "src_field": it["src_field"], "tgt_field": it["tgt_field"], "src": it["src"], "ref": it["ref"],
                   "hyp": (r.get("hyp") or "").strip(), "valid": r.get("valid_json", True),
                   "dd_status": it.get("dd_status")}
            rec["groups"] = group_of(rec)
            mt.append(rec)
    return {"mt": mt, "id": idr}


class TfidfFeatures:
    """Same class as in step06_classical_classifiers.py, so that its saved SVM can be unpickled here."""

    def transform(self, texts):
        import scipy.sparse as sp
        return sp.hstack([self.cv.transform(texts), self.wv.transform(texts)]).tocsr()


class Judges:
    def __init__(self, device: str):
        self.svm = self.muril = None
        f = find_one("/kaggle/input/**/step4a/svm/final/model.joblib")
        if f:
            import joblib
            sys.modules["__main__"].TfidfFeatures = TfidfFeatures
            self.svm = joblib.load(f)
            log.info("judge 1: TF-IDF SVM (%s)", f)
        d = find_one("/kaggle/input/**/step4b/muril-large/final/seed13/model/config.json")
        if d:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(d.parent)
            self.muril = AutoModelForSequenceClassification.from_pretrained(d.parent, dtype=torch.float16).to(device).eval()
            self.device = device
            log.info("judge 2: MuRIL-large (%s)", d.parent)
        if not (self.svm and self.muril):
            log.warning("fidelity judges missing (attach the step 06 and step 07 outputs): svm=%s muril=%s",
                        bool(self.svm), bool(self.muril))

    def svm_labels(self, texts):
        clf, feat = self.svm["classifier"], self.svm["features"]
        return [BANGLA[i] for i in clf.decision_function(feat.transform([strip_punct(t) for t in texts])).argmax(1)]

    def muril_labels(self, texts, bs=128):
        import torch
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([strip_punct(t) or "." for t in texts[i:i + bs]], truncation=True, max_length=128,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out += self.muril(**enc).logits.argmax(-1).tolist()
        return [BANGLA[i] for i in out]

    def label(self, texts) -> dict:
        res = {}
        if self.svm:
            res["svm"] = self.svm_labels(texts)
        if self.muril:
            res["muril"] = self.muril_labels(texts)
        return res


def bleu_metric():
    """sacreBLEU with the flores-200 SentencePiece tokenizer."""
    from sacrebleu.metrics import BLEU
    if BLEU_TOKENIZER["name"] is None:
        try:
            BLEU(tokenize="flores200")
            BLEU_TOKENIZER["name"] = "flores200"
        except Exception as e:
            BLEU_TOKENIZER["name"] = "char"
            log.warning("FLORES-200 SentencePiece model not available (%s): BLEU uses character tokenization", e)
    return BLEU(tokenize=BLEU_TOKENIZER["name"])


class WSTok:
    def tokenize(self, text):
        return text.split()


def mt_direction_metrics(recs: list[dict], bert_f1: dict | None) -> dict:
    """All corpus/sentence metrics for one direction."""
    import numpy as np
    import jiwer
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU, CHRF
    hyps, refs, srcs = [r["hyp"] for r in recs], [r["ref"] for r in recs], [r["src"] for r in recs]
    chrfpp, chrf, bleu = CHRF(word_order=2), CHRF(word_order=0), bleu_metric()
    rs = rouge_scorer.RougeScorer(["rougeL"], tokenizer=WSTok())
    out = {"n": len(recs), "chrF++": chrfpp.corpus_score(hyps, [refs]).score, "chrF": chrf.corpus_score(hyps, [refs]).score,
           "BLEU": bleu.corpus_score(hyps, [refs]).score,
           "METEOR": 100 * float(np.mean([meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)])),
           "ROUGE-L": 100 * float(np.mean([rs.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)])),
           "CER": 100 * jiwer.cer(refs, [h if h else " " for h in hyps]),
           "WER": 100 * jiwer.wer(refs, [h if h else " " for h in hyps]),
           "copy_chrF++": chrfpp.corpus_score(srcs, [refs]).score,
           "copy_rate": 100 * float(np.mean([h == s.strip() for h, s in zip(hyps, srcs)])),
           "valid_rate": 100 * float(np.mean([bool(r["valid"]) for r in recs]))}
    out["gain_over_copy"] = out["chrF++"] - out["copy_chrF++"]
    if bert_f1 is not None:
        out["BERTScore"] = 100 * float(np.mean([bert_f1[r["id"]] for r in recs]))
    return out


def bertscore_f1(recs: list[dict], scorer) -> dict:
    """Sentence-level BERTScore F1 per record id (one batched call)."""
    if not recs or scorer is None:
        return None
    hyps = [r["hyp"] if r["hyp"] else "." for r in recs]
    _, _, f1 = scorer.score(hyps, [r["ref"] for r in recs], batch_size=128)
    return {r["id"]: float(v) for r, v in zip(recs, f1.tolist())}


def error_category(r: dict, svm_label: str | None, muril_label: str | None) -> str:
    """Automatic error categories (first matching rule wins), replacing manual error tags."""
    hyp, tgt = r["hyp"], r["tgt_field"]
    if not r["valid"] and not hyp:
        return "invalid_output"
    if not hyp:
        return "empty"
    if hyp == r["src"].strip():
        return "copied_input"
    lat, ben = len(LATIN.findall(hyp)), len(BENGALI.findall(hyp))
    if tgt == "en:lat" and ben > lat:
        return "wrong_script"
    if tgt != "en:lat" and lat > ben:
        return "wrong_script"
    want = tgt.split(":")[0]
    if want in BANGLA and svm_label and muril_label and svm_label != want and muril_label != want:
        return "wrong_variety_std" if svm_label == muril_label == "std" else "wrong_variety_other"
    ratio = len(hyp.split()) / max(1, len(r["ref"].split()))
    if ratio < 0.5 or ratio > 2.0:
        return "length_anomaly"
    return "no_automatic_error"


def cohen_kappa(a, b) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b)) if a and len(set(a) | set(b)) > 1 else None


def id_metrics(recs: list[dict]) -> list[dict]:
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    rows = []
    for script in ("bn", "rom"):
        sub = [r for r in recs if r["script"] == script]
        if not sub:
            continue
        y = [r["label"] for r in sub]
        p = [r["pred"] if r["pred"] in BANGLA else "invalid" for r in sub]
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=BANGLA, zero_division=0)
        row = {"script": script, "n": len(sub), "accuracy": 100 * accuracy_score(y, p),
               "macro_f1": 100 * f1_score(y, p, labels=BANGLA, average="macro", zero_division=0),
               "weighted_f1": 100 * f1_score(y, p, labels=BANGLA, average="weighted", zero_division=0),
               "valid_rate": 100 * float(np.mean([bool(r["valid"]) for r in sub])),
               "confusion": json.dumps(confusion_matrix(y, p, labels=BANGLA).tolist())}
        for i, v in enumerate(BANGLA):
            row.update({f"P_{v}": 100 * pr[i], f"R_{v}": 100 * rc[i], f"F1_{v}": 100 * f1[i]})
        rows.append(row)
    return rows


def _weights(n_serials, rng):
    import numpy as np
    return np.bincount(rng.integers(0, n_serials, n_serials), minlength=n_serials)


def boot_mt(a: list[dict], b: list[dict], groups: tuple[str, ...]) -> dict:
    """chrF++ macro-averaged over the directions of `groups`; paired over common items, resampled by serial."""
    import numpy as np
    from sacrebleu.metrics import CHRF
    A = {r["id"]: r for r in a if set(r["groups"]) & set(groups)}
    B = {r["id"]: r for r in b if set(r["groups"]) & set(groups)}
    ids = sorted(set(A) & set(B))
    if not ids:
        return {}
    serials = sorted({A[i]["serial"] for i in ids})
    sidx = {s: j for j, s in enumerate(serials)}
    chrf = CHRF(word_order=2)
    dirs = sorted({(A[i]["src_field"], A[i]["tgt_field"], A[i]["tier"]) for i in ids})
    data = []
    for d in dirs:
        di = [i for i in ids if (A[i]["src_field"], A[i]["tgt_field"], A[i]["tier"]) == d]
        refs = [[A[i]["ref"] for i in di]]
        data.append((np.array([sidx[A[i]["serial"]] for i in di]),
                     np.array(chrf._extract_corpus_statistics([A[i]["hyp"] for i in di], refs)),
                     np.array(chrf._extract_corpus_statistics([B[i]["hyp"] for i in di], refs))))

    def s(w, which):
        vals = []
        for ser, sa, sb in data:
            m = sa if which == "A" else sb
            vals.append(chrf._compute_score_from_stats(list((m * w[ser][:, None]).sum(0))).score)
        return float(np.mean(vals))
    ones = np.ones(len(serials), dtype=int)
    pa, pb = s(ones, "A"), s(ones, "B")
    rng = np.random.default_rng(BOOT["seed"])
    diffs = np.array([s(w, "A") - s(w, "B") for w in (_weights(len(serials), rng) for _ in range(BOOT["n"]))])
    return _summary(pa, pb, diffs, len(serials), len(ids), "chrF++")


def boot_id(a: list[dict], b: list[dict], script="bn") -> list[dict]:
    """Macro-F1 (paired bootstrap over serials) and accuracy (McNemar exact) on common items."""
    import numpy as np
    from scipy.stats import binomtest
    from sklearn.metrics import f1_score
    A = {r["id"]: r for r in a if r["script"] == script}
    B = {r["id"]: r for r in b if r["script"] == script}
    ids = sorted(set(A) & set(B))
    if not ids:
        return []
    serials = sorted({A[i]["serial"] for i in ids})
    sidx = {s: j for j, s in enumerate(serials)}
    lab = np.array([BANGLA.index(A[i]["label"]) for i in ids])
    pa = np.array([BANGLA.index(A[i]["pred"]) if A[i]["pred"] in BANGLA else -1 for i in ids])
    pb = np.array([BANGLA.index(B[i]["pred"]) if B[i]["pred"] in BANGLA else -1 for i in ids])
    ser = np.array([sidx[A[i]["serial"]] for i in ids])

    def mf1(w, p):
        ww = w[ser]
        keep = ww > 0
        return 100 * f1_score(np.repeat(lab[keep], ww[keep]), np.repeat(p[keep], ww[keep]), labels=range(10),
                              average="macro", zero_division=0)
    ones = np.ones(len(serials), dtype=int)
    rng = np.random.default_rng(BOOT["seed"])
    diffs = np.array([mf1(w, pa) - mf1(w, pb) for w in (_weights(len(serials), rng) for _ in range(BOOT["n"]))])
    out = [_summary(mf1(ones, pa), mf1(ones, pb), diffs, len(serials), len(ids), f"macro-F1 ({script})")]
    ca, cb = pa == lab, pb == lab
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    out.append({"metric": f"accuracy ({script}, McNemar)", "A": 100 * ca.mean(), "B": 100 * cb.mean(),
                "diff": 100 * (ca.mean() - cb.mean()), "ci_low": None, "ci_high": None, "p": float(p),
                "n_serials": len(serials), "n_items": len(ids), "discordant": f"{n01}/{n10}"})
    return out


def _summary(pa, pb, diffs, n_serials, n_items, metric) -> dict:
    import numpy as np
    p = min(1.0, 2 * min((np.sum(diffs <= 0) + 1) / (len(diffs) + 1), (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)))
    return {"metric": metric, "A": pa, "B": pb, "diff": pa - pb, "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p": float(p), "n_serials": n_serials, "n_items": n_items}


def holm(rows: list[dict]) -> None:
    """Holm-Bonferroni within each family (in place)."""
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    for rs in fams.values():
        order = sorted(range(len(rs)), key=lambda i: rs[i]["p"])
        m, running = len(rs), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * rs[i]["p"]))
            rs[i]["p_holm"] = running
            rs[i]["significant"] = running < ALPHA


def median_seed(family_rows: list[dict], key: str) -> str | None:
    rows = sorted([r for r in family_rows if r.get(key) is not None], key=lambda r: r[key])
    return rows[len(rows) // 2]["system"] if rows else None


def evaluate(args) -> None:
    import numpy as np
    import pandas as pd
    out = Path(args.output_dir, "step8")
    out.mkdir(parents=True, exist_ok=True)
    items = Items(Path(args.artifacts_dir))
    found = discover([args.output_dir, "/kaggle/input"], prompt_subset=not args.no_prompt_subset)
    if args.systems:
        found = {k: v for k, v in found.items() if re.search(args.systems, k[0])}
    if args.splits:
        found = {k: v for k, v in found.items() if k[1] in args.splits.split(",")}
    log.info("prediction sets found: %d", len(found))
    for (sysname, split), m in sorted(found.items()):
        log.info("  %-48s %-11s %s (%d files)", sysname, split, m["kind"], len(m["files"]))
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    judges = Judges(device) if not args.no_fidelity else None
    scorer = None
    if not args.no_bertscore:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type=BERTSCORE_MODEL, device=device)
        log.info("BERTScore: %s on %s", BERTSCORE_MODEL, device)
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception as e:
        log.warning("nltk data not downloaded (%s); METEOR may fail", e)

    mt_rows, dir_rows, id_rows, fid_rows, err_rows, sysrecs = [], [], [], [], [], {}
    ceiling_cache = {}
    for (sysname, split), meta in sorted(found.items()):
        t0 = time.time()
        recs = load_records(meta, split, items)
        sysrecs[(sysname, split)] = recs
        base = {"system": sysname, "split": split, "family": meta["family"], "seed": meta["seed"], "kind": meta["kind"]}
        if recs["id"]:
            for row in id_metrics(recs["id"]):
                id_rows.append({**base, **row})
        mt = recs["mt"]
        if not mt:
            continue
        bert = bertscore_f1(mt, scorer)
        by_dir = defaultdict(list)
        for r in mt:
            by_dir[(r["tier"], r["src_field"], r["tgt_field"])].append(r)
        per_dir = {}
        for d, rs in by_dir.items():
            per_dir[d] = mt_direction_metrics(rs, bert)
            dir_rows.append({**base, "tier": d[0], "src": d[1], "tgt": d[2], "groups": ",".join(rs[0]["groups"]),
                             **{k: round(v, 3) if isinstance(v, float) else v for k, v in per_dir[d].items()}})
        groups = sorted({g for r in mt for g in r["groups"]})
        for g in groups:
            ds = [d for d, rs in by_dir.items() if g in rs[0]["groups"]]
            keys = [k for k in per_dir[ds[0]] if k != "n"]
            mt_rows.append({**base, "group": g, "n_directions": len(ds), "n_items": sum(per_dir[d]["n"] for d in ds),
                            **{k: round(float(np.mean([per_dir[d][k] for d in ds])), 3) for k in keys if
                               all(k in per_dir[d] for d in ds)}})
        if judges and (judges.svm or judges.muril):
            tgt_bn = [r for r in mt if r["tgt_field"] != "en:lat"]
            labs = judges.label([r["hyp"] for r in tgt_bn])
            ckey = (split, tuple(sorted(r["id"] for r in tgt_bn))[:3], len(tgt_bn))
            if ckey not in ceiling_cache:
                ceiling_cache[ckey] = judges.label([r["ref"] for r in tgt_bn])
            ref_labs = ceiling_cache[ckey]
            for gi, g in enumerate(groups):
                idx = [i for i, r in enumerate(tgt_bn) if g in r["groups"]]
                if not idx:
                    continue
                want = [tgt_bn[i]["tgt_field"].split(":")[0] for i in idx]
                row = {**base, "group": g, "n": len(idx)}
                for j in labs:
                    row[f"fidelity_{j}"] = 100 * float(np.mean([labs[j][i] == w for i, w in zip(idx, want)]))
                    row[f"ceiling_{j}"] = 100 * float(np.mean([ref_labs[j][i] == w for i, w in zip(idx, want)]))
                if "svm" in labs and "muril" in labs:
                    row["judge_kappa"] = cohen_kappa([labs["svm"][i] for i in idx], [labs["muril"][i] for i in idx])
                    row["fidelity_both"] = 100 * float(np.mean([labs["svm"][i] == w and labs["muril"][i] == w
                                                                for i, w in zip(idx, want)]))
                    dial = [i for i, w in zip(idx, want) if w != "std"]
                    if dial:
                        row["standardization_rate"] = 100 * float(np.mean([labs["svm"][i] == "std" and labs["muril"][i] == "std"
                                                                            for i in dial]))
                fid_rows.append(row)
            for g in groups:
                rs = [(r, i) for i, r in enumerate(tgt_bn) if g in r["groups"]]
                if not rs:
                    continue
                cats = [error_category(r, labs.get("svm", [None] * len(tgt_bn))[i], labs.get("muril", [None] * len(tgt_bn))[i])
                        for r, i in rs]
                cnt = pd.Series(cats).value_counts(normalize=True) * 100
                err_rows.append({**base, "group": g, "n": len(rs), **{f"err_{k}": round(v, 2) for k, v in cnt.items()}})
        log.info("%s / %s: %d MT + %d ID items evaluated (%.0f s)", sysname, split, len(mt), len(recs["id"]), time.time() - t0)

    comet = sorted(set(glob.glob("/kaggle/input/**/step8b/comet_directions*.csv", recursive=True)
                       + glob.glob(f"{args.output_dir}/step8b/comet_directions*.csv")), key=os.path.getmtime)
    dfd = pd.DataFrame(dir_rows)
    if comet and len(dfd):
        c = pd.concat([pd.read_csv(f) for f in comet]).drop_duplicates(["system", "split", "tier", "src", "tgt"], keep="last")
        dfd = dfd.merge(c[["system", "split", "tier", "src", "tgt", "COMET"]], on=["system", "split", "tier", "src", "tgt"], how="left")
        log.info("COMET merged from %d file(s): %s", len(comet), ", ".join(comet))
        miss = dfd[dfd["COMET"].isna()].groupby(["system", "split"]).size()
        if len(miss):
            log.warning("COMET missing for %d system/split pairs (run step13_comet.py for them): %s", len(miss),
                        ", ".join(f"{a}/{b}" for a, b in miss.index[:20]))
    dfd.to_csv(out / "mt_directions.csv", index=False)
    dfm = pd.DataFrame(mt_rows)
    if comet and len(dfm):
        cg = dfd.assign(g=dfd["groups"].str.split(",")).explode("g").groupby(["system", "split", "g"])["COMET"].mean().reset_index()
        dfm = dfm.merge(cg.rename(columns={"g": "group"}), on=["system", "split", "group"], how="left")
    dfm.to_csv(out / "mt_metrics.csv", index=False)
    dfi = pd.DataFrame(id_rows)
    dfi.to_csv(out / "id_metrics.csv", index=False)
    pd.DataFrame(fid_rows).to_csv(out / "fidelity.csv", index=False)
    pd.DataFrame(err_rows).to_csv(out / "errors.csv", index=False)

    agg = []
    for df, keys, name in ((dfm, ["split", "group"], "mt"), (dfi, ["split", "script"], "id")):
        if not len(df):
            continue
        num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("seed", "n", "n_items", "n_directions")]
        for (fam, *kv), g in df.groupby(["family"] + keys):
            row = {"table": name, "family": fam, **dict(zip(keys, kv)), "n_seeds": g["seed"].nunique() if g["seed"].notna().any() else 1}
            for c in num:
                row[f"{c}_mean"] = round(float(g[c].mean()), 3)
                row[f"{c}_sd"] = round(float(g[c].std(ddof=1)), 3) if len(g) > 1 else None
            agg.append(row)
    pd.DataFrame(agg).to_csv(out / "seeds_summary.csv", index=False)

    sig = significance(dfm, dfi, sysrecs)
    holm(sig)
    pd.DataFrame(sig).to_csv(out / "significance.csv", index=False)
    manifest = {"script": "step14_evaluate.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
                "finished_utc": datetime.now(timezone.utc).isoformat(), "systems": sorted({k[0] for k in found}),
                "bootstrap": BOOT, "alpha": ALPHA, "bertscore_model": BERTSCORE_MODEL if scorer else None,
                "bleu_tokenizer": BLEU_TOKENIZER["name"],
                "judges": {"svm": bool(judges and judges.svm), "muril": bool(judges and judges.muril)},
                "python": sys.version.split()[0]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with pd.option_context("display.width", 230, "display.max_columns", 30, "display.max_rows", 200):
        if len(dfm):
            cols = [c for c in ["system", "split", "group", "chrF++", "chrF", "BLEU", "BERTScore", "COMET", "gain_over_copy",
                                "copy_rate"] if c in dfm.columns]
            log.info("TRANSLATION (macro over directions)\n%s", dfm[dfm.group.isin(["tier1", "d2s", "s2d", "tier2", "tier3"])]
                     [cols].to_string(index=False))
        if len(dfi):
            log.info("IDENTIFICATION\n%s", dfi[["system", "split", "script", "accuracy", "macro_f1", "valid_rate"]].to_string(index=False))
        if sig:
            log.info("SIGNIFICANCE\n%s", pd.DataFrame(sig)[["family", "comparison", "metric", "A", "B", "diff", "ci_low",
                                                           "ci_high", "p", "p_holm", "significant"]].to_string(index=False))


def significance(dfm, dfi, sysrecs) -> list[dict]:
    """Comparisons of the paper, built from the systems that are present."""
    rows = []

    def add(family, comparison, a_key, b_key, kind, **kw):
        a, b = sysrecs.get(a_key), sysrecs.get(b_key)
        if not a or not b:
            return
        if kind == "mt":
            r = boot_mt(a["mt"], b["mt"], kw["groups"])
            if r:
                rows.append({"family": family, "comparison": comparison, "A_system": a_key[0], "B_system": b_key[0],
                             "split": a_key[1], "groups": "+".join(kw["groups"]), **r})
        else:
            for r in boot_id(a["id"], b["id"], kw.get("script", "bn")):
                rows.append({"family": family, "comparison": comparison, "A_system": a_key[0], "B_system": b_key[0],
                             "split": a_key[1], "groups": "identification", **r})

    def median_of(prefix_family, split, df, metric_col, filt):
        sub = df[(df.family == prefix_family) & (df.split == split) & filt(df)]
        if not len(sub):
            return None
        sub = sub.sort_values(metric_col)
        return sub.iloc[len(sub) // 2]["system"]

    if len(dfm):
        split = "test_full"
        qlora_fams = [f for f in dfm.family.unique() if f.startswith("QLoRA/")]
        s2s = dfm[(dfm.family.str.startswith("seq2seq/")) & (dfm.split == split) & (dfm.group == "tier1")]
        if qlora_fams and len(s2s):
            best_fam = s2s.groupby("family")["chrF++"].mean().idxmax()
            d_sys = median_of(qlora_fams[0], split, dfm, "chrF++", lambda d: d.group == "tier1")
            b_sys = median_of(best_fam, split, dfm, "chrF++", lambda d: d.group == "tier1")
            for g in ("d2s", "s2d"):
                add("H1", f"QLoRA vs {best_fam} ({g})", (d_sys, split), (b_sys, split), "mt", groups=(g,))
    def med(fam):
        return median_of(fam, "test_full", dfm, "chrF++", lambda d: d.group == "tier1") if len(dfm) else None

    def both(family, label, a, b, split="test_full", mt_groups=("d2s", "s2d"), with_id=True):
        if not a or not b:
            return
        for g in mt_groups:
            add(family, f"{label} ({g})", (a, split), (b, split), "mt", groups=(g,))
        if with_id:
            add(family, f"{label} (identification)", (a, split), (b, split), "id")

    fams = sorted(dfm.family.unique()) if len(dfm) else []
    best_s2s = None
    if len(dfm):
        s2s = dfm[(dfm.family.str.startswith("seq2seq/")) & (dfm.split == "test_full") & (dfm.group == "tier1")]
        best_s2s = s2s.groupby("family")["chrF++"].mean().idxmax() if len(s2s) else None
    qlora = [f for f in fams if f.startswith("QLoRA/") or (f.startswith("QLoRA-base/") and not f.startswith("QLoRA-base-"))]
    q_sys = {f: med(f) for f in qlora}
    q_sys = {f: v for f, v in q_sys.items() if v}
    best_clf_sys, best_clf = None, None
    if len(dfi):
        clf = dfi[(dfi.family.str.startswith("classifier/")) & (dfi.split == "test_full") & (dfi.script == "bn")]
        if len(clf):
            best_clf = clf.groupby("family")["macro_f1"].mean().idxmax()
            cb = clf[clf.family == best_clf].sort_values("macro_f1")
            best_clf_sys = cb.iloc[len(cb) // 2]["system"]
    names = sorted(q_sys)
    for i, fa in enumerate(names):
        for fb in names[i + 1:]:
            both("LLM-pairs", f"{fa} vs {fb}", q_sys[fa], q_sys[fb])
    for f, s_ in sorted(q_sys.items()):
        if best_s2s:
            both("LLM-vs-seq2seq", f"{f} vs {best_s2s}", s_, med(best_s2s), with_id=False)
        if best_clf_sys:
            add("LLM-vs-classifier", f"{f} vs {best_clf} (identification)", (s_, "test_full"), (best_clf_sys, "test_full"), "id")
    for f in fams:
        if f.startswith("QLoRA-plain/"):
            model = f.split("/", 1)[1]
            p_sys = med(f)
            rec = next((g for g in q_sys if g.split("/", 1)[1] == model), None)
            if rec:
                both("Plain", f"{rec} vs {f}", q_sys[rec], p_sys)
                both("Plain", f"{rec} without examples at inference vs {f}", f"ablate-{q_sys[rec]}-T-noex", p_sys)
            if best_s2s:
                both("Plain", f"{f} vs {best_s2s}", p_sys, med(best_s2s), with_id=False)
    for f, s_ in sorted(q_sys.items()):
        both("NoEx", f"{f}: retrieved examples vs none at inference", s_, f"ablate-{s_}-T-noex")
    trained = dict(q_sys)
    if best_s2s:
        trained[best_s2s] = med(best_s2s)
    if len(dfm):
        pr = dfm[(dfm.family.str.startswith("prompt/")) & (dfm.split == PROMPT_SUBSET) & (dfm.group == "tier1")
                 & dfm.system.str.contains("-mt_t1-") & ~dfm.system.str.endswith("-cot")]
        if len(pr):
            best_p = pr.sort_values("chrF++").iloc[-1]["system"]
            for f, s_ in sorted(trained.items()):
                for g in ("d2s", "s2d"):
                    add("Subset", f"{f} vs {best_p} ({g})", (s_, SUBSET_SPLIT), (best_p, PROMPT_SUBSET), "mt", groups=(g,))
    if len(dfi):
        pi = dfi[(dfi.family.str.startswith("prompt/")) & (dfi.split == PROMPT_SUBSET) & (dfi.script == "bn")
                 & dfi.system.str.contains("-id_bn-") & ~dfi.system.str.endswith("-cot")]
        if len(pi):
            best_pi = pi.sort_values("macro_f1").iloc[-1]["system"]
            for f, s_ in sorted(q_sys.items()):
                add("Subset", f"{f} vs {best_pi} (identification)", (s_, SUBSET_SPLIT), (best_pi, PROMPT_SUBSET), "id")
    for f in fams:
        if not f.startswith("unseen/"):
            continue
        _, model, cond = f.split("/")
        if cond == "full":
            continue
        u_sys = med(f)
        if f"unseen/{model}/full" in fams:
            ref_sys = med(f"unseen/{model}/full")
        elif (f"s2s-{model}-s13", "test_full") in sysrecs:
            ref_sys = f"s2s-{model}-s13"
        else:
            ref_sys = med(f"seq2seq/{model}")
        a_recs, b_recs = sysrecs.get((u_sys, "test_full")), sysrecs.get((ref_sys, "test_full"))
        if not (ref_sys and u_sys and a_recs and b_recs):
            continue
        if cond == "std-only":
            held = list(DIALECTS)
        elif cond.startswith("one-"):
            held = [d for d in DIALECTS if d != cond[4:]]
        elif cond.startswith("lodo-"):
            held = [cond[5:]]
        else:
            held = [d for d in DIALECTS if d not in cond[4:].split("-")]
        sel = lambda r: r["tgt_field"] == "std:bn" and r["src_field"].split(":")[0] in held and r["tier"] == "tier1"
        r = boot_mt([x for x in a_recs["mt"] if sel(x)], [x for x in b_recs["mt"] if sel(x)], ("d2s",))
        if r:
            rows.append({"family": "Unseen", "comparison": f"{model} {cond}: unseen vs seen ({'+'.join(held)} -> std)",
                         "A_system": u_sys, "B_system": ref_sys, "split": "test_full", "groups": "d2s:" + "+".join(held), **r})
    ref = next((k for k in sysrecs if k[1] == "dev_full" and k[0].startswith("primary-") and k[0].endswith("-s13")), None)
    rnd = next((k for k in sysrecs if k[0].endswith("-A2-random") and k[0].startswith("ablate-primary-")), None)
    if ref and rnd:
        add("H3", "QLoRA: retrieved vs random examples (D2S)", ref, rnd, "mt", groups=("d2s",))
        add("H3", "QLoRA: retrieved vs random examples (identification)", ref, rnd, "id")
    for (sysname, split) in sorted(sysrecs):
        m = re.match(r"^prompt-(.+)-(id_bn|mt_t1)-fsretr-k(\d+)$", sysname)
        if m:
            model, task, k = m.groups()
            other = (f"prompt-{model}-{task}-fsrand-k{k}", split)
            if task == "mt_t1":
                add("H3", f"{model}: retrieved vs random, k={k}, Tier 1", (sysname, split), other, "mt", groups=("tier1",))
            else:
                add("H3", f"{model}: retrieved vs random, k={k}, identification", (sysname, split), other, "id")
        m = re.match(r"^prompt-(.+)-(id_bn|mt_t1)-cot$", sysname)
        if m:
            model, task = m.groups()
            zs = (f"prompt-{model}-{task}-zs", split)
            if task == "mt_t1":
                add("H4", f"{model}: CoT vs zero-shot, Tier 1 (CoT subset)", (sysname, split), zs, "mt", groups=("tier1",))
            else:
                add("H4", f"{model}: CoT vs zero-shot, identification", (sysname, split), zs, "id")
    if len(dfi):
        split = "test_full"
        clf = dfi[(dfi.family.str.startswith("classifier/")) & (dfi.split == split) & (dfi.script == "bn")]
        qlora_id = dfi[(dfi.family.str.startswith("QLoRA/")) & (dfi.split == split) & (dfi.script == "bn")]
        if len(clf) and len(qlora_id):
            best = clf.groupby("family")["macro_f1"].mean().idxmax()
            c_sys = clf[clf.family == best].sort_values("macro_f1").iloc[len(clf[clf.family == best]) // 2]["system"]
            d_sys = qlora_id.sort_values("macro_f1").iloc[len(qlora_id) // 2]["system"]
            add("ID-baselines", f"QLoRA vs {best} (identification)", (d_sys, split), (c_sys, split), "id")
    return rows


def self_test(args) -> None:
    """Metric and significance code on synthetic predictions built from the frozen items (no GPU, no judges)."""
    import pandas as pd
    items = Items(Path(args.artifacts_dir))
    ev = items.get("test_full")
    rng = random.Random(0)
    def mt_sys(noise, seed):
        recs = []
        for it in ev["mt"].values():
            words = it["ref"].split()
            hyp = " ".join(w if rng.random() > noise else rng.choice(words) for w in words)
            recs.append({"id": it["id"], "hyp": hyp, "valid_json": True})
        return recs
    tmp = Path(args.output_dir, "selftest")

    def write_run(d, noise, abl_eval_set=None):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "rank0.jsonl", "w", encoding="utf-8") as f:
            for r in mt_sys(noise, 0):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for it in ev["id"].values():
                f.write(json.dumps({"id": it["id"], "label": it["label"], "script": it["script"],
                                    "pred": it["label"] if rng.random() > 0.2 else "ctg", "valid_json": True}, ensure_ascii=False) + "\n")
        if abl_eval_set:
            (d / "metrics.json").write_text(json.dumps({"eval_set": abl_eval_set}))

    for name, noise in (("primary-gemma4-e4b-s13", 0.10), ("primary-gemma4-e4b-s42", 0.12), ("primary-gemma4-e4b-s2026", 0.11),
                        ("base-llama31-8b-s13", 0.13), ("base-llama31-8b-s42", 0.14), ("base-llama31-8b-s2026", 0.12),
                        ("plain-gemma4-e4b-s13", 0.15), ("plain-gemma4-e4b-s42", 0.16), ("plain-gemma4-e4b-s2026", 0.14)):
        write_run(tmp / "runs" / name / "pred_test_full", noise)
    for src, noise in (("primary-gemma4-e4b-s2026", 0.18), ("primary-gemma4-e4b-s13", 0.2), ("base-llama31-8b-s13", 0.2)):
        write_run(tmp / "runs" / f"ablate-{src}" / "abl_T-noex", noise, abl_eval_set="test_full" if "s13" in src else None)
    if (items.root / "data" / "eval" / PROMPT_SUBSET / "mt.jsonl").exists():
        evp = items.get(PROMPT_SUBSET)
        for cond, noise in (("mt_t1-fsretr-k3", 0.3), ("mt_t1-zs", 0.5)):
            d = tmp / "step5" / "toy-llm" / cond
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "rank0.jsonl", "w", encoding="utf-8") as f:
                for it in evp["mt"].values():
                    if it["tier"] == "tier1":
                        w = it["ref"].split()
                        f.write(json.dumps({"id": it["id"], "hyp": " ".join(x if rng.random() > noise else rng.choice(w) for x in w),
                                            "valid_json": True}, ensure_ascii=False) + "\n")
        d = tmp / "step5" / "toy-llm" / "id_bn-zs"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "rank0.jsonl", "w", encoding="utf-8") as f:
            for it in evp["id"].values():
                if it["script"] == "bn":
                    f.write(json.dumps({"id": it["id"], "label": it["label"], "script": "bn",
                                        "pred": it["label"] if rng.random() > 0.5 else "std", "valid_json": True}) + "\n")
    for seed in SEEDS:
        d = tmp / "step4c" / "banglat5" / "final" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "pred_test_full.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in mt_sys(0.25, seed):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for cond, noise in (("lodo-nkh", 0.4), ("two-pab-ctg", 0.45)):
        for seed in SEEDS:
            d = tmp / "step4d" / "banglat5" / cond / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / "pred_test_full.jsonl.gz", "wt", encoding="utf-8") as f:
                for r in mt_sys(noise, seed):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    args.output_dir = str(tmp)
    args.no_bertscore, args.no_fidelity = True, True
    evaluate(args)
    sig = pd.read_csv(tmp / "step8" / "significance.csv")
    log.info("self-test significance rows: %d; H1 significant: %s", len(sig), sig[sig.family == "H1"]["significant"].tolist())
    log.info("self-test significance families: %s", sig.groupby("family").size().to_dict())


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 14: evaluation and significance tests")
    p.add_argument("--artifacts-dir", help="step 02 'outputs' folder (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--systems", help="regular expression: evaluate only matching system names")
    p.add_argument("--splits", help="comma-separated splits (default: all found)")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-fidelity", action="store_true")
    p.add_argument("--no-prompt-subset", action="store_true",
                   help=f"do not re-score test_full predictions on the prompting subset (split {SUBSET_SPLIT})")
    p.add_argument("--bootstrap", type=int, default=BOOT["n"],
                   help="bootstrap resamples (default 10000; with Holm over k tests the smallest attainable adjusted p is "
                        "about 2k/(n+1), so families with many tests need e.g. 10000)")
    p.add_argument("--requirements")
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    BOOT["n"] = args.bootstrap
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if not args.artifacts_dir:
        hit = find_one("/kaggle/input/**/data/aligned.parquet")
        if hit is None:
            raise SystemExit("step-02 output not found: attach the output of step02_prepare_data.py")
        args.artifacts_dir = str(hit.parents[1])
    prepare_environment(args)
    if args.self_test:
        self_test(args)
        return
    evaluate(args)


if __name__ == "__main__":
    main()

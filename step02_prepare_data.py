#!/usr/bin/env python3
"""Shared data artifacts for all experiments.

Stages: env (installs pinned packages), prepare (aligned table, frozen evaluation sets, dialect pairs, marker
list), analysis (copy-input baseline, distance confound, marker tables), retrieval (LaBSE embeddings and hybrid
BM25 + LaBSE neighbours with reciprocal rank fusion), represent (silhouette, MMD, linear probes) and lexicon
(dialect-Standard word lexicon from word alignments).

Usage:
    python step02_prepare_data.py --release-dir BUC_release --output-dir outputs
Offline test (no GPU, no downloads):
    python step02_prepare_data.py --release-dir BUC_release --output-dir /tmp/out --no-install --mock --allow-cpu
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

CONFIG = {
    "seed": 13,
    "data": {
        "near_dup_cosine": 0.90,
        "prompting_subset_size": 200,
        "dev_select_size": 150,
        "dd_pairs_per_band": 2,
        "markers_per_dialect": 5,
        "marker_min_count": 20,
    },
    "retrieval": {
        "dense_encoder": "sentence-transformers/LaBSE",
        "dense_batch_size": 256,
        "bm25_ngram_range": [2, 4],
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "rrf_k": 60,
        "candidates_per_method": 50,
        "top_k": 10,
    },
    "lexicon": {
        "encoder": "FacebookAI/xlm-roberta-base",
        "layer": 8,
        "batch_size": 64,
        "max_length": 128,
        "default_min_count": 3,
        "default_min_prob": 0.5,
    },
    "representation": {"n_serials": 800, "probe_folds": 5},
}
STAGES = ["prepare", "analysis", "retrieval", "represent", "lexicon"]
log = logging.getLogger("buc")


def find_one(pattern: str, what: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    if len(hits) > 1:
        log.warning("several %s found, using the first: %s", what, hits)
    return Path(hits[0]) if hits else None


def pinned(req_file: Path) -> dict[str, str]:
    out = {}
    for line in req_file.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def version_mismatches(req: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    bad = {}
    for pkg, ver in req.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        if have != ver:
            bad[pkg] = (ver, have)
    return bad


def prepare_environment(args) -> dict:
    """Install the locked core requirements (never torch) before any third-party import, then check."""
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise SystemExit(f"Python 3.12 or 3.13 required, found {sys.version.split()[0]}")
    if sys.version_info[:2] == (3, 13):
        log.info("Python %s: newer Kaggle image (the pinned packages install unchanged; versions are recorded)",
                 sys.version.split()[0])
    req_file = Path(args.requirements) if args.requirements else find_one("/kaggle/input/**/requirements.txt",
                                                                           "requirements.txt")
    report = {"requirements_file": str(req_file) if req_file else None}
    if req_file and req_file.exists():
        req = pinned(req_file)
        bad = version_mismatches(req)
        if bad and not args.no_install:
            log.info("installing %d pinned packages from %s (2-5 min) ...", len(bad), req_file)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=True)
            bad = version_mismatches(req)
        report["mismatched_after_install"] = {k: {"locked": v[0], "installed": v[1]} for k, v in bad.items()}
        if bad:
            log.warning("%d packages differ from the lock: %s", len(bad), list(bad)[:10])
    else:
        log.warning("requirements.txt not found: package versions are NOT checked")
    return report


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False}
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(cuda=torch.version.cuda, bf16_supported=torch.cuda.is_bf16_supported(),
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    return info


class Manifest:
    """Records config, arguments, versions, GPUs, wall-clock time and peak GPU memory of a stage."""

    def __init__(self, out_dir: Path, stage: str, args, extra_env: dict | None = None):
        self.path = out_dir / "runs" / f"step1_{stage}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        self.stage, self.args, self.extra_env, self.results = stage, args, extra_env or {}, {}

    def __enter__(self):
        self.t0 = time.time()
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass
        log.info("===== stage %s =====", self.stage)
        return self

    def record(self, **kv):
        self.results.update(kv)

    def __exit__(self, exc_type, exc, tb):
        peak = {}
        try:
            import torch
            peak = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 3)
                    for i in range(torch.cuda.device_count())}
        except Exception:
            pass
        versions = {}
        for mod in ("numpy", "pandas", "sklearn", "scipy", "transformers", "sentence_transformers", "sacrebleu"):
            try:
                versions[mod] = __import__(mod).__version__
            except Exception:
                versions[mod] = None
        m = {"script": "step02_prepare_data.py", "stage": self.stage, "status": "failed" if exc_type else "ok",
             "error": repr(exc) if exc else None, "wall_clock_s": round(time.time() - self.t0, 1),
             "peak_gpu_mem_gib": peak, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
             "python": sys.version.split()[0], "platform": platform.platform(), "gpu": gpu_info(),
             "versions": versions, "environment": self.extra_env, "args": vars(self.args), "config": CONFIG,
             "results": self.results}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.info("stage %s: %s in %.1f s (manifest: %s)", self.stage, m["status"], m["wall_clock_s"], self.path.name)
        return False


DIALECT_FILES = {"02": ("Noakhali", "nkh"), "03": ("Chittagong", "ctg"), "04": ("Sylhet", "syl"),
                 "05": ("Barishal", "bar"), "06": ("Sirajganj", "srj"), "07": ("Pabna", "pab"),
                 "08": ("Shariatpur", "shr"), "09": ("Old Dhaka", "odh"), "10": ("Khulna", "khl")}
DIALECTS = [code for _, code in DIALECT_FILES.values()]
BANGLA = ["std"] + DIALECTS
FIELDS = [f"{v}:bn" for v in BANGLA] + [f"{v}:rom" for v in BANGLA] + ["en:lat", "cm:mix"]
SPLITS = ("train", "dev", "test")
COORD = {"nkh": (22.87, 91.10), "ctg": (22.36, 91.78), "syl": (24.90, 91.87), "bar": (22.70, 90.37),
         "srj": (24.45, 89.70), "pab": (24.01, 89.24), "shr": (23.24, 90.43), "odh": (23.71, 90.40),
         "khl": (22.82, 89.55)}


def tier1():
    return [(f"{d}:bn", "std:bn") for d in DIALECTS] + [("std:bn", f"{d}:bn") for d in DIALECTS]


def tier2():
    return ([(f"{d}:bn", "en:lat") for d in DIALECTS] + [("en:lat", f"{d}:bn") for d in DIALECTS]
            + [("std:bn", "en:lat"), ("en:lat", "std:bn"), ("cm:mix", "std:bn")])


def tier3(dd_pairs):
    return [(f"{a}:bn", f"{b}:bn") for a, b in dd_pairs] + [(f"{d}:rom", "std:bn") for d in DIALECTS]


ZWJ_KEEP = "\u09B0\u200D\u09CD\u09AF"
KHANDA_TA_LEGACY = "\u09A4\u09CD\u200D"
SPACE_BEFORE_PUNCT = re.compile(r"\s+([\u0964\u0965,;:?!.])")
INVISIBLE = ("\u200b", "\ufeff", "\u00ad")
BN_TOKEN = re.compile(r"[\u0980-\u09FF\u200c\u200d]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def normalize_text(s):
    """Identical to the analysis script; idempotent."""
    if s is None or (isinstance(s, float) and s != s):
        return None
    s = unicodedata.normalize("NFC", str(s))
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    s = s.replace(KHANDA_TA_LEGACY, "\u09CE")
    s = s.replace(ZWJ_KEEP, "\x00").replace("\u200d", "").replace("\x00", ZWJ_KEEP)
    s = re.sub(r"\s+", " ", s)
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    return s.strip()


def strip_punct(s: str) -> str:
    """Remove all Unicode punctuation (P*), incl. the danda; used for identification inputs."""
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return BN_TOKEN.findall(s) if isinstance(s, str) else []


def fingerprint(s: str) -> str:
    t = normalize_text(s)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] if t else ""


def read_release(release_dir: Path):
    """Aligned table: index = serial; 22 normalized fields + category + split."""
    import pandas as pd

    def read_csv(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        df["Serial"] = df["Serial"].astype(int)
        return df.set_index("Serial")

    split_file = pd.read_csv(release_dir / "splits_by_serial.csv")
    split_file["Serial"] = split_file["Serial"].astype(int)
    split_file = split_file.set_index("Serial")
    parts = []
    for split in SPLITS:
        files = {m.group(1): p for p in (release_dir / split).glob("*.csv") if (m := re.match(r"^(\d{2})\.", p.name))}
        missing = {"01", *DIALECT_FILES} - set(files)
        if missing:
            raise FileNotFoundError(f"{release_dir / split}: missing files with prefixes {sorted(missing)}")
        main = read_csv(files["01"])
        tab = pd.DataFrame(index=main.index)
        tab["std:bn"], tab["std:rom"] = main["Standard Bangla Sentence"], main["Banglish Sentence"]
        tab["en:lat"], tab["cm:mix"], tab["category"] = main["English Sentence"], main["Code Mixed Sentence"], main["Category"]
        for prefix, (_, code) in DIALECT_FILES.items():
            df = read_csv(files[prefix])
            if set(df.index) != set(tab.index):
                raise ValueError(f"{files[prefix].name} ({split}): serials differ from 01.Main.csv")
            tab[f"{code}:bn"] = df.loc[tab.index, next(c for c in df.columns if c.endswith("Dialect"))]
            tab[f"{code}:rom"] = df.loc[tab.index, next(c for c in df.columns if c.endswith("Banglish"))]
        tab["split"] = split
        parts.append(tab)
    table = pd.concat(parts).sort_index()
    if table.index.duplicated().any():
        raise ValueError("a serial appears in more than one split folder")
    for col in FIELDS + ["category"]:
        table[col] = table[col].map(normalize_text)
    empty = {c: int((table[c].isna() | (table[c] == "")).sum()) for c in FIELDS}
    if any(empty.values()):
        raise ValueError(f"empty text cells: { {k: v for k, v in empty.items() if v} }")
    missing = set(table.index) - set(split_file.index)
    if missing:
        raise ValueError(f"{len(missing)} serials missing from splits_by_serial.csv, e.g. {sorted(missing)[:5]}")
    if (split_file.loc[table.index, "split"].values != table["split"].values).any():
        raise ValueError("folder assignment differs from splits_by_serial.csv")
    if "std_fingerprint" in split_file.columns:
        changed = int((table["std:bn"].map(fingerprint).values != split_file.loc[table.index, "std_fingerprint"].values).sum())
        if changed > 0.10 * len(table):
            raise ValueError(f"{changed} Standard sentences differ from stored fingerprints: serials renumbered?")
        log.info("split verified; %d Standard sentences edited since the split was frozen", changed)
    table.index.name = "serial"
    return table[FIELDS + ["category", "split"]]


def near_duplicate_groups(std, threshold: float):
    """Union-find groups of Standard sentences with char 3-5-gram TF-IDF cosine >= threshold."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(std.values)
    dist, idx = NearestNeighbors(n_neighbors=min(6, len(std)), metric="cosine").fit(X).kneighbors(X)
    parent = np.arange(len(std))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(std)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1 - d >= threshold:
                parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(std))])
    first = pd.Series(std.index.values).groupby(roots).transform("min").values
    return pd.Series(first, index=std.index, name="group")


def stratified_subset(table, split: str, n: int, seed: int) -> list[int]:
    """n serials of one split, proportional to category (largest-remainder allocation)."""
    import numpy as np
    sub = table[table["split"] == split]
    if n >= len(sub):
        log.warning("requested %d serials but %s has %d: using all", n, split, len(sub))
        return sorted(int(s) for s in sub.index)
    counts = sub["category"].value_counts()
    quota = counts / counts.sum() * n
    alloc = np.floor(quota).astype(int)
    for cat in (quota - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[cat] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for cat, k in alloc.sort_index().items():
        chosen += rng.choice(np.sort(sub.index[sub["category"] == cat].values), size=int(k), replace=False).tolist()
    return sorted(int(s) for s in chosen)


def aligned_distance(table, fields):
    """100 - mean sentence chrF++, symmetrized, between aligned renderings."""
    import numpy as np
    import pandas as pd
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    D = pd.DataFrame(0.0, index=fields, columns=fields)
    for i, a in enumerate(fields):
        for b in fields[i + 1:]:
            xa, xb = table[a].tolist(), table[b].tolist()
            ab = np.mean([chrf.sentence_score(x, [y]).score for x, y in zip(xa, xb)])
            ba = np.mean([chrf.sentence_score(y, [x]).score for x, y in zip(xa, xb)])
            D.loc[a, b] = D.loc[b, a] = 100 - (ab + ba) / 2
    return D


def log_odds_dirichlet(fg: Counter, bg: Counter, prior: Counter, a0: float = 1000.0):
    """Weighted log-odds with informative Dirichlet prior (Monroe et al., 2008) -> z-scores."""
    import numpy as np
    import pandas as pd
    n_fg, n_bg, n_pr = sum(fg.values()), sum(bg.values()), sum(prior.values())
    z = {}
    for w in fg:
        aw = a0 * prior[w] / n_pr
        z[w] = ((np.log((fg[w] + aw) / (n_fg + a0 - fg[w] - aw)) - np.log((bg[w] + aw) / (n_bg + a0 - bg[w] - aw)))
                / np.sqrt(1 / (fg[w] + aw) + 1 / (bg[w] + aw)))
    return pd.Series(z, dtype=float).sort_values(ascending=False)


def select_dd_pairs(D, per_band: int) -> dict:
    """Near / median / far dialect pairs by train distance; within each band the closer pair is 'seen' (sampled in training), the next 'held_out' (never trained on)."""
    pairs = sorted((float(D.loc[a, b]), a, b) for i, a in enumerate(DIALECTS) for b in DIALECTS[i + 1:])
    n, mid = len(pairs), len(pairs) // 2 - per_band // 2
    bands = {"near": pairs[:per_band], "median": pairs[mid:mid + per_band], "far": pairs[n - per_band:]}
    out = {"seen": [], "held_out": [], "bands": {}}
    for name, band in bands.items():
        out["bands"][name] = [{"pair": [a, b], "distance": round(d, 2)} for d, a, b in band]
        for j, (_, a, b) in enumerate(band):
            (out["seen"] if j % 2 == 0 else out["held_out"]).extend([[a, b], [b, a]])
    return out


def select_markers(train, k: int, min_count: int) -> dict:
    """Top-k frequent marker words per dialect (log-odds vs all other varieties)."""
    counts = {v: Counter(t for s in train[f"{v}:bn"] for t in tokens(s)) for v in BANGLA}
    total = sum(counts.values(), Counter())
    out = {}
    for d in DIALECTS:
        z = log_odds_dirichlet(counts[d], total - counts[d], total)
        keep = [w for w in z.index if counts[d][w] >= min_count and counts["std"][w] <= 0.5 * counts[d][w]]
        out[d] = [{"word": w, "z": round(float(z[w]), 2), "count": counts[d][w]} for w in keep[:k]]
    return out


def stage_prepare(args, out: Path) -> dict:
    import pandas as pd
    dc, seed = CONFIG["data"], CONFIG["seed"]
    data = out / "data"
    (data / "eval").mkdir(parents=True, exist_ok=True)
    table = read_release(Path(args.release_dir))
    table["group"] = near_duplicate_groups(table["std:bn"], dc["near_dup_cosine"])
    straddle = table.groupby("group")["split"].nunique()
    straddling = straddle[straddle > 1].index.tolist()
    if straddling:
        log.warning("%d near-duplicate groups straddle splits: %s", len(straddling), straddling[:10])
    table.to_parquet(data / "aligned.parquet")
    n_groups = int((table.groupby("group").size() > 1).sum())
    log.info("aligned table: %d serials %s; %d near-duplicate groups", len(table),
             table["split"].value_counts().to_dict(), n_groups)

    subsets = {"test_prompt": stratified_subset(table, "test", dc["prompting_subset_size"], seed),
               "dev_select": stratified_subset(table, "dev", dc["dev_select_size"], seed)}
    (data / "subsets.json").write_text(json.dumps(subsets, indent=1))

    train = table[table["split"] == "train"]
    log.info("aligned distances between the 10 varieties on the train split (5-10 min on CPU) ...")
    D = aligned_distance(train, [f"{v}:bn" for v in BANGLA])
    D.index = D.columns = BANGLA
    D.round(3).to_csv(data / "distances_train.csv")
    dd = select_dd_pairs(D, dc["dd_pairs_per_band"])
    (data / "dd_pairs.json").write_text(json.dumps(dd, indent=1))
    log.info("Dialect->Dialect  seen: %s  held-out: %s", dd["seen"], dd["held_out"])

    markers = select_markers(train, dc["markers_per_dialect"], dc["marker_min_count"])
    (data / "markers.json").write_text(json.dumps(markers, indent=1, ensure_ascii=False), encoding="utf-8")
    mask_set = {m["word"] for ms in markers.values() for m in ms}

    directions = {"tier1": tier1(), "tier2": tier2(), "tier3": tier3(dd["seen"] + dd["held_out"]),
                  "dd_seen": dd["seen"], "dd_held_out": dd["held_out"]}
    (data / "directions.json").write_text(json.dumps(directions, indent=1))

    sets = {"dev_full": table.index[table["split"] == "dev"].tolist(), "dev_select": subsets["dev_select"],
            "test_full": table.index[table["split"] == "test"].tolist(), "test_prompt": subsets["test_prompt"]}
    counts = {}
    for name, serials in sets.items():
        d = data / "eval" / name
        d.mkdir(parents=True, exist_ok=True)
        n_id = n_mt = 0
        with open(d / "id.jsonl", "w", encoding="utf-8") as f:
            for s in serials:
                row = table.loc[s]
                for v in BANGLA:
                    for script in ("bn", "rom"):
                        text = row[f"{v}:{script}"]
                        nop = strip_punct(text)
                        masked = " ".join(t for t in nop.split() if t not in mask_set) if script == "bn" else None
                        f.write(json.dumps({"id": f"id|{s}|{v}:{script}", "serial": int(s), "category": row["category"],
                                            "label": v, "script": script, "text": text, "text_nopunct": nop,
                                            "text_masked": masked}, ensure_ascii=False) + "\n")
                        n_id += 1
        with open(d / "mt.jsonl", "w", encoding="utf-8") as f:
            for tier in ("tier1", "tier2", "tier3"):
                for src, tgt in directions[tier]:
                    for s in serials:
                        row = table.loc[s]
                        item = {"id": f"mt|{s}|{src}>{tgt}", "serial": int(s), "category": row["category"], "tier": tier,
                                "src_field": src, "tgt_field": tgt, "src": row[src], "ref": row[tgt]}
                        if tier == "tier3" and src.endswith(":bn"):
                            item["dd_status"] = "seen" if [src[:3], tgt[:3]] in dd["seen"] else "held_out"
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        n_mt += 1
        counts[name] = {"serials": len(serials), "id_items": n_id, "mt_items": n_mt}
        log.info("eval set %-11s %s", name, counts[name])
    return {"n_serials": len(table), "near_dup_groups": n_groups, "straddling_groups": straddling,
            "eval_sets": counts, "dd_pairs": dd, "n_mask_words": len(mask_set)}


def accuracy_ceiling(texts, labels) -> float:
    """For every distinct string only its most frequent label can be predicted correctly."""
    import pandas as pd
    df = pd.DataFrame({"t": texts, "y": labels})
    return float(df.groupby("t")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


def mantel(A, B, n_perm: int = 9999, seed: int = 13):
    import numpy as np
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(A, 1)
    r0 = np.corrcoef(A[iu], B[iu])[0, 1]
    hits = sum(np.corrcoef(A[np.ix_(p, p)][iu], B[iu])[0, 1] >= r0 for p in (rng.permutation(len(A)) for _ in range(n_perm)))
    return float(r0), float((hits + 1) / (n_perm + 1))


def haversine_km(p, q) -> float:
    import numpy as np
    la1, lo1, la2, lo2 = map(np.radians, (*p, *q))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def stage_analysis(args, out: Path) -> dict:
    import numpy as np
    import pandas as pd
    from sacrebleu.metrics import CHRF
    from sklearn.feature_extraction.text import CountVectorizer
    data, res = out / "data", out / "analysis"
    res.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(data / "aligned.parquet")

    rows = []
    for name in ("dev_full", "test_full"):
        items = pd.read_json(data / "eval" / name / "id.jsonl", lines=True, dtype=False)
        for script in ("bn", "rom"):
            sub = items[items["script"] == script]
            for col, punct in (("text", "kept"), ("text_nopunct", "removed")):
                for classes, s in (("10-way", sub), ("9-way", sub[sub["label"] != "std"])):
                    rows.append({"set": name, "script": script, "punctuation": punct, "classes": classes, "n_items": len(s),
                                 "ceiling_acc": round(100 * accuracy_ceiling(s[col].tolist(), s["label"].tolist()), 3)})
    ceil = pd.DataFrame(rows)
    ceil.to_csv(res / "accuracy_ceiling.csv", index=False)

    chrfpp, chrf = CHRF(word_order=2), CHRF(word_order=0)
    rows = []
    for name in ("dev_select", "dev_full", "test_prompt", "test_full"):
        items = pd.read_json(data / "eval" / name / "mt.jsonl", lines=True, dtype=False)
        for (tier, src, tgt), g in items.groupby(["tier", "src_field", "tgt_field"], sort=False):
            hyp, ref = g["src"].tolist(), [g["ref"].tolist()]
            rows.append({"set": name, "tier": tier, "src": src, "tgt": tgt, "n": len(g),
                         "chrF++": round(chrfpp.corpus_score(hyp, ref).score, 2),
                         "chrF": round(chrf.corpus_score(hyp, ref).score, 2), "signature": str(chrfpp.get_signature())})
    pd.DataFrame(rows).to_csv(res / "copy_baseline.csv", index=False)

    D = pd.read_csv(data / "distances_train.csv", index_col=0)
    A, s = D.loc[DIALECTS, DIALECTS].values, D.loc["std", DIALECTS].values
    S, iu = s[:, None] + s[None, :], np.triu_indices(len(DIALECTS), 1)
    beta = np.polyfit(S[iu], A[iu], 1)
    R = np.zeros_like(A)
    R[iu] = A[iu] - np.polyval(beta, S[iu])
    R = R + R.T
    G = np.array([[haversine_km(COORD[a], COORD[b]) for b in DIALECTS] for a in DIALECTS])
    (r_raw, p_raw), (r_res, p_res) = mantel(A, G), mantel(R, G)
    pairs = sorted((float(R[i, j]), DIALECTS[i], DIALECTS[j]) for i, j in zip(*iu))
    conf = {"corr_pair_distance_vs_sum_distance_to_standard": round(float(np.corrcoef(A[iu], S[iu])[0, 1]), 3),
            "mantel_raw": {"r": round(r_raw, 3), "p": round(p_raw, 4)},
            "mantel_residual": {"r": round(r_res, 3), "p": round(p_res, 4)},
            "closest_pairs_after_control": [[a, b, round(r, 2)] for r, a, b in pairs[:5]],
            "farthest_pairs_after_control": [[a, b, round(r, 2)] for r, a, b in pairs[-5:]]}
    (res / "distance_confound.json").write_text(json.dumps(conf, indent=1))

    train = table[table["split"] == "train"]
    counts = {v: Counter(t for x in train[f"{v}:bn"] for t in tokens(x)) for v in BANGLA}
    total = sum(counts.values(), Counter())
    rest_rows, std_rows = [], []
    for d in DIALECTS:
        for w, z in log_odds_dirichlet(counts[d], total - counts[d], total).head(25).items():
            rest_rows.append({"dialect": d, "unit": "word", "item": w, "z": round(float(z), 2), "count": counts[d][w]})
        for w, z in log_odds_dirichlet(counts[d], counts["std"], counts[d] + counts["std"]).head(25).items():
            std_rows.append({"dialect": d, "item": w, "z": round(float(z), 2), "count": counts[d][w],
                             "count_in_standard": counts["std"][w]})
    cv = CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, lowercase=False)
    X = cv.fit_transform(pd.concat([train[f"{v}:bn"] for v in BANGLA]))
    labels = np.repeat(BANGLA, len(train))
    vocab = cv.get_feature_names_out()
    per = {v: np.asarray(X[labels == v].sum(0)).ravel() for v in BANGLA}
    tot = sum(per.values())
    prior = Counter(dict(zip(vocab, tot)))
    for d in DIALECTS:
        fg, bg = +Counter(dict(zip(vocab, per[d]))), +Counter(dict(zip(vocab, tot - per[d])))
        for w, z in log_odds_dirichlet(fg, bg, prior).head(25).items():
            rest_rows.append({"dialect": d, "unit": "char2-5", "item": w, "z": round(float(z), 2), "count": fg[w]})
    pd.DataFrame(rest_rows).to_csv(res / "markers_vs_rest.csv", index=False)
    pd.DataFrame(std_rows).to_csv(res / "markers_vs_standard.csv", index=False)

    rows, std_len = [], table["std:bn"].str.split().str.len()
    for d in DIALECTS:
        ratio = table[f"{d}:bn"].str.split().str.len() / std_len
        for s_, r in ratio[(ratio > 2) | (ratio < 0.5)].items():
            rows.append({"serial": int(s_), "split": table.at[s_, "split"], "dialect": d, "ratio": round(float(r), 2),
                         "standard": table.at[s_, "std:bn"], "dialect_text": table.at[s_, f"{d}:bn"]})
    lo = pd.DataFrame(rows, columns=["serial", "split", "dialect", "ratio", "standard", "dialect_text"])
    lo.sort_values(["dialect", "ratio"], ascending=[True, False]).to_csv(res / "length_outliers.csv", index=False)
    log.info("accuracy ceiling (test):\n%s", ceil[ceil["set"] == "test_full"].to_string(index=False))
    return {"ceiling_test": ceil[ceil["set"] == "test_full"].to_dict("records"), "distance_confound": conf,
            "length_outliers_by_dialect": lo["dialect"].value_counts().to_dict()}


class CharBM25:
    """Okapi BM25 over character n-grams within word boundaries, as sparse matrix products."""

    def __init__(self, ngram_range=(2, 4), k1=1.5, b=0.75):
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range), lowercase=False, dtype=np.float32)
        self.k1, self.b = k1, b

    def fit(self, docs):
        import numpy as np
        import scipy.sparse as sp
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        w = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        self.W = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
        return self

    def scores(self, queries):
        q = self.vec.transform(queries).tocsr()
        q.data[:] = 1.0
        return (q @ self.W.T).tocsr()


def topk_rows(S, k):
    import numpy as np
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


def rrf_fuse(rank_lists, k_rrf, top_k):
    import numpy as np
    n = rank_lists[0].shape[0]
    out_idx, out_val = np.full((n, top_k), -1, dtype=np.int64), np.zeros((n, top_k), dtype=np.float32)
    for i in range(n):
        score: dict[int, float] = {}
        for ranks in rank_lists:
            for r, doc in enumerate(ranks[i]):
                if doc >= 0:
                    score[int(doc)] = score.get(int(doc), 0.0) + 1.0 / (k_rrf + r + 1)
        for j, (doc, val) in enumerate(sorted(score.items(), key=lambda kv: -kv[1])[:top_k]):
            out_idx[i, j], out_val[i, j] = doc, val
    return out_idx, out_val


def search(q_text, q_emb, q_group, bm25, d_emb, d_group, n_candidates, top_k, k_rrf, chunk=1024):
    """{'bm25'|'dense'|'hybrid': (doc_idx, score)}."""
    import numpy as np
    res = {m: ([], []) for m in ("bm25", "dense", "hybrid")}
    for s in range(0, len(q_text), chunk):
        e = min(s + chunk, len(q_text))
        sb = bm25.scores(q_text[s:e]).toarray()
        sd = q_emb[s:e].astype(np.float32) @ d_emb.astype(np.float32).T
        mask = q_group[s:e, None] == d_group[None, :]
        sb[mask], sd[mask] = -np.inf, -np.inf
        ib, vb = topk_rows(sb, n_candidates)
        idn, vd = topk_rows(sd, n_candidates)
        ib[~np.isfinite(vb)], idn[~np.isfinite(vd)] = -1, -1
        ih, vh = rrf_fuse([ib, idn], k_rrf, top_k)
        for name, (i, v) in {"bm25": (ib[:, :top_k], vb[:, :top_k]), "dense": (idn[:, :top_k], vd[:, :top_k]),
                             "hybrid": (ih, vh)}.items():
            res[name][0].append(i)
            res[name][1].append(v)
    return {m: (np.concatenate(i), np.concatenate(v)) for m, (i, v) in res.items()}


def stage_retrieval(args, out: Path) -> dict:
    import numpy as np
    import pandas as pd
    rc = CONFIG["retrieval"]
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    (out / "retrieval").mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(out / "data" / "aligned.parquet")
    serials, groups = table.index.values, table["group"].values
    is_train = (table["split"] == "train").values

    texts = sorted(set(t for f in FIELDS for t in table[f]))
    if args.mock:
        E = np.random.default_rng(0).standard_normal((len(texts), 32)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(rc["dense_encoder"], device=device)
        if device == "cuda":
            model.half()
        E = model.encode(texts, batch_size=rc["dense_batch_size"], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        del model
    pos = {t: i for i, t in enumerate(texts)}
    emb = {f: E[[pos[t] for t in table[f]]].astype(np.float16) for f in FIELDS}
    np.savez_compressed(out / "embeddings" / "labse.npz", serial=serials, **{f.replace(":", "__"): e for f, e in emb.items()})
    log.info("LaBSE embeddings: %d distinct texts", len(texts))

    kw = dict(n_candidates=rc["candidates_per_method"], top_k=rc["top_k"], k_rrf=rc["rrf_k"])
    bm = dict(ngram_range=rc["bm25_ngram_range"], k1=rc["bm25_k1"], b=rc["bm25_b"])
    own = {"q_serial": serials}
    for f in FIELDS:
        r = search(table[f].tolist(), emb[f], groups, CharBM25(**bm).fit(table.loc[is_train, f].tolist()),
                   emb[f][is_train], groups[is_train], **kw)
        key = f.replace(":", "__")
        for m, (idx, val) in r.items():
            own[f"{key}__{m}__serial"] = np.where(idx >= 0, serials[is_train][np.maximum(idx, 0)], -1)
            own[f"{key}__{m}__score"] = val
    np.savez_compressed(out / "retrieval" / "own.npz", **own)
    log.info("own-field pools done (22 fields)")

    for script in ("bn", "rom"):
        fields = [f"{v}:{script}" for v in BANGLA]
        d_text = sum((table.loc[is_train, f].tolist() for f in fields), [])
        d_emb = np.concatenate([emb[f][is_train] for f in fields])
        d_serial = np.concatenate([serials[is_train]] * len(fields))
        d_group = np.concatenate([groups[is_train]] * len(fields))
        d_var = np.repeat(np.arange(len(fields)), is_train.sum())
        r = search(sum((table[f].tolist() for f in fields), []), np.concatenate([emb[f] for f in fields]),
                   np.concatenate([groups] * len(fields)), CharBM25(**bm).fit(d_text), d_emb, d_group, **kw)
        pool = {"q_serial": np.concatenate([serials] * len(fields)),
                "q_variety": np.repeat(np.arange(len(fields)), len(serials)), "variety_codes": np.array(BANGLA)}
        for m, (idx, val) in r.items():
            ok = idx >= 0
            pool[f"{m}__serial"] = np.where(ok, d_serial[np.maximum(idx, 0)], -1)
            pool[f"{m}__variety"] = np.where(ok, d_var[np.maximum(idx, 0)], -1)
            pool[f"{m}__score"] = val
        np.savez_compressed(out / "retrieval" / f"all_{script}.npz", **pool)
        log.info("all_%s pool done (%d queries x %d documents)", script, len(fields) * len(serials), len(d_text))

    leak = 0
    for f in FIELDS:
        nb = own[f"{f.replace(':', '__')}__hybrid__serial"][is_train]
        leak += int((table["group"].reindex(nb.ravel()).values.reshape(nb.shape) == groups[is_train][:, None]).sum())
    if leak:
        raise AssertionError(f"{leak} retrieved neighbours share the query's near-duplicate group")
    log.info("leakage check passed")
    return {"n_distinct_texts": len(texts), "mock": args.mock, "leakage_check": "passed"}


def stage_represent(args, out: Path) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, silhouette_score
    from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
    from sklearn.model_selection import GroupKFold
    rc, seed = CONFIG["representation"], CONFIG["seed"]
    res = out / "analysis" / "representation"
    res.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(out / "data" / "aligned.parquet")
    emb = np.load(out / "embeddings" / "labse.npz")
    row = {s: i for i, s in enumerate(emb["serial"])}
    sample = np.sort(np.random.default_rng(seed).choice(table.index.values, size=min(rc["n_serials"], len(table)),
                                                        replace=False))
    idx = np.array([row[s] for s in sample])
    E = {v: emb[f"{v}__bn"][idx].astype(np.float32) for v in BANGLA}
    raw, y_raw, g_raw = np.concatenate([E[v] for v in BANGLA]), np.repeat(BANGLA, len(sample)), np.tile(sample, len(BANGLA))
    resid = np.concatenate([E[d] - E["std"] for d in DIALECTS])
    y_res, g_res = np.repeat(DIALECTS, len(sample)), np.tile(sample, len(DIALECTS))

    sil = [{"space": "raw", "classes": 10, "silhouette": round(float(silhouette_score(raw, y_raw, metric="cosine")), 4)},
           {"space": "residual", "classes": 9, "silhouette": round(float(silhouette_score(resid, y_res, metric="cosine")), 4)}]
    pd.DataFrame(sil).to_csv(res / "silhouette.csv", index=False)

    Z = np.concatenate([E[v][:150] for v in BANGLA])
    gamma = 1.0 / (2 * np.median(euclidean_distances(Z, Z)) ** 2)
    M = pd.DataFrame(0.0, index=BANGLA, columns=BANGLA)
    for i, a in enumerate(BANGLA):
        for b in BANGLA[i + 1:]:
            M.loc[a, b] = M.loc[b, a] = (rbf_kernel(E[a], E[a], gamma=gamma).mean() + rbf_kernel(E[b], E[b], gamma=gamma).mean()
                                         - 2 * rbf_kernel(E[a], E[b], gamma=gamma).mean())
    M.round(4).to_csv(res / "mmd.csv")

    texts = np.concatenate([table.loc[sample, f"{v}:bn"].values for v in BANGLA])
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                            lowercase=False).fit_transform(texts)
    probes = []
    for name, X, y, g in (("LaBSE raw (10-way)", raw, y_raw, g_raw), ("LaBSE residual (9-way)", resid, y_res, g_res),
                          ("char 2-5-gram TF-IDF (10-way)", tfidf, y_raw, g_raw)):
        f1s = []
        for tr, te in GroupKFold(n_splits=rc["probe_folds"]).split(X, y, g):
            clf = LogisticRegression(max_iter=3000, random_state=seed).fit(X[tr], y[tr])
            f1s.append(f1_score(y[te], clf.predict(X[te]), average="macro"))
        probes.append({"representation": name, "macro_f1_mean": round(float(np.mean(f1s)), 4),
                       "macro_f1_sd": round(float(np.std(f1s)), 4)})
        log.info("probe %-32s macro-F1 %.3f", name, np.mean(f1s))
    pd.DataFrame(probes).to_csv(res / "probes.csv", index=False)
    np.savez_compressed(res / "sample.npz", serial=sample, raw=raw.astype(np.float16), raw_label=y_raw,
                        residual=resid.astype(np.float16), residual_label=y_res)
    return {"silhouette": sil, "probes": probes}


def word_vectors(sents, tok, model, layer, device, max_length, mock):
    import numpy as np
    if mock:
        return [np.array([np.random.default_rng(int(hashlib.md5(w.encode()).hexdigest()[:8], 16)).standard_normal(16)
                          for w in ws], dtype=np.float32).reshape(len(ws), 16) for ws in sents]
    import torch
    enc = tok(sents, is_split_into_words=True, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    with torch.no_grad():
        hs = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states[layer].float().cpu().numpy()
    out = []
    for b, ws in enumerate(sents):
        vec, cnt = np.zeros((len(ws), hs.shape[-1]), dtype=np.float32), np.zeros(len(ws), dtype=np.float32)
        for t, w in enumerate(enc.word_ids(b)):
            if w is not None:
                vec[w] += hs[b, t]
                cnt[w] += 1
        vec[cnt > 0] /= cnt[cnt > 0, None]
        out.append(vec)
    return out


def mutual_argmax(a, b):
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return []
    sim = (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)) @ (b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)).T
    fwd, bwd = sim.argmax(axis=1), sim.argmax(axis=0)
    return [(i, int(j)) for i, j in enumerate(fwd) if bwd[j] == i]


def stage_lexicon(args, out: Path) -> dict:
    import pandas as pd
    lc = CONFIG["lexicon"]
    res = out / "lexicon"
    res.mkdir(parents=True, exist_ok=True)
    tok = model = None
    device = "cpu"
    if not args.mock:
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(lc["encoder"])
        model = AutoModel.from_pretrained(lc["encoder"], output_hidden_states=True).to(device).eval()
        if device.startswith("cuda"):
            model.half()
    table = pd.read_parquet(out / "data" / "aligned.parquet")
    train = table[table["split"] == "train"]
    rows = []
    for d in DIALECTS:
        pair_counts, dial_freq, std_freq = Counter(), Counter(), Counter()
        src, tgt = [tokens(s) for s in train[f"{d}:bn"]], [tokens(s) for s in train["std:bn"]]
        for i in range(0, len(src), lc["batch_size"]):
            bs, bt = src[i:i + lc["batch_size"]], tgt[i:i + lc["batch_size"]]
            va = word_vectors(bs, tok, model, lc["layer"], device, lc["max_length"], args.mock)
            vb = word_vectors(bt, tok, model, lc["layer"], device, lc["max_length"], args.mock)
            for ws, wt, a, b in zip(bs, bt, va, vb):
                dial_freq.update(ws)
                std_freq.update(wt)
                for i_s, i_t in mutual_argmax(a, b):
                    pair_counts[(ws[i_s], wt[i_t])] += 1
        rows += [{"dialect": d, "dial_word": a, "std_word": b, "count": n, "dial_freq": dial_freq[a], "std_freq": std_freq[b]}
                 for (a, b), n in pair_counts.items()]
        log.info("lexicon %s: %d aligned word-pair types", d, len(pair_counts))
    counts = pd.DataFrame(rows)
    counts.to_parquet(res / "alignment_counts.parquet", index=False)

    c = counts[counts["dial_word"] != counts["std_word"]].copy()
    c["p_std_given_dial"] = (c["count"] / counts.groupby(["dialect", "dial_word"])["count"].transform("sum").loc[c.index]).round(3)
    c["p_dial_given_std"] = (c["count"] / counts.groupby(["dialect", "std_word"])["count"].transform("sum").loc[c.index]).round(3)
    lex, sizes = {"d2s": {}, "s2d": {}}, {}
    for d in DIALECTS:
        cd = c[(c["dialect"] == d) & (c["count"] >= lc["default_min_count"])]
        fwd = cd[cd["p_std_given_dial"] >= lc["default_min_prob"]].sort_values("count", ascending=False).drop_duplicates("dial_word")
        bwd = cd[cd["p_dial_given_std"] >= lc["default_min_prob"]].sort_values("count", ascending=False).drop_duplicates("std_word")
        lex["d2s"][d] = {r.dial_word: [r.std_word, int(r.count), float(r.p_std_given_dial)] for r in fwd.itertuples()}
        lex["s2d"][d] = {r.std_word: [r.dial_word, int(r.count), float(r.p_dial_given_std)] for r in bwd.itertuples()}
        sizes[d] = {"d2s": len(lex["d2s"][d]), "s2d": len(lex["s2d"][d])}
    (res / "lexicon_default.json").write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    log.info("lexicon sizes: %s", sizes)
    return {"lexicon_sizes": sizes, "mock": args.mock}


def main() -> None:
    p = argparse.ArgumentParser(description="BUC step 02: data preparation and shared artifacts")
    p.add_argument("--release-dir", help="folder BUC_release (default: found under /kaggle/input)")
    p.add_argument("--output-dir", default="/kaggle/working/outputs")
    p.add_argument("--requirements", help="requirements.txt (default: found under /kaggle/input)")
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    p.add_argument("--no-install", action="store_true", help="do not pip-install the pinned requirements")
    p.add_argument("--allow-cpu", action="store_true", help="run GPU stages on CPU (slow; local tests only)")
    p.add_argument("--mock", action="store_true", help="random embeddings, no model downloads (pipeline test only)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)

    env = prepare_environment(args)
    if not args.release_dir:
        hit = find_one("/kaggle/input/**/BUC_release/splits_by_serial.csv", "BUC_release folder")
        if hit is None:
            raise SystemExit("BUC_release not found under /kaggle/input: attach the release dataset")
        args.release_dir = str(hit.parent)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    env["gpu"] = gpu
    log.info("release: %s | output: %s | GPU: %s", args.release_dir, out, gpu.get("gpus"))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    if {"retrieval", "lexicon"} & set(stages) and not gpu.get("cuda_available") and not (args.mock or args.allow_cpu):
        raise SystemExit("stages 'retrieval' and 'lexicon' need a GPU: set Accelerator to 'GPU T4 x2'")
    if gpu.get("gpus") and not all("T4" in g for g in gpu["gpus"]):
        log.warning("GPUs are %s, not T4: results and timings may differ from the reported setup", gpu["gpus"])

    if "prepare" not in stages and not (out / "data" / "aligned.parquet").exists():
        prev = find_one("/kaggle/input/**/data/aligned.parquet", "earlier step-02 output")
        if prev is None:
            raise SystemExit("stage 'prepare' not requested and no earlier step-02 output attached")
        import shutil
        shutil.copytree(prev.parents[1], out, dirs_exist_ok=True)
        log.info("resumed from earlier output %s", prev.parents[1])

    fns = {"prepare": stage_prepare, "analysis": stage_analysis, "retrieval": stage_retrieval,
           "represent": stage_represent, "lexicon": stage_lexicon}
    for stage in STAGES:
        if stage in stages:
            with Manifest(out, stage, args, env) as m:
                m.record(**fns[stage](args, out))
    (out / "STEP1_DONE.txt").write_text(f"completed {datetime.now(timezone.utc).isoformat()} stages={stages}\n")
    log.info("step 02 finished. Outputs in %s.", out)


if __name__ == "__main__":
    main()

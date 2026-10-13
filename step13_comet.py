#!/usr/bin/env python3
"""COMET (Unbabel/wmt22-comet-da) for every translation prediction set.

COMET needs numpy < 2, so the script creates its own Python 3.12 environment (uv, requirements-comet.txt)
and re-runs itself there. Output: outputs/step8b/comet_directions.csv, merged by step14_evaluate.py.

Usage:
    python step13_comet.py
    python step13_comet.py --systems '^ablate-base-titullm-3b-' --splits test_full
"""
from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

VENV = Path("/tmp/comet312")
COMET_MODEL = "Unbabel/wmt22-comet-da"
TORCH = "torch==2.10.0"
log = logging.getLogger("buc")


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def bootstrap_env() -> None:
    """Create the isolated Python-3.12 COMET environment (once per session) and re-run this file inside it."""
    req = find_one("/kaggle/input/**/requirements-comet.txt") or Path(__file__).with_name("requirements-comet.txt")
    py = VENV / "bin" / "python"
    if not (VENV / ".ready").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv = [sys.executable, "-m", "uv"]
        subprocess.run(uv + ["python", "install", "3.12"], check=True)
        subprocess.run(uv + ["venv", "-q", "-p", "3.12", str(VENV)], check=True)
        log.info("installing the COMET environment (Python 3.12, %s, %s) ...", TORCH, req.name)
        subprocess.run(uv + ["pip", "install", "-q", "--python", str(py), "-r", str(req), TORCH], check=True)
        (VENV / ".ready").write_text("ok")
    os.execv(str(py), [str(py), __file__, "--inner", *sys.argv[1:]])


def import_step8():
    """Imports step14_evaluate.py from the folder of this script."""
    import importlib
    return importlib.import_module("step14_evaluate")


def inner() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    s8 = import_step8()
    import pandas as pd
    import torch
    from comet import download_model, load_from_checkpoint
    root = find_one("/kaggle/input/**/data/aligned.parquet")
    if root is None:
        raise SystemExit("attach the output of step02_prepare_data.py")
    items = s8.Items(root.parents[1])
    import argparse, re
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--systems", help="regular expression: score only matching system names")
    ap.add_argument("--splits", help="comma-separated splits (e.g. test_full,test_prompt_sub)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip system/split pairs already in an attached step8b/comet_directions*.csv (earlier runs)")
    args = ap.parse_args()
    found = {k: v for k, v in s8.discover(["/kaggle/working/outputs", "/kaggle/input"]).items() if v["kind"] != "clf"}
    if args.systems:
        found = {k: v for k, v in found.items() if re.search(args.systems, k[0])}
    if args.splits:
        found = {k: v for k, v in found.items() if k[1] in args.splits.split(",")}
    if args.skip_done:
        done = set()
        for f in glob.glob("/kaggle/input/**/step8b/comet_directions*.csv", recursive=True):
            d = pd.read_csv(f)
            done |= set(zip(d["system"], d["split"]))
        log.info("already scored in attached outputs: %d system/split pairs", len(done))
        found = {k: v for k, v in found.items() if k not in done}
    log.info("translation prediction sets: %d", len(found))
    if not found:
        n4d = len(glob.glob("/kaggle/input/**/step4d/*/*/seed*/pred_test_full.jsonl.gz", recursive=True))
        log.warning("nothing to score. step 11 translation predictions under /kaggle/input: %d (0 = the collected "
                    "output is not attached, or it holds no step4d/<model>/<cond>/seed13/ folders); "
                    "--systems %r, --splits %r", n4d, args.systems, args.splits)
    for k in sorted(found):
        log.info("  %s / %s", *k)
    model = load_from_checkpoint(download_model(COMET_MODEL))
    gpus = 1 if torch.cuda.is_available() else 0
    rows = []
    out = Path("/kaggle/working/outputs/step8b")
    out.mkdir(parents=True, exist_ok=True)
    for (system, split), meta in sorted(found.items()):
        recs = s8.load_records(meta, split, items)["mt"]
        if not recs:
            continue
        t0 = time.time()
        data = [{"src": r["src"], "mt": r["hyp"] or ".", "ref": r["ref"]} for r in recs]
        scores = model.predict(data, batch_size=64, gpus=gpus, progress_bar=False).scores
        by_dir = {}
        for r, sc in zip(recs, scores):
            by_dir.setdefault((r["tier"], r["src_field"], r["tgt_field"]), []).append(sc)
        for (tier, src, tgt), v in by_dir.items():
            rows.append({"system": system, "split": split, "tier": tier, "src": src, "tgt": tgt,
                         "COMET": round(100 * sum(v) / len(v), 3), "n": len(v)})
        log.info("%s / %s: %d segments (%.0f s)", system, split, len(recs), time.time() - t0)
        pd.DataFrame(rows).to_csv(out / "comet_directions.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"model": COMET_MODEL, "torch": torch.__version__,
                                                   "python": sys.version.split()[0], "systems": sorted({r["system"] for r in rows})}, indent=1))
    log.info("COMET done: %d system/split pairs", len({(r['system'], r['split']) for r in rows}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    if "--inner" in sys.argv:
        inner()
    else:
        bootstrap_env()

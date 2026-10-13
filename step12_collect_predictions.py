#!/usr/bin/env python3
"""Copies the prediction files needed for evaluation from many run outputs into one small folder.

Also writes merged tables of the unseen-dialect runs to <output>/unseen_summary/.

Usage:
    python step12_collect_predictions.py --input <folder with run outputs> --output collected
"""
import argparse
import glob
import shutil
from pathlib import Path

ANCHORS = ("runs", "step4a", "step4b", "step4c", "step4d", "step5", "step8b")
PATTERNS = [
    "**/runs/*/pred_*/rank*.jsonl",
    "**/runs/ablate-*/abl_*/*.jsonl",
    "**/runs/ablate-*/abl_*/metrics.json",
    "**/step4d/*/*/seed*/pred_test_full.jsonl.gz",
    "**/step4d/*/*/seed*/result.json",
    "**/step4c/*/final/seed*/pred_*.jsonl.gz",
    "**/step4a/**/final/**/pred_*_primary.jsonl.gz",
    "**/step4b/**/final/**/pred_*_primary.jsonl.gz",
    "**/step4a/svm/final/model.joblib",
    "**/step4b/muril-large/final/seed13/model/*",
    "**/step5/*/*/*.jsonl",
    "**/step5/*/*/*.json",
    "**/step8b/comet_directions.csv",
    "**/step4d/id/*",
    "**/step4d/id_encoder/*/*/seed*/result.json",
    "**/step4d/id_encoder/*/*/seed*/pred_test_full.jsonl.gz",
    "**/step4d/external/*",
    "**/step4d/*/*/seed*/external_seq2seq.csv",
    "**/runs/external-*/external/*.csv",
]

DIALECTS = ["nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]


def _write(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def unseen_summary(out: Path) -> None:
    """Merged tables from the collected step4d result files (all notebooks together)."""
    import csv
    import json
    res = out / "unseen_summary"
    check, by_dialect, summary, id_rows, id_summary = [], [], [], [], []
    mt = {}
    for f in sorted(out.glob("step4d/*/*/seed*/result.json")):
        model, cond = f.parents[2].name, f.parents[1].name
        if model in ("id", "id_encoder", "external"):
            continue
        r = json.loads(f.read_text())
        if r.get("status") != "ok":
            check.append({"task": "mt", "model": model, "condition": cond, "status": r.get("status"), "error": r.get("error")})
            continue
        per = r.get("evaluation", {}).get("test_full", {}).get("per_direction", {})
        mt[(model, cond)] = {d: (per.get(f"{d}:bn>std:bn", {}).get("chrF++"), per.get(f"std:bn>{d}:bn", {}).get("chrF++"))
                             for d in DIALECTS}
        check.append({"task": "mt", "model": model, "condition": cond, "status": "ok", "code": r.get("code"),
                      "max_epochs": r.get("max_epochs"), "best_epoch": r.get("best_epoch"),
                      "train_minutes": r.get("train_minutes"), "train_dialects": "+".join(r.get("train_dialects") or [])})
    for (model, cond), vals in sorted(mt.items()):
        ref = mt.get((model, "full"), {})
        seen = set() if cond == "std-only" else ({cond[4:]} if cond.startswith("one-") else
                                                 set(DIALECTS) - {cond[5:]} if cond.startswith("lodo-") else set(DIALECTS))
        for d in DIALECTS:
            d2s, s2d = vals[d]
            rd2s, rs2d = ref.get(d, (None, None))
            by_dialect.append({"model": model, "condition": cond, "dialect": d, "seen_in_training": d in seen,
                               "d2s_chrfpp": d2s, "d2s_full_reference": rd2s,
                               "d2s_drop": round(rd2s - d2s, 2) if None not in (d2s, rd2s) else None,
                               "s2d_chrfpp": s2d, "s2d_full_reference": rs2d,
                               "s2d_drop": round(rs2d - s2d, 2) if None not in (s2d, rs2d) else None})
        uns = [d for d in DIALECTS if d not in seen]
        row = {"model": model, "condition": cond, "n_unseen": len(uns),
               "d2s_unseen_mean": _mean([vals[d][0] for d in uns]),
               "d2s_unseen_full_reference": _mean([ref.get(d, (None, None))[0] for d in uns]) if ref else None,
               "d2s_seen_mean": _mean([vals[d][0] for d in seen]),
               "s2d_unseen_mean": _mean([vals[d][1] for d in uns]),
               "s2d_seen_mean": _mean([vals[d][1] for d in seen])}
        if row["d2s_unseen_full_reference"] is not None and row["d2s_unseen_mean"] is not None:
            row["d2s_unseen_drop"] = round(row["d2s_unseen_full_reference"] - row["d2s_unseen_mean"], 2)
        summary.append(row)
    for f in sorted(out.glob("step4d/id/summary_id.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                id_rows.append(r)
    for f in sorted(out.glob("step4d/id/result_*.json")):
        r = json.loads(f.read_text())
        id_summary.append({"model": "svm", "condition": r.get("condition"), "seen_classes": len(r.get("seen_labels", [])),
                           "seen_macro_f1_bn": r.get("seen_macro_f1_bn"), "seen_macro_f1_rom": r.get("seen_macro_f1_rom")})
        check.append({"task": "id", "model": "svm", "condition": r.get("condition"), "status": "ok",
                      "code": "sklearn " + str(r.get("sklearn", "?"))})
    for f in sorted(out.glob("step4d/id_encoder/*/*/seed*/result.json")):
        r = json.loads(f.read_text())
        model, cond = f.parents[2].name, f.parents[1].name
        if r.get("status") != "ok":
            check.append({"task": "id", "model": model, "condition": cond, "status": r.get("status"), "error": r.get("error")})
            continue
        id_rows += [{k: v for k, v in x.items()} for x in r.get("summary_rows", [])]
        id_summary.append({"model": model, "condition": cond, "seen_classes": len(r.get("seen_labels", [])),
                           "seen_macro_f1_bn": r.get("seen_macro_f1_bn"), "seen_macro_f1_rom": r.get("seen_macro_f1_rom")})
        check.append({"task": "id", "model": model, "condition": cond, "status": "ok", "code": r.get("code"),
                      "max_epochs": r.get("max_epochs"), "train_minutes": r.get("train_minutes")})
    _write(res / "unseen_mt_by_dialect.csv", by_dialect)
    _write(res / "unseen_mt_summary.csv", summary)
    _write(res / "unseen_id.csv", id_rows)
    _write(res / "unseen_id_summary.csv", sorted(id_summary, key=lambda r: (r["model"], str(r["condition"]))))
    _write(res / "unseen_check.csv", check)
    if check:
        print(f"\nunseen-dialect runs found: {sum(c['status'] == 'ok' for c in check)} ok, "
              f"{sum(c['status'] != 'ok' for c in check)} not ok -> {res}")
        per = {}
        for c in check:
            per.setdefault((c["task"], c["model"]), []).append(c)
        for (task, model), cs in sorted(per.items()):
            codes = sorted({str(c.get("code")) for c in cs})
            caps = sorted({str(c.get("max_epochs")) for c in cs if c.get("max_epochs") is not None})
            print(f"  {task:3s} {model:18s} {sum(c['status'] == 'ok' for c in cs):2d} runs  code {codes}  epoch cap {caps}")


def relative(path: Path) -> Path | None:
    parts = path.parts
    for i, p in enumerate(parts):
        if p in ANCHORS:
            return Path(*parts[i:])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--output", default="/kaggle/working/collected")
    args = ap.parse_args()
    out = Path(args.output)
    copied, skipped, size = {}, 0, 0
    for pat in PATTERNS:
        for f in glob.glob(f"{args.input}/{pat}", recursive=True):
            src = Path(f)
            if not src.is_file() or "pred_dev_select" in f:
                continue
            rel = relative(src)
            if rel is None:
                continue
            dst = out / rel
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            group = rel.parts[0] if rel.parts[0] != "runs" else f"runs/{rel.parts[1]}"
            copied[group] = copied.get(group, 0) + 1
            size += dst.stat().st_size
    for g in sorted(copied):
        print(f"{copied[g]:6d} files  {g}")
    print(f"copied {sum(copied.values())} files ({size / 2**20:.1f} MB) to {out}; {skipped} duplicates skipped")
    if not copied:
        print("nothing found: are the notebook outputs attached?")
    unseen_summary(out)


if __name__ == "__main__":
    main()

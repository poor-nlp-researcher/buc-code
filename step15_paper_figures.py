#!/usr/bin/env python3
"""Figures of the paper drawn from the outputs of the earlier steps (ACL column width).

    fig_distance.pdf    aligned distance between all varieties (step01 table F8_pairwise_aligned_distance.csv)
    fig_perdialect.pdf  Tier-1 chrF++ per dialect: range and mean of the eight fine-tuned systems and copying
                        the input (step14 mt_directions.csv; several files may be given, later ones take priority)
    fig_fertility.pdf   tokenizer fertility per variety (step03 fertility_long.csv)

Usage:
    python step15_paper_figures.py --distance results/analysis/tables/F8_pairwise_aligned_distance.csv \\
        --fertility outputs/fertility/fertility_long.csv --mt-directions outputs/step8/mt_directions.csv --out figures
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as fm

COLW = 3.03
ABB = {"Standard": "Std", "Noakhali": "Nkh", "Chittagong": "Ctg", "Sylhet": "Syl", "Barishal": "Bar",
       "Sirajganj": "Srj", "Pabna": "Pab", "Shariatpur": "Shr", "Old Dhaka": "ODh", "Khulna": "Khl"}
CODE = {"nkh": "Nkh", "ctg": "Ctg", "syl": "Syl", "bar": "Bar", "srj": "Srj", "pab": "Pab", "shr": "Shr",
        "odh": "ODh", "khl": "Khl", "std": "Std"}
FINE_TUNED = ["seq2seq/banglat5", "seq2seq/mbart50", "seq2seq/nllb-1.3b", "QLoRA/gemma4-e4b",
              "QLoRA-base/llama31-8b", "QLoRA-base/ministral3-8b", "QLoRA-base/qwen3-8b-lr1", "QLoRA-base/titullm-3b"]


def setup_style():
    serif = "serif"
    for f in fm.findSystemFonts():
        if "LiberationSerif-Regular" in f:
            fm.fontManager.addfont(f)
            serif = "Liberation Serif"
    plt.rcParams.update({
        "font.family": serif, "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "savefig.dpi": 300})


def fig_distance(path, out):
    d = pd.read_csv(path, index_col=0)
    order = ["Standard", "Pabna", "Shariatpur", "Khulna", "Barishal", "Sirajganj", "Sylhet", "Noakhali",
             "Chittagong", "Old Dhaka"]
    m = d.loc[order, order].values
    fig, ax = plt.subplots(figsize=(COLW, 2.55))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.array(m, mask=np.eye(len(order), dtype=bool)), cmap=cmap, vmin=30, vmax=82)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if m[i, j] > 62 else "black")
    labels = [ABB[o] for o in order]
    ax.set_xticks(range(len(order)), labels)
    ax.set_yticks(range(len(order)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("aligned distance (100 $-$ chrF++)", fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_distance.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def per_dialect_scores(paths):
    frames = [pd.read_csv(p) for p in paths]
    d = pd.concat(frames).drop_duplicates(["system", "split", "tier", "src", "tgt"], keep="last")
    d = d[(d.split == "test_full") & (d.tier == "tier1") & d.family.isin(FINE_TUNED)]
    missing = sorted(set(FINE_TUNED) - set(d.family))
    if missing:
        raise SystemExit(f"systems missing in mt_directions: {missing}")
    rows = []
    for (fam, src, tgt), g in d.groupby(["family", "src", "tgt"]):
        dial = tgt if src == "std:bn" else src
        rows.append({"family": fam, "dir": "s2d" if src == "std:bn" else "d2s", "dialect": dial.split(":")[0],
                     "chrf": g["chrF++"].mean(), "copy_input": g["copy_chrF++"].mean()})
    return pd.DataFrame(rows)


def fig_perdialect(paths, out):
    t = per_dialect_scores(paths)
    order = ["odh", "nkh", "syl", "ctg", "bar", "srj", "khl", "shr", "pab"]
    fig, axes = plt.subplots(2, 1, figsize=(COLW, 3.2), sharex=True)
    for ax, direction, title in [(axes[0], "d2s", "Dialect $\\rightarrow$ Standard"),
                                 (axes[1], "s2d", "Standard $\\rightarrow$ Dialect")]:
        s = t[t.dir == direction]
        lo = [s[s.dialect == d].chrf.min() for d in order]
        hi = [s[s.dialect == d].chrf.max() for d in order]
        mean = [s[s.dialect == d].chrf.mean() for d in order]
        copy = [s[s.dialect == d].copy_input.iloc[0] for d in order]
        x = np.arange(len(order))
        ax.vlines(x, lo, hi, color="#9aa0a6", lw=5, label="Range of the 8 fine-tuned systems")
        ax.plot(x, mean, "o", color="#1f3b63", ms=4.5, label="Mean of the 8 fine-tuned systems")
        ax.plot(x, copy, "x", color="#c0282d", ms=4.5, mew=1.1, ls="none", label="Copy input")
        ax.set_ylim(20, 95)
        ax.set_yticks([30, 50, 70, 90])
        ax.set_ylabel("chrF++")
        ax.text(0.01, 0.95, title, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.grid(axis="y", lw=0.3, color="0.88")
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[1].set_xticks(np.arange(len(order)), [CODE[d] for d in order])
    axes[1].set_xlabel("dialect (far $\\rightarrow$ near Standard Bangla)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.84))
    fig.savefig(out / "fig_perdialect.pdf")
    plt.close(fig)


def fig_fertility(path, out):
    f = pd.read_csv(path)
    f = f[(f.script == "bn") & (f.tokenizer != "IndicTrans2-1B") & (f.tokenizer != "mBART-50")]
    vorder = ["std", "pab", "shr", "khl", "bar", "srj", "syl", "ctg", "odh", "nkh"]
    piv = f.pivot_table(index="tokenizer", columns="variety", values="fertility")[vorder]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    names = {"XLM-R-large": "XLM-R / mBART-50", "NLLB-200 (1.3B)": "NLLB-200"}
    fig, ax = plt.subplots(figsize=(COLW, 2.9))
    cm = plt.get_cmap("tab10")
    for i, (tok, row) in enumerate(piv.iterrows()):
        ax.plot(range(len(vorder)), row.values, marker="o", ms=2.2, lw=0.8, color=cm(i % 10),
                label=names.get(tok, tok))
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5, 8])
    ax.set_yticklabels(["1", "1.5", "2", "3", "5", "8"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(vorder)), [CODE[v] for v in vorder])
    ax.set_ylabel("tokens per word (log scale)")
    ax.grid(lw=0.3, color="0.88")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=3, fontsize=6.2,
              handlelength=1.5, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_fertility.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="BUC step 15: figures of the paper")
    p.add_argument("--distance", help="F8_pairwise_aligned_distance.csv from step01")
    p.add_argument("--fertility", help="fertility_long.csv from step03")
    p.add_argument("--mt-directions", nargs="+", help="mt_directions.csv from step14 (later files take priority)")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    if args.distance:
        fig_distance(args.distance, out)
    if args.mt_directions:
        fig_perdialect(args.mt_directions, out)
    if args.fertility:
        fig_fertility(args.fertility, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()

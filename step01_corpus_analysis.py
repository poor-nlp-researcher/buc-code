"""Corpus construction and analysis of the Bangla Upovasha Corpus (BUC).

Reads the ten annotated CSV files (Standard Bangla + nine dialects), checks their integrity, creates the
serial-level train/dev/test split, exports the release folder (BUC_release/) and writes the corpus statistics,
figures and tables (lengths, vocabulary, aligned distances, n-gram similarity, MMD, code-mixing, fertility).

Usage:
    python step01_corpus_analysis.py
Paths are set at the top of the file (DATA, PTH, EXPORT).
"""
from __future__ import annotations

import sys

if sys.version_info[:2] != (3, 12):
    sys.exit(f"Python 3.12.* required (Kaggle uses 3.12.13); this is {sys.version.split()[0]}. "
             "See README.md.")

import hashlib
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex
import seaborn as sns
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import CHRF
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

DATA = Path(
    "data/raw")
PTH = "results/analysis"
OUT_F, OUT_T = Path(f"{PTH}/figures"), Path(f"{PTH}/tables")
OUT_F.mkdir(parents=True, exist_ok=True)
OUT_T.mkdir(parents=True, exist_ok=True)

EXPORT = Path("BUC_release")
EXPORT.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = EXPORT / "splits_by_serial.csv"
LEGACY_SPLIT_FILE = OUT_T / "splits_by_serial.csv"
SEED = 13
rng = np.random.default_rng(SEED)

FILES = {
    "Noakhali": "02.Noakhali_Dialect.csv", "Chittagong": "03.Chittagong_Dialect.csv", "Sylhet": "04.Sylhet_Dialect.csv",
    "Barishal": "05.Barishal_Dialect.csv", "Sirajganj": "06.Sirajganj_Dialect.csv", "Pabna": "07.Pabna_Dialect.csv",
    "Shariatpur": "08.Shariatpur_Dialect.csv", "Old Dhaka": "09.Old Dhaka_Dialect.csv", "Khulna": "10.Khulna_Dialect.csv",
}
DIALECTS = list(FILES)
VARIETIES = ["Standard"] + DIALECTS
COORD = {"Noakhali": (22.87, 91.10), "Chittagong": (22.36, 91.78), "Sylhet": (24.90, 91.87),
         "Barishal": (22.70, 90.37), "Sirajganj": (24.45, 89.70), "Pabna": (24.01, 89.24),
         "Shariatpur": (23.24, 90.43), "Old Dhaka": (23.71, 90.40), "Khulna": (22.82, 89.55)}
BN_TOKEN = re.compile(
    r"[\u0980-\u09FF\u200c\u200d]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
chrf = CHRF(word_order=2)


APPLY_ZWJ_POLICY = True
ZWJ_KEEP = "\u09B0\u200D\u09CD\u09AF"
KHANDA_TA_LEGACY = "\u09A4\u09CD\u200D"
SPACE_BEFORE_PUNCT = re.compile(r"\s+([\u0964\u0965,;:?!.])")
INVISIBLE = ("\u200b", "\ufeff", "\u00ad")


def normalize_text(s):
    """NFC; remove invisible junk; ZWJ policy; any whitespace run (tabs, newlines, nbsp, multiple spaces) -> one space; no space before punctuation; strip both ends."""
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFC", str(s))
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    if APPLY_ZWJ_POLICY:
        s = s.replace(KHANDA_TA_LEGACY, "\u09CE")
        s = s.replace(ZWJ_KEEP, "\x00").replace("\u200d", "").replace("\x00", ZWJ_KEEP)
    s = re.sub(r"\s+", " ", s)
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    return s.strip()


clean = normalize_text


def raw_issue_flags(t):
    """What normalisation would fix in one RAW cell (for T5 and the normalisation log)."""
    t = "" if pd.isna(t) else str(t)
    return {
        "leading_or_trailing_space": t != t.strip(),
        "multiple_spaces_or_tabs": bool(re.search(r"\s{2,}|[\t\n\r\u00a0]", t.strip())),
        "space_before_punct": bool(SPACE_BEFORE_PUNCT.search(t)),
        "has_ZWJ": "\u200d" in t,
        "has_ZWNJ": "\u200c" in t,
        "invisible_chars": any(ch in t for ch in INVISIBLE),
        "not_NFC": t != unicodedata.normalize("NFC", t),
        "changed_by_normalization": (t != "" and t != normalize_text(t)),
    }


def tokens(s):
    return BN_TOKEN.findall(s) if isinstance(s, str) else []


def heat(M, name, title, fmt=".2f", cmap="viridis"):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(M.astype(float), annot=True, fmt=fmt, cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_F / f"{name}.png", dpi=300)
    plt.close()
    M.to_csv(OUT_T / f"{name}.csv")


def check_serial_consistency(main):
    """Every dialect file must contain exactly the serials of 01.Main.csv (no gaps, no extras, no duplicates)."""
    ref, rows = set(main.Serial), []
    for d, f in FILES.items():
        s = pd.read_csv(DATA / f, usecols=lambda c: c.strip() == "Serial")
        s = s.iloc[:, 0].dropna().astype(int)
        rows.append(dict(file=f, rows=len(s), duplicated_serials=int(s.duplicated().sum()),
                         missing_vs_main=sorted(ref - set(s))[:20],
                         extra_vs_main=sorted(set(s) - ref)[:20]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / "T5b_serial_consistency.csv", index=False)
    bad = t[(t["duplicated_serials"] > 0) | (t["missing_vs_main"].apply(len) > 0)
            | (t["extra_vs_main"].apply(len) > 0)]
    if len(bad):
        print(bad.to_string(index=False))
        raise ValueError("Serial sets differ between 01.Main.csv and the dialect files (see "
                         "T5b_serial_consistency.csv). Delete/add the same serials in ALL 10 files.")
    print(f"Serial check OK: all 10 files contain the same {len(ref)} serials.")


def fingerprint(s):
    """Short hash of the normalised Standard sentence; detects renumbered serials later."""
    t = normalize_text(s)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] if isinstance(t, str) else ""


def report_category_split(main, split):
    """Print + save the number of serials (= sentences per variety) per category per split."""
    cats = main.set_index("Serial").loc[split.index, "Category"].astype(str)
    tab = pd.crosstab(cats.values, split.values, margins=True, margins_name="Total")
    tab = tab.reindex(columns=["train", "dev", "test", "Total"], fill_value=0)
    tab.index.name, tab.columns.name = "Category", None
    pct = (tab[["train", "dev", "test"]].div(tab["Total"], axis=0) * 100).round(1).add_suffix(" %")
    out = tab.join(pct)
    out.to_csv(OUT_T / "T4_category_by_split.csv")
    print("\nSentences per category per split (same in Main and in every dialect file):")
    print(out.to_string())
    print()


def load():
    main = pd.read_csv(DATA / "01.Main.csv")
    main.columns = main.columns.str.strip()
    main["Serial"] = main["Serial"].astype(int)
    main["Category"] = main["Category"].map(normalize_text)
    if main["Serial"].duplicated().any():
        raise ValueError(f"01.Main.csv has duplicated Serials: "
                         f"{sorted(main.Serial[main.Serial.duplicated()].unique())[:10]}")
    check_serial_consistency(main)
    n = len(main)
    print(f"Corpus: {n} serials -> {9 * n} dialect sentences + {9 * n} dialect Banglish; "
          f"{22 * n} aligned text segments in total")
    print(f"Categories ({main.Category.nunique()}): {sorted(main.Category.unique())}")
    std_main = main.set_index("Serial")["Standard Bangla Sentence"].map(clean)
    rec, mismatch = [], {}
    for r in main.to_dict("records"):
        s = r["Serial"]
        rec += [(s, "Standard", "bn", r["Standard Bangla Sentence"]),
                (s, "Standard", "rom", r["Banglish Sentence"]),
                (s, "English", "en", r["English Sentence"]),
                (s, "CodeMixed", "cm", r["Code Mixed Sentence"])]
    for d, f in FILES.items():
        df = pd.read_csv(DATA / f)
        df.columns = df.columns.str.strip()
        dcol = next(c for c in df.columns if c.endswith("Dialect"))
        rcol = next(c for c in df.columns if c.endswith("Banglish"))
        here = df.set_index(df["Serial"].astype(int))["Standard Bangla Sentence"].map(clean)
        mismatch[d] = int((here != std_main.reindex(here.index)).sum())
        for r in df.to_dict("records"):
            s = int(r["Serial"])
            rec += [(s, d, "bn", r[dcol]), (s, d, "rom", r[rcol])]
    long = pd.DataFrame(rec, columns=["serial", "variety", "script", "raw"])
    n_dup = int(long.duplicated(["serial", "variety", "script"]).sum())
    print(f"duplicate (serial, variety, script) rows dropped: {n_dup}")
    long = long.drop_duplicates(["serial", "variety", "script"])
    long["category"] = long.serial.map(dict(zip(main.Serial, main.Category)))
    long["text"] = long.raw.map(clean)
    pd.Series(mismatch, name="std_mismatch_vs_main").to_csv(OUT_T / "T5a_std_mismatch.csv")
    return main, long


def integrity(long):
    """Issues in the RAW text (before normalisation) + duplicates/missing after normalisation."""
    flags = pd.DataFrame([raw_issue_flags(t) for t in long.raw], index=long.index)
    df = pd.concat([long[["variety", "script", "text"]], flags], axis=1)
    g = df.groupby(["variety", "script"])
    out = pd.DataFrame({
        "n": g.size(),
        "missing": g.text.apply(lambda x: int(x.isna().sum())),
        "exact_dup_within": g.text.apply(lambda x: int(x.dropna().duplicated().sum())),
        "has_latin": g.text.apply(lambda x: int(x.dropna().str.contains(r"[A-Za-z]").sum())),
    })
    out = out.join(g[list(flags.columns)].sum().astype(int))
    out.to_csv(OUT_T / "T5_integrity.csv")


def collisions(long):
    bn = long[(long.script == "bn") & long.variety.isin(VARIETIES)].dropna(subset=["text"])
    piv = bn.pivot(index="serial", columns="variety", values="text")[VARIETIES]
    M = pd.DataFrame({a: [(piv[a] == piv[b]).mean() * 100 for b in VARIETIES] for a in VARIETIES},
                     index=VARIETIES)
    heat(M, "F6_identical_rendering_pct", "% serials with identical renderings", fmt=".1f", cmap="Reds")
    n_labels = bn.groupby("text").variety.nunique()
    amb = bn.text.map(n_labels).gt(1).groupby(bn.variety).mean() * 100
    amb.rename("pct_items_string_shared_with_other_label").to_csv(OUT_T / "T8_ambiguous_strings.csv")


def load_frozen_split(main):
    """Return the frozen split (Series Serial -> split) if it exists, else None."""
    for path in (SPLIT_FILE, LEGACY_SPLIT_FILE):
        if not path.exists():
            continue
        sf = pd.read_csv(path)
        sf["Serial"] = sf["Serial"].astype(int)
        sf = sf.set_index("Serial")
        cur = main.set_index("Serial")["Standard Bangla Sentence"].map(fingerprint)
        new = sorted(set(cur.index) - set(sf.index))
        if new:
            raise ValueError(f"{len(new)} serials are not in {path} (e.g. {new[:5]}). If serials "
                             "were renumbered, delete the split file and re-run (only allowed "
                             "before any experiment has used it).")
        gone = sorted(set(sf.index) - set(cur.index))
        has_fp = "std_fingerprint" in sf.columns
        if has_fp:
            common = cur.index
            changed = common[sf.loc[common, "std_fingerprint"].values != cur.values]
            if len(changed) > 0.10 * len(common):
                raise RuntimeError(f"{len(changed)}/{len(common)} Standard sentences differ from the "
                                   "frozen fingerprints -> serials were probably RENUMBERED.")
            if len(changed):
                print(f"NOTE: {len(changed)} Standard sentences were edited since the split was "
                      "frozen (fine for manual corrections): tables/split_fingerprint_changes.csv")
                pd.Series(changed, name="Serial").to_csv(OUT_T / "split_fingerprint_changes.csv",
                                                         index=False)
        else:
            print("NOTE: split file has no fingerprints yet -> adding them from the current data. "
                  "This assumes serial numbers were NOT renumbered since the split was created.")
        out = pd.DataFrame({"split": sf.loc[cur.index, "split"],
                            "std_fingerprint": sf.loc[cur.index, "std_fingerprint"] if has_fp else cur})
        if gone or not has_fp or path != SPLIT_FILE:
            backup = EXPORT / "splits_by_serial.backup.csv"
            if path == SPLIT_FILE and not backup.exists():
                shutil.copy(SPLIT_FILE, backup)
            out.rename_axis("Serial").to_csv(SPLIT_FILE)
        if gone:
            print(f"{len(gone)} serials were removed from the dataset since the split was frozen "
                  f"and are dropped from it: {gone[:20]} (old file kept as {backup.name})")
        print(f"Using FROZEN split from {path} (assignments unchanged).")
        return out["split"]
    return None


def make_splits(main, near_dup_cos=0.90):
    frozen = load_frozen_split(main)
    if frozen is not None:
        report_category_split(main, frozen)
        return frozen
    std = main.set_index("Serial")["Standard Bangla Sentence"].map(clean).fillna("")
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(std.values)
    dist, idx = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X).kneighbors(X)
    parent = np.arange(len(std))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(std)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1 - d >= near_dup_cos:
                parent[find(i)] = find(j)
    groups = np.array([find(i) for i in range(len(std))])
    cats = main.set_index("Serial").loc[std.index, "Category"].astype(str).values
    fold = np.empty(len(std), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(std.index, cats, groups)):
        fold[te] = k
    split = pd.Series(np.select([fold == 0, fold == 1], ["test", "dev"], "train"),
                      index=std.index, name="split")
    pd.DataFrame({"split": split, "std_fingerprint": std.map(fingerprint)}) \
        .rename_axis("Serial").to_csv(SPLIT_FILE)
    print(f"NEW split written to {SPLIT_FILE} -- keep this file forever; never regenerate it.")
    print("near-duplicate groups merged:", int((pd.Series(groups).value_counts() > 1).sum()))
    report_category_split(main, split)
    return split


def descriptive(long, main):
    vc = main.Category.value_counts()
    vc.to_csv(OUT_T / "T4a_category_counts.csv")
    plt.figure(figsize=(9, 4))
    vc.plot.barh()
    plt.xlabel("# serials (identical for every variety)")
    plt.tight_layout()
    plt.savefig(OUT_F / "F3_category_distribution.png", dpi=300)
    plt.close()

    df = long.dropna(subset=["text"]).copy()
    df["n_words"] = df.text.str.split().str.len()
    df["n_graphemes"] = df.text.map(lambda s: len(regex.findall(r"\X", s.replace(" ", ""))))
    df.groupby(["variety", "script"])[["n_words", "n_graphemes"]] \
      .agg(["min", "max", "mean", "median", "std"]).round(2).to_csv(OUT_T / "T6_length_stats.csv")

    sub = df[(df.script == "bn") & df.variety.isin(VARIETIES)]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.violinplot(data=sub, x="variety", y="n_words", order=VARIETIES, cut=0, ax=ax[0])
    sns.violinplot(data=sub, x="variety", y="n_graphemes", order=VARIETIES, cut=0, ax=ax[1])
    plt.tight_layout()
    plt.savefig(OUT_F / "F4_length_violins.png", dpi=300)
    plt.close()

    piv = sub.pivot(index="serial", columns="variety", values="n_words")
    ratio = piv[DIALECTS].div(piv["Standard"], axis=0).melt(var_name="dialect", value_name="len_ratio")
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=ratio, x="dialect", y="len_ratio")
    plt.axhline(1, ls="--", c="grey")
    plt.tight_layout()
    plt.savefig(OUT_F / "F5_length_ratio.png", dpi=300)
    plt.close()


def vocabulary(long):
    bn = long[(long.script == "bn") & long.variety.isin(VARIETIES)]
    tr, te = bn[bn.split == "train"], bn[bn.split == "test"]
    V = {v: Counter(t for s in g.text for t in tokens(s)) for v, g in tr.groupby("variety")}
    std = set(V["Standard"])
    rows = []
    for v in VARIETIES:
        c, types = V[v], set(V[v])
        others = set().union(*(set(V[u]) for u in VARIETIES if u != v))
        n_tok = sum(c.values())
        test_toks = [t for s in te[te.variety == v].text for t in tokens(s)]
        rows.append(dict(
            variety=v, tokens=n_tok, types=len(types), TTR=len(types) / n_tok,
            OOV_type_vs_std=100 * len(types - std) / len(types),
            OOV_token_vs_std=100 * sum(n for w, n in c.items() if w not in std) / n_tok,
            unique_types_pct=100 * len(types - others) / len(types),
            test_OOV_vs_own_train=100 * np.mean([t not in types for t in test_toks]),
        ))
    pd.DataFrame(rows).round(2).to_csv(OUT_T / "T7_vocabulary.csv", index=False)
    J = pd.DataFrame({a: [len(set(V[a]) & set(V[b])) / len(set(V[a]) | set(V[b])) for b in VARIETIES]
                      for a in VARIETIES}, index=VARIETIES)
    heat(J, "F10_jaccard_word_types", "Word-type Jaccard (train; under-separates close dialects)")


def haversine(p, q):
    la1, lo1, la2, lo2 = map(np.radians, (*p, *q))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def mantel(A, B, n_perm=9999):
    iu = np.triu_indices_from(A, 1)
    r0 = np.corrcoef(A[iu], B[iu])[0, 1]
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(len(A))
        hits += np.corrcoef(A[np.ix_(p, p)][iu], B[iu])[0, 1] >= r0
    return r0, (hits + 1) / (n_perm + 1)


def aligned_divergence(long):
    piv = long[long.script == "bn"].pivot(index="serial", columns="variety", values="text")
    rows = []
    for d in DIALECTS:
        for s, a, b in zip(piv.index, piv[d], piv["Standard"]):
            if isinstance(a, str) and isinstance(b, str):
                rows.append((s, d, chrf.sentence_score(a, [b]).score,
                             Levenshtein.normalized_distance(a, b), a == b))
    div = pd.DataFrame(rows, columns=["serial", "dialect", "chrfpp_vs_std", "norm_edit_dist", "identical"])
    div.groupby("dialect").agg(chrF_mean=("chrfpp_vs_std", "mean"), chrF_sd=("chrfpp_vs_std", "std"),
                               ned_mean=("norm_edit_dist", "mean"), pct_identical=("identical", "mean")) \
       .sort_values("chrF_mean").round(3).to_csv(OUT_T / "T9_divergence_from_standard.csv")
    order = div.groupby("dialect").chrfpp_vs_std.median().sort_values().index
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=div, x="dialect", y="chrfpp_vs_std", order=order, cut=0)
    plt.ylabel("chrF++ vs. own Standard source")
    plt.tight_layout()
    plt.savefig(OUT_F / "F7_divergence_violin.png", dpi=300)
    plt.close()

    D = pd.DataFrame(0.0, index=VARIETIES, columns=VARIETIES)
    for i, a in enumerate(VARIETIES):
        for b in VARIETIES[i + 1:]:
            ok = piv[a].notna() & piv[b].notna()
            xa, xb = piv[a][ok].tolist(), piv[b][ok].tolist()
            ab = np.mean([chrf.sentence_score(x, [y]).score for x, y in zip(xa, xb)])
            ba = np.mean([chrf.sentence_score(y, [x]).score for x, y in zip(xa, xb)])
            D.loc[a, b] = D.loc[b, a] = 100 - (ab + ba) / 2
    heat(D, "F8_pairwise_aligned_distance", "Aligned distance (100 - chrF++)", fmt=".1f", cmap="magma_r")
    plt.figure(figsize=(8, 4))
    dendrogram(linkage(squareform(D.values, checks=False), "average"), labels=VARIETIES)
    plt.tight_layout()
    plt.savefig(OUT_F / "F9_dendrogram.png", dpi=300)
    plt.close()
    G = np.array([[haversine(COORD[a], COORD[b]) for b in DIALECTS] for a in DIALECTS])
    r, p = mantel(D.loc[DIALECTS, DIALECTS].values, G)
    print(f"Mantel r={r:.3f}, p={p:.4f}  (9 dialects -> low power; report as exploratory)")


def log_odds_dirichlet(fg: Counter, bg: Counter, prior: Counter, a0: float = 1000.0):
    """Monroe, Colaresi & Quinn (2008): log-odds ratio with informative Dirichlet prior -> z-scores."""
    n_fg, n_bg, n_pr = sum(fg.values()), sum(bg.values()), sum(prior.values())
    z = {}
    for w in fg:
        aw = a0 * prior[w] / n_pr
        l1 = np.log((fg[w] + aw) / (n_fg + a0 - fg[w] - aw))
        l2 = np.log((bg[w] + aw) / (n_bg + a0 - bg[w] - aw))
        z[w] = (l1 - l2) / np.sqrt(1 / (fg[w] + aw) + 1 / (bg[w] + aw))
    return pd.Series(z).sort_values(ascending=False)


def discriminative_ngrams(long, topk=25):
    tr = long[(long.script == "bn") & (long.split == "train") & long.variety.isin(VARIETIES)]
    specs = {"word1-3": dict(analyzer="word", ngram_range=(1, 3), token_pattern=BN_TOKEN.pattern),
             "char2-5": dict(analyzer="char_wb", ngram_range=(2, 5))}
    for name, kw in specs.items():
        cv = CountVectorizer(min_df=3, lowercase=False, **kw)
        X = cv.fit_transform(tr.text.fillna(""))
        vocab = cv.get_feature_names_out()
        per = {v: np.asarray(X[(tr.variety == v).values].sum(0)).ravel() for v in VARIETIES}
        total = sum(per.values())
        prior = Counter(dict(zip(vocab, total)))
        out = []
        for v in DIALECTS:
            fg = +Counter(dict(zip(vocab, per[v])))
            bg = +Counter(dict(zip(vocab, total - per[v])))
            z = log_odds_dirichlet(fg, bg, prior).head(topk)
            out.append(pd.DataFrame({"dialect": v, "ngram": z.index, "z": z.values.round(2),
                                     "count": [fg[w] for w in z.index]}))
        pd.concat(out).to_csv(OUT_T / f"T10_top_{name}.csv", index=False)


def char_ngram_similarity(long):
    tr = long[(long.script == "bn") & (long.split == "train") & long.variety.isin(VARIETIES)]
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                        lowercase=False).fit_transform(tr.text.fillna(""))
    C = np.vstack([np.asarray(X[(tr.variety == v).values].mean(0)) for v in VARIETIES])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    heat(pd.DataFrame(C @ C.T, index=VARIETIES, columns=VARIETIES), "F11_char_ngram_cosine",
         "Char 2-5-gram TF-IDF centroid cosine (under-separates close dialects)", fmt=".3f")


def representation(long, n_serials=800, model_name="sentence-transformers/LaBSE"):
    import umap
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE

    bn = long[(long.script == "bn") & long.variety.isin(VARIETIES)].dropna(subset=["text"])
    complete = bn.groupby("serial").variety.nunique().eq(len(VARIETIES))
    keep = rng.choice(complete[complete].index.values, n_serials, replace=False)
    bn = bn[bn.serial.isin(keep)].sort_values(["serial", "variety"]).reset_index(drop=True)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding with {model_name} on {device}")
    E = SentenceTransformer(model_name, device=device).encode(
        bn.text.tolist(), batch_size=128 if device == "cuda" else 32,
        normalize_embeddings=True, show_progress_bar=True)
    std_emb = {s: E[i] for i, (s, v) in enumerate(zip(bn.serial, bn.variety)) if v == "Standard"}
    mask = (bn.variety != "Standard").values
    R = E[mask] - np.vstack([std_emb[s] for s in bn.serial[mask]])
    sil = {}
    for tag, M, y in [("raw", E, bn.variety.values), ("residual", R, bn.variety.values[mask])]:
        sil[tag] = silhouette_score(M, y, metric="cosine")
        for pname, proj in [("umap", umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                                                random_state=SEED)),
                            ("tsne", TSNE(perplexity=40, init="pca", random_state=SEED))]:
            Z = proj.fit_transform(M)
            plt.figure(figsize=(7, 6))
            sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, s=6, linewidth=0, palette="tab10")
            plt.title(f"{model_name.split('/')[-1]} | {tag} | {pname.upper()}")
            plt.legend(markerscale=3, fontsize=7)
            plt.tight_layout()
            plt.savefig(OUT_F / f"F13_{tag}_{pname}.png", dpi=300)
            plt.close()
    pd.Series(sil, name="silhouette_cosine").to_csv(OUT_T / "T11_silhouette.csv")

    groups = {v: E[(bn.variety == v).values] for v in VARIETIES}
    Zs = np.vstack([g[:150] for g in groups.values()])
    gamma = 1.0 / (2 * np.median(euclidean_distances(Zs, Zs)) ** 2)
    M = pd.DataFrame(0.0, index=VARIETIES, columns=VARIETIES)
    for i, a in enumerate(VARIETIES):
        for b in VARIETIES[i + 1:]:
            X, Y = groups[a], groups[b]
            M.loc[a, b] = M.loc[b, a] = (rbf_kernel(X, X, gamma=gamma).mean()
                                         + rbf_kernel(Y, Y, gamma=gamma).mean()
                                         - 2 * rbf_kernel(X, Y, gamma=gamma).mean())
    heat(M, "F14_mmd", "MMD^2 between varieties (LaBSE, RBF)", fmt=".3f")


TOKENIZERS = ["csebuetnlp/banglabert", "csebuetnlp/banglat5", "Qwen/Qwen3-8B",
              "unsloth/Meta-Llama-3.1-8B-Instruct",
              "google/gemma-4-E4B",
              "FacebookAI/xlm-roberta-base",
              "hishab/titulm-llama-3.2-3b-v2.0",
              "mistralai/Ministral-3-8B-Instruct-2512-BF16",
              "facebook/nllb-200-distilled-600M"]


def fertility(long, names=TOKENIZERS):
    from transformers import AutoTokenizer
    df = long[long.variety.isin(VARIETIES)].dropna(subset=["text"])
    rows = []
    for name in names:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"[fertility] SKIPPED {name}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue
        for (v, sc), g in df.groupby(["variety", "script"]):
            ids = tok(g.text.tolist(), add_special_tokens=False).input_ids
            rows.append(dict(tokenizer=name, variety=v, script=sc,
                             fertility=sum(map(len, ids)) / g.text.str.split().str.len().sum()))
    if not rows:
        print("[fertility] no tokenizer could be loaded (internet/HF access?) -> T12 not written")
        return
    pd.DataFrame(rows).pivot_table(index=["variety", "script"], columns="tokenizer",
                                   values="fertility").round(3).to_csv(OUT_T / "T12_fertility.csv")


def code_mixing(long):
    cm = long[long.script == "cm"].dropna(subset=["text"]).copy()

    def stats(s):
        toks = BN_TOKEN.findall(s)
        n_bn = sum(bool(re.match(r"[\u0980-\u09FF]", t)) for t in toks)
        n_en = sum(bool(re.fullmatch(r"[A-Za-z']+", t)) for t in toks)
        n, u = len(toks), len(toks) - n_bn - n_en
        cmi = 0.0 if n == u else 100 * (1 - max(n_bn, n_en) / (n - u))
        return pd.Series({"CMI": cmi, "pct_latin_tokens": 100 * n_en / max(n, 1)})

    cm = cm.join(cm.text.apply(stats))
    cm[["CMI", "pct_latin_tokens"]].describe().round(2).to_csv(OUT_T / "T13_code_mixing.csv")
    plt.figure(figsize=(7, 4))
    sns.histplot(cm.CMI, bins=30)
    plt.xlabel("Code-Mixing Index (script-based lower bound)")
    plt.tight_layout()
    plt.savefig(OUT_F / "F16_cmi_hist.png", dpi=300)
    plt.close()
    types = {(v, sc): len({t for s in g.text for t in tokens(s)})
             for (v, sc), g in long[long.script.isin(["bn", "rom"])].dropna(subset=["text"])
             .groupby(["variety", "script"])}
    pd.Series({v: types[(v, "rom")] / types[(v, "bn")] for v in VARIETIES if (v, "rom") in types},
              name="romanised_type_inflation").to_csv(OUT_T / "T13b_romanisation_inflation.csv")


ALL_FILES = ["01.Main.csv", *FILES.values()]
SPLITS = ("train", "dev", "test")


def export_splits(split):
    """Write BUC_release/{train,dev,test}/ each holding the same 10 CSVs (original columns, text passed through normalize_text), filtered by the frozen split."""
    serial_split = split.to_dict()
    counts, log = {}, []
    for fname in ALL_FILES:
        df = pd.read_csv(DATA / fname)
        df.columns = df.columns.str.strip()
        for col in [c for c in df.columns if c != "Serial"]:
            raw = df[col]
            flags = pd.DataFrame([raw_issue_flags(t) for t in raw])
            log.append({"file": fname, "column": col, "cells": len(raw), **flags.sum().astype(int)})
            df[col] = raw.map(normalize_text)
        n_blank = int(df["Serial"].isna().sum())
        if n_blank:
            print(f"{fname}: dropping {n_blank} rows with empty Serial")
            df = df.dropna(subset=["Serial"])
        df["Serial"] = df["Serial"].astype(int)
        lab = df["Serial"].map(serial_split)
        if lab.isna().any():
            raise ValueError(f"{fname}: {int(lab.isna().sum())} serials not in {SPLIT_FILE}")
        for name in SPLITS:
            (EXPORT / name).mkdir(parents=True, exist_ok=True)
            part = df[lab.values == name].sort_values("Serial")
            part.to_csv(EXPORT / name / fname, index=False, encoding="utf-8")
            counts[(name, fname)] = len(part)

    for name in SPLITS:
        sets = {f: set(pd.read_csv(EXPORT / name / f)["Serial"]) for f in ALL_FILES}
        bad = [f for f, s in sets.items() if s != sets["01.Main.csv"]]
        if bad:
            raise AssertionError(f"{name}: serial sets differ from 01.Main.csv in {bad}")
    pd.DataFrame(log).to_csv(EXPORT / "normalization_log.csv", index=False)
    table = pd.Series(counts).unstack(0)[list(SPLITS)]
    table.to_csv(EXPORT / "split_counts.csv")
    print(table)
    write_hf_readme()


def write_hf_readme():
    """readme.md with YAML 'configs' so Hugging Face shows one subset per file (main, noakhali, ...) each with train/dev/test."""
    def cfg(fname):
        return "main" if fname == "01.Main.csv" else \
            next(k for k, v in FILES.items() if v == fname).lower().replace(" ", "_")
    lines = ["---", "license: cc-by-nc-sa-4.0", "language:", "- bn", "- en", "configs:"]
    for fname in ALL_FILES:
        lines += [f"- config_name: {cfg(fname)}", "  data_files:"]
        lines += [f'  - split: {s}\n    path: "{s}/{fname}"' for s in SPLITS]
    lines += ["---", "", "# Bangla Upovasha Corpus (BUC)", "",
              "See the paper for the description of the corpus. Fields, splits (serial-level, see splits_by_serial.csv) "
              "and license: CC BY-NC-SA 4.0.", ""]
    (EXPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main_df, long = load()
    integrity(long)
    collisions(long)
    split = make_splits(main_df)
    export_splits(split)
    long["split"] = long.serial.map(split)
    descriptive(long, main_df)
    vocabulary(long)
    aligned_divergence(long)
    discriminative_ngrams(long)
    char_ngram_similarity(long)
    code_mixing(long)
    fertility(long)
    representation(long)
    long.to_parquet(Path(PTH) / "buc_long.parquet", index=False)

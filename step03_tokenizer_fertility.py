#!/usr/bin/env python3
"""Tokenizer fertility (subword tokens per whitespace word) of every model, per variety and script.

Usage:
    python step03_tokenizer_fertility.py
The output of step02 is found under /kaggle/input.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import sys
import types
from importlib import metadata
from pathlib import Path

TOKENIZERS = [
    ("BanglaBERT-large", "csebuetnlp/banglabert_large"),
    ("BanglaT5", "csebuetnlp/banglat5"),
    ("MuRIL-large", "google/muril-large-cased"),
    ("TituLLM-3B v2.0", "hishab/titulm-llama-3.2-3b-v2.0"),
    ("Gemma-4-E4B", "google/gemma-4-E4B-it"),
    ("IndicTrans2-1B", "ai4bharat/indictrans2-indic-indic-1B"),
    ("NLLB-200 (1.3B)", "facebook/nllb-200-distilled-1.3B"),
    ("XLM-R-large", "FacebookAI/xlm-roberta-large"),
    ("mBART-50", "facebook/mbart-large-50-many-to-many-mmt"),
    ("Ministral-3-8B", "mistralai/Ministral-3-8B-Instruct-2512-BF16"),
    ("Qwen3-8B", "Qwen/Qwen3-8B"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
]
SAME_AS_BASE = [
    ("csebuetnlp/banglabert_large", "csebuetnlp/banglabert"),
    ("FacebookAI/xlm-roberta-large", "FacebookAI/xlm-roberta-base"),
    ("facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-600M"),
]
BANGLA = ["std", "nkh", "ctg", "syl", "bar", "srj", "pab", "shr", "odh", "khl"]
NAMES = {"std": "Standard", "nkh": "Noakhali", "ctg": "Chittagong", "syl": "Sylhet", "bar": "Barishal",
         "srj": "Sirajganj", "pab": "Pabna", "shr": "Shariatpur", "odh": "Old Dhaka", "khl": "Khulna"}
ITT_DIR = Path("/tmp/buc_itt")
log = logging.getLogger("buc")


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def prepare_environment() -> None:
    req = find_one("/kaggle/input/**/requirements.txt")
    if req:
        pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", req.read_text(), re.M))
        bad = []
        for k, v in pins.items():
            try:
                if metadata.version(k) != v:
                    bad.append(k)
            except metadata.PackageNotFoundError:
                bad.append(k)
        if bad:
            log.info("installing pinned requirements (%d differ) ...", len(bad))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    if not (ITT_DIR / "IndicTransToolkit").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(ITT_DIR),
                        "IndicTransToolkit==1.1.1", "indic-nlp-library-itt==0.1.1", "morfessor==2.0.6",
                        "sacremoses==0.2.0"], check=True)
    sys.path.insert(0, str(ITT_DIR))


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def indic_processor(inference: bool = True):
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


def counter(repo: str, token):
    """Function texts -> list of token counts (no special tokens)."""
    if "indictrans2" in repo:
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download
        sp = spm.SentencePieceProcessor(model_file=hf_hub_download(repo, "model.SRC", token=token))
        ip = indic_processor()

        def count(texts):
            out = []
            for i in range(0, len(texts), 512):
                pre = ip.preprocess_batch(texts[i:i + 512], src_lang="ben_Beng", tgt_lang="ben_Beng")
                out += [len(sp.encode(" ".join(p.split()[2:]), out_type=str)) for p in pre]
            return out
        return count
    from transformers import AutoTokenizer
    kw = {"use_fast": False} if repo.endswith("banglat5") else {}
    tok = AutoTokenizer.from_pretrained(repo, token=token, **kw)
    tok = getattr(tok, "tokenizer", tok)

    def count(texts):
        return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    prepare_environment()
    import pandas as pd
    token = hf_token()
    hit = find_one("/kaggle/input/**/data/aligned.parquet")
    if hit is None:
        raise SystemExit("step-02 output not found: attach the output of step02_prepare_data.py")
    table = pd.read_parquet(hit)
    out = Path("/kaggle/working/outputs/fertility")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    same = []
    for large, base in SAME_AS_BASE:
        try:
            a = AutoTokenizer.from_pretrained(large, token=token).get_vocab()
            b = AutoTokenizer.from_pretrained(base, token=token).get_vocab()
            same.append({"large": large, "base": base, "identical_vocabulary": a == b, "vocab_large": len(a),
                         "vocab_base": len(b)})
        except Exception as e:
            same.append({"large": large, "base": base, "error": f"{type(e).__name__}: {e}"})
        log.info("tokenizer identity: %s", same[-1])
    (out / "same_tokenizer_as_base.json").write_text(json.dumps(same, indent=1))

    rows = []
    for label, repo in TOKENIZERS:
        try:
            count = counter(repo, token)
        except Exception as e:
            log.error("%s: tokenizer not available (%s)", label, e)
            continue
        for v in BANGLA:
            for script in ("bn", "rom"):
                texts = table[f"{v}:{script}"].tolist()
                n_tok = sum(count(texts))
                n_words = sum(len(t.split()) for t in texts)
                rows.append({"tokenizer": label, "repo": repo, "variety": v, "script": script,
                             "fertility": round(n_tok / n_words, 4)})
        log.info("%s done", label)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fertility_long.csv", index=False)

    summ = []
    for label, _ in TOKENIZERS:
        d = df[df.tokenizer == label]
        if d.empty:
            continue
        bn = d[d.script == "bn"].set_index("variety")["fertility"]
        dia = bn.drop("std")
        summ.append({"tokenizer": label, "standard": round(bn["std"], 2), "dialects_mean": round(dia.mean(), 2),
                     "dialects_min": round(dia.min(), 2), "dialects_max": round(dia.max(), 2),
                     "dialect_max_at": NAMES[dia.idxmax()],
                     "romanized_mean": round(d[d.script == "rom"]["fertility"].mean(), 2)})
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "fertility_table.csv", index=False)
    with pd.option_context("display.width", 200):
        log.info("TABLE 3.12\n%s", summary.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = df[df.script == "bn"].pivot(index="variety", columns="tokenizer", values="fertility").loc[BANGLA]
    groups = {}
    for t in summary.sort_values("dialects_mean")["tokenizer"]:
        key = tuple(df[df.tokenizer == t].sort_values(["variety", "script"])["fertility"].round(4))
        groups.setdefault(key, []).append(t)
    merged = {" / ".join(ts): piv[ts[0]] for ts in groups.values()}
    piv = __import__("pandas").DataFrame(merged)
    order = list(merged)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, t in enumerate(order):
        ax.plot(range(len(BANGLA)), piv[t].values, marker="o", ms=4, lw=1.4, label=t,
                color=plt.cm.tab20(i / max(1, len(order) - 1)))
    ax.set_xticks(range(len(BANGLA)), [NAMES[v] for v in BANGLA], rotation=30, ha="right")
    ax.set_ylabel("subword tokens per word")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4, 6, 8], ["1", "1.5", "2", "3", "4", "6", "8"])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out / "F15_fertility.pdf")
    fig.savefig(out / "F15_fertility.png", dpi=300)
    log.info("written: %s", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

# Bangla Upovasha Corpus (BUC): code

Code for the corpus analysis and all experiments of the paper *The Bangla Upovasha Corpus: A Multi-Parallel
Corpus of Nine Bangla Regional Dialects for Dialect Identification and Translation*.

The scripts are numbered in the order in which the experiments were run. Each script is self-contained, is
started with one command and writes its results to an output folder that later scripts read.

## Environment

All experiments ran on Kaggle notebooks with two NVIDIA T4 GPUs (16 GB each), Python 3.12/3.13 and
PyTorch 2.10.0 (`step01_corpus_analysis.py` was run locally and requires Python 3.12). Package versions are pinned:

| File | Used by |
|---|---|
| `requirements.txt` | all scripts (core environment) |
| `requirements-gpu.txt` | QLoRA scripts (Unsloth), installed with `--no-deps` |
| `requirements-legacy.txt` | BanglaT5 in step 08 (Transformers 4.x); the script creates this environment itself |
| `requirements-comet.txt` | COMET in step 13 (numpy < 2); the script creates this environment itself |

Every script installs the pinned packages at start (skip with `--no-install`). Gated models (Llama-3.1, Gemma-4)
need a Hugging Face token in the environment variable `HF_TOKEN`.

**Paths.** On Kaggle, inputs are found automatically under `/kaggle/input` and results are written to
`/kaggle/working/outputs`. Elsewhere, pass `--artifacts-dir <outputs of step 02>` and `--output-dir <folder>`.
The output sub-folders keep the names used during the experiments (`step4a/`, `step4b/`, `step4c/`,
`step4d/`, `step5/`, `step8b/`, `step8/`, `runs/`); the table below lists which script writes which folder.

**Long runs.** QLoRA training and evaluation stop cleanly before the 12-hour session limit
(`--time-budget-h`, default 10.5). Rerun the same command with the previous output attached (or in the same
output folder) and the run continues from its last checkpoint or its last generated item.

## Scripts

| Script | What it does | Hardware | Output folder |
|---|---|---|---|
| `step01_corpus_analysis.py` | integrity checks, serial-level split, release export, corpus statistics, figures and tables | CPU | `BUC_release/`, `results/analysis/` |
| `step02_prepare_data.py` | aligned table, frozen evaluation sets, dialect pairs, retrieval index (BM25 + LaBSE), word lexicon | 2×T4, 45–75 min | `data/`, `retrieval/`, `lexicon/`, `analysis/` |
| `step03_tokenizer_fertility.py` | tokens per word for every model's tokenizer | CPU, 15–30 min | `fertility/` |
| `step04_llm_pilot.py` | QLoRA pilot of the five LLMs (fp16 stability, memory, speed, prompt lengths) | 2×T4 | `pilot/` |
| `step05_qlora_llms.py` | QLoRA model selection, hyperparameter search, learning-rate transfer, full training of Llama-3.1, Ministral-3, Qwen3, TituLLM-3B | 2×T4, 5.5–24 h per run | `runs/`, `search/` |
| `step06_classical_classifiers.py` | majority class, TF-IDF SVM, logistic regression, fastText | CPU, 1.5–2.5 h | `step4a/` |
| `step07_encoder_classifiers.py` | BanglaBERT-large, XLM-R-large, MuRIL-large | 2×T4 | `step4b/` |
| `step08_seq2seq.py` | BanglaT5, mBART-50, NLLB-200-distilled-1.3B | 2×T4 | `step4c/` |
| `step09_prompting.py` | zero-shot, few-shot (random / retrieved), chain-of-thought, self-consistency prompting | 2×T4 | `step5/` |
| `step10_qlora_gemma_and_evals.py` | QLoRA Gemma-4-E4B, plain QLoRA, learning curve, test-set evaluations of all adapters, external corpora (LLM) | 2×T4 | `runs/` |
| `step11_unseen_and_external.py` | unseen-dialect experiments; external corpora for SVM, encoders and BanglaT5 | CPU / 2×T4 | `step4d/` |
| `step12_collect_predictions.py` | copies the prediction files of all runs into one small folder | CPU | `collected/` |
| `step13_comet.py` | COMET (wmt22-comet-da) for every translation run | 1×T4 | `step8b/` |
| `step14_evaluate.py` | all metrics, dialect fidelity, error categories, significance tests | 1×T4 | `step8/` |
| `step15_paper_figures.py` | aligned-distance, per-dialect and fertility figures of the paper | CPU | `figures/` |

## How to run

### 1. Corpus

```bash
python step01_corpus_analysis.py        # set DATA (ten annotated CSV files), PTH and EXPORT at the top
python step02_prepare_data.py --release-dir BUC_release --output-dir outputs
python step03_tokenizer_fertility.py
```

### 2. Pilot and QLoRA LLMs (Llama-3.1, Ministral-3, Qwen3, TituLLM-3B)

```bash
python step04_llm_pilot.py                                   # optional
python step05_qlora_llms.py --preset select --models qwen3-8b,llama31-8b,ministral3-8b,gemma4-e4b,titullm-3b
python step05_qlora_llms.py --compare
python step05_qlora_llms.py --preset search --models gemma4-e4b      # hyperparameter search
python step05_qlora_llms.py --compare-search
python step05_qlora_llms.py --preset transfer --models llama31-8b     # learning-rate sweep for the other LLMs
python step05_qlora_llms.py --compare-transfer
```

Full training with the selected settings, seeds 13, 42 and 2026:

```bash
python step05_qlora_llms.py --preset base --models llama31-8b --seed 13 --lr 0.000202 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --targets all --eff-batch 16
python step05_qlora_llms.py --preset base --models ministral3-8b --seed 13 --lr 0.000153 --lora-r 64 --lora-alpha 128 --lora-dropout 0.0 --targets all --eff-batch 32
python step05_qlora_llms.py --preset base --models titullm-3b --seed 13 --lr 0.000202 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --targets all --eff-batch 16
python step05_qlora_llms.py --preset base --models qwen3-8b --seed 13 --run-name base-qwen3-8b-lr1-s13 --lr 0.000202 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --targets all --eff-batch 16
```

Repeat with `--seed 42` and `--seed 2026` (for Qwen3 also change `--run-name`).

### 3. Baselines

```bash
python step06_classical_classifiers.py
python step07_encoder_classifiers.py
python step08_seq2seq.py --models banglat5
python step08_seq2seq.py --models mbart50
python step08_seq2seq.py --models nllb-1.3b
```

### 4. Prompting

```bash
python step09_prompting.py --models gemma4-e4b      # one run per model: qwen3-8b, llama31-8b, ministral3-8b, titullm-3b
python step09_prompting.py --compare
```

### 5. Gemma-4 and test-set evaluations

```bash
python step10_qlora_gemma_and_evals.py --preset primary --seed 13       # also 42 and 2026
python step10_qlora_gemma_and_evals.py --preset plain --seed 13         # also 42 and 2026
python step10_qlora_gemma_and_evals.py --preset data10 --seed 13        # learning curve: data10, data25, data50
python step10_qlora_gemma_and_evals.py --preset ablate --from-run primary-gemma4-e4b-s13 --variants T-noex
python step10_qlora_gemma_and_evals.py --preset ablate --from-run base-llama31-8b-s13 --variants T-noex,T-extra
python step10_qlora_gemma_and_evals.py --preset external --from-run primary-gemma4-e4b-s13
```

`T-noex` tests an adapter without retrieved examples and word hints (identification, Tier 1); `T-extra` adds
Tier 2, Tier 3 and translation with an unknown source dialect. Run them for every QLoRA run.

### 6. Unseen dialects and external corpora (baselines)

```bash
python step11_unseen_and_external.py --task mt --models banglat5 --conditions std-only,one
python step11_unseen_and_external.py --task mt --models banglat5 --conditions lodo
python step11_unseen_and_external.py --task id --conditions one,lodo
python step11_unseen_and_external.py --task id-encoder --encoders banglabert-large --conditions one,lodo
python step11_unseen_and_external.py --task id-encoder --encoders muril-large --conditions one,lodo
python step11_unseen_and_external.py --task mt --models banglat5 --conditions full --external
python step11_unseen_and_external.py --task external-clf --encoders muril-large,banglabert-large
```

The external corpora (Vashantor, ONUBAD, BanglaRegionalTextCorpus) are not redistributed here; download them from
their authors and pass their folder with `--external-dir`.

### 7. Evaluation and figures

```bash
python step12_collect_predictions.py --input <folder with all run outputs> --output collected
python step13_comet.py
python step14_evaluate.py --bootstrap 10000
python step15_paper_figures.py --distance results/analysis/tables/F8_pairwise_aligned_distance.csv \
    --fertility outputs/fertility/fertility_long.csv --mt-directions outputs/step8/mt_directions.csv --out figures
```

`step14_evaluate.py` writes `mt_metrics.csv`, `mt_directions.csv`, `id_metrics.csv`, `seeds_summary.csv`,
`fidelity.csv`, `errors.csv` and `significance.csv`. Use `--systems <regex>` to score a subset of runs.

## Offline tests

```bash
python step02_prepare_data.py --release-dir BUC_release --output-dir /tmp/out --no-install --mock --allow-cpu
python step05_qlora_llms.py --dry-run --artifacts-dir outputs --no-install
python step09_prompting.py --dry-run --artifacts-dir outputs --no-install
python step14_evaluate.py --self-test --artifacts-dir outputs --no-install
```

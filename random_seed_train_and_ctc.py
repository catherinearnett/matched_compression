"""
Train monolingual SentencePiece tokenizers and calculate CTC against FLORES+.

For each language × subset × model_type × whitespace combo:
  1. Train tokenizer (vocab_size=65536)
  2. Immediately compute CTC on that language's FLORES+ sentences
  3. Append result to ctc_random_results.csv

Dataset source: catherinearnett/bilingual-tokenizer-training-data
  Configs named like: afr_Latn_subset_1, afr_Latn_subset_2, ...
  Each subset is a different data seed for the same language.
  Each config contains parquet shards loaded via load_dataset().

Strategy:
  - Main process pre-downloads all training text files before spawning workers.
  - Workers train + compute CTC in parallel. Each worker loads FLORES from the
    HF datasets disk cache (populated on first access), so no redundant downloads.
  - Results are written to CSV in the main process as futures complete.

Output:
  spm_tokenizers_monolingual/  — trained .model and .vocab files
  ctc_random_results.csv       — columns:
    tokenizer, lang, tok_type, whitespace, vocab_size, flores_lang, ctc

Usage:
    export HF_TOKEN_READ=hf_...
    python train_monolingual_tokenizers.py

Requirements:
    pip install sentencepiece datasets huggingface_hub pandas
"""

import sentencepiece as spm
import os
import sys
import traceback
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from huggingface_hub import HfApi
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

DATA_REPO     = "catherinearnett/bilingual-tokenizer-training-data"
FLORES_REPO   = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN   = "sentence"

SPM_DIR     = "spm_tokenizers_monolingual"
DATA_DIR    = "monolingual_data"
LOG_FILE    = "tokenizer_errors.txt"
CTC_OUT     = "ctc_random_results.csv"
VOCAB_SIZE  = 65536
NUM_WORKERS = 4

CTC_COLS = ["tokenizer", "lang", "tok_type", "whitespace",
            "vocab_size", "flores_lang", "ctc"]

# ── logging ───────────────────────────────────────────────────────────────────

def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())

# ── dataset discovery ─────────────────────────────────────────────────────────

def discover_datasets(token):
    """
    Discover dataset configs by listing top-level directories in the HF repo.
    Each directory is a config named afr_Latn_subset_1, etc.
    Returns list of (lang, subset, config_name) tuples.
    """
    api = HfApi()
    print(f"Listing configs in {DATA_REPO} ...")
    all_files = list(api.list_repo_files(
        repo_id=DATA_REPO, repo_type="dataset", token=token
    ))
    seen = set()
    datasets = []
    for f in all_files:
        top = f.split("/")[0]
        if top in seen or "_subset_" not in top:
            continue
        seen.add(top)
        parts = top.split("_subset_")
        lang   = parts[0]
        subset = parts[1]
        datasets.append((lang, subset, top))
    print(f"  Found {len(datasets)} configs.")
    return datasets


def download_training_text(config_name, token):
    """
    Load training data for a config via load_dataset (parquet shards).
    Writes sentences to DATA_DIR/<config_name>.txt for SPM input.
    Returns the local .txt path. Safe to call multiple times (idempotent).
    """
    local_path = os.path.join(DATA_DIR, f"{config_name}.txt")
    if os.path.exists(local_path):
        return local_path
    print(f"  Downloading {config_name} ...")
    ds = load_dataset(DATA_REPO, config_name, split="train",
                      token=token, trust_remote_code=False)
    text_col = next(
        (c for c in ["text", "sentence"] if c in ds.column_names),
        ds.column_names[0]
    )
    with open(local_path, "w", encoding="utf-8") as f:
        for row in ds[text_col]:
            f.write(row.strip() + "\n")
    return local_path

# ── job building ──────────────────────────────────────────────────────────────

def build_jobs(datasets, done_tokenizers):
    jobs = []
    for lang, subset, config_name in datasets:
        for model_type in ["bpe", "unigram"]:
            for split_by_whitespace in [True, False]:
                pretok = "whitespace" if split_by_whitespace else "nowhitespace"
                tokenizer_name = f"{lang}_subset_{subset}_{model_type}_{pretok}_{VOCAB_SIZE}"
                if tokenizer_name in done_tokenizers:
                    continue
                jobs.append({
                    "config_name":         config_name,
                    "tokenizer_name":      tokenizer_name,
                    "lang":                lang,
                    "model_type":          model_type,
                    "split_by_whitespace": split_by_whitespace,
                    "pretok":              pretok,
                })
    return jobs

# ── FLORES helpers (called inside workers) ───────────────────────────────────

def load_flores_sentences(lang, token, flores_configs):
    """Load FLORES+ sentences for a language. Uses HF datasets disk cache."""
    if lang not in flores_configs:
        return None
    sentences = []
    for split in FLORES_SPLITS:
        try:
            ds = load_dataset(FLORES_REPO, lang, split=split,
                              token=token, trust_remote_code=False)
            sentences.extend(ds[TEXT_COLUMN])
        except Exception as e:
            print(f"  WARNING: could not load FLORES+ {lang}/{split}: {e}")
    return sentences if sentences else None

# ── CTC ───────────────────────────────────────────────────────────────────────

def compute_ctc_spm(sentences, sp):
    total = 0
    for sent in sentences:
        total += len(sp.EncodeAsIds(sent))
    return total

# ── worker function (runs in subprocess) ─────────────────────────────────────

def run_job(job, token, flores_configs):
    """
    1. Train tokenizer (skip if .model already exists).
    2. Load with SentencePieceProcessor.
    3. Compute CTC for the tokenizer's own language.
    Returns (status, tokenizer_name, row_dict_or_None, error_str_or_None)
    """
    tokenizer_name = job["tokenizer_name"]
    spm_path = os.path.join(SPM_DIR, f"{tokenizer_name}.model")
    local_data = os.path.join(DATA_DIR, f"{job['config_name']}.txt")

    # 1. Train
    if not os.path.exists(spm_path):
        try:
            spm.SentencePieceTrainer.train(
                input=local_data,
                model_prefix=os.path.join(SPM_DIR, tokenizer_name),
                vocab_size=VOCAB_SIZE,
                model_type=job["model_type"],
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                normalization_rule_name="identity",
                split_by_whitespace=job["split_by_whitespace"],
                byte_fallback=True,
                num_threads=1,
            )
        except Exception as e:
            log_error(f"train:{tokenizer_name}", e)
            return "failed", tokenizer_name, None, str(e)

    # 2. Load
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_path)
    except Exception as e:
        log_error(f"load:{tokenizer_name}", e)
        return "failed", tokenizer_name, None, str(e)

    # 3. CTC
    sentences = load_flores_sentences(job["lang"], token, flores_configs)
    flores_lang = job["lang"]
    ctc_value = None

    if sentences is None:
        flores_lang = None
    else:
        try:
            ctc_value = compute_ctc_spm(sentences, sp)
        except Exception as e:
            log_error(f"ctc:{tokenizer_name}", e)
            return "failed", tokenizer_name, None, str(e)

    row = {
        "tokenizer":   tokenizer_name,
        "lang":        job["lang"],
        "tok_type":    job["model_type"],
        "whitespace":  job["pretok"],
        "vocab_size":  VOCAB_SIZE,
        "flores_lang": flores_lang,
        "ctc":         ctc_value,
    }
    return "done", tokenizer_name, row, None

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    token = os.environ.get("HF_TOKEN_READ")
    if not token:
        sys.exit("ERROR: set HF_TOKEN_READ environment variable")

    os.makedirs(SPM_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(LOG_FILE, "a") as f:
        f.write(f"\nMonolingual run — started {datetime.now().isoformat()}\n")

    # Resume: load already-computed tokenizer names
    done_tokenizers = set()
    if os.path.exists(CTC_OUT):
        try:
            done_tokenizers = set(pd.read_csv(CTC_OUT)["tokenizer"].unique())
            print(f"Resuming: {len(done_tokenizers)} tokenizers already in {CTC_OUT}")
        except Exception:
            pass
    else:
        pd.DataFrame(columns=CTC_COLS).to_csv(CTC_OUT, index=False)

    datasets = discover_datasets(token)
    jobs = build_jobs(datasets, done_tokenizers)
    total = len(jobs)
    print(f"Total jobs: {total} | Workers: {NUM_WORKERS}\n")

    if total == 0:
        print("All tokenizers trained and CTC computed. Nothing to do.")
        sys.exit(0)

    # Pre-download all training text files sequentially in the main process
    # before spawning workers, so workers never race on the same download.
    print("Pre-downloading training data...")
    needed_configs = {job["config_name"] for job in jobs}
    for config_name in sorted(needed_configs):
        download_training_text(config_name, token)
    print(f"  All {len(needed_configs)} configs ready.\n")

    # Pre-fetch FLORES+ config list and warm the disk cache for all languages
    # that appear in our jobs, so workers hit disk rather than network.
    print("Fetching FLORES+ language configs...")
    flores_configs = set(get_dataset_config_names(FLORES_REPO, token=token))
    print(f"  {len(flores_configs)} configs available.")

    needed_langs = {job["lang"] for job in jobs} & flores_configs
    print(f"  Pre-caching FLORES+ sentences for {len(needed_langs)} languages...")
    for lang in sorted(needed_langs):
        load_flores_sentences(lang, token, flores_configs)
    print()

    counts = {"done": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_job, job, token, flores_configs): job
            for job in jobs
        }
        for future in as_completed(futures):
            status, tokenizer_name, row, err = future.result()
            counts[status] += 1

            if status == "done":
                pd.DataFrame([row]).to_csv(CTC_OUT, mode="a", header=False, index=False)
                ctc_str = f"ctc={row['ctc']}" if row["ctc"] is not None else "ctc=N/A (not in FLORES+)"
                print(f"[DONE] {tokenizer_name}  {ctc_str}")
            else:
                print(f"[FAIL] {tokenizer_name}  {err}")

            d, f_ = counts["done"], counts["failed"]
            print(f"  Progress: {d+f_}/{total}  (✓ {d}  ✗ {f_})")

    d, f_ = counts["done"], counts["failed"]
    print(f"\nDone. Completed: {d}  Failed: {f_}")
    print(f"CTC results → {CTC_OUT}")
    print(f"Errors      → {LOG_FILE}")


if __name__ == "__main__":
    main()

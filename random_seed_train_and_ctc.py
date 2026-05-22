"""
Train monolingual SentencePiece tokenizers and calculate CTC against FLORES+.

For each language × subset × model_type × whitespace combo:
  1. Train tokenizer (vocab_size=65536)
  2. Immediately compute CTC on that language's FLORES+ sentences
  3. Append result to ctc_random_results.csv

Dataset source: catherinearnett/bilingual-tokenizer-training-data
  Files named like: afr_Latn_subset_1, afr_Latn_subset_2, ...
  Each subset is a different data seed for the same language.

Output:
  spm_tokenizers_monolingual/  — trained .model and .vocab files
  ctc_random_results.csv       — columns:
    tokenizer, lang, tok_type, whitespace, vocab_size, flores_lang, ctc

Usage:
    export HF_TOKEN_READ=hf_...
    python train_monolingual_tokenizers.py

Requirements:
    pip install sentencepiece transformers datasets huggingface_hub pandas
"""

import sentencepiece as spm
import os
import sys
import traceback
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

DATA_REPO     = "catherinearnett/bilingual-tokenizer-training-data"
FLORES_REPO   = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN   = "sentence"

SPM_DIR    = "spm_tokenizers_monolingual"
DATA_DIR   = "monolingual_data"
LOG_FILE   = "tokenizer_errors.txt"
CTC_OUT    = "ctc_random_results.csv"
VOCAB_SIZE = 65536

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
    List files in the HF training data repo.
    Expects filenames like: afr_Latn_subset_1[.txt[.gz]]
    Returns list of (lang, repo_path) tuples. The subset number is encoded
    in the tokenizer name but not tracked as a separate field.
    """
    api = HfApi()
    print(f"Listing files in {DATA_REPO} ...")
    all_files = list(api.list_repo_files(
        repo_id=DATA_REPO, repo_type="dataset", token=token
    ))
    datasets = []
    for f in all_files:
        basename = os.path.basename(f)
        name = basename.split(".")[0]  # strip all extensions
        parts = name.split("_subset_")
        if len(parts) == 2:
            lang   = parts[0]   # e.g. afr_Latn
            subset = parts[1]   # e.g. 1, 2, 3
            datasets.append((lang, subset, f))
    print(f"  Found {len(datasets)} dataset files.")
    return datasets


def download_dataset(repo_path, token):
    """Download a dataset file from HF to DATA_DIR; return local path."""
    local_path = os.path.join(DATA_DIR, os.path.basename(repo_path))
    if os.path.exists(local_path):
        return local_path
    print(f"  Downloading {repo_path} ...")
    downloaded = hf_hub_download(
        repo_id=DATA_REPO,
        filename=repo_path,
        repo_type="dataset",
        local_dir=DATA_DIR,
        token=token,
    )
    return downloaded

# ── job building ──────────────────────────────────────────────────────────────

def build_jobs(datasets, done_tokenizers):
    """
    For each (lang, subset, file), produce 4 jobs: bpe/unigram × whitespace/nowhitespace.
    Skip if tokenizer name is already in done_tokenizers (CTC already computed).
    """
    jobs = []
    for lang, subset, repo_path in datasets:
        for model_type in ["bpe", "unigram"]:
            for split_by_whitespace in [True, False]:
                pretok = "whitespace" if split_by_whitespace else "nowhitespace"
                tokenizer_name = f"{lang}_subset_{subset}_{model_type}_{pretok}_{VOCAB_SIZE}"
                if tokenizer_name in done_tokenizers:
                    continue
                jobs.append({
                    "repo_path":           repo_path,
                    "tokenizer_name":      tokenizer_name,
                    "lang":                lang,
                    "model_type":          model_type,
                    "split_by_whitespace": split_by_whitespace,
                    "pretok":              pretok,
                })
    return jobs

# ── FLORES helpers ────────────────────────────────────────────────────────────

_flores_configs = None
_flores_cache   = {}

def get_flores_configs(token):
    global _flores_configs
    if _flores_configs is not None:
        return _flores_configs
    try:
        _flores_configs = set(get_dataset_config_names(FLORES_REPO, token=token))
    except Exception as e:
        print(f"WARNING: could not fetch FLORES+ configs: {e}")
        _flores_configs = set()
    return _flores_configs


def load_flores_sentences(lang, token, flores_configs):
    if lang in _flores_cache:
        return _flores_cache[lang]
    if lang not in flores_configs:
        _flores_cache[lang] = None
        return None
    sentences = []
    for split in FLORES_SPLITS:
        try:
            ds = load_dataset(FLORES_REPO, lang, split=split,
                              token=token, trust_remote_code=False)
            sentences.extend(ds[TEXT_COLUMN])
        except Exception as e:
            print(f"  WARNING: could not load FLORES+ {lang}/{split}: {e}")
    result = sentences if sentences else None
    _flores_cache[lang] = result
    return result

# ── CTC ───────────────────────────────────────────────────────────────────────

def compute_ctc_spm(sentences, sp):
    """Compute CTC using a raw SentencePieceProcessor (no special token ids)."""
    total = 0
    for sent in sentences:
        total += len(sp.EncodeAsIds(sent))
    return total

# ── training + CTC ────────────────────────────────────────────────────────────

def run_job(job, token, flores_configs):
    """
    1. Train tokenizer (skip if .model already exists).
    2. Load with SentencePieceProcessor.
    3. Compute CTC for the tokenizer's own language.
    Returns (status, tokenizer_name, row_dict_or_None, error_str_or_None)
    """
    tokenizer_name = job["tokenizer_name"]
    spm_path = os.path.join(SPM_DIR, f"{tokenizer_name}.model")

    # 1. Train
    if not os.path.exists(spm_path):
        try:
            local_data = download_dataset(job["repo_path"], token)
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
        flores_lang = None  # language not in FLORES+
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
    print(f"Total jobs: {total}\n")

    if total == 0:
        print("All tokenizers trained and CTC computed. Nothing to do.")
        sys.exit(0)

    print("Fetching FLORES+ language configs...")
    flores_configs = get_flores_configs(token)
    print(f"  {len(flores_configs)} configs available\n")

    counts = {"done": 0, "failed": 0}

    # Sequential: FLORES sentences are cached in memory per language, so
    # all 4 tokenizer variants for a language reuse the same loaded sentences.
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{total}] {job['tokenizer_name']}")
        status, tokenizer_name, row, err = run_job(job, token, flores_configs)
        counts[status] += 1

        if status == "done":
            pd.DataFrame([row]).to_csv(CTC_OUT, mode="a", header=False, index=False)
            ctc_str = f"ctc={row['ctc']}" if row["ctc"] is not None else "ctc=N/A (not in FLORES+)"
            print(f"  ✓ {ctc_str}")
        else:
            print(f"  ✗ {err}")

        d, f_ = counts["done"], counts["failed"]
        print(f"  Progress: {d+f_}/{total}  (✓ {d}  ✗ {f_})")

    d, f_ = counts["done"], counts["failed"]
    print(f"\nDone. Completed: {d}  Failed: {f_}")
    print(f"CTC results → {CTC_OUT}")
    print(f"Errors      → {LOG_FILE}")


if __name__ == "__main__":
    main()

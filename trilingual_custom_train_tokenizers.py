"""
Create mixed data files and train SentencePiece tokenizers for the
estimated optimal mixes from trilingual_optimal_mix.csv.

Reuses the same pipeline as make_trilingual_data.py and
train_trilingual_tokenizers.py, but only for the estimated optimal ratios.

Output data:    trilingual_data/{l1}_{l2}_{l3}_{r1}_{r2}_{r3}_subset_1_nfc.txt
Output models:  trilingual_spm_tokenizers/{l1}_{l2}_{l3}_{r1}_{r2}_{r3}.model/.vocab

Upload with:
  huggingface-cli upload catherinearnett/trilingual-tokenizers \
    ./trilingual_spm_tokenizers trilingual_spm_tokenizers --repo-type dataset
"""

import os
import random
import unicodedata
import traceback
import sentencepiece as spm
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── config ────────────────────────────────────────────────────────────────────
OPTIMAL_CSV   = "trilingual_optimal_mix.csv"
DATA_DIR      = "trilingual_data"
SPM_DIR       = "trilingual_spm_tokenizers"
CACHE_DIR     = "/tmp"
LOG_FILE      = "trilingual_optimal_tokenizer_errors.txt"
TOTAL_SIZE    = 500_000_000
VOCAB_SIZE    = 65536
TOK_TYPE      = "bpe"
WHITESPACE    = "whitespace"
NUM_WORKERS   = 16

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPM_DIR, exist_ok=True)


# ── data mixing (mirrors make_trilingual_data.py) ─────────────────────────────

def normalize_string(text):
    return unicodedata.normalize("NFC", text)


def read_cached(lang_code, max_bytes):
    path = os.path.join(CACHE_DIR, f"{lang_code}_subset_1.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache missing for {lang_code}: {path}\n"
                                f"Run make_trilingual_data.py first to populate /tmp cache.")
    lines = []
    total_bytes = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            encoded = line.encode("utf-8")
            if total_bytes + len(encoded) > max_bytes:
                break
            lines.append(line)
            total_bytes += len(encoded)
    return lines


def make_data_file(l1, l2, l3, r1, r2, r3):
    stem     = f"{l1}_{l2}_{l3}_{r1}_{r2}_{r3}"
    out_path = os.path.join(DATA_DIR, f"{stem}_subset_1_nfc.txt")

    if os.path.exists(out_path):
        return f"SKIP (exists): {out_path}"

    try:
        d1 = read_cached(l1, TOTAL_SIZE * (r1 / 100))
        d2 = read_cached(l2, TOTAL_SIZE * (r2 / 100))
        d3 = read_cached(l3, TOTAL_SIZE * (r3 / 100))
    except Exception as e:
        return f"FAILED reading cache for {stem}: {e}"

    mixed = d1 + d2 + d3
    random.shuffle(mixed)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for line in mixed:
                f.write(normalize_string(line))
        return f"OK: {out_path}"
    except Exception as e:
        return f"FAILED writing {out_path}: {e}"


# ── tokenizer training (mirrors train_trilingual_tokenizers.py) ───────────────

def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())


def train_tokenizer(job):
    stem       = job["stem"]
    input_file = job["input_file"]
    spm_path   = os.path.join(SPM_DIR, f"{stem}.model")

    if os.path.exists(spm_path):
        return "skipped", stem

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=os.path.join(SPM_DIR, stem),
            vocab_size=VOCAB_SIZE,
            model_type=TOK_TYPE,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name="identity",
            split_by_whitespace=(WHITESPACE == "whitespace"),
            byte_fallback=True,
            num_threads=1,
        )
        return "trained", stem
    except Exception as e:
        log_error(stem, e)
        return "failed", stem


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(OPTIMAL_CSV)
    print(f"Loaded {len(df)} optimal mixes from {OPTIMAL_CSV}\n")

    with open(LOG_FILE, "w") as f:
        f.write(f"Optimal mix tokenizer log — started {datetime.now().isoformat()}\n")

    # ── Step 1: create data files ─────────────────────────────────────────────
    print("Step 1: Creating mixed data files...")
    jobs = []
    for _, row in df.iterrows():
        l1, l2, l3 = row["l1"], row["l2"], row["l3"]
        r1, r2, r3 = int(row["est_r1"]), int(row["est_r2"]), int(row["est_r3"])
        stem     = f"{l1}_{l2}_{l3}_{r1}_{r2}_{r3}"
        data_file = os.path.join(DATA_DIR, f"{stem}_subset_1_nfc.txt")
        result = make_data_file(l1, l2, l3, r1, r2, r3)
        status = "SKIP" if result.startswith("SKIP") else ("OK" if result.startswith("OK") else "FAIL")
        print(f"  [{status}] {stem}")
        if not result.startswith("FAILED"):
            jobs.append({"stem": stem, "input_file": data_file})

    print(f"\n{len(jobs)} data files ready.\n")

    # ── Step 2: train tokenizers ──────────────────────────────────────────────
    print(f"Step 2: Training {len(jobs)} tokenizers with {NUM_WORKERS} workers...")
    counts = {"trained": 0, "skipped": 0, "failed": 0}
    total = len(jobs)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(train_tokenizer, job): job for job in jobs}
        for future in as_completed(futures):
            result, label = future.result()
            counts[result] += 1
            t, s, f = counts["trained"], counts["skipped"], counts["failed"]
            done = t + s + f
            print(f"  [{result.upper():7s}] {label}  ({done}/{total}  ✓{t} ⏭{s} ✗{f})")

    t, s, f = counts["trained"], counts["skipped"], counts["failed"]
    print(f"\nDone. Trained: {t} | Skipped: {s} | Failed: {f}")
    print(f"Errors logged to: {LOG_FILE}")
    print(f"\nUpload with:")
    print(f"  huggingface-cli upload catherinearnett/trilingual-tokenizers "
          f"./{SPM_DIR} trilingual_spm_tokenizers --repo-type dataset")


if __name__ == "__main__":
    main()

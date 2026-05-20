"""
Train SentencePiece tokenizers for all trilingual data files.

One tokenizer per triple+ratio combination (130 total):
  Input:  trilingual_data/{l1}_{l2}_{l3}_{r1}_{r2}_{r3}_subset_1_nfc.txt
  Output: trilingual_spm_tokenizers/{l1}_{l2}_{l3}_{r1}_{r2}_{r3}.model/.vocab

Upload with:
  huggingface-cli upload catherinearnett/trilingual-tokenizers \
    ./trilingual_spm_tokenizers trilingual_spm_tokenizers --repo-type dataset
"""

import os
import glob
import traceback
import sentencepiece as spm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "trilingual_data"
SPM_DIR     = "trilingual_spm_tokenizers"
LOG_FILE    = "trilingual_tokenizer_errors.txt"
NUM_WORKERS = 16
VOCAB_SIZE  = 65536

# Fixed SPM settings — matching bilingual setup
TOK_TYPE   = "bpe"
WHITESPACE = "whitespace"   # → split_by_whitespace=True

# ── Build jobs ────────────────────────────────────────────────────────────────

def build_jobs():
    """Scan trilingual_data/ and build one job per file."""
    ready   = []
    missing = []

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_subset_1_nfc.txt"))):
        fname = os.path.basename(path)
        # stem: {l1}_{l2}_{l3}_{r1}_{r2}_{r3}
        stem = fname.replace("_subset_1_nfc.txt", "")
        spm_path = os.path.join(SPM_DIR, f"{stem}.model")

        if not os.path.exists(path):
            missing.append(fname)
            continue

        ready.append({
            "stem":       stem,
            "input_file": path,
            "spm_path":   spm_path,
        })

    return ready, missing

# ── Training ──────────────────────────────────────────────────────────────────

def log_error(label, exc):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIME:  {datetime.now().isoformat()}\n")
        f.write(f"JOB:   {label}\n")
        f.write(f"ERROR: {exc}\n")
        f.write(traceback.format_exc())


def run_job(job):
    stem       = job["stem"]
    input_file = job["input_file"]
    spm_path   = job["spm_path"]

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SPM_DIR, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write(f"Trilingual tokenizer error log — started {datetime.now().isoformat()}\n")

    ready_jobs, missing_files = build_jobs()

    print(f"Jobs ready to train : {len(ready_jobs)}")
    print(f"Vocab size          : {VOCAB_SIZE}")
    print(f"Model type          : {TOK_TYPE}")
    print(f"Split by whitespace : {WHITESPACE == 'whitespace'}")
    if missing_files:
        print(f"\nInput files not found ({len(missing_files)}) — skipping:")
        for f in missing_files:
            print(f"  {f}")
    print()

    if not ready_jobs:
        print("Nothing to train! Run make_trilingual_data.py first.")
        return

    total  = len(ready_jobs)
    counts = {"trained": 0, "skipped": 0, "failed": 0}

    print(f"Training {total} tokenizers with {NUM_WORKERS} workers...\n")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in ready_jobs}
        for future in as_completed(futures):
            result, label = future.result()
            counts[result] += 1
            t, s, f = counts["trained"], counts["skipped"], counts["failed"]
            done = t + s + f
            print(f"[{result.upper():7s}] {label}")
            print(f"  Progress: {done}/{total}  (✓ {t}  ⏭ {s}  ✗ {f})")

    t, s, f = counts["trained"], counts["skipped"], counts["failed"]
    print(f"\nDone. Trained: {t} | Skipped: {s} | Failed: {f}")
    print(f"Errors logged to: {LOG_FILE}")
    print(f"\nUpload with:")
    print(f"  huggingface-cli upload catherinearnett/trilingual-tokenizers "
          f"./{SPM_DIR} trilingual_spm_tokenizers --repo-type dataset")


if __name__ == "__main__":
    main()

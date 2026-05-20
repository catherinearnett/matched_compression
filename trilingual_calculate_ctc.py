"""
Calculate CTC (token count) for all trilingual SentencePiece tokenizers
against FLORES+ for each of the three languages in the triple.

Tokenizers are loaded locally from: trilingual_spm_tokenizers/
Each .model file stem: {l1}_{l2}_{l3}_{r1}_{r2}_{r3}

CTC = total non-special tokens across all sentences (dev + devtest).

Output:
  trilingual_ctc_results.csv — columns:
    tokenizer, l1, l2, l3, r1, r2, r3, flores_lang, ctc

Usage:
    export HF_TOKEN_READ=hf_...
    python calculate_trilingual_ctc.py
    python calculate_trilingual_ctc.py --out trilingual_ctc_results.csv --resume

Requirements:
    pip install sentencepiece datasets pandas huggingface_hub
"""

import os
import sys
import glob
import argparse
import sentencepiece as spm
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

SPM_DIR       = "trilingual_spm_tokenizers"
FLORES_REPO   = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN   = "text"

OUT_COLS = ["tokenizer", "l1", "l2", "l3", "r1", "r2", "r3", "flores_lang", "ctc"]


# ── stem parsing ──────────────────────────────────────────────────────────────

def parse_stem(stem):
    """
    Parse stem of the form: {l1}_{l2}_{l3}_{r1}_{r2}_{r3}
    e.g. gle_Latn_cym_Latn_swh_Latn_34_33_33
    Parts: lang(2) script(1) lang(2) script(1) lang(2) script(1) r1 r2 r3 = 9 parts
    """
    parts = stem.split("_")
    if len(parts) != 9:
        return None
    try:
        l1 = f"{parts[0]}_{parts[1]}"
        l2 = f"{parts[2]}_{parts[3]}"
        l3 = f"{parts[4]}_{parts[5]}"
        r1 = int(parts[6])
        r2 = int(parts[7])
        r3 = int(parts[8])
        if r1 + r2 + r3 != 100:
            return None
        return {"l1": l1, "l2": l2, "l3": l3, "r1": r1, "r2": r2, "r3": r3}
    except (ValueError, IndexError):
        return None


# ── FLORES helpers ────────────────────────────────────────────────────────────

def get_flores_configs(token):
    try:
        return set(get_dataset_config_names(FLORES_REPO, token=token))
    except Exception as e:
        print(f"WARNING: could not fetch FLORES+ config list: {e}")
        return set()


_flores_cache = {}

def load_flores_sentences(lang, token, flores_configs):
    if lang not in flores_configs:
        return None
    if lang in _flores_cache:
        return _flores_cache[lang]
    sentences = []
    for split in FLORES_SPLITS:
        try:
            ds = load_dataset(FLORES_REPO, lang, split=split,
                              token=token, trust_remote_code=False)
            sentences.extend(ds[TEXT_COLUMN])
        except Exception as e:
            print(f"  WARNING: could not load {lang}/{split}: {e}")
    result = sentences if sentences else None
    _flores_cache[lang] = result
    return result


# ── CTC ───────────────────────────────────────────────────────────────────────

def compute_ctc(sentences, sp):
    """Count non-special tokens using SentencePiece model directly."""
    total = 0
    for sent in sentences:
        pieces = sp.encode(sent, out_type=str)
        total += len(pieces)
    return total


def process_one(stem, model_path, info, flores_configs, token):
    """Load local .model file and compute CTC for l1, l2, l3."""
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
    except Exception as e:
        return [], f"ERROR loading model {model_path}: {e}"

    rows = []
    for flores_lang in [info["l1"], info["l2"], info["l3"]]:
        sentences = load_flores_sentences(flores_lang, token, flores_configs)
        if sentences is None:
            print(f"  SKIP {flores_lang} (not in FLORES+)")
            continue
        try:
            ctc = compute_ctc(sentences, sp)
        except Exception as e:
            print(f"  ERROR computing CTC for {flores_lang}: {e}")
            ctc = None
        rows.append({
            "tokenizer":   stem,
            "l1":          info["l1"],
            "l2":          info["l2"],
            "l3":          info["l3"],
            "r1":          info["r1"],
            "r2":          info["r2"],
            "r3":          info["r3"],
            "flores_lang": flores_lang,
            "ctc":         ctc,
        })

    return rows, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="trilingual_ctc_results.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tokenizers already in output CSV")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_READ")
    if not token:
        sys.exit("ERROR: set HF_TOKEN_READ environment variable")

    # ── collect local .model files ────────────────────────────────────────────
    model_files = sorted(glob.glob(os.path.join(SPM_DIR, "*.model")))
    print(f"Found {len(model_files)} .model files in {SPM_DIR}/")

    jobs = []
    for model_path in model_files:
        stem = os.path.basename(model_path).replace(".model", "")
        info = parse_stem(stem)
        if info:
            jobs.append((stem, model_path, info))
        else:
            print(f"  SKIP (unparseable): {stem}")

    print(f"  {len(jobs)} valid tokenizers\n")

    # ── resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["tokenizer"].unique())
        before = len(jobs)
        jobs = [(s, p, i) for s, p, i in jobs if s not in done]
        print(f"Resuming: skipped {before - len(jobs)} done, {len(jobs)} remaining\n")

    # ── FLORES+ configs ───────────────────────────────────────────────────────
    print("Fetching FLORES+ language configs...")
    flores_configs = get_flores_configs(token)
    print(f"  {len(flores_configs)} configs available\n")

    # ── write CSV header ──────────────────────────────────────────────────────
    if not (args.resume and os.path.exists(args.out)):
        pd.DataFrame(columns=OUT_COLS).to_csv(args.out, index=False)

    # ── process ───────────────────────────────────────────────────────────────
    total = len(jobs)
    done_count = err_count = 0

    for i, (stem, model_path, info) in enumerate(jobs, 1):
        print(f"[{i}/{total}] {stem}")

        rows, err = process_one(stem, model_path, info, flores_configs, token)

        if err:
            print(f"  {err}")
            err_count += 1
        else:
            if rows:
                pd.DataFrame(rows).to_csv(args.out, mode="a", header=False, index=False)
            done_count += 1
            lang_ctc = ", ".join(r["flores_lang"] + "=" + str(r["ctc"]) for r in rows)
            print(f"  → {len(rows)} CTC values written  ({lang_ctc})")

    print(f"\nDone. Processed: {done_count}  Errors: {err_count}")
    print(f"Results: {args.out}  ({pd.read_csv(args.out).shape[0]} rows)")


if __name__ == "__main__":
    main()

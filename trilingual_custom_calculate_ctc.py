"""
Calculate CTC for the 10 optimally-mixed trilingual tokenizers.
Reads stems from trilingual_optimal_mix.csv (est_r1/r2/r3 columns),
loads models from trilingual_spm_tokenizers/, saves to trilingual_custom_ctc.csv.
"""

import os
import sys
import argparse
import sentencepiece as spm
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
import warnings
warnings.filterwarnings("ignore")

OPTIMAL_CSV   = "trilingual_optimal_mix.csv"
SPM_DIR       = "trilingual_spm_tokenizers"
FLORES_REPO   = "openlanguagedata/flores_plus"
FLORES_SPLITS = ["dev", "devtest"]
TEXT_COLUMN   = "text"
TOK_TYPE      = "bpe"
WHITESPACE    = "whitespace"
VOCAB_SIZE    = 65536

OUT_COLS = ["tokenizer", "l1", "l2", "l3", "r1", "r2", "r3",
            "tok_type", "whitespace", "vocab_size", "flores_lang", "ctc"]


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
    total = 0
    for sent in sentences:
        total += len(sp.encode(sent, out_type=str))
    return total


def process_one(stem, model_path, l1, l2, l3, r1, r2, r3, flores_configs, token):
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
    except Exception as e:
        return [], f"ERROR loading model {model_path}: {e}"

    rows = []
    for flores_lang in [l1, l2, l3]:
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
            "l1":          l1,
            "l2":          l2,
            "l3":          l3,
            "r1":          r1,
            "r2":          r2,
            "r3":          r3,
            "tok_type":    TOK_TYPE,
            "whitespace":  WHITESPACE,
            "vocab_size":  VOCAB_SIZE,
            "flores_lang": flores_lang,
            "ctc":         ctc,
        })

    return rows, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="trilingual_custom_ctc.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN_READ")
    if not token:
        sys.exit("ERROR: set HF_TOKEN_READ environment variable")

    # ── build job list from optimal CSV ───────────────────────────────────────
    df = pd.read_csv(OPTIMAL_CSV)
    jobs = []
    for _, row in df.iterrows():
        l1, l2, l3 = row["l1"], row["l2"], row["l3"]
        r1, r2, r3 = int(row["est_r1"]), int(row["est_r2"]), int(row["est_r3"])
        stem       = f"{l1}_{l2}_{l3}_{r1}_{r2}_{r3}"
        model_path = os.path.join(SPM_DIR, f"{stem}.model")
        if not os.path.exists(model_path):
            print(f"SKIP (model not found): {stem}")
            continue
        jobs.append((stem, model_path, l1, l2, l3, r1, r2, r3))

    print(f"Found {len(jobs)} optimal models to evaluate\n")

    # ── resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out)["tokenizer"].unique())
        before = len(jobs)
        jobs = [j for j in jobs if j[0] not in done]
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

    for i, (stem, model_path, l1, l2, l3, r1, r2, r3) in enumerate(jobs, 1):
        print(f"[{i}/{total}] {stem}")

        rows, err = process_one(stem, model_path, l1, l2, l3, r1, r2, r3,
                                flores_configs, token)
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

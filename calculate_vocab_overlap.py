"""
Calculate pairwise vocabulary overlap between monolingual SentencePiece tokenizers.

Monolingual tokenizers have exactly one language code in the name:
  {lang}_{script}_{tok_type}_{whitespace}_{vocab_size}.model
  e.g. zsm_Latn_bpe_nowhitespace_16384.model

Only compares tokenizers with matching tok_type, whitespace, and vocab_size.
Overlap = number of vocab tokens that appear in both tokenizers.

Output: vocab_overlap.csv
  lang1, lang2, tok_type, whitespace, vocab_size, overlap, jaccard
"""

import os
import glob
import re
import itertools
import sentencepiece as spm
import pandas as pd

SPM_DIR = "/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/spm_tokenizers"
OUT_FILE = "vocab_overlap.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_mono_stem(stem):
    """
    Parse a monolingual stem: {lang}_{script}_{tok_type}_{whitespace}_{vocab_size}
    e.g. zsm_Latn_bpe_nowhitespace_16384
    Returns dict or None if not monolingual.
    """
    parts = stem.split("_")
    # Monolingual: lang(1) + script(1) + tok_type(1) + whitespace(1) + vocab_size(1) = 5 parts
    # Bilingual would have 9+ parts
    if len(parts) != 5:
        return None
    lang_code, script, tok_type, whitespace, vocab_size = parts
    if tok_type not in ("bpe", "unigram"):
        return None
    if whitespace not in ("whitespace", "nowhitespace"):
        return None
    try:
        int(vocab_size)
    except ValueError:
        return None
    return {
        "lang":       f"{lang_code}_{script}",
        "tok_type":   tok_type,
        "whitespace": whitespace,
        "vocab_size": vocab_size,
    }


def get_vocab(model_path):
    """Return set of vocab tokens from a .model file."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return set(sp.id_to_piece(i) for i in range(sp.get_piece_size()))


# ── collect monolingual .model files ─────────────────────────────────────────

model_files = sorted(glob.glob(os.path.join(SPM_DIR, "*.model")))

tokenizers = {}  # stem -> {lang, tok_type, whitespace, vocab_size, path}
for path in model_files:
    stem = os.path.basename(path).replace(".model", "")
    info = parse_mono_stem(stem)
    if info:
        tokenizers[stem] = {**info, "path": path}

print(f"Found {len(tokenizers)} monolingual tokenizers:")
for stem in sorted(tokenizers):
    print(f"  {stem}")
print()

# ── group by (tok_type, whitespace, vocab_size) ───────────────────────────────

from collections import defaultdict
groups = defaultdict(list)
for stem, info in tokenizers.items():
    key = (info["tok_type"], info["whitespace"], info["vocab_size"])
    groups[key].append(stem)

total_pairs = sum(len(stems) * (len(stems) - 1) // 2 for stems in groups.values())
print(f"Comparison groups: {len(groups)}")
print(f"Total pairs to compare: {total_pairs}\n")

# ── compute pairwise overlap ──────────────────────────────────────────────────

rows = []
done = 0

for (tok_type, whitespace, vocab_size), stems in sorted(groups.items()):
    if len(stems) < 2:
        continue
    print(f"[{tok_type} / {whitespace} / {vocab_size}] — {len(stems)} tokenizers, "
          f"{len(stems)*(len(stems)-1)//2} pairs")

    # Load all vocabs in this group
    vocabs = {}
    for stem in stems:
        vocabs[stem] = get_vocab(tokenizers[stem]["path"])
        print(f"  loaded {stem}: {len(vocabs[stem])} tokens")

    # Pairwise comparisons
    for stem1, stem2 in itertools.combinations(stems, 2):
        v1, v2 = vocabs[stem1], vocabs[stem2]
        overlap = len(v1 & v2)
        jaccard = overlap / len(v1 | v2) if v1 | v2 else 0.0
        rows.append({
            "lang1":      tokenizers[stem1]["lang"],
            "lang2":      tokenizers[stem2]["lang"],
            "tok_type":   tok_type,
            "whitespace": whitespace,
            "vocab_size": vocab_size,
            "overlap":    overlap,
            "jaccard":    round(jaccard, 6),
        })
        done += 1

    print(f"  → {len(stems)*(len(stems)-1)//2} pairs done\n")

# ── save ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows, columns=["lang1", "lang2", "tok_type", "whitespace",
                                  "vocab_size", "overlap", "jaccard"])
df.to_csv(OUT_FILE, index=False)
print(f"Done. {len(df)} pairs written to {OUT_FILE}")

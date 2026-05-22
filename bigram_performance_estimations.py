"""
Bigram stupid-backoff baseline evaluated on FLORES.

For each model stem in CUSTOM_MODELS_DIR:
  1. Load the stem's tokenizer
  2. Train a token-level bigram LM on the same tokenized training data
     (custom_tokenized_data/{stem}.txt — one space-separated token-id sequence per line)
  3. Evaluate mean sentence NLL (nats) on FLORES devtest for both languages,
     second-half-only, matching the GPT eval script exactly.

Smoothing: stupid backoff (Brants et al. 2007)
  P_sb(w | c) = count(c,w)/count(c)          if count(c,w) > 0
              = λ * count(w)/N                otherwise   (λ = 0.40)

NLL per token = -ln P_sb(w | context)
Sent-NLL = raw sum of token NLLs in nats over second-half tokens.

Output: bigram_flores_nll_results.csv
  columns: model, flores_lang, mean_nll, se_nll, n_sequences
"""

import os
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
FLORES_SPLIT       = "devtest"
MAX_SEQ_LEN        = 512
ONLY_SECOND_HALF   = True
LAMBDA_BACKOFF     = 0.40
CUSTOM_MODELS_DIR  = "/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models"
TOKENIZED_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_tokenized_data")
TOKENIZERS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_tokenizers")
CSV_PATH           = "bigram_flores_nll_results.csv"

FLORES_LANG_MAPPING = {
    'ace_Arab': 'urd_Arab', 'acm_Arab': 'arb_Arab', 'acq_Arab': 'arb_Arab',
    'aeb_Arab': 'arb_Arab', 'ajp_Arab': 'arb_Arab', 'als_Latn': 'sqi_Latn',
    'arb_Latn': 'mlt_Latn', 'ars_Arab': 'arb_Arab', 'ary_Arab': 'arb_Arab',
    'awa_Deva': 'hin_Deva', 'ayr_Latn': 'aym_Latn', 'azb_Arab': 'aze_Arab',
    'azj_Latn': 'aze_Latn', 'bjn_Arab': 'urd_Arab', 'dik_Latn': 'din_Latn',
    'gaz_Latn': 'orm_Latn', 'kam_Latn': 'kik_Latn', 'kas_Arab': 'urd_Arab',
    'khk_Cyrl': 'mon_Cyrl', 'kmr_Latn': 'kur_Latn', 'lvs_Latn': 'lav_Latn',
    'min_Arab': 'urd_Arab', 'mni_Beng': 'ben_Beng', 'npi_Deva': 'nep_Deva',
    'nus_Latn': 'din_Latn', 'ory_Orya': 'ori_Orya', 'pbt_Arab': 'pus_Arab',
    'plt_Latn': 'mlg_Latn', 'quy_Latn': 'que_Latn', 'swh_Latn': 'swa_Latn',
    'taq_Latn': 'kab_Latn', 'taq_Tfng': None, 'tzm_Tfng': None,
    'uzn_Latn': 'uzb_Latn', 'ydd_Hebr': 'yid_Hebr', 'yue_Hant': 'zho_Hant',
    'zsm_Latn': 'msa_Latn',
}

LOG_LAMBDA = math.log(LAMBDA_BACKOFF)  # nats


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_langs_from_model_name(model_name):
    m = re.match(r'^([a-z]{2,3}_[A-Z][a-z]{3})_([a-z]{2,3}_[A-Z][a-z]{3})_', model_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


def load_tokenizer(stem):
    """Load tokenizer for a given stem from local TOKENIZERS_DIR."""
    tok_dir  = os.path.join(TOKENIZERS_DIR, stem)
    tok_json = os.path.join(tok_dir, 'tokenizer.json')
    if not os.path.isfile(tok_json):
        raise FileNotFoundError(f"No tokenizer.json at {tok_json}")
    try:
        return AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    except Exception:
        base_tok  = Tokenizer.from_file(tok_json)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tok)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({'cls_token': '<s>'})
        return tokenizer


def build_bigram_counts(tokenized_file):
    """
    Read tokenized file (one sequence per line, space-separated token ids as strings).
    Returns:
        unigram_counts : dict  token_id -> int
        bigram_counts  : dict  (context_id, token_id) -> int
        context_totals : dict  context_id -> int
        total_tokens   : int
    """
    unigram_counts = defaultdict(int)
    bigram_counts  = defaultdict(int)
    total_tokens   = 0

    with open(tokenized_file) as f:
        for line in f:
            ids = list(map(int, line.strip().split()))
            if not ids:
                continue
            for tok in ids:
                unigram_counts[tok] += 1
            for ctx, tok in zip(ids[:-1], ids[1:]):
                bigram_counts[(ctx, tok)] += 1
            total_tokens += len(ids)

    context_totals = defaultdict(int)
    for (ctx, tok), cnt in bigram_counts.items():
        context_totals[ctx] += cnt

    return unigram_counts, bigram_counts, context_totals, total_tokens


def stupid_backoff_log_prob(ctx, tok, unigram_counts, bigram_counts, context_totals, total_tokens):
    """
    ln P_sb(tok | ctx)  — nats
    = ln(count(ctx,tok) / count(ctx))          if count(ctx,tok) > 0
    = ln(λ) + ln(count(tok) / total_tokens)    otherwise
    """
    bi_cnt = bigram_counts.get((ctx, tok), 0)
    if bi_cnt > 0:
        return math.log(bi_cnt) - math.log(context_totals[ctx])
    uni_cnt = unigram_counts.get(tok, 0)
    if uni_cnt > 0:
        return LOG_LAMBDA + math.log(uni_cnt) - math.log(total_tokens)
    # Unseen unigram: floor at 1 count
    return LOG_LAMBDA + math.log(1) - math.log(total_tokens)


def load_flores_lines(flores_lang_code, split, token):
    try:
        ds = load_dataset("openlanguagedata/flores_plus", flores_lang_code, split=split, token=token)
        return [row['text'] for row in ds]
    except Exception:
        pass
    mapped = FLORES_LANG_MAPPING.get(flores_lang_code)
    if mapped is None:
        print(f"  No FLORES mapping for {flores_lang_code}, skipping.")
        return None
    try:
        ds = load_dataset("openlanguagedata/flores_plus", mapped, split=split, token=token)
        print(f"  Loaded {flores_lang_code} via mapping -> {mapped}")
        return [row['text'] for row in ds]
    except Exception as e:
        print(f"  Could not load FLORES for {flores_lang_code} (or {mapped}): {e}")
        return None


def compute_bigram_sentence_nlls(tokenizer, lines,
                                  unigram_counts, bigram_counts,
                                  context_totals, total_tokens):
    """
    Returns one Sent-NLL value (raw sum of token NLLs in nats) per sequence,
    matching the GPT eval script exactly.
    """
    prepend_token_id = (
        tokenizer.cls_token_id
        or tokenizer.bos_token_id
        or tokenizer.eos_token_id
    )
    unk_token_id = tokenizer.unk_token_id
    vocab_size   = tokenizer.vocab_size
    unk_nll      = math.log(vocab_size) if vocab_size else 20.0  # nats, matches GPT script

    nlls = []

    for line in tqdm(lines, leave=False):
        ids = tokenizer(line, add_special_tokens=False)['input_ids']

        if prepend_token_id is not None:
            ids = [prepend_token_id] + ids

        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]

        if len(ids) < 2:
            continue

        if ONLY_SECOND_HALF:
            halfline     = line[: len(line) // 2]
            halfline_len = len(tokenizer(halfline, add_special_tokens=False)['input_ids'])
        else:
            halfline_len = 0

        token_nlls       = []
        second_half_mask = []

        for t in range(1, len(ids)):
            ctx = ids[t - 1]
            tok = ids[t]

            if unk_token_id is not None and tok == unk_token_id:
                nll = unk_nll
            else:
                nll = -stupid_backoff_log_prob(
                    ctx, tok,
                    unigram_counts, bigram_counts,
                    context_totals, total_tokens
                )

            token_nlls.append(nll)
            second_half_mask.append(t - 1 >= halfline_len)

        if ONLY_SECOND_HALF:
            selected = [n for n, m in zip(token_nlls, second_half_mask) if m]
        else:
            selected = token_nlls

        if not selected:
            continue

        nlls.append(sum(selected))  # raw sum, not mean

    return nlls


def already_done(existing_df, model_name, flores_code):
    if existing_df.empty:
        return False
    mask = (
        (existing_df['model'].str.strip() == model_name.strip()) &
        (existing_df['flores_lang'].str.strip() == flores_code.strip())
    )
    return mask.any()


# ── main ──────────────────────────────────────────────────────────────────────

token = os.environ.get("HF_TOKEN_READ")

if os.path.isfile(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    existing_df['model']       = existing_df['model'].astype(str)
    existing_df['flores_lang'] = existing_df['flores_lang'].astype(str)
    print(f"Loaded {len(existing_df)} existing rows from {CSV_PATH}")
else:
    existing_df = pd.DataFrame()
    print("No existing CSV found, starting fresh.")

model_names = sorted([
    d for d in os.listdir(CUSTOM_MODELS_DIR)
    if os.path.isdir(os.path.join(CUSTOM_MODELS_DIR, d))
])
print(f"Found {len(model_names)} model directories.")

all_lang_codes = set()
for name in model_names:
    l1, l2 = parse_langs_from_model_name(name)
    if l1: all_lang_codes.add(l1)
    if l2: all_lang_codes.add(l2)

flores_sentences = {}
for code in sorted(all_lang_codes):
    lines = load_flores_lines(code, FLORES_SPLIT, token)
    if lines is not None:
        flores_sentences[code] = lines
        print(f"  FLORES {code}: {len(lines)} sentences")

new_rows = []

for model_name in model_names:
    lang1, lang2 = parse_langs_from_model_name(model_name)
    if lang1 is None:
        print(f"\nSkipping {model_name} — could not parse language codes.")
        continue

    stem           = model_name
    tokenized_file = os.path.join(TOKENIZED_DIR, f"{stem}.txt")

    if not os.path.exists(tokenized_file):
        print(f"\nSkipping {stem} — no tokenized file at {tokenized_file}")
        continue

    target_langs = [c for c in [lang1, lang2] if c in flores_sentences]
    needed = [c for c in target_langs if not already_done(existing_df, model_name, c)]

    if not needed:
        print(f"\n{model_name}: already done, skipping.")
        continue

    print(f"\n{model_name}  |  langs: {lang1}, {lang2}")

    try:
        tokenizer = load_tokenizer(stem)
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        continue

    print(f"  Building bigram counts from {tokenized_file} ...")
    unigram_counts, bigram_counts, context_totals, total_tokens = build_bigram_counts(tokenized_file)
    print(f"  Unigrams: {len(unigram_counts):,}  Bigrams: {len(bigram_counts):,}  Tokens: {total_tokens:,}")

    for flores_code in needed:
        print(f"  Evaluating {flores_code} ...")
        nlls     = compute_bigram_sentence_nlls(
            tokenizer, flores_sentences[flores_code],
            unigram_counts, bigram_counts, context_totals, total_tokens
        )
        mean_nll = float(np.mean(nlls))
        se_nll   = float(np.std(nlls, ddof=1) / np.sqrt(len(nlls)))
        print(f"  {flores_code} mean Sent-NLL: {mean_nll:.4f} ± {se_nll:.4f} nats")
        new_rows.append({
            'model':       model_name,
            'flores_lang': flores_code,
            'mean_nll':    mean_nll,
            'se_nll':      se_nll,
            'n_sequences': len(nlls),
        })

    del unigram_counts, bigram_counts, context_totals

new_df   = pd.DataFrame(new_rows)
combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.to_csv(CSV_PATH, index=False)
print(f"\nSaved {len(combined)} total rows to {CSV_PATH}")
print(combined)

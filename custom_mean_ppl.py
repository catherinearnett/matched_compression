import os
import re
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from datasets import load_dataset
import pandas as pd


# --- Config ---
FLORES_SPLIT = "devtest"
MAX_SEQ_LEN = 512
ONLY_SECOND_HALF = True
CUSTOM_MODELS_DIR = "/mnt/ssd-3/catherine/bilingual_tokenizers/matched_compression/custom_models"
CSV_PATH = "flores_nll_results.csv"

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


def parse_langs_from_model_name(model_name):
    """Extract the two language codes from a model folder name.
    e.g. afr_Latn_als_Latn_50_50_bpe_nowhitespace_16384 -> ('afr_Latn', 'als_Latn')
    """
    m = re.match(r'^([a-z]{2,3}_[A-Z][a-z]{3})_([a-z]{2,3}_[A-Z][a-z]{3})_', model_name)
    if m:
        return m.group(1), m.group(2)
    return None, None


# def get_local_checkpoints(model_dir):
#     """Return sorted checkpoint subdirs plus the model root ('main')."""
#     checkpoints = sorted(
#         [d for d in os.listdir(model_dir)
#          if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('checkpoint-')],
#         key=lambda x: int(x.split('-')[-1])
#     )
#     return checkpoints + ['main']

def get_local_checkpoints(model_dir):
    """Return only the final model checkpoint."""
    return ['main']

def load_tokenizer_and_model(model_dir, checkpoint):
    """Load tokenizer and model from a local checkpoint (or model root if 'main')."""
    ckpt_dir = model_dir if checkpoint == 'main' else os.path.join(model_dir, checkpoint)
    print(f"  Loading from {ckpt_dir}")

    # Tokenizer: prefer tokenizer.json in checkpoint dir, fall back to model root
    tokenizer_path = os.path.join(ckpt_dir, 'tokenizer.json')
    if not os.path.isfile(tokenizer_path):
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer.json found in {ckpt_dir} or {model_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:
            print(f"    AutoTokenizer failed ({e}), falling back to tokenizers library...")
            base_tok = Tokenizer.from_file(tokenizer_path)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tok)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.cls_token is None:
                tokenizer.add_special_tokens({'cls_token': '<s>'})

    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, config=config)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model


def load_flores_lines(flores_lang_code, split, token):
    """Load FLORES sentences, falling back via FLORES_LANG_MAPPING if needed."""
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


def compute_sequence_nlls(tokenizer, model, lines):
    """Returns one Sent-NLL value (raw sum of token NLLs in nats) per sequence."""
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=(-100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id),
        reduction='none'
    )
    unk_token_id = tokenizer.unk_token_id
    prepend_token_id = tokenizer.cls_token_id
    if prepend_token_id is None:
        prepend_token_id = tokenizer.bos_token_id
    if prepend_token_id is None:
        prepend_token_id = tokenizer.eos_token_id
    use_prepend = prepend_token_id is not None
    if not use_prepend:
        print("  Warning: no CLS/BOS/EOS token found, skipping prepend token.")

    sequence_nlls = []

    for line in tqdm(lines, leave=False):
        inputs = tokenizer([line], add_special_tokens=False)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if use_prepend:
            input_ids[0].insert(0, prepend_token_id)
            attention_mask[0].insert(0, 1)

        if len(input_ids[0]) > MAX_SEQ_LEN:
            input_ids[0] = input_ids[0][:MAX_SEQ_LEN]
            attention_mask[0] = attention_mask[0][:MAX_SEQ_LEN]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        logits = outputs['logits'].detach()
        labels = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        logits = torch.transpose(logits, 1, 2)
        losses = loss_fn(logits, labels).cpu()  # nats, no conversion

        if unk_token_id is not None:
            losses[labels.cpu() == unk_token_id] = np.log(tokenizer.vocab_size)  # nats

        if ONLY_SECOND_HALF:
            halfline = line[:(len(line) // 2)]
            halfline_len = len(tokenizer([halfline], add_special_tokens=False)['input_ids'][0])
            losses[0, :halfline_len] = 0.0

        sentence_nll = losses[0].sum().item()
        sequence_nlls.append(sentence_nll)

    return sequence_nlls


def already_done(existing_df, model_name, ckpt, flores_code):
    if existing_df.empty:
        return False
    mask = (
        (existing_df['model'].str.strip() == model_name.strip()) &
        (existing_df['checkpoint'].str.strip() == ckpt.strip()) &
        (existing_df['flores_lang'].str.strip() == flores_code.strip())
    )
    return mask.any()


# --- Main ---
token = os.environ.get("HF_TOKEN_READ")

if os.path.isfile(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    existing_df['model'] = existing_df['model'].astype(str)
    existing_df['checkpoint'] = existing_df['checkpoint'].astype(str)
    existing_df['flores_lang'] = existing_df['flores_lang'].astype(str)
    print(f"Loaded {len(existing_df)} existing rows from {CSV_PATH}")
else:
    existing_df = pd.DataFrame()
    print("No existing CSV found, starting fresh.")

# Discover model directories
model_names = sorted([
    d for d in os.listdir(CUSTOM_MODELS_DIR)
    if os.path.isdir(os.path.join(CUSTOM_MODELS_DIR, d))
])
print(f"Found {len(model_names)} model directories.")

# Collect all unique language codes
all_lang_codes = set()
for model_name in model_names:
    lang1, lang2 = parse_langs_from_model_name(model_name)
    if lang1: all_lang_codes.add(lang1)
    if lang2: all_lang_codes.add(lang2)

# Pre-load all FLORES sentence sets
flores_sentences = {}
for code in sorted(all_lang_codes):
    print(f"Loading FLORES for {code} ...")
    lines = load_flores_lines(code, FLORES_SPLIT, token)
    if lines is not None:
        flores_sentences[code] = lines
        print(f"  -> {len(lines)} sentences")

# Main loop
new_rows = []

for model_name in model_names:
    model_dir = os.path.join(CUSTOM_MODELS_DIR, model_name)
    lang1, lang2 = parse_langs_from_model_name(model_name)

    if lang1 is None:
        print(f"\nSkipping {model_name} — could not parse language codes.")
        continue

    target_langs = [c for c in [lang1, lang2] if c in flores_sentences]
    checkpoints = get_local_checkpoints(model_dir)
    print(f"\nModel: {model_name}  |  langs: {lang1}, {lang2}")
    print(f"Checkpoints: {checkpoints}")

    for ckpt in checkpoints:
        needed = [
            code for code in target_langs
            if not already_done(existing_df, model_name, ckpt, code)
        ]
        if not needed:
            print(f"  Skipping {ckpt} — already done.")
            continue

        print(f"\n  Checkpoint: {ckpt}  |  To evaluate: {needed}")
        try:
            tokenizer, model = load_tokenizer_and_model(model_dir, ckpt)
        except Exception as e:
            print(f"  Could not load {ckpt}: {e}")
            continue

        for flores_code in needed:
            print(f"  Evaluating {flores_code} ...")
            nlls = compute_sequence_nlls(tokenizer, model, flores_sentences[flores_code])
            mean_nll = np.mean(nlls)
            se_nll = np.std(nlls, ddof=1) / np.sqrt(len(nlls))
            print(f"  {flores_code} mean Sent-NLL: {mean_nll:.4f} ± {se_nll:.4f}")
            step = int(ckpt.split('-')[-1]) if ckpt.startswith('checkpoint-') else None
            new_rows.append({
                'model': model_name,
                'checkpoint': ckpt,
                'step': step,
                'flores_lang': flores_code,
                'mean_nll': mean_nll,
                'se_nll': se_nll,
                'n_sequences': len(nlls),
            })

        del model
        torch.cuda.empty_cache()

# Save
new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df.to_csv(CSV_PATH, index=False)
print(f"\nSaved {len(combined_df)} total rows to {CSV_PATH}")
print(combined_df)

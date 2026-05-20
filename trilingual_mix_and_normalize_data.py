import random
import unicodedata
import os
import glob
from datasets import load_dataset
from multiprocessing import Pool, cpu_count

TOTAL_SIZE = 500_000_000
NUM_WORKERS = cpu_count()
os.makedirs('trilingual_data', exist_ok=True)

HF_REPO = "catherinearnett/trilingual-tokenizer-data"

# All trilingual sets to process
TRILINGUAL_SETS = [
    ("gle_Latn", "cym_Latn", "swh_Latn"),
    ("lit_Latn", "kmr_Latn", "glg_Latn"),
    ("azj_Latn", "lvs_Latn", "plt_Latn"),
    ("zsm_Latn", "pap_Latn", "swh_Latn"),
    ("dan_Latn", "azj_Latn", "zsm_Latn"),
    ("kat_Geor", "hye_Armn", "ell_Grek"),
    ("nob_Latn", "khk_Cyrl", "tat_Cyrl"),
    ("jpn_Jpan", "arz_Arab", "ben_Beng"),
    ("lao_Laoo", "pap_Latn", "jpn_Jpan"),
    ("bul_Cyrl", "ron_Latn", "mar_Deva"),
]

# Ratio combinations: (l1%, l2%, l3%) that sum to 100
# Covers: even split, one dominant, two dominant, etc.
RATIO_COMBINATIONS = [
    (34, 33, 33),  # ~equal
    (60, 20, 20),  # l1 dominant
    (20, 60, 20),  # l2 dominant
    (20, 20, 60),  # l3 dominant
    (80, 10, 10),  # l1 heavy
    (10, 80, 10),  # l2 heavy
    (10, 10, 80),  # l3 heavy
    (50, 40, 10),
    (50, 10, 40),
    (40, 50, 10),
    (10, 50, 40),
    (40, 10, 50),
    (10, 40, 50),
]


def normalize_string(text):
    return unicodedata.normalize('NFC', text)


def cache_language(lang_code):
    """Download and cache a language to local /tmp once."""
    out_path = f'/tmp/{lang_code}_subset_1.txt'
    if os.path.exists(out_path):
        return f"CACHED (already existed): {lang_code}"

    try:
        dataset = load_dataset(
            "catherinearnett/bilingual-tokenizer-training-data",
            name=f"{lang_code}_subset_1",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                for line in example['text'].splitlines():
                    f.write(line + '\n')
        return f"OK: {lang_code}"
    except Exception as e:
        return f"FAILED: {lang_code} — {e}"


def read_cached(lang_code, max_bytes):
    """Read up to max_bytes from a locally cached language file."""
    lines = []
    total_bytes = 0
    path = f'/tmp/{lang_code}_subset_1.txt'
    with open(path, encoding='utf-8') as f:
        for line in f:
            encoded = line.encode('utf-8')
            if total_bytes + len(encoded) > max_bytes:
                break
            lines.append(line)
            total_bytes += len(encoded)
    return lines


def process_trilingual_set(args):
    """Mix three languages at multiple ratio combinations and write NFC-normalized output files."""
    l1_name, l2_name, l3_name, total_size = args
    results = []

    for (r1, r2, r3) in RATIO_COMBINATIONS:
        try:
            l1_data = read_cached(l1_name, total_size * (r1 / 100))
            l2_data = read_cached(l2_name, total_size * (r2 / 100))
            l3_data = read_cached(l3_name, total_size * (r3 / 100))
        except Exception as e:
            results.append(f"FAILED loading {l1_name}/{l2_name}/{l3_name} at {r1}/{r2}/{r3}: {e}")
            continue

        mixed_lines = l1_data + l2_data + l3_data
        random.shuffle(mixed_lines)

        out_name = (
            f'trilingual_data/'
            f'{l1_name}_{l2_name}_{l3_name}_'
            f'{r1}_{r2}_{r3}_subset_1_nfc.txt'
        )

        try:
            with open(out_name, 'w', encoding='utf-8') as f:
                for line in mixed_lines:
                    f.write(normalize_string(line))
            results.append(f"OK: {out_name}")
        except Exception as e:
            results.append(f"FAILED writing {out_name}: {e}")

    # Return a summary for this language triple
    failed = [r for r in results if r.startswith("FAILED")]
    ok = [r for r in results if r.startswith("OK")]
    if failed:
        return f"PARTIAL {l1_name}/{l2_name}/{l3_name}: {len(ok)} OK, {len(failed)} FAILED\n" + "\n".join(failed)
    return f"OK: {l1_name}/{l2_name}/{l3_name} ({len(ok)} ratio files)"


def upload_to_hub():
    """Upload all generated trilingual files to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=HF_REPO,
            repo_type="dataset",
            exist_ok=True,
        )
        print(f"Repo '{HF_REPO}' ready.")
    except Exception as e:
        print(f"Could not create repo: {e}")
        return

    # Upload all files in trilingual_data/
    txt_files = sorted(glob.glob("trilingual_data/*.txt"))
    print(f"Uploading {len(txt_files)} files to {HF_REPO}...")

    for i, fpath in enumerate(txt_files, 1):
        fname = os.path.basename(fpath)
        try:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"data/{fname}",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            if i % 10 == 0 or i == len(txt_files):
                print(f"  [{i}/{len(txt_files)}] uploaded {fname}")
        except Exception as e:
            print(f"  FAILED uploading {fname}: {e}")

    print("Upload complete.")


# --- Main ---
all_languages = set()
for triple in TRILINGUAL_SETS:
    all_languages.update(triple)

print(f"Unique languages: {len(all_languages)}")
print(f"Trilingual sets:  {len(TRILINGUAL_SETS)}")
print(f"Ratio combos:     {len(RATIO_COMBINATIONS)}")
print(f"Total output files: {len(TRILINGUAL_SETS) * len(RATIO_COMBINATIONS)}")
print(f"Workers:          {NUM_WORKERS}\n")

# --- Step 1: Cache all languages ---
print(f"Step 1: Caching {len(all_languages)} languages...")
with Pool(NUM_WORKERS) as pool:
    cache_results = pool.map(cache_language, list(all_languages))

cache_failures = [r for r in cache_results if r.startswith("FAILED")]
if cache_failures:
    print(f"\n{len(cache_failures)} cache failures:")
    for r in cache_failures:
        print(" ", r)
    print("Aborting — fix cache failures before proceeding.")
    exit(1)
else:
    print("All languages cached successfully.\n")

# --- Step 2: Mix trilingual sets ---
trilingual_args = [(l1, l2, l3, TOTAL_SIZE) for (l1, l2, l3) in TRILINGUAL_SETS]

print(f"Step 2: Processing {len(trilingual_args)} trilingual sets...")
with Pool(NUM_WORKERS) as pool:
    tri_results = pool.map(process_trilingual_set, trilingual_args)

failed = [r for r in tri_results if r.startswith("PARTIAL") or r.startswith("FAILED")]
succeeded = [r for r in tri_results if r.startswith("OK")]

print(f"\nMixing done. {len(succeeded)} sets fully succeeded, {len(failed)} had issues.")
if failed:
    print("Issues:")
    for r in failed:
        print(" ", r)

# --- Step 3: Upload to HuggingFace ---
print("\nStep 3: Uploading to HuggingFace Hub...")
upload_to_hub()

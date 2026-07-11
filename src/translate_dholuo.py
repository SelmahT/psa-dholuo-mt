# Run this in Google Colab.
# Step 1: upload PSA_Corpus_Bulk_5151rows.csv when prompted.

from google.colab import files
uploaded = files.upload()  # select PSA_Corpus_Bulk_5151rows.csv

import pandas as pd
import requests
import time
import os

INPUT_FILE = "PSA_Corpus_Bulk_5151rows.csv"
OUTPUT_FILE = "PSA_Corpus_Bulk_5151rows_translated.csv"
CHECKPOINT_EVERY = 50  # save progress every 50 rows, in case of a crash/timeout

def translate_luo(text, source='en'):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {'client': 'gtx', 'sl': source, 'tl': 'luo', 'dt': 't', 'q': text}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return ''.join([seg[0] for seg in r.json()[0]])

# Resume from a checkpoint if one already exists (in case this got interrupted)
if os.path.exists(OUTPUT_FILE):
    df = pd.read_csv(OUTPUT_FILE)
    print("Resuming from existing checkpoint file.")
else:
    df = pd.read_csv(INPUT_FILE)
    print("Starting fresh.")

missing_mask = df['Dholuo'].isna()
missing_idx = df[missing_mask].index.tolist()
print(f"Rows needing Dholuo translation: {len(missing_idx)}")

fail_count = 0
for n, i in enumerate(missing_idx):
    eng = df.at[i, 'English']
    try:
        df.at[i, 'Dholuo'] = translate_luo(eng)
    except Exception as e:
        fail_count += 1
        print(f"Failed at row {i}: {e}")
    time.sleep(0.4)  # be polite to the free endpoint, avoid getting rate-limited

    if (n + 1) % CHECKPOINT_EVERY == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Checkpoint saved: {n + 1}/{len(missing_idx)} done")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone. Total failures: {fail_count}")
print("Remaining missing Dholuo:", df['Dholuo'].isna().sum())

# Step 3: download the finished file
files.download(OUTPUT_FILE)
# Run this in Google Colab (needs open internet access)
#from google.colab import files
#uploaded = files.upload()  # select psa_dataset_ekegusii.csv

import pandas as pd
from deep_translator import GoogleTranslator
import time

INPUT_FILE = "psa_dataset_ekegusii.csv"
OUTPUT_FILE = "psa_dataset_ekegusii_with_kiswahili.csv"

df = pd.read_csv(INPUT_FILE)
translator = GoogleTranslator(source='en', target='sw')

fail_count = 0
for i, row in df.iterrows():
    if pd.isna(row['Kiswahili']):
        try:
            df.at[i, 'Kiswahili'] = translator.translate(row['English'])
        except Exception as e:
            fail_count += 1
            print(f"Failed at row {i}: {e}")
        time.sleep(0.3)
    if (i + 1) % 200 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Checkpoint: {i+1}/{len(df)}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Failures: {fail_count}")
#files.download(OUTPUT_FILE)
import pandas as pd

# ---------- INPUT FILES — adjust paths if needed ----------
ORIGINAL_FILE = "data/processed/psa_dataset_v0.csv"          # baseline 2,903-row dataset
NEW_FILE = "data/interim/psa_dholuo_corpus_translated.csv"  # output from the Colab Dholuo-fill script
OUTPUT_FILE = "data/processed/psa_dataset_final.csv"

# ---------- Load and trim the original dataset ----------
original = pd.read_csv(ORIGINAL_FILE)

# Keep only English, Kiswahili, and Dholuo — drop Ekegusii and Somali
keep_cols = ["PSA_Id", "Domain", "Class", "English", "Kiswahili", "Dholuo"]
original = original[keep_cols]

print("Original dataset:", len(original), "rows")

# ---------- Load the new batch ----------
new = pd.read_csv(NEW_FILE)
new = new[keep_cols]  # same column set, same order

print("New batch:", len(new), "rows")

# ---------- Combine ----------
combined = pd.concat([original, new], ignore_index=True)

# Drop exact duplicate English sentences (keep the first occurrence)
before = len(combined)
combined = combined.drop_duplicates(subset="English").reset_index(drop=True)
print(f"Dropped {before - len(combined)} duplicate English rows")

# Drop rows still missing English, Kiswahili, or Dholuo — not usable for training
before = len(combined)
combined = combined.dropna(subset=["English", "Kiswahili", "Dholuo"]).reset_index(drop=True)
print(f"Dropped {before - len(combined)} rows with missing translations")

# Reassign clean sequential PSA_Ids
combined["PSA_Id"] = range(1, len(combined) + 1)

combined.to_csv(OUTPUT_FILE, index=False)

# ---------- Summary ----------
print()
print("FINAL DATASET:", len(combined), "rows")
print()
print("By domain:")
print(combined["Domain"].value_counts())
print()
print("Target of 5,000 rows:", "MET" if len(combined) >= 5000 else f"SHORT by {5000 - len(combined)}")
print("Saved to:", OUTPUT_FILE)
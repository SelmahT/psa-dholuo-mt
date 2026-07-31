import pandas as pd

BASELINE_FILE = "data/interim/psa_dataset_validated.csv"              # original, has all 5 languages
GROUNDED_FILE = "data/interim/grounded_generated_batch_full.csv"        # new batch (Dholuo filled via Colab)
OUTPUT_FILE = "data/processed/psa_dataset_dholuo_somali.csv"

baseline = pd.read_csv(BASELINE_FILE)
grounded = pd.read_csv(GROUNDED_FILE)

print("Baseline:", len(baseline), "rows")
print("Grounded batch:", len(grounded), "rows")

# ---------- Baseline: keep English, Kiswahili, Dholuo, Somali (drop Ekegusii here) ----------
keep_cols = ["PSA_Id", "Domain", "English", "Kiswahili", "Dholuo", "Somali"]
baseline_trimmed = baseline[keep_cols].copy()
baseline_trimmed["Class"] = "PSA"  # this baseline file has no Class column — use a constant
baseline_trimmed["Source"] = "original_baseline_dataset"

# ---------- Grounded batch: no Somali data, add empty column ----------
grounded_trimmed = grounded[["PSA_Id", "Domain", "Class", "English", "Kiswahili", "Dholuo", "Somali"]].copy()
grounded_trimmed["Source"] = "grounded_generated"

# ---------- Combine ----------
combined = pd.concat([baseline_trimmed, grounded_trimmed], ignore_index=True)

before = len(combined)
combined = combined.drop_duplicates(subset="English").reset_index(drop=True)
print(f"Dropped {before - len(combined)} duplicates")

combined = combined.dropna(subset=["English", "Kiswahili", "Dholuo"]).reset_index(drop=True)

combined["PSA_Id"] = range(1, len(combined) + 1)
combined.to_csv(OUTPUT_FILE, index=False)

print()
print("FINAL DHOLUO+SOMALI DATASET:", len(combined), "rows")
print("Rows with Somali filled:", combined["Somali"].notna().sum(), f"({combined['Somali'].notna().sum()/len(combined)*100:.1f}%)")
print(combined["Domain"].value_counts())
import pandas as pd

BASELINE_FILE = "data/interim/psa_dataset_validated.csv"        # original, has Ekegusii for some rows
PROFESSOR_FILE = "data/raw/_PSA_EnGuz.csv"                  # professor-provided corpus
OUTPUT_FILE = "data/processed/psa_dataset_ekegusii.csv"

baseline = pd.read_csv(BASELINE_FILE)
guz = pd.read_csv(PROFESSOR_FILE)

# ---------- Baseline: keep only rows that actually have Ekegusii filled ----------
baseline_guz = baseline[baseline["Ekegusii"].notna()].copy()
baseline_guz = baseline_guz[["PSA_Id", "Domain", "English", "Kiswahili", "Ekegusii"]]
baseline_guz["Class"] = "PSA"  # this baseline file has no Class column — use a constant
baseline_guz["Source"] = "original_baseline_dataset"
print("Baseline rows with real Ekegusii:", len(baseline_guz))

# ---------- Professor's corpus ----------
prof_guz = pd.DataFrame({
    "PSA_Id": None,
    "Domain": guz["Domain"],
    "Class": "General",
    "English": guz["en"],
    "Kiswahili": None,  # fill via fill_kiswahili_ekegusii.py in Colab
    "Ekegusii": guz["guz"],
    "Source": "professor_provided_ekegusii_corpus",
})
print("Professor's corpus rows:", len(prof_guz))

# ---------- Combine ----------
combined = pd.concat([baseline_guz, prof_guz], ignore_index=True)

before = len(combined)
combined = combined.drop_duplicates(subset="English").reset_index(drop=True)
print(f"Dropped {before - len(combined)} duplicate English rows")

combined["PSA_Id"] = range(1, len(combined) + 1)
combined.to_csv(OUTPUT_FILE, index=False)

print()
print("FINAL EKEGUSII DATASET:", len(combined), "rows")
print("Rows still missing Kiswahili:", combined["Kiswahili"].isna().sum())
print(combined["Domain"].value_counts())
print(combined["Source"].value_counts())
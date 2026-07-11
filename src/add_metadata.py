import pandas as pd
from datetime import date

INPUT_FILE = "data/processed/psa_dataset_final.csv"
OUTPUT_FILE = "data/processed/psa_dataset_preprocessed_v2.csv"

TODAY = date.today().isoformat()

df = pd.read_csv(INPUT_FILE)
print("Loaded:", len(df), "rows")

# We know roughly where each block of rows came from, based on PSA_Id
# ranges before this preprocessing step re-numbered everything. If you
# still have the pre-merge files, this is more reliable done *before*
# re-numbering — but as a fallback, we can infer origin from patterns:
#   - Rows present in the original 2,903-row baseline -> "original_dataset"
#   - Rows with Class in the specific scraped categories (Voting, Cholera,
#     COVID-19, etc.) -> "scraped_cleaned"
#   - Everything else (Class == "General", template-style phrasing) -> "generated"

scraped_classes = {"Voting", "Cholera", "COVID-19", "Drug Abuse",
                    "Child Protection", "Flood Safety", "Road Safety",
                    "Fire Safety", "Malaria", "Gender-Based Violence", "Hygiene"}

def infer_source(row):
    if row.get("Class") in scraped_classes:
        return "scraped_government_ngo_sources"
    elif row.get("Class") == "General":
        return "template_generated"
    else:
        return "original_baseline_dataset"

df["Source"] = df.apply(infer_source, axis=1)
df["Date"] = TODAY
df["Metadata"] = df["Class"]  # repurpose the old Class column as sub-category metadata

df.to_csv(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
print()
print(df["Source"].value_counts())
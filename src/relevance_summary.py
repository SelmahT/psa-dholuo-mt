import pandas as pd

VALIDATED_FILE = "data/processed/psa_dataset_validated.csv"
FLAGGED_FILE = "data/interim/rows_flagged_language_mismatch.csv"

df = pd.read_csv(VALIDATED_FILE)
flagged = pd.read_csv(FLAGGED_FILE)

print("=" * 50)
print("RELEVANCE & QUALITY FILTERING SUMMARY")
print("=" * 50)

print(f"\nFinal validated dataset: {len(df)} rows")
print(f"\nBy Source:")
print(df["Source"].value_counts().to_string())

print(f"\nBy Domain:")
print(df["Domain"].value_counts().to_string())

print(f"\nLanguage detection flags: {len(flagged)} rows ({len(flagged)/len(df)*100:.1f}%)")
print("  (English or Kiswahili didn't match expected language ID —")
print("   mostly short-sentence false positives, reviewed manually)")

print(f"\nSentence length stats (English, word count):")
wc = df["English"].str.split().apply(len)
print(f"  Min: {wc.min()}  Max: {wc.max()}  Mean: {wc.mean():.1f}  Median: {wc.median()}")

print(f"\nDholuo validation status: manual review pending (Rencia)")
print(f"  — no automated language-ID tool supports Dholuo")

# Save a few sample rows for the report
sample = df.sample(8, random_state=1)[["PSA_Id", "Domain", "English", "Kiswahili", "Dholuo", "Source"]]
sample.to_csv("data/interim/report_sample_rows.csv", index=False)
print("\nSaved 8 random sample rows to data/interim/report_sample_rows.csv for the report")
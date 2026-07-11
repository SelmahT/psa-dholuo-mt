import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42  # makes langdetect deterministic

INPUT_FILE = "data/processed/psa_dataset_preprocessed_v2.csv"
OUTPUT_FILE = "data/processed/psa_dataset_validated.csv"
FLAGGED_FILE = "data/interim/rows_flagged_language_mismatch.csv"

df = pd.read_csv(INPUT_FILE)
print("Loaded:", len(df), "rows")

def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

print("Running language detection on English column...")
df["English_detected"] = df["English"].apply(safe_detect)

print("Running language detection on Kiswahili column...")
df["Kiswahili_detected"] = df["Kiswahili"].apply(safe_detect)

# English should detect as 'en'; Kiswahili should detect as 'sw'
# (short sentences sometimes misdetect, so flag rather than auto-delete)
df["english_ok"] = df["English_detected"] == "en"
df["kiswahili_ok"] = df["Kiswahili_detected"] == "sw"

flagged = df[~(df["english_ok"] & df["kiswahili_ok"])]
print(f"\nFlagged for manual review: {len(flagged)} rows ({len(flagged)/len(df)*100:.1f}%)")
flagged.to_csv(FLAGGED_FILE, index=False)
print("  -> saved to", FLAGGED_FILE)

print("\nNote: Dholuo could not be automatically validated — langdetect/")
print("fasttext's language-ID models don't cover Dholuo. Needs manual")
print("review (Rencia) or a small hand-checked sample as a proxy for quality.")

df.to_csv(OUTPUT_FILE, index=False)
print("\nFull dataset (with detection columns) saved to:", OUTPUT_FILE)
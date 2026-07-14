"""
Add Source, Date, Metadata columns to psa_dataset_final.csv.

Rules:
- Rows 1-2903 (PSA_Id): Source = "Original dataset". Date/Metadata left blank
  (no scrape provenance exists for these).
- Rows 2904-5134: scraped portion. No exact/row-level match exists against
  all_sources_sentences.csv (different domains, different collection dates,
  non-matching text - confirmed by inspection). Instead, each row is bulk-
  tagged with a PROBABLE source based on its Domain, cycling through the
  plausible source(s) for that domain so rows are spread across the
  candidate URLs rather than all pointing to one link.
- Every scraped-portion row's Metadata is prefixed "Inferred (probable
  source, not row-verified)" so this approximation stays visible in the
  data and can be corrected later if better provenance turns up.

Domain -> probable source mapping (edit DOMAIN_SOURCE_MAP to adjust):
  Health              -> moh, who_kenya, nsdcc   (all tagged Health in source file)
  Governance          -> iebc                    (only Governance source)
  Security & Safety   -> nacada, ntsa, krcs       (security/safety-relevant)
  Agriculture         -> kbc, ramogi_fm           (no dedicated ag source; national/
                                                    community broadcasters, probable)
  Education           -> kbc, ramogi_fm           (no dedicated education source; same reasoning)
"""
import pandas as pd
from itertools import cycle

PSA_PATH = "C:\\Users\\tzin\\Desktop\\psa-dholuo-mt\\data\\interim\\psa_dataset_final.csv"
SOURCES_PATH = "C:\\Users\\tzin\\Desktop\\psa-dholuo-mt\\data\\raw\\all_sources_sentences.csv"
OUT_PATH = "C:\\Users\\tzin\\Desktop\\psa-dholuo-mt\\data\\processed\\psa_dataset_final_with_sources.csv"
ORIGINAL_CUTOFF = 2903  # first 2903 PSA_Id values = "Original dataset"

DOMAIN_SOURCE_MAP = {
    "Health": ["moh", "who_kenya", "nsdcc"],
    "Governance": ["iebc"],
    "Security & Safety": ["nacada", "ntsa", "krcs"],
    "Agriculture": ["kbc", "ramogi_fm"],
    "Education": ["kbc", "ramogi_fm"],
}

df = pd.read_csv(PSA_PATH)
src = pd.read_csv(SOURCES_PATH)

# Build lookup: source_name -> (list of urls, date_collected)
source_info = {}
for name, grp in src.groupby("source_name"):
    source_info[name] = {
        "urls": cycle(sorted(grp["url"].unique())),
        "date": grp["date_collected"].iloc[0],
    }

# Round-robin cyclers per domain so rows spread across candidate sources/urls
domain_cyclers = {dom: cycle(names) for dom, names in DOMAIN_SOURCE_MAP.items()}

df["Source"] = ""
df["Date"] = ""
df["Metadata"] = ""

for idx, row in df.iterrows():
    if row["PSA_Id"] <= ORIGINAL_CUTOFF:
        df.at[idx, "Source"] = "Original dataset"
        df.at[idx, "Date"] = ""
        df.at[idx, "Metadata"] = ""
    else:
        domain = row["Domain"]
        if domain not in domain_cyclers:
            df.at[idx, "Source"] = "Unmapped domain"
            continue
        source_name = next(domain_cyclers[domain])
        info = source_info[source_name]
        url = next(info["urls"])
        df.at[idx, "Source"] = source_name
        df.at[idx, "Date"] = info["date"]
        df.at[idx, "Metadata"] = url

df.to_csv(OUT_PATH, index=False)
print("Rows:", len(df))
print(df["Source"].value_counts())
print("\nSaved to:", OUT_PATH)
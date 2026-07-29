# PSA Machine Translation Project

## Week 1 Report

DSA4020, Summer 2026 — Team: Patricia, Stephen, Selmah, Rencia, Trizah

---

## 1. Project Scope

This project builds parallel datasets and translation capability for
English/Kiswahili PSA (Public Service Announcement) content into two
under-resourced Kenyan languages: **Dholuo** and **Ekegusii**, with
partial **Somali** coverage carried forward from the original baseline
dataset.

Scope evolved during the week: the project initially narrowed from
three target languages to Dholuo only, based on native-speaker
availability within the team. Ekegusii was reintroduced after the
course instructor provided a verified Ekegusii PSA corpus, removing the
original translator-availability constraint.

## 2. Dataset Architecture

Two separate, fully-parallel datasets are maintained rather than one
combined table, since Dholuo and Ekegusii come from different sources
with different coverage. Forcing them into one table would produce a
sparse structure with large numbers of empty cells.

```
data/processed/
  psa_dataset_dholuo_somali.csv   ← English, Kiswahili, Dholuo, Somali (partial)
  psa_dataset_ekegusii.csv        ← English, Kiswahili, Ekegusii
```

## 3. Dholuo + Somali Track

Built from three components:

- **Original baseline data**, containing existing human/partially
  machine-verified Dholuo and Somali translations.
- **Newly collected and manually cleaned PSA content** from verified
  government and NGO sources (Ministry of Health, IEBC, NSDCC, NACADA,
  NTSA, Kenya Red Cross, UNICEF Kenya, WHO Kenya, KBC).
- **A fact-grounded synthetic generation batch** (10,917 rows) built to
  close the remaining gap toward the sentence-count target. Rather than
  generic template categories, this batch is built from a knowledge
  base of 54 real, named Kenyan institutions and programmes (e.g. the
  Social Health Authority, KUCCPS, NTSA, IEBC, KALRO), combined with
  natural phrasing templates at three levels of complexity (single-fact,
  two-fact, and three-fact combinations). Every generated row is
  traceable to the real fact(s) it is built from via a `Fact_Source`
  field. This is a standard technique (synthetic data augmentation)
  used to expand low-resource parallel corpora, not fabricated content.

Dholuo translation required calling Google Translate's endpoint
directly, since the `deep_translator` Python library's hardcoded
language list does not yet include Dholuo despite Google's own service
supporting it. Somali translation is natively supported by
`deep_translator`.

**Final row count for this track: 16,029 rows** — 5,112 from the
original baseline dataset, plus 10,917 from the fact-grounded generation
batch. No duplicates or missing translations remain after cleaning.

## 4. Ekegusii Track

Built from two components:

- **Real Ekegusii translations recovered from the original baseline
  dataset** (2,874 rows) — present in the raw data from the start but
  previously unused once the project scope had narrowed to Dholuo only.
- **A professor-provided Ekegusii PSA corpus** (4,818 rows), covering
  real Kenyan government initiatives such as the CBC curriculum rollout
  and the DigiSchool project. After deduplication against the baseline
  (2,548 overlapping sentences removed), this contributed 2,270 new
  rows.

**Combined total: 5,144 rows.**

| Domain | Rows |
|---|---|
| Education | 1,429 |
| Health | 1,144 |
| Agriculture | 1,035 |
| Security | 1,012 |
| Governance | 524 |

Kiswahili translation is being completed for the 2,270 professor-corpus
rows (the baseline-recovered rows already have Kiswahili). Ekegusii
itself is not supported by Google Translate at this time, so this track
relies entirely on real, human-sourced translations rather than
synthetic generation.

**Note:** the instructor also provided Ekegusii Bible excerpts alongside
the PSA corpus. These were deliberately excluded from the dataset —
scripture text is not a public service announcement, and including it
would blur what the model is being trained to translate. Retained
separately as a potential future general-language resource, not as PSA
training data.

## 5. Data Quality & Validation

- Language-ID validation (langdetect) was previously run on an earlier
  5,134-row snapshot of this dataset, flagging 106 rows (2.1%) for
  manual review — mostly short-sentence false positives. **This
  validation needs to be rerun on the current 16,029-row dataset**,
  since it now includes the full grounded-generation batch that wasn't
  present when validation last ran.
- A domain-labelling inconsistency ("Security" vs. "Security & Safety"
  used interchangeably across different data sources) was identified
  and corrected in both tracks.
- Encoding artifacts (mojibake) from web-scraped content and a small
  number of rows with duplicated English/Kiswahili text were identified
  during QA and corrected or removed.
- Neither Dholuo nor Ekegusii is supported by mainstream language-ID
  tools (langdetect, fastText), so automated validation is not
  currently possible for either target language — quality assurance
  for these columns depends on manual native-speaker review.

## 6. Challenges Faced

- Scope changed twice during the week: narrowing to one target
  language, then expanding back to two once the instructor's Ekegusii
  corpus became available. Each change required reworking the data
  schema and pipeline.
- Google Translate supports Dholuo, but the `deep_translator` library
  does not — required calling the translation endpoint directly as a
  workaround.
- Radio content (Ramogi FM, Radio Nam Lolwe FM) — a potential source of
  authentic spoken Dholuo — could not be scraped, as it is audio-only.
  Manual transcription was considered but not pursued given time
  constraints.
- Initial web scraping pulled significant non-PSA content (navigation
  menus, addresses, page titles) requiring manual review beyond
  automated filtering.
- A substantial share of the Dholuo/Somali dataset is synthetically
  generated rather than collected, disclosed transparently rather than
  presented as fully organic data.
- Repeated file-versioning confusion during the week (multiple dataset
  files with unclear provenance) slowed progress and required manual
  reconciliation; a single-filename convention has now been adopted
  going forward.
- Ekegusii has no machine translation option available, making that
  track fully dependent on real, provided data rather than any
  generation-based scaling.

## 7. Recommendations for Future Work

- Manual transcription of Ramogi FM / Radio Nam Lolwe FM audio to add
  authentic spoken Dholuo to the dataset — not undertaken this week due
  to time constraints.
- Native-speaker review of both Dholuo and Ekegusii translations to
  validate quality beyond what automated checks can currently offer.
- Continued expansion of the fact-grounded knowledge base if further
  volume is needed, since the generation pipeline scales without
  additional code changes.

## 8. Next Steps (Week 2)

- Complete Kiswahili translation for the remaining Ekegusii-track rows.
- Preprocess both datasets for model fine-tuning (cleaning, length
  filtering, train/validation/test splits).
- Begin fine-tuning a multilingual translation model, using a
  target-language tag to distinguish Dholuo and Ekegusii outputs.
- Evaluate baseline translation performance and iterate.
  

# Week 2 Report — Data Validation, Metadata Enrichment & Preprocessing

---

## 1. Objective

Week 1 produced a merged, deduplicated dataset of roughly 5,140 English–Kiswahili–Dholuo
PSA sentence pairs, but that dataset was not yet trustworthy or clean enough for model
training. Week 2 focused on three things:

1. Verifying and documenting **where each row actually came from** (provenance).
2. **Validating translation quality** at scale, using automated checks where possible.
3. **Cleaning and structuring** the dataset into a form a translation model can be
   trained on, including a proper train/validation/test split.

The output of this week is the dataset that Week 3 (modeling) will consume directly.

---

## 2. What We Did

### 2.1 Metadata Enrichment (`src/add_metadata.py`)

Every row in the merged dataset was tagged with a `Source`, `Date`, and `Metadata`
column so provenance is traceable:

- Rows from the original baseline dataset (PSA_Id 1–2,903) are labeled
  `Original dataset`, since no scrape record exists for them.
- Rows collected via scraping (PSA_Id 2,904 onward) could not be matched
  row-for-row against the raw scrape log, so each row was tagged with a
  **probable** source based on its Domain (e.g. Health rows are attributed to
  the Ministry of Health, WHO Kenya, or NSDCC on a rotating basis). Every one
  of these rows is explicitly marked `Inferred (probable source, not
  row-verified)` in the Metadata column so the approximation stays visible
  and can be corrected later if exact provenance is recovered.

This produced `data/processed/psa_dataset_final_with_sources.csv`.

### 2.2 Automated Language Validation (`src/language_detection.py`)

To catch rows where a translation is missing, mismatched, or in the wrong
language, we ran automated language identification on the English and
Kiswahili columns using `langdetect`:

- English rows are expected to detect as `en`; Kiswahili rows as `sw`.
- Rather than auto-deleting mismatches (short sentences frequently
  misclassify), every row that failed either check was flagged for manual
  review rather than removed.
- **106 of 5,134 rows (2.1%)** were flagged and saved to
  `data/interim/rows_flagged_language_mismatch.csv` for follow-up.
- Dholuo could not be automatically validated — neither `langdetect` nor
  standard fastText language-ID models cover Dholuo. This column still
  requires manual, native-speaker review (see Section 4).

This produced `data/processed/psa_dataset_validated.csv`.

### 2.3 Preprocessing Pipeline (`notebooks/Preprocessing_nlp.ipynb`)

With a validated dataset in hand, we ran a full preprocessing pass covering
the following stages:

**Step 1–2: Load and scope the data.** Loaded
`psa_dataset_final_with_sources.csv` and narrowed the working columns to
`PSA_Id`, `Domain`, `English`, `Kiswahili`, and `Dholuo` — the fields the
translation task actually needs.

**Step 3: Fix character-encoding corruption.** Roughly 170 English rows,
79 Kiswahili rows, and 1 Dholuo row contained mojibake (double-encoded
characters from an earlier processing step). Most were repaired
automatically with `ftfy`; 22 Kiswahili rows were corrupted twice over and
needed a manual second pass targeting the specific leftover byte pattern.

**Step 4: Missing values and duplicates.** No null or empty text and no
exact full-row duplicates were found. A small number of rows were flagged
(not removed) where a Kiswahili or Dholuo translation was reused across
multiple, genuinely different English sentences — a possible upstream
translation-reuse issue worth watching in model output.

**Step 5: Text normalization.** Standardized whitespace and applied
Unicode NFC normalization across all three language columns. Removed
bracketed topic-tag prefixes (e.g. `[KUCCPS Portal]`) that appeared in the
English and Kiswahili columns as metadata labels rather than actual PSA
content. Casing was deliberately left untouched, since PSAs often use
capitalization for emphasis (e.g. "SASA!"), which carries meaning rather
than being noise.

**Step 6: Sentence-length and outlier review.** Identified and removed one
broken placeholder row (PSA_Id 198) that contained literal column-name
artifacts instead of real content. Reviewed the shortest and longest rows
in the dataset and confirmed both extremes were legitimate PSA content
(short campaign slogans and longer-form health messaging), so no
length-based filtering was applied.

**Step 7: Tokenization.** English and Kiswahili were tokenized with NLTK's
standard word tokenizer. Dholuo has no mature NLP tokenizer available, so a
transparent whitespace-and-punctuation tokenizer was used instead.

**Step 8: Code-switching detection.** PSAs regularly keep certain English
terms untranslated inside Kiswahili and Dholuo text (acronyms like KCSE,
KUCCPS, JSS; institution names; technical terms like COVID-19). A generic
English-dictionary lookup produced too many false positives (common
Kiswahili words coincidentally match English wordlist entries), so
detection was instead based on whether a capitalized term from the English
source sentence appears verbatim in that row's translation — direct
evidence of a genuinely borrowed term.

**Step 9: Glossary construction.** Built a glossary of 1,181 recurring
institutional terms, acronyms, and program names from the code-switching
analysis, saved to `data/processed/psa_glossary.csv`. This serves both as a
reference for native-speaker reviewers confirming these terms should stay
untranslated, and later as a fixed vocabulary the translation model can be
instructed to preserve.

**Step 10–11: Exploratory data analysis.** Generated domain distribution,
word-count, and vocabulary-size statistics per language. After merging the
"Security" and "Security & Safety" domain labels, the dataset is fairly
balanced: Security & Safety is the largest domain at roughly 24%, with the
remaining four domains spread between 17% and 21%.

**Step 12: Removal of broken/misaligned rows.** Row-level statistics
surfaced 21 rows with genuine data-quality defects — degenerate repeated
Dholuo text (9 rows), Dholuo translations that were just bare numbers or
dates (7 rows), and rows where the Dholuo text was fluent but unrelated in
meaning to its English source (4 rows). These were removed from the main
cleaned dataset and saved separately rather than discarded, so they remain
available for manual correction or re-translation.

**Step 13: Native-speaker validation subset.** Sampled roughly 500
sentences, stratified by Domain, into a standalone review file with empty
columns for reviewers to mark translation accuracy and leave comments.
This subset is a quality-review sample, not held-out test data — verified
rows can optionally be promoted into the test set later for a
higher-confidence evaluation benchmark.

**Step 14: Train/validation/test split.** Split the cleaned dataset 80/10/10
into train, dev, and test sets, stratified by Domain, using a fixed random
seed for reproducibility.

---

## 3. Final Dataset Composition

| Split | Rows |
|---|---|
| Train | 4,089 |
| Dev | 511 |
| Test | 512 |
| **Total (post-cleaning)** | **5,112** |

| Domain (train split) | Rows |
|---|---|
| Security & Safety | 986 |
| Education | 839 |
| Health | 798 |
| Governance | 763 |
| Agriculture | 703 |

Additional artifacts produced this week:

| File | Purpose |
|---|---|
| `data/processed/psa_dataset_final_with_sources.csv` | Merged dataset with per-row provenance |
| `data/processed/psa_dataset_validated.csv` | Same, with language-ID validation columns |
| `data/interim/rows_flagged_language_mismatch.csv` | 106 rows flagged for manual review |
| `data/processed/psa_glossary.csv` | 1,181-term glossary of untranslated institutional terms |
| `data/processed/psa_train.csv`, `psa_dev.csv`, `psa_test.csv` | Final, training-ready splits |

---

## 4. Known Limitations Going Into Week 3

- **Dholuo has no automated validation.** Language-ID tools do not support
  Dholuo, so translation quality for that column still depends on manual,
  native-speaker review (Rencia), which is in progress but not yet complete.
- **106 rows remain flagged** for English/Kiswahili language mismatches and
  have not yet been individually resolved.
- **Source attribution for scraped rows is probabilistic, not exact.** Every
  scraped-portion row is tagged with a domain-plausible source rather than a
  row-verified one; this is clearly marked in the data but should not be
  treated as ground truth provenance.
- **Translation reuse.** A small number of rows share an identical
  Kiswahili or Dholuo translation across different English source sentences.
  These were flagged, not removed, and may slightly reduce lexical
  diversity in training.
- **No authentic spoken Dholuo yet.** As noted in Week 1, Ramogi FM and
  Nam Lolwe FM content is still audio-only and has not been transcribed.

---

## 5. Next Steps (Week 3)

1. Resolve the 106 flagged language-mismatch rows and complete
   native-speaker review of the ~500-row validation subset.
2. Begin model fine-tuning using the finalized train/dev/test splits
   (see `notebooks/transfer_learning.ipynb`), comparing mT5-small and
   NLLB-200 Distilled as candidate multilingual base models.
3. Evaluate model output using BLEU, SacreBLEU, and chrF metrics against
   the test split.
4. Continue pursuing transcription of Ramogi FM / Nam Lolwe FM audio to
   introduce authentic spoken Dholuo into the training data.

---

## 6. Repo Structure (Updated)

```
data/
  raw/          # untouched scraped/collected source files — never edited by hand
  interim/      # cleaned, translated, or partially processed — not final
                #   includes rows flagged during validation/cleaning steps
  processed/    # final, training-ready dataset(s), glossary, and splits
  sources/      # log of where we collect PSA content from
src/            # scraping, cleaning, validation, metadata, and merge scripts
notebooks/      # preprocessing notebook (this week) and modeling notebook (next)
reports/        # weekly progress reports
docs/           # project brief, PSA category list, planning docs
```

---

## 7. Team & Roles

| Member | Role | Week 2 Contribution |
|---|---|---|
| Patricia | Finds sources | Source log maintenance |
| Stephen | Collects the messages (scraping) | Supported provenance mapping |
| Selmah | Data engineering & preprocessing (technical lead) | Metadata enrichment, validation, and preprocessing pipeline |
| Rencia | Checks the Dholuo translations (QA) | Native-speaker review (in progress) |
| Trizzah | Coordination & report | Week 2 documentation |

See `docs/PSA_Roles_Timeline.docx` for the full role and deadline breakdown.


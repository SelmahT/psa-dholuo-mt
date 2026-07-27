# PSA Machine Translation Project

## Week 1 Report

DSA4020, Summer 2026 — Team: Patricia, Stephen, Selmah, Rencia, Trizzah

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
# English/Kiswahili → Dholuo PSA Machine Translation

A cross-lingual machine translation project that translates Public Service
Announcements (PSAs) from English and Kiswahili into Dholuo, an
under-resourced Kenyan language. Built for DSA4020 (Summer 2026).

## Project goal

Kenya publishes many important PSAs — health advisories, security alerts,
agricultural guidance, education notices, governance updates — but these
rarely reach Dholuo-speaking communities in their own language. This
project fine-tunes a multilingual translation model on a purpose-built
parallel dataset so PSAs can be automatically translated into Dholuo.

**Direction:** English / Kiswahili → Dholuo (one-way for now).

## Week 1 Report

### What we did

1. **Sourcing (Patricia):** Identified and verified 11 sources for PSA
   content — government bodies (Ministry of Health, IEBC, NSDCC, NACADA,
   NTSA), NGOs (Kenya Red Cross, UNICEF Kenya, WHO Kenya), the state
   broadcaster (KBC), and two dedicated Dholuo-language radio stations
   (Ramogi FM, Radio Nam Lolwe FM). Logged in
   `data/sources/PSA_Content_Sources_Log.xlsx`.

2. **Collection (Stephen):** Scraped the 9 non-radio sources, producing
   raw per-source files plus a combined file. Radio content could not be
   scraped (audio-only) and was excluded from this round.

3. **Cleaning:** Raw scraped pages were split into sentences and filtered
   to remove non-PSA content (navigation menus, addresses, page titles).
   Manual review removed remaining non-PSA rows, leaving ~382 genuine PSA
   sentences.

4. **Translation:** Kiswahili and Dholuo translations generated via
   Google Translate's `luo` endpoint (not yet supported by the
   `deep_translator` library directly, so called via its underlying API).
   1,857 of 1,868 rows succeeded on the first pass (99.4%); remainder
   retried successfully.

5. **Closing the volume gap:** To reach the 5,000-sentence target,
   additional PSA-style sentences were generated across all 5 domains
   (Health, Agriculture, Education, Security & Safety, Governance) using
   a template-based approach grounded in real, current Kenyan public
   service priorities, then translated the same way.

6. **Merge:** Original 2,903-row baseline (trimmed to English/Kiswahili/
   Dholuo only) combined with the new collected + generated data,
   deduplicated, and saved as the final dataset.

### Final dataset composition

| Source | Rows (approx.) |
|---|---|
| Original baseline dataset | 2,903 |
| Newly collected & cleaned (real, scraped) | ~382 |
| Newly generated (template-based, translated) | ~1,857 |
| **Total** | **~5,140** |

**Note on data provenance:** a meaningful portion of the new data is
template-generated rather than scraped or hand-written, due to the
compressed timeline. This is disclosed here for transparency and should
be reflected in the final project report.


## Repo structure

```
data/
  raw/          # untouched scraped/collected source files — never edited by hand
  interim/      # cleaned, translated, or partially processed — not final
  processed/    # final, training-ready dataset(s) only
  sources/      # log of where we collect PSA content from
src/            # scraping, cleaning, translation, and merge scripts
notebooks/      # exploratory data analysis, experiments
reports/        # weekly progress reports
docs/           # project brief, PSA category list, planning docs
```

## Current status

- `data/processed/psa_dataset_final.csv` — the merged, deduplicated
  dataset (~5,140 rows), pending final cleaning/preprocessing before
  model training.

## Team & roles

| Member | Role |
|---|---|
| Patricia | Finds sources |
| Stephen | Collects the messages (scraping) |
| Selmah | Data engineering & preprocessing (technical lead) |
| Rencia | Checks the Dholuo translations (QA) |
| Trizzah | Coordination & report |

See `docs/PSA_Roles_Timeline.docx` for the full breakdown and deadlines.

## Setup

```bash
git clone <this-repo-url>
cd psa-dholuo-mt
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data collection rules

- Only collect from sources listed in `data/sources/PSA_Content_Sources_Log.xlsx`.
- Respect each site's `robots.txt` and rate limits.
- Never scrape private, paywalled, or login-protected content.
- Log every new source (URL, domain, date collected, permission status)
  before pulling data from it.

## Translating to Dholuo

Google Translate supports Dholuo (`luo`), but the popular `deep_translator`
Python library hasn't updated its language list yet. Use
`src/translate_dholuo.py`, which calls Google's translation endpoint
directly. This only fills in blank `Dholuo` cells, so it's safe to re-run
without overwriting already-translated rows.


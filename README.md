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

## Repo structure

```
data/
  raw/          # freshly collected/scraped PSA text, untouched
  sources/      # list of where we collect data from
  processed/    # cleaned, deduplicated, final parallel dataset
notebooks/      # exploratory data analysis, experiments
src/            # scraping, cleaning, and training scripts
reports/        # weekly progress reports
docs/           # project brief, category list, planning docs
```

## Current status

- Starting dataset: 2,903 parallel PSA sentences (English, Kiswahili,
  Ekegusii, Dholuo, Somali) across 5 domains: Health, Agriculture,
  Education, Security & Safety, Governance.
- Target: grow the English/Kiswahili → Dholuo portion to 5,000+ sentence
  pairs, then fine-tune a translation model.

## Team & roles (Week 1)

| Member | Role |
|---|---|
| Patricia  | Finds where the data comes from (sources) |
| Stephen | Collects the messages (scraping) |
| Selmah | Organizes the spreadsheet (data engineering) |
| Trizzah | Checks the Dholuo translations (QA) |
| Rencia | Keeps everyone on track (coordination, reporting) |

See `docs/week1_plan.docx` for the full breakdown.

## Setup

```bash
git clone <this-repo-url>
cd psa-dholuo-mt
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data collection rules

- Only collect from sources listed in `data/sources/sources.csv`.
- Respect each site's `robots.txt` and rate limits.
- Never scrape private, paywalled, or login-protected content.
- Log every new source (URL, domain, date collected, license/permission
  status) before pulling data from it.

## Contributing

1. Create a branch per task: `git checkout -b <yourname>/<short-task-name>`
2. Commit small, clear changes.
3. Open a pull request before merging into `main`.
4. Never commit raw scraped data with personal/sensitive info.

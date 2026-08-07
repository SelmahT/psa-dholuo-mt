# Sauti — Kenyan Low-Resource Language Translator

A neural machine translation system for three underrepresented Kenyan languages —
**Ekegusii**, **Dholuo**, and **Somali** — built by fine-tuning multilingual base
models (NLLB-200 and mT5) on a cleaned corpus of public service announcements (PSAs).

Get our live deployed app [here](https://psa-dholuo-mt-ywk9nbhs9mynmfv3wtczvw.streamlit.app/).



## Project Overview

Most large multilingual translation models have little or no prior exposure to
Ekegusii, Dholuo, or Somali — three languages spoken across Kenya that are
underrepresented in NLP training data. This project fine-tunes NLLB-200 and
mT5 checkpoints on parallel PSA text (health, civic, education, agriculture,
and security announcements) to produce usable translation between these
languages, English, and Kiswahili.

The end product is an interactive Streamlit app (`app.py`) that lets a user:
- Translate text in either direction for each language
- Auto-detect input language (heuristic, not a trained classifier)
- Batch-translate a CSV of sentences
- Compare the same English sentence translated across all three languages side by side
- View evaluation metrics (BLEU / chrF) per language and direction
- Browse an institutional glossary of acronyms the model intentionally preserves rather than mistranslates
- Review known issues and limitations transparently

## Team

| Member | Focus |
|---|---|
| Patricia | Source identification (Week 1); mT5 fine-tuning for Somali (Week 3) |
| Stephen (Steve) | Data collection / scraping (Week 1); NLLB fine-tuning for Dholuo (Week 3) |
| Selmah | Data engineering & preprocessing lead (Weeks 1–2); NLLB fine-tuning for Ekegusii (Week 3);deployment of the app(Week 4) |
| Rencia | Translation QA (Weeks 1–2); mT5 fine-tuning for Ekegusii (Week 3) |
| Trizzah | Coordination & reporting (Weeks 1–2); mT5 fine-tuning for Dholuo (Week 3) |

---

## Week 1 — Data Sourcing & Collection

### Scope decisions
The project initially targeted three languages (Ekegusii, Dholuo, Somali).
Given the team's timeline and the fact that translation-quality checking
needs a native speaker, scope was deliberately narrowed to **Dholuo only**
for active data collection, decided by team poll based on speaker
availability. (Ekegusii was reintroduced in Week 2 — see below — once a
verified corpus became available from the course instructor.)

### Sources
Eleven verified sources were identified and logged: Ministry of Health,
IEBC, NSDCC, NACADA, NTSA, Kenya Red Cross, UNICEF Kenya, WHO Kenya, KBC,
and two Dholuo-language radio stations (Ramogi FM, Radio Nam Lolwe FM).
Radio content could not be scraped, since it's audio-only — a limitation
carried forward rather than solved.

### Collection & early cleaning challenges
- Initial scraping pulled significant non-PSA content (navigation menus,
  addresses, page titles) requiring manual review beyond automated filtering.
- Dholuo translation required calling Google Translate's endpoint directly,
  since the `deep_translator` Python library's hardcoded language list
  doesn't include Dholuo despite Google's own service supporting it.
- The team's real dataset was augmented with a **fact-grounded synthetic
  generation batch** to reach the 5,000+ sentence target within the
  timeline — built from a knowledge base of real, named Kenyan institutions
  and programmes (e.g. SHA, KUCCPS, NTSA, IEBC, KALRO) combined with natural
  phrasing templates, rather than generic filler. This is disclosed
  transparently as synthetic data augmentation, a standard low-resource NLP
  technique, not presented as organically collected data.

### Week 1 outcome
A combined English/Kiswahili → Dholuo dataset reaching the target sentence
count, documented in the Week 1 report with dataset statistics, sample
entries, and known challenges.

---

### Documentation, Dataset & Pipeline Access

The complete documentation of the data sourcing process, collection methodology, cleaning procedures, translation workflow, and challenges is available in the comprehensive Week 1 report:

📄 **Full Week 1 Data Collection Report:**  
[View the complete report here](reports/Week1_Collecting_Data_Report/Week1_Collecting_data_report.docx)

The final processed dataset used for subsequent model development and experimentation is available here:

📂 **Processed Dataset:**  
[Access the final collected data here](data/processed/)

The complete data preparation pipeline, including scripts used for dataset merging, cleaning, translation generation, formatting, and preparation of training-ready datasets is available here:

💻 **Data Processing & Pipeline Scripts:**  
[View all data processing scripts here](src/)

These scripts contain the reproducible workflow used to transform raw collected data into the final datasets used for model training, including:
- Dataset merging and consolidation
- Data cleaning and normalization
- Translation generation and alignment
- Synthetic data augmentation preparation
- Dataset formatting for multilingual machine translation models
- Final preprocessing before model training

---


## Week 2 — Preprocessing & EDA

### Dataset architecture decision
Rather than one sparse combined table, the project maintains **two separate,
fully-parallel datasets**, since Dholuo/Somali and Ekegusii come from
different sources with different coverage:

```
data/processed/
  psa_dataset_dholuo_somali.csv   ← English, Kiswahili, Dholuo, Somali (partial)
  psa_dataset_ekegusii.csv        ← English, Kiswahili, Ekegusii
```

### Ekegusii reintroduced
The course instructor provided a verified Ekegusii PSA corpus (real content
covering initiatives like the CBC curriculum rollout and the DigiSchool
project), plus Ekegusii Bible excerpts. The Bible excerpts were **deliberately
excluded** from the training data — scripture is not a public service
announcement, and including it would blur what the model is learning to
translate. Real Ekegusii translations already present in the original
baseline dataset (but unused after the earlier scope narrowing) were also
recovered and merged with the instructor's corpus.

### Preprocessing pipeline
Built as a parametrized, reusable pipeline (run once per language), with
each stage logged in a funnel table so nothing is silently dropped without
a record of why:

1. **Structural completeness** — drop rows missing required fields.
2. **Text normalization** — mojibake repair (`ftfy`), Unicode NFKC
   normalization, control-character stripping, HTML/URL removal, invisible
   Unicode removal, smart-quote normalization, whitespace collapsing.
3. **Garbage/junk detection** — repeated consecutive words, leftover
   template placeholders, punctuation/digit-only rows, repeated-character
   runs.
4. **Length + cross-language length-ratio filtering** — catches
   truncated/merged/mistranslated rows even when individual lengths look
   fine alone.
5. **Duplicate detection** — exact and near-duplicate matching.
6. **Domain & label validation** — flags anything outside the 5 expected
   domains (Health, Agriculture, Education, Security, Governance);
   corrected a recurring "Security" vs "Security & Safety" naming
   inconsistency across data sources.
7. **Language-ID validation** — English/Kiswahili checked via `langdetect`
   (Dholuo and Ekegusii aren't supported by any mainstream language-ID
   tool, so those columns rely on manual review instead).

Additional Ekegusii-specific QA caught and fixed:
- Multi-layer mojibake that survived `ftfy`'s automatic repair, traced and
  fixed via exact byte-level substring replacement.
- Off-topic content contamination (a Ukraine/Russia news article and a
  South African legal reference, neither of which is a Kenyan PSA) —
  removed after keyword-based detection.

### EDA
Domain distribution (bar/pie), sentence-length histograms per language,
vocabulary size estimates, and source composition breakdowns — produced via
a Jupyter notebook template shared across both language tracks so results
are directly comparable.

---
### Documentation, Notebooks & Report Access

The complete Week 2 preprocessing methodology, exploratory analysis, visualizations, dataset statistics, and findings are documented in the full report:

📄 **Full Week 2 Preprocessing & EDA Report:**  
[View the complete report here](reports/Week2_Preprocessing_EDA_Report/Week2_Preprocessing_EDA_Report.docx)

The complete EDA notebooks used to generate the analysis, visualizations, and dataset quality assessments are available below:

📓 **Dholuo/Somali Preprocessing & EDA Notebook:**  
[Access notebook here](notebooks/preprocesssing_eda/dholuo_somali_preprocessing_eda.ipynb)

📓 **Ekegusii Preprocessing & EDA Notebook:**  
[Access notebook here](notebooks/preprocesssing_eda/preprocessing_eda_ekegusii.ipynb)

These notebooks provide a reproducible record of:
- Dataset loading and inspection
- Cleaning verification
- Statistical summaries
- Visualization generation
- Language-track comparisons
- Final dataset readiness checks before model training

---

### Week 2 outcome
- `psa_dataset_dholuo_somali.csv`: **16,029 rows** (5,112 from the original
  baseline + 10,917 from fact-grounded generation), zero duplicates, zero
  missing translations.
- `psa_dataset_ekegusii.csv`: **5,126 rows** after final cleaning (2,874
  baseline-recovered + 2,270 from the instructor's corpus, after
  deduplication and off-topic removal), zero mojibake remaining.
- Train/validation/test splits (80/10/10, stratified by domain) prepared
  for both.

---

## Week 3 — Model Training

### Model assignments

| Person | Model | Target Language | New-Language Workaround? |
|---|---|---|---|
| Trizzah | mT5 | Dholuo | Partial — mT5 has no per-language embedding to add; relies on consistent text-prefix fine-tuning instead. |
| Patricia | mT5 | Somali | No — natively supported by mT5. |
| Rencia | mT5 | Ekegusii | Yes — new language token added, embeddings resized and warm-started before fine-tuning. |
| Steve | NLLB | Dholuo | No — Dholuo (`luo_Latn`) is a native NLLB-200 token. |
| Selmah | NLLB | Ekegusii | Yes — same new-token adaptation approach as Rencia's mT5 track, applied to NLLB. |

### Training requirements
- 10 epochs, batch training, `save_strategy="epoch"` for automatic
  checkpointing (no manual checkpoint saving).
- Google Drive mounted before training, to prevent checkpoint loss from a
  runtime disconnect (a real risk experienced earlier in the project).
- Each member's submission: final trained model + a report covering
  per-epoch performance, challenges faced, and limitations.

---

## Week 4 — Evaluation & Deployment

### Model Performance

| Language | Direction(s) | BLEU | chrF | Notes |
|---|---|---|---|---|
| Ekegusii | English → Ekegusii | 17.04 | 47.71 | |
| Ekegusii | Ekegusii → English | 16.89 | 41.71 | |
| Ekegusii | Kiswahili → Ekegusii | 15.83 | 48.12 | |
| Ekegusii | Ekegusii → Kiswahili | 17.18 | 44.37 | |
| Dholuo | Overall (epoch 10) | 71.77 | 82.11 | Single overall score, not broken out by direction |
| Somali | Overall (epoch 6) | 69.87 | N/A | SacreBLEU only, no chrF reported |

**A note on the Dholuo numbers**: they are markedly higher than Ekegusii's —
plausible given Dholuo already has native NLLB-200 support, but this has not
yet been independently sanity-checked for train/test leakage. Treat as
provisional pending review.

chrF is generally more informative than BLEU for morphologically rich
languages like Ekegusii, since BLEU penalizes valid word-ending variation as
a full mismatch.

The Ekegusii oversampling results above reflect a corrected re-run after an
earlier labeling bug was identified and fixed during evaluation — the
original (bugged) numbers are superseded and should not be cited.

### Known Issues & Limitations

**Data**
- ~370 rows (7.2%) of the Ekegusii training corpus have truncated source
  text (scraped from a preview snippet) — exclusion decision still pending.
- A small number of placeholder/corrupted rows (e.g. literal `"English_text"`)
  were found during qualitative review and may still be present.

**Model**
- Occasional repetition looping in beam search on long or syntactically
  complex sentences.
- Ekegusii model is somewhat stronger translating OUT of Ekegusii than INTO
  it, despite oversampling narrowing this gap substantially.
- Automatic metrics were computed on models with no prior exposure to
  Ekegusii's morphology — scores should be read as directional signals, not
  absolute quality.
- Dholuo's high BLEU/chrF have not yet been independently sanity-checked
  (see above).
- No manual transcription of authentic spoken Dholuo (from Ramogi FM /
  Radio Nam Lolwe FM) was completed — considered but not pursued given time
  constraints.

**Evaluation**
- COMET could not be computed in the current environment due to a
  dependency conflict.
- Human evaluation by native speakers is in progress; no confirmed native
  Ekegusii reviewer was available on the team as of Week 3, addressed as a
  documented limitation rather than a blocker.

### Institutional Glossary
Certain institutional acronyms (e.g. `TVET`, `IEBC`, `KCSE`, `KRA`) have no
native-language equivalent, so the model is designed to preserve them as-is
in translation rather than mistranslate or drop them. The app highlights
these terms inline with tooltips explaining the full name.

### Project Structure

```
psa-dholuo-mt/
├── app.py                  # Streamlit app entry point
├── requirements.txt
├── Models/
│   ├── ekegusii_models/nllb-ekegusii-final/
│   ├── dholuo_models/nllb-dholuo-final/
│   └── somali_models/mt5-somali-final/
├── data/
│   ├── raw/                  # scraped/provided source data, never edited
│   ├── interim/               # cleaning / validation intermediate files
│   ├── processed/             # final per-language datasets + splits
│   ├── sources/                # verified source log
│   └── knowledge_base/        # kenya_facts.json — real institutions used for grounded generation
├── notebooks/
│   ├── preprocessing_eda/      # data cleaning and exploratory analysis
│   └── training_models/        # fine-tuning notebooks per language/architecture
├── src/                        # data collection, cleaning, and translation scripts
├── reports/                    # weekly progress reports (Weeks 1–4)
└── docs/                       # project brief, category definitions, planning docs
```

### Running Locally

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Local inference runs on CPU unless a CUDA GPU is available, and will be
noticeably slower than a GPU-backed deployment — expect longer translation
times, particularly on first load per language (model weights loading from
disk) and during beam search (`num_beams=4`).

### Deployment Options

**Hugging Face Spaces (free, recommended for the demo)** — model folders and
`app.py` uploaded directly to a Space repo via the browser; Spaces builds
and hosts the app automatically with a public URL, no server management
needed.

**Modal (GPU-backed)** — the app can be deployed with model weights bundled
directly into the container image:
```powershell
pip install modal
modal setup
modal deploy modal_app.py
```
This provisions a GPU container, bundling `app.py` and the local `Models/`
folder into the image. GPU inference removes the CPU bottleneck from beam
search, making translation near-instant compared to local CPU runs.

### Tech Stack
- **UI**: Streamlit
- **Models**: NLLB-200 (Ekegusii, Dholuo), mT5 (Ekegusii, Dholuo, Somali)
- **ML framework**: PyTorch, Hugging Face Transformers
- **Deployment**: Hugging Face Spaces, optionally Modal (GPU) or Docker

---

## Data Pipeline Summary (end to end)

```
Raw PSA corpus (scraped + instructor-provided)
  → encoding / mojibake repair
  → boilerplate & off-topic removal
  → code-switch detection & glossary extraction
  → fact-grounded synthetic augmentation (Dholuo/Somali track only)
  → leak-safe train / val / test split
  → baseline fine-tune
  → error-driven oversampling
  → evaluation (BLEU, chrF, SacreBLEU, qualitative review, human evaluation in progress)
  → Streamlit deployment
```

# Sauti — Kenyan Low-Resource Language Translator

A neural machine translation system for three underrepresented Kenyan languages —
**Ekegusii**, **Dholuo**, and **Somali** — built by fine-tuning multilingual base
models (NLLB-200 and mT5) on a cleaned corpus of public service announcements (PSAs).

Get our live deployed app [here](https://psa-dholuo-mt-ywk9nbhs9mynmfv3wtczvw.streamlit.app/).

## Table of Contents

- [Project Overview](#project-overview)
- [Team](#team)
- [Week 1 — Data Sourcing & Collection](#week-1--data-sourcing--collection)
  - [Scope Decisions](#scope-decisions)
  - [Sources](#sources)
  - [Collection & Early Cleaning Challenges](#collection--early-cleaning-challenges)
  - [Documentation, Dataset & Pipeline Access](#documentation-dataset--pipeline-access)

- [Week 2 — Preprocessing & EDA](#week-2--preprocessing--eda)
  - [Dataset Architecture Decision](#dataset-architecture-decision)
  - [Ekegusii Reintroduced](#ekegusii-reintroduced)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [EDA](#eda)
  - [Documentation, Notebooks & Report Access](#documentation-notebooks--report-access)
  - [Week 2 Outcome](#week-2-outcome)

- [Week 3 — Model Training](#week-3--model-training)
  - [Model Assignments](#model-assignments)
  - [Training Requirements](#training-requirements)
  - [Model Adaptation Approaches](#model-adaptation-approaches)
  - [Training Documentation & Results Access](#training-documentation--results-access)
  - [Training Notebooks](#training-notebooks)
  - [Final Trained Models](#final-trained-models)

- [Week 4 — Evaluation & Deployment](#week-4--evaluation--deployment)
  - [Model Performance](#model-performance)
  - [Known Issues & Limitations](#known-issues--limitations)
  - [Institutional Glossary](#institutional-glossary)
  - [Project Structure](#project-structure)
  - [Running Locally](#running-locally)

- [Deployment](#deployment)
  - [Deployment Architecture](#deployment-architecture)
  - [Deployment Workflow](#deployment-workflow)
  - [Alternative Deployment Options](#alternative-deployment-options)
    - [Hugging Face Spaces](#hugging-face-spaces)
    - [Modal GPU Deployment](#modal-gpu-backed-deployment)
    - [Docker Deployment](#docker-deployment)
  - [Tech Stack](#tech-stack)
  - [Deployment Challenges & Solutions](#deployment-challenges--solutions)

- [Data Pipeline Summary (End-to-End)](#data-pipeline-summary-end-to-end)

- [Closing Remarks](#closing-remarks)

- [Future Work](#future-work)

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

| Member | GitHub | Weeks 1–2 | Week 3 (Fine-tuning) | Week 4 |
|---|---|---|---|---|
| Selmah | [@SelmahT](https://github.com/SelmahT) | Data engineering & preprocessing lead | NLLB fine-tuning — Ekegusii | App deployment |
| Stephen (Steve) | [@Stephen-Austine](https://github.com/Stephen-Austine) | Data collection / scraping | NLLB fine-tuning — Dholuo | — |
| Patricia | [@PatriciaKiarie04](https://github.com/PatriciaKiarie04) | Source identification | mT5 fine-tuning — Somali | — |
| Rencia | [@RenciaSeda](https://github.com/RenciaSeda) | Translation QA | mT5 fine-tuning — Ekegusii | — |
| Trizzah | [@Trizah250000](https://github.com/Trizah250000) | Coordination & reporting | mT5 fine-tuning — Dholuo | — |

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

| Person   | Model | Target Language | New-Language Workaround? |
| -------- | ----- | --------------- | ------------------------- |
| Trizzah  | mT5   | Dholuo          | Partial — mT5 has no per-language embedding addition mechanism; training relies on consistent text-prefix fine-tuning instead. |
| Patricia | mT5   | Somali          | No — Somali is natively supported by mT5. |
| Rencia   | mT5   | Ekegusii        | Yes — a new language token was introduced, embeddings were resized, and warm-start initialization was applied before fine-tuning. |
| Steve    | NLLB  | Dholuo          | No — Dholuo (`luo_Latn`) is already supported as a native NLLB-200 language token. |
| Selmah   | NLLB  | Ekegusii        | Yes — the same new-language token adaptation approach was applied by extending NLLB with Ekegusii support before fine-tuning. |

---

### Training requirements

All models followed a standardized training workflow to ensure comparable experimentation across languages:

- Training was performed for **10 epochs** using batch-based fine-tuning.
- `save_strategy="epoch"` was used to automatically save checkpoints after every epoch.
- Google Drive was mounted before training to reduce the risk of checkpoint loss caused by runtime disconnections.
- Each team member submitted:
  - The final fine-tuned model checkpoint.
  - A training report documenting epoch-level performance, challenges encountered, and model limitations.

---

### Model adaptation approaches

Because some target languages are low-resource and not directly supported by pretrained multilingual models, different adaptation strategies were applied:

- **Native language support**
  - Languages already present in the pretrained tokenizer vocabulary (e.g., Dholuo in NLLB and Somali in mT5) were fine-tuned directly.

- **New language token adaptation**
  - For Ekegusii, additional language tokens were introduced.
  - Tokenizer vocabulary was extended.
  - Model embeddings were resized.
  - New embeddings were warm-started before fine-tuning to allow the pretrained model to adapt to the new language.

- **Prefix-based fine-tuning**
  - For mT5 Dholuo, consistent text prefixes were used because mT5 does not support the same explicit language-token extension mechanism as NLLB.

---

### Training Documentation & Results Access

The complete training process, model configurations, experiments, evaluation results, challenges, and limitations are documented in the individual project reports:

📄 **NLLB Ekegusii Training Report:**  
[View report here](reports/Week3_training_reports/Ekegusii_NLLB/Project_Report.docx)

📄 **mT5 Somali Training Report:**  
[View report here](reports/Week3_training_reports/mT5_Somali_Report.docx)

📄 **NLLB Dholuo Training Report:**  
[View report here](reports/Week3_training_reports/NLLB_Dholuo_Report.docx)

---

### Training Notebooks

The complete training notebooks used for model fine-tuning, configuration, checkpointing, and evaluation are available below:

📓 **mT5 Somali Training Notebook:**  
[Access notebook here](notebooks/training_models/train-mt5-somali.ipynb)

📓 **NLLB Dholuo Training Notebook:**  
[Access notebook here](notebooks/training_models/train-nllb-dholuo.ipynb)

📓 **NLLB Ekegusii Training Notebook:**  
[Access notebook here](notebooks/training_models/train-nllb-ekegusii.ipynb)

📓 **mT5 Dholuo Training Notebook:**  
[Access notebook here](notebooks/training_models/train-mt5-dholuo.ipynb)

These notebooks provide a reproducible record of:

- Dataset loading and preparation.
- Tokenizer configuration.
- Language-token adaptation steps.
- Model initialization.
- Fine-tuning hyperparameters.
- Training loops and checkpoint management.
- Validation monitoring.
- Final model saving and evaluation.

---

### Final trained models

The resulting fine-tuned models from each training track were prepared for downstream evaluation and deployment as part of the final translation system.

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

## Deployment

### Deployment Architecture

The final translation system was deployed as an interactive **Streamlit web application**. To avoid storing large model checkpoints directly inside the application repository, the trained translation models were uploaded and hosted on the **Hugging Face Hub**.

The deployed system follows the architecture below:

```text
User
 │
 ▼
Streamlit Web Application
(app.py hosted on GitHub)
 │
 ▼
Loads fine-tuned models from Hugging Face Hub
 │
 ▼
NLLB-200 / mT5 Translation Models
 │
 ▼
Translated PSA Output
```

The Streamlit application (`app.py`) acts as the interface between users and the trained translation models. Instead of loading local model files, the application retrieves the required tokenizer and model weights directly from their Hugging Face repositories using the Hugging Face Transformers library.

This deployment strategy:

- Keeps the GitHub repository lightweight.
- Avoids GitHub storage limitations for large model checkpoints.
- Allows models and application code to be updated independently.
- Provides a reproducible deployment pipeline through version-controlled model repositories.

---

## Deployment Workflow

The production deployment process consisted of the following stages:

### 1. Model Hosting

The fine-tuned translation models were uploaded to Hugging Face Hub repositories.

The hosted models include:

- **NLLB-200**
  - Ekegusii translation model
  - Dholuo translation model

- **mT5**
  - Somali translation model

Each Hugging Face repository contains the required:

- Model weights
- Tokenizer files
- Configuration files
- Language-specific adaptations

---

### 2. Application Integration

The Streamlit application was configured to load models dynamically from Hugging Face.

The application workflow:

1. The user selects:
   - Source language.
   - Target language.
   - PSA text input.

2. The application identifies the appropriate translation model.

3. The tokenizer and model are loaded from Hugging Face Hub.

4. The input text is passed through the model.

5. The translated PSA output is displayed to the user.

Example model-loading workflow:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "HuggingFace_Model_Repository"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

---

### 3. Streamlit Deployment

The final application was deployed using **Streamlit Cloud** through GitHub integration.

Deployment process:

1. The Streamlit application code was pushed to GitHub.
2. Streamlit Cloud was connected to the GitHub repository.
3. Dependencies were installed from `requirements.txt`.
4. The application started and retrieved the translation models from Hugging Face Hub.
5. The application became publicly accessible through a Streamlit URL.

🚀 **Live deployed application:**  
[Access the translator here](https://psa-dholuo-mt-ywk9nbhs9mynmfv3wtczvw.streamlit.app/)

---

# Alternative Deployment Options

Although Streamlit Cloud was used for the final deployment, other deployment approaches were explored as alternative hosting strategies.

---

## Hugging Face Spaces

Hugging Face Spaces provides an alternative platform for hosting machine learning demonstrations.

A Space can contain:

- Application code.
- Model-loading logic.
- User interface components.

Advantages:

- Direct integration with Hugging Face model repositories.
- Automatic build and deployment.
- Public demo URL generation.
- Minimal server management.

For this project, Hugging Face was primarily used as the **model hosting platform**, while Streamlit Cloud was selected as the final application hosting platform.

---

## Modal (GPU-backed Deployment)

Modal was explored as an alternative deployment option for GPU-enabled inference.

Deployment commands:

```powershell
pip install modal
modal setup
modal deploy modal_app.py
```

The Modal deployment approach packages:

- Streamlit application code.
- Required Python dependencies.
- Model files or model-loading configuration.

into a GPU-enabled container.

GPU inference provides significant performance improvements compared to CPU execution, especially for larger multilingual models such as NLLB-200 where beam search generation can become computationally expensive.

Benefits:

- Faster translation inference.
- GPU acceleration.
- Scalable container-based deployment.

---

## Docker Deployment

The application can also be deployed using Docker by packaging:

- Streamlit application code.
- Python dependencies.
- Runtime environment.
- Model-loading configuration.

Docker provides a portable deployment option suitable for:

- Local hosting.
- Cloud servers.
- Institutional infrastructure.

---

# Tech Stack

| Component | Technology |
|---|---|
| User Interface | Streamlit |
| Programming Language | Python |
| Translation Models | NLLB-200, mT5 |
| Target Languages | Ekegusii, Dholuo, Somali |
| Machine Learning Framework | PyTorch |
| NLP Framework | Hugging Face Transformers |
| Model Hosting | Hugging Face Hub |
| Application Hosting | Streamlit Cloud |
| Version Control | GitHub |
| Alternative Deployment | Hugging Face Spaces, Modal (GPU), Docker |

---

# Deployment Challenges & Solutions

| Challenge | Solution |
|---|---|
| Large model files exceeded GitHub storage limits | Hosted models externally on Hugging Face Hub |
| Streamlit Cloud resource limitations | Avoided bundling multi-gigabyte checkpoints inside the repository |
| Maintaining reproducibility | Linked application code to specific Hugging Face model repositories |
| Slow CPU inference | Explored GPU-backed deployment using Modal |
| Managing application dependencies | Used `requirements.txt` for automated environment setup |
---

## Data Pipeline Summary (End-to-End)

The complete project workflow follows a reproducible end-to-end machine translation pipeline, beginning with raw public service announcement (PSA) data collection and ending with a deployed translation application.

```text
Raw PSA Corpus
(scraped sources + instructor-provided datasets)
        │
        ▼
Data Cleaning & Quality Control
(encoding repair, mojibake correction,
Unicode normalization, noise removal)
        │
        ▼
Content Filtering
(boilerplate removal, off-topic detection,
domain validation)
        │
        ▼
Language-Specific Processing
(code-switch detection,
language validation,
glossary extraction)
        │
        ▼
Dataset Enhancement
(fact-grounded synthetic augmentation
for Dholuo/Somali low-resource tracks)
        │
        ▼
Dataset Preparation
(aligned language pairs,
train/validation/test split,
leak prevention)
        │
        ▼
Model Development
(baseline fine-tuning of NLLB-200 and mT5)
        │
        ▼
Model Improvement
(error analysis,
error-driven oversampling,
targeted fine-tuning)
        │
        ▼
Model Evaluation
(BLEU, chrF, SacreBLEU,
qualitative analysis,
human evaluation in progress)
        │
        ▼
Model Hosting
(fine-tuned models uploaded to Hugging Face Hub)
        │
        ▼
Application Deployment
(Streamlit web application)
```

The final system integrates the complete machine translation workflow:

- Data collection from verified Kenyan PSA sources.
- Language-specific preprocessing and quality assurance.
- Low-resource dataset enhancement techniques.
- Multilingual model adaptation and fine-tuning.
- Automated and human-centered evaluation.
- Deployment of the final translation system as an accessible web application.

---

# Closing Remarks

This project demonstrates the challenges and opportunities involved in developing machine translation systems for **low-resource Kenyan languages**. By combining multilingual pretrained models, careful dataset engineering, language-specific adaptation strategies, and transparent documentation, we developed a practical translation pipeline for **Ekegusii, Dholuo, and Somali** public service communication.

The project highlights that improving translation quality for underrepresented languages requires more than model selection alone. Reliable results depend heavily on:

- High-quality and domain-relevant datasets.
- Careful preprocessing and validation.
- Appropriate handling of languages missing from pretrained model vocabularies.
- Continuous error analysis and model refinement.
- Evaluation approaches that combine automatic metrics with human judgment.

Although challenges remain, including limited parallel data availability and the difficulty of fully capturing linguistic nuances in low-resource languages, this work provides a foundation for future expansion into additional Kenyan languages and broader public communication applications.

The resulting system demonstrates the potential of modern multilingual NLP approaches to improve accessibility, information sharing, and digital inclusion for communities whose languages remain underrepresented in artificial intelligence technologies.

---

## Future Work

Potential directions for future improvement include:

- Expanding the dataset with additional verified PSA sources.
- Incorporating more native-speaker evaluation and feedback.
- Improving domain-specific terminology handling through larger glossaries.
- Exploring larger multilingual foundation models.
- Developing continuous learning pipelines as new PSA content becomes available.
- Extending support to additional Kenyan languages.

---

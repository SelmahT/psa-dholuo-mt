"""
Kenyan Low-Resource Language Translator
=========================================
A multi-language (Ekegusii / Dholuo / Somali) neural machine translation
demo app, built on fine-tuned NLLB-200 / mT5 checkpoints.

Run with:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import torch
import pandas as pd
import time
import re
from datetime import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =============================================================================
# PAGE CONFIG — must be the first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Sauti | Kenyan Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIG — flip this to False once real model paths are filled in below
# =============================================================================
USE_MOCK_MODELS = False  # set True to demo the UI without loading real weights

# NOTE: paths below use /root/Models/... because modal_app.py bundles the
# local "Models" folder into the container at that path via add_local_dir().
# If you're running this locally (not on Modal), change these back to
# relative paths like "./Models/ekegusii_models/nllb-ekegusii-final".
MODEL_CONFIG = {
    "Ekegusii": {
        "ready": True,
        "path": "Models/ekegusii_models/nllb-ekegusii-final",
        "arch": "nllb",
        "lang_code": "guz_Latn",
        "pairs": [
            ("English", "Ekegusii"), ("Ekegusii", "English"),
            ("Kiswahili", "Ekegusii"), ("Ekegusii", "Kiswahili"),
        ],
        "owner": "Selmah",
        "metrics": {
            "English → Ekegusii":   {"bleu": 17.04, "chrf": 47.71},
            "Ekegusii → English":   {"bleu": 16.89, "chrf": 41.71},
            "Kiswahili → Ekegusii": {"bleu": 15.83, "chrf": 48.12},
            "Ekegusii → Kiswahili": {"bleu": 17.18, "chrf": 44.37},
        },
    },
    "Dholuo": {
        "ready": True,
        "path": "Models/dholuo_models/nllb-dholuo-final",
        "arch": "nllb",
        "lang_code": "luo_Latn",   # already a native NLLB-200 token -- no embedding-add trick needed
        "pairs": [
            ("English", "Dholuo"), ("Dholuo", "English"),
            ("Kiswahili", "Dholuo"), ("Dholuo", "Kiswahili"),
        ],
        "owner": "Steve",
        "metrics": {
            # Steve's report gives one overall score (epoch 10, final checkpoint),
            # not a per-direction breakdown like Ekegusii's table.
            # NOTE: these numbers are notably higher than Ekegusii's — worth a
            # quick sanity check (train/test leakage, manual spot-check of a
            # few predictions) before treating them as final for the report.
            "Overall (epoch 10)": {"bleu": 71.77, "chrf": 82.11},
        },
    },
    "Somali": {
        "ready": True,
        "path": "Models/somali_models/mt5-somali-final",
        "arch": "mt5",
        "lang_code": "som",
        "pairs": [
            ("English", "Somali"), ("Somali", "English"),
        ],
        "owner": "Patricia (mT5)",
        "metrics": {
            # Patricia's report only gives one overall sacreBLEU figure (epoch 6),
            # no chrF and no per-direction breakdown yet -- stored as "bleu" since
            # sacreBLEU and BLEU share the same 0-100 scale, chrf left as None so
            # the Performance page displays "N/A" instead of crashing (the
            # original {"sacrebleu": 69.87} key name would have crashed this
            # page outright -- the display code requires exactly "bleu"/"chrf").
            # NOTE: 69.87 is dramatically higher than Ekegusii's rigorously
            # validated 15-18 BLEU range, and close to Dholuo's already-flagged
            # 71.77 -- both warrant the same sanity check (train/test leakage,
            # confirm this is a held-out test score, not train or best-epoch
            # cherry-picking) before being presented as final in the report.
            "Overall (epoch 6)": {"bleu": 69.87, "chrf": None},
        },
    },
}

LANG_TO_NLLB_CODE = {
    "English": "eng_Latn", "Kiswahili": "swh_Latn",
    "Ekegusii": "guz_Latn", "Dholuo": "luo_Latn", "Somali": "som_Latn",
}

# Institutional glossary (from Week 2 code-switching analysis) — used to
# highlight untranslated acronyms in output rather than leave them unexplained
GLOSSARY = {
    "TVET": "Technical and Vocational Education and Training",
    "IEBC": "Independent Electoral and Boundaries Commission",
    "CSA": "Climate-Smart Agriculture",
    "KCSE": "Kenya Certificate of Secondary Education",
    "KICD": "Kenya Institute of Curriculum Development",
    "EACC": "Ethics and Anti-Corruption Commission",
    "CBC": "Competency-Based Curriculum",
    "TSC": "Teachers Service Commission",
    "KNEC": "Kenya National Examinations Council",
    "KUCCPS": "Kenya Universities and Colleges Central Placement Service",
    "SHA": "Social Health Authority (formerly NHIF)",
    "NPS": "National Police Service",
    "KNCHR": "Kenya National Commission on Human Rights",
    "HELB": "Higher Education Loans Board",
    "KRA": "Kenya Revenue Authority",
}

KNOWN_ISSUES = [
    ("Data", "~370 rows (7.2%) of the Ekegusii training corpus have truncated source text (source scraped from a preview snippet) — team decision on exclusion still pending."),
    ("Data", "A small number of placeholder/corrupted rows (e.g. literal 'English_text') were found during qualitative review and may still be present in training data."),
    ("Model", "Occasional repetition looping in beam search on long or syntactically complex sentences."),
    ("Model", "Model is somewhat stronger at translating OUT of Ekegusii than INTO it, despite oversampling narrowing this gap substantially."),
    ("Model", "Automatic metrics (BLEU, chrF) were computed on models with no prior exposure to Ekegusii's morphology — scores should be read as directional signals, not absolute quality."),
    ("Model", "Dholuo's overall BLEU/chrF (71.77 / 82.11) and Somali's overall BLEU (69.87) are markedly higher than Ekegusii's rigorously validated 15-18 BLEU range — plausible given Dholuo/Somali's native/near-native support in their base models, but neither has been independently sanity-checked for train/test leakage or confirmed as a true held-out test score. Treat both as provisional pending review with Steve and Patricia."),
    ("Evaluation", "COMET could not be computed in the current environment due to a dependency conflict; human evaluation by native speakers is in progress."),
]

# =============================================================================
# STYLE — custom CSS for a distinctive look instead of default Streamlit chrome
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .app-title { font-family: 'Space Grotesk', sans-serif; }

    .main { background: linear-gradient(180deg, #0f1420 0%, #161b2e 100%); }

    .hero {
        background: linear-gradient(135deg, #1b2340 0%, #2d1b4e 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        font-size: 2.1rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #7dd3fc, #c084fc, #f0abfc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; font-size: 1.02rem; margin: 0; }

    .lang-badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem;
    }
    .badge-ready { background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .badge-pending { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

    .translation-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.5rem; margin-top: 1rem;
    }
    .translation-output {
        font-size: 1.25rem; font-weight: 500; color: #f1f5f9;
        line-height: 1.6; padding: 1rem 0;
    }
    .glossary-chip {
        display: inline-block; background: rgba(192, 132, 252, 0.15);
        color: #d8b4fe; border: 1px solid rgba(192,132,252,0.3);
        border-radius: 8px; padding: 0.15rem 0.5rem; margin: 0.15rem;
        font-size: 0.8rem; cursor: help;
    }
    .metric-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #7dd3fc; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

    .footer-note { color: #64748b; font-size: 0.85rem; text-align: center; margin-top: 3rem; }

    div[data-testid="stSidebar"] { background: #0d1121; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING (cached so switching pages/directions doesn't reload weights)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model(lang_key: str):
    cfg = MODEL_CONFIG[lang_key]
    if USE_MOCK_MODELS or not cfg["ready"]:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["path"], local_files_only=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def mock_translate(text: str) -> str:
    """Placeholder output when a model isn't ready yet or mock mode is on."""
    return f"[demo output — model not yet connected] {text[:60]}..."


def run_translation(text: str, src_lang: str, tgt_lang: str, lang_key: str) -> tuple[str, float]:
    cfg = MODEL_CONFIG[lang_key]
    start = time.time()

    if USE_MOCK_MODELS or not cfg["ready"]:
        time.sleep(0.6)  # simulate latency so the mock UX still feels real
        return mock_translate(text), time.time() - start

    model, tokenizer = load_model(lang_key)

    if cfg["arch"] == "nllb":
        tokenizer.src_lang = LANG_TO_NLLB_CODE[src_lang]
        forced_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB_CODE[tgt_lang])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, forced_bos_token_id=forced_id, max_length=128,
                num_beams=4, no_repeat_ngram_size=3,
            )
        result = tokenizer.decode(out[0], skip_special_tokens=True)
    else:  # mT5 — prefix-based, no language token
        prefixed = f"translate to {tgt_lang}: {text}"
        inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=4, no_repeat_ngram_size=3)
        result = tokenizer.decode(out[0], skip_special_tokens=True)

    return result, time.time() - start


def detect_language(text: str) -> str:
    """Lightweight heuristic language guesser — not a trained classifier,
    just common function-word matching to give the 'auto-detect' feature
    something reasonable to do without adding a whole extra model."""
    text_lower = text.lower()
    swahili_markers = [" ya ", " na ", " wa ", " kwa ", " za ", "ni ", "hii "]
    ekegusii_markers = ["nigo", "obon", "aba", "ase ", "ekero", "eng'", "bwa"]
    scores = {
        "Kiswahili": sum(m in text_lower for m in swahili_markers),
        "Ekegusii": sum(m in text_lower for m in ekegusii_markers),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "English"


def highlight_glossary_terms(text: str) -> str:
    """Wrap known institutional acronyms in a styled chip with a tooltip,
    so untranslated code-switched terms are explained inline rather than
    left as unexplained noise in the output."""
    for term, full_name in GLOSSARY.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(
            pattern,
            f'<span class="glossary-chip" title="{full_name}">{term}</span>',
            text,
        )
    return text


# =============================================================================
# SESSION STATE
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("### 🌍 Sauti")
    st.caption("Kenyan low-resource language MT")
    page = st.radio(
        "Navigate",
        ["🔤 Translate", "🆚 Compare Languages", "📊 Model Performance", "📖 Glossary", "⚠️ Known Issues", "👥 About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model status**")
    for lang, cfg in MODEL_CONFIG.items():
        badge = '<span class="lang-badge badge-ready">● Ready</span>' if cfg["ready"] else '<span class="lang-badge badge-pending">● Pending</span>'
        st.markdown(f"{lang} {badge}", unsafe_allow_html=True)
    st.divider()
    st.caption(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '⚠️ CPU only'}")

# =============================================================================
# PAGE: TRANSLATE
# =============================================================================
if page == "🔤 Translate":
    st.markdown("""
    <div class="hero">
        <h1>Sauti — Kenyan Language Translator</h1>
        <p>Fine-tuned neural machine translation for Ekegusii, Dholuo, and Somali — 
        low-resource languages built into models that never saw them before.</p>
    </div>
    """, unsafe_allow_html=True)

    col_lang, col_dir = st.columns([1, 2])
    with col_lang:
        target_language = st.selectbox("Language track", list(MODEL_CONFIG.keys()))

    cfg = MODEL_CONFIG[target_language]
    if not cfg["ready"] and not USE_MOCK_MODELS:
        st.warning(
            f"**{target_language} model not yet available** — pending delivery from {cfg['owner']}. "
            f"Showing demo mode with placeholder output so the interface can be reviewed ahead of time.",
            icon="⏳",
        )

    available_pairs = cfg["pairs"]
    pair_labels = [f"{s} → {t}" for s, t in available_pairs]

    with col_dir:
        direction_label = st.selectbox("Translation direction", pair_labels)
    src_lang, tgt_lang = available_pairs[pair_labels.index(direction_label)]

    auto_detect = st.toggle("🔎 Auto-detect input language", value=False)

    input_text = st.text_area(
        f"Enter text in {src_lang}" if not auto_detect else "Enter text (language will be detected)",
        height=140,
        placeholder="Type or paste a sentence to translate…",
    )

    if auto_detect and input_text.strip():
        detected = detect_language(input_text)
        st.caption(f"Detected language: **{detected}**")

    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        translate_clicked = st.button("Translate →", type="primary", width='stretch')

    if translate_clicked and input_text.strip():
        with st.spinner(f"Translating {src_lang} → {tgt_lang}…"):
            result, elapsed = run_translation(input_text, src_lang, tgt_lang, target_language)

        highlighted = highlight_glossary_terms(result)

        st.markdown(f"""
        <div class="translation-card">
            <div style="color:#64748b; font-size:0.85rem;">{src_lang} → {tgt_lang} · {elapsed:.2f}s</div>
            <div class="translation-output">{highlighted}</div>
        </div>
        """, unsafe_allow_html=True)

        col_copy, col_dl = st.columns([1, 1])
        with col_copy:
            st.code(result, language=None)
        with col_dl:
            st.download_button("⬇ Download translation", result, file_name="translation.txt")

        st.session_state.history.insert(0, {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "language": target_language,
            "direction": direction_label,
            "source": input_text,
            "translation": result,
        })

    elif translate_clicked:
        st.info("Enter some text first.")

    # ---- Batch translation via CSV upload ----
    with st.expander("📁 Batch translate from a CSV"):
        st.caption("Upload a CSV with a text column — every row gets translated in the current direction.")
        uploaded = st.file_uploader("Upload CSV", type="csv", key="batch_upload")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            text_col = st.selectbox("Which column has the source text?", batch_df.columns)
            if st.button("Run batch translation"):
                progress = st.progress(0, text="Translating…")
                outputs = []
                for i, row_text in enumerate(batch_df[text_col].astype(str)):
                    out, _ = run_translation(row_text, src_lang, tgt_lang, target_language)
                    outputs.append(out)
                    progress.progress((i + 1) / len(batch_df), text=f"Translating… {i+1}/{len(batch_df)}")
                batch_df["translation"] = outputs
                st.dataframe(batch_df, width='stretch')
                st.download_button(
                    "⬇ Download results CSV",
                    batch_df.to_csv(index=False),
                    file_name="batch_translations.csv",
                )

    # ---- History ----
    if st.session_state.history:
        with st.expander(f"🕘 Session history ({len(st.session_state.history)})"):
            for item in st.session_state.history[:10]:
                st.markdown(f"**{item['timestamp']}** · {item['direction']}")
                st.markdown(f"> {item['source']}")
                st.markdown(f"→ {item['translation']}")
                st.divider()

# =============================================================================
# PAGE: COMPARE LANGUAGES
# =============================================================================
elif page == "🆚 Compare Languages":
    st.markdown("""
    <div class="hero">
        <h1>Compare Across Languages</h1>
        <p>Translate one English sentence into every ready language side by side — 
        a quick way to see how each model handles the same input.</p>
    </div>
    """, unsafe_allow_html=True)

    compare_text = st.text_area(
        "Enter English text to translate into all available languages",
        height=120,
        placeholder="e.g. Free maternal healthcare is available at all public hospitals.",
    )

    if st.button("Compare →", type="primary"):
        if not compare_text.strip():
            st.info("Enter some text first.")
        else:
            cols = st.columns(len(MODEL_CONFIG))
            for col, (lang, cfg) in zip(cols, MODEL_CONFIG.items()):
                with col:
                    st.markdown(f"#### {lang}")
                    if not cfg["ready"] and not USE_MOCK_MODELS:
                        st.markdown(
                            f'<span class="lang-badge badge-pending">● Pending</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Awaiting model from {cfg['owner']}")
                        continue

                    # English is always the source here, so pick whichever
                    # pair in this language's config starts from English
                    eng_pair = next((p for p in cfg["pairs"] if p[0] == "English"), None)
                    if eng_pair is None:
                        st.caption("No English-source direction configured for this language.")
                        continue

                    with st.spinner("Translating…"):
                        result, elapsed = run_translation(compare_text, "English", eng_pair[1], lang)
                    st.markdown(
                        f'<div class="translation-card"><div class="translation-output" style="font-size:1.05rem;">{highlight_glossary_terms(result)}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{elapsed:.2f}s")

    st.divider()
    st.caption(
        "This view is most useful once all three languages are live — right now it "
        "shows real output for ready models and an honest 'pending' state for the rest, "
        "rather than hiding what's incomplete."
    )

# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="hero">
        <h1>Model Performance</h1>
        <p>Automatic evaluation metrics across all translation directions and languages.</p>
    </div>
    """, unsafe_allow_html=True)

    for lang, cfg in MODEL_CONFIG.items():
        st.subheader(lang)
        if not cfg["metrics"]:
            st.info(f"No evaluation results yet — model pending from {cfg['owner']}.", icon="⏳")
            continue

        metric_df = pd.DataFrame(cfg["metrics"]).T.reset_index()
        metric_df.columns = ["Direction", "BLEU", "chrF"]

        cols = st.columns(len(metric_df))
        for col, (_, row) in zip(cols, metric_df.iterrows()):
            with col:
                chrf_display = f"{row['chrF']:.1f}" if pd.notna(row['chrF']) else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{row['Direction']}</div>
                    <div class="metric-value">{row['BLEU']:.1f}</div>
                    <div class="metric-label">BLEU</div>
                    <div style="height:0.5rem;"></div>
                    <div class="metric-value" style="color:#c084fc;">{chrf_display}</div>
                    <div class="metric-label">chrF</div>
                </div>
                """, unsafe_allow_html=True)

        chart_df = metric_df.set_index("Direction")[["BLEU", "chrF"]].fillna(0)
        st.bar_chart(chart_df, height=280)
        st.caption(
            "chrF is generally more informative than BLEU for morphologically rich languages "
            "like Ekegusii, since BLEU penalizes valid word-ending variation as a full mismatch."
        )
        st.divider()

# =============================================================================
# PAGE: GLOSSARY
# =============================================================================
elif page == "📖 Glossary":
    st.markdown("""
    <div class="hero">
        <h1>Institutional Glossary</h1>
        <p>Code-switched acronyms retained as-is in translation — these institutions 
        have no native-language equivalent, so the model preserves them intentionally.</p>
    </div>
    """, unsafe_allow_html=True)

    glossary_df = pd.DataFrame(
        [{"Term": k, "Full Name": v} for k, v in GLOSSARY.items()]
    )
    search = st.text_input("🔎 Search glossary", "")
    if search:
        glossary_df = glossary_df[glossary_df["Term"].str.contains(search, case=False) |
                                   glossary_df["Full Name"].str.contains(search, case=False)]
    st.dataframe(glossary_df, width='stretch', hide_index=True)

# =============================================================================
# PAGE: KNOWN ISSUES
# =============================================================================
elif page == "⚠️ Known Issues":
    st.markdown("""
    <div class="hero">
        <h1>Known Issues & Limitations</h1>
        <p>Documented transparently — an honest account of what this project 
        knows it still needs to address.</p>
    </div>
    """, unsafe_allow_html=True)

    categories = sorted(set(c for c, _ in KNOWN_ISSUES))
    for cat in categories:
        st.subheader(cat)
        for c, issue in KNOWN_ISSUES:
            if c == cat:
                st.markdown(f"- {issue}")

# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif page == "👥 About":
    st.markdown("""
    <div class="hero">
        <h1>About This Project</h1>
        <p>Building neural machine translation for underrepresented Kenyan languages, 
        starting from base models that had never seen them before.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Team
    | Member | Focus |
    |---|---|
    | Selmah | NLLB fine-tuning for Ekegusii |
    | Rencia | mT5 fine-tuning for Ekegusii |
    | Steve | NLLB fine-tuning for Dholuo |
    | Trizzah | mT5 fine-tuning for Dholuo |
    | Patricia | mT5 fine-tuning for Somali |

    ### Approach
    Each language track adapts a multilingual base model (NLLB-200 or mT5) to a
    language the model has little or no prior exposure to — via new-token
    embedding initialization (for NLLB) or task-prefix fine-tuning (for
    mT5), trained on a cleaned parallel corpus of public service announcements.

    ### Data pipeline
    Raw PSA corpus → encoding/mojibake repair → boilerplate removal →
    code-switch detection & glossary → leak-safe train/val/test split →
    baseline fine-tune → error-driven oversampling → evaluation (BLEU, chrF,
    qualitative review, human evaluation).
    """)

st.markdown(
    '<div class="footer-note">Sauti — built for low-resource Kenyan language translation · '
    'Fine-tuned on NLLB-200 & mT5</div>',
    unsafe_allow_html=True,
)
import streamlit as st
import numpy as np
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Arabic:wght@300;400;600&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}

.stApp { background-color: #0a0a0f; }

/* Header */
.main-header {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2.5rem;
}
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8e8e8;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.main-subtitle {
    font-size: 0.9rem;
    color: #555570;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Track labels */
.track-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 0.6rem;
}

/* Input sections */
.input-section {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.input-section:hover { border-color: #2e2e4e; }

/* Result cards */
.result-card {
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    background: #0f0f1a;
}
.result-card.credible {
    border-left: 3px solid #00c896;
}
.result-card.fake {
    border-left: 3px solid #ff4466;
}
.result-card.authentic {
    border-left: 3px solid #00c896;
}
.result-card.tampered {
    border-left: 3px solid #ff4466;
}

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.verdict-credible { color: #00c896; }
.verdict-fake { color: #ff4466; }
.verdict-authentic { color: #00c896; }
.verdict-tampered { color: #ff4466; }

.confidence-bar-bg {
    height: 3px;
    background: #1e1e2e;
    border-radius: 2px;
    margin-top: 0.8rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}

/* Divider */
.track-divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}

/* Analyze button */
.stButton > button {
    background: #e8e8e8 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Text area */
.stTextArea > div > div > textarea {
    background: #0a0a0f !important;
    border: 1px solid #1e1e2e !important;
    color: #e8e8e8 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    font-size: 0.9rem !important;
    direction: rtl;
}
.stTextArea > div > div > textarea:focus {
    border-color: #2e2e4e !important;
    box-shadow: none !important;
}

/* File uploader */
.stFileUploader > div {
    background: #0a0a0f !important;
    border: 1px dashed #1e1e2e !important;
    border-radius: 4px !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding-top: 0 !important; max-width: 900px; }

/* Model info badge */
.model-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: #555570;
    border: 1px solid #1e1e2e;
    border-radius: 2px;
    padding: 0.15rem 0.4rem;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS (replace with your actual model loading)
# ─────────────────────────────────────────────
@st.cache_resource
def load_text_model():
    """
    Load your best text model here.
    Replace this with your actual LSTM model loading code.
    Example:
        from tensorflow.keras.models import load_model
        import pickle
        model = load_model('models/lstm_text_model.h5')
        tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
        return model, tokenizer
    """
    return None, None

@st.cache_resource
def load_image_model():
    """
    Load your best image model here.
    Replace this with your actual ELA + ResNet50 + XGBoost loading code.
    Example:
        import torch
        import joblib
        resnet = torch.load('models/resnet50_ela.pt')
        xgb = joblib.load('models/xgboost_image.pkl')
        return resnet, xgb
    """
    return None, None


def predict_text(text, model, tokenizer):
    """
    Replace with your actual text prediction logic.
    Should return (label, confidence) where:
        label: 'Credible' or 'Fake'
        confidence: float between 0 and 1
    """
    # ── PLACEHOLDER — replace with real inference ──
    import random
    label = random.choice(["Credible", "Fake"])
    confidence = random.uniform(0.65, 0.98)
    return label, confidence


def apply_ela(image: Image.Image, quality: int = 90) -> np.ndarray:
    """Apply Error Level Analysis to a PIL image."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')
    original = np.array(image.convert('RGB')).astype(np.float32)
    recomp = np.array(recompressed).astype(np.float32)
    ela = np.abs(original - recomp)
    ela = (ela / ela.max() * 255).astype(np.uint8)
    return ela


def predict_image(image: Image.Image, resnet, xgb):
    """
    Replace with your actual image prediction logic.
    Should return (label, confidence) where:
        label: 'Authentic' or 'Tampered'
        confidence: float between 0 and 1
    """
    # ── PLACEHOLDER — replace with real inference ──
    import random
    label = random.choice(["Authentic", "Tampered"])
    confidence = random.uniform(0.65, 0.98)
    return label, confidence


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Arabic Misinformation Detector</div>
    <div class="main-subtitle">Dual-Track · NLP + Computer Vision · Deep Learning</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="track-label">Track 01 — Text</div>', unsafe_allow_html=True)
    news_text = st.text_area(
        label="",
        placeholder="أدخل نص الخبر هنا...",
        height=200,
        label_visibility="collapsed"
    )
    st.markdown("""
    <div>
        <span class="model-badge">LSTM</span>
        <span class="model-badge">AFND · 606,912 articles</span>
        <span class="model-badge">95% acc</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="track-label">Track 02 — Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    st.markdown("""
    <div>
        <span class="model-badge">ELA + ResNet50 + XGBoost</span>
        <span class="model-badge">CASIA v1+v2</span>
        <span class="model-badge">92.89% acc</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("Analyze")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if analyze:
    has_text = bool(news_text and news_text.strip())
    has_image = uploaded_file is not None

    if not has_text and not has_image:
        st.warning("Please provide at least a text input or an image.")
    else:
        st.markdown('<hr class="track-divider">', unsafe_allow_html=True)
        st.markdown('<div class="track-label" style="margin-bottom:1.5rem;">Analysis Results</div>', unsafe_allow_html=True)

        text_model, tokenizer = load_text_model()
        image_model, xgb = load_image_model()

        res_col1, res_col2 = st.columns([1, 1], gap="large")

        # ── TEXT RESULT ──
        with res_col1:
            if has_text:
                with st.spinner("Analyzing text..."):
                    label, conf = predict_text(news_text, text_model, tokenizer)

                verdict_class = "credible" if label == "Credible" else "fake"
                verdict_color = "verdict-credible" if label == "Credible" else "verdict-fake"
                bar_color = "#00c896" if label == "Credible" else "#ff4466"
                conf_pct = int(conf * 100)

                st.markdown(f"""
                <div class="result-card {verdict_class}">
                    <div class="result-label">Text Classification</div>
                    <div class="result-verdict {verdict_color}">{label}</div>
                    <div style="font-size:0.8rem; color:#555570;">Confidence: {conf_pct}%</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill"
                             style="width:{conf_pct}%; background:{bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Text Classification</div>
                    <div style="color:#555570; font-size:0.85rem;">No text provided</div>
                </div>
                """, unsafe_allow_html=True)

        # ── IMAGE RESULT ──
        with res_col2:
            if has_image:
                image = Image.open(uploaded_file)

                with st.spinner("Applying ELA and analyzing image..."):
                    ela_map = apply_ela(image)
                    label, conf = predict_image(image, image_model, xgb)

                verdict_class = "authentic" if label == "Authentic" else "tampered"
                verdict_color = "verdict-authentic" if label == "Authentic" else "verdict-tampered"
                bar_color = "#00c896" if label == "Authentic" else "#ff4466"
                conf_pct = int(conf * 100)

                st.markdown(f"""
                <div class="result-card {verdict_class}">
                    <div class="result-label">Image Forgery Detection</div>
                    <div class="result-verdict {verdict_color}">{label}</div>
                    <div style="font-size:0.8rem; color:#555570;">Confidence: {conf_pct}%</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill"
                             style="width:{conf_pct}%; background:{bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("View ELA Map"):
                    st.image(ela_map, caption="Error Level Analysis — brighter areas indicate potential manipulation", use_container_width=True)
            else:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Image Forgery Detection</div>
                    <div style="color:#555570; font-size:0.85rem;">No image provided</div>
                </div>
                """, unsafe_allow_html=True)

        # ── COMBINED VERDICT (if both provided) ──
        if has_text and has_image:
            st.markdown('<hr class="track-divider">', unsafe_allow_html=True)
            t_label, t_conf = predict_text(news_text, text_model, tokenizer)
            i_label, i_conf = predict_image(Image.open(uploaded_file), image_model, xgb)

            flags = sum([t_label == "Fake", i_label == "Tampered"])
            if flags == 2:
                overall = "High Risk — Both text and image flagged"
                overall_color = "#ff4466"
            elif flags == 1:
                overall = "Partial Risk — One signal flagged"
                overall_color = "#ffaa00"
            else:
                overall = "Low Risk — No signals flagged"
                overall_color = "#00c896"

            st.markdown(f"""
            <div style="text-align:center; padding: 1.5rem 0;">
                <div class="result-label" style="margin-bottom:0.5rem;">Combined Assessment</div>
                <div style="font-size:1.1rem; font-weight:600; color:{overall_color};">{overall}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 1rem 0; color: #2e2e4e;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
            letter-spacing: 0.15em; border-top: 1px solid #1e1e2e; margin-top: 3rem;">
    TISHREEN UNIVERSITY · FACULTY OF INFORMATICS ENGINEERING · AI DEPARTMENT · 2024–2025
</div>
""", unsafe_allow_html=True)

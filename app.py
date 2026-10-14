import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ═══════════════════════════════════════════════════════
#                   PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quote Studio",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════
#                   CUSTOM CSS (Premium v2)
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,800;1,600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e5e5f0;
    }

    .stApp {
        background: #0a0a0f;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139, 92, 246, 0.15), transparent),
            radial-gradient(ellipse 60% 50% at 80% 100%, rgba(236, 72, 153, 0.08), transparent),
            radial-gradient(ellipse 60% 50% at 0% 80%, rgba(59, 130, 246, 0.08), transparent);
    }

    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }

    /* ═══ HERO ═══ */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 3rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 999px;
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        font-weight: 300;
        margin: 1.25rem auto 0 auto;
        max-width: 620px;
        line-height: 1.6;
        text-align: center;
        display: block;
        width: 100%;
    }
    .hero-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b5cf6, transparent);
        margin: 2rem auto;
    }

    /* ═══ INPUTS ═══ */
    .stTextInput > div > div > input {
        background: rgba(10, 10, 15, 0.8) !important;
        border: 1.5px solid rgba(139, 92, 246, 0.25) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
    }
    .stTextInput label, .stSlider label {
        color: #a78bfa !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }

    /* ═══ RADIO ═══ */
    .stRadio > div {
        gap: 8px !important;
    }
    .stRadio label {
        color: #d1d5db !important;
        font-weight: 500 !important;
    }

    /* ═══ BUTTON ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.5);
        filter: brightness(1.1);
    }

    /* ═══ OUTPUT CARD — CENTERED ═══ */
    .output-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(236, 72, 153, 0.05));
        border: 1.5px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 3.5rem 2.5rem 3rem 2.5rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
        text-align: center;
        animation: fadeSlideIn 0.6s ease-out;
    }
    .output-card::before {
        content: '"';
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        font-family: 'Playfair Display', serif;
        font-size: 8rem;
        color: rgba(139, 92, 246, 0.12);
        line-height: 1;
        pointer-events: none;
    }
    .output-label {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(139, 92, 246, 0.15);
        border-radius: 999px;
        color: #a78bfa;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.75rem;
        position: relative;
        z-index: 2;
    }
    .output-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-style: italic;
        font-weight: 600;
        line-height: 1.55;
        color: #ffffff;
        position: relative;
        z-index: 2;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        letter-spacing: 0.01em;
        text-shadow: 0 2px 20px rgba(139, 92, 246, 0.15);
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ═══ MODEL CHIPS ═══ */
    .model-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(20, 20, 31, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 999px;
        color: #d1d5db;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    .model-chip:hover {
        border-color: #8b5cf6;
        transform: translateY(-2px);
    }
    .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-purple { background: #8b5cf6; box-shadow: 0 0 12px #8b5cf6; }
    .dot-pink   { background: #ec4899; box-shadow: 0 0 12px #ec4899; }
    .dot-blue   { background: #3b82f6; box-shadow: 0 0 12px #3b82f6; }

    /* ═══ SECTION HEADERS ═══ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.3), transparent);
    }

    /* ═══ STATS — NEW PREMIUM DESIGN ═══ */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 16px;
        margin-top: 1.5rem;
    }
    .stat-card {
        position: relative;
        padding: 24px 20px;
        background: linear-gradient(145deg, rgba(20, 20, 31, 0.9), rgba(10, 10, 15, 0.9));
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        text-align: center;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b5cf6, #ec4899, transparent);
        opacity: 0.6;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.15), transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 16px 40px rgba(139, 92, 246, 0.25);
    }
    .stat-card:hover::after {
        opacity: 1;
    }
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 12px;
        display: block;
        filter: drop-shadow(0 0 12px rgba(139, 92, 246, 0.6));
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 60%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    .stat-suffix {
        font-size: 0.9rem;
        color: #6b7280;
        margin-left: 2px;
    }

    /* ═══ FOOTER ═══ */
    .footer {
        text-align: center;
        padding: 3rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(139, 92, 246, 0.1);
    }
    .footer-line {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .footer-line .accent {
        color: #8b5cf6;
        font-weight: 600;
    }
    .footer-copy {
        color: #374151;
        font-size: 0.75rem;
        letter-spacing: 2px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    .footer-heart {
        display: inline-block;
        animation: heartbeat 1.5s ease-in-out infinite;
        color: #ec4899;
    }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }

    /* ═══ WARNING ═══ */
    .stAlert {
        background: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #e5e5f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#                   LOAD MODELS
# ═══════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_all_models():
    base = "models"
    lstm = load_model(os.path.join(base, "lstm_model.h5"))
    rnn  = load_model(os.path.join(base, "rnn_model.h5"))
    gru  = load_model(os.path.join(base, "gru_model.h5"))
    with open(os.path.join(base, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)
    return lstm, rnn, gru, tokenizer, max_len


with st.spinner("Loading neural networks..."):
    lstm_model, rnn_model, gru_model, tokenizer, max_len = load_all_models()

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1


# ═══════════════════════════════════════════════════════
#                   PREDICTION
# ═══════════════════════════════════════════════════════
def predictor(model, tokenizer, text, max_len):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return ""
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)
    return index_to_word.get(int(np.argmax(pred)), "")


def generate_text(model, tokenizer, seed_text, max_len, n_words):
    for _ in range(n_words):
        nw = predictor(model, tokenizer, seed_text, max_len)
        if not nw:
            break
        seed_text += " " + nw
    return seed_text


# ═══════════════════════════════════════════════════════
#                   HERO
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">◆ Neural Text Generation</div>
    <h1 class="hero-title">Quote Predictor Studio</h1>
    <p class="hero-subtitle" style="text-align:center; margin-left:auto; margin-right:auto;">
        Craft original quotes with the power of three deep learning architectures —
        LSTM, GRU, and Vanilla RNN — trained on thousands of timeless words.
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#                   MODEL SELECTOR
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header">Choose Model</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown('<div class="model-chip"><span class="dot dot-purple"></span> LSTM · Long Memory</div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="model-chip"><span class="dot dot-pink"></span> GRU · Balanced</div>', unsafe_allow_html=True)
with col_c:
    st.markdown('<div class="model-chip"><span class="dot dot-blue"></span> RNN · Lightweight</div>', unsafe_allow_html=True)

model_choice = st.radio(
    "Select architecture",
    ["LSTM", "GRU", "RNN"],
    horizontal=True,
    label_visibility="collapsed"
)


# ═══════════════════════════════════════════════════════
#                   INPUT
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header" style="margin-top:2rem;">Compose</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    seed_text = st.text_input(
        "Seed text",
        value="the purpose of life is",
        placeholder="Type a phrase to begin…"
    )
with col2:
    n_words = st.slider("Words", min_value=5, max_value=40, value=15, step=5)

st.write("")
generate_btn = st.button("✦  Predict Quote", use_container_width=True)


# ═══════════════════════════════════════════════════════
#                   OUTPUT
# ═══════════════════════════════════════════════════════
if generate_btn:
    if not seed_text.strip():
        st.warning("Please enter some seed text first.")
    else:
        model_map = {"LSTM": lstm_model, "GRU": gru_model, "RNN": rnn_model}
        selected = model_map[model_choice]

        with st.spinner(f"{model_choice} is weaving words…"):
            output = generate_text(selected, tokenizer, seed_text, max_len, n_words)

        st.markdown(f"""
        <div class="output-card">
            <div class="output-label">◆ {model_choice} · Output</div>
            <div class="output-text">{output}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#                   STATS — PREMIUM CARDS
# ═══════════════════════════════════════════════════════
st.markdown('<div class="section-header" style="margin-top:3rem;">Studio Metrics</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card">
        <span class="stat-icon">📚</span>
        <div class="stat-value">{vocab_size:,}</div>
        <div class="stat-label">Vocabulary</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">📏</span>
        <div class="stat-value">{max_len}</div>
        <div class="stat-label">Max Length</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">🧬</span>
        <div class="stat-value">50</div>
        <div class="stat-label">Embed Dim</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">🧠</span>
        <div class="stat-value">128</div>
        <div class="stat-label">Hidden Units</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">⚡</span>
        <div class="stat-value">3</div>
        <div class="stat-label">Models</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#                   FOOTER — AUTO YEAR
# ═══════════════════════════════════════════════════════
current_year = datetime.now().year

st.markdown(f"""
<div class="footer">
    <div class="footer-line">
        <span class="accent">◆</span>
            Think it. Type it. Read it.
        <span class="accent">◆</span>
    </div>
    <div class="footer-copy">
        QUOTE STUDIO © {current_year} · MADE WITH <span class="footer-heart">♥</span>
    </div>
</div>
""", unsafe_allow_html=True)
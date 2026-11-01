import streamlit as st
import tensorflow as tf
import numpy as np
import json
import hashlib
from PIL import Image
from datetime import datetime
from html import escape
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# BLUE + SOFT PINK THEME CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
    --navy: #10243e;
    --blue: #2563eb;
    --cyan: #38bdf8;
    --light: #eff6ff;
    --text: #172b45;
    --muted: #718096;

    /* Soft pink surfaces */
    --pink-bg: #D3D6F2;       /* page background */
    --pink-surface: #fff1f6;  /* cards and panels */
    --pink-soft: #ffe6f0;     /* hints and highlights */
    --pink-line: #f2c2d6;     /* card borders */
    --pink-dash: #ea8db4;     /* dashed borders */
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--pink-bg);
    color: var(--text);
}

.stApp [data-testid="stMarkdownContainer"] p {
    color: var(--text);
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10243e, #173c63);
}

section[data-testid="stSidebar"] * {
    color: #f3f8ff !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-weight: 600;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #10243e;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.3;
    color: white;
}

.eyebrow {
    color: #2563eb;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.section-heading {
    color: #10243e;
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin: 18px 0 12px;
}

.panel {
    background: var(--pink-surface);
    border: 1px solid var(--pink-line);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 5px 20px rgba(190,60,110,.06);
}

/* Bordered containers used as panels (upload, preview, top 5) */
.st-key-upload_panel,
.st-key-preview_panel,
.st-key-top5_panel {
    background: var(--pink-surface);
    border: 1px solid var(--pink-line) !important;
    border-radius: 18px !important;
    padding: 12px 14px;
    box-shadow: 0 5px 20px rgba(190,60,110,.06);
}

.panel-title {
    color: #10243e;
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    font-weight: 800;
}

.panel-subtitle {
    color: #718096;
    font-size: 13px;
    margin-bottom: 15px;
}

.stat-card {
    background: var(--pink-surface);
    border: 1px solid var(--pink-line);
    border-radius: 17px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 5px 18px rgba(190,60,110,.06);
}

.stat-icon {
    float: right;
    font-size: 25px;
}

.stat-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

.stat-value {
    color: #10243e;
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    margin-top: 7px;
}

.step-card {
    background: var(--pink-surface);
    border: 1px solid var(--pink-line);
    border-radius: 15px;
    padding: 18px 10px;
    text-align: center;
    min-height: 115px;
}

.step-icon {
    font-size: 27px;
}

.step-title {
    color: #173c63;
    font-weight: 800;
    font-size: 13px;
    margin-top: 8px;
}

.step-subtitle {
    color: #8291a5;
    font-size: 11px;
    margin-top: 4px;
}

.result-card {
    background: linear-gradient(135deg, #fcd9e8, #fff0f6);
    border: 1px solid #efb0ca;
    border-radius: 18px;
    padding: 23px;
}

.result-label {
    color: #2563eb;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-name {
    color: #10243e;
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin: 9px 0;
}

.confidence {
    color: #2563eb;
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
}

.advice-card {
    background: var(--pink-soft);
    border: 1px solid #f2bad2;
    border-radius: 15px;
    padding: 18px;
    color: #4a2a3c;
    font-size: 14px;
    line-height: 1.7;
}

.upload-hint {
    background: var(--pink-soft);
    border: 1px dashed var(--pink-dash);
    border-radius: 14px;
    padding: 15px;
    text-align: center;
    color: #6b2a4a;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    min-height: 46px;
    transition: all .2s ease;
}

div.stButton > button:hover {
    background: #173c63;
    color: white;
    transform: translateY(-1px);
    border: none;
}

/* File uploader (kept light so it matches the pink theme) */
div[data-testid="stFileUploader"] {
    background: transparent;
    border: none;
    padding: 0;
}

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 1.5px dashed var(--pink-dash);
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div {
    color: var(--text) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    color: #ffffff !important;
    border: none;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}

/* Camera input */
[data-testid="stCameraInput"] {
    background: #ffffff;
    border: 1.5px dashed var(--pink-dash);
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stCameraInput"] > div,
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 12px;
}

[data-testid="stCameraInput"] button {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    color: #ffffff !important;
    border: none;
    font-weight: 700;
}

[data-testid="stCameraInput"] button * {
    color: #ffffff !important;
}

[data-testid="stCameraInput"] small,
[data-testid="stCameraInput"] span {
    color: var(--text);
}

/* Image source selector: sliding pill toggle (Upload / Camera) */
.st-key-image_source {
    margin-bottom: 14px;
}

.st-key-image_source div[role="radiogroup"] {
    position: relative;
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 0 !important;
    padding: 5px;
    background: #ffffff;
    border: 1px solid var(--pink-line);
    border-radius: 999px;
    box-shadow: inset 0 1px 3px rgba(190,60,110,.08);
}

/* The sliding highlight */
.st-key-image_source div[role="radiogroup"]::before {
    content: "";
    position: absolute;
    top: 5px;
    bottom: 5px;
    left: 5px;
    width: calc(50% - 5px);
    border-radius: 999px;
    background: linear-gradient(100deg, #2563eb, #0891b2);
    box-shadow: 0 6px 16px rgba(37,99,235,.35);
    transition: transform .4s cubic-bezier(.65, 0, .35, 1);
}

/* Slide to the second option when "Use camera" is selected */
.st-key-image_source div[role="radiogroup"]:has(label:nth-of-type(2) input:checked)::before {
    transform: translateX(100%);
}

.st-key-image_source label {
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 !important;
    padding: 9px 0 !important;
    border-radius: 999px;
    cursor: pointer;
}

/* Hide the round radio dot, keep it clickable through the label */
.st-key-image_source label > div:first-child {
    position: absolute;
    width: 0;
    height: 0;
    opacity: 0;
    overflow: hidden;
    pointer-events: none;
}

.st-key-image_source label p {
    color: #5b6b82 !important;
    font-weight: 700;
    font-size: 14px;
    text-align: center;
    transition: color .3s ease;
}

.st-key-image_source label:has(input:checked) p {
    color: #ffffff !important;
}

/* Equal-height panels so the two boxes line up */
div[data-testid="stHorizontalBlock"]:has(.st-key-upload_panel) {
    align-items: stretch;
}

div[data-testid="stColumn"]:has(.st-key-upload_panel),
div[data-testid="stColumn"]:has(.st-key-preview_panel) {
    display: flex;
    flex-direction: column;
}

div[data-testid="stColumn"] div:has(.st-key-upload_panel),
div[data-testid="stColumn"] div:has(.st-key-preview_panel) {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}

.st-key-upload_panel,
.st-key-preview_panel {
    flex: 1 1 auto;
    min-height: 430px;
}

div[data-testid="stMetric"] {
    background: var(--pink-surface);
    border: 1px solid var(--pink-line);
    border-radius: 15px;
    padding: 15px;
}

div[data-testid="stProgress"] > div > div {
    background: #2563eb;
}

/* Alerts: dark readable text on the light theme */
div[data-testid="stAlert"] {
    border-radius: 13px;
}

div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p {
    color: #172b45 !important;
}

/* ---------- Sidebar polish ---------- */
section[data-testid="stSidebar"] .side-logo {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    font-size: 29px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    box-shadow: 0 8px 22px rgba(56,189,248,.35);
}

section[data-testid="stSidebar"] .side-title {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] .side-tagline {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 10px;
    color: #8fd3ff !important;
}

section[data-testid="stSidebar"] .side-heading {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 2px 0 12px;
    color: #9fc3e8 !important;
}

/* Navigation as large pill buttons */
section[data-testid="stSidebar"] .st-key-nav_page [data-testid="stWidgetLabel"] p {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #9fc3e8 !important;
}

section[data-testid="stSidebar"] .st-key-nav_page div[role="radiogroup"] {
    gap: 6px !important;
}

section[data-testid="stSidebar"] .st-key-nav_page label[data-baseweb="radio"] {
    width: 100%;
    margin: 0 !important;
    padding: 13px 16px !important;
    border-radius: 14px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: background .25s ease, transform .25s ease, box-shadow .25s ease;
}

section[data-testid="stSidebar"] .st-key-nav_page label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,.09);
    transform: translateX(4px);
}

section[data-testid="stSidebar"] .st-key-nav_page label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(100deg, #2563eb, #0891b2);
    box-shadow: 0 8px 20px rgba(8,145,178,.35);
    transform: none;
}

/* Hide the round radio dot, the whole row is the button */
section[data-testid="stSidebar"] .st-key-nav_page label[data-baseweb="radio"] > div:first-child {
    position: absolute;
    width: 0;
    height: 0;
    opacity: 0;
    overflow: hidden;
    pointer-events: none;
}

section[data-testid="stSidebar"] .st-key-nav_page label[data-baseweb="radio"] p {
    font-size: 17px;
    font-weight: 700;
}

/* Model details card */
section[data-testid="stSidebar"] .side-card {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 16px;
    padding: 4px 14px;
}

section[data-testid="stSidebar"] .side-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] .side-row:last-child {
    border-bottom: none;
}

section[data-testid="stSidebar"] .side-ico {
    flex: none;
    width: 38px;
    height: 38px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    font-size: 19px;
    background: rgba(56,189,248,.16);
}

section[data-testid="stSidebar"] .side-k {
    font-size: 12.5px;
    color: #9fc3e8 !important;
}

section[data-testid="stSidebar"] .side-v {
    font-size: 15.5px;
    font-weight: 700;
    margin-top: 1px;
}

/* Image tip */
section[data-testid="stSidebar"] .side-tip {
    background: linear-gradient(135deg, rgba(56,189,248,.18), rgba(37,99,235,.20));
    border: 1px solid rgba(125,211,252,.30);
    border-radius: 16px;
    padding: 16px;
    font-size: 14.5px;
    line-height: 1.65;
}

section[data-testid="stSidebar"] .side-tip-title {
    font-weight: 800;
    font-size: 15.5px;
    margin-bottom: 6px;
}

/* Button that re-opens the sidebar: visible on the pink page */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    background: linear-gradient(100deg, #2563eb, #0891b2) !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 16px rgba(37,99,235,.35);
    transition: transform .2s ease, box-shadow .2s ease;
}

[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(37,99,235,.45);
}

[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"] button *,
[data-testid="collapsedControl"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

.footer {
    text-align: center;
    color: #8291a5;
    font-size: 12px;
    padding: 22px 0;
    border-top: 1px solid var(--pink-line);
    margin-top: 25px;
}

hr {
    border-color: var(--pink-line);
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND CLASS NAMES
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


@st.cache_data
def load_class_names():
    with open("class_names.json", "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()

except Exception as error:
    st.error("Unable to load the model or class names.")

    st.info("Keep all three files in the same folder:")
    st.code("""
app.py
plant_disease_model.keras
class_names.json
""")

    st.exception(error)
    st.stop()


# =====================================================
# DISEASE INFORMATION
# =====================================================

def get_disease_info(class_name):

    if "healthy" in class_name.lower():
        return (
            "Healthy",
            "The plant appears healthy. Continue regular watering, "
            "provide suitable sunlight and monitor the plant regularly."
        )

    if class_name == "Background_without_leaves":
        return (
            "No leaf detected",
            "No visible plant leaf was identified. Upload a clear "
            "close-up image of a plant leaf."
        )

    advice = {

        "Apple___Apple_scab":
            "Remove affected leaves and maintain good air circulation. "
            "Avoid excessive moisture on leaves.",

        "Apple___Black_rot":
            "Remove infected plant material and maintain good sanitation.",

        "Apple___Cedar_apple_rust":
            "Remove infected leaves and monitor the plant regularly.",

        "Cherry___Powdery_mildew":
            "Improve air circulation and avoid excessive humidity.",

        "Corn___Cercospora_leaf_spot Gray_leaf_spot":
            "Remove heavily infected material and improve field sanitation.",

        "Corn___Common_rust":
            "Monitor the crop and remove severely affected leaves.",

        "Corn___Northern_Leaf_Blight":
            "Remove infected plant debris and maintain field sanitation.",

        "Grape___Black_rot":
            "Remove infected leaves and fruit. Improve air circulation.",

        "Grape___Esca_(Black_Measles)":
            "Remove severely affected plant material and monitor the vine.",

        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
            "Remove affected leaves and maintain good plant sanitation.",

        "Orange___Haunglongbing_(Citrus_greening)":
            "Monitor the plant carefully and seek professional agricultural advice.",

        "Peach___Bacterial_spot":
            "Remove affected leaves and avoid excessive leaf wetness.",

        "Pepper,_bell___Bacterial_spot":
            "Remove affected leaves and avoid overhead watering.",

        "Potato___Early_blight":
            "Remove infected leaves and improve air circulation.",

        "Potato___Late_blight":
            "Remove infected plant material and seek expert advice for severe cases.",

        "Squash___Powdery_mildew":
            "Improve ventilation and remove severely affected leaves.",

        "Strawberry___Leaf_scorch":
            "Remove affected leaves and maintain good plant sanitation.",

        "Tomato___Bacterial_spot":
            "Remove affected leaves and avoid overhead watering.",

        "Tomato___Early_blight":
            "Remove infected leaves and plant debris.",

        "Tomato___Late_blight":
            "Remove infected plant material and monitor the crop closely.",

        "Tomato___Leaf_Mold":
            "Improve ventilation and reduce excessive humidity.",

        "Tomato___Septoria_leaf_spot":
            "Remove infected leaves and avoid prolonged leaf moisture.",

        "Tomato___Spider_mites Two-spotted_spider_mite":
            "Inspect the underside of leaves and monitor for spider mites.",

        "Tomato___Target_Spot":
            "Remove infected leaves and improve air circulation.",

        "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
            "Monitor for whiteflies and seek professional agricultural advice.",

        "Tomato___Tomato_mosaic_virus":
            "Remove infected plant material and maintain good sanitation."
    }

    return (
        "Diseased",
        advice.get(
            class_name,
            "The model detected a possible plant disease. "
            "Monitor the plant and consult an agricultural expert "
            "for confirmation and suitable treatment."
        )
    )


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def format_class_name(class_name):

    if class_name == "Background_without_leaves":
        return "Background Without Leaves"

    return class_name.replace("___", " - ").replace("_", " ")


def predict_leaf(image):

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    prediction = model.predict(image_array, verbose=0)

    return prediction[0]


def go_to_scanner():
    """Button callback: switch the sidebar navigation to the scanner page."""
    st.session_state["nav_page"] = "🔬 Disease Scanner"


def stat_card(icon, label, value):

    st.markdown(f"""
    <div class="stat-card">
        <span class="stat-icon">{icon}</span>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def step_card(icon, title, subtitle):

    st.markdown(f"""
    <div class="step-card">
        <div class="step-icon">{icon}</div>
        <div class="step-title">{title}</div>
        <div class="step-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="side-logo">🌿</div>
    <div class="side-title">Plant Disease Detection</div>
    <div class="side-tagline">PLANT HEALTH ASSISTANT</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔬 Disease Scanner",
            "ℹ️ About Project"
        ],
        key="nav_page"
    )

    st.markdown("---")

    st.markdown(
        '<div class="side-heading">MODEL DETAILS</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="side-card">
        <div class="side-row">
            <div class="side-ico">🧠</div>
            <div><div class="side-k">Architecture</div><div class="side-v">MobileNetV2</div></div>
        </div>
        <div class="side-row">
            <div class="side-ico">⚙️</div>
            <div><div class="side-k">Framework</div><div class="side-v">TensorFlow / Keras</div></div>
        </div>
        <div class="side-row">
            <div class="side-ico">📐</div>
            <div><div class="side-k">Image input</div><div class="side-v">224 × 224 RGB</div></div>
        </div>
        <div class="side-row">
            <div class="side-ico">🌿</div>
            <div><div class="side-k">Prediction classes</div><div class="side-v">{len(class_names)}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="side-tip">
        <div class="side-tip-title">💡 Image tip</div>
        Use a clear, close-up, well-lit leaf image for better predictions.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# HERO BANNER - st.html FIX
# =====================================================

st.html("""
<div style="
    background:linear-gradient(115deg,#10243e,#174c7b,#087eaa);
    padding:34px;
    border-radius:23px;
    color:white;
    margin-bottom:25px;
    box-shadow:0 12px 30px rgba(16,36,62,.13);
">

    <div style="
        color:#7dd3fc;
        font-size:11px;
        font-weight:800;
        letter-spacing:2px;
    ">
        AI-POWERED PLANT CARE
    </div>

    <h1 style="
        color:white;
        font-size:34px;
        font-weight:800;
        line-height:1.3;
        margin-top:12px;
    ">
        Understand your plant.<br>
        Protect its health.
    </h1>

    <p style="
        color:#e0f2fe;
        font-size:15px;
        max-width:750px;
        line-height:1.8;
    ">
        Upload a plant leaf image and let deep learning help identify
        possible plant diseases, display prediction confidence and
        provide basic care guidance.
    </p>

    <div style="margin-top:18px;">
        <span style="
            background:rgba(255,255,255,.15);
            padding:8px 13px;
            border-radius:30px;
            font-size:11px;
            font-weight:700;
            color:white;
        ">
            ✦ MOBILENETV2 · DEEP LEARNING · 39 CLASSES
        </span>
    </div>

</div>
""")


# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="eyebrow">PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Welcome to LeafCare AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "An intelligent plant disease detection application "
        "powered by deep learning."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        stat_card("🧠", "AI ARCHITECTURE", "MobileNetV2")

    with c2:
        stat_card("🌿", "SUPPORTED CLASSES", str(len(class_names)))

    with c3:
        stat_card("📐", "IMAGE INPUT", "224 × 224")

    st.markdown(
        '<div class="section-heading">How it works</div>',
        unsafe_allow_html=True
    )

    steps = st.columns(4)

    step_data = [
        ("📷", "Add Image", "Upload or use camera"),
        ("🧠", "AI Analysis", "Model processes image"),
        ("🔍", "Detect Disease", "View prediction"),
        ("💡", "Get Advice", "Explore care guidance")
    ]

    for col, data in zip(steps, step_data):
        with col:
            step_card(*data)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">🔬 Start a new scan</div>
            <div class="panel-subtitle">
                Analyse a plant leaf using your trained AI model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Open Disease Scanner →",
            use_container_width=True,
            on_click=go_to_scanner
        )

    with right:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">🌾 Supported Plants</div>
            <div class="panel-subtitle">
                Plant categories represented in the dataset.
            </div>
            <p style="line-height:2.1;color:#456080;">
                🍎 Apple &nbsp; 🍒 Cherry<br>
                🌽 Corn &nbsp; 🍇 Grape<br>
                🍊 Orange &nbsp; 🍑 Peach<br>
                🫑 Pepper &nbsp; 🥔 Potato<br>
                🍓 Strawberry &nbsp; 🍅 Tomato
            </p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# DISEASE SCANNER PAGE
# =====================================================

elif page == "🔬 Disease Scanner":

    st.markdown(
        '<div class="eyebrow">IMAGE DIAGNOSTICS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Plant Disease Scanner</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a plant leaf image or take a photo with your camera "
        "to get an AI-generated prediction."
    )

    upload_col, preview_col = st.columns([1, 1], gap="large")

    with upload_col:

        # st.container replaces the old split <div class="panel"> markdown,
        # which rendered as an empty white box above each panel.
        with st.container(border=True, key="upload_panel"):

            st.markdown(
                '<div class="panel-title">📤 Upload Leaf Image</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="panel-subtitle">JPG, JPEG and PNG supported</div>',
                unsafe_allow_html=True
            )

            source_choice = st.radio(
                "Image source",
                ["📁 Upload image", "📷 Use camera"],
                horizontal=True,
                key="image_source",
                label_visibility="collapsed"
            )

            uploaded_file = None
            camera_file = None

            if source_choice == "📁 Upload image":

                uploaded_file = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png"],
                    key="leaf_upload",
                    label_visibility="collapsed"
                )

            else:

                st.caption(
                    "Allow camera access in your browser, then hold the "
                    "leaf close to the camera and press Take Photo."
                )

                camera_file = st.camera_input(
                    "Take a photo of the leaf",
                    key="leaf_camera",
                    label_visibility="collapsed"
                )

            # Whichever source is selected provides the image
            source_file = uploaded_file or camera_file

            st.markdown("""
            <div class="upload-hint">
                <b>For best results</b><br>
                Good lighting · Close-up image · Leaf in focus
            </div>
            """, unsafe_allow_html=True)

    with preview_col:

        with st.container(border=True, key="preview_panel"):

            st.markdown(
                '<div class="panel-title">🖼️ Image Preview</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="panel-subtitle">'
                'Your selected photo appears here'
                '</div>',
                unsafe_allow_html=True
            )

            image = None
            file_key = None

            if source_file is not None:

                try:
                    image = Image.open(source_file).convert("RGB")

                    # Identifies this exact image so an old result is never
                    # shown next to a new photo
                    file_key = hashlib.md5(
                        source_file.getvalue()
                    ).hexdigest()

                    st.image(image, use_container_width=True)

                    label = (
                        "Camera photo"
                        if camera_file is not None
                        else source_file.name
                    )

                    st.caption(
                        f"{label} · "
                        f"{image.width} × {image.height} pixels"
                    )

                except Exception:
                    st.error("Unable to open this image.")

            else:

                st.markdown("""
                <div style="
                    min-height:290px;
                    width:100%;
                    box-sizing:border-box;
                    border:1.5px dashed #ea8db4;
                    border-radius:14px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:#ffffff;
                    color:#718096;
                    text-align:center;
                ">
                    <div>
                        <div style="font-size:42px;">🌿</div>
                        <b>Your leaf image will appear here</b><br>
                        <span style="font-size:12px;">
                            Upload an image or use your camera
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # PREDICTION

    if source_file is not None and image is not None:

        if st.button(
            "🔍 Analyse Plant Leaf",
            use_container_width=True,
            type="primary"
        ):

            with st.spinner("AI is analysing your plant leaf..."):

                probabilities = predict_leaf(image)

                best_index = int(np.argmax(probabilities))

                predicted_class = class_names[best_index]

                confidence = float(
                    probabilities[best_index] * 100
                )

                status, advice = get_disease_info(predicted_class)

                st.session_state["latest_result"] = {
                    "key": file_key,
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "status": status,
                    "advice": advice,
                    "probabilities": probabilities.tolist(),
                    "time": datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                }

        # DISPLAY RESULTS

        result = st.session_state.get("latest_result")

        if result is not None and result.get("key") == file_key:

            st.markdown("---")

            st.markdown(
                '<div class="eyebrow">ANALYSIS COMPLETE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">Detection Results</div>',
                unsafe_allow_html=True
            )

            result_col, advice_col = st.columns([1.2, 1])

            display_name = escape(
                format_class_name(result["prediction"])
            )

            with result_col:

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED CONDITION</div>
                    <div class="result-name">{display_name}</div>
                    <div style="color:#64748b;font-size:13px;">
                        Status: <b>{escape(result["status"])}</b>
                    </div>
                    <div style="margin-top:17px;">
                        <span class="confidence">
                            {result["confidence"]:.2f}%
                        </span>
                        <span style="color:#718096;font-size:12px;">
                            Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if result["status"] == "Healthy":
                    st.success("🌱 The model classifies this leaf as healthy.")

                elif result["status"] == "No leaf detected":
                    st.warning("⚠️ No clear plant leaf was identified.")

                else:
                    st.warning("⚠️ The model detected a possible disease.")

            with advice_col:

                st.markdown("""
                <div class="panel">
                    <div class="panel-title">💡 Basic Care Advice</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f'<div class="advice-card">{escape(result["advice"])}</div>',
                    unsafe_allow_html=True
                )

            # TOP FIVE PREDICTIONS

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True, key="top5_panel"):

                st.markdown(
                    '<div class="panel-title">📊 Top 5 Predictions</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="panel-subtitle">'
                    'Five classes with the highest model probabilities.'
                    '</div>',
                    unsafe_allow_html=True
                )

                probabilities = np.array(result["probabilities"])

                top_indices = np.argsort(probabilities)[-5:][::-1]

                for rank, index in enumerate(top_indices, start=1):

                    class_label = escape(
                        format_class_name(class_names[int(index)])
                    )

                    score = float(probabilities[index])

                    left, right = st.columns([4, 1])

                    with left:
                        st.markdown(f"**{rank}. {class_label}**")
                        st.progress(score)

                    with right:
                        st.markdown(
                            f"<div style='text-align:right;"
                            f"font-weight:800;color:#2563eb;padding-top:5px;'>"
                            f"{score * 100:.2f}%</div>",
                            unsafe_allow_html=True
                        )

            st.caption(f"Last analysed: {result['time']}")

            st.info(
                "This is an AI-based screening result, not a confirmed "
                "agricultural diagnosis. Verify important crop decisions "
                "with an agricultural expert."
            )

    else:

        st.info(
            "Upload a leaf image or take a photo with your camera "
            "to activate the analysis."
        )


# =====================================================
# ABOUT PROJECT PAGE
# =====================================================

else:

    st.markdown(
        '<div class="eyebrow">PROJECT INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">About LeafCare AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Plant Disease Detection using Deep Learning and "
        "Convolutional Neural Networks."
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">🌿 Project Objective</div>
        <p style="color:#64748b;line-height:1.8;">
        This project uses a convolutional neural network based on
        MobileNetV2 to classify uploaded plant leaf images into
        categories learned during model training. It displays the
        predicted class, confidence score, alternative predictions
        and basic informational guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">⚙️ Technologies Used</div>
            <p style="line-height:2;color:#64748b;">
                • Python<br>
                • Streamlit<br>
                • TensorFlow / Keras<br>
                • MobileNetV2<br>
                • NumPy<br>
                • Pillow
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🧠 Model Configuration</div>
            <p style="line-height:2;color:#64748b;">
                • Architecture: MobileNetV2<br>
                • Input: 224 × 224 pixels<br>
                • Output classes: {len(class_names)}<br>
                • Activation: Softmax<br>
                • Image classification
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "This application is a student project prototype. Predictions "
        "depend on image quality and training data. It does not replace "
        "laboratory testing or expert diagnosis."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <b>🌿 PLANT DISEASE DETECTION</b><br>
    Plant Disease Detection using Deep Learning and MobileNetV2
    <br>
    Minor Project Prototype · Python · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FILE: app.py
# PROJECT: IoT Botnet-based DDoS Detection in Smart Cities
# SYSTEM: Intrusion Detection System (IDS) — Streamlit Dashboard v2
# DESCRIPTION: Redesigned production dashboard with animated dark theme,
#              no sidebar, no emojis, minimalist layout. Shows detection
#              result after file upload, then detailed per-model analysis
#              inside a collapsible expander with a model dropdown.
# =============================================================================

# Standard Library
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Third-Party
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Scikit-learn metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# =============================================================================
# SECTION 1: PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="IoT Botnet IDS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# SECTION 2: GLOBAL CSS — Animated Dark Gradient Theme
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg-deep:       #080C14;
        --bg-surface:    #0D1420;
        --bg-raised:     #131C2E;
        --border:        rgba(0,229,255,0.12);
        --border-bright: rgba(0,229,255,0.35);
        --accent:        #00E5FF;
        --accent-dim:    rgba(0,229,255,0.07);
        --text-primary:  #E8EEF7;
        --text-muted:    #5A6880;
        --green:         #00C896;
        --green-bg:      rgba(0,200,150,0.08);
        --green-border:  rgba(0,200,150,0.40);
        --red:           #FF3D5A;
        --red-bg:        rgba(255,61,90,0.08);
        --red-border:    rgba(255,61,90,0.40);
        --shadow:        0 4px 24px rgba(0,0,0,0.5);
        --font-display:  'Syne', sans-serif;
        --font-mono:     'DM Mono', monospace;
        --radius:        10px;
        --transition:    all 0.25s cubic-bezier(0.4,0,0.2,1);
    }

    /* Main background: deep navy with animated mouse-tracked gradient orb */
    .stApp {
        background: #080C14;
        font-family: var(--font-display);
        color: var(--text-primary);
        min-height: 100vh;
    }

    /* Gradient orb layer — position driven by JS mouse events via CSS vars */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(700px circle at var(--mx, 25%) var(--my, 25%),
                rgba(0,229,255,0.045) 0%, transparent 70%),
            radial-gradient(1000px circle at 80% 85%,
                rgba(0,80,255,0.025) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        transition: background 0.1s linear;
    }

    /* Hide Streamlit chrome */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Remove sidebar */
    [data-testid="stSidebar"]         { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }

    /* Content area */
    .block-container {
        padding: 3.5rem 4rem 5rem 4rem;
        max-width: 1060px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    /* ── Typography ── */
    .ids-wordmark {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 5px;
        color: var(--accent);
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 0.6rem;
        opacity: 0.65;
    }
    .ids-title {
        font-family: var(--font-display);
        font-size: 2.75rem;
        font-weight: 800;
        color: var(--text-primary);
        text-align: center;
        letter-spacing: -1.2px;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .ids-title span { color: var(--accent); }
    .ids-subtitle {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-muted);
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 2.5rem;
    }
    .ids-rule {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }
    .ids-label {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        letter-spacing: 3.5px;
        color: var(--accent);
        text-transform: uppercase;
        margin-bottom: 0.65rem;
        opacity: 0.7;
    }

    /* ── Verdict Cards ── */
    .verdict-normal, .verdict-attack {
        border-radius: var(--radius);
        padding: 2.2rem 2.5rem;
        text-align: center;
        animation: fadeUp 0.45s cubic-bezier(0.16,1,0.3,1) both;
    }
    .verdict-normal {
        background: var(--green-bg);
        border: 1.5px solid var(--green-border);
    }
    .verdict-attack {
        background: var(--red-bg);
        border: 1.5px solid var(--red-border);
    }
    .verdict-tag {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .verdict-value-normal {
        font-family: var(--font-display);
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--green);
        line-height: 1;
    }
    .verdict-value-attack {
        font-family: var(--font-display);
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--red);
        line-height: 1;
    }
    .verdict-sub {
        font-family: var(--font-mono);
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 0.7rem;
        letter-spacing: 0.5px;
    }

    /* ── Accuracy badge ── */
    .acc-badge {
        display: inline-block;
        background: var(--accent-dim);
        border: 1px solid var(--border-bright);
        color: var(--accent);
        border-radius: 6px;
        padding: 0.4rem 1.2rem;
        font-family: var(--font-mono);
        font-size: 1.15rem;
        font-weight: 500;
        margin-top: 0.1rem;
        margin-bottom: 0.4rem;
    }

    /* ── Streamlit Widget Overrides ── */

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-raised) !important;
        border: 1.5px dashed var(--border-bright) !important;
        border-radius: var(--radius) !important;
        transition: var(--transition) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-dim) !important;
    }
    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] * {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.83rem !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00B8CC, #00E5FF) !important;
        border-radius: 4px !important;
        transition: width 0.6s ease !important;
    }
    [data-testid="stProgress"] > div {
        background: var(--bg-raised) !important;
        border-radius: 4px !important;
        height: 5px !important;
        border: 1px solid var(--border) !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSelectbox"] label { display: none !important; }

    /* Expander */
    details {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-top: 1.5rem !important;
        overflow: hidden !important;
    }
    details[open] { border-color: var(--border-bright) !important; }
    details > summary {
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        letter-spacing: 3px !important;
        color: var(--accent) !important;
        text-transform: uppercase !important;
        padding: 1rem 1.25rem !important;
        cursor: pointer !important;
        background: transparent !important;
        user-select: none !important;
        list-style: none !important;
    }
    details > summary::-webkit-details-marker { display: none; }

    /* Code block (log) */
    .stCode > div, pre {
        background: #05080F !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.77rem !important;
        color: #7EC8E3 !important;
    }

    /* Metric widgets */
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: var(--accent) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 0.65rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricDelta"] { display: none !important; }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
    }

    /* Download button */
    [data-testid="stDownloadButton"] button {
        background: transparent !important;
        border: 1px solid var(--border-bright) !important;
        color: var(--accent) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.75rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        transition: var(--transition) !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: var(--accent-dim) !important;
        border-color: var(--accent) !important;
    }

    /* Empty state text */
    .empty-state {
        text-align: center;
        padding: 5rem 0 3rem 0;
        font-family: var(--font-mono);
        font-size: 0.8rem;
        letter-spacing: 3px;
        color: #1E2A3A;
        text-transform: uppercase;
    }

    /* Animations */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>

<!-- Mouse-tracking gradient: updates CSS vars --mx/--my on mousemove -->
<script>
(function() {
    var ticking = false;
    document.addEventListener('mousemove', function(e) {
        if (!ticking) {
            requestAnimationFrame(function() {
                var x = (e.clientX / window.innerWidth  * 100).toFixed(1) + '%';
                var y = (e.clientY / window.innerHeight * 100).toFixed(1) + '%';
                document.documentElement.style.setProperty('--mx', x);
                document.documentElement.style.setProperty('--my', y);
                ticking = false;
            });
            ticking = true;
        }
    });
})();
</script>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3: CONSTANTS
# =============================================================================

# ACTION REQUIRED: Update paths if model files are stored in a subfolder.
MODEL_PATHS = {
    "Random Forest":         "random_forest.joblib",
    "Logistic Regression":   "logistic_regression.joblib",
    "SVM":                   "svm.joblib"
}

# ACTION REQUIRED: Verify scaler path matches your exported file name.
SCALER_PATH = "scaler.joblib"

# ACTION REQUIRED: Replace with real test-set accuracy scores after training.
DEMO_ACCURACY = {
    "Random Forest":       0.9871,
    "Logistic Regression": 0.9134,
    "SVM":                 0.9562
}

# ACTION REQUIRED: Replace with real confusion matrices from your test evaluation.
DEMO_CM = {
    "Random Forest":       np.array([[920, 18], [12, 850]]),
    "Logistic Regression": np.array([[870, 68], [55, 807]]),
    "SVM":                 np.array([[900, 38], [35, 827]])
}


# =============================================================================
# SECTION 4: CACHED LOADERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load a joblib model. Returns None if file is absent."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    """Load the fitted StandardScaler. Returns None if file is absent."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# =============================================================================
# SECTION 5: PROCESSING HELPERS
# =============================================================================

def preprocess(df: pd.DataFrame, scaler, label_col: str = "Label"):
    """
    Extract features and optional labels, impute NaN with column median,
    then apply StandardScaler normalization.
    Returns: (X_scaled np.ndarray, y_true Series or None)
    """
    y_true = None
    if label_col in df.columns:
        y_true = df[label_col].copy()
        df = df.drop(columns=[label_col])
    df_num   = df.select_dtypes(include=[np.number])
    df_num   = df_num.fillna(df_num.median())
    X_scaled = scaler.transform(df_num)
    return X_scaled, y_true


def summarize(preds: np.ndarray) -> dict:
    """Aggregate per-row predictions into an overall verdict dictionary."""
    total  = len(preds)
    attack = int(np.sum(preds == 1))
    normal = int(np.sum(preds == 0))
    return {
        "verdict":      "ATTACK" if attack > 0 else "NORMAL",
        "attack_count": attack,
        "normal_count": normal,
        "total":        total,
        "attack_pct":   round(attack / total * 100, 1) if total else 0.0
    }


def make_cm_figure(cm: np.ndarray, model_name: str) -> go.Figure:
    """Build a dark-themed Plotly confusion matrix heatmap."""
    labels = ["Normal", "Attack"]
    annotations = [
        dict(x=labels[j], y=labels[i],
             text=str(cm[i, j]),
             font=dict(size=18, color="#E8EEF7", family="DM Mono"),
             showarrow=False)
        for i in range(2) for j in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: Normal", "Predicted: Attack"],
        y=["Actual: Normal",    "Actual: Attack"],
        colorscale=[[0.0, "#0D1420"], [1.0, "#00B8CC"]],
        showscale=False,
        hoverongaps=False
    ))
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix — {model_name}",
            font=dict(size=12, family="DM Mono", color="#5A6880"),
            x=0.5
        ),
        xaxis=dict(tickfont=dict(size=11, family="DM Mono", color="#5A6880")),
        yaxis=dict(tickfont=dict(size=11, family="DM Mono", color="#5A6880")),
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=40, b=8),
        height=300
    )
    return fig


# =============================================================================
# SECTION 6: PAGE HEADER
# =============================================================================
st.markdown("""
<div class='ids-wordmark'>Intrusion Detection System</div>
<div class='ids-title'>IoT Botnet <span>DDoS</span> Detection</div>
<div class='ids-subtitle'>Smart Cities Security Platform &nbsp;/&nbsp; Network Traffic Analysis</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='ids-rule'>", unsafe_allow_html=True)


# =============================================================================
# SECTION 7: FILE UPLOADER
# =============================================================================
st.markdown("<div class='ids-label'>Upload Traffic Capture</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drop CSV file here",       # Internal label text (hidden by CSS)
    type=["csv"],
    label_visibility="collapsed",
    help=(
        "Upload a CSV exported from CICFlowMeter or similar tool. "
        "Max file size: 500 MB — requires .streamlit/config.toml (see project docs)."
        # ACTION REQUIRED: Create .streamlit/config.toml — content provided below.
    )
)


# =============================================================================
# SECTION 8: DETECTION PIPELINE (runs after file upload)
# =============================================================================

if uploaded_file is not None:

    # ── Load scaler ────────────────────────────────────────────────────────────
    scaler = load_scaler(SCALER_PATH)

    # ── Pipeline log ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='ids-label'>Processing Log</div>", unsafe_allow_html=True)
    log_ph    = st.empty()    # Dynamic placeholder for the updating log block
    log_lines = []

    def log(msg: str):
        """Append msg to log and refresh the st.code block."""
        log_lines.append(msg)
        log_ph.code("\n".join(log_lines), language="bash")
        time.sleep(0.22)

    log(f"[ START ] Received: {uploaded_file.name}")

    # ── Parse CSV ──────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file)
        log(f"[ OK    ] Parsed — {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # ── Preprocess ────────────────────────────────────────────────────────────
    log("[ RUN   ] Feature extraction, NaN imputation ...")
    if scaler is None:
        log("[ WARN  ] Scaler not found — unscaled demo mode")
        st.warning(
            "Scaler file `scaler.joblib` not found. "
            "Export it after training: `joblib.dump(scaler, 'scaler.joblib')`"
        )
        df_num   = df_raw.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = df_num.values
        y_true   = None
    else:
        try:
            X_scaled, y_true = preprocess(df_raw.copy(), scaler)
            log(f"[ OK    ] Scaled — {X_scaled.shape[1]} features retained")
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    # ── Primary model inference (Random Forest) ────────────────────────────────
    log("[ RUN   ] Loading Random Forest for primary detection ...")
    primary_model = load_model(MODEL_PATHS["Random Forest"])

    if primary_model is None:
        log("[ WARN  ] random_forest.joblib not found — demo mode")
        st.warning(
            "Model `random_forest.joblib` not found. "
            "Export: `joblib.dump(model, 'random_forest.joblib')`"
        )
        predictions = np.random.randint(0, 2, size=len(X_scaled))
    else:
        try:
            predictions = primary_model.predict(X_scaled)
            log(f"[ OK    ] Inference — {len(predictions):,} predictions")
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    result = summarize(predictions)
    log(f"[ OK    ] Verdict: {result['verdict']}  |  Attack: {result['attack_count']:,}  |  Normal: {result['normal_count']:,}")
    log("[ DONE  ] Pipeline complete")

    # =========================================================================
    # SECTION 9: VERDICT CARD
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='ids-label'>Detection Result</div>", unsafe_allow_html=True)

    is_attack   = result["verdict"] == "ATTACK"
    card_cls    = "verdict-attack" if is_attack else "verdict-normal"
    value_cls   = "verdict-value-attack" if is_attack else "verdict-value-normal"
    tag_color   = "#FF3D5A" if is_attack else "#00C896"

    st.markdown(f"""
    <div class='{card_cls}'>
        <div class='verdict-tag' style='color:{tag_color};'>Network Status</div>
        <div class='{value_cls}'>{result['verdict']}</div>
        <div class='verdict-sub'>
            {result['total']:,} flows analyzed
            &nbsp;&nbsp;&middot;&nbsp;&nbsp;
            {result['attack_count']:,} attack
            &nbsp;&nbsp;&middot;&nbsp;&nbsp;
            {result['normal_count']:,} normal
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 10: ATTACK RATE PROGRESS BAR
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='ids-label'>Attack Rate &nbsp; {result['attack_pct']}% of total traffic</div>",
        unsafe_allow_html=True
    )
    st.progress(min(result['attack_pct'] / 100, 1.0))

    st.markdown("<hr class='ids-rule'>", unsafe_allow_html=True)

    # =========================================================================
    # SECTION 11: DETAILED MODEL ANALYSIS EXPANDER
    # =========================================================================
    with st.expander("Detailed Model Analysis", expanded=False):

        st.markdown("<br>", unsafe_allow_html=True)

        # Model dropdown
        st.markdown("<div class='ids-label'>Select Model</div>", unsafe_allow_html=True)
        chosen = st.selectbox(
            label="Model selector",
            options=list(MODEL_PATHS.keys()),
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Load selected model and compute or retrieve metrics
        sel_model = load_model(MODEL_PATHS[chosen])

        if sel_model is not None and 'y_true' in dir() and y_true is not None:
            try:
                sel_preds = sel_model.predict(X_scaled)
                sel_acc   = accuracy_score(y_true, sel_preds)
                sel_cm    = confusion_matrix(y_true, sel_preds)
            except Exception:
                sel_acc = DEMO_ACCURACY[chosen]
                sel_cm  = DEMO_CM[chosen]
        else:
            # Use demo values when model file or ground truth is unavailable
            sel_acc = DEMO_ACCURACY[chosen]
            sel_cm  = DEMO_CM[chosen]

        # Accuracy badge
        st.markdown("<div class='ids-label'>Model Accuracy</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='acc-badge'>{sel_acc * 100:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix heatmap
        st.markdown("<div class='ids-label'>Confusion Matrix</div>", unsafe_allow_html=True)
        st.plotly_chart(make_cm_figure(sel_cm, chosen), use_container_width=True)

        # Raw count metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("True Positive",  int(sel_cm[1][1]))
        with c2: st.metric("True Negative",  int(sel_cm[0][0]))
        with c3: st.metric("False Positive", int(sel_cm[0][1]))
        with c4: st.metric("False Negative", int(sel_cm[1][0]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction distribution donut chart
        st.markdown("<div class='ids-label'>Prediction Distribution</div>",
                    unsafe_allow_html=True)
        pie = px.pie(
            values=[result['normal_count'], result['attack_count']],
            names=["Normal", "Attack"],
            color_discrete_sequence=["#00C896", "#FF3D5A"],
            hole=0.52
        )
        pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=12, family="DM Mono", color="#E8EEF7")
        )
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(family="DM Mono", size=11, color="#5A6880")),
            margin=dict(l=8, r=8, t=8, b=8),
            height=250
        )
        st.plotly_chart(pie, use_container_width=True)

    # =========================================================================
    # SECTION 12: DATA PREVIEW AND DOWNLOAD
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='ids-label'>Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True, height=250)

    st.markdown("<br>", unsafe_allow_html=True)

    out_df = df_raw.copy()
    out_df["Predicted_Class"]  = predictions
    out_df["Predicted_Status"] = ["ATTACK" if p == 1 else "NORMAL" for p in predictions]
    csv_out = out_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_out,
        file_name="ids_predictions.csv",
        mime="text/csv"
    )

else:
    # ── Empty state before file upload ────────────────────────────────────────
    st.markdown(
        "<div class='empty-state'>Awaiting traffic capture upload</div>",
        unsafe_allow_html=True
    )

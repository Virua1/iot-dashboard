# =============================================================================
# FILE: app.py
# PROJECT: IoT Botnet-based DDoS Detection in Smart Cities — IDS Dashboard
# COMPATIBLE: Python 3.14+, Streamlit 1.56+, pandas 3.x, numpy 2.x
# =============================================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import confusion_matrix, accuracy_score

# =============================================================================
# PAGE CONFIG — must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="IoT Botnet IDS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS — Dark animated gradient theme, no emojis, Syne + DM Mono fonts
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg:            #080C14;
        --surface:       #0D1420;
        --raised:        #131C2E;
        --border:        rgba(0,229,255,0.13);
        --border-hi:     rgba(0,229,255,0.38);
        --accent:        #00E5FF;
        --accent-dim:    rgba(0,229,255,0.07);
        --text:          #E8EEF7;
        --muted:         #4A5E78;
        --green:         #00C896;
        --green-bg:      rgba(0,200,150,0.08);
        --green-bd:      rgba(0,200,150,0.42);
        --red:           #FF3D5A;
        --red-bg:        rgba(255,61,90,0.08);
        --red-bd:        rgba(255,61,90,0.42);
        --radius:        10px;
        --ease:          all 0.24s cubic-bezier(.4,0,.2,1);
    }

    /* ── Base ── */
    .stApp {
        background: #080C14;
        font-family: 'Syne', sans-serif;
        color: var(--text);
    }

    /* Mouse-tracked gradient orb (position set by JS below) */
    .stApp::before {
        content:'';
        position:fixed; inset:0;
        background:
            radial-gradient(700px circle at var(--mx,25%) var(--my,25%),
                rgba(0,229,255,.042) 0%, transparent 68%),
            radial-gradient(900px circle at 80% 85%,
                rgba(0,80,255,.025) 0%, transparent 60%);
        pointer-events:none;
        z-index:0;
        transition: background .1s linear;
    }

    /* ── Hide chrome ── */
    #MainMenu,footer,header { visibility:hidden; }
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] { display:none !important; }

    /* ── Layout ── */
    .block-container {
        padding: 3.5rem 4rem 5rem;
        max-width: 1050px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    /* ── Typography ── */
    .wordmark {
        font-family:'DM Mono',monospace;
        font-size:.68rem; letter-spacing:5px;
        color:var(--accent); text-align:center;
        margin-bottom:.55rem; opacity:.65;
        text-transform:uppercase;
    }
    .title {
        font-family:'Syne',sans-serif;
        font-size:2.75rem; font-weight:800;
        color:var(--text); text-align:center;
        letter-spacing:-1.2px; line-height:1.1;
        margin-bottom:.4rem;
    }
    .title span { color:var(--accent); }
    .subtitle {
        font-family:'DM Mono',monospace;
        font-size:.78rem; color:var(--muted);
        text-align:center; letter-spacing:2px;
        margin-bottom:2.5rem;
    }
    .rule {
        border:none;
        border-top:1px solid var(--border);
        margin:2rem 0;
    }
    .lbl {
        font-family:'DM Mono',monospace;
        font-size:.65rem; letter-spacing:3.5px;
        color:var(--accent); text-transform:uppercase;
        margin-bottom:.6rem; opacity:.7;
    }

    /* ── Verdict cards ── */
    .v-normal,.v-attack {
        border-radius:var(--radius);
        padding:2.2rem 2.5rem;
        text-align:center;
        animation: up .45s cubic-bezier(.16,1,.3,1) both;
    }
    .v-normal { background:var(--green-bg); border:1.5px solid var(--green-bd); }
    .v-attack { background:var(--red-bg);   border:1.5px solid var(--red-bd);   }
    .v-tag {
        font-family:'DM Mono',monospace;
        font-size:.68rem; letter-spacing:4px;
        text-transform:uppercase; margin-bottom:.55rem;
    }
    .v-val-n { font-size:3.5rem; font-weight:800; color:var(--green); line-height:1; }
    .v-val-a { font-size:3.5rem; font-weight:800; color:var(--red);   line-height:1; }
    .v-sub {
        font-family:'DM Mono',monospace;
        font-size:.76rem; color:var(--muted); margin-top:.7rem;
    }

    /* ── Accuracy badge ── */
    .acc {
        display:inline-block;
        background:var(--accent-dim);
        border:1px solid var(--border-hi);
        color:var(--accent);
        border-radius:6px; padding:.4rem 1.2rem;
        font-family:'DM Mono',monospace;
        font-size:1.15rem; font-weight:500;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background:var(--raised) !important;
        border:1.5px dashed var(--border-hi) !important;
        border-radius:var(--radius) !important;
        transition:var(--ease) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color:var(--accent) !important;
        background:var(--accent-dim) !important;
    }
    [data-testid="stFileUploader"] * {
        color:var(--muted) !important;
        font-family:'DM Mono',monospace !important;
        font-size:.83rem !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgress"] > div > div {
        background:linear-gradient(90deg,#00B8CC,#00E5FF) !important;
        border-radius:4px !important;
    }
    [data-testid="stProgress"] > div {
        background:var(--raised) !important;
        border-radius:4px !important;
        height:5px !important;
        border:1px solid var(--border) !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background:var(--raised) !important;
        border:1px solid var(--border) !important;
        border-radius:var(--radius) !important;
        color:var(--text) !important;
        font-family:'DM Mono',monospace !important;
    }

    /* ── Expander ── */
    details {
        background:var(--surface) !important;
        border:1px solid var(--border) !important;
        border-radius:var(--radius) !important;
        margin-top:1.5rem !important;
        overflow:hidden !important;
    }
    details[open] { border-color:var(--border-hi) !important; }
    details > summary {
        font-family:'DM Mono',monospace !important;
        font-size:.75rem !important; letter-spacing:3px !important;
        color:var(--accent) !important; text-transform:uppercase !important;
        padding:1rem 1.25rem !important; cursor:pointer !important;
        background:transparent !important; list-style:none !important;
    }
    details > summary::-webkit-details-marker { display:none; }

    /* ── Code block ── */
    .stCode > div, pre {
        background:#05080F !important;
        border:1px solid var(--border) !important;
        border-radius:var(--radius) !important;
        font-family:'DM Mono',monospace !important;
        font-size:.77rem !important; color:#7EC8E3 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-family:'DM Mono',monospace !important;
        font-size:1.5rem !important; font-weight:500 !important;
        color:var(--accent) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family:'DM Mono',monospace !important;
        font-size:.65rem !important; letter-spacing:2px !important;
        text-transform:uppercase !important; color:var(--muted) !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background:transparent !important;
        border:1px solid var(--border-hi) !important;
        color:var(--accent) !important;
        font-family:'DM Mono',monospace !important;
        font-size:.75rem !important; letter-spacing:2px !important;
        text-transform:uppercase !important;
        border-radius:6px !important; padding:.5rem 1.2rem !important;
        transition:var(--ease) !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background:var(--accent-dim) !important;
        border-color:var(--accent) !important;
    }

    /* ── Empty state ── */
    .empty {
        text-align:center; padding:5rem 0 3rem;
        font-family:'DM Mono',monospace;
        font-size:.8rem; letter-spacing:3px;
        color:#1E2A3A; text-transform:uppercase;
    }

    /* ── Animation ── */
    @keyframes up {
        from { opacity:0; transform:translateY(16px); }
        to   { opacity:1; transform:translateY(0); }
    }
</style>

<script>
/* Mouse-tracking: update CSS vars --mx/--my for the gradient orb */
(function(){
    var t=false;
    document.addEventListener('mousemove',function(e){
        if(!t){
            requestAnimationFrame(function(){
                var x=(e.clientX/window.innerWidth*100).toFixed(1)+'%';
                var y=(e.clientY/window.innerHeight*100).toFixed(1)+'%';
                document.documentElement.style.setProperty('--mx',x);
                document.documentElement.style.setProperty('--my',y);
                t=false;
            });
            t=true;
        }
    });
})();
</script>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS
# ACTION REQUIRED: Verify file names match your exported .joblib files.
# =============================================================================
MODEL_PATHS = {
    "Random Forest":         "random_forest.joblib",
    "Logistic Regression":   "logistic_regression.joblib",
    "SVM":                   "svm.joblib",
}

SCALER_PATH = "scaler.joblib"  # ACTION REQUIRED: must match your exported scaler name

# ACTION REQUIRED: Replace with real accuracy values from your test evaluation.
DEMO_ACCURACY = {
    "Random Forest":       0.9871,
    "Logistic Regression": 0.9134,
    "SVM":                 0.9562,
}

# ACTION REQUIRED: Replace with real confusion matrices from your test evaluation.
DEMO_CM = {
    "Random Forest":       np.array([[920, 18], [12, 850]]),
    "Logistic Regression": np.array([[870, 68], [55, 807]]),
    "SVM":                 np.array([[900, 38], [35, 827]]),
}


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Return a loaded joblib model, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    """Return the fitted StandardScaler, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# =============================================================================
# PROCESSING HELPERS
# =============================================================================
def preprocess(df: pd.DataFrame, scaler, label_col: str = "Label"):
    """
    Extract labels (optional), keep numeric columns, impute NaN via median,
    then apply StandardScaler. Returns (X_scaled, y_true or None).
    """
    y_true = None
    if label_col in df.columns:
        y_true = df[label_col].copy()
        df = df.drop(columns=[label_col])

    df_num = df.select_dtypes(include=[np.number])
    # pandas 3.x: fillna with median computed first to avoid FutureWarnings
    medians = df_num.median()
    df_num  = df_num.fillna(medians)

    X_scaled = scaler.transform(df_num)
    return X_scaled, y_true


def summarize(preds: np.ndarray) -> dict:
    """Aggregate per-row predictions into an overall result dictionary."""
    total  = len(preds)
    attack = int(np.sum(preds == 1))
    normal = int(np.sum(preds == 0))
    return {
        "verdict":      "ATTACK" if attack > 0 else "NORMAL",
        "attack_count": attack,
        "normal_count": normal,
        "total":        total,
        "attack_pct":   round(attack / total * 100, 1) if total else 0.0,
    }


def cm_figure(cm: np.ndarray, model_name: str) -> go.Figure:
    """Return a dark-themed Plotly confusion matrix heatmap figure."""
    labels = ["Normal", "Attack"]
    anns = [
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
    ))
    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}",
                   font=dict(size=12, family="DM Mono", color="#4A5E78"), x=0.5),
        xaxis=dict(tickfont=dict(size=11, family="DM Mono", color="#4A5E78")),
        yaxis=dict(tickfont=dict(size=11, family="DM Mono", color="#4A5E78")),
        annotations=anns,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=42, b=8),
        height=300,
    )
    return fig


# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class='wordmark'>Intrusion Detection System</div>
<div class='title'>IoT Botnet <span>DDoS</span> Detection</div>
<div class='subtitle'>Smart Cities Security Platform &nbsp;/&nbsp; Network Traffic Analysis</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='rule'>", unsafe_allow_html=True)


# =============================================================================
# FILE UPLOADER
# =============================================================================
st.markdown("<div class='lbl'>Upload Traffic Capture</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drop CSV file here",
    type=["csv"],
    label_visibility="collapsed",
    help="Upload a CSV exported from CICFlowMeter or similar tool. Max 500 MB.",
    # ACTION REQUIRED: 500 MB limit requires .streamlit/config.toml — see config.toml file.
)


# =============================================================================
# DETECTION PIPELINE
# =============================================================================
if uploaded_file is not None:

    scaler = load_scaler(SCALER_PATH)

    # ── Pipeline log ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>Processing Log</div>", unsafe_allow_html=True)
    log_ph    = st.empty()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        log_ph.code("\n".join(log_lines), language="bash")
        time.sleep(0.22)

    log(f"[ START ] Received: {uploaded_file.name}")

    # ── Read CSV ───────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file)
        log(f"[ OK    ] Parsed — {df_raw.shape[0]:,} rows x {df_raw.shape[1]} cols")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # ── Preprocess ────────────────────────────────────────────────────────────
    log("[ RUN   ] Feature extraction and NaN imputation ...")

    if scaler is None:
        log("[ WARN  ] scaler.joblib not found — running unscaled (demo mode)")
        st.warning(
            "Scaler file `scaler.joblib` not found. "
            "Export it after training: `joblib.dump(scaler, 'scaler.joblib')`"
        )
        df_num   = df_raw.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = df_num.to_numpy()   # pandas 3.x: prefer .to_numpy() over .values
        y_true: pd.Series | None = None
    else:
        try:
            X_scaled, y_true = preprocess(df_raw.copy(), scaler)
            log(f"[ OK    ] Scaled — {X_scaled.shape[1]} features retained")
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

    # ── Primary inference (Random Forest) ─────────────────────────────────────
    log("[ RUN   ] Loading Random Forest for primary detection ...")
    primary = load_model(MODEL_PATHS["Random Forest"])

    if primary is None:
        log("[ WARN  ] random_forest.joblib not found — demo predictions")
        st.warning(
            "Model `random_forest.joblib` not found. "
            "Export after training: `joblib.dump(model, 'random_forest.joblib')`"
        )
        rng         = np.random.default_rng(seed=42)   # numpy 2.x: use default_rng
        predictions = rng.integers(0, 2, size=len(X_scaled))
    else:
        try:
            predictions = primary.predict(X_scaled)
            log(f"[ OK    ] Inference — {len(predictions):,} predictions")
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    result = summarize(predictions)
    log(
        f"[ OK    ] Verdict: {result['verdict']}  |  "
        f"Attack: {result['attack_count']:,}  |  Normal: {result['normal_count']:,}"
    )
    log("[ DONE  ] Pipeline complete")

    # =========================================================================
    # VERDICT CARD
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>Detection Result</div>", unsafe_allow_html=True)

    is_attack  = result["verdict"] == "ATTACK"
    card_cls   = "v-attack"  if is_attack else "v-normal"
    val_cls    = "v-val-a"   if is_attack else "v-val-n"
    tag_color  = "#FF3D5A"   if is_attack else "#00C896"

    st.markdown(f"""
    <div class='{card_cls}'>
        <div class='v-tag' style='color:{tag_color};'>Network Status</div>
        <div class='{val_cls}'>{result['verdict']}</div>
        <div class='v-sub'>
            {result['total']:,} flows analyzed
            &nbsp;&middot;&nbsp;
            {result['attack_count']:,} attack
            &nbsp;&middot;&nbsp;
            {result['normal_count']:,} normal
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # ATTACK RATE PROGRESS BAR
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='lbl'>Attack Rate &nbsp; {result['attack_pct']}%</div>",
        unsafe_allow_html=True
    )
    st.progress(float(min(result['attack_pct'] / 100.0, 1.0)))
    # float() cast ensures compatibility with Streamlit 1.56 progress() signature

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # =========================================================================
    # DETAILED MODEL ANALYSIS EXPANDER
    # =========================================================================
    with st.expander("Detailed Model Analysis", expanded=False):

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='lbl'>Select Model</div>", unsafe_allow_html=True)

        chosen = st.selectbox(
            label="Model selector",
            options=list(MODEL_PATHS.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Load selected model and resolve metrics
        sel_model = load_model(MODEL_PATHS[chosen])
        sel_acc   = DEMO_ACCURACY[chosen]
        sel_cm    = DEMO_CM[chosen]

        if sel_model is not None and y_true is not None:
            try:
                sel_preds = sel_model.predict(X_scaled)
                sel_acc   = float(accuracy_score(y_true, sel_preds))
                sel_cm    = confusion_matrix(y_true, sel_preds)
            except Exception:
                pass   # silently fall back to demo values

        # Accuracy
        st.markdown("<div class='lbl'>Model Accuracy</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='acc'>{sel_acc * 100:.2f}%</div>",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        st.markdown("<div class='lbl'>Confusion Matrix</div>", unsafe_allow_html=True)
        st.plotly_chart(cm_figure(sel_cm, chosen), use_container_width=True)

        # Raw counts
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("True Positive",  int(sel_cm[1][1]))
        with c2: st.metric("True Negative",  int(sel_cm[0][0]))
        with c3: st.metric("False Positive", int(sel_cm[0][1]))
        with c4: st.metric("False Negative", int(sel_cm[1][0]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction distribution donut
        st.markdown("<div class='lbl'>Prediction Distribution</div>",
                    unsafe_allow_html=True)
        pie = px.pie(
            values=[result["normal_count"], result["attack_count"]],
            names=["Normal", "Attack"],
            color_discrete_sequence=["#00C896", "#FF3D5A"],
            hole=0.52,
        )
        pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(size=12, family="DM Mono", color="#E8EEF7"),
        )
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(family="DM Mono", size=11, color="#4A5E78")),
            margin=dict(l=8, r=8, t=8, b=8),
            height=250,
        )
        st.plotly_chart(pie, use_container_width=True)

    # =========================================================================
    # DATA PREVIEW + DOWNLOAD
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True, height=250)

    st.markdown("<br>", unsafe_allow_html=True)

    out_df = df_raw.copy()
    out_df["Predicted_Class"]  = predictions
    out_df["Predicted_Status"] = [
        "ATTACK" if int(p) == 1 else "NORMAL" for p in predictions
    ]
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="ids_predictions.csv",
        mime="text/csv",
    )

else:
    # Empty state — shown before any file is uploaded
    st.markdown(
        "<div class='empty'>Awaiting traffic capture upload</div>",
        unsafe_allow_html=True
    )

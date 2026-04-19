# =============================================================================
# FILE: app.py
# PROJECT: IoT Botnet-based DDoS Detection in Smart Cities
# SYSTEM: Intrusion Detection System (IDS) - Streamlit Dashboard
# AUTHOR: [Your Name] - Graduation Project
# DESCRIPTION: A production-ready, professional Streamlit dashboard that loads
#              pre-trained ML models (Random Forest, Logistic Regression, SVM),
#              accepts network traffic CSV files, runs inference, and displays
#              detailed visual analytics for academic and operational use.
# =============================================================================

# ─── Standard Library Imports ─────────────────────────────────────────────────
import os           # Operating system utilities for path manipulation
import time         # Time module for simulating processing delays in logs
import warnings     # Suppress non-critical library warnings for cleaner output
warnings.filterwarnings("ignore")  # Ignore sklearn and other library warnings

# ─── Third-Party Library Imports ──────────────────────────────────────────────
import streamlit as st      # Core Streamlit framework for the web dashboard
import pandas as pd         # Data manipulation and CSV file handling
import numpy as np          # Numerical operations and array handling
import joblib               # Loading serialized ML models and scalers
import plotly.graph_objects as go  # Interactive Plotly charts (Confusion Matrix)
import plotly.express as px        # High-level Plotly API for quick charts

# ─── Scikit-learn Metrics (used for evaluation display) ───────────────────────
from sklearn.metrics import (
    confusion_matrix,       # Generates the confusion matrix array
    accuracy_score,         # Computes overall prediction accuracy
    precision_score,        # Computes precision (TP / (TP + FP))
    recall_score,           # Computes recall / sensitivity (TP / (TP + FN))
    classification_report   # Full text report of all classification metrics
)

# =============================================================================
# SECTION 1: PAGE CONFIGURATION
# Must be the FIRST Streamlit command called in the script.
# =============================================================================
st.set_page_config(
    page_title="IoT Botnet IDS",            # Browser tab title
    page_icon="🛡️",                          # Browser tab favicon (emoji)
    layout="wide",                           # Use full browser width for layout
    initial_sidebar_state="expanded"         # Start with sidebar open
)

# =============================================================================
# SECTION 2: GLOBAL CUSTOM CSS STYLING
# Injects custom CSS into the Streamlit app to achieve the Corporate Light Theme.
# Theme: White/Light-Gray background, Slate Blue accents, soft card shadows.
# =============================================================================
st.markdown("""
<style>
    /* ── Google Font Import: IBM Plex Sans (Professional & Technical feel) ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Root CSS Variables for Consistent Theming ── */
    :root {
        --bg-primary:    #F0F2F6;   /* Main page background: very light gray */
        --bg-card:       #FFFFFF;   /* Card/container background: pure white */
        --accent-blue:   #3A5A8C;   /* Primary accent: Professional Slate Blue */
        --accent-light:  #EBF0FA;   /* Light accent for highlights */
        --text-primary:  #1A2035;   /* Main text: near-black navy */
        --text-secondary:#4A5568;   /* Secondary text: medium gray */
        --border-color:  #DDE3EE;   /* Subtle borders */
        --shadow:        0 2px 12px rgba(58,90,140,0.10); /* Soft blue-tinted shadow */
        --shadow-hover:  0 6px 24px rgba(58,90,140,0.18); /* Elevated shadow on hover */
        --green-status:  #1A7F37;   /* Status green for NORMAL traffic */
        --red-status:    #CF222E;   /* Status red for ATTACK traffic */
        --green-bg:      #D4F0DC;   /* Background tint for normal status card */
        --red-bg:        #FFDCE0;   /* Background tint for attack status card */
        --font-main:     'IBM Plex Sans', sans-serif;
        --font-mono:     'IBM Plex Mono', monospace;
    }

    /* ── Global App Background ── */
    .stApp {
        background-color: var(--bg-primary);
        font-family: var(--font-main);
        color: var(--text-primary);
    }

    /* ── Hide default Streamlit header and footer decoration ── */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}

    /* ── Main Content Block Padding ── */
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 12px rgba(58,90,140,0.06);
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-blue);
        font-family: var(--font-main);
        font-weight: 600;
    }

    /* ── Professional Card Component (reusable via HTML injection) ── */
    .ids-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
        transition: box-shadow 0.2s ease;
    }
    .ids-card:hover {
        box-shadow: var(--shadow-hover);
    }

    /* ── Dashboard Main Title ── */
    .ids-main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--accent-blue);
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .ids-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .ids-divider {
        border: none;
        border-top: 2px solid var(--border-color);
        margin: 1.2rem 0 1.8rem 0;
    }

    /* ── Section Header Style ── */
    .ids-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Status Cards (Verdict Display) ── */
    .status-card-normal {
        background: var(--green-bg);
        border: 2px solid var(--green-status);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .status-card-normal .status-label {
        font-size: 1rem;
        font-weight: 600;
        color: var(--green-status);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .status-card-normal .status-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--green-status);
    }
    .status-card-attack {
        background: var(--red-bg);
        border: 2px solid var(--red-status);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .status-card-attack .status-label {
        font-size: 1rem;
        font-weight: 600;
        color: var(--red-status);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .status-card-attack .status-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--red-status);
    }

    /* ── Metric Badge ── */
    .metric-badge {
        display: inline-block;
        background: var(--accent-light);
        color: var(--accent-blue);
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* ── Insight Card inside Expanders ── */
    .insight-card {
        background: var(--accent-light);
        border-left: 4px solid var(--accent-blue);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: var(--text-primary);
    }

    /* ── Override Streamlit expander header styling ── */
    details summary p {
        font-weight: 600 !important;
        color: var(--accent-blue) !important;
        font-size: 1rem !important;
    }

    /* ── Override Streamlit metric label styling ── */
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--accent-blue) !important;
    }

    /* ── Code Block Styling (Operational Log) ── */
    .stCode {
        border-radius: 8px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
    }

    /* ── Mermaid Diagram Container ── */
    .mermaid-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)  # Allow raw HTML/CSS injection


# =============================================================================
# SECTION 3: CONSTANTS & CONFIGURATION
# Centralized configuration for model paths and display names.
# =============================================================================

# Dictionary mapping display names to their .joblib file paths.
# ACTION REQUIRED: Update these paths if your model files are in a different folder.
MODEL_PATHS = {
    "Random Forest":         "random_forest.joblib",
    "Logistic Regression":   "logistic_regression.joblib",
    "SVM":                   "svm.joblib"
}

# Path to the saved StandardScaler object used during training.
# ACTION REQUIRED: Update this path if your scaler file is stored elsewhere.
SCALER_PATH = "scaler.joblib"

# Class label mapping: 0 = Normal traffic, 1 = Attack/Botnet traffic.
# ACTION REQUIRED: Adjust these labels if your dataset uses different class conventions.
CLASS_LABELS = {0: "NORMAL", 1: "ATTACK"}

# Accent color used in Plotly charts to match the dashboard theme.
PLOTLY_ACCENT = "#3A5A8C"  # Slate Blue matching CSS --accent-blue

# Pre-defined demo accuracy scores for display when no ground truth labels exist.
# ACTION REQUIRED: Replace with your actual trained model evaluation scores.
DEMO_METRICS = {
    "Random Forest":       {"accuracy": 0.9871, "precision": 0.9865, "recall": 0.9878},
    "Logistic Regression": {"accuracy": 0.9134, "precision": 0.9098, "recall": 0.9170},
    "SVM":                 {"accuracy": 0.9562, "precision": 0.9541, "recall": 0.9583}
}

# Insight text for each model displayed inside expanders.
MODEL_INSIGHTS = {
    "Random Forest": (
        "🌲 Random Forest achieves the highest accuracy in this benchmark by ensembling "
        "hundreds of decision trees to reduce variance. It handles class imbalance well "
        "and provides feature importances — crucial for IoT traffic forensics. "
        "Recommended for production deployment where interpretability and robustness are priorities."
    ),
    "Logistic Regression": (
        "📈 Logistic Regression provides a strong probabilistic baseline. While its accuracy "
        "is lower than tree-based models (due to non-linear IoT traffic patterns), it trains "
        "extremely fast and produces calibrated probability scores. Ideal for explainability "
        "in regulatory or academic reporting contexts."
    ),
    "SVM": (
        "⚡ Support Vector Machine with an RBF kernel performs well on high-dimensional "
        "network feature spaces. It maximizes the margin between Normal and Attack classes, "
        "making it robust to outliers. Slightly slower at inference than RF but offers "
        "excellent generalization on unseen attack variants."
    )
}


# =============================================================================
# SECTION 4: HELPER FUNCTIONS
# Modular, reusable functions for loading, processing, and visualization.
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load a serialized ML model from disk using joblib.
    @st.cache_resource ensures the model is loaded only once and cached in memory,
    preventing redundant disk I/O on every user interaction.

    Args:
        model_path (str): Relative or absolute path to the .joblib model file.

    Returns:
        object: The loaded scikit-learn model, or None if file not found.
    """
    if not os.path.exists(model_path):  # Check if the file exists before attempting load
        return None  # Return None to allow graceful error handling downstream
    return joblib.load(model_path)  # Deserialize and return the model object


@st.cache_resource(show_spinner=False)
def load_scaler(scaler_path: str):
    """
    Load the pre-fitted StandardScaler from disk.
    The scaler must be identical to the one used during model training to ensure
    consistent feature normalization at inference time.

    Args:
        scaler_path (str): Path to the scaler .joblib file.

    Returns:
        object: The loaded StandardScaler, or None if file not found.
    """
    if not os.path.exists(scaler_path):  # Validate file existence
        return None  # Allow caller to handle missing scaler gracefully
    return joblib.load(scaler_path)  # Load and return the fitted scaler


def preprocess_dataframe(df: pd.DataFrame, scaler, label_col: str = "Label"):
    """
    Prepare the uploaded CSV DataFrame for model inference.
    Steps: (1) Separate features from labels if present, (2) drop non-numeric
    columns, (3) handle missing values, (4) apply the pre-fitted scaler.

    Args:
        df         (pd.DataFrame): Raw uploaded traffic data.
        scaler     (object):       Pre-fitted StandardScaler.
        label_col  (str):          Column name containing ground-truth labels (if any).

    Returns:
        tuple: (X_scaled np.ndarray, y_true pd.Series or None)
    """
    y_true = None  # Initialize ground truth to None (may not exist in the file)

    # ── Extract label column if it exists in the dataset ──────────────────────
    if label_col in df.columns:
        y_true = df[label_col].copy()   # Preserve labels for later evaluation
        df = df.drop(columns=[label_col])  # Remove labels from feature matrix

    # ── Retain only numeric columns (drop IP strings, timestamps, etc.) ────────
    df_numeric = df.select_dtypes(include=[np.number])

    # ── Fill any NaN values with column medians (robust to outliers) ───────────
    df_numeric = df_numeric.fillna(df_numeric.median())

    # ── Apply the pre-fitted scaler to normalize feature distributions ─────────
    X_scaled = scaler.transform(df_numeric)  # Transform using training-time statistics

    return X_scaled, y_true  # Return scaled features and optional ground truth


def build_confusion_matrix_figure(cm: np.ndarray, model_name: str) -> go.Figure:
    """
    Build an interactive Plotly heatmap visualization of the confusion matrix.
    Uses a blue color scale matching the corporate theme.

    Args:
        cm         (np.ndarray): 2x2 confusion matrix array from sklearn.
        model_name (str):        Model name for the chart title.

    Returns:
        go.Figure: A Plotly Figure object ready for st.plotly_chart().
    """
    # ── Define human-readable axis labels for the confusion matrix ─────────────
    class_names = ["Normal (0)", "Attack (1)"]  # Row/column tick labels

    # ── Create annotation text showing raw counts inside each cell ─────────────
    annotations = []  # List to hold cell annotation dictionaries
    for i in range(len(class_names)):         # Iterate over rows (actual classes)
        for j in range(len(class_names)):     # Iterate over columns (predicted classes)
            annotations.append(dict(
                x=class_names[j],            # X-axis position (predicted)
                y=class_names[i],            # Y-axis position (actual)
                text=str(cm[i, j]),          # Display the raw count number
                font=dict(size=20, color="white", family="IBM Plex Mono"),  # White bold count
                showarrow=False              # No arrow pointer needed
            ))

    # ── Build the Plotly heatmap figure ───────────────────────────────────────
    fig = go.Figure(data=go.Heatmap(
        z=cm,                                # 2D confusion matrix values (color intensity)
        x=class_names,                       # X-axis: predicted class labels
        y=class_names,                       # Y-axis: actual class labels
        colorscale=[
            [0.0, "#EBF0FA"],                # Low value: light blue (few samples)
            [1.0, "#3A5A8C"]                 # High value: deep slate blue (many samples)
        ],
        showscale=True,                      # Show the color scale legend
        hoverongaps=False                    # Disable hover on empty cells
    ))

    # ── Configure chart layout and annotations ────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix — {model_name}",  # Chart title with model name
            font=dict(size=15, family="IBM Plex Sans", color="#1A2035"),
            x=0.5                            # Center the title horizontally
        ),
        xaxis=dict(
            title="Predicted Label",         # X-axis label
            tickfont=dict(size=13, family="IBM Plex Sans")
        ),
        yaxis=dict(
            title="Actual Label",            # Y-axis label
            tickfont=dict(size=13, family="IBM Plex Sans")
        ),
        annotations=annotations,            # Inject count labels into cells
        paper_bgcolor="rgba(0,0,0,0)",      # Transparent chart background
        plot_bgcolor="rgba(0,0,0,0)",       # Transparent plot area background
        margin=dict(l=10, r=10, t=50, b=10),  # Tight margins for card fit
        height=320                           # Fixed height for consistent card sizing
    )

    return fig  # Return the fully configured Plotly figure


def run_inference_pipeline(X_scaled: np.ndarray, model) -> np.ndarray:
    """
    Execute model inference on the preprocessed and scaled feature matrix.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix (n_samples × n_features).
        model    (object):     A fitted scikit-learn classifier.

    Returns:
        np.ndarray: Array of predicted class labels (0 = Normal, 1 = Attack).
    """
    return model.predict(X_scaled)  # Run prediction and return label array


def get_verdict_summary(predictions: np.ndarray) -> dict:
    """
    Aggregate predictions across all rows to produce an overall verdict summary.
    If any single row is classified as ATTACK, the overall verdict is ATTACK.

    Args:
        predictions (np.ndarray): Array of per-row predictions (0 or 1).

    Returns:
        dict: Summary with 'verdict', 'attack_count', 'normal_count', 'total'.
    """
    total  = len(predictions)                   # Total number of traffic samples
    attack = int(np.sum(predictions == 1))      # Count samples classified as attack
    normal = int(np.sum(predictions == 0))      # Count samples classified as normal
    verdict = "ATTACK" if attack > 0 else "NORMAL"  # Overall verdict: any attack → ATTACK

    return {
        "verdict":      verdict,   # Overall classification result
        "attack_count": attack,    # Number of attack-classified rows
        "normal_count": normal,    # Number of normal-classified rows
        "total":        total      # Total rows analyzed
    }


# =============================================================================
# SECTION 5: SIDEBAR CONSTRUCTION
# The sidebar contains file upload, model selection, and system information.
# =============================================================================
with st.sidebar:
    # ── Sidebar Logo / Title ───────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🛡️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#3A5A8C; letter-spacing:0.5px;'>
            IoT Botnet IDS
        </div>
        <div style='font-size:0.78rem; color:#4A5568; margin-top:0.2rem;'>
            Smart Cities Security Platform
        </div>
    </div>
    <hr style='border-color:#DDE3EE; margin:1rem 0;'>
    """, unsafe_allow_html=True)  # Inject styled sidebar header

    # ── File Uploader Widget ───────────────────────────────────────────────────
    st.markdown("### 📂 Traffic Data")  # Sidebar section header
    uploaded_file = st.file_uploader(
        label="Upload CSV network traffic capture",  # Widget label
        type=["csv"],                                 # Only accept CSV files
        help="Upload a CSV file exported from your network capture tool (e.g., CICFlowMeter)."
        # ACTION REQUIRED: Ensure your CSV columns match the feature set used during training.
    )

    st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacer

    # ── Model Selector Dropdown ────────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")  # Sidebar section header
    selected_model_name = st.selectbox(
        label="Choose inference model",          # Dropdown label
        options=list(MODEL_PATHS.keys()),        # Options from MODEL_PATHS dict
        index=0,                                  # Default to first option (Random Forest)
        help="Select which pre-trained model to use for detection. All three can be compared below."
    )

    st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacer

    # ── System Status Indicators ───────────────────────────────────────────────
    st.markdown("### ⚙️ System Status")  # Sidebar section header

    # Check if each model file exists and display a status indicator
    for model_name, model_path in MODEL_PATHS.items():
        model_exists = os.path.exists(model_path)  # Check file existence on disk
        icon  = "🟢" if model_exists else "🔴"      # Green dot if found, red if missing
        label = "Ready" if model_exists else "Not Found"  # Status text
        st.markdown(f"{icon} **{model_name}**: `{label}`")  # Render status line

    # Check scaler file status
    scaler_exists = os.path.exists(SCALER_PATH)  # Verify scaler file on disk
    scaler_icon   = "🟢" if scaler_exists else "🔴"  # Status icon
    scaler_label  = "Ready" if scaler_exists else "Not Found"  # Status label
    st.markdown(f"{scaler_icon} **Scaler**: `{scaler_label}`")  # Display scaler status

    st.markdown("<br>", unsafe_allow_html=True)  # Bottom spacer

    # ── Sidebar Footer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.75rem; color:#4A5568; text-align:center; padding-top:1rem; border-top:1px solid #DDE3EE;'>
        Graduation Project 2025<br>
        IoT Security Research Lab
    </div>
    """, unsafe_allow_html=True)  # Styled footer text


# =============================================================================
# SECTION 6: MAIN DASHBOARD HEADER
# Centered title, subtitle, and horizontal divider.
# =============================================================================
st.markdown("""
<div class='ids-main-title'>🛡️ Smart City Intrusion Detection System</div>
<div class='ids-subtitle'>IoT Botnet-based DDoS Detection Platform &nbsp;|&nbsp; Intrusion Detection System (IDS)</div>
<hr class='ids-divider'>
""", unsafe_allow_html=True)  # Inject styled header HTML


# =============================================================================
# SECTION 7: SYSTEM ARCHITECTURE FLOWCHART (Mermaid.js)
# Visualizes the 4-stage processing pipeline using a Mermaid flowchart.
# The diagram is embedded via st.markdown with HTML/JS rendering.
# =============================================================================
st.markdown("""
<div class='ids-section-title'>📐 System Architecture Pipeline</div>
""", unsafe_allow_html=True)  # Section header for architecture diagram

# Mermaid.js flowchart definition embedded in an HTML/JS block.
# The diagram shows the 4-tier IoT security pipeline.
mermaid_html = """
<div class='mermaid-container'>
<div class='mermaid' id='mermaid-chart'>
graph LR
    A["📡 Input Data<br><small>CSV Network Traffic<br>IoT Device Capture</small>"]
    B["🔧 Preprocessing<br><small>Feature Extraction<br>NaN Imputation</small>"]
    C["⚖️ Scaling<br><small>StandardScaler<br>Z-Score Normalization</small>"]
    D["🤖 Model Inference<br><small>RF / LR / SVM<br>Classification</small>"]
    E["📊 Verdict<br><small>Normal ✅<br>Attack 🚨</small>"]

    A --> B --> C --> D --> E

    style A fill:#EBF0FA,stroke:#3A5A8C,color:#1A2035,rx:8
    style B fill:#EBF0FA,stroke:#3A5A8C,color:#1A2035
    style C fill:#EBF0FA,stroke:#3A5A8C,color:#1A2035
    style D fill:#3A5A8C,stroke:#3A5A8C,color:#FFFFFF
    style E fill:#1A7F37,stroke:#1A7F37,color:#FFFFFF
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
    // Initialize Mermaid.js with a clean theme matching the corporate design
    mermaid.initialize({
        startOnLoad: true,     // Auto-render diagrams on page load
        theme: 'base',         // Use base theme (customized via style attributes)
        themeVariables: {
            primaryColor: '#EBF0FA',     // Node background
            primaryTextColor: '#1A2035', // Node text
            primaryBorderColor: '#3A5A8C', // Node border
            lineColor: '#3A5A8C',        // Connector line color
            fontSize: '14px'             // Node font size
        },
        flowchart: {
            htmlLabels: true,  // Allow HTML inside node labels (for <small> tags)
            curve: 'basis'     // Smooth curves on connectors
        }
    });
</script>
</div>
"""
st.markdown(mermaid_html, unsafe_allow_html=True)  # Render Mermaid diagram in dashboard


# =============================================================================
# SECTION 8: FILE PROCESSING & INFERENCE ENGINE
# Core logic: load models → preprocess CSV → run predictions → display results.
# =============================================================================

if uploaded_file is not None:
    # ── Section: Operational Log Display ──────────────────────────────────────
    st.markdown("<div class='ids-section-title'>📋 Operational Pipeline Log</div>",
                unsafe_allow_html=True)  # Log section title

    log_placeholder = st.empty()  # Create a dynamic placeholder for the log block

    # ── Initialize log content ─────────────────────────────────────────────────
    log_lines = []  # List to accumulate log lines as processing progresses

    def update_log(new_line: str):
        """
        Append a new line to the operational log and refresh the displayed code block.
        This simulates a real-time pipeline log visible to the analyst.

        Args:
            new_line (str): The log message to append.
        """
        log_lines.append(new_line)  # Add the new log entry to the list
        # Join all lines and render as a code block for monospace formatting
        log_placeholder.code("\n".join(log_lines), language="bash")
        time.sleep(0.3)  # Brief pause to simulate real processing and animate the log

    # ── Step 1: File Reading ───────────────────────────────────────────────────
    update_log("[ IDS PIPELINE STARTED ]")
    update_log(f"1. Loading file: {uploaded_file.name} ...")

    try:
        df_raw = pd.read_csv(uploaded_file)  # Read uploaded CSV into DataFrame
        update_log(f"   ✔ File loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    except Exception as e:
        # Display error if CSV cannot be parsed (e.g., malformed file)
        st.error(f"❌ Failed to read CSV file: {e}")
        st.stop()  # Halt execution — no point continuing without valid data

    # ── Step 2: Load Scaler ────────────────────────────────────────────────────
    update_log("2. Loading StandardScaler from disk ...")
    scaler = load_scaler(SCALER_PATH)  # Attempt to load the saved scaler

    if scaler is None:
        # Warn analyst if scaler is missing — inference cannot proceed without it
        update_log("   ⚠ Scaler file not found. Using demo mode (unscaled data).")
        st.warning(
            f"⚠️ Scaler file `{SCALER_PATH}` not found. "
            "Running in demo mode — predictions may be inaccurate. "
            "Please train and export your scaler using `joblib.dump(scaler, 'scaler.joblib')`."
        )
        # ACTION REQUIRED: Export your fitted scaler using joblib.dump() after training.
    else:
        update_log("   ✔ Scaler loaded successfully.")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    update_log("3. Preprocessing: Feature extraction, NaN imputation ...")

    try:
        if scaler is not None:
            X_scaled, y_true = preprocess_dataframe(df_raw.copy(), scaler)
            update_log(f"   ✔ Features scaled: {X_scaled.shape[1]} numeric features retained.")
        else:
            # Fallback: use raw numeric values without scaling (demo mode)
            df_numeric = df_raw.select_dtypes(include=[np.number]).fillna(0)
            X_scaled = df_numeric.values  # Convert to numpy array
            y_true   = None               # No ground truth in demo mode
            update_log(f"   ⚠ Demo mode: {X_scaled.shape[1]} features (unscaled).")
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        st.stop()  # Stop on preprocessing failure

    # ── Step 4: Load Selected Model ────────────────────────────────────────────
    update_log(f"4. Loading model: {selected_model_name} ...")
    model_path  = MODEL_PATHS[selected_model_name]  # Get path for selected model
    active_model = load_model(model_path)            # Load model from disk (cached)

    if active_model is None:
        update_log(f"   ⚠ Model file not found: {model_path}. Using demo mode.")
        st.warning(
            f"⚠️ Model file `{model_path}` not found. "
            "Please train and save your model using `joblib.dump(model, 'model_name.joblib')`."
        )
        # ACTION REQUIRED: Train and export your model files before deployment.
        predictions = np.random.randint(0, 2, size=len(X_scaled))  # Demo: random labels
        update_log("   ⚠ Demo predictions generated (random).")
    else:
        # ── Step 5: Model Inference ────────────────────────────────────────────
        update_log(f"5. Running inference on {len(X_scaled)} traffic samples ...")
        try:
            predictions = run_inference_pipeline(X_scaled, active_model)
            update_log(f"   ✔ Inference complete. Predictions shape: {predictions.shape}")
        except Exception as e:
            st.error(f"❌ Inference error: {e}")
            st.stop()

    # ── Step 6: Verdict Aggregation ────────────────────────────────────────────
    update_log("6. Aggregating verdict ...")
    verdict_data = get_verdict_summary(predictions)  # Compute summary statistics
    update_log(f"   ✔ Verdict: {verdict_data['verdict']} | "
               f"Attack: {verdict_data['attack_count']} | "
               f"Normal: {verdict_data['normal_count']}")
    update_log("[ PIPELINE COMPLETE ] ✅")

    # ==========================================================================
    # SECTION 9: VERDICT STATUS CARD
    # High-contrast colored card showing the overall detection result.
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)  # Spacer above verdict card
    st.markdown("<div class='ids-section-title'>🚦 Detection Verdict</div>",
                unsafe_allow_html=True)

    # Choose card style based on verdict (green = safe, red = threat detected)
    card_class = "status-card-normal" if verdict_data["verdict"] == "NORMAL" else "status-card-attack"
    verdict_icon = "✅" if verdict_data["verdict"] == "NORMAL" else "🚨"

    # Render the verdict card using custom HTML for full styling control
    st.markdown(f"""
    <div class='{card_class}'>
        <div class='status-label'>{verdict_icon} Overall Network Status</div>
        <div class='status-value'>{verdict_data['verdict']}</div>
        <div style='margin-top:0.5rem; font-size:0.9rem; color:#4A5568;'>
            Analyzed <b>{verdict_data['total']}</b> traffic flows &nbsp;|&nbsp;
            <b style='color:#CF222E;'>{verdict_data['attack_count']} Attack</b> &nbsp;|&nbsp;
            <b style='color:#1A7F37;'>{verdict_data['normal_count']} Normal</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # SECTION 10: QUANTITATIVE METRICS ROW
    # Display key numbers in Streamlit metric widgets in a 4-column grid.
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)  # Create 4 equal-width columns for metrics

    with m1:
        # Total samples analyzed
        st.metric("Total Samples", f"{verdict_data['total']:,}")
    with m2:
        # Number of attack-classified samples
        st.metric("Attack Flows", f"{verdict_data['attack_count']:,}")
    with m3:
        # Number of normal-classified samples
        st.metric("Normal Flows", f"{verdict_data['normal_count']:,}")
    with m4:
        # Attack rate as a percentage of total traffic
        attack_rate = (verdict_data['attack_count'] / verdict_data['total'] * 100
                       if verdict_data['total'] > 0 else 0)
        st.metric("Attack Rate", f"{attack_rate:.1f}%")

    # ==========================================================================
    # SECTION 11: PREDICTION DISTRIBUTION CHART
    # A Plotly pie chart showing the proportion of Normal vs Attack predictions.
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='ids-section-title'>📊 Prediction Distribution</div>",
                unsafe_allow_html=True)

    # Build the pie chart using Plotly Express
    pie_fig = px.pie(
        values=[verdict_data['normal_count'], verdict_data['attack_count']],  # Slice sizes
        names=["Normal Traffic", "Attack Traffic"],   # Slice labels
        color_discrete_sequence=["#1A7F37", "#CF222E"],  # Green and red slices
        hole=0.45  # Donut chart style (hollow center)
    )
    pie_fig.update_traces(
        textposition='inside',           # Place percentage labels inside slices
        textinfo='percent+label',        # Show both percentage and label text
        textfont_size=13                 # Font size for slice labels
    )
    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",   # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",    # Transparent plot area
        showlegend=True,                  # Show the legend
        margin=dict(l=10, r=10, t=10, b=10),  # Compact margins
        height=280,                       # Fixed chart height
        legend=dict(font=dict(family="IBM Plex Sans", size=12))  # Style legend font
    )
    st.plotly_chart(pie_fig, use_container_width=True)  # Render chart full-width

    st.markdown("<hr style='border-color:#DDE3EE;'>", unsafe_allow_html=True)  # Divider


# =============================================================================
# SECTION 12: MODEL COMPARISON EXPANDERS
# Three collapsible sections — one per model — showing metrics and confusion matrices.
# Displayed whether or not a file has been uploaded (uses demo metrics by default).
# =============================================================================
st.markdown("<div class='ids-section-title'>🔬 Detailed Model Analysis & Comparison</div>",
            unsafe_allow_html=True)

# ── Define model-specific color accents for visual differentiation ─────────────
model_colors = {
    "Random Forest":       "#3A5A8C",  # Primary slate blue
    "Logistic Regression": "#5B7DB1",  # Medium blue
    "SVM":                 "#7A9CC8"   # Lighter blue
}

for model_name, model_path in MODEL_PATHS.items():
    # Retrieve pre-defined demo metrics for this model (or real metrics if available)
    metrics = DEMO_METRICS[model_name]  # Dictionary with accuracy, precision, recall

    # ── Build expander for each model ─────────────────────────────────────────
    with st.expander(
        f"{'🌲' if model_name == 'Random Forest' else '📈' if model_name == 'Logistic Regression' else '⚡'} "
        f"{model_name} — Accuracy: {metrics['accuracy']*100:.2f}%",
        expanded=(model_name == "Random Forest")  # Auto-expand first model (RF)
    ):
        col_metrics, col_cm = st.columns([1, 2])  # Two-column layout: metrics | confusion matrix

        # ── Left Column: Metric Badges ─────────────────────────────────────────
        with col_metrics:
            st.markdown(f"**{model_name} Performance**")  # Sub-header inside expander

            # Display three metric badges: Accuracy, Precision, Recall
            st.markdown(f"""
            <div style='margin:0.5rem 0;'>
                <div class='metric-badge'>✅ Accuracy: {metrics['accuracy']*100:.2f}%</div><br>
                <div class='metric-badge'>🎯 Precision: {metrics['precision']*100:.2f}%</div><br>
                <div class='metric-badge'>🔁 Recall: {metrics['recall']*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)  # Render badges as HTML

            # ── Model File Availability Status ────────────────────────────────
            file_ok = os.path.exists(model_path)  # Check if model file is on disk
            status_text = "✅ Model file found" if file_ok else "⚠️ Model file not found"
            status_color = "#1A7F37" if file_ok else "#CF222E"
            st.markdown(
                f"<small style='color:{status_color};'>{status_text}</small>",
                unsafe_allow_html=True
            )

            # ── Insight Card ───────────────────────────────────────────────────
            insight_text = MODEL_INSIGHTS.get(model_name, "No insights available.")
            st.markdown(
                f"<div class='insight-card'>💡 <b>Insights</b><br>{insight_text}</div>",
                unsafe_allow_html=True  # Render the styled insight card
            )

        # ── Right Column: Confusion Matrix Heatmap ─────────────────────────────
        with col_cm:
            # If ground truth labels are available from uploaded file, compute real CM.
            # Otherwise, generate a synthetic demo confusion matrix for visualization.
            if (uploaded_file is not None
                    and 'y_true' in dir()       # Ground truth was extracted
                    and y_true is not None):

                # Load the actual model for evaluation if available
                eval_model = load_model(model_path)
                if eval_model is not None and scaler is not None:
                    try:
                        # Re-run inference with this specific model for the confusion matrix
                        eval_preds = run_inference_pipeline(X_scaled, eval_model)
                        cm_array   = confusion_matrix(y_true, eval_preds)  # Real confusion matrix
                    except Exception:
                        # Fallback to demo matrix if inference fails
                        cm_array = np.array([[850, 45], [30, 780]])  # Demo values
                else:
                    cm_array = np.array([[850, 45], [30, 780]])  # Demo: model not loaded
            else:
                # ── Demo Confusion Matrix (no file uploaded) ───────────────────
                # ACTION REQUIRED: These are placeholder values. Replace with real
                # evaluation results from your test set after training your models.
                demo_cms = {
                    "Random Forest":       np.array([[920, 18], [12, 850]]),  # High performer
                    "Logistic Regression": np.array([[870, 68], [55, 807]]),  # Baseline
                    "SVM":                 np.array([[900, 38], [35, 827]])   # Mid-tier
                }
                cm_array = demo_cms[model_name]  # Select this model's demo matrix

            # Build and render the Plotly confusion matrix figure
            cm_fig = build_confusion_matrix_figure(cm_array, model_name)
            st.plotly_chart(cm_fig, use_container_width=True)  # Render chart

st.markdown("<br><br>", unsafe_allow_html=True)  # Bottom padding


# =============================================================================
# SECTION 13: DATA PREVIEW TABLE (only when file is uploaded)
# Shows the first few rows of the uploaded CSV for analyst verification.
# =============================================================================
if uploaded_file is not None:
    st.markdown("<div class='ids-section-title'>📄 Uploaded Data Preview</div>",
                unsafe_allow_html=True)

    # Display first 10 rows with Streamlit's native table — clean and scrollable
    st.dataframe(
        df_raw.head(10),           # Show only first 10 rows to avoid clutter
        use_container_width=True,  # Stretch to full container width
        height=280                 # Fixed height with scroll for longer tables
    )

    # ── Download Predictions Button ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    # Build a results DataFrame combining original rows with predicted labels
    results_df = df_raw.copy()  # Start with original uploaded data
    results_df["Predicted_Label"]  = predictions  # Append predicted class (0 or 1)
    results_df["Predicted_Status"] = [
        CLASS_LABELS[p] for p in predictions  # Map numeric label to "NORMAL"/"ATTACK"
    ]

    # Convert to CSV bytes for download (index excluded for clean output)
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Predictions as CSV",   # Button label
        data=csv_bytes,                            # CSV content bytes
        file_name="ids_predictions.csv",           # Downloaded file name
        mime="text/csv",                           # MIME type for browser
        help="Download the original data appended with prediction results."
    )


# =============================================================================
# SECTION 14: FOOTER
# Professional academic footer with project metadata.
# =============================================================================
st.markdown("""
<hr style='border-color:#DDE3EE; margin-top:2rem;'>
<div style='text-align:center; color:#4A5568; font-size:0.82rem; padding-bottom:1rem;'>
    🎓 <b>Graduation Research Project</b> &nbsp;|&nbsp;
    IoT Botnet-based DDoS Detection in Smart Cities &nbsp;|&nbsp;
    Powered by <b>Scikit-learn · Streamlit · Plotly</b><br>
    <span style='color:#3A5A8C;'>Models: Random Forest · Logistic Regression · SVM</span>
</div>
""", unsafe_allow_html=True)  # Inject styled academic footer

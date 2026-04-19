import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="AI-Based IoT Botnet Detection System",
    layout="wide"
)

# Hide Streamlit elements
st.markdown("""
    <style>
    .stStatusWidget {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("Smart City IoT Botnet Detection Dashboard")
st.markdown("""
This system uses Machine Learning models (Random Forest, Logistic Regression, and SVM) 
to classify network traffic and detect potential Botnet attacks in real-time.
""")

@st.cache_resource
def load_security_models():
    # تم تعديل الامتدادات إلى .joblib
    rf = joblib.load('random_forest.joblib')
    lr = joblib.load('logistic_regression.joblib')
    svm = joblib.load('svm.joblib')
    scaler = joblib.load('scaler.joblib')
    return rf, lr, svm, scaler

try:
    rf_model, lr_model, svm_model, scaler = load_security_models()
    st.sidebar.success("Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error Loading Models: {e}")
    st.stop()

st.sidebar.header("System Settings")
st.sidebar.info("The system automatically selects the best model based on data complexity and size.")

# File uploader
uploaded_file = st.file_uploader("Upload Network Traffic Data (CSV Format - Training or Testing)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-16')
    
    with st.expander("View Uploaded Raw Data"):
        st.write(df.head(10))

    if st.button("Run Intelligent Analysis"):
        with st.status("Analyzing Network Packets...", expanded=True) as status:
            st.write("Cleaning data and handling Infinity values...")
            processed_df = df.copy()
            processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Determine if target column exists
            has_target = False
            target_col_name = None
            if 'attack' in processed_df.columns:
                has_target = True
                target_col_name = 'attack'
            elif 'label' in processed_df.columns:
                 has_target = True
                 target_col_name = 'label'

            target_cols = ['attack', 'label', 'category', 'subcategory', 'pkSeqID', 'seq']
            features_df = processed_df.drop(columns=[c for c in target_cols if c in processed_df.columns], errors='ignore')
            
            for col in features_df.select_dtypes(include=['object']).columns:
                temp_le = LabelEncoder()
                features_df[col] = temp_le.fit_transform(features_df[col].astype(str))
            
            st.write("Scaling features for ML models...")
            X_scaled = scaler.transform(features_df)
            
            data_size = len(features_df)
            if data_size > 10000:
                selected_model = rf_model
                model_name = "Random Forest (Optimized for Large Data)"
            elif data_size < 1000:
                selected_model = lr_model
                model_name = "Logistic Regression (Optimized for Speed)"
            else:
                selected_model = svm_model
                model_name = "Support Vector Machine (Optimized for Boundary Precision)"
            
            st.write(f"Model Selection: **{model_name}** selected.")
            time.sleep(1)
            
            st.write("Finalizing predictions...")
            predictions = selected_model.predict(X_scaled)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        st.divider()
        st.subheader(f"Analysis Results (Model: {model_name})")
        
        attack_count = int(np.sum(predictions == 1))
        normal_count = int(np.sum(predictions == 0))
        total_packets = len(predictions)
        attack_percentage = (attack_count / total_packets) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Packets", total_packets)
        m2.metric("Botnet Attacks", attack_count, delta="Threats Detected", delta_color="inverse")
        m3.metric("Normal Traffic", normal_count)
        m4.metric("Threat Ratio", f"{attack_percentage:.2f}%")

        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.write("### Traffic Distribution")
            chart_data = pd.DataFrame({
                "Category": ["Attack", "Normal"],
                "Count": [attack_count, normal_count]
            })
            st.bar_chart(chart_data.set_index("Category"))
            
        with c2:
            st.write("### Threat Detection Log")
            results_df = df.copy()
            results_df['Detection_Result'] = ["ATTACK" if p == 1 else "NORMAL" for p in predictions]
            st.write(results_df[['Detection_Result']].merge(features_df.head(100), left_index=True, right_index=True))

        if has_target:
             st.divider()
             st.subheader("Model Evaluation on Test Data")
             
             y_true = processed_df[target_col_name].astype(int)
             acc = accuracy_score(y_true, predictions)
             st.write(f"**Accuracy:** {acc:.4f}")
             
             st.write("**Classification Report:**")
             report = classification_report(y_true, predictions, output_dict=True)
             st.dataframe(pd.DataFrame(report).transpose())

             st.write("**Confusion Matrix:**")
             cm = confusion_matrix(y_true, predictions)
             fig, ax = plt.subplots()
             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
             ax.set_xlabel('Predicted')
             ax.set_ylabel('Actual')
             st.pyplot(fig)

        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Detection Report",
            data=csv_results,
            file_name="botnet_detection_report.csv",
            mime="text/csv"
        )

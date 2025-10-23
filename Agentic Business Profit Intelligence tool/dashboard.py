import streamlit as st
import pandas as pd
import os
import shutil
from pipeline.perfect_flow import run_business_pipeline
from ydata_profiling import ProfileReport
from fpdf import FPDF
import json

# Streamlit Page Setup
st.set_page_config(page_title="Agentic Business Profit Intelligence", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
        .block-container {
            max-width: 1000px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin: auto;
        }
        h1 {
            text-align: center;
            color: #2C3E50;
            font-size: 2rem !important;
        }
        .stButton>button {
            width: 100%;
            background-color: #3B82F6;
            color: white;
            font-weight: 600;
            border-radius: 6px;
            height: 3rem;
        }
        .stButton>button:hover {
            background-color: #2563EB;
        }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit Header ---
st.title("📊 Agentic Business Profit Intelligence Dashboard")

# --- Initialize Session State ---
if "insights_text" not in st.session_state:
    st.session_state.insights_text = ""
if "images" not in st.session_state:
    st.session_state.images = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# --- File Upload ---
uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded:
    os.makedirs("Data/raw", exist_ok=True)
    file_path = os.path.join("Data/raw", uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"✅ File saved to {file_path}")

    # --- Data Preview ---
    df = pd.read_csv(file_path, encoding="latin1")
    st.write("### 👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- Profiling Report ---
    if st.button("📈 Generate Data Profiling Report"):
        with st.spinner("Generating advanced profiling report... ⏳"):
            profile = ProfileReport(df, title="Business Data Profiling Report", explorative=True)
            os.makedirs("Results", exist_ok=True)
            profile_path = "Results/data_profile.html"
            profile.to_file(profile_path)
            st.success("✅ Data profiling report generated successfully!")

            # Display HTML inline
            with open(profile_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

    # --- Run Full AI Analysis ---
    if st.button("🚀 Run Full AI Analysis"):
        with st.spinner("Running Prefect pipeline... Please wait ⏳"):
            try:
                # Clean old results
                if os.path.exists("Results"):
                    shutil.rmtree("Results")
                os.makedirs("Results/visualization", exist_ok=True)

                # Run your pipeline
                run_business_pipeline(file_path)
                st.success("✅ Pipeline executed successfully!")

                # Load metrics
                metrics_path = "Results/model_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        st.session_state.metrics = json.load(f)

                # Load insights
                insights_path = "Results/insights.txt"
                if os.path.exists(insights_path):
                    with open(insights_path, encoding="utf-8") as f:
                        st.session_state.insights_text = f.read()

                # Load images
                vis_folder = "Results/visualization"
                if os.path.exists(vis_folder):
                    st.session_state.images = [
                        img for img in os.listdir(vis_folder) if img.endswith(".png")
                    ]

            except Exception as e:
                st.error(f"❌ Error while running pipeline: {e}")

    # --- Show Results (Persistent with session_state) ---
    metrics = st.session_state.metrics
    insights_text = st.session_state.insights_text
    images = st.session_state.images

    # Model Metrics
    if metrics:
        st.subheader("📈 Model Metrics")
        target = metrics.get("Target", "Unknown")
        r2 = metrics.get("R²", "N/A")
        top_features = metrics.get("Top Features", {})

        st.write(f"**🎯 Target Variable:** `{target}`")
        st.write(f"**📊 Model R² Score:** `{r2}`")

        if top_features:
            st.write("**💡 Top Influencing Features:**")
            feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
            st.dataframe(feat_df, use_container_width=True)

    # 🧠 Show AI Insights only if available
    if insights_text.strip():
        st.subheader("🧠 AI-Generated Insights")
        st.write(insights_text)

    # 📊 Show Visualizations only if available
    if images:
        st.subheader("📊 Visualizations")
        for img in images:
            st.image(os.path.join("Results/visualization", img), use_container_width=True)

        # --- 📄 Generate AI Business Report (only after analysis is ready) ---
        st.subheader("📄 Generate AI Business Report")
        if st.button("📝 Generate PDF Report"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Agentic Business Profit Intelligence Report", ln=True, align="C")

                # Dataset info
                pdf.set_font("Arial", size=12)
                pdf.ln(10)
                pdf.multi_cell(0, 10, f"Dataset: {uploaded.name}\n")

                # AI Insights
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, txt="AI Insights", ln=True)
                pdf.set_font("Arial", size=12)
                from textwrap import wrap
                safe_insights = insights_text if insights_text else "No AI insights available."
                safe_insights_wrapped = "\n".join(wrap(safe_insights, width=110))
                pdf.multi_cell(0, 8, safe_insights_wrapped)

                # Visualizations
                pdf.ln(5)
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, txt="Visualizations", ln=True)
                for img in images[:3]:
                    img_path = os.path.join("Results/visualization", img)
                    if os.path.exists(img_path):
                        pdf.image(img_path, w=160)
                        pdf.ln(5)

                # ✅ Properly generate bytes for Streamlit download
                pdf_bytes = bytes(pdf.output(dest="S"))
                st.session_state.pdf_bytes = pdf_bytes
                st.success("✅ PDF Report Generated! Scroll down to download it.")

            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")

        # --- Download Button ---
        if st.session_state.pdf_bytes:
            st.download_button(
                label="⬇️ Download AI Business Report (PDF)",
                data=st.session_state.pdf_bytes,
                file_name=f"AI_Report_{uploaded.name.split('.')[0]}.pdf",
                mime="application/pdf"
            )

else:
    st.info("👈 Please upload a CSV file to start your analysis.")

import streamlit as st
import pandas as pd
import os
import shutil
from pipeline.perfect_flow import run_business_pipeline
from fpdf import FPDF
from ydata_profiling import ProfileReport
from io import BytesIO

# Streamlit Page Setup
st.set_page_config(page_title="Agentic Business Profit Intelligence", layout="centered")

# CSS Styling
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

# Header
st.title("📊 Agentic Business Profit Intelligence Dashboard")

# File Upload
uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded:
    os.makedirs("Data/raw", exist_ok=True)
    file_path = os.path.join("Data/raw", uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"✅ File saved to {file_path}")

    # 🧾 Preview Data
    df = pd.read_csv(file_path, encoding="latin1")
    st.write("### 👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # 🧠 Generate Data Profiling Report
    if st.button("📈 Generate Data Profiling Report"):
        with st.spinner("Generating advanced profiling report... ⏳"):
            profile = ProfileReport(df, title="Business Data Profiling Report", explorative=True)
            os.makedirs("Results", exist_ok=True)
            profile.to_file("Results/data_profile.html")
            st.success("✅ Data profiling report generated successfully!")

            # Show HTML report inline
            with open("Results/data_profile.html", "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

    # 🚀 Run Full Pipeline
    if st.button("🚀 Run Full AI Analysis"):
        with st.spinner("Running Prefect pipeline... Please wait ⏳"):
            try:
                # Clear old results
                if os.path.exists("Results"):
                    shutil.rmtree("Results")
                os.makedirs("Results/visualization", exist_ok=True)

                run_business_pipeline(file_path)
                st.success("✅ Pipeline executed successfully!")

                # 📈 Model Metrics Bar
                st.subheader("📈 Model Metrics")
                if os.path.exists("Results/model_metrics.json"):
                    import json
                    with open("Results/model_metrics.json", "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                    # Show metrics dynamically
                    target = metrics.get("Target", "Unknown")
                    r2 = metrics.get("R²", "N/A")
                    top_features = metrics.get("Top Features", {})

                    st.write(f"**🎯 Target Variable:** `{target}`")
                    st.write(f"**📊 Model R² Score:** `{r2}`")

                    if top_features:
                        st.write("**💡 Top Influencing Features:**")
                        feat_df = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"])
                        st.dataframe(feat_df, use_container_width=True)

                # 🧠 AI Insights
                st.subheader("🧠 AI-Generated Insights")
                insights_text = ""
                if os.path.exists("Results/insights.txt"):
                    with open("Results/insights.txt", encoding="utf-8") as f:
                        insights_text = f.read()
                        st.write(insights_text)

                # 📊 Visualizations
                st.subheader("📊 Visualizations")
                vis_folder = "Results/visualization"
                images = []
                if os.path.exists(vis_folder):
                    images = [img for img in os.listdir(vis_folder) if img.endswith(".png")]
                    for img in images:
                        st.image(os.path.join(vis_folder, img), use_container_width=True)
                else:
                    st.warning("No visualizations found.")

                # 📄 Generate Dynamic Report (PDF)
                st.subheader("📄 Generate Dynamic Report")
                if st.button("📝 Generate AI Business Report (PDF)"):
                    pdf_buffer = BytesIO()
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, txt="Agentic Business Profit Intelligence Report", ln=True, align="C")

                    pdf.set_font("Arial", size=12)
                    pdf.ln(10)
                    pdf.multi_cell(0, 10, f"Dataset: {uploaded.name}\n\nAI Insights:\n{insights_text}")

                    pdf.ln(10)
                    pdf.cell(0, 10, txt="Visualizations:", ln=True)
                    for img in images[:3]:
                        img_path = os.path.join(vis_folder, img)
                        pdf.image(img_path, w=160)
                        pdf.ln(5)

                    pdf.output(pdf_buffer)
                    pdf_bytes = pdf_buffer.getvalue()

                    st.success("✅ PDF Report Generated Successfully!")
                    st.download_button(
                        label="⬇️ Download AI Business Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"AI_Report_{uploaded.name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error(f"❌ Error while running pipeline: {e}")

else:
    st.info("👈 Please upload a CSV file to start your analysis.")

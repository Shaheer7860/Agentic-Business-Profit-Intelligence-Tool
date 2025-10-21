# Pipeline/prefect_flow.py

import sys, os, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prefect import flow, task
from Src_code.data_ingestion import load_data
from Src_code.data_cleaning import clean_data
from Src_code.model_training import train_model
from Src_code.visualization import generate_visuals
from Src_code.agentic_ai import generate_ai_insights


# 🧹 Clean previous runs
def clear_previous_results():
    """Remove all old outputs before a new pipeline run."""
    folders_to_clear = ["Results", "Data/Processed"]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    os.makedirs("Results/visualization", exist_ok=True)
    os.makedirs("Data/Processed", exist_ok=True)
    print("🧹 Cleared old Results and Processed folders for a fresh run.")


# Ensure directory structure exists
def ensure_dirs():
    os.makedirs("Data/Processed", exist_ok=True)
    os.makedirs("Results/visualization", exist_ok=True)


# ============================
# Prefect Tasks
# ============================

@task(name="Load Data", cache_key_fn=None)
def task_load_data(path: str):
    print(f"📥 Loading dataset from: {path}")
    df = load_data(path)
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


@task(name="Clean Data", cache_key_fn=None)
def task_clean_data(df):
    print("🧹 Cleaning data...")
    df_cleaned = clean_data(df)
    df_cleaned.to_csv("Data/Processed/cleaned_dataset.csv", index=False)
    print("✅ Cleaned data saved to Data/Processed/cleaned_dataset.csv")
    return df_cleaned


@task(name="Train Model", cache_key_fn=None)
def task_train_model(df_cleaned):
    """Train model only if numeric target exists."""
    numeric_cols = df_cleaned.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("⚠️ No numeric columns available. Skipping model training.")
        return None

    print("🧠 Training model...")
    model, metrics = train_model(df_cleaned)
    with open("Results/metrics.txt", "w", encoding="utf-8") as f:
        f.write(str(metrics))
    print(f"✅ Model trained successfully: {metrics}")
    return metrics


@task(name="Generate Visuals", cache_key_fn=None)
def task_generate_visuals(df_cleaned):
    print("🎨 Letting AI decide the best visualizations for this dataset...")
    generate_visuals(df_cleaned)
    print("✅ AI-driven visuals saved to Results/visualization/")


@task(name="Generate AI Insights", cache_key_fn=None)
def task_generate_ai_insights(df_cleaned):
    print("🧠 Generating AI-driven insights...")
    generate_ai_insights(df_cleaned)
    print("✅ Insights saved to Results/insights.txt")


# ============================
# Prefect Flow
# ============================

@flow(name="Agentic Business Profit Intelligence")
def business_pipeline(file_path: str = "Data/raw/Superstore.csv"):
    """
    Main Prefect flow that orchestrates the entire pipeline.
    You can pass a different dataset path to process other files.
    """
    print(f"🚀 Starting Prefect pipeline using dataset: {file_path}")
    clear_previous_results()
    ensure_dirs()

    df = task_load_data(file_path)
    df_clean = task_clean_data(df)
    task_train_model(df_clean)
    task_generate_visuals(df_clean)
    task_generate_ai_insights(df_clean)

    print("🎯 Prefect Flow completed successfully!")


# ============================
# Streamlit / External Entry
# ============================

def run_business_pipeline(file_path: str):
    """
    Runs the Prefect pipeline directly (for Streamlit or CLI).
    Ensures a clean, fresh run each time.
    """
    print(f"⚙️ Running Prefect pipeline from Streamlit using: {file_path}")
    clear_previous_results()
    ensure_dirs()

    df = task_load_data.fn(file_path)
    df_clean = task_clean_data.fn(df)
    task_train_model.fn(df_clean)
    task_generate_visuals.fn(df_clean)
    task_generate_ai_insights.fn(df_clean)

    print("✅ Pipeline execution finished successfully via Streamlit.")


# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    # Default fallback (you can run other datasets by passing --file argument)
    business_pipeline("Data/raw/Superstore.csv")

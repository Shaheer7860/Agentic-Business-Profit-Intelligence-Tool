import os
import pandas as pd

def clean_data(df, dataset_name="dataset"):
    """
    Cleans and standardizes the dataset for further analysis or AI processing.
    Compatible with any dataset used in the Prefect pipeline.
    """

    print("🧹 Cleaning data...")

    # Normalize column names
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True).str.lower()

    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)

    # Detect common business columns (flexible for any dataset)
    profit_col = None
    sales_col = None
    order_date_col = None

    for col in df.columns:
        if "profit" in col:
            profit_col = col
        elif "sale" in col:
            sales_col = col
        elif "order date" in col or "orderdate" in col:
            order_date_col = col

    # Add derived features when possible
    if profit_col and sales_col:
        try:
            df["profit_margin"] = (df[profit_col] / df[sales_col]) * 100
        except Exception:
            print("⚠️ Could not calculate profit margin due to invalid data types.")

    if order_date_col:
        try:
            df[order_date_col] = pd.to_datetime(df[order_date_col], errors="coerce")
            df["order_month"] = df[order_date_col].dt.month
            df["order_year"] = df[order_date_col].dt.year
        except Exception:
            print("⚠️ Could not parse order dates properly.")

    # Ensure processed data folder exists
    os.makedirs("Data/Processed", exist_ok=True)

    # Save cleaned dataset dynamically
    cleaned_path = f"Data/Processed/cleaned_{dataset_name}.csv"
    df.to_csv(cleaned_path, index=False)

    print(f"✅ Cleaned data saved to {cleaned_path}")
    return df

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from langchain_openai import ChatOpenAI
from sklearn.metrics import confusion_matrix
import numpy as np

def generate_visuals(df):
    """
    Professional-grade AI-driven visualization generator.
    Automatically suggests and creates diverse, publication-ready charts.
    """

    print("🎨 Generating AI-powered professional visualizations...")

    # Ensure output folder
    os.makedirs("Results/visualization", exist_ok=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\xa0", "", regex=True).str.lower()

    # Identify column types
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Build concise dataset summary
    dataset_summary = f"""
    Numeric columns: {numeric_cols}
    Categorical columns: {categorical_cols}
    """

    # Ask AI for a visualization plan
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    prompt = f"""
    You are an expert data visualization analyst.
    Given the following dataset schema:

    {dataset_summary}

    Suggest 5 visualizations that best explain patterns, relationships, or trends.
    Include a variety of types (bar, line, scatter, box, histogram, heatmap, pie, confusion matrix etc.)
    Return ONLY valid JSON in this exact structure:

    [
      {{"type": "bar", "x": "region", "y": "sales", "title": "Sales by Region"}},
      {{"type": "scatter", "x": "discount", "y": "profit", "title": "Discount vs Profit"}}
    ]

    Only use columns that exist in the dataset.
    """

    response = llm.invoke(prompt)

    try:
        vis_plan = json.loads(response.content)
    except Exception:
        print("⚠️ AI visualization plan invalid. Using fallback visuals.")
        vis_plan = []

    # Fallback visuals
    if not vis_plan:
        vis_plan = []
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            vis_plan.append({
                "type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0].capitalize()} by {categorical_cols[0].capitalize()}"
            })
        if len(numeric_cols) > 1:
            vis_plan.append({
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": f"{numeric_cols[1].capitalize()} vs {numeric_cols[0].capitalize()}"
            })
        if len(numeric_cols) > 2:
            vis_plan.append({"type": "heatmap", "title": "Correlation Heatmap"})

    # Universal aesthetic improvements
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (8, 5)
    })

    # Generate visuals
    for i, vis in enumerate(vis_plan, start=1):
        try:
            chart_type = vis["type"].lower()
            title = vis.get("title", f"AI Chart {i}")
            x = vis.get("x")
            y = vis.get("y")

            plt.figure()

            # BAR CHART
            if chart_type == "bar" and x in df.columns and y in df.columns:
                top_vals = df[x].value_counts().nlargest(10).index
                sns.barplot(x=x, y=y, data=df[df[x].isin(top_vals)], ci=None)
                plt.xticks(rotation=45, ha="right")

            # SCATTER
            elif chart_type == "scatter" and x in df.columns and y in df.columns:
                sns.scatterplot(x=x, y=y, data=df, alpha=0.7)

            # LINE CHART
            elif chart_type == "line" and x in df.columns and y in df.columns:
                sns.lineplot(x=x, y=y, data=df)

            # HISTOGRAM
            elif chart_type == "hist" and x in df.columns:
                sns.histplot(df[x], bins=20, kde=True)

            # BOX PLOT
            elif chart_type == "box" and x in df.columns and y in df.columns:
                sns.boxplot(x=x, y=y, data=df)

            # HEATMAP
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include="number").corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

            # PIE CHART
            elif chart_type == "pie" and x in df.columns:
                data_counts = df[x].value_counts().nlargest(6)
                plt.pie(data_counts, labels=data_counts.index, autopct="%1.1f%%", startangle=90)
                plt.axis("equal")

            # CONFUSION MATRIX (NEW FEATURE)
            elif chart_type == "confusion_matrix":
                # Automatically pick two categorical columns if not specified
                if len(categorical_cols) >= 2:
                    cat1, cat2 = categorical_cols[:2]
                    conf_data = pd.crosstab(df[cat1], df[cat2])
                    sns.heatmap(conf_data, annot=True, cmap="Blues", fmt="d")
                    plt.title(f"Confusion Matrix: {cat1} vs {cat2}")
                else:
                    print("⚠️ Not enough categorical columns for confusion matrix.")
                    plt.close()
                    continue

            else:
                print(f"⚠️ Unsupported or invalid config: {vis}")
                plt.close()
                continue

            plt.title(title)
            plt.tight_layout()
            plt.savefig(f"Results/visualization/ai_visual_{i}.png", dpi=300)
            plt.close()
            print(f"✅ Saved: ai_visual_{i}.png")

        except Exception as e:
            print(f"❌ Failed to create {vis.get('title', f'Chart {i}')}: {e}")

    print("✅ All AI-enhanced professional visualizations saved successfully.")


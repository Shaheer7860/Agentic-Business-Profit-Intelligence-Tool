import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import json

def train_model(df):
    """
    Train a RandomForest model to predict 'Profit' (or closest numeric target)
    and generate a feature importance visualization for business insight.
    """

    df = df.copy()
    df.columns = df.columns.str.lower()

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    # Identify the target variable
    target_col = None
    for col in ['profit', 'sales', 'revenue']:
        if col in numeric_cols:
            target_col = col
            break

    if not target_col:
        print("⚠️ No suitable numeric target found for model training.")
        return None, {}

    # Encode categorical variables
    le = LabelEncoder()
    for c in categorical_cols:
        df[c] = le.fit_transform(df[c].astype(str))

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Compute feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Save visualization
    os.makedirs("Results/visualization", exist_ok=True)
    plt.figure(figsize=(8, 5))
    importances.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Top Factors Influencing {target_col.capitalize()}")
    plt.ylabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("Results/visualization/feature_importance.png")
    plt.close()

    # Save numeric summary
    top_features = importances.head(10).to_dict()
    metrics = {
        "Target": target_col,
        "R²": round(score, 3),
        "Top Features": top_features
    }

    # Save to JSON for dashboard or AI use
    with open("Results/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Model trained to predict '{target_col}'. R² Score: {score:.3f}")
    print("📊 Feature importance chart and metrics saved to Results/visualization and Results/model_metrics.json")

    return model, metrics

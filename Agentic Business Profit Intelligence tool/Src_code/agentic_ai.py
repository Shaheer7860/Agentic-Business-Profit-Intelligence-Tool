from langchain_openai import ChatOpenAI
import pandas as pd
import json

def generate_ai_insights(df):
    """
    Uses OpenAI LLM to produce 20–30 deep insights from any dataset.
    Works dynamically based on dataset columns .
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    # Create structured dataset summary
    dataset_summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    # Convert the summary to a JSON string safely (outside the f-string)
    dataset_json = json.dumps(dataset_summary, indent=2)

    # Build the LLM prompt
    prompt = f"""
You are an expert Data Science and Business Intelligence Analyst.

The user has provided a dataset with the following summary:
{dataset_json}

Your task:
- Perform a thorough analytical review and extract **20–30 meaningful, data-driven insights**.
- Each insight should be concise, specific, and written in **natural business language** — no placeholders or generic statements.
- Do not use placeholders like "$X" or "Region B".
- Avoid numbering (no "Insight 1", "Insight 2", etc.).
- Instead, organize your response into short **paragraphs or bullet points**, grouped by themes such as:
  • Sales and Profit Trends  
  • Customer Behavior  
  • Regional or Category Insights  
  • Discount and Pricing Patterns  
  • Operational or Shipping Observations 
  • Strategic Recommendations 
  • Statistical summaries (averages, trends, distributions)
      . Relationships between key numeric columns
      . Category-based performance or differences
      . Potential anomalies or outliers
      . Patterns that may influence performance 

Each insight should:
- Reference real patterns implied by the dataset structure (e.g., relationships, differences, anomalies).
- Read like a business consultant’s summary — crisp, confident, and actionable.
- Be at least **20 separate insights** (but merge related points naturally for flow).

End your response with a short section titled:
**Strategic Recommendations**
Summarize 3–5 key actions the business could take based on these findings.

"""


    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Generate insights
    response = llm.invoke(prompt)

    # Save output
    with open("Results/insights.txt", "w", encoding="utf-8") as f:
        f.write(response.content)

    print("✅ 20–30 detailed insights generated and saved to Results/insights.txt.")

Agentic Business Profit Intelligence Dashboard

1. Overview

The Agentic Business Profit Intelligence Dashboard is a powerful, user-friendly web application designed to automate the complex process of business data analysis. It empowers users to upload their own datasets (in CSV format) and receive a comprehensive analysis that includes data profiling, predictive modeling, AI-generated insights, and rich visualizations.

The core of the application is an automated pipeline that leverages machine learning to identify key drivers of profitability and a large language model (LLM) agent to interpret these findings in plain English. The final output is a downloadable, professional report summarizing the entire analysis.

2. Key Features

Easy Data Upload: A simple drag-and-drop interface to upload CSV files.

Instant Data Preview: Quickly view the first few rows of your dataset to ensure it's loaded correctly.

Automated Data Profiling: Generate an in-depth exploratory data analysis (EDA) report using ydata-profiling to understand data distributions, correlations, and missing values.

One-Click AI Analysis: Execute a complete backend pipeline that performs:

Data cleaning and preprocessing.

Training a predictive machine learning model (e.g., regression to predict a target variable).

Identifying the most influential features affecting the target.

Generating relevant data visualizations.

AI-Generated Insights: Utilizes an AI agent (powered by LangChain and OpenAI) to translate model metrics and feature importance into actionable, human-readable business insights.

Dynamic PDF Reporting: Automatically compile the model metrics, AI insights, and key visualizations into a downloadable PDF report.

3. System Architecture

The application consists of three main components:

Streamlit Frontend (dashboard.py): A modern web interface that handles user interaction, file uploads, and the display of results. It acts as the control center for the application.

Prefect Workflow Pipeline (pipeline/perfect_flow.py): An orchestrated backend workflow that manages the entire data analysis process. Prefect ensures that each step (data loading, cleaning, training, visualization, insight generation) runs reliably and in the correct order.

AI Core (Scikit-learn & LangChain):

scikit-learn is used for the machine learning tasks, including training a model to understand the data's underlying patterns.

LangChain and OpenAI form the "agentic" part of the system. The AI agent receives the structured output from the machine learning model (like feature importance) and synthesizes it into a narrative with actionable advice.

4. Technology Stack

Frontend: Streamlit

Backend & Pipeline: Prefect

Data Manipulation: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn, Plotly

Machine Learning: Scikit-learn

AI & Language Models: LangChain, OpenAI

Data Profiling: ydata-profiling

PDF Generation: FPDF2

5. Setup and Usage

Prerequisites

Python 3.8+

An OpenAI API Key

Installation

Clone the repository (or set up your project folder):

git clone <your-repo-url>
cd <your-repo-folder>


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required dependencies:

pip install -r requirements.txt


Set up your OpenAI API Key:
Create a .env file in the root of your project directory and add your API key:

OPENAI_API_KEY="your-secret-key-here"


The application code will need to be configured to load this key.

Running the Application

Launch the Streamlit app:

streamlit run dashboard.py


Open your web browser to the local URL provided by Streamlit (usually http://localhost:8501).

How to Use the Dashboard

Upload a CSV file using the file uploader.

Preview the data to confirm it's correct.

(Optional) Click "Generate Data Profiling Report" for a deep dive into your dataset's statistics.

Click "Run Full AI Analysis" to execute the main pipeline. Wait for the process to complete.

Review the results:

Model Metrics (R² score, Top Features)

AI-Generated Insights

Visualizations

Click "Generate AI Business Report (PDF)" to create and download your final report.
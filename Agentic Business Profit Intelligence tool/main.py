import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipeline.perfect_flow import business_pipeline

def main():
    print("🚀 Launching Agentic Business Intelligence Pipeline via Prefect...")
    business_pipeline()

if __name__ == "__main__":
    main()
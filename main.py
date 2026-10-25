"""
main.py
-------
Single-command entry point to run the entire Credit Scoring pipeline:
1. Verify or generate synthetic credit dataset.
2. Run feature engineering.
3. Train Logistic Regression, Decision Tree, and Random Forest models.
4. Evaluate models (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Generate high-resolution visual evaluation charts.
6. Save the champion model for inference and the Streamlit web app.
"""

import os
import sys

# Ensure current workspace root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_credit_dataset
from src.model_training import train_and_evaluate_all


def main():
    print("""
    =======================================================
       CREDIT SCORING MODEL - MACHINE LEARNING PIPELINE
                 CodeAlpha Internship Project
    =======================================================
    """)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    data_file = os.path.join(data_dir, 'credit_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    vis_dir = os.path.join(base_dir, 'visualizations')
    
    # Check if dataset exists, if not generate it
    if not os.path.exists(data_file):
        print("Data file not detected. Generating synthetic credit dataset (7,500 records)...")
        os.makedirs(data_dir, exist_ok=True)
        df = generate_credit_dataset(num_samples=7500, random_state=42)
        df.to_csv(data_file, index=False)
        print(f"Dataset generated and saved to {data_file}")
    else:
        print(f"Found existing credit dataset: {data_file}")
        
    # Run training and evaluation pipeline
    metrics_df, _, _ = train_and_evaluate_all(
        data_path=data_file,
        output_dir=models_dir,
        vis_dir=vis_dir,
        random_state=42
    )
    
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED!")
    print("="*60)
    print("To test an individual applicant on the command line:")
    print("  python predict.py")
    print("\nTo launch the interactive Web Dashboard:")
    print("  streamlit run app.py")
    print("="*60)


if __name__ == '__main__':
    main()

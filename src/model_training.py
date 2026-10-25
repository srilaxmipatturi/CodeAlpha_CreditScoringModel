"""
model_training.py
-----------------
Orchestrates model training, evaluation, comparison, and persistence.

Algorithms Implemented:
1. Logistic Regression:
   - Linear probabilistic classifier.
   - High interpretability; industry standard for regulatory credit scorecards.
2. Decision Tree Classifier:
   - Rule-based tree model.
   - Captures non-linear thresholds and decision paths.
3. Random Forest Classifier:
   - Bagging ensemble of decision trees.
   - Reduces variance, robust against outliers, high predictive power.
"""

import os
import json
import tempfile
import joblib
import pandas as pd
import numpy as np

# Set writable MPLCONFIGDIR to silence matplotlib warnings on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import (
    engineer_financial_features,
    build_preprocessor,
    get_column_definitions
)
from src.evaluation import (
    evaluate_single_model,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metrics_comparison_chart
)


def train_and_evaluate_all(
    data_path: str,
    output_dir: str = 'models',
    vis_dir: str = 'visualizations',
    random_state: int = 42
):
    """
    Full pipeline execution:
    1. Load data
    2. Engineer features
    3. Split train/test
    4. Fit preprocessing pipeline
    5. Train classifiers
    6. Evaluate & save plots
    7. Save champion model
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 1: LOADING & EXPLORING RAW DATASET")
    print("="*60)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}. Run generate_data.py first.")
        
    raw_df = pd.read_csv(data_path)
    print(f"Loaded {raw_df.shape[0]} records with {raw_df.shape[1]} columns.")
    
    target_col = 'loan_status'
    print(f"Target distribution:\n{raw_df[target_col].value_counts(normalize=True).round(3)}")
    
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    df_engineered = engineer_financial_features(raw_df)
    cat_cols, num_cols = get_column_definitions()
    
    print(f"Engineered features added. Total columns now: {df_engineered.shape[1]}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    
    X = df_engineered[cat_cols + num_cols]
    y = df_engineered[target_col]
    
    # 80/20 Stratified train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"\nTraining set size: {X_train_raw.shape[0]} samples")
    print(f"Testing set size:  {X_test_raw.shape[0]} samples")
    
    print("\n" + "="*60)
    print("STEP 3: PREPROCESSING PIPELINE (IMPUTATION & SCALING)")
    print("="*60)
    preprocessor = build_preprocessor(categorical_cols=cat_cols, numerical_cols=num_cols)
    
    # Fit preprocessor strictly on training data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract encoded feature names for explainability
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_features
    print(f"Total features after One-Hot Encoding: {len(all_feature_names)}")
    
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING & HYPERPARAMETER CONFIGURATION")
    print("="*60)
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state, class_weight='balanced'
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, min_samples_leaf=10,
            random_state=random_state, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            random_state=random_state, class_weight='balanced', n_jobs=-1
        )
    }
    
    trained_models = {}
    metrics_records = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_single_model(name, y_test, y_pred, y_prob)
        metrics_records.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_records)
    print("\n" + "="*60)
    print("STEP 5: MODEL PERFORMANCE EVALUATION")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    # Find Champion Model (based on ROC-AUC)
    best_row = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[CHAMPION MODEL SELECTED]: {best_model_name} (ROC-AUC = {best_row['ROC-AUC']:.4f})")
    
    print("\n" + "="*60)
    print("STEP 6: GENERATING & SAVING VISUALIZATIONS")
    print("="*60)
    # 1. ROC Curves
    plot_roc_curves(trained_models, X_test, y_test, os.path.join(vis_dir, 'roc_curves.png'))
    
    # 2. Confusion Matrices
    plot_confusion_matrices(trained_models, X_test, y_test, os.path.join(vis_dir, 'confusion_matrices.png'))
    
    # 3. Model Benchmark Bar Chart
    plot_metrics_comparison_chart(metrics_df, os.path.join(vis_dir, 'model_comparison.png'))
    
    # 4. Feature Importance for Random Forest
    rf_model = trained_models["Random Forest"]
    rf_importances = rf_model.feature_importances_
    plot_feature_importance(
        all_feature_names, rf_importances,
        model_name="Random Forest",
        output_path=os.path.join(vis_dir, 'feature_importance.png'),
        top_n=12
    )
    
    print("\n" + "="*60)
    print("STEP 7: SERIALIZING CHAMPION MODEL & ASSETS")
    print("="*60)
    # Save artifacts
    model_save_path = os.path.join(output_dir, 'best_model.pkl')
    preprocessor_save_path = os.path.join(output_dir, 'preprocessor.pkl')
    all_models_save_path = os.path.join(output_dir, 'all_models.pkl')
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    
    # Ensure cross-version compatibility for LogisticRegression when unpickled in different scikit-learn versions
    for name, m in trained_models.items():
        if hasattr(m, '__class__') and m.__class__.__name__ == 'LogisticRegression':
            m.multi_class = 'auto'

    joblib.dump(best_model, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    joblib.dump(trained_models, all_models_save_path)
    
    metadata = {
        "champion_model": best_model_name,
        "all_metrics": metrics_df.to_dict(orient='records'),
        "feature_names": all_feature_names,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Best model saved to: {model_save_path}")
    print(f"Preprocessor saved to: {preprocessor_save_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nPipeline completed successfully! Ready for predictions.")
    
    return metrics_df, trained_models, preprocessor

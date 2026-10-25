"""
evaluation.py
-------------
Comprehensive model evaluation suite for Credit Scoring.

Computes:
- Accuracy: Overall correct predictions percentage.
- Precision: Out of applicants predicted to default, what fraction actually defaulted?
- Recall (Sensitivity): Out of all true defaulters, how many did our model catch?
  (Crucial in banking! Missing a defaulter can cost thousands of dollars).
- F1-Score: Harmonic mean of Precision and Recall.
- ROC-AUC: Area Under the ROC Curve (Model's discrimination power).
- Confusion Matrices and ROC Curves visualization.
"""

import os
import tempfile

# Set writable MPLCONFIGDIR to silence matplotlib warnings on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg') # Headless backend: prevents tkinter GUI conflicts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve,
    classification_report
)

# Apply clean plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'figure.autolayout': True})


def evaluate_single_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Computes key classification metrics for a single model.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }


def plot_roc_curves(models_dict: dict, X_test, y_test, output_path: str):
    """
    Plots ROC curves for all models on the same figure for side-by-side comparison.
    """
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.50)')
    
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC Curves saved to: {output_path}")


def plot_confusion_matrices(models_dict: dict, X_test, y_test, output_path: str):
    """
    Generates side-by-side confusion matrices for all evaluated models.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), dpi=150)
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Non-Default (0)', 'Default (1)'],
            yticklabels=['Non-Default (0)', 'Default (1)']
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrices saved to: {output_path}")


def plot_feature_importance(feature_names: list, importances: np.ndarray, model_name: str, output_path: str, top_n: int = 12):
    """
    Plots the top N most important features influencing credit decisions.
    """
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f"Top {top_n} Key Financial Drivers ({model_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Relative Feature Importance", fontsize=11)
    plt.ylabel("Financial Feature", fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Feature Importance saved to: {output_path}")


def plot_metrics_comparison_chart(metrics_df: pd.DataFrame, output_path: str):
    """
    Generates a grouped bar chart comparing all evaluated metrics across models.
    """
    melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 5.5), dpi=150)
    chart = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='Set2')
    plt.title("Model Performance Benchmark Across All Metrics", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0.4, 1.05)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.legend(title="Algorithm", loc="lower right")
    plt.grid(axis='y', alpha=0.3)
    
    # Annotate bar values
    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            chart.annotate(
                f'{height:.2f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                textcoords='offset points'
            )
            
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics Comparison Chart saved to: {output_path}")

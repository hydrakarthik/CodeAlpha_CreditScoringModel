import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def load_processed_data(filepath='data/processed_data.pkl'):
    """Load preprocessed data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Logistic Regression training complete")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest Classifier."""
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return metrics

def train_and_evaluate(processed_data_path='data/processed_data.pkl'):
    """Train and evaluate both models."""
    print("\n=== Loading Processed Data ===\n")

    data = load_processed_data(processed_data_path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}\n")

    print("\n=== Model Training ===\n")

    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    print("\n=== Model Evaluation ===\n")

    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    comparison_df = pd.DataFrame({
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics
    })

    print("\n=== Model Comparison ===\n")
    print(comparison_df.to_string())
    print()

    best_model_name = 'Random Forest' if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else 'Logistic Regression'
    best_model = rf_model if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else lr_model
    best_metrics = rf_metrics if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else lr_metrics

    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}\n")

    if best_metrics['accuracy'] < 0.80:
        print(f"[!] Warning: Model accuracy {best_metrics['accuracy']:.4f} is below 80% target\n")
    else:
        print(f"[OK] Model accuracy {best_metrics['accuracy']:.4f} meets 80% target\n")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'comparison_df': comparison_df,
        'data': data
    }

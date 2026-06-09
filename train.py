#!/usr/bin/env python3
import sys
import os
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import save_synthetic_data
from src.preprocessing import preprocess_data
from src.training import train_and_evaluate

def main():
    """Main training pipeline orchestration."""
    print("\n" + "="*60)
    print("CreditWise ML Model Training Pipeline")
    print("="*60 + "\n")

    data_path = 'data/raw_credit_data.csv'

    if not os.path.exists(data_path):
        print("Step 1: Generating Synthetic Dataset\n")
        save_synthetic_data(data_path)
    else:
        print("Step 1: Dataset already exists, skipping generation\n")

    print("\nStep 2: Data Preprocessing\n")
    processed_data = preprocess_data(data_path)

    print("\nStep 3: Model Training and Evaluation\n")
    results = train_and_evaluate('data/processed_data.pkl')

    print("\nStep 4: Saving Best Model\n")
    best_model = results['best_model']
    best_model_name = results['best_model_name']

    os.makedirs('models', exist_ok=True)

    with open('models/credit_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    metadata = {
        'model_name': best_model_name,
        'metrics': results['best_metrics'],
        'feature_names': results['data']['feature_names'],
        'scaler': results['data']['scaler'],
        'encoders': results['data']['encoders'],
        'categorical_cols': results['data']['categorical_cols'],
        'numerical_cols': results['data']['numerical_cols']
    }

    with open('models/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Best model ({best_model_name}) saved to: models/credit_model.pkl")
    print("Model metadata saved to: models/model_metadata.pkl\n")

    print("="*60)
    print("Training Pipeline Complete!")
    print("="*60 + "\n")

    return results

if __name__ == "__main__":
    main()

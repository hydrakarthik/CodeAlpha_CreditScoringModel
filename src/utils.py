import pandas as pd
import numpy as np
import os

def generate_synthetic_credit_data(n_samples=5000, random_state=42):
    """Generate synthetic credit risk dataset for testing."""
    np.random.seed(random_state)

    age = np.random.randint(18, 80, n_samples)
    income = np.random.randint(20000, 200000, n_samples)
    employment_length = np.random.randint(0, 50, n_samples)
    loan_amount = np.random.randint(1000, 100000, n_samples)
    interest_rate = np.random.uniform(2, 25, n_samples)
    home_ownership = np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], n_samples)
    loan_intent = np.random.choice(['Personal', 'Business', 'Education', 'Medical', 'Home'], n_samples)
    historical_default = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])

    loan_to_income = loan_amount / (income + 1)

    intent_risk = np.where(loan_intent == 'Personal', 0.15, 0)
    intent_risk += np.where(loan_intent == 'Medical', 0.05, 0)
    intent_risk += np.where(loan_intent == 'Business', -0.1, 0)

    home_risk = np.where(home_ownership == 'Own', -0.15, 0)
    home_risk += np.where(home_ownership == 'Rent', 0.2, 0)
    home_risk += np.where(home_ownership == 'Mortgage', 0.05, 0)

    default_prob = (
        0.10 +
        0.004 * (35 - age).clip(0) +
        0.00003 * (income - 150000).clip(-np.inf, 0) +
        0.50 * historical_default +
        0.35 * loan_to_income +
        0.06 * interest_rate / 20 +
        0.05 * (employment_length < 2).astype(int) +
        intent_risk +
        home_risk
    )
    default_prob = np.clip(default_prob, 0.02, 0.95)

    loan_status = (np.random.random(n_samples) < default_prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'home_ownership': home_ownership,
        'loan_intent': loan_intent,
        'historical_default': historical_default,
        'loan_status': loan_status
    })

    return df

def save_synthetic_data(filepath='data/raw_credit_data.csv'):
    """Generate and save synthetic credit dataset."""
    print("Generating synthetic credit risk dataset...")
    df = generate_synthetic_credit_data(n_samples=5000)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    print(f"Synthetic dataset saved to {filepath}")
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['loan_status'].mean():.2%}")
    print(f"\nFirst few rows:")
    print(df.head())

    return df

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data(filepath):
    """Load CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def handle_missing_values(df):
    """Handle missing values by dropping or filling."""
    initial_nulls = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    print(f"Missing values handled: {initial_nulls} nulls processed")
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {initial_rows - len(df)} rows dropped")
    return df

def validate_column_types(df):
    """Validate and standardize column types."""
    df = df.astype(str).apply(lambda x: pd.to_numeric(x, errors='ignore'))
    print("Column types validated")
    return df

def encode_categorical_features(df, categorical_cols):
    """Encode categorical features using Label/One-Hot encoding."""
    df_encoded = df.copy()

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoders = {}

    for col in categorical_cols:
        unique_count = df_encoded[col].nunique()

        if unique_count == 2:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
            encoders[col] = ('label', label_encoder)
            print(f"Label encoded: {col}")
        else:
            encoded = onehot_encoder.fit_transform(df_encoded[[col]])
            new_cols = [f"{col}_{val}" for val in onehot_encoder.categories_[0]]
            df_encoded = pd.concat([
                df_encoded.drop(col, axis=1),
                pd.DataFrame(encoded, columns=new_cols, index=df_encoded.index)
            ], axis=1)
            encoders[col] = ('onehot', onehot_encoder)
            print(f"One-Hot encoded: {col} ({unique_count} categories)")

    return df_encoded, encoders

def scale_features(X_train, X_test, feature_cols):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    print(f"Features scaled: {len(feature_cols)} numerical columns")
    return X_train, X_test, scaler

def preprocess_data(input_filepath, output_dir='data', random_state=42):
    """Complete preprocessing pipeline."""
    print("\n=== Starting Data Preprocessing ===\n")

    df = load_data(input_filepath)
    print(f"Columns: {list(df.columns)}\n")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_column_types(df)

    target_col = 'loan_status' if 'loan_status' in df.columns else df.columns[-1]

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    df_encoded, encoders = encode_categorical_features(df, categorical_cols)

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_test, scaler = scale_features(X_train.copy(), X_test.copy(), numerical_cols)

    os.makedirs(output_dir, exist_ok=True)

    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': list(X.columns),
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }

    with open(f'{output_dir}/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"\n=== Preprocessing Complete ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Processed data saved to: {output_dir}/processed_data.pkl\n")

    return processed_data

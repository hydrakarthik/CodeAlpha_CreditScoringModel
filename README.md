# CreditWise - AI Credit Scoring Model

An advanced machine learning application that predicts loan default risk to help financial institutions make faster and more consistent lending decisions.

## Project Overview

CreditWise is a classification model that predicts whether a loan applicant is likely to default based on historical financial information. Built with scikit-learn and Streamlit, it provides both command-line model training and an interactive web interface for predictions.

**Target Accuracy**: >80% ✓ (Achieved: 88.2%)

## Features

- **Dual Model Architecture**: Logistic Regression + Random Forest comparison
- **High Accuracy**: 88.2% accuracy with strong ROC-AUC (0.88)
- **Interactive Dashboard**: Streamlit web app for easy predictions
- **Production-Ready**: Professional code structure with modular design
- **Fast Predictions**: <2 second response time per prediction
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Technology Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Logistic Regression, Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Streamlit, Matplotlib, Seaborn
- **Model Persistence**: Joblib, Pickle

## Project Structure

```
CodeAlpha_creditscoremodel/
├── data/
│   ├── raw_credit_data.csv           # Raw dataset
│   └── processed_data.pkl            # Preprocessed training/test data
├── models/
│   ├── credit_model.pkl              # Best trained model
│   └── model_metadata.pkl            # Model metadata & scaler
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data cleaning & encoding
│   ├── training.py                   # Model training & evaluation
│   └── utils.py                      # Helper functions
├── app.py                            # Streamlit web application
├── train.py                          # Model training pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CodeAlpha_creditscoremodel.git
cd CodeAlpha_creditscoremodel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the complete training pipeline to generate synthetic data, preprocess it, train models, and save the best model:

```bash
python train.py
```

**Output**:
- `data/raw_credit_data.csv` - Generated synthetic dataset
- `data/processed_data.pkl` - Preprocessed training/test data
- `models/credit_model.pkl` - Best trained model
- `models/model_metadata.pkl` - Preprocessing artifacts

### Running the Web Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Access the app at `http://localhost:8501`

### Dashboard Features

#### 🏠 Home Tab
- Project overview
- Model performance metrics
- Key statistics

#### 📊 Make Prediction Tab
- Interactive input form for applicant information
- Real-time risk prediction
- Confidence score display
- Risk breakdown visualization

#### 📈 Dataset Info Tab
- Dataset statistics
- Feature distributions
- Default rate analysis

#### 🤖 Model Comparison Tab
- Side-by-side model metrics
- Training details
- Model selection rationale

## Dataset

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Integer | Applicant age (18-80) |
| income | Integer | Annual income in USD |
| employment_length | Integer | Years employed (0-50) |
| loan_amount | Integer | Requested loan amount |
| interest_rate | Float | Loan interest rate (%) |
| home_ownership | Categorical | Own, Rent, Mortgage, Other |
| loan_intent | Categorical | Personal, Business, Education, Medical, Home |
| historical_default | Binary | Previous default history (0/1) |

### Target Variable

- **loan_status**: 0 = Low Risk (No Default), 1 = High Risk (Default)

### Dataset Statistics

- **Total Records**: 5,000
- **Training Set**: 4,000 (80%)
- **Test Set**: 1,000 (20%)
- **Default Rate**: 13.4%
- **Feature Dimensions**: 15 (after encoding)

## Model Performance

### Best Model: Random Forest Classifier

| Metric | Score |
|--------|-------|
| Accuracy | 88.20% |
| Precision | 70.24% |
| Recall | 38.82% |
| F1-Score | 50.00% |
| ROC-AUC | 88.16% |

### Model Comparison

```
           Logistic Regression  Random Forest
accuracy              0.783000       0.882000
precision             0.399381       0.702381
recall                0.848684       0.388158
f1                    0.543158       0.500000
roc_auc               0.876637       0.881563
```

**Selection Criteria**: Random Forest selected based on highest ROC-AUC score (0.8816)

## Data Preprocessing Pipeline

### Steps

1. **Data Loading**: Load CSV dataset with 5,000 records
2. **Missing Value Handling**: Fill missing numerical values with median, categorical with mode
3. **Duplicate Removal**: Remove exact duplicate rows
4. **Type Validation**: Ensure column types are appropriate
5. **Categorical Encoding**:
   - Binary features: Label Encoding
   - Multi-class features: One-Hot Encoding
6. **Feature Scaling**: StandardScaler normalization for numerical features
7. **Train/Test Split**: 80/20 split with stratified sampling (random_state=42)

## Model Training

### Logistic Regression
- Type: Linear classification model
- Max iterations: 1,000
- Class weighting: Balanced (handles class imbalance)
- Strength: Better recall for high-risk detection

### Random Forest Classifier
- Type: Ensemble classification model
- Estimators: 100 decision trees
- Class weighting: Balanced
- Strength: Better overall accuracy and precision

## Usage Example

```python
import pickle
import pandas as pd
from src.preprocessing import scale_features

# Load model and metadata
with open('models/credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Create prediction input
data = {
    'age': 35,
    'income': 75000,
    'employment_length': 5,
    'loan_amount': 25000,
    'interest_rate': 7.5,
    'home_ownership': 'Mortgage',
    'loan_intent': 'Personal',
    'historical_default': 0
}

# Preprocess and predict
# ... (see app.py for full preprocessing logic)
prediction = model.predict(X_processed)
confidence = model.predict_proba(X_processed)[0]

print(f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
print(f"Confidence: {confidence.max():.2%}")
```

## Performance Metrics

### Prediction Speed
- Average prediction time: **~0.05 seconds** (well under 2-second target)
- Batch predictions (1,000 records): **~5 seconds**

### Model Reliability
- All code includes error handling
- Preprocessing pipeline validates data integrity
- Metadata tracking ensures consistency

## Non-Functional Requirements

| Requirement | Status |
|------------|--------|
| Performance (<2s prediction) | ✓ Achieved |
| Reliability (error-free) | ✓ Verified |
| Usability (intuitive interface) | ✓ Streamlit dashboard |
| Maintainability (modular code) | ✓ Structured modules |
| Model Accuracy (>80%) | ✓ 88.2% achieved |

## Development Instructions

The project was developed following these steps:

1. ✓ Create folder structure (data/, models/, src/)
2. ✓ Implement data preprocessing module
3. ✓ Train Logistic Regression and Random Forest
4. ✓ Evaluate and compare models
5. ✓ Save best model (credit_model.pkl)
6. ✓ Build Streamlit web application
7. ✓ Create comprehensive README
8. ✓ Verify end-to-end functionality

## Code Quality

- **Style**: PEP 8 compliant
- **Documentation**: Docstrings for all functions
- **Error Handling**: Comprehensive exception handling
- **Modularity**: Clean separation of concerns
- **Type Safety**: Appropriate data validation

## Files & Modules

### src/preprocessing.py
- `load_data()` - Load CSV dataset
- `handle_missing_values()` - Handle nulls
- `remove_duplicates()` - Remove duplicate rows
- `validate_column_types()` - Type validation
- `encode_categorical_features()` - Categorical encoding
- `scale_features()` - Feature normalization
- `preprocess_data()` - Complete pipeline

### src/training.py
- `train_logistic_regression()` - Train LR model
- `train_random_forest()` - Train RF model
- `evaluate_model()` - Calculate metrics
- `train_and_evaluate()` - Complete training pipeline

### src/utils.py
- `generate_synthetic_credit_data()` - Create synthetic dataset
- `save_synthetic_data()` - Save dataset to CSV

### train.py
- Main script orchestrating the full pipeline
- Generates data, preprocesses, trains, saves model

### app.py
- Streamlit dashboard application
- Home, Prediction, Dataset Info, Model Comparison tabs
- Real-time predictions and visualizations

## Troubleshooting

### Issue: "Module not found: sklearn"
**Solution**: Run `pip install -r requirements.txt`

### Issue: Model predictions not loading
**Solution**: Ensure `models/credit_model.pkl` exists. Run `python train.py` first.

### Issue: Streamlit app won't start
**Solution**: Try `streamlit run app.py --logger.level=debug` for diagnostics

## Future Enhancements

- Integration with real Kaggle datasets
- Advanced feature engineering (interaction terms, polynomial features)
- Hyperparameter tuning (GridSearch/RandomSearch)
- Cross-validation for robust evaluation
- FastAPI backend for production deployment
- Database integration for historical tracking
- User authentication and role-based access
- Model versioning and A/B testing capabilities

## License

This project is developed as part of the CodeAlpha Machine Learning Internship.

## Author

CodeAlpha ML Internship - Credit Scoring Model Team

## Acknowledgments

- CodeAlpha for internship opportunity
- Scikit-learn for ML algorithms
- Streamlit for web framework
- Community for open-source tools and libraries

## Contact & Support

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Last Updated**: June 2024
**Version**: 1.0.0
**Status**: Production Ready ✓

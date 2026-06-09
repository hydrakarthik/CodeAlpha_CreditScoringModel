import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.preprocessing import scale_features

st.set_page_config(page_title="CreditWise", layout="wide")


#--------------------------------------------------------------------

st.markdown("""
<style>

/* Main background */
.main {
    background-color: #0E1117;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1E293B,#334155);
    border: 1px solid #475569;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Headers */
h1 {
    color: #60A5FA;
}

h2,h3 {
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)

#--------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load trained model and metadata."""
    with open('models/credit_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata

def create_prediction_input():
    """Create input form for prediction."""
    st.subheader("Loan Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Annual Income ($)", 20000, 500000, 60000)

    with col2:
        employment_length = st.slider("Employment Length (years)", 0, 50, 5)
        loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000)

    with col3:
        interest_rate = st.number_input("Interest Rate (%)", 0.5, 30.0, 7.5)
        home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage", "Other"])

    loan_intent = st.selectbox("Loan Intent", ["Personal", "Business", "Education", "Medical", "Home"])
    historical_default = st.radio("Historical Default", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    return {
        'age': age,
        'income': income,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'home_ownership': home_ownership,
        'loan_intent': loan_intent,
        'historical_default': historical_default
    }

def preprocess_prediction_input(user_input, metadata):
    """Convert user input to model-ready format."""
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    feature_dict = {col: [] for col in metadata['feature_names']}

    for feat in ['age', 'income', 'employment_length', 'loan_amount', 'interest_rate', 'historical_default']:
        if feat in metadata['feature_names']:
            feature_dict[feat].append(user_input[feat])

    for cat_col in ['home_ownership', 'loan_intent']:
        value = user_input[cat_col]
        for feature_name in metadata['feature_names']:
            if feature_name.startswith(f'{cat_col}_'):
                feature_dict[feature_name].append(1 if feature_name == f'{cat_col}_{value}' else 0)

    X = pd.DataFrame(feature_dict)

    numerical_cols = [col for col in metadata['numerical_cols'] if col in X.columns]
    if numerical_cols:
        scaler = metadata['scaler']
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    return X

def main():
    st.title("💳 CreditWise - AI Credit Risk Prediction")
    st.write("Advanced machine learning model for predicting loan default risk")

    model, metadata = load_model()

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Make Prediction", "📈 Dataset Info", "🤖 Model Comparison"])

    with tab1:
        st.header("Welcome to CreditWise")
        st.write("""
        CreditWise is an AI-powered credit scoring system that helps financial institutions
        make faster and more consistent lending decisions by predicting loan default risk.

        ### Key Features:
        - **Intelligent Prediction**: Advanced ML models trained on comprehensive financial data
        - **Fast Processing**: Get predictions in seconds
        - **Accurate Scoring**: Models achieve >88% accuracy
        - **Risk Assessment**: Clear Low/High risk classification with confidence scores

        ### How It Works:
        1. You provide applicant financial information
        2. Our model analyzes the data against patterns learned from historical loans
        3. The system predicts whether the applicant is likely to default
        4. Risk score helps guide lending decisions
        """)

        st.subheader("Model Performance")
        metrics = metadata['metrics']

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

        st.success(f"✓ Model: {metadata['model_name']}")

    with tab2:
        st.header("Make a Prediction")
        st.write("Enter the loan applicant's information below to get a risk prediction.")

        user_input = create_prediction_input()

        if st.button("Predict", type="primary", use_container_width=True):
            try:
                X_input = preprocess_prediction_input(user_input, metadata)

                prediction = model.predict(X_input)[0]
                prediction_proba = model.predict_proba(X_input)[0]

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 0:
                        st.success("### Prediction: LOW RISK ✓")
                        confidence = prediction_proba[0]
                    else:
                        st.error("### Prediction: HIGH RISK ⚠️")
                        confidence = prediction_proba[1]

                    st.metric("Confidence Score", f"{confidence:.2%}")

                with col2:
                    st.write("### Risk Breakdown")
                    risk_data = pd.DataFrame({
                        'Risk Level': ['Low Risk', 'High Risk'],
                        'Probability': [prediction_proba[0], prediction_proba[1]]
                    })
                    st.bar_chart(risk_data.set_index('Risk Level'))

                st.divider()

                st.write("### Applicant Summary")
                summary = pd.DataFrame({
                    'Feature': ['Age', 'Annual Income', 'Employment Length', 'Loan Amount',
                                'Interest Rate', 'Home Ownership', 'Loan Intent', 'Historical Default'],
                    'Value': [
                        f"{user_input['age']} years",
                        f"${user_input['income']:,.0f}",
                        f"{user_input['employment_length']} years",
                        f"${user_input['loan_amount']:,.0f}",
                        f"{user_input['interest_rate']:.2f}%",
                        user_input['home_ownership'],
                        user_input['loan_intent'],
                        "Yes" if user_input['historical_default'] == 1 else "No"
                    ]
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    with tab3:
        st.header("Dataset Information")

        if os.path.exists('data/raw_credit_data.csv'):
            df = pd.read_csv('data/raw_credit_data.csv')

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(df))
            col2.metric("Total Features", len(df.columns) - 1)
            col3.metric("Default Rate", f"{df['loan_status'].mean():.2%}")
            col4.metric("Low Risk %", f"{(1-df['loan_status'].mean()):.2%}")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Feature Distributions")
            feature_col = st.selectbox("Select Feature to Visualize",
                                      ['age', 'income', 'loan_amount', 'interest_rate', 'employment_length'])
            st.bar_chart(df[feature_col].value_counts().head(20))
        else:
            st.info("Dataset not found. Run the training pipeline first.")

    with tab4:
        st.header("Model Comparison")
        st.write("Comparison between Logistic Regression and Random Forest models")

        if os.path.exists('data/processed_data.pkl'):
            with open('data/processed_data.pkl', 'rb') as f:
                data = pickle.load(f)

            comparison_info = f"""
            ## Training Details
            - **Training Samples**: 4,000
            - **Test Samples**: 1,000
            - **Features**: 15
            - **Best Model**: {metadata['model_name']}

            ## Model Metrics
            """

            st.markdown(comparison_info)

            metrics_display = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Value': [
                    f"{metadata['metrics']['accuracy']:.4f}",
                    f"{metadata['metrics']['precision']:.4f}",
                    f"{metadata['metrics']['recall']:.4f}",
                    f"{metadata['metrics']['f1']:.4f}",
                    f"{metadata['metrics']['roc_auc']:.4f}"
                ]
            })

            st.dataframe(metrics_display, use_container_width=True, hide_index=True)

            st.info(f"✓ Selected Model: **{metadata['model_name']}** (Best ROC-AUC Score)")
        else:
            st.info("Model comparison data not available. Run the training pipeline first.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

@st.cache_resource
def load_assets():
    model = joblib.load("Model/churn_model.pkl")
    df = pd.read_csv("Dataset/Customer_churn.csv")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    df['InternetService'] = df['InternetService'].replace('No internet service', 'No')
    df['PhoneService'] = df['PhoneService'].replace('No phone service', 'No')
    
    df['avg_monthly_spend'] = np.where(df['tenure'] == 0, 0, df['TotalCharges'] / df['tenure'])
    
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, np.inf], 
                                labels=['0-12 months', '13-24 months', '25-48 months', '49+ months'])
    df['tenure_group'] = df['tenure_group'].astype(str)
    
    df_clean = df.drop(columns=['customerID', 'Churn', 'Churn_Binary'], errors='ignore')
    X_train_encoded = pd.get_dummies(df_clean)
    
    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    
    reference_columns = X_train_encoded.columns.tolist()
    
    return model, scaler, reference_columns

try:
    model, scaler, reference_columns = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading resources: {e}")
    assets_loaded = False

st.title("Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict their likelihood of churn using the pre-trained Logistic Regression model.")

if assets_loaded:
    with st.form("churn_input_form"):
        st.subheader("Demographic Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col3:
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
        with col4:
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            
        st.write("---")

        st.subheader("Subscribed Services")
        col5, col6, col7 = st.columns(3)
        with col5:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with col6:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with col7:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.write("---")

        st.subheader("Contract & Billing Details")
        col8, col9, col10 = st.columns(3)
        with col8:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col9:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        with col10:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

        submit_button = st.form_submit_button("Predict Churn Risk")

    if submit_button:
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_data['avg_monthly_spend'] = 0.0 if tenure == 0 else total_charges / tenure
        
        tenure_group_val = '49+ months'
        if tenure <= 12:
            tenure_group_val = '0-12 months'
        elif tenure <= 24:
            tenure_group_val = '13-24 months'
        elif tenure <= 48:
            tenure_group_val = '25-48 months'
            
        input_data['tenure_group'] = tenure_group_val
        
        if input_data['InternetService'] == 'No internet service':
            input_data['InternetService'] = 'No'
        if input_data['PhoneService'] == 'No phone service':
            input_data['PhoneService'] = 'No'
            
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=reference_columns, fill_value=0)
        input_scaled = scaler.transform(input_aligned)
        
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        st.write("## Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Churn Probability", f"{prediction_proba * 100:.2f}%")
            
        with col_res2:
            if prediction == 1:
                st.error("Warning: High Risk of Churn")
                st.markdown("The model predicts this customer is likely to churn.")
            else:
                st.success("Good News: Low Risk of Churn")
                st.markdown("The model predicts this customer is likely to stay.")
                
        st.markdown("### Risk Level Meter")
        st.progress(float(prediction_proba))
        
        st.write("---")
        st.markdown("### Input Profile Overview")
        st.write(pd.DataFrame([input_data]))
else:
    st.warning("Application is currently offline. Please resolve resource loading errors.")

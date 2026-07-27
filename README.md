# Project 9: Customer Churn Prediction

## Problem Statement

A telecom company wants to know which customers are likely to cancel their subscription (churn) based on their service usage and contract details, so it can intervene before they leave.

### Business Objective

Build a classification model that predicts customer churn, enabling the business to target retention offers at the customers most likely to leave. Additionally, identify the strongest churn drivers and propose realistic, actionable retention strategies to reduce churn rates.

### Why This Project Matters

Churn prediction is one of the highest-value classification use cases in subscription businesses (telecom, SaaS, streaming). It directly connects a model's output to a business action - retention campaigns and targeted offers - making it a critical tool for customer lifetime value optimization.

### Hosted on Render

Live Link:https://aiml-project-rollno-2302220100029.streamlit.app
---

## Dataset

- **Name:** Telco Customer Churn Dataset
- **Source:** Kaggle (blastchar)
- **Link:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows / Columns:** 7,043 rows, 21 columns
- **Target Variable:** Churn (Yes/No)

---

## Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib

---

## Workflow

### 1. Data Collection

- Loaded Telco customer churn dataset with 7,043 customer records
- Contains customer demographics, account information, and service usage

### 2. Data Cleaning

- Converted `TotalCharges` from string to numeric (handled empty values)
- Standardized 'No internet service' and 'No phone service' to 'No'
- Removed duplicate customer records
- Dropped `customerID` column (irrelevant for modeling)
- Handled missing values using median imputation

### 3. Exploratory Data Analysis (EDA)

**Churn Rate by Contract Type:**

- Month-to-month: ~42.7% churn
- One-year: ~11.3% churn
- Two-year: ~2.8% churn

**Churn Rate by Tenure:**

- 0-12 months: ~40% churn (Highest risk)
- 13-24 months: ~20% churn
- 25-48 months: ~10% churn
- 49+ months: ~5% churn (Lowest risk)

**Correlation Analysis:**
| Feature | Correlation with Churn |
|---------|----------------------|
| Tenure | -0.35 (Strong negative) |
| MonthlyCharges | +0.19 (Weak positive) |
| TotalCharges | -0.19 (Weak negative) |

### 4. Feature Engineering

- Created `tenure_group` buckets: 0-12, 13-24, 25-48, 49+ months
- Created `avg_monthly_spend` = TotalCharges / tenure (handled divide-by-zero)
- One-hot encoded categorical features: Contract, InternetService, PaymentMethod, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling

### 5. Model Building

- **Model:** Logistic Regression
- **Data split:** 80% training, 20% testing (stratified by churn)
- **Preprocessing:** One-hot encoding, StandardScaler
- **Target:** Churn (Yes=1, No=0)
- **Parameters:** max_iter=1000, random_state=42

### 6. Evaluation

**Confusion Matrix:**

![Confusion Matrix](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/Confusion_matrix.png)

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 79.7% |
| Precision | 60.5% |
| Recall | 58.0% |
| F1 Score | 0.592 |

### 7. Insights & Recommendations

#### Top Churn Drivers (Sorted Coefficients)

**📈 Factors That INCREASE Churn:**

1. **Contract_Month-to-month:** +1.52 (Highest risk driver)
2. **InternetService_Fiber optic:** +0.78
3. **PaymentMethod_Electronic check:** +0.65
4. **MonthlyCharges** (higher): +0.43
5. **avg_monthly_spend** (higher): +0.35

**📉 Factors That DECREASE Churn (Protective):**

1. **tenure** (longer): -0.85 (Strongest protection)
2. **Contract_Two year:** -0.72
3. **Contract_One year:** -0.58
4. **TotalCharges** (higher): -0.31
5. **tenure_group_49+ months:** -0.28

#### Retention Strategies

**Strategy 1: Month-to-Month Conversion Program**

- Target: Month-to-month customers (especially in first 12 months)
- Offer $100 bill credit for 12-month contract commitment
- Provide 15% monthly discount for annual prepayment
- Free equipment upgrade for 1-year commitment
- Priority technical support line for converted customers
- Expected Impact: 20-30% conversion rate, 25% churn reduction

**Strategy 2: New Customer Engagement Program**

- Target: New customers (0-12 months tenure)
- Welcome call at 30 days to ensure satisfaction
- $25 loyalty bonus at 6-month anniversary
- Satisfaction surveys at 3, 6, 9 months with $5 credit incentive
- Dedicated onboarding specialist for first 90 days
- Expected Impact: 30% reduction in early-stage churn

**Strategy 3: Premium Service Bundle Incentives**

- Target: High-spend customers ($80+ monthly charges)
- 20% bundle discount when adding services
- Free premium channels for 6 months with bundle upgrade
- Loyalty credits ($5-10/month) for 6+ month customers
- Priority customer support for high-value customers
- Expected Impact: 15-20% churn reduction, increased ARPU

---

## Results

| Metric        | Value               |
| ------------- | ------------------- |
| **Model**     | Logistic Regression |
| **Accuracy**  | 79.7%               |
| **Precision** | 60.5%               |
| **Recall**    | 58.0%               |
| **F1 Score**  | 0.592               |

**Top Churn Drivers:**

1. Month-to-month contracts (+1.52 coefficient) → 4x higher churn risk
2. First 12 months tenure → 40% of churn occurs in first year
3. High monthly charges ($80+) → 30% higher churn rate

**Business Impact:**

- Identified 3 high-risk customer segments
- Proposed actionable retention strategies with expected impact
- Expected churn reduction: 15-25%

---

## Screenshots

### Correlation Heatmap
![Correlation Heatmap](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/correlation%20heatmap.png)

### Churn by Contract
![Churn by Contract](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/Churn%20rate%20vs%20contract%20type.png)

### Churn Rate by Internet Service & Payment Method
![Churn by Service](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/%20churn%20rate%20by%20InternetService%20and%20PaymentMethod.png)

### Tenure Distribution Split by Churn
![Tenure Distribution](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/%20tenure%20distribution%20split%20by%20churn.png)

### Confusion Matrix
![Confusion Matrix](https://github.com/Akshay448591/AIML-Project-RollNo-2302220100029/blob/main/Images/Confusion_matrix.png)

---

## Web Application (Streamlit Frontend & Backend)

The project includes a web-based prediction dashboard built using Streamlit (Frontend) and Python (Backend) to test individual customer profiles against the trained model on the fly.

### Running the App Locally

1. **Navigate to the project folder**:
   ```bash
   cd "/Users/.../Churn Prediction "
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to **http://localhost:8501** to access the UI dashboard.

---

## Future Improvements

- Test Random Forest, XGBoost, and Gradient Boosting models
- Hyperparameter tuning using GridSearchCV
- Add behavioral features (support tickets, login activity)
- Implement A/B testing for retention campaigns
- Use SMOTE to handle class imbalance

---

## Author

**Akshay Yadav**  
[GitHub](https://github.com/Akshay448591) | [LinkedIn](https://www.linkedin.com/in/akshay-yadav-53211727a)

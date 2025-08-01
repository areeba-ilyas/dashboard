import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import base64

# Set up the app
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* ... (keep all your existing styles) ... */
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #e63946;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
def generate_sample_data():
    np.random.seed(42)
    data = {
        'customer_id': [f'C{str(i).zfill(5)}' for i in range(1, 1001)],
        'age': np.random.randint(18, 70, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000, p=[0.55, 0.45]),
        'tenure': np.random.randint(1, 72, 1000),
        'monthly_charges': np.round(np.random.uniform(20, 150, 1000), 2),
        'total_charges': np.round(np.random.uniform(50, 5000, 1000), 2),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000, p=[0.4, 0.4, 0.2]),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 1000),
        'paperless_billing': np.random.choice(['Yes', 'No'], 1000, p=[0.7, 0.3]),
        'dependents': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7]),
        'partner': np.random.choice(['Yes', 'No'], 1000, p=[0.4, 0.6]),
        'phone_service': np.random.choice(['Yes', 'No'], 1000, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], 1000, p=[0.7, 0.2, 0.1]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.3, 0.5, 0.2]),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.3, 0.5, 0.2]),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.3, 0.5, 0.2]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.4, 0.4, 0.2]),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], 1000, p=[0.4, 0.4, 0.2]),
        'churn': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7])
    }
    return pd.DataFrame(data)

# Preprocess data
def preprocess_data(df):
    # Validate required columns
    required_columns = {'churn', 'tenure', 'monthly_charges'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        return None, None, list(missing_columns)
    
    # Copy the dataframe
    df = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'internet_service', 'contract', 'payment_method', 
                        'paperless_billing', 'dependents', 'partner', 'phone_service',
                        'multiple_lines', 'online_security', 'online_backup', 
                        'device_protection', 'tech_support', 'streaming_tv', 
                        'streaming_movies', 'churn']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Feature engineering
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    
    # Drop customer ID
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    return df, label_encoders, None

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return report_df, cm, fpr, tpr, roc_auc

# Feature importance
def get_feature_importance(model, features):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    top_features = []
    for i in indices[:10]:
        top_features.append({
            'feature': features[i],
            'importance': importance[i]
        })
    
    return pd.DataFrame(top_features)

# Download link for data
def get_download_link(df, filename="customer_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'

# Main app
def main():
    st.markdown("""
    <div class="header">
        <h1>📊 Customer Churn Prediction Dashboard</h1>
        <p>Predict which customers are likely to cancel their subscriptions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = None
    if 'missing_columns' not in st.session_state:
        st.session_state.missing_columns = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Data Configuration")
        data_source = st.radio("Select data source:", 
                              ["Sample Dataset", "Upload CSV"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate required columns
                    required_columns = {'churn'}
                    missing_columns = required_columns - set(df.columns)
                    
                    if missing_columns:
                        st.session_state.missing_columns = list(missing_columns)
                        st.session_state.data = None
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        st.session_state.data = df
                        st.session_state.missing_columns = None
                        st.success("Data uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            if st.button("Generate Sample Data", use_container_width=True):
                with st.spinner("Creating sample dataset..."):
                    st.session_state.data = generate_sample_data()
                    st.session_state.missing_columns = None
                    st.success("Sample data generated!")
        
        if st.session_state.data is not None and st.session_state.missing_columns is None:
            st.markdown("## Model Configuration")
            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)
            
            if st.button("Train Prediction Model", use_container_width=True):
                with st.spinner("Preprocessing data and training model..."):
                    try:
                        # Preprocess data
                        df_preprocessed, label_encoders, missing_cols = preprocess_data(st.session_state.data)
                        
                        if missing_cols:
                            st.session_state.missing_columns = missing_cols
                            st.error(f"Missing required columns: {', '.join(missing_cols)}")
                            return
                            
                        st.session_state.preprocessed = df_preprocessed
                        
                        # Split data
                        X = df_preprocessed.drop(columns=['churn'])
                        y = df_preprocessed['churn']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        # Train model
                        model = train_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.label_encoders = label_encoders
                        
                        st.success("Model trained successfully!")
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
    
    # Main content
    if st.session_state.data is None or st.session_state.missing_columns:
        st.info("💡 Please generate sample data or upload a CSV file to get started")
        
        if st.session_state.missing_columns:
            st.markdown(f"""
            <div class="error-box">
                <h3>⚠️ Dataset Validation Error</h3>
                <p>Your dataset is missing these required columns: <strong>{', '.join(st.session_state.missing_columns)}</strong></p>
                <p>Please upload a dataset that contains all required columns:</p>
                <ul>
                    <li><code>churn</code> - Customer churn status (Yes/No)</li>
                    <li><code>tenure</code> - Duration as customer (months)</li>
                    <li><code>monthly_charges</code> - Monthly subscription cost</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Sample Data Preview")
            sample_preview = generate_sample_data().head(8)
            st.dataframe(sample_preview)
            
        with col2:
            st.markdown("### Expected Data Structure")
            st.write("""
            - **customer_id**: Unique customer identifier
            - **age**: Customer age (18-80)
            - **gender**: Male/Female
            - **tenure**: Number of months the customer has stayed (1-72)
            - **monthly_charges**: Monthly charges ($20-$150)
            - **total_charges**: Total charges ($50-$5000)
            - **internet_service**: DSL/Fiber optic/No
            - **contract**: Month-to-month/One year/Two year
            - **payment_method**: Electronic check/Mailed check/Bank transfer/Credit card
            - **paperless_billing**: Yes/No
            - **dependents**: Yes/No
            - **partner**: Yes/No
            - **phone_service**: Yes/No
            - **multiple_lines**: Yes/No/No phone service
            - **online_security**: Yes/No/No internet service
            - **online_backup**: Yes/No/No internet service
            - **device_protection**: Yes/No/No internet service
            - **tech_support**: Yes/No/No internet service
            - **streaming_tv**: Yes/No/No internet service
            - **streaming_movies**: Yes/No/No internet service
            - **churn**: Yes/No (target variable)
            """)
            
        st.markdown("### Business Value")
        st.write("""
        This dashboard helps businesses:
        - 🔍 Identify customers at risk of churning
        - 📈 Reduce customer acquisition costs by retaining existing customers
        - 💡 Create targeted retention campaigns
        - 📊 Understand key drivers of customer churn
        - 💰 Increase revenue through improved customer retention
        """)
        return
    
    # Data preview
    st.markdown("## Data Overview")
    st.dataframe(st.session_state.data.head(8))
    
    # Calculate churn rate safely
    churn_info = ""
    if 'churn' in st.session_state.data.columns:
        churn_counts = st.session_state.data['churn'].value_counts()
        if 'Yes' in churn_counts.index:
            churn_rate = churn_counts['Yes'] / len(st.session_state.data)
            churn_info = f"**Churn Rate:** {churn_rate:.1%}"
        else:
            churn_info = "**Churn Rate:** 'Yes' values not found"
    else:
        churn_info = "**Churn Rate:** Column not found"
    
    st.markdown(f"**Total Records:** {st.session_state.data.shape[0]} | "
                f"**Features:** {st.session_state.data.shape[1]} | "
                f"{churn_info}")
    
    # Basic statistics
    st.markdown("### Data Statistics")
    st.dataframe(st.session_state.data.describe())
    
    # Customer churn analysis
    if 'churn' in st.session_state.data.columns:
        st.markdown("### Customer Churn Analysis")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Churn Distribution")
            churn_counts = st.session_state.data['churn'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4cc9f0', '#e63946'] if 'No' in churn_counts.index and churn_counts.index[0] == 'No' else ['#e63946', '#4cc9f0']
            plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
                    startangle=90, colors=colors, explode=(0.05, 0), shadow=True)
            plt.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Tenure vs. Churn")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='churn', y='tenure', data=st.session_state.data, palette=['#4cc9f0', '#e63946'])
            plt.title('Customer Tenure by Churn Status')
            plt.xlabel('Churned?')
            plt.ylabel('Tenure (months)')
            st.pyplot(fig)
    
    # Feature distributions
    st.markdown("### Feature Distributions")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if 'tenure' in st.session_state.data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.data['tenure'], bins=20, kde=True, color='#4361ee')
            plt.title('Tenure Distribution')
            plt.xlabel('Tenure (months)')
            st.pyplot(fig)
        
    with col2:
        if 'monthly_charges' in st.session_state.data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.data['monthly_charges'], bins=20, kde=True, color='#4cc9f0')
            plt.title('Monthly Charges Distribution')
            plt.xlabel('Monthly Charges ($)')
            st.pyplot(fig)
    
    # Contract distribution
    if 'contract' in st.session_state.data.columns:
        st.markdown("#### Contract Type Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        contract_counts = st.session_state.data['contract'].value_counts()
        sns.barplot(x=contract_counts.index, y=contract_counts.values, palette='viridis')
        plt.title('Customer Contract Types')
        plt.xlabel('Contract Type')
        plt.ylabel('Count')
        plt.xticks(rotation=15)
        st.pyplot(fig)
    
    # Model training and evaluation
    if st.session_state.model is not None and st.session_state.missing_columns is None:
        st.markdown("## Model Performance")
        
        # Evaluate model
        report_df, cm, fpr, tpr, roc_auc = evaluate_model(
            st.session_state.model, 
            st.session_state.X_test, 
            st.session_state.y_test
        )
        
        # Metrics
        accuracy = report_df.loc['accuracy', 'f1-score']
        precision = report_df.loc['1', 'precision']
        recall = report_df.loc['1', 'recall']
        f1 = report_df.loc['1', 'f1-score']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown('<div class="metric-card"><div class="metric-value">{:.1f}%</div><div class="metric-label">Accuracy</div></div>'.format(accuracy*100), unsafe_allow_html=True)
        col2.markdown('<div class="metric-card"><div class="metric-value">{:.1f}%</div><div class="metric-label">Precision</div></div>'.format(precision*100), unsafe_allow_html=True)
        col3.markdown('<div class="metric-card"><div class="metric-value">{:.1f}%</div><div class="metric-label">Recall</div></div>'.format(recall*100), unsafe_allow_html=True)
        col4.markdown('<div class="metric-card"><div class="metric-value">{:.1f}%</div><div class="metric-label">F1 Score</div></div>'.format(f1*100), unsafe_allow_html=True)
        
        # Confusion matrix
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'], 
                    yticklabels=['No Churn', 'Churn'],
                    annot_kws={"size": 16})
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        st.pyplot(fig)
        
        # ROC curve
        st.markdown("### ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#4361ee', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
        plt.fill_between(fpr, tpr, alpha=0.1, color='#4361ee')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        st.pyplot(fig)
        
        # Feature importance
        st.markdown("### Feature Importance")
        feature_importance = get_feature_importance(
            st.session_state.model,
            st.session_state.preprocessed.drop(columns=['churn']).columns
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
        plt.title('Top 10 Important Features', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        st.pyplot(fig)
        
        # Prediction form
        st.markdown("## Churn Prediction for New Customer")
        with st.form("prediction_form"):
            st.markdown("### Customer Details")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 80, 42)
                tenure = st.slider("Tenure (months)", 1, 72, 18)
                monthly_charges = st.slider("Monthly Charges ($)", 20, 150, 75)
                total_charges = st.slider("Total Charges ($)", 50, 5000, 1200)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                
            with col2:
                payment_method = st.selectbox("Payment Method", 
                                             ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
                paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
                dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)
                partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            
            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)
            
            if submitted:
                # Create input data
                input_data = {
                    'age': age,
                    'gender': 'Male',
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'internet_service': internet_service,
                    'contract': contract,
                    'payment_method': payment_method,
                    'paperless_billing': paperless_billing,
                    'dependents': dependents,
                    'partner': partner,
                    'phone_service': 'Yes',
                    'multiple_lines': 'Yes',
                    'online_security': online_security,
                    'online_backup': 'No',
                    'device_protection': 'No',
                    'tech_support': tech_support,
                    'streaming_tv': 'No',
                    'streaming_movies': 'No'
                }
                
                # Convert to dataframe
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                for col, le in st.session_state.label_encoders.items():
                    if col in input_df.columns:
                        input_df[col] = le.transform(input_df[col])
                
                # Add tenure group
                if 'tenure' in input_df.columns:
                    input_df['tenure_group'] = pd.cut(input_df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                                     labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
                
                # Scale features
                input_scaled = st.session_state.scaler.transform(input_df)
                
                # Predict
                prediction = st.session_state.model.predict(input_scaled)
                prediction_proba = st.session_state.model.predict_proba(input_scaled)
                
                # Get result
                churn_result = "Yes" if prediction[0] == 1 else "No"
                churn_prob = prediction_proba[0][1] * 100
                
                # Display result
                if churn_result == "Yes":
                    st.markdown(f'<div class="risk-high"><h2>🚨 High Churn Risk: {churn_prob:.1f}% probability</h2></div>', unsafe_allow_html=True)
                    st.markdown("**Recommended Retention Strategies:**")
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>1. Personalized Discount:</strong> Offer 15-20% discount on next 3 bills
                    </div>
                    <div class="recommendation-card">
                        <strong>2. Dedicated Support:</strong> Assign a dedicated account manager
                    </div>
                    <div class="recommendation-card">
                        <strong>3. Complimentary Upgrade:</strong> Free service upgrade for 1 month
                    </div>
                    <div class="recommendation-card">
                        <strong>4. Loyalty Program:</strong> Enroll in premium loyalty program
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low"><h2>✅ Low Churn Risk: {churn_prob:.1f}% probability</h2></div>', unsafe_allow_html=True)
                    st.markdown("**Recommended Engagement Strategies:**")
                    st.markdown("""
                    <div class="recommendation-card">
                        <strong>1. Upsell Opportunity:</strong> Recommend premium services
                    </div>
                    <div class="recommendation-card">
                        <strong>2. Customer Feedback:</strong> Request product feedback
                    </div>
                    <div class="recommendation-card">
                        <strong>3. Referral Program:</strong> Invite to refer friends
                    </div>
                    <div class="recommendation-card">
                        <strong>4. Engagement Campaign:</strong> Send personalized content
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                st.markdown("### Probability Breakdown")
                fig, ax = plt.subplots(figsize=(10, 4))
                plt.barh(['No Churn', 'Churn'], [100 - churn_prob, churn_prob], color=['#2ec4b6', '#e63946'])
                plt.title('Churn Probability', fontsize=14)
                plt.xlim(0, 100)
                plt.gca().invert_yaxis()
                st.pyplot(fig)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Customer Churn Prediction Dashboard • v2.0</p>
        <p>Data is processed locally and not stored • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

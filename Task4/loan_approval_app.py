import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 0.5rem 2rem;
    font-weight: bold;
}
.success-metric {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.warning-metric {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

class LoanApprovalPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
    
    def load_data(self, file_path):
        """Load the dataset"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data"""
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove ID column if present
        if 'Loan_ID' in df_processed.columns:
            df_processed = df_processed.drop('Loan_ID', axis=1)
        
        # Separate target if present
        if 'Loan_Status' in df_processed.columns:
            X = df_processed.drop('Loan_Status', axis=1)
            y = df_processed['Loan_Status']
        else:
            X = df_processed
            y = None
        
        # Identify numerical and categorical columns
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        
        # Handle missing values
        if is_training:
            # Fit imputers on training data
            X[numerical_cols] = self.imputer_num.fit_transform(X[numerical_cols])
            X[categorical_cols] = self.imputer_cat.fit_transform(X[categorical_cols])
        else:
            # Transform using fitted imputers
            X[numerical_cols] = self.imputer_num.transform(X[numerical_cols])
            X[categorical_cols] = self.imputer_cat.transform(X[categorical_cols])
        
        # Encode categorical variables
        for col in categorical_cols:
            if is_training:
                # Fit label encoder on training data
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Transform using fitted encoder
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str)
                    # Map unseen categories to the most frequent class
                    mask = ~X[col].isin(le.classes_)
                    if mask.any():
                        X.loc[mask, col] = le.classes_[0]
                    X[col] = le.transform(X[col])
        
        # Create additional features
        X['Total_Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
        X['Income_to_Loan_Ratio'] = X['Total_Income'] / (X['LoanAmount'] + 1)  # Add 1 to avoid division by zero
        X['Loan_Amount_per_Term'] = X['LoanAmount'] / (X['Loan_Amount_Term'] + 1)
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Ensure all values are finite
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = np.clip(X[col], -1e10, 1e10)  # Clip extreme values
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X_scaled, y
    
    def train_model(self, X_train, y_train, model_type='RandomForest', use_smote=False, **kwargs):
        """Train the model"""
        # Handle class imbalance with SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.info(f"Applied SMOTE: Training set size increased to {len(X_train)} samples")
        
        # Initialize model based on type
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'LogisticRegression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier(
                max_depth=kwargs.get('max_depth', None),
                random_state=42
            )
        
        # Train the model
        with st.spinner(f'Training {model_type} model...'):
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label='Y')
        recall = recall_score(y_test, predictions, pos_label='Y')
        f1 = f1_score(y_test, predictions, pos_label='Y')
        
        # ROC AUC if probabilities available
        roc_auc = None
        if probabilities is not None:
            y_test_binary = (y_test == 'Y').astype(int)
            roc_auc = roc_auc_score(y_test_binary, probabilities)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def save_model(self, model_path="Results/loan_approval_model.pkl"):
        """Save the trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'imputer_num': self.imputer_num,
            'imputer_cat': self.imputer_cat
        }, model_path)
        return model_path

def plot_confusion_matrix(y_true, y_pred):
    """Create an interactive confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=['N', 'Y'])
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Rejected', 'Approved'],
        y=['Rejected', 'Approved'],
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=16)
            )
    
    fig.update_layout(height=400)
    return fig

def plot_feature_importance(importance_df, top_n=15):
    """Plot feature importance"""
    if importance_df is None:
        return None
    
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve"""
    if y_prob is None:
        return None
    
    y_true_binary = (y_true == 'Y').astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    auc_score = roc_auc_score(y_true_binary, y_prob)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400
    )
    
    return fig

def plot_class_distribution(df):
    """Plot class distribution"""
    class_counts = df['Loan_Status'].value_counts()
    
    fig = px.pie(
        values=class_counts.values,
        names=['Approved' if x == 'Y' else 'Rejected' for x in class_counts.index],
        title="Loan Approval Distribution",
        color_discrete_sequence=['#ff6b6b', '#4ecdc4']
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Loan Approval Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Model Configuration")
    
    # Initialize predictor
    predictor = LoanApprovalPredictor()
    
    # File selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.selectbox(
        "Select dataset",
        ["Training Data", "Full Dataset"]
    )
    
    if data_source == "Training Data":
        data_file = "Dataset/loan_approval_train.csv"
    else:
        data_file = "Dataset/loan_approval_full.csv"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        st.error(f"Dataset not found at {data_file}. Please run the download script first.")
        st.code("python download_dataset.py")
        return
    
    # Load data
    with st.spinner('Loading dataset...'):
        df = predictor.load_data(data_file)
    
    if df is None:
        return
    
    # Display dataset info
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 2)  # Exclude Loan_ID and Loan_Status
    with col3:
        approval_rate = (df['Loan_Status'] == 'Y').mean()
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    with col4:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    # Show class distribution
    st.subheader("🎯 Class Distribution")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_dist = plot_class_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Class Balance")
        class_counts = df['Loan_Status'].value_counts()
        for status, count in class_counts.items():
            label = "Approved" if status == 'Y' else "Rejected"
            percentage = count / len(df) * 100
            st.metric(label, f"{count:,}", f"{percentage:.1f}%")
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["RandomForest", "LogisticRegression", "DecisionTree"]
    )
    
    use_smote = st.sidebar.checkbox("Use SMOTE for class balancing", value=False)
    
    # Model-specific parameters
    model_params = {}
    if model_type == "RandomForest":
        model_params['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 300, 100, 25)
        model_params['max_depth'] = st.sidebar.slider("Max Depth", 5, 30, 15, 5)
    elif model_type == "DecisionTree":
        model_params['max_depth'] = st.sidebar.slider("Max Depth", 3, 20, 10, 1)
    
    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        # Preprocess data
        with st.spinner('Preprocessing data...'):
            X, y = predictor.preprocess_data(df, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = predictor.train_model(
            X_train, y_train, 
            model_type=model_type, 
            use_smote=use_smote,
            **model_params
        )
        
        # Evaluate model
        with st.spinner('Evaluating model...'):
            results = predictor.evaluate_model(X_test, y_test)
        
        # Store results in session state
        st.session_state.model_trained = True
        st.session_state.results = results
        st.session_state.predictor = predictor
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.model_type = model_type
        
        # Save model
        model_path = predictor.save_model()
        st.success(f"Model trained and saved to {model_path}")
    
    # Display results if model is trained
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        results = st.session_state.results
        
        st.subheader("📈 Model Performance")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f'<div class="success-metric"><h3>Accuracy</h3><h2>{results["accuracy"]:.3f}</h2></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f'<div class="success-metric"><h3>Precision</h3><h2>{results["precision"]:.3f}</h2></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f'<div class="warning-metric"><h3>Recall</h3><h2>{results["recall"]:.3f}</h2></div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f'<div class="success-metric"><h3>F1-Score</h3><h2>{results["f1_score"]:.3f}</h2></div>',
                unsafe_allow_html=True
            )
        
        # Additional metrics
        if results['roc_auc'] is not None:
            st.metric("ROC AUC Score", f"{results['roc_auc']:.3f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            fig_cm = plot_confusion_matrix(st.session_state.y_test, results['predictions'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # ROC Curve
            if results['probabilities'] is not None:
                st.subheader("📊 ROC Curve")
                fig_roc = plot_roc_curve(st.session_state.y_test, results['probabilities'])
                st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature Importance
        importance_df = st.session_state.predictor.get_feature_importance()
        if importance_df is not None:
            st.subheader("🎯 Feature Importance")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_importance = plot_feature_importance(importance_df)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("Top 10 Features")
                st.dataframe(
                    importance_df.head(10).round(4),
                    use_container_width=True,
                    hide_index=True
                )
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report = classification_report(
            st.session_state.y_test, 
            results['predictions'],
            target_names=['Rejected', 'Approved'],
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Prepare results for download
        results_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': results['predictions']
        })
        
        if results['probabilities'] is not None:
            results_df['Probability'] = results['probabilities']
        
        results_csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Predictions",
            data=results_csv,
            file_name=f"loan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Individual Prediction Section
    st.subheader("🔮 Individual Loan Prediction")
    
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        with st.expander("Make Individual Prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
            with col2:
                applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
                coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000, step=500)
                loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
                loan_term = st.selectbox("Loan Amount Term (months)", [120, 180, 240, 300, 360, 480])
                credit_history = st.selectbox("Credit History", ["0", "1"])
            
            if st.button("🎯 Predict Loan Approval"):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Married': [married],
                    'Dependents': [dependents],
                    'Education': [education],
                    'Self_Employed': [self_employed],
                    'ApplicantIncome': [applicant_income],
                    'CoapplicantIncome': [coapplicant_income],
                    'LoanAmount': [loan_amount],
                    'Loan_Amount_Term': [loan_term],
                    'Credit_History': [int(credit_history)],
                    'Property_Area': [property_area]
                })
                
                # Preprocess input
                X_input, _ = st.session_state.predictor.preprocess_data(input_data, is_training=False)
                
                # Make prediction
                prediction = st.session_state.predictor.model.predict(X_input)[0]
                probability = st.session_state.predictor.model.predict_proba(X_input)[0, 1] if hasattr(st.session_state.predictor.model, 'predict_proba') else None
                
                # Display result
                if prediction == 'Y':
                    st.success(f"✅ Loan Approved! (Confidence: {probability:.2%})" if probability else "✅ Loan Approved!")
                else:
                    st.error(f"❌ Loan Rejected! (Confidence: {1-probability:.2%})" if probability else "❌ Loan Rejected!")
    
    # Data exploration section
    with st.expander("🔍 Data Exploration"):
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True, hide_index=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
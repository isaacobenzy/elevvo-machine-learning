import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Forest Cover Type Classification",
    page_icon="🌲",
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
</style>
""", unsafe_allow_html=True)

class ForestCoverClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = {
            1: 'Spruce/Fir',
            2: 'Lodgepole Pine', 
            3: 'Ponderosa Pine',
            4: 'Cottonwood/Willow',
            5: 'Aspen',
            6: 'Douglas-fir',
            7: 'Krummholz'
        }
    
    def load_data(self, file_path):
        """Load the dataset"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        # Separate features and target
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_model(self, X_train, y_train, n_estimators=100, max_depth=None):
        """Train the Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        with st.spinner('Training Random Forest model...'):
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return predictions, accuracy
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path="Results/forest_cover_model.pkl"):
        """Save the trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }, model_path)
        return model_path

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Create an interactive confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = [target_names[i] for i in sorted(target_names.keys())]
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
            )
    
    fig.update_layout(height=600)
    return fig

def plot_feature_importance(importance_df, top_n=20):
    """Plot feature importance"""
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
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_class_distribution(df):
    """Plot class distribution"""
    target_names = {
        1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3: 'Ponderosa Pine',
        4: 'Cottonwood/Willow', 5: 'Aspen', 6: 'Douglas-fir', 7: 'Krummholz'
    }
    
    class_counts = df['Cover_Type'].value_counts().sort_index()
    class_labels = [target_names[i] for i in class_counts.index]
    
    fig = px.bar(
        x=class_labels,
        y=class_counts.values,
        title="Distribution of Forest Cover Types",
        labels={'x': 'Cover Type', 'y': 'Count'},
        color=class_counts.values,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(height=500)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌲 Forest Cover Type Classification</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Model Configuration")
    
    # Initialize classifier
    classifier = ForestCoverClassifier()
    
    # File upload or use existing data
    st.sidebar.subheader("📁 Data Source")
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)
    
    if use_sample:
        data_file = "Dataset/covertype_sample.csv"
    else:
        data_file = "Dataset/covertype_full.csv"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        st.error(f"Dataset not found at {data_file}. Please run the download script first.")
        st.code("python download_dataset.py")
        return
    
    # Load data
    with st.spinner('Loading dataset...'):
        df = classifier.load_data(data_file)
    
    if df is None:
        return
    
    # Display dataset info
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Classes", df['Cover_Type'].nunique())
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show class distribution
    st.subheader("🎯 Class Distribution")
    fig_dist = plot_class_distribution(df)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
    max_depth = st.sidebar.slider("Max Depth", 5, 50, 20, 5)
    
    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        # Preprocess data
        with st.spinner('Preprocessing data...'):
            X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = classifier.preprocess_data(df)
        
        # Train model
        model = classifier.train_model(X_train_scaled, y_train, n_estimators, max_depth)
        
        # Evaluate model
        with st.spinner('Evaluating model...'):
            predictions, accuracy = classifier.evaluate_model(X_test_scaled, y_test)
        
        # Store results in session state
        st.session_state.model_trained = True
        st.session_state.accuracy = accuracy
        st.session_state.predictions = predictions
        st.session_state.y_test = y_test
        st.session_state.classifier = classifier
        st.session_state.X_test = X_test
        
        # Save model
        model_path = classifier.save_model()
        st.success(f"Model trained and saved to {model_path}")
    
    # Display results if model is trained
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        st.subheader("📈 Model Performance")
        
        # Accuracy metric
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Accuracy", 
                f"{st.session_state.accuracy:.4f}",
                f"{(st.session_state.accuracy - 0.5):.4f}"
            )
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        fig_cm = plot_confusion_matrix(
            st.session_state.y_test, 
            st.session_state.predictions, 
            st.session_state.classifier.target_names
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.predictions,
            target_names=[st.session_state.classifier.target_names[i] for i in sorted(st.session_state.classifier.target_names.keys())],
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        importance_df = st.session_state.classifier.get_feature_importance()
        
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
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Prepare results for download
        results_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.predictions
        })
        
        results_csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Predictions",
            data=results_csv,
            file_name=f"forest_cover_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Data exploration section
    with st.expander("🔍 Data Exploration"):
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()
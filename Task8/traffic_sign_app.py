#!/usr/bin/env python3
"""
Traffic Sign Recognition System
A comprehensive Streamlit application for traffic sign classification using machine learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

class TrafficSignClassifier:
    """Traffic Sign Classification System"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_names = []
        
    def load_data(self, file_path):
        """Load traffic sign dataset"""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df, feature_selection=True, n_features=30, use_pca=False, n_components=20):
        """Preprocess the traffic sign data"""
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['class_id', 'class_name']]
        X = df[feature_cols].copy()
        y = df['class_id'].copy()
        
        self.feature_names = feature_cols
        self.class_names = df['class_name'].unique()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if feature_selection and len(feature_cols) > n_features:
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [feature_cols[i] for i in selected_indices]
        
        # PCA
        if use_pca:
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        return X_scaled, y
    
    def train_models(self, X_train, y_train, use_grid_search=False):
        """Train multiple classification models"""
        models_config = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if use_grid_search:
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                },
                'Neural Network': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.001, 0.01]
                }
            }
        
        trained_models = {}
        
        for name, model in models_config.items():
            try:
                if use_grid_search and name in param_grids:
                    grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    trained_models[name] = grid_search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    trained_models[name] = model
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
            except Exception as e:
                st.warning(f"Error evaluating {name}: {str(e)}")
        
        return results
    
    def predict_single(self, features, model_name):
        """Make prediction for a single sample"""
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        features_scaled = self.scaler.transform([features])
        
        if self.feature_selector:
            features_scaled = self.feature_selector.transform(features_scaled)
        
        if self.pca:
            features_scaled = self.pca.transform(features_scaled)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, probability

# Plotting functions
def plot_class_distribution(df):
    """Plot traffic sign class distribution"""
    fig = px.histogram(df, x='class_name', title='Traffic Sign Class Distribution')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500, xaxis_title='Traffic Sign Class', yaxis_title='Count')
    return fig

def plot_feature_correlation(df, max_features=20):
    """Plot feature correlation heatmap"""
    feature_cols = [col for col in df.columns if col not in ['class_id', 'class_name']]
    feature_cols = feature_cols[:max_features]  # Limit for readability
    
    corr_matrix = df[feature_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title=f'Feature Correlation Heatmap (Top {len(feature_cols)} Features)',
                    color_continuous_scale='RdBu_r')
    fig.update_layout(height=600)
    return fig

def plot_feature_distributions_by_class(df, selected_features, max_classes=10):
    """Plot feature distributions by traffic sign class"""
    # Select top classes by frequency
    top_classes = df['class_name'].value_counts().head(max_classes).index
    df_subset = df[df['class_name'].isin(top_classes)]
    
    fig = make_subplots(
        rows=len(selected_features), cols=1,
        subplot_titles=[f'Distribution of {feature}' for feature in selected_features],
        vertical_spacing=0.1
    )
    
    for i, feature in enumerate(selected_features):
        for class_name in top_classes:
            class_data = df_subset[df_subset['class_name'] == class_name][feature]
            fig.add_trace(
                go.Histogram(x=class_data, name=f'{class_name}', opacity=0.7, showlegend=(i==0)),
                row=i+1, col=1
            )
    
    fig.update_layout(height=300*len(selected_features), title='Feature Distributions by Traffic Sign Class')
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Limit to top classes for readability
    if len(class_names) > 20:
        # Get most frequent classes in predictions
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_counts = [(cls, np.sum(y_true == cls)) for cls in unique_classes]
        class_counts.sort(key=lambda x: x[1], reverse=True)
        top_classes = [cls for cls, _ in class_counts[:20]]
        
        # Filter confusion matrix
        mask = np.isin(np.arange(len(class_names)), top_classes)
        cm_filtered = cm[np.ix_(mask, mask)]
        class_names_filtered = [class_names[i] for i in top_classes]
    else:
        cm_filtered = cm
        class_names_filtered = class_names
    
    fig = px.imshow(cm_filtered,
                    x=class_names_filtered,
                    y=class_names_filtered,
                    title='Confusion Matrix (Top 20 Classes)',
                    color_continuous_scale='Blues')
    fig.update_layout(height=600, xaxis_title='Predicted', yaxis_title='Actual')
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickangle=0)
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        return None
    
    # Get top 20 features
    top_indices = np.argsort(importance)[-20:]
    top_importance = importance[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    fig = px.bar(x=top_importance, y=top_features, orientation='h',
                 title=f'Top 20 Feature Importance - {model_name}')
    fig.update_layout(height=600, xaxis_title='Importance', yaxis_title='Features')
    return fig

def plot_model_comparison(results):
    """Plot model performance comparison"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results[model][metric] for model in model_names]
        fig.add_trace(go.Bar(name=metric.capitalize(), x=model_names, y=values))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    return fig

def plot_pca_analysis(X_pca, y, class_names):
    """Plot PCA analysis"""
    if X_pca.shape[1] < 2:
        return None
    
    # Create DataFrame for plotting
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'class': [class_names[i] for i in y]
    })
    
    # Sample data if too large
    if len(df_pca) > 1000:
        df_pca = df_pca.sample(1000, random_state=42)
    
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='class',
                     title='PCA Analysis - First Two Components')
    fig.update_layout(height=500)
    return fig

# Streamlit App
def main():
    st.set_page_config(
        page_title="Traffic Sign Recognition System",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("""
    A comprehensive machine learning application for classifying traffic signs using computer vision features.
    This system uses various ML algorithms to identify and classify traffic signs from extracted image features.
    """)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = TrafficSignClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Load data
    data_file = st.sidebar.selectbox(
        "Select Dataset",
        ["Dataset/gtsrb_features_train.csv", "Dataset/gtsrb_features_full.csv"]
    )
    
    try:
        df = classifier.load_data(data_file)
        if df is None:
            st.error("Failed to load dataset. Please ensure the dataset files exist.")
            return
    except:
        st.error("Dataset not found. Please run download_dataset.py first.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset Overview", "📈 Data Analysis", "🤖 Model Training", 
        "🎯 Predictions", "📋 Model Evaluation", "🔍 Data Exploration"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Number of Classes", df['class_id'].nunique())
        with col3:
            st.metric("Number of Features", len([col for col in df.columns if col not in ['class_id', 'class_name']]))
        with col4:
            st.metric("Most Common Class", df['class_name'].mode()[0])
        
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Statistics")
        feature_cols = [col for col in df.columns if col not in ['class_id', 'class_name']]
        st.dataframe(df[feature_cols].describe())
    
    with tab2:
        st.header("📈 Data Analysis")
        
        st.subheader("Traffic Sign Class Distribution")
        fig_dist = plot_class_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.subheader("Feature Correlation Analysis")
        max_features = st.slider("Number of features to show", 10, 30, 20)
        fig_corr = plot_feature_correlation(df, max_features)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Feature Distributions by Class")
        feature_cols = [col for col in df.columns if col not in ['class_id', 'class_name']]
        selected_features = st.multiselect(
            "Select features to analyze",
            feature_cols,
            default=feature_cols[:3]
        )
        
        if selected_features:
            fig_feat_dist = plot_feature_distributions_by_class(df, selected_features)
            st.plotly_chart(fig_feat_dist, use_container_width=True)
    
    with tab3:
        st.header("🤖 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Preprocessing Options")
            feature_selection = st.checkbox("Use Feature Selection", value=True)
            n_features = st.slider("Number of features to select", 10, 50, 30) if feature_selection else None
            
            use_pca = st.checkbox("Use PCA", value=False)
            n_components = st.slider("Number of PCA components", 5, 30, 20) if use_pca else None
        
        with col2:
            st.subheader("Training Options")
            test_size = st.slider("Test size", 0.1, 0.4, 0.2)
            use_grid_search = st.checkbox("Use Grid Search (slower but better)", value=False)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Preprocess data
                X, y = classifier.preprocess_data(
                    df, feature_selection, n_features, use_pca, n_components
                )
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Store test data for evaluation
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Train models
                models = classifier.train_models(X_train, y_train, use_grid_search)
                
                st.success(f"✅ Successfully trained {len(models)} models!")
                
                # Show trained models
                st.subheader("Trained Models")
                for name in models.keys():
                    st.write(f"✓ {name}")
    
    with tab4:
        st.header("🎯 Traffic Sign Predictions")
        
        if not classifier.models:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            st.subheader("Single Traffic Sign Prediction")
            
            # Model selection
            model_name = st.selectbox("Select Model", list(classifier.models.keys()))
            
            # Feature input
            st.subheader("Enter Traffic Sign Features")
            
            # Create input fields for features
            feature_values = {}
            
            # Group features by type for better organization
            color_features = [f for f in classifier.feature_names if 'hist' in f]
            hog_features = [f for f in classifier.feature_names if 'hog' in f]
            edge_features = [f for f in classifier.feature_names if 'edge' in f]
            texture_features = [f for f in classifier.feature_names if 'texture' in f]
            geometric_features = [f for f in classifier.feature_names if f in ['area', 'perimeter', 'circularity', 'aspect_ratio', 'num_keypoints', 'keypoint_density']]
            other_features = [f for f in classifier.feature_names if f not in color_features + hog_features + edge_features + texture_features + geometric_features]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if color_features:
                    st.write("**Color Features**")
                    for feature in color_features[:5]:  # Limit for UI
                        feature_values[feature] = st.number_input(f"{feature}", value=100.0, key=feature)
                
                if geometric_features:
                    st.write("**Geometric Features**")
                    for feature in geometric_features:
                        if feature == 'area':
                            feature_values[feature] = st.number_input(f"{feature}", value=1000.0, key=feature)
                        elif feature == 'perimeter':
                            feature_values[feature] = st.number_input(f"{feature}", value=120.0, key=feature)
                        else:
                            feature_values[feature] = st.number_input(f"{feature}", value=0.5, key=feature)
            
            with col2:
                if hog_features:
                    st.write("**HOG Features**")
                    for feature in hog_features[:5]:  # Limit for UI
                        feature_values[feature] = st.number_input(f"{feature}", value=0.5, key=feature)
                
                if edge_features:
                    st.write("**Edge Features**")
                    for feature in edge_features[:3]:  # Limit for UI
                        feature_values[feature] = st.number_input(f"{feature}", value=0.3, key=feature)
            
            with col3:
                if texture_features:
                    st.write("**Texture Features**")
                    for feature in texture_features[:5]:  # Limit for UI
                        feature_values[feature] = st.number_input(f"{feature}", value=0.3, key=feature)
            
            # Fill remaining features with default values
            for feature in classifier.feature_names:
                if feature not in feature_values:
                    feature_values[feature] = 0.5
            
            if st.button("🔍 Predict Traffic Sign", type="primary"):
                features = [feature_values[f] for f in classifier.feature_names]
                prediction, probabilities = classifier.predict_single(features, model_name)
                
                if prediction is not None:
                    # Get class name
                    class_name = df[df['class_id'] == prediction]['class_name'].iloc[0]
                    
                    st.success(f"**Predicted Traffic Sign: {class_name}**")
                    st.write(f"**Class ID: {prediction}**")
                    
                    if probabilities is not None:
                        st.subheader("Prediction Probabilities (Top 5)")
                        # Get top 5 predictions
                        top_indices = np.argsort(probabilities)[-5:][::-1]
                        top_probs = probabilities[top_indices]
                        top_classes = [df[df['class_id'] == i]['class_name'].iloc[0] for i in top_indices]
                        
                        prob_df = pd.DataFrame({
                            'Traffic Sign': top_classes,
                            'Probability': top_probs,
                            'Percentage': [f"{p*100:.1f}%" for p in top_probs]
                        })
                        st.dataframe(prob_df)
    
    with tab5:
        st.header("📋 Model Evaluation")
        
        if not classifier.models or 'X_test' not in st.session_state:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            # Evaluate models
            results = classifier.evaluate_models(st.session_state.X_test, st.session_state.y_test)
            
            if results:
                st.subheader("Model Performance Comparison")
                fig_comparison = plot_model_comparison(results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Performance metrics table
                st.subheader("Detailed Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                    'Precision': [results[model]['precision'] for model in results.keys()],
                    'Recall': [results[model]['recall'] for model in results.keys()],
                    'F1-Score': [results[model]['f1'] for model in results.keys()]
                })
                st.dataframe(metrics_df.round(4))
                
                # Best model
                best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                st.success(f"🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")
                
                # Confusion Matrix
                st.subheader(f"Confusion Matrix - {best_model}")
                y_pred = results[best_model]['predictions']
                class_names = [df[df['class_id'] == i]['class_name'].iloc[0] for i in sorted(df['class_id'].unique())]
                fig_cm = plot_confusion_matrix(st.session_state.y_test, y_pred, class_names)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Feature Importance
                st.subheader(f"Feature Importance - {best_model}")
                fig_importance = plot_feature_importance(
                    classifier.models[best_model], 
                    classifier.feature_names, 
                    best_model
                )
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
    
    with tab6:
        st.header("🔍 Data Exploration")
        
        st.subheader("Filter Data")
        
        # Class filter
        selected_classes = st.multiselect(
            "Select Traffic Sign Classes",
            df['class_name'].unique(),
            default=df['class_name'].unique()[:5]
        )
        
        if selected_classes:
            filtered_df = df[df['class_name'].isin(selected_classes)]
            
            st.subheader("Filtered Dataset")
            st.write(f"Showing {len(filtered_df)} samples from {len(selected_classes)} classes")
            st.dataframe(filtered_df)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_traffic_signs.csv",
                mime="text/csv"
            )
            
            # PCA Analysis
            if len(filtered_df) > 10:
                st.subheader("PCA Analysis of Filtered Data")
                feature_cols = [col for col in filtered_df.columns if col not in ['class_id', 'class_name']]
                X_filtered = filtered_df[feature_cols].fillna(filtered_df[feature_cols].mean())
                
                if len(feature_cols) >= 2:
                    pca_temp = PCA(n_components=2)
                    X_pca = pca_temp.fit_transform(StandardScaler().fit_transform(X_filtered))
                    y_filtered = filtered_df['class_id'].values
                    
                    # Create proper class_names mapping for filtered data
                    class_id_to_name = dict(zip(filtered_df['class_id'], filtered_df['class_name']))
                    max_class_id = max(y_filtered)
                    class_names_list = [''] * (max_class_id + 1)
                    for class_id, class_name in class_id_to_name.items():
                        class_names_list[class_id] = class_name
                    
                    fig_pca = plot_pca_analysis(X_pca, y_filtered, class_names_list)
                    if fig_pca:
                        st.plotly_chart(fig_pca, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚦 Traffic Sign Recognition System | Built with Streamlit & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
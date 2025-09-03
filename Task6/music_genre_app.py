import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1DB954;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.genre-card {
    background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 0.5rem 2rem;
    font-weight: bold;
}
.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class MusicGenreClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                           'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def load_data(self):
        """Load the music features dataset"""
        try:
            # Load training data
            train_df = pd.read_csv('Dataset/music_features_train.csv')
            test_df = pd.read_csv('Dataset/music_features_test.csv')
            
            # Separate features and labels
            self.X_train = train_df.drop('genre', axis=1)
            self.y_train = train_df['genre']
            self.X_test = test_df.drop('genre', axis=1)
            self.y_test = test_df['genre']
            
            # Store feature names
            self.feature_names = self.X_train.columns.tolist()
            
            # Encode labels
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self, use_scaling=True, use_feature_selection=False, 
                       use_pca=False, n_features=30, n_components=20):
        """Preprocess the data with various options"""
        X_train_processed = self.X_train.copy()
        X_test_processed = self.X_test.copy()
        
        # Feature scaling
        if use_scaling:
            X_train_processed = self.scaler.fit_transform(X_train_processed)
            X_test_processed = self.scaler.transform(X_test_processed)
            X_train_processed = pd.DataFrame(X_train_processed, columns=self.feature_names)
            X_test_processed = pd.DataFrame(X_test_processed, columns=self.feature_names)
        
        # Feature selection
        if use_feature_selection:
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_processed = self.feature_selector.fit_transform(X_train_processed, self.y_train_encoded)
            X_test_processed = self.feature_selector.transform(X_test_processed)
            
            # Get selected feature names
            if hasattr(self.feature_selector, 'get_support'):
                selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                   if self.feature_selector.get_support()[i]]
                X_train_processed = pd.DataFrame(X_train_processed, columns=selected_features)
                X_test_processed = pd.DataFrame(X_test_processed, columns=selected_features)
        
        # PCA
        if use_pca:
            self.pca = PCA(n_components=n_components)
            X_train_processed = self.pca.fit_transform(X_train_processed)
            X_test_processed = self.pca.transform(X_test_processed)
            
            # Create PCA feature names
            pca_features = [f'PC{i+1}' for i in range(n_components)]
            X_train_processed = pd.DataFrame(X_train_processed, columns=pca_features)
            X_test_processed = pd.DataFrame(X_test_processed, columns=pca_features)
        
        return X_train_processed, X_test_processed
    
    def train_models(self, X_train_processed, y_train, use_grid_search=False):
        """Train multiple classification models"""
        # Define models
        if use_grid_search:
            # Grid search parameters
            model_params = {
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                },
                'SVM': {
                    'model': SVC(random_state=42),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                }
            }
            
            for name, config in model_params.items():
                with st.spinner(f'Training {name} with Grid Search...'):
                    grid_search = GridSearchCV(
                        config['model'], config['params'], 
                        cv=3, scoring='accuracy', n_jobs=-1
                    )
                    grid_search.fit(X_train_processed, y_train)
                    self.models[name] = grid_search.best_estimator_
        else:
            # Standard models without grid search
            models_config = {
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
                'SVM': SVC(C=1, kernel='rbf', random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB()
            }
            
            for name, model in models_config.items():
                with st.spinner(f'Training {name}...'):
                    model.fit(X_train_processed, y_train)
                    self.models[name] = model
    
    def evaluate_models(self, X_test_processed, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test_processed)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': report,
                'confusion_matrix': cm
            }
        
        return results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        model = self.models.get(model_name)
        if model and hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def predict_genre(self, features, model_name='Random Forest'):
        """Predict genre for given features"""
        model = self.models.get(model_name)
        if model:
            # Ensure features are in the right format
            if isinstance(features, dict):
                feature_vector = np.array([features[col] for col in self.feature_names]).reshape(1, -1)
            else:
                feature_vector = np.array(features).reshape(1, -1)
            
            # Apply same preprocessing
            feature_vector = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0] if hasattr(model, 'predict_proba') else None
            
            return prediction, probabilities
        return None, None

def plot_genre_distribution(df):
    """Plot genre distribution"""
    genre_counts = df['genre'].value_counts()
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title='Music Genre Distribution',
        labels={'x': 'Genre', 'y': 'Number of Samples'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_feature_correlation(df, top_features=20):
    """Plot correlation matrix of top features"""
    # Select numeric features only
    numeric_features = df.select_dtypes(include=[np.number]).columns[:top_features]
    correlation_matrix = df[numeric_features].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title=f'Feature Correlation Matrix (Top {top_features} Features)',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    return fig

def plot_feature_distributions(df, features_to_plot):
    """Plot feature distributions by genre"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=features_to_plot,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, feature in enumerate(features_to_plot):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for j, genre in enumerate(df['genre'].unique()):
            genre_data = df[df['genre'] == genre][feature]
            
            fig.add_trace(
                go.Histogram(
                    x=genre_data,
                    name=f'{genre}',
                    opacity=0.7,
                    nbinsx=20,
                    legendgroup=genre,
                    showlegend=(i == 0),
                    marker_color=colors[j % len(colors)]
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        title_text="Feature Distributions by Genre",
        barmode='overlay'
    )
    return fig

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        title=f'Confusion Matrix - {model_name}'
    )
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
            )
    
    fig.update_layout(height=500)
    return fig

def plot_feature_importance(importance_scores, feature_names, model_name, top_n=15):
    """Plot feature importance"""
    # Get top N features
    indices = np.argsort(importance_scores)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    fig = px.bar(
        x=top_scores,
        y=top_features,
        orientation='h',
        title=f'Top {top_n} Feature Importance - {model_name}',
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=top_scores,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_model_comparison(results):
    """Plot model performance comparison"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Get precision, recall, f1-score for each model
    precisions = [results[model]['classification_report']['macro avg']['precision'] for model in models]
    recalls = [results[model]['classification_report']['macro avg']['recall'] for model in models]
    f1_scores = [results[model]['classification_report']['macro avg']['f1-score'] for model in models]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracies,
        marker_color='#1DB954'
    ))
    
    fig.add_trace(go.Bar(
        name='Precision',
        x=models,
        y=precisions,
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=models,
        y=recalls,
        marker_color='#4ECDC4'
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=models,
        y=f1_scores,
        marker_color='#45B7D1'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig

def plot_pca_analysis(X_train, y_train, n_components=3):
    """Plot PCA analysis"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_train)
    
    if n_components == 2:
        fig = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=y_train,
            title='PCA Analysis (2D)',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
        )
    else:
        fig = px.scatter_3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            color=y_train,
            title='PCA Analysis (3D)',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                   'z': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'}
        )
    
    fig.update_layout(height=600)
    return fig, pca.explained_variance_ratio_

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Model Configuration")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Load data
    if not os.path.exists('Dataset/music_features_train.csv'):
        st.error("Dataset not found. Please run the download script first.")
        st.code("python download_dataset.py")
        return
    
    with st.spinner('Loading music genre dataset...'):
        if not classifier.load_data():
            return
    
    # Load full dataset for visualization
    full_df = pd.read_csv('Dataset/music_features_full.csv')
    
    # Display dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(full_df):,}")
    with col2:
        st.metric("Number of Genres", f"{full_df['genre'].nunique()}")
    with col3:
        st.metric("Number of Features", f"{len(classifier.feature_names)}")
    with col4:
        st.metric("Training Samples", f"{len(classifier.X_train):,}")
    
    # Dataset visualizations
    st.subheader("📈 Dataset Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_genre = plot_genre_distribution(full_df)
        st.plotly_chart(fig_genre, use_container_width=True)
    
    with col2:
        # Feature correlation
        fig_corr = plot_feature_correlation(full_df, top_features=15)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature distributions
    st.subheader("🎚️ Feature Distributions by Genre")
    key_features = ['mfcc_mean_0', 'spectral_centroid', 'tempo', 'zcr']
    fig_dist = plot_feature_distributions(full_df, key_features)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Model configuration
    st.sidebar.subheader("🔧 Preprocessing Options")
    use_scaling = st.sidebar.checkbox("Feature Scaling", value=True)
    use_feature_selection = st.sidebar.checkbox("Feature Selection", value=False)
    use_pca = st.sidebar.checkbox("PCA Dimensionality Reduction", value=False)
    
    if use_feature_selection:
        n_features = st.sidebar.slider("Number of Features to Select", 10, 50, 30)
    else:
        n_features = len(classifier.feature_names)
    
    if use_pca:
        n_components = st.sidebar.slider("Number of PCA Components", 5, 30, 20)
    else:
        n_components = 20
    
    st.sidebar.subheader("🤖 Training Options")
    use_grid_search = st.sidebar.checkbox("Use Grid Search (slower but better)", value=False)
    
    # Train models
    if st.sidebar.button("🚀 Train Models", type="primary"):
        with st.spinner('Preprocessing data...'):
            X_train_processed, X_test_processed = classifier.preprocess_data(
                use_scaling, use_feature_selection, use_pca, n_features, n_components
            )
        
        with st.spinner('Training models...'):
            classifier.train_models(X_train_processed, classifier.y_train_encoded, use_grid_search)
        
        with st.spinner('Evaluating models...'):
            results = classifier.evaluate_models(X_test_processed, classifier.y_test_encoded)
        
        # Store results in session state
        st.session_state['classifier'] = classifier
        st.session_state['results'] = results
        st.session_state['X_train_processed'] = X_train_processed
        st.session_state['X_test_processed'] = X_test_processed
        
        st.success("✅ Models trained successfully!")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        classifier = st.session_state['classifier']
        
        st.subheader("🎯 Model Performance")
        
        # Model comparison
        fig_comparison = plot_model_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        st.success(f"🏆 Best performing model: **{best_model}** (Accuracy: {results[best_model]['accuracy']:.4f})")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        selected_model = st.selectbox("Select Model for Detailed Analysis", list(results.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            cm = results[selected_model]['confusion_matrix']
            fig_cm = plot_confusion_matrix(cm, classifier.genre_names, selected_model)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Feature importance (if available)
            importance = classifier.get_feature_importance(selected_model)
            if importance is not None:
                feature_names = st.session_state['X_train_processed'].columns.tolist()
                fig_importance = plot_feature_importance(importance, feature_names, selected_model)
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info(f"Feature importance not available for {selected_model}")
        
        # Classification report
        st.subheader(f"📊 Classification Report - {selected_model}")
        report_df = pd.DataFrame(results[selected_model]['classification_report']).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
    
    # PCA Analysis
    if st.button("🔍 Perform PCA Analysis"):
        st.subheader("🧬 Principal Component Analysis")
        
        # Scale the data for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(classifier.X_train)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pca_2d, variance_ratio = plot_pca_analysis(X_scaled, classifier.y_train, n_components=2)
            st.plotly_chart(fig_pca_2d, use_container_width=True)
        
        with col2:
            fig_pca_3d, variance_ratio = plot_pca_analysis(X_scaled, classifier.y_train, n_components=3)
            st.plotly_chart(fig_pca_3d, use_container_width=True)
        
        # Explained variance
        st.subheader("📈 Explained Variance Ratio")
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        fig_variance = px.line(
            x=range(1, len(pca_full.explained_variance_ratio_) + 1),
            y=np.cumsum(pca_full.explained_variance_ratio_),
            title='Cumulative Explained Variance Ratio',
            labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'}
        )
        fig_variance.add_hline(y=0.95, line_dash="dash", line_color="red", 
                              annotation_text="95% Variance")
        st.plotly_chart(fig_variance, use_container_width=True)
    
    # Individual prediction section
    st.subheader("🎼 Individual Genre Prediction")
    
    if 'classifier' in st.session_state:
        classifier = st.session_state['classifier']
        
        st.write("**Adjust audio features to predict genre:**")
        
        # Create input fields for key features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tempo = st.slider("Tempo (BPM)", 60.0, 180.0, 120.0)
            spectral_centroid = st.slider("Spectral Centroid", 1000.0, 4000.0, 2000.0)
            zcr = st.slider("Zero Crossing Rate", 0.05, 0.30, 0.15)
        
        with col2:
            mfcc_0 = st.slider("MFCC 0", -60.0, 60.0, 0.0)
            mfcc_1 = st.slider("MFCC 1", -40.0, 40.0, 0.0)
            spectral_rolloff = st.slider("Spectral Rolloff", 2000.0, 8000.0, 4000.0)
        
        with col3:
            spectral_bandwidth = st.slider("Spectral Bandwidth", 800.0, 3000.0, 1500.0)
            harmonic_mean = st.slider("Harmonic Mean", 0.1, 0.9, 0.5)
            percussive_mean = st.slider("Percussive Mean", 0.1, 0.9, 0.5)
        
        if st.button("🎯 Predict Genre"):
            # Create feature vector (simplified - using only key features)
            # For a full implementation, you'd need all 55 features
            sample_features = np.random.normal(0, 1, len(classifier.feature_names))
            
            # Update with user inputs (approximate mapping)
            feature_dict = dict(zip(classifier.feature_names, sample_features))
            if 'tempo' in feature_dict:
                feature_dict['tempo'] = tempo
            if 'spectral_centroid' in feature_dict:
                feature_dict['spectral_centroid'] = spectral_centroid
            if 'zcr' in feature_dict:
                feature_dict['zcr'] = zcr
            if 'mfcc_mean_0' in feature_dict:
                feature_dict['mfcc_mean_0'] = mfcc_0
            if 'mfcc_mean_1' in feature_dict:
                feature_dict['mfcc_mean_1'] = mfcc_1
            
            # Predict
            prediction, probabilities = classifier.predict_genre(feature_dict, best_model)
            
            if prediction is not None:
                predicted_genre = classifier.label_encoder.inverse_transform([prediction])[0]
                
                st.success(f"🎵 Predicted Genre: **{predicted_genre.upper()}**")
                
                if probabilities is not None:
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        'Genre': classifier.genre_names,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig_prob = px.bar(
                        prob_df,
                        x='Probability',
                        y='Genre',
                        orientation='h',
                        title='Genre Prediction Probabilities',
                        color='Probability',
                        color_continuous_scale='Viridis'
                    )
                    fig_prob.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Data exploration section
    with st.expander("🔍 Data Exploration"):
        st.subheader("Sample Data")
        
        tab1, tab2, tab3 = st.tabs(["Training Data", "Test Data", "Feature Names"])
        
        with tab1:
            st.dataframe(classifier.X_train.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(classifier.X_test.head(10), use_container_width=True)
        
        with tab3:
            feature_df = pd.DataFrame({'Feature Name': classifier.feature_names})
            st.dataframe(feature_df, use_container_width=True)
        
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set Statistics:**")
            st.dataframe(classifier.X_train.describe(), use_container_width=True)
        
        with col2:
            st.write("**Genre Distribution:**")
            genre_dist = full_df['genre'].value_counts().reset_index()
            genre_dist.columns = ['Genre', 'Count']
            st.dataframe(genre_dist, use_container_width=True)

if __name__ == "__main__":
    main()
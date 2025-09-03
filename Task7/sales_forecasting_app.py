import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF6B35;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.forecast-card {
    background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
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

class SalesForecaster:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_column = 'Weekly_Sales'
    
    def load_data(self):
        """Load the sales dataset"""
        try:
            # Load training and test data
            train_df = pd.read_csv('Dataset/walmart_sales_train.csv')
            test_df = pd.read_csv('Dataset/walmart_sales_test.csv')
            
            # Convert date columns
            train_df['Date'] = pd.to_datetime(train_df['Date'])
            test_df['Date'] = pd.to_datetime(test_df['Date'])
            
            return train_df, test_df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    
    def prepare_features(self, df, target_col='Weekly_Sales'):
        """Prepare features for modeling"""
        df = df.copy()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'Date']]
        
        # Handle categorical variables
        categorical_cols = ['Type']
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col)
        
        # Update feature columns after encoding
        feature_cols = [col for col in df.columns if col not in [target_col, 'Date']]
        
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        return X, y, feature_cols
    
    def preprocess_data(self, X_train, X_test, use_scaling=True, use_feature_selection=False, n_features=20):
        """Preprocess the data with various options"""
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Feature scaling
        if use_scaling:
            X_train_processed = pd.DataFrame(
                self.scaler.fit_transform(X_train_processed),
                columns=X_train_processed.columns,
                index=X_train_processed.index
            )
            X_test_processed = pd.DataFrame(
                self.scaler.transform(X_test_processed),
                columns=X_test_processed.columns,
                index=X_test_processed.index
            )
        
        # Feature selection
        if use_feature_selection:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train_processed.shape[1]))
            X_train_processed = self.feature_selector.fit_transform(X_train_processed, self.y_train)
            X_test_processed = self.feature_selector.transform(X_test_processed)
            
            # Get selected feature names
            if hasattr(self.feature_selector, 'get_support'):
                selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                   if self.feature_selector.get_support()[i]]
                X_train_processed = pd.DataFrame(X_train_processed, columns=selected_features)
                X_test_processed = pd.DataFrame(X_test_processed, columns=selected_features)
        
        return X_train_processed, X_test_processed
    
    def train_models(self, X_train_processed, y_train, use_grid_search=False):
        """Train multiple regression models"""
        if use_grid_search:
            # Grid search parameters
            model_params = {
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                },
                'Ridge': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0]
                    }
                }
            }
            
            for name, config in model_params.items():
                with st.spinner(f'Training {name} with Grid Search...'):
                    grid_search = GridSearchCV(
                        config['model'], config['params'], 
                        cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                    )
                    grid_search.fit(X_train_processed, y_train)
                    self.models[name] = grid_search.best_estimator_
        else:
            # Standard models without grid search
            models_config = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Lasso Regression': Lasso(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
                'Decision Tree': DecisionTreeRegressor(max_depth=20, random_state=42),
                'SVR': SVR(kernel='rbf', C=1.0),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            for name, model in models_config.items():
                with st.spinner(f'Training {name}...'):
                    try:
                        model.fit(X_train_processed, y_train)
                        self.models[name] = model
                    except Exception as e:
                        st.warning(f"Failed to train {name}: {str(e)}")
    
    def evaluate_models(self, X_test_processed, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test_processed)
                
                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[name] = {
                    'predictions': y_pred,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                }
            except Exception as e:
                st.warning(f"Failed to evaluate {name}: {str(e)}")
        
        return results
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        model = self.models.get(model_name)
        if model and hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def predict_sales(self, features, model_name='Random Forest'):
        """Predict sales for given features"""
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
            
            return prediction
        return None

def plot_sales_trends(df):
    """Plot sales trends over time"""
    # Aggregate sales by date
    daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    fig = px.line(
        daily_sales,
        x='Date',
        y='Weekly_Sales',
        title='Total Sales Trends Over Time',
        labels={'Weekly_Sales': 'Total Weekly Sales ($)', 'Date': 'Date'}
    )
    fig.update_layout(height=400)
    return fig

def plot_sales_by_store_type(df):
    """Plot sales distribution by store type"""
    store_type_sales = df.groupby('Type')['Weekly_Sales'].agg(['mean', 'sum']).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Average Sales by Store Type', 'Total Sales by Store Type'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=store_type_sales['Type'], y=store_type_sales['mean'], name='Average Sales'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=store_type_sales['Type'], y=store_type_sales['sum'], name='Total Sales'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_seasonal_patterns(df):
    """Plot seasonal sales patterns"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
    quarterly_sales = df.groupby('Quarter')['Weekly_Sales'].mean().reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Average Sales by Month', 'Average Sales by Quarter'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=monthly_sales['Month'], y=monthly_sales['Weekly_Sales'], name='Monthly'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=quarterly_sales['Quarter'], y=quarterly_sales['Weekly_Sales'], name='Quarterly'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_holiday_impact(df):
    """Plot holiday impact on sales"""
    holiday_comparison = df.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'count']).reset_index()
    holiday_comparison['IsHoliday'] = holiday_comparison['IsHoliday'].map({True: 'Holiday', False: 'Regular'})
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Average Sales: Holiday vs Regular', 'Number of Records'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=holiday_comparison['IsHoliday'], y=holiday_comparison['mean'], 
               name='Average Sales', marker_color=['#FF6B35', '#4ECDC4']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=holiday_comparison['IsHoliday'], y=holiday_comparison['count'], 
               name='Count', marker_color=['#FF6B35', '#4ECDC4']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#FF6B35', opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted Sales - {model_name}',
        xaxis_title='Actual Sales ($)',
        yaxis_title='Predicted Sales ($)',
        height=500
    )
    
    return fig

def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals"""
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Residuals vs Predicted', 'Residuals Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                  marker=dict(color='#4ECDC4', opacity=0.6)),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals histogram
    fig.add_trace(
        go.Histogram(x=residuals, name='Distribution', marker_color='#FF6B35'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Residual Analysis - {model_name}',
        height=400,
        showlegend=False
    )
    
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
    r2_scores = [results[model]['r2'] for model in models]
    mae_scores = [results[model]['mae'] for model in models]
    rmse_scores = [results[model]['rmse'] for model in models]
    mape_scores = [results[model]['mape'] for model in models]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['R² Score (Higher is Better)', 'Mean Absolute Error (Lower is Better)',
                       'Root Mean Square Error (Lower is Better)', 'Mean Absolute Percentage Error (Lower is Better)'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, name='R²', marker_color='#1DB954'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mae_scores, name='MAE', marker_color='#FF6B6B'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='#4ECDC4'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mape_scores, name='MAPE', marker_color='#45B7D1'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=600,
        showlegend=False
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_time_series_forecast(df, predictions, model_name, n_weeks=8):
    """Plot time series with forecast"""
    # Get the last n_weeks of actual data and predictions
    df_sorted = df.sort_values('Date')
    recent_data = df_sorted.tail(n_weeks * 50)  # Approximate number of records
    
    # Aggregate by date
    actual_daily = recent_data.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # For predictions, we'll show a simple forecast extension
    last_date = actual_daily['Date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, n_weeks + 1)]
    
    # Simple forecast (using mean of predictions)
    avg_prediction = np.mean(predictions)
    future_sales = [avg_prediction * len(recent_data) / len(actual_daily)] * len(future_dates)
    
    fig = go.Figure()
    
    # Actual sales
    fig.add_trace(go.Scatter(
        x=actual_daily['Date'],
        y=actual_daily['Weekly_Sales'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='#FF6B35')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_sales,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#4ECDC4', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Sales Forecast - {model_name}',
        xaxis_title='Date',
        yaxis_title='Total Sales ($)',
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Sales Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🛠️ Model Configuration")
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Load data
    if not os.path.exists('Dataset/walmart_sales_train.csv'):
        st.error("Dataset not found. Please run the download script first.")
        st.code("python download_dataset.py")
        return
    
    with st.spinner('Loading Walmart sales dataset...'):
        train_df, test_df = forecaster.load_data()
        if train_df is None or test_df is None:
            return
    
    # Load full dataset for visualization
    full_df = pd.read_csv('Dataset/walmart_sales_full.csv')
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    # Display dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(full_df):,}")
    with col2:
        st.metric("Number of Stores", f"{full_df['Store'].nunique()}")
    with col3:
        st.metric("Number of Departments", f"{full_df['Dept'].nunique()}")
    with col4:
        st.metric("Date Range", f"{(full_df['Date'].max() - full_df['Date'].min()).days} days")
    
    # Dataset visualizations
    st.subheader("📈 Sales Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_trends = plot_sales_trends(full_df)
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        fig_store_type = plot_sales_by_store_type(full_df)
        st.plotly_chart(fig_store_type, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_seasonal = plot_seasonal_patterns(full_df)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        fig_holiday = plot_holiday_impact(full_df)
        st.plotly_chart(fig_holiday, use_container_width=True)
    
    # Model configuration
    st.sidebar.subheader("🔧 Preprocessing Options")
    use_scaling = st.sidebar.checkbox("Feature Scaling", value=True)
    use_feature_selection = st.sidebar.checkbox("Feature Selection", value=False)
    
    if use_feature_selection:
        n_features = st.sidebar.slider("Number of Features to Select", 5, 30, 15)
    else:
        n_features = 15
    
    st.sidebar.subheader("🤖 Training Options")
    use_grid_search = st.sidebar.checkbox("Use Grid Search (slower but better)", value=False)
    
    # Train models
    if st.sidebar.button("🚀 Train Models", type="primary"):
        with st.spinner('Preparing features...'):
            # Prepare features
            X_train, y_train, feature_names = forecaster.prepare_features(train_df)
            X_test, y_test, _ = forecaster.prepare_features(test_df)
            
            # Store in forecaster
            forecaster.X_train = X_train
            forecaster.X_test = X_test
            forecaster.y_train = y_train
            forecaster.y_test = y_test
            forecaster.feature_names = feature_names
        
        with st.spinner('Preprocessing data...'):
            X_train_processed, X_test_processed = forecaster.preprocess_data(
                X_train, X_test, use_scaling, use_feature_selection, n_features
            )
        
        with st.spinner('Training models...'):
            forecaster.train_models(X_train_processed, y_train, use_grid_search)
        
        with st.spinner('Evaluating models...'):
            results = forecaster.evaluate_models(X_test_processed, y_test)
        
        # Store results in session state
        st.session_state['forecaster'] = forecaster
        st.session_state['results'] = results
        st.session_state['X_train_processed'] = X_train_processed
        st.session_state['X_test_processed'] = X_test_processed
        st.session_state['test_df'] = test_df
        
        st.success("✅ Models trained successfully!")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        forecaster = st.session_state['forecaster']
        test_df = st.session_state['test_df']
        
        st.subheader("🎯 Model Performance")
        
        # Model comparison
        fig_comparison = plot_model_comparison(results)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Best model
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        best_r2 = results[best_model]['r2']
        st.success(f"🏆 Best performing model: **{best_model}** (R² Score: {best_r2:.4f})")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        selected_model = st.selectbox("Select Model for Detailed Analysis", list(results.keys()))
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{results[selected_model]['r2']:.4f}")
        with col2:
            st.metric("MAE", f"${results[selected_model]['mae']:,.2f}")
        with col3:
            st.metric("RMSE", f"${results[selected_model]['rmse']:,.2f}")
        with col4:
            st.metric("MAPE", f"{results[selected_model]['mape']:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            y_true = forecaster.y_test
            y_pred = results[selected_model]['predictions']
            fig_actual_pred = plot_actual_vs_predicted(y_true, y_pred, selected_model)
            st.plotly_chart(fig_actual_pred, use_container_width=True)
        
        with col2:
            # Residuals
            fig_residuals = plot_residuals(y_true, y_pred, selected_model)
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature importance (if available)
        importance = forecaster.get_feature_importance(selected_model)
        if importance is not None:
            st.subheader(f"🔍 Feature Importance - {selected_model}")
            feature_names = st.session_state['X_train_processed'].columns.tolist()
            fig_importance = plot_feature_importance(importance, feature_names, selected_model)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Time series forecast
        st.subheader(f"📅 Sales Forecast - {selected_model}")
        fig_forecast = plot_time_series_forecast(test_df, y_pred, selected_model)
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Individual prediction section
    st.subheader("🎯 Individual Sales Prediction")
    
    if 'forecaster' in st.session_state:
        forecaster = st.session_state['forecaster']
        
        st.write("**Adjust store and economic features to predict sales:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            store = st.selectbox("Store", options=sorted(full_df['Store'].unique()))
            dept = st.selectbox("Department", options=sorted(full_df['Dept'].unique()))
            store_type = st.selectbox("Store Type", options=['A', 'B', 'C'])
        
        with col2:
            temperature = st.slider("Temperature (°F)", 0.0, 100.0, 70.0)
            fuel_price = st.slider("Fuel Price ($)", 2.0, 5.0, 3.5)
            cpi = st.slider("Consumer Price Index", 200.0, 230.0, 215.0)
        
        with col3:
            unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0)
            is_holiday = st.checkbox("Is Holiday Week")
            markdown1 = st.slider("MarkDown1 ($)", 0.0, 10000.0, 0.0)
        
        if st.button("🎯 Predict Sales"):
            # Create feature vector (simplified)
            # Note: This is a simplified prediction - in practice, you'd need all features
            st.info("Note: This is a simplified prediction using key features. For full accuracy, all features would be needed.")
            
            # Show prediction range based on historical data
            similar_records = full_df[
                (full_df['Store'] == store) & 
                (full_df['Dept'] == dept) & 
                (full_df['IsHoliday'] == is_holiday)
            ]
            
            if len(similar_records) > 0:
                avg_sales = similar_records['Weekly_Sales'].mean()
                min_sales = similar_records['Weekly_Sales'].min()
                max_sales = similar_records['Weekly_Sales'].max()
                
                st.success(f"📊 **Predicted Sales Range for Store {store}, Dept {dept}:**")
                st.write(f"- **Average Historical Sales**: ${avg_sales:,.2f}")
                st.write(f"- **Range**: ${min_sales:,.2f} - ${max_sales:,.2f}")
                
                # Adjust based on economic factors
                economic_factor = 1.0
                if temperature > 80 or temperature < 40:
                    economic_factor *= 0.95  # Extreme weather reduces sales
                if fuel_price > 4.0:
                    economic_factor *= 0.9   # High fuel prices reduce sales
                if unemployment > 10.0:
                    economic_factor *= 0.85  # High unemployment reduces sales
                if is_holiday:
                    economic_factor *= 1.15  # Holidays increase sales
                
                adjusted_prediction = avg_sales * economic_factor
                st.write(f"- **Adjusted Prediction**: ${adjusted_prediction:,.2f}")
                
                # Show factors
                st.write(f"- **Economic Adjustment Factor**: {economic_factor:.3f}")
            else:
                st.warning("No historical data found for this store/department combination.")
    
    # Data exploration section
    with st.expander("🔍 Data Exploration"):
        st.subheader("Sample Data")
        
        tab1, tab2, tab3 = st.tabs(["Training Data", "Test Data", "Full Dataset"])
        
        with tab1:
            st.dataframe(train_df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(test_df.head(10), use_container_width=True)
        
        with tab3:
            st.dataframe(full_df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sales Statistics:**")
            sales_stats = full_df['Weekly_Sales'].describe()
            st.dataframe(sales_stats, use_container_width=True)
        
        with col2:
            st.write("**Store Information:**")
            store_info = full_df.groupby('Store').agg({
                'Weekly_Sales': ['count', 'mean', 'sum'],
                'Dept': 'nunique'
            }).round(2)
            store_info.columns = ['Records', 'Avg_Sales', 'Total_Sales', 'Departments']
            st.dataframe(store_info.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
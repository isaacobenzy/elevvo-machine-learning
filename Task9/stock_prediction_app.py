#!/usr/bin/env python3
"""
Stock Price Prediction System
A comprehensive Streamlit application for stock price forecasting using machine learning and time series models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. LSTM models will be disabled.")

class StockPredictor:
    """Stock Price Prediction System"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_names = []
        self.lstm_model = None
        self.arima_model = None
        
    def load_data(self, file_path):
        """Load stock market dataset"""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df, target_column='Next_Close', feature_selection=True, n_features=20):
        """Preprocess the stock data"""
        # Remove rows with NaN values
        df_clean = df.dropna().copy()
        
        # Separate features and target
        exclude_cols = ['Date', 'Next_Close', 'Next_3_Close', 'Next_5_Close', 'Next_10_Close', 
                       'Price_Change', 'Price_Change_Pct', 'Price_Up', 'Strong_Up', 'Strong_Down', 'Next_5_Volatility']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols].copy()
        y = df_clean[target_column].copy()
        
        self.feature_names = feature_cols
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if feature_selection and len(feature_cols) > n_features:
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [feature_cols[i] for i in selected_indices]
        
        return X_scaled, y.values, df_clean
    
    def create_lstm_sequences(self, data, sequence_length=60):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_traditional_models(self, X_train, y_train, use_grid_search=False):
        """Train traditional ML models"""
        models_config = {
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50)),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42)
        }
        
        if use_grid_search:
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'SVR': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                },
                'Ridge': {
                    'alpha': [0.1, 1, 10]
                }
            }
        
        trained_models = {}
        
        for name, model in models_config.items():
            try:
                if use_grid_search and name in param_grids:
                    # Use TimeSeriesSplit for time series data
                    tscv = TimeSeriesSplit(n_splits=3)
                    grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    trained_models[name] = grid_search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    trained_models[name] = model
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
        
        self.models = trained_models
        return trained_models
    
    def train_lstm_model(self, df, target_column='Close', sequence_length=60, epochs=50):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[target_column].values.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_lstm_sequences(scaled_data, sequence_length)
        
        if len(X) == 0:
            st.error("Not enough data for LSTM training")
            return None
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        if model is None:
            return None
            
        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.lstm_model = model
        self.lstm_scaler = scaler
        self.lstm_sequence_length = sequence_length
        
        return model, history
    
    def train_arima_model(self, df, target_column='Close', order=(5,1,0)):
        """Train ARIMA model"""
        try:
            # Prepare time series data
            ts_data = df.set_index('Date')[target_column]
            
            # Check stationarity
            adf_result = adfuller(ts_data.dropna())
            
            # Train ARIMA model
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            self.arima_model = fitted_model
            return fitted_model
            
        except Exception as e:
            st.warning(f"Error training ARIMA model: {str(e)}")
            return None
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                
                results[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                    'predictions': y_pred
                }
            except Exception as e:
                st.warning(f"Error evaluating {name}: {str(e)}")
        
        return results
    
    def predict_future_prices(self, model_name, days_ahead=30):
        """Predict future stock prices"""
        if model_name not in self.models:
            return None
        
        # This is a simplified prediction - in practice, you'd need to handle
        # the recursive nature of multi-step forecasting
        model = self.models[model_name]
        
        # For demonstration, we'll use the last known features
        # In practice, you'd need to forecast features as well
        last_features = self.last_X_test[-1:] if hasattr(self, 'last_X_test') else None
        
        if last_features is not None:
            predictions = []
            current_features = last_features.copy()
            
            for _ in range(days_ahead):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                # Update features (simplified - in practice this would be more complex)
                current_features = current_features  # Placeholder
            
            return predictions
        
        return None

# Plotting functions
def plot_stock_price_history(df):
    """Plot stock price history"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Stock Price History', 'Volume'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title='Stock Price and Volume History')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_technical_indicators(df):
    """Plot technical indicators"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'Volume Indicators'],
        vertical_spacing=0.1
    )
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='green')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=2)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
    
    # MACD
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='green')), row=2, col=2)
    
    # Stochastic
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Stoch_K'], name='%K', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Stoch_D'], name='%D', line=dict(color='red')), row=3, col=1)
    
    # Volume indicators
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_Ratio'], name='Volume Ratio', line=dict(color='orange')), row=3, col=2)
    
    fig.update_layout(height=800, title='Technical Indicators Analysis')
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plot actual vs predicted prices"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Actual vs Predicted - {model_name}', 'Residuals'],
        vertical_spacing=0.1
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=list(range(len(y_true))), y=y_true, name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(y_pred))), y=y_pred, name='Predicted', line=dict(color='red')),
        row=1, col=1
    )
    
    # Residuals
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(x=list(range(len(residuals))), y=residuals, name='Residuals', mode='markers'),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    fig.update_layout(height=600)
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    # Get top 15 features
    top_indices = np.argsort(importance)[-15:]
    top_importance = importance[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    fig = px.bar(x=top_importance, y=top_features, orientation='h',
                 title=f'Top 15 Feature Importance - {model_name}')
    fig.update_layout(height=500, xaxis_title='Importance', yaxis_title='Features')
    return fig

def plot_model_comparison(results):
    """Plot model performance comparison"""
    metrics = ['rmse', 'mae', 'r2', 'mape']
    model_names = list(results.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['RMSE (Lower is Better)', 'MAE (Lower is Better)', 'R² (Higher is Better)', 'MAPE (Lower is Better)'],
        vertical_spacing=0.1
    )
    
    # RMSE
    rmse_values = [results[model]['rmse'] for model in model_names]
    fig.add_trace(go.Bar(x=model_names, y=rmse_values, name='RMSE'), row=1, col=1)
    
    # MAE
    mae_values = [results[model]['mae'] for model in model_names]
    fig.add_trace(go.Bar(x=model_names, y=mae_values, name='MAE'), row=1, col=2)
    
    # R²
    r2_values = [results[model]['r2'] for model in model_names]
    fig.add_trace(go.Bar(x=model_names, y=r2_values, name='R²'), row=2, col=1)
    
    # MAPE
    mape_values = [results[model]['mape'] for model in model_names]
    fig.add_trace(go.Bar(x=model_names, y=mape_values, name='MAPE'), row=2, col=2)
    
    fig.update_layout(height=600, title='Model Performance Comparison', showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig

def plot_price_forecast(df, predictions, model_name, days_ahead=30):
    """Plot price forecast"""
    # Create future dates
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=df['Date'][-100:], y=df['Close'][-100:],
        name='Historical Prices', line=dict(color='blue')
    ))
    
    # Predictions
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=future_dates, y=predictions,
            name=f'Predicted Prices ({model_name})', line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title=f'Stock Price Forecast - {model_name}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500
    )
    
    return fig

# Streamlit App
def main():
    st.set_page_config(
        page_title="Stock Price Prediction System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Stock Price Prediction System")
    st.markdown("""
    A comprehensive machine learning application for stock price forecasting using traditional ML models, 
    LSTM neural networks, and ARIMA time series analysis.
    """)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Load data
    data_file = st.sidebar.selectbox(
        "Select Dataset",
        ["Dataset/stock_data_train.csv", "Dataset/stock_data_full.csv"]
    )
    
    try:
        df = predictor.load_data(data_file)
        if df is None:
            st.error("Failed to load dataset. Please ensure the dataset files exist.")
            return
    except:
        st.error("Dataset not found. Please run download_dataset.py first.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dataset Overview", "📈 Technical Analysis", "🤖 Model Training", 
        "🎯 Predictions", "📋 Model Evaluation", "🔮 Forecasting"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col3:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        with col4:
            daily_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            st.metric("Daily Change", f"{daily_return:.2f}%")
        
        st.subheader("Stock Price History")
        fig_history = plot_stock_price_history(df)
        st.plotly_chart(fig_history, use_container_width=True)
        
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10))
        
        st.subheader("Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe())
    
    with tab2:
        st.header("📈 Technical Analysis")
        
        st.subheader("Technical Indicators")
        fig_indicators = plot_technical_indicators(df)
        st.plotly_chart(fig_indicators, use_container_width=True)
        
        st.subheader("Market Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Price Statistics**")
            st.write(f"Highest: ${df['High'].max():.2f}")
            st.write(f"Lowest: ${df['Low'].min():.2f}")
            st.write(f"Average: ${df['Close'].mean():.2f}")
        
        with col2:
            st.write("**Volatility**")
            daily_returns = df['Close'].pct_change().dropna()
            st.write(f"Daily Vol: {daily_returns.std()*100:.2f}%")
            st.write(f"Annual Vol: {daily_returns.std()*np.sqrt(252)*100:.1f}%")
        
        with col3:
            st.write("**Technical Levels**")
            st.write(f"RSI: {df['RSI'].iloc[-1]:.1f}")
            st.write(f"MACD: {df['MACD'].iloc[-1]:.2f}")
    
    with tab3:
        st.header("🤖 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traditional ML Models")
            target_column = st.selectbox("Select Target", ['Next_Close', 'Price_Change', 'Price_Change_Pct'])
            feature_selection = st.checkbox("Use Feature Selection", value=True)
            n_features = st.slider("Number of features", 10, 30, 20) if feature_selection else None
            use_grid_search = st.checkbox("Use Grid Search", value=False)
            
            if st.button("🚀 Train Traditional Models", type="primary"):
                with st.spinner("Training models..."):
                    # Preprocess data
                    X, y, df_clean = predictor.preprocess_data(df, target_column, feature_selection, n_features)
                    
                    # Time series split
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Store for evaluation
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    predictor.last_X_test = X_test
                    
                    # Train models
                    models = predictor.train_traditional_models(X_train, y_train, use_grid_search)
                    
                    st.success(f"✅ Successfully trained {len(models)} models!")
                    
                    # Show trained models
                    for name in models.keys():
                        st.write(f"✓ {name}")
        
        with col2:
            st.subheader("Advanced Models")
            
            # LSTM Training
            if TENSORFLOW_AVAILABLE:
                lstm_epochs = st.slider("LSTM Epochs", 10, 100, 50)
                lstm_sequence = st.slider("Sequence Length", 30, 120, 60)
                
                if st.button("🧠 Train LSTM Model"):
                    with st.spinner("Training LSTM model..."):
                        model, history = predictor.train_lstm_model(df, 'Close', lstm_sequence, lstm_epochs)
                        if model is not None:
                            st.success("✅ LSTM model trained successfully!")
                            
                            # Plot training history
                            if history is not None:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                                fig.update_layout(title='LSTM Training History', xaxis_title='Epoch', yaxis_title='Loss')
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install TensorFlow to enable LSTM models")
            
            # ARIMA Training
            arima_p = st.slider("ARIMA p", 1, 10, 5)
            arima_d = st.slider("ARIMA d", 0, 2, 1)
            arima_q = st.slider("ARIMA q", 0, 5, 0)
            
            if st.button("📊 Train ARIMA Model"):
                with st.spinner("Training ARIMA model..."):
                    arima_model = predictor.train_arima_model(df, 'Close', (arima_p, arima_d, arima_q))
                    if arima_model is not None:
                        st.success("✅ ARIMA model trained successfully!")
                        st.text(str(arima_model.summary()))
    
    with tab4:
        st.header("🎯 Stock Price Predictions")
        
        if not predictor.models:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            st.subheader("Single Prediction")
            
            # Model selection
            model_name = st.selectbox("Select Model", list(predictor.models.keys()))
            
            # Feature input (simplified)
            st.subheader("Current Market Conditions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_price = st.number_input("Current Price", value=float(df['Close'].iloc[-1]), step=0.01)
                volume = st.number_input("Volume", value=int(df['Volume'].iloc[-1]), step=1000)
            
            with col2:
                rsi = st.number_input("RSI", value=float(df['RSI'].iloc[-1]), min_value=0.0, max_value=100.0)
                macd = st.number_input("MACD", value=float(df['MACD'].iloc[-1]))
            
            with col3:
                sma_20 = st.number_input("SMA 20", value=float(df['SMA_20'].iloc[-1]))
                atr = st.number_input("ATR", value=float(df['ATR'].iloc[-1]))
            
            if st.button("🔍 Predict Price", type="primary"):
                # This is a simplified prediction interface
                # In practice, you'd need all features
                st.info("Prediction functionality requires all features. Use the model evaluation tab for detailed predictions.")
    
    with tab5:
        st.header("📋 Model Evaluation")
        
        if not predictor.models or 'X_test' not in st.session_state:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            # Evaluate models
            results = predictor.evaluate_models(st.session_state.X_test, st.session_state.y_test)
            
            if results:
                st.subheader("Model Performance Comparison")
                fig_comparison = plot_model_comparison(results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Performance metrics table
                st.subheader("Detailed Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [results[model]['rmse'] for model in results.keys()],
                    'MAE': [results[model]['mae'] for model in results.keys()],
                    'R²': [results[model]['r2'] for model in results.keys()],
                    'MAPE (%)': [results[model]['mape'] for model in results.keys()]
                })
                st.dataframe(metrics_df.round(4))
                
                # Best model
                best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
                st.success(f"🏆 Best Model: {best_model} (RMSE: {results[best_model]['rmse']:.4f})")
                
                # Actual vs Predicted
                st.subheader(f"Actual vs Predicted - {best_model}")
                y_pred = results[best_model]['predictions']
                fig_pred = plot_actual_vs_predicted(st.session_state.y_test, y_pred, best_model)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Feature Importance
                st.subheader(f"Feature Importance - {best_model}")
                fig_importance = plot_feature_importance(
                    predictor.models[best_model], 
                    predictor.feature_names, 
                    best_model
                )
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
    
    with tab6:
        st.header("🔮 Price Forecasting")
        
        if not predictor.models:
            st.warning("⚠️ Please train models first in the Model Training tab.")
        else:
            st.subheader("Future Price Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_model = st.selectbox("Select Forecasting Model", list(predictor.models.keys()))
                days_ahead = st.slider("Days to Forecast", 1, 60, 30)
            
            with col2:
                st.write("**Forecast Settings**")
                confidence_interval = st.checkbox("Show Confidence Interval", value=True)
                show_technical = st.checkbox("Show Technical Levels", value=True)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    # Generate predictions
                    predictions = predictor.predict_future_prices(forecast_model, days_ahead)
                    
                    if predictions is not None:
                        # Plot forecast
                        fig_forecast = plot_price_forecast(df, predictions, forecast_model, days_ahead)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast summary
                        st.subheader("Forecast Summary")
                        current_price = df['Close'].iloc[-1]
                        final_price = predictions[-1]
                        price_change = final_price - current_price
                        price_change_pct = (price_change / current_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Predicted Price", f"${final_price:.2f}")
                        with col3:
                            st.metric("Expected Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}")
                    else:
                        st.error("Unable to generate forecast. Please ensure models are properly trained.")
            
            # ARIMA Forecast
            if predictor.arima_model is not None:
                st.subheader("ARIMA Time Series Forecast")
                
                if st.button("📊 Generate ARIMA Forecast"):
                    try:
                        forecast = predictor.arima_model.forecast(steps=days_ahead)
                        forecast_ci = predictor.arima_model.get_forecast(steps=days_ahead).conf_int()
                        
                        # Create forecast plot
                        last_date = df['Date'].max()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df['Date'][-100:], y=df['Close'][-100:],
                            name='Historical', line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=forecast,
                            name='ARIMA Forecast', line=dict(color='red', dash='dash')
                        ))
                        
                        # Confidence interval
                        if confidence_interval:
                            fig.add_trace(go.Scatter(
                                x=future_dates, y=forecast_ci.iloc[:, 0],
                                fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=future_dates, y=forecast_ci.iloc[:, 1],
                                fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                                name='Confidence Interval', fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig.update_layout(
                            title='ARIMA Stock Price Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating ARIMA forecast: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>📈 Stock Price Prediction System | Built with Streamlit, Scikit-learn & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
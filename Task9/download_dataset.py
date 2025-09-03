#!/usr/bin/env python3
"""
Stock Market Data Generator
Generates synthetic stock market data with technical indicators for stock price prediction
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories"""
    directories = ['Dataset', 'Results', 'Screenshots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created successfully!")

def generate_stock_price_data(n_days=1000, start_price=100.0, volatility=0.02):
    """Generate synthetic stock price data using geometric Brownian motion"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price data using geometric Brownian motion
    prices = [start_price]
    
    for i in range(1, n_days):
        # Random walk with drift
        drift = 0.0005  # Small positive drift
        shock = np.random.normal(0, volatility)
        price_change = prices[-1] * (drift + shock)
        new_price = max(prices[-1] + price_change, 1.0)  # Ensure price stays positive
        prices.append(new_price)
    
    # Generate volume data (correlated with price changes)
    volumes = []
    base_volume = 1000000
    
    for i in range(n_days):
        if i == 0:
            volume = base_volume + np.random.normal(0, base_volume * 0.3)
        else:
            # Higher volume on larger price changes
            price_change_pct = abs((prices[i] - prices[i-1]) / prices[i-1])
            volume_multiplier = 1 + price_change_pct * 10
            volume = base_volume * volume_multiplier + np.random.normal(0, base_volume * 0.2)
        
        volumes.append(max(int(volume), 100000))  # Ensure minimum volume
    
    # Create OHLC data
    data = []
    for i in range(n_days):
        # Generate Open, High, Low based on Close price
        close = prices[i]
        
        if i == 0:
            open_price = close
        else:
            # Open is close to previous close with some gap
            gap = np.random.normal(0, close * 0.005)
            open_price = max(prices[i-1] + gap, 1.0)
        
        # High and Low around the close price
        daily_range = abs(np.random.normal(0, close * 0.02))
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volumes[i]
        })
    
    return pd.DataFrame(data)

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # Average True Range (ATR)
    df['TR1'] = df['High'] - df['Low']
    df['TR2'] = abs(df['High'] - df['Close'].shift(1))
    df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    
    # Price Rate of Change
    df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # Clean up temporary columns
    df.drop(['TR1', 'TR2', 'TR3'], axis=1, inplace=True)
    
    return df

def create_target_variables(df):
    """Create target variables for prediction"""
    # Next day price prediction
    df['Next_Close'] = df['Close'].shift(-1)
    df['Price_Change'] = df['Next_Close'] - df['Close']
    df['Price_Change_Pct'] = (df['Price_Change'] / df['Close']) * 100
    
    # Binary classification targets
    df['Price_Up'] = (df['Price_Change'] > 0).astype(int)
    df['Strong_Up'] = (df['Price_Change_Pct'] > 2).astype(int)
    df['Strong_Down'] = (df['Price_Change_Pct'] < -2).astype(int)
    
    # Multi-day predictions
    df['Next_3_Close'] = df['Close'].shift(-3)
    df['Next_5_Close'] = df['Close'].shift(-5)
    df['Next_10_Close'] = df['Close'].shift(-10)
    
    # Volatility prediction
    df['Next_5_Volatility'] = df['Close'].rolling(window=5).std().shift(-5)
    
    return df

def create_lag_features(df):
    """Create lagged features for time series modeling"""
    # Price lags
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
    
    # Moving average ratios
    df['Price_to_SMA_5'] = df['Close'] / df['SMA_5']
    df['Price_to_SMA_20'] = df['Close'] / df['SMA_20']
    df['Price_to_SMA_50'] = df['Close'] / df['SMA_50']
    
    # Volatility features
    df['Price_Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Price_Volatility_20'] = df['Close'].rolling(window=20).std()
    
    return df

def add_market_regime_features(df):
    """Add market regime and trend features"""
    # Trend identification
    df['Trend_5'] = np.where(df['SMA_5'] > df['SMA_5'].shift(1), 1, 
                            np.where(df['SMA_5'] < df['SMA_5'].shift(1), -1, 0))
    df['Trend_20'] = np.where(df['SMA_20'] > df['SMA_20'].shift(1), 1, 
                             np.where(df['SMA_20'] < df['SMA_20'].shift(1), -1, 0))
    
    # Market regime (bull/bear/sideways)
    df['Bull_Market'] = ((df['Close'] > df['SMA_50']) & (df['SMA_20'] > df['SMA_50'])).astype(int)
    df['Bear_Market'] = ((df['Close'] < df['SMA_50']) & (df['SMA_20'] < df['SMA_50'])).astype(int)
    
    # Volatility regime
    vol_median = df['ATR'].rolling(window=50).median()
    df['High_Volatility'] = (df['ATR'] > vol_median * 1.5).astype(int)
    df['Low_Volatility'] = (df['ATR'] < vol_median * 0.5).astype(int)
    
    return df

def save_datasets(df):
    """Save different versions of the dataset"""
    # Remove rows with NaN values (due to rolling calculations)
    df_clean = df.dropna().copy()
    
    # Full dataset
    df_clean.to_csv('Dataset/stock_data_full.csv', index=False)
    print(f"✅ Saved full dataset: {len(df_clean)} samples")
    
    # Training dataset (80% of data)
    train_size = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:train_size].copy()
    df_train.to_csv('Dataset/stock_data_train.csv', index=False)
    print(f"✅ Saved training dataset: {len(df_train)} samples")
    
    # Test dataset (20% of data)
    df_test = df_clean.iloc[train_size:].copy()
    df_test.to_csv('Dataset/stock_data_test.csv', index=False)
    print(f"✅ Saved test dataset: {len(df_test)} samples")
    
    # Feature names
    feature_columns = [col for col in df_clean.columns if col not in 
                      ['Date', 'Next_Close', 'Next_3_Close', 'Next_5_Close', 'Next_10_Close', 
                       'Price_Change', 'Price_Change_Pct', 'Price_Up', 'Strong_Up', 'Strong_Down', 'Next_5_Volatility']]
    
    with open('Dataset/feature_names.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    # Target names
    target_columns = ['Next_Close', 'Price_Change', 'Price_Change_Pct', 'Price_Up', 'Strong_Up', 'Strong_Down']
    with open('Dataset/target_names.txt', 'w') as f:
        for target in target_columns:
            f.write(f"{target}\n")
    
    return df_clean

def generate_dataset_statistics(df):
    """Generate and display dataset statistics"""
    print("\n" + "="*60)
    print("📊 STOCK MARKET DATASET STATISTICS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Dataset Overview:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Total features: {len([col for col in df.columns if col not in ['Date']])}")    
    print(f"   • Technical indicators: {len([col for col in df.columns if col in ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Stoch_K', 'ATR']])}")
    
    # Price statistics
    print(f"\n💰 Price Statistics:")
    print(f"   • Starting price: ${df['Close'].iloc[0]:.2f}")
    print(f"   • Ending price: ${df['Close'].iloc[-1]:.2f}")
    print(f"   • Highest price: ${df['High'].max():.2f}")
    print(f"   • Lowest price: ${df['Low'].min():.2f}")
    print(f"   • Average daily volume: {df['Volume'].mean():,.0f}")
    
    # Technical indicator statistics
    print(f"\n📊 Technical Indicators:")
    print(f"   • Average RSI: {df['RSI'].mean():.2f}")
    print(f"   • RSI range: {df['RSI'].min():.2f} - {df['RSI'].max():.2f}")
    print(f"   • Average ATR: {df['ATR'].mean():.2f}")
    print(f"   • MACD range: {df['MACD'].min():.2f} - {df['MACD'].max():.2f}")
    
    # Market regime statistics
    print(f"\n🐂 Market Regimes:")
    bull_days = df['Bull_Market'].sum()
    bear_days = df['Bear_Market'].sum()
    sideways_days = len(df) - bull_days - bear_days
    print(f"   • Bull market days: {bull_days} ({bull_days/len(df)*100:.1f}%)")
    print(f"   • Bear market days: {bear_days} ({bear_days/len(df)*100:.1f}%)")
    print(f"   • Sideways market days: {sideways_days} ({sideways_days/len(df)*100:.1f}%)")
    
    # Volatility statistics
    print(f"\n📈 Volatility Analysis:")
    daily_returns = df['Close'].pct_change().dropna()
    print(f"   • Average daily return: {daily_returns.mean()*100:.3f}%")
    print(f"   • Daily volatility: {daily_returns.std()*100:.3f}%")
    print(f"   • Annualized volatility: {daily_returns.std()*np.sqrt(252)*100:.1f}%")
    
    # Target variable statistics
    print(f"\n🎯 Target Variables:")
    up_days = df['Price_Up'].sum()
    print(f"   • Days with price increase: {up_days} ({up_days/len(df)*100:.1f}%)")
    strong_up = df['Strong_Up'].sum()
    strong_down = df['Strong_Down'].sum()
    print(f"   • Strong up days (>2%): {strong_up} ({strong_up/len(df)*100:.1f}%)")
    print(f"   • Strong down days (<-2%): {strong_down} ({strong_down/len(df)*100:.1f}%)")
    
    # Feature correlation insights
    print(f"\n🔗 Feature Insights:")
    feature_cols = [col for col in df.columns if col not in ['Date', 'Next_Close', 'Price_Change', 'Price_Change_Pct']]
    corr_with_target = df[feature_cols + ['Price_Change_Pct']].corr()['Price_Change_Pct'].abs().sort_values(ascending=False)
    print(f"   • Most predictive features:")
    for i, (feature, corr) in enumerate(corr_with_target.head(5).items()):
        if feature != 'Price_Change_Pct':
            print(f"     {i+1}. {feature}: {corr:.3f}")
    
    print(f"\n✅ Dataset generation completed successfully!")
    print(f"📁 Files saved in Dataset/ directory")
    print(f"🚀 Ready to run: streamlit run stock_prediction_app.py")
    print("="*60)

def main():
    """Main function to generate the stock market dataset"""
    print("🚀 Generating Stock Market Dataset...")
    
    # Create directories
    create_directories()
    
    # Generate stock price data
    print("📈 Generating stock price data...")
    df = generate_stock_price_data(n_days=1000)
    
    # Calculate technical indicators
    print("📊 Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Create target variables
    print("🎯 Creating target variables...")
    df = create_target_variables(df)
    
    # Create lag features
    print("⏰ Creating lag features...")
    df = create_lag_features(df)
    
    # Add market regime features
    print("🐂 Adding market regime features...")
    df = add_market_regime_features(df)
    
    # Save datasets
    print("💾 Saving datasets...")
    df_final = save_datasets(df)
    
    # Generate statistics
    generate_dataset_statistics(df_final)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories"""
    directories = ['Dataset', 'Results', 'Screenshots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_walmart_dataset():
    """Download Walmart sales dataset"""
    print("Attempting to download Walmart sales dataset...")
    
    # Note: The original Walmart dataset from Kaggle requires authentication
    # For this demo, we'll create a synthetic dataset with similar characteristics
    print("Note: Creating synthetic Walmart-style sales dataset...")
    print("In a real scenario, you would download from Kaggle or other sources.")
    
    return create_synthetic_walmart_dataset()

def create_synthetic_walmart_dataset():
    """Create synthetic Walmart sales dataset"""
    print("Generating synthetic Walmart sales dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Date range: 3 years of weekly data
    start_date = datetime(2010, 2, 5)
    end_date = datetime(2012, 10, 26)
    
    # Generate weekly dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=7)
    
    # Store information
    stores = list(range(1, 46))  # 45 stores
    departments = list(range(1, 100))  # 99 departments
    
    # Generate sales data
    sales_data = []
    
    for store in stores:
        # Store-specific characteristics
        store_size = np.random.choice(['A', 'B', 'C'], p=[0.3, 0.4, 0.3])
        base_sales = {
            'A': np.random.normal(50000, 10000),
            'B': np.random.normal(30000, 8000),
            'C': np.random.normal(20000, 5000)
        }[store_size]
        
        # Select random departments for this store (not all stores have all departments)
        store_departments = np.random.choice(departments, 
                                           size=np.random.randint(30, 80), 
                                           replace=False)
        
        for dept in store_departments:
            # Department-specific characteristics
            dept_multiplier = np.random.uniform(0.1, 2.0)
            
            for date in dates:
                # Seasonal patterns
                week_of_year = date.isocalendar()[1]
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)
                
                # Holiday effects (simplified)
                holiday_factor = 1.0
                if week_of_year in [47, 48, 49, 50, 51, 52]:  # Holiday season
                    holiday_factor = 1.5
                elif week_of_year in [1, 2]:  # Post-holiday
                    holiday_factor = 0.7
                
                # Random noise
                noise = np.random.normal(1, 0.2)
                
                # Calculate weekly sales
                weekly_sales = (base_sales * dept_multiplier * seasonal_factor * 
                              holiday_factor * noise)
                weekly_sales = max(0, weekly_sales)  # Ensure non-negative
                
                # Markdown data (promotional discounts)
                markdown_prob = 0.3  # 30% chance of markdown
                markdowns = {}
                for i in range(1, 6):  # MarkDown1 to MarkDown5
                    if np.random.random() < markdown_prob:
                        markdowns[f'MarkDown{i}'] = np.random.uniform(0, 5000)
                    else:
                        markdowns[f'MarkDown{i}'] = 0
                
                sales_data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Weekly_Sales': round(weekly_sales, 2),
                    'IsHoliday': week_of_year in [6, 36, 47, 52],  # Simplified holiday weeks
                    **markdowns
                })
    
    # Create DataFrame
    sales_df = pd.DataFrame(sales_data)
    
    # Create stores dataset
    stores_data = []
    for store in stores:
        stores_data.append({
            'Store': store,
            'Type': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.4, 0.3]),
            'Size': np.random.randint(34000, 220000)
        })
    
    stores_df = pd.DataFrame(stores_data)
    
    # Create features dataset (economic indicators)
    features_data = []
    for store in stores:
        for date in dates:
            # Generate economic indicators
            temperature = 45 + 30 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 5)
            fuel_price = 2.5 + 1.5 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 0.2)
            cpi = 210 + 5 * (date.year - 2010) + np.random.normal(0, 2)
            unemployment = 8 + 2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 0.5)
            
            features_data.append({
                'Store': store,
                'Date': date.strftime('%Y-%m-%d'),
                'Temperature': round(temperature, 2),
                'Fuel_Price': round(fuel_price, 3),
                'CPI': round(cpi, 3),
                'Unemployment': round(unemployment, 3)
            })
    
    features_df = pd.DataFrame(features_data)
    
    return sales_df, stores_df, features_df

def create_time_features(df):
    """Create time-based features"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    
    # Cyclical features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    
    return df

def merge_datasets(sales_df, stores_df, features_df):
    """Merge all datasets"""
    # Merge sales with stores
    merged_df = sales_df.merge(stores_df, on='Store', how='left')
    
    # Merge with features
    merged_df = merged_df.merge(features_df, on=['Store', 'Date'], how='left')
    
    return merged_df

def create_train_test_split(df, test_weeks=8):
    """Create train/test split based on time"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Get unique dates and split
    unique_dates = sorted(df['Date'].unique())
    split_date = unique_dates[-test_weeks]
    
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    
    print(f"Train set: {len(train_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    print(f"Split date: {split_date.strftime('%Y-%m-%d')}")
    
    return train_df, test_df

def save_datasets(sales_df, stores_df, features_df, merged_df, train_df, test_df):
    """Save all datasets"""
    datasets = {
        'sales_data.csv': sales_df,
        'stores_data.csv': stores_df,
        'features_data.csv': features_df,
        'walmart_sales_full.csv': merged_df,
        'walmart_sales_train.csv': train_df,
        'walmart_sales_test.csv': test_df
    }
    
    for filename, df in datasets.items():
        filepath = os.path.join('Dataset', filename)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath} ({len(df):,} records)")

def generate_dataset_statistics(sales_df, stores_df, features_df, merged_df):
    """Generate and display dataset statistics"""
    print("\n" + "="*50)
    print("WALMART SALES DATASET STATISTICS")
    print("="*50)
    
    # Basic statistics
    print(f"\n📊 Dataset Overview:")
    print(f"Total sales records: {len(sales_df):,}")
    print(f"Number of stores: {sales_df['Store'].nunique()}")
    print(f"Number of departments: {sales_df['Dept'].nunique()}")
    print(f"Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
    print(f"Total weeks: {len(pd.to_datetime(sales_df['Date']).dt.to_period('W').unique())}")
    
    # Sales statistics
    print(f"\n💰 Sales Statistics:")
    print(f"Total sales: ${sales_df['Weekly_Sales'].sum():,.2f}")
    print(f"Average weekly sales: ${sales_df['Weekly_Sales'].mean():,.2f}")
    print(f"Median weekly sales: ${sales_df['Weekly_Sales'].median():,.2f}")
    print(f"Sales range: ${sales_df['Weekly_Sales'].min():,.2f} - ${sales_df['Weekly_Sales'].max():,.2f}")
    
    # Store statistics
    print(f"\n🏪 Store Statistics:")
    store_types = stores_df['Type'].value_counts()
    for store_type, count in store_types.items():
        print(f"Type {store_type} stores: {count}")
    
    print(f"Store size range: {stores_df['Size'].min():,} - {stores_df['Size'].max():,} sq ft")
    print(f"Average store size: {stores_df['Size'].mean():,.0f} sq ft")
    
    # Holiday statistics
    print(f"\n🎉 Holiday Statistics:")
    holiday_sales = sales_df[sales_df['IsHoliday'] == True]['Weekly_Sales'].mean()
    regular_sales = sales_df[sales_df['IsHoliday'] == False]['Weekly_Sales'].mean()
    print(f"Average holiday sales: ${holiday_sales:,.2f}")
    print(f"Average regular sales: ${regular_sales:,.2f}")
    print(f"Holiday sales boost: {((holiday_sales/regular_sales - 1) * 100):+.1f}%")
    
    # Markdown statistics
    markdown_cols = [col for col in sales_df.columns if 'MarkDown' in col]
    print(f"\n🏷️ Markdown Statistics:")
    for col in markdown_cols:
        non_zero = (sales_df[col] > 0).sum()
        total = len(sales_df)
        avg_markdown = sales_df[sales_df[col] > 0][col].mean()
        print(f"{col}: {non_zero:,}/{total:,} records ({non_zero/total*100:.1f}%), Avg: ${avg_markdown:.2f}")
    
    # Economic indicators
    print(f"\n📈 Economic Indicators:")
    print(f"Temperature range: {features_df['Temperature'].min():.1f}°F - {features_df['Temperature'].max():.1f}°F")
    print(f"Fuel price range: ${features_df['Fuel_Price'].min():.3f} - ${features_df['Fuel_Price'].max():.3f}")
    print(f"CPI range: {features_df['CPI'].min():.3f} - {features_df['CPI'].max():.3f}")
    print(f"Unemployment range: {features_df['Unemployment'].min():.3f}% - {features_df['Unemployment'].max():.3f}%")
    
    # Top performing stores and departments
    print(f"\n🏆 Top Performers:")
    top_stores = sales_df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(5)
    print("Top 5 stores by total sales:")
    for store, sales in top_stores.items():
        print(f"  Store {store}: ${sales:,.2f}")
    
    top_depts = sales_df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(5)
    print("Top 5 departments by total sales:")
    for dept, sales in top_depts.items():
        print(f"  Dept {dept}: ${sales:,.2f}")
    
    print("\n" + "="*50)

def main():
    """Main function to download and process the dataset"""
    print("Starting Walmart Sales Dataset Download and Processing...")
    
    # Create directories
    create_directories()
    
    # Download/create dataset
    sales_df, stores_df, features_df = download_walmart_dataset()
    
    # Create time features
    print("\nCreating time-based features...")
    sales_df = create_time_features(sales_df)
    features_df = create_time_features(features_df)
    
    # Merge datasets
    print("Merging datasets...")
    merged_df = merge_datasets(sales_df, stores_df, features_df)
    
    # Create train/test split
    print("Creating train/test split...")
    train_df, test_df = create_train_test_split(merged_df)
    
    # Save datasets
    print("\nSaving datasets...")
    save_datasets(sales_df, stores_df, features_df, merged_df, train_df, test_df)
    
    # Generate statistics
    generate_dataset_statistics(sales_df, stores_df, features_df, merged_df)
    
    print("\n✅ Dataset download and processing completed successfully!")
    print("\n📁 Files created:")
    print("  - Dataset/sales_data.csv")
    print("  - Dataset/stores_data.csv")
    print("  - Dataset/features_data.csv")
    print("  - Dataset/walmart_sales_full.csv")
    print("  - Dataset/walmart_sales_train.csv")
    print("  - Dataset/walmart_sales_test.csv")
    
    print("\n🚀 Ready to run the Streamlit application!")
    print("Run: streamlit run sales_forecasting_app.py")

if __name__ == "__main__":
    main()
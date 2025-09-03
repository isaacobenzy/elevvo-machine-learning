import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

def download_covertype_dataset():
    """
    Download and prepare the Covertype dataset from sklearn
    """
    print("Downloading Covertype dataset...")
    
    # Create Dataset directory if it doesn't exist
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    try:
        # Fetch the dataset
        covertype = fetch_covtype()
        
        # Create DataFrame
        feature_names = [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ]
        
        # Add wilderness area columns (binary)
        wilderness_areas = [f'Wilderness_Area_{i}' for i in range(1, 5)]
        
        # Add soil type columns (binary)
        soil_types = [f'Soil_Type_{i}' for i in range(1, 41)]
        
        # Combine all feature names
        all_features = feature_names + wilderness_areas + soil_types
        
        # Create DataFrame
        df = pd.DataFrame(covertype.data, columns=all_features)
        df['Cover_Type'] = covertype.target
        
        # Save the full dataset
        full_path = os.path.join(dataset_dir, "covertype_full.csv")
        df.to_csv(full_path, index=False)
        print(f"Full dataset saved to {full_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Cover types: {sorted(df['Cover_Type'].unique())}")
        
        # Create a smaller sample for faster processing during development
        sample_df = df.sample(n=50000, random_state=42)
        sample_path = os.path.join(dataset_dir, "covertype_sample.csv")
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample dataset saved to {sample_path}")
        print(f"Sample shape: {sample_df.shape}")
        
        # Split and save train/test sets
        X = sample_df.drop('Cover_Type', axis=1)
        y = sample_df['Cover_Type']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save train set
        train_df = pd.concat([X_train, y_train], axis=1)
        train_path = os.path.join(dataset_dir, "covertype_train.csv")
        train_df.to_csv(train_path, index=False)
        
        # Save test set
        test_df = pd.concat([X_test, y_test], axis=1)
        test_path = os.path.join(dataset_dir, "covertype_test.csv")
        test_df.to_csv(test_path, index=False)
        
        print(f"Train set saved to {train_path} - Shape: {train_df.shape}")
        print(f"Test set saved to {test_path} - Shape: {test_df.shape}")
        
        print("\nDataset info:")
        print(f"Features: {len(all_features)}")
        print(f"Samples: {len(df)}")
        print(f"Classes: {len(df['Cover_Type'].unique())}")
        print("\nClass distribution:")
        print(sample_df['Cover_Type'].value_counts().sort_index())
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_covertype_dataset()
    if success:
        print("\nDataset download completed successfully!")
    else:
        print("\nDataset download failed!")
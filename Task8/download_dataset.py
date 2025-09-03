#!/usr/bin/env python3
"""
Traffic Sign Recognition Dataset Generator
Generates synthetic traffic sign image features for classification

This script creates a synthetic dataset mimicking the German Traffic Sign Recognition Benchmark (GTSRB)
with image features extracted from traffic sign images for classification tasks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime

def create_directories():
    """Create necessary directories for the dataset"""
    directories = ['Dataset', 'Results', 'Screenshots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_gtsrb_dataset():
    """
    Note: The actual GTSRB dataset is quite large (~300MB)
    For this demo, we'll create synthetic traffic sign features
    
    In a real scenario, you would download from:
    https://benchmark.ini.rub.de/gtsrb_dataset.html
    """
    print("Note: Creating synthetic traffic sign features instead of downloading large GTSRB dataset")
    print("In production, you would download the actual GTSRB dataset from the official source")
    return create_synthetic_traffic_sign_features()

def create_synthetic_traffic_sign_features(num_samples=5000):
    """
    Create synthetic traffic sign image features
    Simulates features that would be extracted from actual traffic sign images
    """
    print(f"Generating {num_samples} synthetic traffic sign samples...")
    
    # Define 43 traffic sign classes (same as GTSRB)
    class_names = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
        'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
        'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    
    # Generate features for each sample
    features = []
    labels = []
    
    for i in range(num_samples):
        # Random class assignment
        class_id = random.randint(0, 42)
        class_name = class_names[class_id]
        
        # Generate image-like features
        # Simulate features extracted from traffic sign images
        
        # Color features (RGB histograms)
        red_hist = np.random.normal(128 + class_id * 2, 30, 8)  # Red channel histogram
        green_hist = np.random.normal(100 + class_id * 1.5, 25, 8)  # Green channel histogram
        blue_hist = np.random.normal(120 + class_id * 1.8, 28, 8)  # Blue channel histogram
        
        # Shape features (HOG-like features)
        hog_features = np.random.normal(0.5 + class_id * 0.01, 0.2, 16)
        
        # Edge features (Canny edge density)
        edge_density = random.uniform(0.1 + class_id * 0.005, 0.8)
        edge_orientation = np.random.normal(class_id * 8, 45, 4)  # Edge orientations
        
        # Texture features (LBP-like)
        texture_features = np.random.normal(0.3 + class_id * 0.008, 0.15, 8)
        
        # Geometric features
        area = random.uniform(500 + class_id * 10, 2000)
        perimeter = random.uniform(80 + class_id * 2, 200)
        circularity = random.uniform(0.3, 0.9)
        aspect_ratio = random.uniform(0.8, 1.2)
        
        # SIFT/SURF-like keypoint features
        num_keypoints = random.randint(10 + class_id, 50 + class_id * 2)
        keypoint_density = num_keypoints / area
        
        # Combine all features
        sample_features = np.concatenate([
            red_hist, green_hist, blue_hist,
            hog_features,
            [edge_density], edge_orientation,
            texture_features,
            [area, perimeter, circularity, aspect_ratio, num_keypoints, keypoint_density]
        ])
        
        features.append(sample_features)
        labels.append(class_id)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Create feature names
    feature_names = []
    
    # Color histogram features
    for channel in ['red', 'green', 'blue']:
        for i in range(8):
            feature_names.append(f'{channel}_hist_{i}')
    
    # HOG features
    for i in range(16):
        feature_names.append(f'hog_feature_{i}')
    
    # Edge features
    feature_names.append('edge_density')
    for i in range(4):
        feature_names.append(f'edge_orientation_{i}')
    
    # Texture features
    for i in range(8):
        feature_names.append(f'texture_feature_{i}')
    
    # Geometric features
    feature_names.extend(['area', 'perimeter', 'circularity', 'aspect_ratio', 'num_keypoints', 'keypoint_density'])
    
    return features, labels, class_names, feature_names

def create_feature_dataframe(features, labels, class_names, feature_names):
    """Create a pandas DataFrame with features and labels"""
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['class_id'] = labels
    df['class_name'] = [class_names[label] for label in labels]
    
    return df

def save_datasets(df, feature_names, class_names):
    """Save the datasets in various formats"""
    print("Saving datasets...")
    
    # Save full dataset
    df.to_csv('Dataset/gtsrb_features_full.csv', index=False)
    print("Saved: Dataset/gtsrb_features_full.csv")
    
    # Create train/test split (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_id'])
    
    # Save train and test sets
    train_df.to_csv('Dataset/gtsrb_features_train.csv', index=False)
    test_df.to_csv('Dataset/gtsrb_features_test.csv', index=False)
    print("Saved: Dataset/gtsrb_features_train.csv")
    print("Saved: Dataset/gtsrb_features_test.csv")
    
    # Save feature names
    pd.DataFrame({'feature_names': feature_names}).to_csv('Dataset/feature_names.csv', index=False)
    print("Saved: Dataset/feature_names.csv")
    
    # Save class names
    pd.DataFrame({'class_id': range(len(class_names)), 'class_name': class_names}).to_csv('Dataset/class_names.csv', index=False)
    print("Saved: Dataset/class_names.csv")
    
    return train_df, test_df

def generate_dataset_statistics(df, train_df, test_df, class_names):
    """Generate and display dataset statistics"""
    print("\n" + "="*60)
    print("GTSRB TRAFFIC SIGN DATASET STATISTICS")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"- Total samples: {len(df):,}")
    print(f"- Number of classes: {len(class_names)}")
    print(f"- Number of features: {len(df.columns) - 2}")
    print(f"- Training samples: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"- Testing samples: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nClass Distribution:")
    class_counts = df['class_name'].value_counts().sort_index()
    for i, (class_name, count) in enumerate(class_counts.head(10).items()):
        print(f"- {class_name}: {count} samples")
    if len(class_counts) > 10:
        print(f"... and {len(class_counts) - 10} more classes")
    
    print(f"\nFeature Statistics:")
    feature_cols = [col for col in df.columns if col not in ['class_id', 'class_name']]
    feature_stats = df[feature_cols].describe()
    print(f"- Mean feature value: {feature_stats.loc['mean'].mean():.3f}")
    print(f"- Std feature value: {feature_stats.loc['std'].mean():.3f}")
    print(f"- Min feature value: {feature_stats.loc['min'].min():.3f}")
    print(f"- Max feature value: {feature_stats.loc['max'].max():.3f}")
    
    print(f"\nSample Feature Ranges:")
    sample_features = ['red_hist_0', 'hog_feature_0', 'edge_density', 'area']
    for feature in sample_features:
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            mean_val = df[feature].mean()
            print(f"- {feature}: {min_val:.3f} to {max_val:.3f} (mean: {mean_val:.3f})")
    
    print(f"\nTop 5 Most Common Classes:")
    top_classes = df['class_name'].value_counts().head(5)
    for class_name, count in top_classes.items():
        percentage = (count / len(df)) * 100
        print(f"- {class_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nDataset Generation Complete!")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ready to run: streamlit run traffic_sign_app.py")
    print("="*60)

def main():
    """Main function to generate the traffic sign dataset"""
    print("Starting GTSRB Traffic Sign Dataset Generation...")
    
    # Create directories
    create_directories()
    
    # Generate synthetic traffic sign features
    features, labels, class_names, feature_names = download_gtsrb_dataset()
    
    # Create DataFrame
    df = create_feature_dataframe(features, labels, class_names, feature_names)
    
    # Save datasets
    train_df, test_df = save_datasets(df, feature_names, class_names)
    
    # Generate statistics
    generate_dataset_statistics(df, train_df, test_df, class_names)

if __name__ == "__main__":
    main()
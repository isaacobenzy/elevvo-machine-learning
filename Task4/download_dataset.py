import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import requests
from io import StringIO

def create_synthetic_loan_dataset():
    """
    Create a synthetic loan approval dataset since the Kaggle dataset requires authentication
    """
    print("Creating synthetic loan approval dataset...")
    
    # Create Dataset directory if it doesn't exist
    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Number of samples
        n_samples = 10000
        
        # Generate synthetic data
        data = {
            'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
            'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25]),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
            'ApplicantIncome': np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples).astype(int),
            'CoapplicantIncome': np.random.lognormal(mean=7.0, sigma=1.2, size=n_samples).astype(int),
            'LoanAmount': np.random.normal(loc=150, scale=50, size=n_samples).astype(int),
            'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360, 480], n_samples, p=[0.05, 0.1, 0.15, 0.2, 0.4, 0.1]),
            'Credit_History': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.35, 0.25])
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Introduce some missing values to make it realistic
        missing_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in missing_cols:
            missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        # Create target variable based on logical rules
        # Higher income, good credit history, lower loan amount -> higher approval chance
        total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
        income_to_loan_ratio = total_income / (df['LoanAmount'] * 1000)  # Convert to thousands
        
        # Calculate approval probability
        approval_prob = np.full(n_samples, 0.3)  # Base probability array
        
        # Income factor
        approval_prob += 0.3 * (total_income > total_income.median()).astype(int)
        
        # Credit history factor
        approval_prob += 0.25 * df['Credit_History'].fillna(0)
        
        # Education factor
        approval_prob += 0.1 * (df['Education'] == 'Graduate').astype(int)
        
        # Property area factor
        approval_prob += 0.05 * (df['Property_Area'] == 'Urban').astype(int)
        
        # Income to loan ratio factor
        approval_prob += 0.1 * (income_to_loan_ratio > income_to_loan_ratio.median()).astype(int)
        
        # Ensure probabilities are between 0 and 1
        approval_prob = np.clip(approval_prob, 0, 1)
        
        # Generate loan status based on probability
        df['Loan_Status'] = np.array([np.random.binomial(1, p) for p in approval_prob])
        df['Loan_Status'] = df['Loan_Status'].map({1: 'Y', 0: 'N'})
        
        # Save the full dataset
        full_path = os.path.join(dataset_dir, "loan_approval_full.csv")
        df.to_csv(full_path, index=False)
        print(f"Full dataset saved to {full_path}")
        print(f"Dataset shape: {df.shape}")
        
        # Split and save train/test sets
        X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = df['Loan_Status']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Add back IDs for tracking
        train_ids = df.loc[X_train.index, 'Loan_ID']
        test_ids = df.loc[X_test.index, 'Loan_ID']
        
        # Save train set
        train_df = pd.concat([train_ids, X_train, y_train], axis=1)
        train_path = os.path.join(dataset_dir, "loan_approval_train.csv")
        train_df.to_csv(train_path, index=False)
        
        # Save test set
        test_df = pd.concat([test_ids, X_test, y_test], axis=1)
        test_path = os.path.join(dataset_dir, "loan_approval_test.csv")
        test_df.to_csv(test_path, index=False)
        
        print(f"Train set saved to {train_path} - Shape: {train_df.shape}")
        print(f"Test set saved to {test_path} - Shape: {test_df.shape}")
        
        print("\nDataset info:")
        print(f"Features: {len(X.columns)}")
        print(f"Samples: {len(df)}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        print("\nLoan approval distribution:")
        print(df['Loan_Status'].value_counts())
        print(f"Approval rate: {(df['Loan_Status'] == 'Y').mean():.2%}")
        
        # Display sample statistics
        print("\nSample statistics:")
        print(f"Average Applicant Income: ${df['ApplicantIncome'].mean():,.0f}")
        print(f"Average Loan Amount: ${df['LoanAmount'].mean() * 1000:,.0f}")
        print(f"Credit History Rate: {df['Credit_History'].mean():.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return False

def download_alternative_dataset():
    """
    Try to download a real loan dataset from a public source
    """
    print("Attempting to download alternative loan dataset...")
    
    try:
        # This is a sample URL - in practice, you'd use a real dataset URL
        # For now, we'll create synthetic data
        return create_synthetic_loan_dataset()
        
    except Exception as e:
        print(f"Could not download alternative dataset: {str(e)}")
        print("Falling back to synthetic dataset creation...")
        return create_synthetic_loan_dataset()

if __name__ == "__main__":
    success = download_alternative_dataset()
    if success:
        print("\nDataset creation completed successfully!")
        print("\nNote: This is a synthetic dataset created for demonstration purposes.")
        print("In a real scenario, you would use actual loan approval data from Kaggle or other sources.")
    else:
        print("\nDataset creation failed!")
# Task 4: Loan Approval Prediction 💰

A comprehensive machine learning application for predicting loan approval decisions using various classification algorithms with advanced handling of imbalanced datasets.

## 🎯 Project Overview

This project implements a loan approval prediction system that helps financial institutions make data-driven decisions about loan applications. The system handles class imbalance using SMOTE (Synthetic Minority Oversampling Technique) and provides detailed performance metrics including precision, recall, and F1-score.

## 📊 Dataset Information

### Synthetic Loan Dataset
- **Total Samples**: 1,000 loan applications
- **Features**: 11 input features + 1 target variable
- **Target Variable**: Loan_Status (Y = Approved, N = Rejected)
- **Class Distribution**: Imbalanced dataset reflecting real-world scenarios

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| Loan_ID | Categorical | Unique loan identifier |
| Gender | Categorical | Applicant's gender (Male/Female) |
| Married | Categorical | Marital status (Yes/No) |
| Dependents | Categorical | Number of dependents (0/1/2/3+) |
| Education | Categorical | Education level (Graduate/Not Graduate) |
| Self_Employed | Categorical | Employment type (Yes/No) |
| ApplicantIncome | Numerical | Primary applicant's income |
| CoapplicantIncome | Numerical | Co-applicant's income |
| LoanAmount | Numerical | Loan amount in thousands |
| Loan_Amount_Term | Numerical | Loan term in months |
| Credit_History | Categorical | Credit history (1=Good, 0=Bad) |
| Property_Area | Categorical | Property location (Urban/Semiurban/Rural) |

## 🏗️ Project Structure

```
Task4/
├── Dataset/
│   ├── loan_approval_full.csv      # Complete dataset
│   ├── loan_approval_train.csv     # Training subset
│   └── loan_approval_test.csv      # Testing subset
├── Results/
│   └── loan_approval_model.pkl     # Trained model (generated)
├── Screenshots/
│   └── (application screenshots)
├── download_dataset.py             # Dataset generation script
├── loan_approval_app.py           # Main Streamlit application
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to the Task4 directory**:
   ```bash
   cd Task4
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset**:
   ```bash
   python download_dataset.py
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run loan_approval_app.py
   ```

## 🔧 Features

### Data Processing
- **Missing Value Handling**: Automatic imputation using median for numerical and mode for categorical features
- **Feature Engineering**: Creation of derived features like Total_Income, Income_to_Loan_Ratio
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Class Balancing**: Optional SMOTE implementation for handling imbalanced datasets

### Machine Learning Models
- **Random Forest Classifier**: Ensemble method with configurable parameters
- **Logistic Regression**: Linear classification with regularization
- **Decision Tree Classifier**: Interpretable tree-based model

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value (important for loan approval)
- **Recall**: Sensitivity (important for identifying all potential approvals)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

### Interactive Features
- **Model Configuration**: Real-time parameter tuning
- **Data Exploration**: Interactive dataset analysis
- **Individual Predictions**: Single loan application assessment
- **Feature Importance**: Understanding model decision factors
- **Results Export**: Download predictions and model performance

## 📈 Model Performance

### Evaluation Metrics
The application provides comprehensive evaluation including:

- **Classification Report**: Detailed per-class metrics
- **Confusion Matrix**: Visual representation of predictions vs actual
- **ROC Curve**: Performance across different thresholds
- **Feature Importance**: Top contributing features

### Handling Class Imbalance
- **SMOTE Integration**: Synthetic minority oversampling
- **Stratified Splitting**: Maintains class distribution in train/test splits
- **Balanced Metrics**: Focus on precision, recall, and F1-score

## 🎨 User Interface

### Dashboard Components
1. **Dataset Overview**: Key statistics and class distribution
2. **Model Configuration**: Parameter selection and training controls
3. **Performance Metrics**: Real-time evaluation results
4. **Visualizations**: Interactive charts and plots
5. **Individual Prediction**: Single application assessment tool
6. **Data Exploration**: Dataset analysis and insights

### Key Visualizations
- **Class Distribution Pie Chart**: Loan approval rates
- **Confusion Matrix Heatmap**: Prediction accuracy breakdown
- **ROC Curve**: Model performance visualization
- **Feature Importance Bar Chart**: Top predictive features

## 🔍 Usage Examples

### Training a Model
1. Select dataset (Training Data or Full Dataset)
2. Choose model type (Random Forest, Logistic Regression, Decision Tree)
3. Configure parameters (if applicable)
4. Enable SMOTE for class balancing (optional)
5. Click "Train Model"

### Making Individual Predictions
1. Train a model first
2. Navigate to "Individual Loan Prediction" section
3. Fill in applicant details
4. Click "Predict Loan Approval"
5. View prediction result with confidence score

## 📊 Technical Implementation

### Data Pipeline
```python
# Data preprocessing pipeline
1. Load dataset
2. Handle missing values
3. Encode categorical variables
4. Create engineered features
5. Scale numerical features
6. Apply SMOTE (if enabled)
7. Train model
8. Evaluate performance
```

### Model Architecture
- **Input Layer**: 14 features (11 original + 3 engineered)
- **Processing**: Preprocessing pipeline with scaling and encoding
- **Model**: Configurable classifier (RF/LR/DT)
- **Output**: Binary classification (Approved/Rejected)

## 🛠️ Technologies Used

- **Frontend**: Streamlit for interactive web application
- **Machine Learning**: scikit-learn for models and preprocessing
- **Data Processing**: pandas and numpy for data manipulation
- **Visualization**: plotly, matplotlib, and seaborn for charts
- **Class Balancing**: imbalanced-learn for SMOTE implementation
- **Model Persistence**: joblib for saving trained models

## 📝 Key Insights

### Important Features for Loan Approval
1. **Credit History**: Most significant predictor
2. **Total Income**: Combined applicant and co-applicant income
3. **Income to Loan Ratio**: Debt-to-income assessment
4. **Loan Amount**: Requested loan size
5. **Property Area**: Geographic risk factors

### Model Recommendations
- **Random Forest**: Best overall performance with feature importance
- **SMOTE**: Recommended for handling class imbalance
- **Feature Engineering**: Significantly improves model performance

## 🔮 Future Enhancements

- **Advanced Models**: XGBoost, Neural Networks
- **Feature Selection**: Automated feature importance ranking
- **Hyperparameter Tuning**: Grid search optimization
- **Model Ensemble**: Combining multiple models
- **Real-time API**: REST API for production deployment
- **A/B Testing**: Model comparison framework

## 📞 Support

For questions or issues related to this loan approval prediction system, please refer to the documentation or create an issue in the project repository.

---

**Note**: This is a demonstration project using synthetic data. For production use, ensure compliance with financial regulations and fair lending practices.
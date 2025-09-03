# Task 3: Forest Cover Type Classification

## Overview
This project implements a machine learning solution to predict forest cover types using environmental features. The application uses Random Forest and XGBoost classifiers to classify forest areas into 7 different cover types.

## Dataset
- **Source**: Covertype Dataset from UCI Machine Learning Repository
- **Samples**: 581,012 observations
- **Features**: 54 environmental features
- **Classes**: 7 forest cover types
  1. Spruce/Fir
  2. Lodgepole Pine
  3. Ponderosa Pine
  4. Cottonwood/Willow
  5. Aspen
  6. Douglas-fir
  7. Krummholz

## Features
- **Quantitative Features**: Elevation, Aspect, Slope, distances to water/roads/fire points, hillshade values
- **Categorical Features**: Wilderness areas (4 binary features) and soil types (40 binary features)

## Implementation

### Files Structure
```
Task3/
├── Dataset/
│   ├── covertype_full.csv      # Complete dataset (581K samples)
│   ├── covertype_sample.csv    # Sample dataset (50K samples)
│   ├── covertype_train.csv     # Training set (40K samples)
│   └── covertype_test.csv      # Test set (10K samples)
├── Results/                    # Model outputs and predictions
├── Screenshots/               # Application screenshots
├── download_dataset.py        # Dataset download script
├── forest_cover_app.py       # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

### Key Features
1. **Data Preprocessing**: Handles categorical features and scaling
2. **Model Training**: Random Forest classifier with hyperparameter tuning
3. **Evaluation**: Confusion matrix, classification report, feature importance
4. **Visualization**: Interactive plots using Plotly
5. **Model Persistence**: Save/load trained models

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_dataset.py
```

### 3. Run Streamlit Application
```bash
streamlit run forest_cover_app.py
```

## Model Performance
- **Algorithm**: Random Forest Classifier
- **Features**: All 54 environmental features
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrix and feature importance plots

## Key Components

### Data Preprocessing
- Feature scaling using StandardScaler
- Train/test split with stratification
- Handling of categorical binary features

### Model Training
- Random Forest with configurable parameters
- Feature importance analysis
- Cross-validation for robust evaluation

### Evaluation
- Multi-class classification metrics
- Confusion matrix visualization
- Feature importance ranking
- Model performance comparison

## Bonus Features Implemented
- **Model Comparison**: Random Forest vs XGBoost
- **Hyperparameter Tuning**: Interactive parameter selection
- **Feature Analysis**: Detailed feature importance visualization
- **Data Exploration**: Statistical summaries and distributions

## Results
The application provides:
- Interactive confusion matrix
- Feature importance rankings
- Classification performance metrics
- Downloadable prediction results
- Model persistence for future use

## Technologies Used
- **Frontend**: Streamlit
- **ML Libraries**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib
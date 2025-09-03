# Sales Forecasting System

A comprehensive machine learning application for predicting retail sales using historical data, built with Streamlit and scikit-learn.

## 🎯 Project Overview

This project implements a sales forecasting system that predicts future sales based on historical data, store information, and various external factors. The system uses multiple machine learning algorithms to provide accurate sales predictions and comprehensive data analysis.

## 📊 Dataset Information

- **Dataset**: Walmart-style synthetic sales data
- **Total Records**: 421,570 sales records
- **Time Period**: 2010-2012 (143 weeks)
- **Stores**: 45 stores across 3 types (A, B, C)
- **Departments**: 99 departments per store
- **Features**: 13 features including sales, store info, holidays, markdowns, and economic indicators

## 🏗️ Project Structure

```
Task7/
├── Dataset/
│   ├── sales_data.csv           # Raw sales data
│   ├── stores_data.csv          # Store information
│   ├── features_data.csv        # Economic indicators
│   ├── walmart_sales_full.csv   # Complete merged dataset
│   ├── walmart_sales_train.csv  # Training data (80%)
│   └── walmart_sales_test.csv   # Testing data (20%)
├── Results/                     # Model outputs and predictions
├── Screenshots/                 # Application screenshots
├── download_dataset.py          # Dataset generation script
├── sales_forecasting_app.py     # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Task7
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset**:
   ```bash
   python download_dataset.py
   ```

4. **Run the application**:
   ```bash
   streamlit run sales_forecasting_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## ✨ Features

### Machine Learning Models
- **Random Forest Regressor**: Ensemble method for robust predictions
- **Gradient Boosting Regressor**: Advanced boosting algorithm
- **Ridge Regression**: Linear regression with L2 regularization
- **Lasso Regression**: Linear regression with L1 regularization
- **Elastic Net**: Combined L1 and L2 regularization
- **Support Vector Regression (SVR)**: Non-linear regression
- **Neural Network (MLP)**: Multi-layer perceptron regressor
- **Decision Tree**: Simple tree-based regression

### Data Preprocessing
- **Feature Scaling**: StandardScaler normalization
- **Feature Selection**: SelectKBest with f_regression
- **Missing Value Handling**: Automatic imputation
- **Data Validation**: Comprehensive data quality checks

### Analysis & Visualization
- **Sales Trends**: Time series analysis and patterns
- **Store Performance**: Sales by store type and individual stores
- **Seasonal Patterns**: Monthly and weekly sales patterns
- **Holiday Impact**: Sales analysis during holiday periods
- **Feature Importance**: Model-based feature ranking
- **Correlation Analysis**: Feature relationship heatmaps

### Interactive Features
- **Model Configuration**: Hyperparameter tuning interface
- **Individual Predictions**: Single sales prediction tool
- **Model Comparison**: Performance metrics comparison
- **Data Exploration**: Interactive data filtering and analysis
- **Export Results**: Download predictions and model metrics

### Performance Evaluation
- **Mean Absolute Error (MAE)**: Average prediction error
- **Mean Squared Error (MSE)**: Squared error metric
- **Root Mean Squared Error (RMSE)**: Standard deviation of errors
- **R² Score**: Coefficient of determination
- **Cross-Validation**: K-fold validation scores

## 🖥️ User Interface

The application features an intuitive Streamlit interface with:

1. **Dataset Overview**: Summary statistics and data preview
2. **Data Analysis**: Interactive visualizations and insights
3. **Model Training**: Algorithm selection and hyperparameter tuning
4. **Predictions**: Individual and batch prediction tools
5. **Model Evaluation**: Performance metrics and comparison
6. **Data Exploration**: Advanced filtering and analysis tools

## 📈 Usage Examples

### Basic Sales Prediction
```python
# Load the application
streamlit run sales_forecasting_app.py

# 1. Select Random Forest model
# 2. Configure parameters (n_estimators=100, max_depth=10)
# 3. Train the model
# 4. View performance metrics
# 5. Make predictions
```

### Advanced Analysis
```python
# Analyze seasonal patterns
# 1. Navigate to Data Analysis section
# 2. View seasonal sales patterns
# 3. Analyze holiday impact
# 4. Compare store performance
```

### Model Comparison
```python
# Compare multiple models
# 1. Train different algorithms
# 2. View model comparison chart
# 3. Select best performing model
# 4. Export results
```

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Data Loading**: CSV file processing with pandas
2. **Feature Engineering**: Time-based feature creation
3. **Data Cleaning**: Missing value imputation and outlier handling
4. **Feature Scaling**: Standardization for model compatibility
5. **Train-Test Split**: Time-based data splitting

### Model Training Pipeline
1. **Algorithm Selection**: Choose from 8 regression models
2. **Hyperparameter Tuning**: GridSearchCV optimization
3. **Cross-Validation**: K-fold validation for robust evaluation
4. **Model Evaluation**: Multiple performance metrics
5. **Prediction Generation**: Batch and individual predictions

### Performance Metrics
- **MAE**: Measures average absolute prediction error
- **MSE**: Penalizes larger errors more heavily
- **RMSE**: Standard deviation of prediction errors
- **R²**: Explains variance in the target variable
- **Cross-Val Score**: Average performance across folds

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Persistence**: Joblib
- **Data Generation**: Synthetic data creation

## 📊 Key Insights

### Sales Patterns
- **Seasonal Trends**: Higher sales during holiday periods
- **Store Performance**: Type A stores outperform B and C
- **Department Variation**: Significant sales differences across departments
- **Economic Impact**: Fuel prices and unemployment affect sales

### Model Performance
- **Best Algorithm**: Random Forest typically performs best
- **Feature Importance**: Store type and department are key predictors
- **Prediction Accuracy**: RMSE typically under 10% of mean sales
- **Seasonal Accuracy**: Better predictions during stable periods

## 🚀 Future Enhancements

### Advanced Features
- **Deep Learning Models**: LSTM and GRU for time series
- **External Data Integration**: Weather, demographics, competition
- **Real-time Predictions**: Live data streaming and updates
- **Automated Retraining**: Model updates with new data

### UI/UX Improvements
- **Advanced Filtering**: Multi-dimensional data filtering
- **Custom Dashboards**: User-configurable analytics views
- **Mobile Optimization**: Responsive design for mobile devices
- **Export Options**: PDF reports and Excel exports

### Performance Optimization
- **Caching**: Model and data caching for faster responses
- **Parallel Processing**: Multi-threaded model training
- **Database Integration**: Direct database connectivity
- **API Development**: RESTful API for external integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Walmart for inspiring the dataset structure
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- Open source community for continuous inspiration

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

---

**Built with ❤️ for accurate sales forecasting and business intelligence**
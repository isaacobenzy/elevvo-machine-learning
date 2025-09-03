# Traffic Sign Recognition System 🚦

A comprehensive machine learning application for classifying traffic signs using computer vision features and various ML algorithms.

## Project Overview

This project implements a traffic sign recognition system that uses extracted image features to classify different types of traffic signs. The system employs multiple machine learning algorithms and provides an interactive Streamlit interface for data analysis, model training, and predictions.

## Dataset Information

### Synthetic GTSRB-Style Dataset
- **Total Samples**: 5,000 traffic sign samples
- **Classes**: 43 different traffic sign types
- **Features**: 59 extracted image features including:
  - Color histograms (RGB channels)
  - HOG (Histogram of Oriented Gradients) features
  - Edge detection features
  - Texture analysis (LBP - Local Binary Patterns)
  - Geometric properties (area, perimeter, circularity, aspect ratio)
  - Keypoint features (SIFT-based)

### Feature Categories
1. **Color Features**: RGB histogram bins for color distribution analysis
2. **Shape Features**: HOG descriptors capturing edge orientations
3. **Edge Features**: Canny edge detection statistics
4. **Texture Features**: Local Binary Pattern analysis
5. **Geometric Features**: Basic shape measurements
6. **Keypoint Features**: SIFT keypoint density and distribution

## Project Structure

```
Task8/
├── Dataset/
│   ├── gtsrb_features_full.csv      # Complete dataset
│   ├── gtsrb_features_train.csv     # Training subset
│   ├── gtsrb_features_test.csv      # Test subset
│   ├── feature_names.txt            # List of feature names
│   └── class_names.txt              # List of class names
├── Results/                         # Model outputs and results
├── Screenshots/                     # Application screenshots
├── download_dataset.py              # Dataset generation script
├── traffic_sign_app.py             # Main Streamlit application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd Task8
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset** (if not already done):
   ```bash
   python download_dataset.py
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run traffic_sign_app.py
   ```

5. **Open your browser** and navigate to the provided URL (typically `http://localhost:8501`)

## Features

### 🤖 Machine Learning Models
- **Random Forest**: Ensemble method for robust classification
- **Gradient Boosting**: Advanced boosting algorithm
- **Support Vector Machine**: Kernel-based classification
- **Neural Network**: Multi-layer perceptron
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier
- **Decision Tree**: Interpretable tree-based model
- **Logistic Regression**: Linear classification model

### 📊 Data Preprocessing
- **Feature Scaling**: StandardScaler normalization
- **Feature Selection**: SelectKBest with f_classif
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Missing Value Handling**: Mean imputation

### 📈 Analysis & Visualization
- **Class Distribution**: Traffic sign frequency analysis
- **Feature Correlation**: Heatmap visualization
- **Feature Distributions**: Class-wise feature analysis
- **PCA Analysis**: Dimensionality reduction visualization
- **Confusion Matrix**: Classification performance matrix
- **Feature Importance**: Model interpretability

### 🎯 Interactive Elements
- **Model Comparison**: Performance metrics comparison
- **Single Predictions**: Individual traffic sign classification
- **Data Filtering**: Custom dataset exploration
- **Parameter Tuning**: Preprocessing and training options

### 📋 Performance Evaluation
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision scores
- **Recall**: Class-wise recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## User Interface

The application features six main tabs:

1. **📊 Dataset Overview**: Basic statistics and data preview
2. **📈 Data Analysis**: Visualizations and exploratory analysis
3. **🤖 Model Training**: Configure and train ML models
4. **🎯 Predictions**: Make individual traffic sign predictions
5. **📋 Model Evaluation**: Compare model performance
6. **🔍 Data Exploration**: Filter and explore the dataset

## Usage Examples

### Training Models
1. Navigate to the "Model Training" tab
2. Configure preprocessing options (feature selection, PCA)
3. Set training parameters (test size, grid search)
4. Click "Train Models" to train all algorithms
5. View trained models and their status

### Making Predictions
1. Go to the "Predictions" tab
2. Select a trained model
3. Enter traffic sign feature values
4. Click "Predict Traffic Sign" to get classification
5. View prediction probabilities and confidence scores

### Analyzing Performance
1. Visit the "Model Evaluation" tab
2. Compare model performance metrics
3. Examine confusion matrices
4. Analyze feature importance
5. Identify the best performing model

## Technical Implementation

### Data Processing Pipeline
1. **Feature Extraction**: Computer vision features from synthetic images
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Scale and select relevant features
4. **Train-Test Split**: Stratified splitting for balanced evaluation

### Model Training Pipeline
1. **Preprocessing**: Apply scaling, selection, and PCA
2. **Model Training**: Train multiple algorithms simultaneously
3. **Hyperparameter Tuning**: Optional GridSearchCV optimization
4. **Cross-Validation**: Robust performance estimation

### Performance Metrics
- **Multi-class Classification**: Weighted averages for imbalanced classes
- **Confusion Matrix**: Detailed per-class performance
- **Feature Importance**: Model interpretability analysis
- **ROC Analysis**: Classification threshold optimization

## Technologies Used

- **Frontend**: Streamlit for interactive web interface
- **Machine Learning**: Scikit-learn for ML algorithms
- **Data Processing**: Pandas and NumPy for data manipulation
- **Visualization**: Plotly, Matplotlib, and Seaborn
- **Computer Vision**: OpenCV for image processing
- **Scientific Computing**: SciPy for statistical functions

## Key Insights

### Traffic Sign Classification Challenges
- **Feature Importance**: Color and shape features are most discriminative
- **Class Imbalance**: Some traffic signs are more common than others
- **Feature Correlation**: Certain features show high correlation
- **Model Performance**: Ensemble methods generally perform best

### Best Practices
- **Feature Selection**: Reduces overfitting and improves performance
- **Cross-Validation**: Ensures robust model evaluation
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Preprocessing**: Proper scaling is crucial for many algorithms

## Future Enhancements

### Technical Improvements
- **Deep Learning**: Implement CNN models for raw image processing
- **Real-time Processing**: Add webcam integration for live classification
- **Model Deployment**: Create REST API for production use
- **Advanced Features**: Add YOLO for object detection

### User Experience
- **Batch Predictions**: Process multiple images simultaneously
- **Model Export**: Save and load trained models
- **Custom Datasets**: Upload and train on user datasets
- **Mobile App**: Develop mobile application interface

### Performance Optimization
- **GPU Acceleration**: Utilize CUDA for faster training
- **Distributed Computing**: Scale to larger datasets
- **Model Compression**: Optimize models for edge deployment
- **Caching**: Implement result caching for faster responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **GTSRB Dataset**: Inspiration from the German Traffic Sign Recognition Benchmark
- **Scikit-learn**: Comprehensive machine learning library
- **Streamlit**: Excellent framework for ML applications
- **OpenCV**: Computer vision and image processing capabilities

## Support

If you encounter any issues or have questions:

1. Check the troubleshooting section in this README
2. Review the application logs for error messages
3. Ensure all dependencies are properly installed
4. Verify that the dataset has been generated correctly

### Common Issues

- **Import Errors**: Run `pip install -r requirements.txt`
- **Dataset Missing**: Execute `python download_dataset.py`
- **Memory Issues**: Reduce dataset size or use feature selection
- **Slow Performance**: Enable feature selection and disable grid search

---

**Built with ❤️ using Streamlit and Scikit-learn**
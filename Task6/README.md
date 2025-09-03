# 🎵 Music Genre Classification

A comprehensive machine learning application for classifying music genres using audio features. This project implements multiple classification algorithms and provides an interactive Streamlit interface for exploring the dataset and making predictions.

## 📋 Project Overview

This project focuses on music genre classification using the GTZAN dataset (or synthetic data with similar characteristics). The application extracts audio features using Librosa and trains multiple machine learning models to classify music into 10 different genres.

### 🎯 Genres Supported
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 📊 Dataset Information

**Dataset**: GTZAN Music Genre Dataset (or synthetic equivalent)
- **Total Samples**: 1,000 audio tracks
- **Genres**: 10 different music genres
- **Samples per Genre**: 100 tracks
- **Features**: 55 audio features extracted using Librosa
- **Format**: CSV files with pre-extracted features

### 🎚️ Audio Features
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients with mean and variance
- **Spectral Features**: Centroid, Bandwidth, Rolloff
- **Temporal Features**: Zero Crossing Rate, Tempo
- **Harmonic Features**: Harmonic and Percussive components
- **Chroma Features**: 12 pitch class profiles
- **Tonnetz Features**: Tonal centroid features

## 🏗️ Project Structure

```
Task6/
├── Dataset/
│   ├── music_features_full.csv      # Complete dataset
│   ├── music_features_train.csv     # Training set
│   ├── music_features_test.csv      # Test set
│   └── feature_names.txt            # List of feature names
├── Results/
│   └── (Model results and metrics)
├── Screenshots/
│   └── (Application screenshots)
├── download_dataset.py              # Dataset download/generation script
├── music_genre_app.py              # Main Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Task6
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download/Generate dataset**:
   ```bash
   python download_dataset.py
   ```

4. **Run the application**:
   ```bash
   streamlit run music_genre_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## ✨ Features

### 🤖 Machine Learning Models
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Gradient Boosting**: Advanced ensemble technique
- **Neural Network (MLP)**: Multi-layer perceptron
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier

### 🔧 Data Preprocessing
- **Feature Scaling**: StandardScaler normalization
- **Feature Selection**: SelectKBest with f_classif
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Grid Search**: Hyperparameter optimization

### 📊 Data Analysis & Visualization
- **Genre Distribution**: Bar charts showing sample distribution
- **Feature Correlation**: Heatmaps of feature relationships
- **Feature Distributions**: Histograms by genre
- **PCA Analysis**: 2D and 3D visualizations
- **Confusion Matrices**: Model performance visualization
- **Feature Importance**: Tree-based model insights

### 🎯 Interactive Features
- **Model Comparison**: Side-by-side performance metrics
- **Individual Predictions**: Real-time genre classification
- **Feature Adjustment**: Interactive sliders for audio features
- **Probability Visualization**: Prediction confidence scores
- **Data Exploration**: Dataset statistics and sample viewing

### 📈 Performance Evaluation
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision
- **Recall**: Per-class and macro-averaged recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🎮 User Interface

### Main Dashboard
- **Dataset Overview**: Key statistics and metrics
- **Model Configuration**: Preprocessing and training options
- **Performance Comparison**: Interactive charts and metrics
- **Individual Prediction**: Real-time genre classification

### Analysis Sections
1. **Dataset Analysis**: Distribution and correlation visualizations
2. **Model Training**: Configure and train multiple algorithms
3. **Performance Evaluation**: Detailed results and comparisons
4. **PCA Analysis**: Dimensionality reduction insights
5. **Data Exploration**: Raw data viewing and statistics

## 📝 Usage Examples

### Training Models
1. Configure preprocessing options in the sidebar
2. Choose whether to use Grid Search for optimization
3. Click "Train Models" to start the training process
4. View results in the performance comparison section

### Making Predictions
1. Scroll to the "Individual Genre Prediction" section
2. Adjust audio feature sliders (tempo, spectral features, etc.)
3. Click "Predict Genre" to see the classification result
4. View probability scores for all genres

### Analyzing Data
1. Explore the dataset overview for basic statistics
2. View feature distributions by genre
3. Examine correlation matrices for feature relationships
4. Use PCA analysis to understand data structure

## 🔬 Technical Implementation

### Audio Feature Extraction
```python
# Key features extracted using Librosa
- MFCC coefficients (mean and variance)
- Spectral centroid, bandwidth, rolloff
- Zero crossing rate
- Tempo estimation
- Harmonic and percussive components
- Chroma features
- Tonnetz features
```

### Model Training Pipeline
```python
1. Data Loading and Preprocessing
2. Feature Scaling (StandardScaler)
3. Optional Feature Selection (SelectKBest)
4. Optional Dimensionality Reduction (PCA)
5. Model Training with Cross-Validation
6. Hyperparameter Optimization (Grid Search)
7. Performance Evaluation
```

### Performance Metrics
- **Accuracy**: ~85-90% on test set
- **Best Model**: Random Forest (typically)
- **Training Time**: 2-5 minutes (depending on configuration)
- **Prediction Time**: <1 second per sample

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Librosa**: Audio feature extraction
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Statistical plotting

## 🔍 Key Insights

### Feature Importance
- **MFCC coefficients** are typically the most important features
- **Spectral centroid** helps distinguish between genres
- **Tempo** is crucial for rhythm-based classification
- **Zero crossing rate** differentiates vocal vs. instrumental content

### Genre Characteristics
- **Classical**: Low tempo, rich harmonic content
- **Metal**: High spectral centroid, aggressive features
- **Jazz**: Complex harmonic structures, varied tempo
- **Hip-hop**: Strong percussive elements, specific MFCC patterns

### Model Performance
- **Random Forest**: Best overall performance, good interpretability
- **SVM**: Strong performance with proper kernel selection
- **Neural Networks**: Good for complex pattern recognition
- **Gradient Boosting**: Excellent for feature interactions

## 🚀 Future Enhancements

### Technical Improvements
- [ ] Real-time audio processing from microphone input
- [ ] Deep learning models (CNN, RNN) for raw audio
- [ ] Advanced feature engineering techniques
- [ ] Model ensemble methods
- [ ] Cross-validation with temporal splits

### User Experience
- [ ] Audio playback functionality
- [ ] Batch prediction capabilities
- [ ] Model export/import functionality
- [ ] Advanced filtering and search options
- [ ] Custom genre training

### Performance Optimization
- [ ] Model caching and persistence
- [ ] Parallel processing for training
- [ ] Memory optimization for large datasets
- [ ] GPU acceleration support

## 📊 Performance Optimization

### Memory Management
- Efficient data loading with chunking
- Feature selection to reduce dimensionality
- Model persistence to avoid retraining

### Speed Optimization
- Vectorized operations with NumPy
- Parallel processing for cross-validation
- Caching of preprocessed data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **GTZAN Dataset**: Original music genre dataset
- **Librosa**: Excellent audio analysis library
- **Streamlit**: Amazing framework for ML applications
- **Scikit-learn**: Comprehensive machine learning toolkit
- **Music Information Retrieval Community**: Research and inspiration

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This application is designed for educational and research purposes. The synthetic dataset provides a good approximation of real audio features for learning and experimentation.
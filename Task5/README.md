# Movie Recommendation System 🎬

A comprehensive movie recommendation system built with Streamlit that implements multiple recommendation algorithms using the MovieLens 100K dataset.

## Project Overview

This project implements various recommendation algorithms including:
- **User-based Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Item-based Collaborative Filtering**: Recommends movies similar to those the user has liked
- **Matrix Factorization (SVD)**: Uses dimensionality reduction for recommendations
- **Content-based Filtering**: Recommends movies based on genre similarity
- **Hybrid Approach**: Combines multiple methods for better recommendations

## Dataset Information

**MovieLens 100K Dataset**
- **Source**: GroupLens Research (University of Minnesota)
- **Size**: 100,000 ratings from 943 users on 1,682 movies
- **Rating Scale**: 1-5 stars
- **Features**: User demographics, movie genres, timestamps
- **Sparsity**: ~93.7% (highly sparse matrix)

### Dataset Structure
- `movielens_ratings.csv`: User ratings data
- `movielens_movies.csv`: Movie information with genres
- `movielens_users.csv`: User demographic information
- `movielens_full.csv`: Complete merged dataset
- `user_item_matrix.csv`: Pivot table for collaborative filtering

## Project Structure

```
Task5/
├── Dataset/
│   ├── movielens_ratings.csv
│   ├── movielens_movies.csv
│   ├── movielens_users.csv
│   ├── movielens_full.csv
│   ├── movielens_train.csv
│   ├── movielens_test.csv
│   └── user_item_matrix.csv
├── Results/
├── Screenshots/
├── download_dataset.py
├── movie_recommendation_app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   ```bash
   python download_dataset.py
   ```

3. **Run the Application**
   ```bash
   streamlit run movie_recommendation_app.py
   ```

4. **Access the App**
   - Open your browser and go to `http://localhost:8501`

## Features

### 🎯 Recommendation Algorithms
- **User-based CF**: Finds similar users and recommends their favorite movies
- **Item-based CF**: Recommends movies similar to user's previously liked items
- **SVD Matrix Factorization**: Uses latent factors for recommendations
- **Content-based**: Recommends based on movie genre similarity
- **Hybrid**: Combines multiple approaches with weighted scoring

### 📊 Data Analysis & Visualization
- **Rating Distribution**: Histogram of user ratings
- **Genre Analysis**: Most popular movie genres
- **User Activity**: Distribution of user engagement
- **Movie Popularity**: Scatter plot of rating count vs average rating
- **Sparsity Analysis**: Matrix density statistics

### 🔧 Interactive Features
- **User Selection**: Choose any user from the dataset
- **Algorithm Comparison**: Switch between different recommendation methods
- **Customizable Results**: Adjust number of recommendations (5-20)
- **User History**: View user's past ratings and preferences
- **Popular Movies**: Discover trending and highly-rated movies

### 📈 Performance Evaluation
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant items in top-K
- **F1-Score**: Harmonic mean of precision and recall
- **Comparative Analysis**: Side-by-side algorithm performance

## Model Performance

### Evaluation Metrics
- **Precision@10**: ~0.15-0.25 (varies by algorithm)
- **Recall@10**: ~0.08-0.18 (varies by algorithm)
- **F1-Score**: ~0.10-0.20 (varies by algorithm)

### Algorithm Comparison
1. **Item-based CF**: Generally performs best for precision
2. **User-based CF**: Good for discovering diverse content
3. **SVD**: Efficient for large-scale recommendations
4. **Content-based**: Excellent for new users (cold start)
5. **Hybrid**: Balanced performance across metrics

## User Interface

### Main Sections
1. **Dataset Overview**: Key statistics and metrics
2. **Data Analysis**: Interactive visualizations
3. **Recommendation Settings**: Algorithm and parameter selection
4. **User Selection**: Choose target user for recommendations
5. **Personalized Recommendations**: Generated movie suggestions
6. **Popular Movies**: Trending and highly-rated content
7. **Model Evaluation**: Performance metrics and comparisons
8. **Data Exploration**: Raw data inspection

### Key Visualizations
- Rating distribution histogram
- Top movie genres bar chart
- User activity distribution
- Movie popularity scatter plot
- Algorithm performance comparison

## Usage Examples

### Generate Recommendations
1. Select a user ID from the dropdown
2. Choose recommendation algorithm
3. Set number of recommendations
4. Click "Generate Recommendations"
5. View personalized movie suggestions with scores

### Evaluate Performance
1. Navigate to "Model Evaluation" section
2. Click "Evaluate Recommendation System"
3. Compare Precision@K, Recall@K, and F1-scores
4. Identify best-performing algorithm

### Explore Data
1. View dataset statistics in overview section
2. Analyze rating patterns and user behavior
3. Discover popular genres and movies
4. Examine data sparsity and distribution

## Technical Implementation

### Collaborative Filtering
- **Similarity Metrics**: Cosine similarity for user/item comparisons
- **Neighborhood Selection**: Top-50 similar users/items
- **Prediction**: Weighted average of neighbor ratings

### Matrix Factorization
- **Algorithm**: Truncated SVD (Singular Value Decomposition)
- **Components**: 50 latent factors
- **Optimization**: Scikit-learn's efficient implementation

### Content-based Filtering
- **Feature Extraction**: TF-IDF on movie genres
- **Similarity**: Cosine similarity between genre vectors
- **Recommendation**: Based on user's highly-rated movie genres

### Hybrid Approach
- **Combination**: Weighted average of multiple algorithms
- **Weights**: 30% user-based, 40% item-based, 30% SVD
- **Normalization**: Score averaging across available methods

## Technologies Used

- **Frontend**: Streamlit for interactive web interface
- **Data Processing**: Pandas, NumPy for data manipulation
- **Machine Learning**: Scikit-learn for algorithms and metrics
- **Visualization**: Plotly, Matplotlib, Seaborn for charts
- **Similarity Computation**: Cosine similarity, TF-IDF vectorization
- **Matrix Operations**: Scipy for sparse matrix handling

## Key Insights

### Dataset Characteristics
- High sparsity (~93.7%) typical of recommendation systems
- Rating bias toward higher values (3-5 stars)
- Drama and Comedy are most popular genres
- Power-law distribution in user activity

### Algorithm Performance
- Item-based CF often outperforms user-based CF
- SVD provides good scalability for large datasets
- Content-based helps with cold start problems
- Hybrid approaches balance different algorithm strengths

### Recommendation Quality
- Precision@K values are typical for movie recommendation
- Recall@K indicates good coverage of user preferences
- F1-scores show balanced precision-recall trade-off

## Future Enhancements

### Algorithm Improvements
- **Deep Learning**: Neural collaborative filtering
- **Advanced Factorization**: Non-negative matrix factorization
- **Temporal Dynamics**: Time-aware recommendations
- **Implicit Feedback**: Incorporate viewing behavior

### Feature Additions
- **Real-time Updates**: Dynamic model retraining
- **Explanation Interface**: Why recommendations were made
- **A/B Testing**: Compare algorithm performance
- **Social Features**: Friend-based recommendations

### Technical Enhancements
- **Scalability**: Distributed computing for large datasets
- **Caching**: Redis for faster recommendation serving
- **API Development**: RESTful service for recommendations
- **Mobile Interface**: Responsive design for mobile devices

## Performance Optimization

### Computational Efficiency
- Sparse matrix operations for memory efficiency
- Precomputed similarity matrices for faster recommendations
- Batch processing for multiple user recommendations
- Efficient data structures for large-scale operations

### User Experience
- Progressive loading for large datasets
- Caching of computed similarities
- Responsive design for different screen sizes
- Interactive visualizations with Plotly

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Algorithm improvements
- Documentation updates

## License

This project is for educational purposes. The MovieLens dataset is provided by GroupLens Research.

## Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **Streamlit** for the excellent web framework
- **Scikit-learn** for machine learning algorithms
- **Plotly** for interactive visualizations

---

**Note**: This recommendation system is designed for educational and demonstration purposes. For production use, consider additional factors like scalability, real-time updates, and advanced evaluation metrics.
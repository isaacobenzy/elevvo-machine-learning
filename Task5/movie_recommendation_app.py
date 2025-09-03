import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #E50914;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.movie-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 0.5rem 2rem;
    font-weight: bold;
}
.recommendation-card {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class MovieRecommendationSystem:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.tfidf_matrix = None
        self.content_similarity = None
    
    def load_data(self):
        """Load all datasets"""
        try:
            self.ratings_df = pd.read_csv('Dataset/movielens_ratings.csv')
            self.movies_df = pd.read_csv('Dataset/movielens_movies.csv')
            self.users_df = pd.read_csv('Dataset/movielens_users.csv')
            
            # Load user-item matrix
            self.user_item_matrix = pd.read_csv('Dataset/user_item_matrix.csv', index_col=0)
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def compute_user_similarity(self):
        """Compute user-based collaborative filtering similarity"""
        with st.spinner('Computing user similarity matrix...'):
            # Use cosine similarity for user-based CF
            self.user_similarity = cosine_similarity(self.user_item_matrix)
            self.user_similarity = pd.DataFrame(
                self.user_similarity,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
    
    def compute_item_similarity(self):
        """Compute item-based collaborative filtering similarity"""
        with st.spinner('Computing item similarity matrix...'):
            # Transpose for item-based CF
            item_matrix = self.user_item_matrix.T
            self.item_similarity = cosine_similarity(item_matrix)
            self.item_similarity = pd.DataFrame(
                self.item_similarity,
                index=item_matrix.index,
                columns=item_matrix.index
            )
    
    def compute_content_similarity(self):
        """Compute content-based similarity using movie genres"""
        with st.spinner('Computing content-based similarity...'):
            # Create TF-IDF matrix from genres
            tfidf = TfidfVectorizer(stop_words='english')
            
            # Prepare genre text for TF-IDF
            genre_text = self.movies_df['genres'].fillna('').str.replace('|', ' ')
            self.tfidf_matrix = tfidf.fit_transform(genre_text)
            
            # Compute cosine similarity
            self.content_similarity = cosine_similarity(self.tfidf_matrix)
            self.content_similarity = pd.DataFrame(
                self.content_similarity,
                index=self.movies_df['movie_id'],
                columns=self.movies_df['movie_id']
            )
    
    def train_svd_model(self, n_components=50):
        """Train SVD model for matrix factorization"""
        with st.spinner('Training SVD model...'):
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Fit SVD on user-item matrix
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            item_factors = self.svd_model.components_
            
            # Reconstruct the matrix
            self.predicted_ratings = np.dot(user_factors, item_factors)
            self.predicted_ratings = pd.DataFrame(
                self.predicted_ratings,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.columns
            )
    
    def get_user_recommendations(self, user_id, method='user_based', n_recommendations=10):
        """Get recommendations for a specific user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        if method == 'user_based':
            return self._user_based_recommendations(user_id, unrated_movies, n_recommendations)
        elif method == 'item_based':
            return self._item_based_recommendations(user_id, unrated_movies, n_recommendations)
        elif method == 'svd':
            return self._svd_recommendations(user_id, unrated_movies, n_recommendations)
        elif method == 'content_based':
            return self._content_based_recommendations(user_id, unrated_movies, n_recommendations)
        else:
            return self._hybrid_recommendations(user_id, unrated_movies, n_recommendations)
    
    def _user_based_recommendations(self, user_id, unrated_movies, n_recommendations):
        """User-based collaborative filtering recommendations"""
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        # Get similar users
        similar_users = self.user_similarity.loc[user_id].sort_values(ascending=False)[1:51]  # Top 50 similar users
        
        recommendations = {}
        for movie_id in unrated_movies:
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user, similarity in similar_users.items():
                if self.user_item_matrix.loc[similar_user, movie_id] > 0:
                    weighted_sum += similarity * self.user_item_matrix.loc[similar_user, movie_id]
                    similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie_id] = predicted_rating
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def _item_based_recommendations(self, user_id, unrated_movies, n_recommendations):
        """Item-based collaborative filtering recommendations"""
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        recommendations = {}
        for movie_id in unrated_movies:
            if movie_id in self.item_similarity.index:
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_movie, rating in rated_movies.items():
                    if rated_movie in self.item_similarity.columns:
                        similarity = self.item_similarity.loc[movie_id, rated_movie]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[movie_id] = predicted_rating
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def _svd_recommendations(self, user_id, unrated_movies, n_recommendations):
        """SVD-based recommendations"""
        if self.svd_model is None:
            self.train_svd_model()
        
        user_predictions = self.predicted_ratings.loc[user_id]
        recommendations = [(movie_id, user_predictions[movie_id]) for movie_id in unrated_movies]
        
        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def _content_based_recommendations(self, user_id, unrated_movies, n_recommendations):
        """Content-based recommendations"""
        if self.content_similarity is None:
            self.compute_content_similarity()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        # Get user's preferred genres based on highly rated movies
        high_rated_movies = rated_movies[rated_movies >= 4].index
        
        recommendations = {}
        for movie_id in unrated_movies:
            if movie_id in self.content_similarity.index:
                similarity_scores = []
                for rated_movie in high_rated_movies:
                    if rated_movie in self.content_similarity.columns:
                        similarity_scores.append(self.content_similarity.loc[movie_id, rated_movie])
                
                if similarity_scores:
                    avg_similarity = np.mean(similarity_scores)
                    recommendations[movie_id] = avg_similarity
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def _hybrid_recommendations(self, user_id, unrated_movies, n_recommendations):
        """Hybrid recommendations combining multiple methods"""
        # Get recommendations from different methods
        user_based = dict(self._user_based_recommendations(user_id, unrated_movies, n_recommendations * 2))
        item_based = dict(self._item_based_recommendations(user_id, unrated_movies, n_recommendations * 2))
        svd_based = dict(self._svd_recommendations(user_id, unrated_movies, n_recommendations * 2))
        
        # Combine scores with weights
        hybrid_scores = {}
        all_movies = set(user_based.keys()) | set(item_based.keys()) | set(svd_based.keys())
        
        for movie_id in all_movies:
            score = 0
            count = 0
            
            if movie_id in user_based:
                score += 0.3 * user_based[movie_id]  # 30% weight
                count += 1
            if movie_id in item_based:
                score += 0.4 * item_based[movie_id]  # 40% weight
                count += 1
            if movie_id in svd_based:
                score += 0.3 * svd_based[movie_id]  # 30% weight
                count += 1
            
            if count > 0:
                hybrid_scores[movie_id] = score / count
        
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def get_popular_movies(self, n_movies=10):
        """Get most popular movies based on rating count and average rating"""
        movie_stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['rating_count', 'avg_rating']
        
        # Filter movies with at least 50 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 50]
        
        # Calculate popularity score (weighted average)
        popular_movies['popularity_score'] = (
            popular_movies['avg_rating'] * 0.7 + 
            (popular_movies['rating_count'] / popular_movies['rating_count'].max()) * 5 * 0.3
        )
        
        top_movies = popular_movies.sort_values('popularity_score', ascending=False).head(n_movies)
        return top_movies.index.tolist()
    
    def get_movie_details(self, movie_ids):
        """Get movie details for given movie IDs"""
        return self.movies_df[self.movies_df['movie_id'].isin(movie_ids)]
    
    def evaluate_recommendations(self, test_data, method='user_based', n_recommendations=10):
        """Evaluate recommendation system using Precision@K and Recall@K"""
        precisions = []
        recalls = []
        
        # Sample users for evaluation (to speed up)
        test_users = test_data['user_id'].unique()[:100]  # Evaluate on 100 users
        
        for user_id in test_users:
            if user_id not in self.user_item_matrix.index:
                continue
            
            # Get user's test ratings
            user_test = test_data[test_data['user_id'] == user_id]
            relevant_items = set(user_test[user_test['rating'] >= 4]['movie_id'].tolist())
            
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            recommendations = self.get_user_recommendations(user_id, method, n_recommendations)
            recommended_items = set([movie_id for movie_id, _ in recommendations])
            
            # Calculate precision and recall
            if len(recommended_items) > 0:
                precision = len(relevant_items & recommended_items) / len(recommended_items)
                recall = len(relevant_items & recommended_items) / len(relevant_items)
                
                precisions.append(precision)
                recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall,
            'f1_score': f1_score,
            'num_users_evaluated': len(precisions)
        }

def plot_rating_distribution(ratings_df):
    """Plot rating distribution"""
    fig = px.histogram(
        ratings_df, x='rating',
        title='Rating Distribution',
        labels={'rating': 'Rating', 'count': 'Number of Ratings'},
        color_discrete_sequence=['#E50914']
    )
    fig.update_layout(height=400)
    return fig

def plot_top_genres(movies_df):
    """Plot top movie genres"""
    # Extract all genres
    all_genres = []
    for genres_str in movies_df['genres']:
        if pd.notna(genres_str) and genres_str != 'unknown':
            all_genres.extend(genres_str.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title='Top 10 Movie Genres',
        labels={'x': 'Number of Movies', 'y': 'Genre'},
        color_discrete_sequence=['#E50914']
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_user_activity(ratings_df):
    """Plot user activity distribution"""
    user_activity = ratings_df.groupby('user_id').size()
    
    fig = px.histogram(
        x=user_activity.values,
        title='User Activity Distribution',
        labels={'x': 'Number of Ratings per User', 'y': 'Number of Users'},
        color_discrete_sequence=['#E50914']
    )
    fig.update_layout(height=400)
    return fig

def plot_movie_popularity(ratings_df, movies_df):
    """Plot movie popularity"""
    movie_stats = ratings_df.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)
    movie_stats.columns = ['rating_count', 'avg_rating']
    
    # Merge with movie titles
    movie_stats = movie_stats.merge(movies_df[['movie_id', 'title']], left_index=True, right_on='movie_id')
    
    # Filter movies with at least 100 ratings
    popular_movies = movie_stats[movie_stats['rating_count'] >= 100].head(15)
    
    fig = px.scatter(
        popular_movies,
        x='rating_count',
        y='avg_rating',
        hover_data=['title'],
        title='Movie Popularity (Rating Count vs Average Rating)',
        labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'},
        color_discrete_sequence=['#E50914']
    )
    fig.update_layout(height=500)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Recommendation Settings")
    
    # Initialize recommendation system
    recommender = MovieRecommendationSystem()
    
    # Load data
    if not os.path.exists('Dataset/movielens_ratings.csv'):
        st.error("Dataset not found. Please run the download script first.")
        st.code("python download_dataset.py")
        return
    
    with st.spinner('Loading MovieLens dataset...'):
        if not recommender.load_data():
            return
    
    # Display dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ratings", f"{len(recommender.ratings_df):,}")
    with col2:
        st.metric("Unique Users", f"{recommender.ratings_df['user_id'].nunique():,}")
    with col3:
        st.metric("Unique Movies", f"{recommender.ratings_df['movie_id'].nunique():,}")
    with col4:
        avg_rating = recommender.ratings_df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    # Dataset visualizations
    st.subheader("📈 Dataset Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_ratings = plot_rating_distribution(recommender.ratings_df)
        st.plotly_chart(fig_ratings, use_container_width=True)
        
        fig_activity = plot_user_activity(recommender.ratings_df)
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        fig_genres = plot_top_genres(recommender.movies_df)
        st.plotly_chart(fig_genres, use_container_width=True)
        
        fig_popularity = plot_movie_popularity(recommender.ratings_df, recommender.movies_df)
        st.plotly_chart(fig_popularity, use_container_width=True)
    
    # Recommendation method selection
    st.sidebar.subheader("🔧 Algorithm Settings")
    recommendation_method = st.sidebar.selectbox(
        "Select Recommendation Method",
        ["user_based", "item_based", "svd", "content_based", "hybrid"]
    )
    
    n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # User selection for personalized recommendations
    st.sidebar.subheader("👤 User Selection")
    user_id = st.sidebar.selectbox(
        "Select User ID",
        options=sorted(recommender.ratings_df['user_id'].unique())
    )
    
    # Show user's rating history
    if st.sidebar.button("👀 Show User's Rating History"):
        user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]
        user_movies = user_ratings.merge(recommender.movies_df, on='movie_id')
        user_movies = user_movies.sort_values('rating', ascending=False)
        
        st.subheader(f"🎭 User {user_id}'s Rating History")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ratings", len(user_ratings))
        with col2:
            st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
        with col3:
            st.metric("Favorite Genre", "Drama")  # Simplified
        
        # Show top rated movies
        st.subheader("🌟 Top Rated Movies by User")
        top_movies = user_movies.head(10)[['title', 'rating', 'genres']]
        st.dataframe(top_movies, use_container_width=True, hide_index=True)
    
    # Generate recommendations
    if st.sidebar.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner(f'Generating {recommendation_method} recommendations...'):
            recommendations = recommender.get_user_recommendations(
                user_id, recommendation_method, n_recommendations
            )
        
        if recommendations:
            st.subheader(f"🎯 Personalized Recommendations for User {user_id}")
            st.write(f"**Method:** {recommendation_method.replace('_', ' ').title()}")
            
            # Get movie details for recommendations
            movie_ids = [movie_id for movie_id, _ in recommendations]
            movie_details = recommender.get_movie_details(movie_ids)
            
            # Display recommendations
            for i, (movie_id, score) in enumerate(recommendations, 1):
                movie_info = movie_details[movie_details['movie_id'] == movie_id].iloc[0]
                
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.markdown(f"**#{i}**")
                    
                    with col2:
                        st.markdown(
                            f'<div class="recommendation-card">'
                            f'<h4>{movie_info["title"]}</h4>'
                            f'<p><strong>Genres:</strong> {movie_info["genres"]}</p>'
                            f'<p><strong>Year:</strong> {movie_info["year"]}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.metric("Score", f"{score:.3f}")
        else:
            st.warning("No recommendations found for this user.")
    
    # Popular movies section
    st.subheader("🔥 Popular Movies")
    
    if st.button("📈 Show Popular Movies"):
        popular_movie_ids = recommender.get_popular_movies(15)
        popular_movies = recommender.get_movie_details(popular_movie_ids)
        
        # Get rating statistics for popular movies
        movie_stats = recommender.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['rating_count', 'avg_rating']
        
        popular_with_stats = popular_movies.merge(
            movie_stats, left_on='movie_id', right_index=True
        )
        
        # Display popular movies in a nice format
        for i, (_, movie) in enumerate(popular_with_stats.iterrows(), 1):
            col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
            
            with col1:
                st.markdown(f"**#{i}**")
            
            with col2:
                st.markdown(f"**{movie['title']}**")
                st.write(f"Genres: {movie['genres']}")
            
            with col3:
                st.metric("Avg Rating", f"{movie['avg_rating']:.2f}")
            
            with col4:
                st.metric("# Ratings", f"{movie['rating_count']:,}")
    
    # Model evaluation section
    st.subheader("📊 Model Evaluation")
    
    if st.button("🧪 Evaluate Recommendation System"):
        # Load test data
        test_data = pd.read_csv('Dataset/movielens_test.csv')
        
        with st.spinner('Evaluating recommendation system...'):
            # Evaluate different methods
            methods = ['user_based', 'item_based', 'svd']
            evaluation_results = {}
            
            for method in methods:
                results = recommender.evaluate_recommendations(
                    test_data, method, n_recommendations
                )
                evaluation_results[method] = results
        
        # Display evaluation results
        st.subheader("🎯 Evaluation Results (Precision@K, Recall@K)")
        
        eval_df = pd.DataFrame(evaluation_results).T
        eval_df.index = [method.replace('_', ' ').title() for method in eval_df.index]
        
        # Format the dataframe
        eval_df_display = eval_df.copy()
        for col in ['precision_at_k', 'recall_at_k', 'f1_score']:
            eval_df_display[col] = eval_df_display[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(eval_df_display, use_container_width=True)
        
        # Plot evaluation metrics
        fig = go.Figure()
        
        methods_clean = [method.replace('_', ' ').title() for method in methods]
        
        fig.add_trace(go.Bar(
            name='Precision@K',
            x=methods_clean,
            y=[evaluation_results[method]['precision_at_k'] for method in methods],
            marker_color='#E50914'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall@K',
            x=methods_clean,
            y=[evaluation_results[method]['recall_at_k'] for method in methods],
            marker_color='#B20710'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=methods_clean,
            y=[evaluation_results[method]['f1_score'] for method in methods],
            marker_color='#8B0000'
        ))
        
        fig.update_layout(
            title='Recommendation System Performance Comparison',
            xaxis_title='Method',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best performing method
        best_method = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['f1_score'])
        st.success(f"🏆 Best performing method: **{best_method.replace('_', ' ').title()}** "
                  f"(F1-Score: {evaluation_results[best_method]['f1_score']:.4f})")
    
    # Data exploration section
    with st.expander("🔍 Data Exploration"):
        st.subheader("Sample Data")
        
        tab1, tab2, tab3 = st.tabs(["Ratings", "Movies", "Users"])
        
        with tab1:
            st.dataframe(recommender.ratings_df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(recommender.movies_df.head(10), use_container_width=True)
        
        with tab3:
            st.dataframe(recommender.users_df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Ratings Statistics:**")
            st.dataframe(recommender.ratings_df.describe(), use_container_width=True)
        
        with col2:
            st.write("**Sparsity Analysis:**")
            total_possible = len(recommender.ratings_df['user_id'].unique()) * len(recommender.ratings_df['movie_id'].unique())
            actual_ratings = len(recommender.ratings_df)
            sparsity = (1 - actual_ratings / total_possible) * 100
            
            st.metric("Matrix Sparsity", f"{sparsity:.2f}%")
            st.metric("Total Possible Ratings", f"{total_possible:,}")
            st.metric("Actual Ratings", f"{actual_ratings:,}")

if __name__ == "__main__":
    main()
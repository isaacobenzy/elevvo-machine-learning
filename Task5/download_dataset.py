import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def download_movielens_100k():
    """
    Download and prepare the MovieLens 100K dataset
    """
    print("Downloading MovieLens 100K dataset...")
    
    # Create Dataset directory if it doesn't exist
    os.makedirs('Dataset', exist_ok=True)
    
    # MovieLens 100K dataset URL
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "Dataset/ml-100k.zip"
    extract_path = "Dataset/ml-100k"
    
    try:
        # Download the dataset
        print("Downloading from GroupLens...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")
        
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("Dataset")
        print(f"Extracted to {extract_path}")
        
        # Load the main ratings data
        ratings_file = os.path.join(extract_path, 'u.data')
        users_file = os.path.join(extract_path, 'u.user')
        items_file = os.path.join(extract_path, 'u.item')
        
        # Read ratings data (user_id, item_id, rating, timestamp)
        print("Processing ratings data...")
        ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(ratings_file, sep='\t', names=ratings_columns)
        
        # Read users data
        print("Processing users data...")
        users_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        users_df = pd.read_csv(users_file, sep='|', names=users_columns)
        
        # Read items data (movies)
        print("Processing movies data...")
        items_columns = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                       [f'genre_{i}' for i in range(19)]  # 19 genre columns
        
        # Read with latin-1 encoding to handle special characters
        items_df = pd.read_csv(items_file, sep='|', names=items_columns, encoding='latin-1')
        
        # Clean up movie titles and extract year
        items_df['year'] = items_df['title'].str.extract(r'\((\d{4})\)$')
        items_df['clean_title'] = items_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Create genre list for each movie
        genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        def get_genres(row):
            genres = []
            for i, genre in enumerate(genre_names):
                if row[f'genre_{i}'] == 1:
                    genres.append(genre)
            return '|'.join(genres) if genres else 'unknown'
        
        items_df['genres'] = items_df.apply(get_genres, axis=1)
        
        # Keep only essential movie columns
        movies_df = items_df[['item_id', 'title', 'clean_title', 'year', 'genres']].copy()
        movies_df.columns = ['movie_id', 'title', 'clean_title', 'year', 'genres']
        
        # Merge ratings with movie and user information
        print("Merging datasets...")
        full_df = ratings_df.merge(movies_df, left_on='item_id', right_on='movie_id', how='left')
        full_df = full_df.merge(users_df, on='user_id', how='left')
        
        # Create a simplified ratings dataset
        ratings_simple = ratings_df[['user_id', 'item_id', 'rating']].copy()
        ratings_simple.columns = ['user_id', 'movie_id', 'rating']
        
        # Save processed datasets
        print("Saving processed datasets...")
        
        # Full dataset with all information
        full_df.to_csv('Dataset/movielens_full.csv', index=False)
        print(f"Saved full dataset: {len(full_df)} ratings")
        
        # Simple ratings matrix
        ratings_simple.to_csv('Dataset/movielens_ratings.csv', index=False)
        print(f"Saved ratings dataset: {len(ratings_simple)} ratings")
        
        # Movies dataset
        movies_df.to_csv('Dataset/movielens_movies.csv', index=False)
        print(f"Saved movies dataset: {len(movies_df)} movies")
        
        # Users dataset
        users_df.to_csv('Dataset/movielens_users.csv', index=False)
        print(f"Saved users dataset: {len(users_df)} users")
        
        # Create train/test split for collaborative filtering
        print("Creating train/test split...")
        train_ratings, test_ratings = train_test_split(
            ratings_simple, test_size=0.2, random_state=42, stratify=ratings_simple['user_id']
        )
        
        train_ratings.to_csv('Dataset/movielens_train.csv', index=False)
        test_ratings.to_csv('Dataset/movielens_test.csv', index=False)
        print(f"Train set: {len(train_ratings)} ratings")
        print(f"Test set: {len(test_ratings)} ratings")
        
        # Create user-item matrix for collaborative filtering
        print("Creating user-item matrix...")
        user_item_matrix = ratings_simple.pivot(index='user_id', columns='movie_id', values='rating')
        user_item_matrix.fillna(0, inplace=True)
        user_item_matrix.to_csv('Dataset/user_item_matrix.csv')
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        
        # Dataset statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total ratings: {len(ratings_simple):,}")
        print(f"Unique users: {ratings_simple['user_id'].nunique():,}")
        print(f"Unique movies: {ratings_simple['movie_id'].nunique():,}")
        print(f"Rating range: {ratings_simple['rating'].min()} - {ratings_simple['rating'].max()}")
        print(f"Average rating: {ratings_simple['rating'].mean():.2f}")
        print(f"Sparsity: {(1 - len(ratings_simple) / (ratings_simple['user_id'].nunique() * ratings_simple['movie_id'].nunique())) * 100:.2f}%")
        
        # Rating distribution
        print("\nRating distribution:")
        rating_counts = ratings_simple['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  {rating} stars: {count:,} ({count/len(ratings_simple)*100:.1f}%)")
        
        # Top genres
        print("\nTop movie genres:")
        all_genres = []
        for genres_str in movies_df['genres']:
            if genres_str != 'unknown':
                all_genres.extend(genres_str.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} movies")
        
        # Most rated movies
        print("\nMost rated movies:")
        movie_ratings = ratings_simple.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_ratings.columns = ['rating_count', 'avg_rating']
        movie_ratings = movie_ratings.merge(movies_df, on='movie_id', how='left')
        top_rated = movie_ratings.sort_values('rating_count', ascending=False).head(10)
        
        for _, movie in top_rated.iterrows():
            print(f"  {movie['title']}: {movie['rating_count']} ratings (avg: {movie['avg_rating']})")
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"\nCleaned up {zip_path}")
        
        print("\n✅ MovieLens 100K dataset downloaded and processed successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        # If download fails, create a small synthetic dataset for demonstration
        print("Creating synthetic dataset for demonstration...")
        create_synthetic_movielens()

def create_synthetic_movielens():
    """
    Create a synthetic MovieLens-like dataset for demonstration
    """
    print("Creating synthetic MovieLens dataset...")
    
    np.random.seed(42)
    
    # Create synthetic movies
    movie_titles = [
        "The Matrix (1999)", "Titanic (1997)", "Star Wars (1977)", "Jurassic Park (1993)",
        "Forrest Gump (1994)", "The Lion King (1994)", "Pulp Fiction (1994)", "The Shawshank Redemption (1994)",
        "Goodfellas (1990)", "The Godfather (1972)", "Casablanca (1942)", "Gone with the Wind (1939)",
        "Lawrence of Arabia (1962)", "Schindler's List (1993)", "Vertigo (1958)", "Citizen Kane (1941)",
        "The Wizard of Oz (1939)", "Singin' in the Rain (1952)", "It's a Wonderful Life (1946)",
        "Sunset Blvd. (1950)", "The Bridge on the River Kwai (1957)", "Some Like It Hot (1959)",
        "All About Eve (1950)", "The African Queen (1951)", "Psycho (1960)", "Chinatown (1974)",
        "One Flew Over the Cuckoo's Nest (1975)", "The Grapes of Wrath (1940)", "2001: A Space Odyssey (1968)",
        "The Maltese Falcon (1941)", "Raging Bull (1980)", "E.T. the Extra-Terrestrial (1982)",
        "Dr. Strangelove (1964)", "Bonnie and Clyde (1967)", "Apocalypse Now (1979)", "Mr. Smith Goes to Washington (1939)",
        "The Treasure of the Sierra Madre (1948)", "Annie Hall (1977)", "The Godfather: Part II (1974)",
        "High Noon (1952)", "To Kill a Mockingbird (1962)", "It Happened One Night (1934)",
        "Midnight Cowboy (1969)", "The Best Years of Our Lives (1946)", "Double Indemnity (1944)",
        "Doctor Zhivago (1965)", "North by Northwest (1959)", "West Side Story (1961)",
        "Rear Window (1954)", "King Kong (1933)", "The Birth of a Nation (1915)", "A Streetcar Named Desire (1951)",
        "A Clockwork Orange (1971)", "Taxi Driver (1976)", "Jaws (1975)", "Snow White and the Seven Dwarfs (1937)",
        "Butch Cassidy and the Sundance Kid (1969)", "The Philadelphia Story (1940)", "From Here to Eternity (1953)",
        "Amadeus (1984)", "All Quiet on the Western Front (1930)", "The Sound of Music (1965)",
        "M*A*S*H (1970)", "The Third Man (1949)", "Fantasia (1940)", "Rebel Without a Cause (1955)",
        "Raiders of the Lost Ark (1981)", "The Deer Hunter (1978)", "The Wild Bunch (1969)",
        "Modern Times (1936)", "Giant (1956)", "Platoon (1986)", "Fargo (1996)", "Duck Soup (1933)",
        "Mutiny on the Bounty (1935)", "Frankenstein (1931)", "Easy Rider (1969)", "Patton (1970)",
        "The Jazz Singer (1927)", "My Fair Lady (1964)", "A Place in the Sun (1951)", "The Apartment (1960)",
        "GoodFellas (1990)", "Pulp Fiction (1994)", "The Searchers (1956)", "Bringing Up Baby (1938)",
        "Unforgiven (1992)", "Guess Who's Coming to Dinner (1967)", "Yankee Doodle Dandy (1942)",
        "The Silence of the Lambs (1991)", "In the Heat of the Night (1967)", "Forrest Gump (1994)",
        "All the President's Men (1976)", "On the Waterfront (1954)", "The Exorcist (1973)",
        "Singin' in the Rain (1952)", "The French Connection (1971)", "Terms of Endearment (1983)",
        "The Sting (1973)", "Gandhi (1982)", "American Graffiti (1973)", "Cabaret (1972)",
        "Nashville (1975)", "Network (1976)", "The Hustler (1961)", "The Hospital (1971)",
        "The Graduate (1967)", "American Beauty (1999)", "Toy Story (1995)", "Saving Private Ryan (1998)"
    ]
    
    genres_list = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", 
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    # Create movies dataframe
    movies_data = []
    for i, title in enumerate(movie_titles, 1):
        year = title.split('(')[-1].replace(')', '') if '(' in title else '1990'
        clean_title = title.split(' (')[0] if '(' in title else title
        # Assign 1-3 random genres
        num_genres = np.random.randint(1, 4)
        movie_genres = '|'.join(np.random.choice(genres_list, num_genres, replace=False))
        
        movies_data.append({
            'movie_id': i,
            'title': title,
            'clean_title': clean_title,
            'year': year,
            'genres': movie_genres
        })
    
    movies_df = pd.DataFrame(movies_data)
    
    # Create users dataframe
    users_data = []
    occupations = ['administrator', 'artist', 'doctor', 'educator', 'engineer', 'entertainment', 
                  'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian', 'marketing', 
                  'none', 'other', 'programmer', 'retired', 'salesman', 'scientist', 'student', 
                  'technician', 'writer']
    
    for user_id in range(1, 944):  # 943 users like original dataset
        users_data.append({
            'user_id': user_id,
            'age': np.random.randint(7, 73),
            'gender': np.random.choice(['M', 'F']),
            'occupation': np.random.choice(occupations),
            'zip_code': f"{np.random.randint(10000, 99999)}"
        })
    
    users_df = pd.DataFrame(users_data)
    
    # Create ratings dataframe
    ratings_data = []
    num_ratings = 80000  # Approximate number of ratings
    
    for _ in range(num_ratings):
        user_id = np.random.randint(1, 944)
        movie_id = np.random.randint(1, len(movies_df) + 1)
        
        # Create realistic rating distribution (more 3s and 4s)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.3, 0.35, 0.2])
        timestamp = np.random.randint(874965758, 893286638)  # Approximate timestamp range
        
        ratings_data.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    ratings_df = pd.DataFrame(ratings_data)
    
    # Remove duplicates (same user rating same movie multiple times)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')
    
    # Create simple ratings dataset
    ratings_simple = ratings_df[['user_id', 'movie_id', 'rating']].copy()
    
    # Merge for full dataset
    full_df = ratings_df.merge(movies_df, on='movie_id', how='left')
    full_df = full_df.merge(users_df, on='user_id', how='left')
    
    # Save all datasets
    os.makedirs('Dataset', exist_ok=True)
    
    full_df.to_csv('Dataset/movielens_full.csv', index=False)
    ratings_simple.to_csv('Dataset/movielens_ratings.csv', index=False)
    movies_df.to_csv('Dataset/movielens_movies.csv', index=False)
    users_df.to_csv('Dataset/movielens_users.csv', index=False)
    
    # Train/test split
    train_ratings, test_ratings = train_test_split(
        ratings_simple, test_size=0.2, random_state=42
    )
    
    train_ratings.to_csv('Dataset/movielens_train.csv', index=False)
    test_ratings.to_csv('Dataset/movielens_test.csv', index=False)
    
    # User-item matrix
    user_item_matrix = ratings_simple.pivot(index='user_id', columns='movie_id', values='rating')
    user_item_matrix.fillna(0, inplace=True)
    user_item_matrix.to_csv('Dataset/user_item_matrix.csv')
    
    print(f"✅ Synthetic MovieLens dataset created successfully!")
    print(f"Total ratings: {len(ratings_simple):,}")
    print(f"Unique users: {ratings_simple['user_id'].nunique():,}")
    print(f"Unique movies: {ratings_simple['movie_id'].nunique():,}")
    print(f"Average rating: {ratings_simple['rating'].mean():.2f}")

if __name__ == "__main__":
    download_movielens_100k()
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    print("MYNEXTMOVIE RECOMMENDATION SYSTEM")
    print("=" * 50)
    
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        print("SUCCESS: Data loaded successfully!")
        print(f"Movies dataset: {len(movies)} movies")
        print(f"Ratings dataset: {len(ratings)} ratings")
        return movies, ratings
    except Exception as e:
        print(f"ERROR: Could not load data files: {e}")
        print("Please ensure movies.csv and ratings.csv are in the same folder")
        return None, None

def popularity_based_recommender(movies, ratings):

    movies['movieId']=movies['movieId'].astype(int)
    ratings['movieId']=ratings['movieId'].astype(int)
    merged=pd.merge(ratings,movies,on='movieId')
    print("\nPOPULARITY-BASED RECOMMENDER")
    print("-" * 40)
    
    genre = input("Enter genre: ").strip()
    min_ratings = int(input("Minimum number of ratings: "))
    num_recommendations = int(input("Number of recommendations: "))
    
    genre_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    
    if len(genre_movies) == 0:
        print(f"No movies found in genre: {genre}")
        return
    
    ratings_summary = ratings.groupby('movieId').agg(
        average_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    merged_data = genre_movies.merge(ratings_summary, on='movieId')
    filtered_movies = merged_data[merged_data['rating_count'] >= min_ratings]
    
    if len(filtered_movies) == 0:
        print(f"No movies found with at least {min_ratings} ratings")
        return
    
    top_movies = filtered_movies.sort_values('average_rating', ascending=False).head(num_recommendations)
    
    print(f"\nTOP {num_recommendations} {genre.upper()} MOVIES:")
    print("=" * 50)
    for _, row in top_movies.iterrows():
        print(f"Movie: {row['title']}")
        print(f"Average Rating: {row['average_rating']:.2f}")
        print(f"Number of Ratings: {row['rating_count']}")
        print(f"Genres: {row['genres']}")
        print()

def content_based_recommender(movies):
    print("\nCONTENT-BASED RECOMMENDER")
    print("-" * 40)
    
    print("Sample movies from dataset:")
    sample_movies = movies['title'].head(10).tolist()
    for i, movie in enumerate(sample_movies, 1):
        print(f"{i}. {movie}")
    
    movie_title = input("\nEnter movie title from above list: ").strip()
    num_recommendations = int(input("Number of recommendations: "))
    
    if movie_title not in movies['title'].values:
        print(f"Movie '{movie_title}' not found in dataset")
        return
    
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = vectorizer.fit_transform(movies['genres'])
    
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    
    movie_index = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_similar = similarity_scores[1:num_recommendations + 1]
    
    print(f"\nMOVIES SIMILAR TO '{movie_title}':")
    print("=" * 50)
    for i, (index, score) in enumerate(top_similar, 1):
        similar_movie = movies.iloc[index]
        print(f"{i}. {similar_movie['title']}")
        print(f"   Similarity Score: {score:.2f}")
        print(f"   Genres: {similar_movie['genres']}")
        print()

def run_demo(movies, ratings):
    print("\nDEMONSTRATION MODE")
    print("=" * 40)
    
    available_genres = set()
    for genres in movies['genres']:
        available_genres.update(genres.split('|'))
    
    demo_genre = list(available_genres)[0] if available_genres else "Drama"
    
    print(f"1. Popularity-based demo for genre: {demo_genre}")
    print("   Minimum ratings: 10, Recommendations: 3")
    print("-" * 50)
    
    genre_movies = movies[movies['genres'].str.contains(demo_genre, case=False, na=False)]
    if len(genre_movies) > 0:
        ratings_summary = ratings.groupby('movieId').agg(
            average_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        
        merged_data = genre_movies.merge(ratings_summary, on='movieId')
        filtered_movies = merged_data[merged_data['rating_count'] >= 10]
        
        if len(filtered_movies) > 0:
            top_movies = filtered_movies.sort_values('average_rating', ascending=False).head(3)
            for _, row in top_movies.iterrows():
                print(f"   {row['title']} - Rating: {row['average_rating']:.2f}")
        else:
            print("   No movies meet the criteria")
    else:
        print(f"   No movies found in genre: {demo_genre}")
    
    print(f"\n2. Content-based demo")
    demo_movie = movies['title'].iloc[0] if len(movies) > 0 else "Sample Movie"
    print(f"   Movies similar to: {demo_movie}")
    print("-" * 50)
    
    if len(movies) > 0:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
        genre_matrix = vectorizer.fit_transform(movies['genres'])
        cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
        
        movie_index = 0
        similarity_scores = list(enumerate(cosine_sim[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        top_similar = similarity_scores[1:4]
        
        for i, (index, score) in enumerate(top_similar, 1):
            similar_movie = movies.iloc[index]
            print(f"   {i}. {similar_movie['title']}")

def main():
    movies, ratings = load_data()
    
    if movies is None or ratings is None:
        return
    
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("1. Popularity-based Recommendations")
        print("2. Content-based Recommendations")
        print("3. Run Demonstration")
        print("4. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            popularity_based_recommender(movies, ratings)
        elif choice == '2':
            content_based_recommender(movies)
        elif choice == '3':
            run_demo(movies, ratings)
        elif choice == '4':
            print("Thank you for using MyNextMovie Recommendation System!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()
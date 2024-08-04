# Install the necessary libraries (uncomment if running locally)
# !pip install pandas numpy scikit-learn surprise

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load the MovieLens dataset
movies_path = 'path_to/movies.csv'
ratings_path = 'path_to/ratings.csv'

# Read movies and ratings data
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Display the first few rows of the movies and ratings dataframes
print("Movies DataFrame:")
print(movies.head())
print("\nRatings DataFrame:")
print(ratings.head())

# Merge the movies and ratings dataframes on the 'movieId' column
merged_data = pd.merge(ratings, movies, on='movieId')

# Create a Surprise dataset from the merged data
reader = Reader(rating_scale=(0.5, 5.0))
surprise_data = Dataset.load_from_df(merged_data[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets (75% train, 25% test)
trainset, testset = train_test_split(surprise_data, test_size=0.25)

# Use the SVD algorithm for collaborative filtering
svd_model = SVD()

# Train the model on the training set
svd_model.fit(trainset)

# Make predictions on the test set
predictions = svd_model.test(testset)

# Evaluate the accuracy of the model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"\nModel Evaluation:\nRMSE: {rmse}\nMAE: {mae}")

# Function to get movie recommendations for a given user
def get_movie_recommendations(user_id, model, movies_df, num_recommendations=10):
    # Get all unique movieIds
    unique_movie_ids = movies_df['movieId'].unique()
    
    # Predict ratings for all movies for the given user
    predicted_ratings = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unique_movie_ids]
    
    # Sort the movies by predicted rating in descending order
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top n recommended movies
    top_recommendations = [movie_id for movie_id, rating in predicted_ratings[:num_recommendations]]
    
    # Retrieve the titles of the recommended movies
    recommended_movies = movies_df[movies_df['movieId'].isin(top_recommendations)]
    
    return recommended_movies[['movieId', 'title']]

# Example: Get top 10 movie recommendations for user with userId=1
user_id_example = 1
recommendations = get_movie_recommendations(user_id_example, svd_model, movies)

print("\nTop 10 Movie Recommendations:")
print(recommendations)

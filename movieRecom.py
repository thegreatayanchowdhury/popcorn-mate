import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load both datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge on title
movies = movies.merge(credits, on='title')

# Keep only required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Parse JSON-like columns
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # top 3 cast members
movies['crew'] = movies['crew'].apply(get_director)

# Create a combined column
movies['tags'] = (
    movies['overview'] + " " +
    movies['genres'].apply(lambda x: " ".join(x)) + " " +
    movies['keywords'].apply(lambda x: " ".join(x)) + " " +
    movies['cast'].apply(lambda x: " ".join(x)) + " " +
    movies['crew']
)

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommend function
def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        print("Movie not found in dataset.")
        return

    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nTop 5 recommendations for '{movies.iloc[index]['title']}':")
    for i in movie_list:
        print("→", movies.iloc[i[0]]['title'])

print("=== AI Movie Recommender ===")
name = input("Enter a movie you like: ")
recommend(name)

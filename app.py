from flask import Flask, render_template, request
from surprise import SVDpp, Dataset, Reader
import pandas as pd
import os

app = Flask(__name__, template_folder='notebooks/templates')


# Load the model and data
reader = Reader()
ratings_file = os.path.join(os.getcwd(), 'ratings.csv')
df_ratings = pd.read_csv('./data/ratings.csv')
movies_file = os.path.join(os.getcwd(), 'movies.csv')
df_movies = pd.read_csv('./data/movies.csv')
data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)

# Train the model
algo = SVDpp()
trainset = data.build_full_trainset()
algo.fit(trainset)

# Function to get movie title
def get_movie_title(movie_id):
    title = df_movies[df_movies['movieId'] == movie_id]['title'].values[0]
    return title

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    uid = request.form.get('userId')
    uid = str(uid)

    if uid:
        rated_movies = df_ratings[df_ratings['userId'] == int(uid)]['movieId']
        unrated_movies = df_movies[~df_movies['movieId'].isin(rated_movies)]['movieId']
        
        preds = []
        for iid in unrated_movies:
            pred = algo.predict(uid, str(iid))
            preds.append((get_movie_title(iid), pred.est))

        # Get the top 5 predictions
        preds.sort(key=lambda x: x[1], reverse=True)
        top_preds = preds[:5]

        return render_template('recommendations.html', predictions=top_preds)

    return "No user ID provided. Please go back and enter a user ID."

if __name__ == '__main__':
    app.run(debug=True)

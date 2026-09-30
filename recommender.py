"""
Movie Recommendation Engine
---------------------------
Two approaches, both built on cosine similarity:

1. Collaborative filtering (item-based): movies that are rated similarly by the
   same users are "close" to each other.
2. Content-based: movies that share genres are "close" to each other.

A hybrid score blends the two, which also helps with movies that have few ratings.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


class MovieRecommender:
    def __init__(self, movies_path=None, ratings_path=None):
        movies_path = movies_path or DATA_DIR / "movies.csv"
        ratings_path = ratings_path or DATA_DIR / "ratings.csv"
        self.movies, self.ratings = self._load_and_clean(movies_path, ratings_path)
        self._build_matrices()

    # ---------- Step 1 & 2: load, clean, build user-item matrix ----------
    @staticmethod
    def _load_and_clean(movies_path, ratings_path):
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        movies = movies.drop_duplicates("movieId").dropna(subset=["title"])
        # MovieLens style "Matrix, The (1999)" -> "The Matrix (1999)"
        movies["title"] = movies["title"].str.replace(r"^(.*), (The|A|An) (\(\d{4}\))$", r"\2 \1 \3", regex=True)
        movies = movies.drop_duplicates("title")
        movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")
        if "poster_path" not in movies:
            movies["poster_path"] = ""
        movies["poster_path"] = movies["poster_path"].fillna("")
        ratings = ratings.dropna().drop_duplicates(["userId", "movieId"])
        ratings = ratings[ratings["rating"].between(0.5, 5)]
        ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
        return movies.reset_index(drop=True), ratings.reset_index(drop=True)

    def _build_matrices(self):
        # rows = movies, columns = users, values = rating (0 = not rated)
        self.item_user = (
            self.ratings.pivot_table(index="movieId", columns="userId", values="rating")
            .reindex(self.movies["movieId"])
            .fillna(0)
        )
        # Mean-center per user so a harsh rater's "3" and a generous rater's "5" compare fairly
        user_means = self.ratings.groupby("userId")["rating"].mean()
        centered = self.item_user.sub(user_means.reindex(self.item_user.columns), axis=1).where(self.item_user > 0, 0)

        ids = self.movies["movieId"].values
        self.cf_sim = pd.DataFrame(cosine_similarity(centered.values), index=ids, columns=ids)

        # Content similarity from one-hot genres
        genre_matrix = self.movies["genres"].str.get_dummies(sep="|")
        self.genre_sim = pd.DataFrame(cosine_similarity(genre_matrix.values), index=ids, columns=ids)

        self.title_to_id = dict(zip(self.movies["title"], self.movies["movieId"]))
        self.id_to_title = {v: k for k, v in self.title_to_id.items()}

        stats = self.ratings.groupby("movieId")["rating"].agg(avg_rating="mean", num_ratings="count")
        self.movies = self.movies.merge(stats, on="movieId", how="left")

    # ---------- Step 3 & 4: similarity + recommendation functions ----------
    def _combined_sim(self, alpha):
        """alpha = weight for collaborative filtering; (1 - alpha) for genres."""
        # clip at 0: "dissimilar" shouldn't count as evidence in either direction
        return alpha * self.cf_sim.clip(lower=0) + (1 - alpha) * self.genre_sim

    def similar_movies(self, title, n=5, alpha=0.5):
        """Movies most similar to a given title."""
        if title not in self.title_to_id:
            raise ValueError(f"'{title}' not found in dataset")
        mid = self.title_to_id[title]
        scores = self._combined_sim(alpha)[mid].drop(mid).sort_values(ascending=False).head(n)
        return self._format(scores)

    def recommend_for_ratings(self, my_ratings: dict, n=5, alpha=0.5):
        """
        Recommend for a new user.
        my_ratings: {"Inception": 5, "Toy Story": 2, ...}
        Score of candidate = similarity-weighted average of the user's ratings.
        """
        sim = self._combined_sim(alpha)
        liked = {self.title_to_id[t]: r for t, r in my_ratings.items() if t in self.title_to_id}
        if not liked:
            return pd.DataFrame(columns=["title", "genres", "score", "avg_rating", "poster_path"])

        ids = list(liked)
        weights = np.array([liked[i] - 2.5 for i in ids])  # >2.5 pulls up, <2.5 pushes down
        S = sim.loc[:, ids].values
        # "+ 1" shrinks scores for movies with weak evidence (only faintly similar to what you rated)
        scores = S @ weights / (S.sum(axis=1) + 1.0)
        scores = pd.Series(scores, index=sim.index).drop(ids)  # don't recommend already-rated
        return self._format(scores.sort_values(ascending=False).head(n))

    def recommend_for_user(self, user_id, n=5, alpha=0.5):
        """Recommend for an existing user in the ratings data."""
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id} not found")
        as_dict = {self.id_to_title[m]: r for m, r in zip(user_ratings["movieId"], user_ratings["rating"])}
        return self.recommend_for_ratings(as_dict, n=n, alpha=alpha)

    # ---------- Step 5: clean output ----------
    def _format(self, scores: pd.Series):
        out = self.movies.set_index("movieId").loc[scores.index, ["title", "genres", "avg_rating", "poster_path"]].copy()
        out["score"] = scores.round(3).values
        out["avg_rating"] = out["avg_rating"].round(2)
        return out[["title", "genres", "score", "avg_rating", "poster_path"]].reset_index(drop=True)


# ---------- Step 6: test with different inputs ----------
if __name__ == "__main__":
    rec = MovieRecommender()
    pd.set_option("display.width", 120)
    popular = rec.movies.sort_values("num_ratings", ascending=False)["title"].tolist()
    print(f"{len(rec.movies)} movies, {rec.ratings['userId'].nunique()} users\n")
    for t in popular[:2]:
        print(f"=== Because you liked '{t}' ===")
        print(rec.similar_movies(t).drop(columns="poster_path"), "\n")
    print("=== Existing user #1 ===")
    print(rec.recommend_for_user(1).drop(columns="poster_path"))

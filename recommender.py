"""CineMind hybrid ML recommendation engine.

Uses:
- TF-IDF content vectors from title + genres + optional metadata
- Item-based collaborative filtering from user ratings
- Genre similarity
- Bayesian-style popularity prior
- Natural-language query scoring for the AI layer
"""
from pathlib import Path
import os, re, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"

class MovieRecommender:
    def __init__(self, movies_path=None, ratings_path=None):
        movies_path = movies_path or DATA_DIR / "movies.csv"
        ratings_path = ratings_path or DATA_DIR / "ratings.csv"
        self.movies, self.ratings = self._load_and_clean(movies_path, ratings_path)
        self._build_models()

    @staticmethod
    def _load_and_clean(movies_path, ratings_path):
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        movies = movies.drop_duplicates("movieId").dropna(subset=["title"]).copy()
        movies["title"] = movies["title"].astype(str).str.replace(
            r"^(.*), (The|A|An) (\(\d{4}\))$", r"\2 \1 \3", regex=True
        )
        movies = movies.drop_duplicates("title")
        movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")
        if "poster_path" not in movies:
            movies["poster_path"] = ""
        movies["poster_path"] = movies["poster_path"].fillna("").astype(str)
        ratings = ratings.dropna().drop_duplicates(["userId", "movieId"])
        ratings = ratings[ratings["rating"].between(0.5, 5)]
        ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
        return movies.reset_index(drop=True), ratings.reset_index(drop=True)

    def _build_models(self):
        ids = self.movies["movieId"].values
        # Collaborative filtering: mean-centred item/user matrix.
        item_user = self.ratings.pivot_table(index="movieId", columns="userId", values="rating") \
            .reindex(ids).fillna(0)
        user_means = self.ratings.groupby("userId")["rating"].mean()
        centered = item_user.sub(user_means.reindex(item_user.columns), axis=1).where(item_user > 0, 0)
        self.cf_sim = pd.DataFrame(cosine_similarity(centered.values), index=ids, columns=ids)

        # Content ML: TF-IDF over movie title + genres. This is stronger than genres alone
        # while staying lightweight enough for Render's free tier.
        text = (self.movies["title"].str.replace(r"\(\d{4}\)", "", regex=True) + " " +
                self.movies["genres"].str.replace("|", " ", regex=False)).str.lower()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.content_matrix = self.vectorizer.fit_transform(text)
        self.content_sim = cosine_similarity(self.content_matrix)
        self.content_sim_df = pd.DataFrame(self.content_sim, index=ids, columns=ids)

        genre_matrix = self.movies["genres"].str.get_dummies(sep="|")
        self.genre_sim = pd.DataFrame(cosine_similarity(genre_matrix.values), index=ids, columns=ids)

        self.title_to_id = dict(zip(self.movies["title"], self.movies["movieId"]))
        self.id_to_title = {v: k for k, v in self.title_to_id.items()}

        stats = self.ratings.groupby("movieId")["rating"].agg(avg_rating="mean", num_ratings="count")
        self.movies = self.movies.merge(stats, on="movieId", how="left")
        self.movies["avg_rating"] = self.movies["avg_rating"].fillna(0.0)
        self.movies["num_ratings"] = self.movies["num_ratings"].fillna(0).astype(int)
        # A smooth popularity score prevents one 5.0-rated movie with one vote from dominating.
        C = float(self.ratings["rating"].mean()) if not self.ratings.empty else 3.5
        m = max(float(self.movies["num_ratings"].quantile(0.65)), 1.0)
        self.movies["quality"] = (
            (self.movies["num_ratings"] / (self.movies["num_ratings"] + m)) * self.movies["avg_rating"] +
            (m / (self.movies["num_ratings"] + m)) * C
        )
        self.movies["popularity"] = np.log1p(self.movies["num_ratings"])
        self.movies["popularity"] = self.movies["popularity"] / max(float(self.movies["popularity"].max()), 1.0)

    def _combined_sim(self, alpha=0.5):
        alpha = float(np.clip(alpha, 0, 1))
        # Content gets the largest share; CF is still useful when rating overlap exists.
        return 0.50 * self.content_sim_df + 0.30 * self.genre_sim + 0.20 * self.cf_sim.clip(lower=0) if alpha == 0.5 else ((1-alpha)*self.content_sim_df + 0.25*(1-alpha)*self.genre_sim + alpha*self.cf_sim.clip(lower=0))

    def similar_movies(self, title, n=5, alpha=0.5):
        if title not in self.title_to_id:
            raise ValueError(f"'{title}' not found in dataset")
        mid = self.title_to_id[title]
        sim = self._combined_sim(alpha)
        scores = sim[mid].drop(mid)
        # Small quality prior breaks ties without overwhelming similarity.
        q = self.movies.set_index("movieId")["quality"]
        scores = scores + 0.025 * (q.reindex(scores.index).fillna(0) / 5.0)
        return self._format(scores.sort_values(ascending=False).head(n))

    def recommend_for_ratings(self, my_ratings: dict, n=5, alpha=0.5):
        sim = self._combined_sim(alpha)
        liked = {self.title_to_id[t]: float(r) for t, r in my_ratings.items() if t in self.title_to_id}
        if not liked:
            return pd.DataFrame(columns=["title", "genres", "score", "avg_rating", "poster_path"])
        ids = list(liked)
        weights = np.array([liked[i] - 2.5 for i in ids])
        S = sim.loc[:, ids].values
        scores = S @ weights / (np.abs(S).sum(axis=1) + 1.0)
        scores = pd.Series(scores, index=sim.index).drop(ids)
        quality = self.movies.set_index("movieId")["quality"]
        scores += 0.04 * ((quality.reindex(scores.index).fillna(0) - 2.5) / 2.5)
        return self._format(scores.sort_values(ascending=False).head(n))

    def recommend_for_prompt(self, prompt, n=8, parsed=None):
        """Rank catalogue against a natural-language preference/query."""
        parsed = parsed or {}
        query_parts = [prompt]
        query_parts += parsed.get("genres", [])
        query_parts += parsed.get("keywords", [])
        query_parts += parsed.get("mood", []) if isinstance(parsed.get("mood"), list) else [parsed.get("mood", "")]
        query = " ".join(str(x) for x in query_parts if x).lower()
        qv = self.vectorizer.transform([query])
        content_scores = cosine_similarity(qv, self.content_matrix).ravel()
        scores = pd.Series(content_scores, index=self.movies.movieId.values)

        # Seed titles supplied by the AI layer: blend their learned neighbourhoods.
        seeds = parsed.get("seed_titles", []) or []
        valid = [self.title_to_id[t] for t in seeds if t in self.title_to_id]
        if valid:
            seed_sim = self.content_sim_df.loc[:, valid].mean(axis=1)
            scores = 0.70 * scores + 0.30 * seed_sim

        genres = [str(g).lower() for g in (parsed.get("genres", []) or [])]
        if genres:
            gscore = self.movies["genres"].str.lower().apply(lambda x: sum(g in x for g in genres) / len(genres)).values
            scores += 0.18 * pd.Series(gscore, index=scores.index)

        negative = [str(x).lower() for x in (parsed.get("avoid_genres", []) or [])]
        if negative:
            penalty = self.movies["genres"].str.lower().apply(lambda x: sum(g in x for g in negative)).values
            scores -= 0.20 * pd.Series(np.minimum(penalty, 1), index=scores.index)

        quality = self.movies.set_index("movieId")["quality"]
        scores += 0.025 * ((quality.reindex(scores.index).fillna(0) - 3.0) / 2.0)
        return self._format(scores.sort_values(ascending=False).head(n))

    def recommend_for_user(self, user_id, n=5, alpha=0.5):
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id} not found")
        as_dict = {self.id_to_title[m]: r for m, r in zip(user_ratings["movieId"], user_ratings["rating"])}
        return self.recommend_for_ratings(as_dict, n=n, alpha=alpha)

    def _format(self, scores):
        out = self.movies.set_index("movieId").loc[scores.index, ["title", "genres", "avg_rating", "poster_path"]].copy()
        out["score"] = np.asarray(scores.round(3))
        out["avg_rating"] = out["avg_rating"].round(2)
        return out[["title", "genres", "score", "avg_rating", "poster_path"]].reset_index(drop=True)

if __name__ == "__main__":
    rec = MovieRecommender()
    print(f"{len(rec.movies)} movies, {rec.ratings['userId'].nunique()} users")
    print(rec.similar_movies("Inception (2010)"))

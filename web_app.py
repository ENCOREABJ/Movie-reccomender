"""Web app: Flask backend + custom HTML/CSS/JS frontend.  Run with:  python web_app.py"""
from flask import Flask, jsonify, render_template, request

from recommender import MovieRecommender

app = Flask(__name__)
engine = MovieRecommender()  # loaded once at startup


@app.get("/")
def home():
    movies = engine.movies.fillna({"avg_rating": 0, "num_ratings": 0}).sort_values("title")
    return render_template("index.html", movies=movies.rename(columns={"num_ratings": "n"})[["title", "genres", "n", "poster_path"]].fillna({"n": 0}).to_dict("records"))


@app.post("/api/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    ratings = {t: float(r) for t, r in (body.get("ratings") or {}).items()}
    n = min(max(int(body.get("n", 6)), 1), 12)
    alpha = min(max(float(body.get("alpha", 0.5)), 0.0), 1.0)

    if not ratings:
        return jsonify(results=[])
    df = engine.recommend_for_ratings(ratings, n=n, alpha=alpha).fillna(0)
    top = float(df["score"].max()) or 1.0
    df["match"] = (df["score"].clip(lower=0) / top).round(3)  # 0-1, relative to best pick
    return jsonify(results=df.to_dict("records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

"""CineMind Flask web application."""
from flask import Flask, jsonify, render_template, request
from recommender import MovieRecommender
from ai_layer import parse_preferences

app = Flask(__name__)
engine = MovieRecommender()

@app.get("/")
def home():
    movies = engine.movies.fillna({"avg_rating": 0, "num_ratings": 0}).sort_values("title")
    return render_template("index.html", movies=movies.rename(columns={"num_ratings":"n"})[["title","genres","n","avg_rating","poster_path"]].fillna({"n":0,"poster_path":""}).to_dict("records"))

@app.get("/api/similar")
def similar():
    title=request.args.get("title","")
    if title not in engine.title_to_id: return jsonify(results=[])
    n=min(max(int(request.args.get("n",12)),1),20)
    df=engine.similar_movies(title,n=n).fillna(0)
    return jsonify(results=df.to_dict("records"))

@app.post("/api/recommend")
def recommend():
    body=request.get_json(silent=True) or {}
    ratings={t:float(r) for t,r in (body.get("ratings") or {}).items()}
    n=min(max(int(body.get("n",6)),1),20)
    alpha=min(max(float(body.get("alpha",0.5)),0.0),1.0)
    if not ratings: return jsonify(results=[])
    df=engine.recommend_for_ratings(ratings,n=n,alpha=alpha).fillna(0)
    top=float(df["score"].max()) if len(df) else 1.0
    df["match"]=(df["score"].clip(lower=0)/top).round(3) if top else 0
    return jsonify(results=df.to_dict("records"))

@app.post("/api/ai-recommend")
def ai_recommend():
    body=request.get_json(silent=True) or {}
    prompt=str(body.get("prompt","")).strip()
    if not prompt: return jsonify(results=[],error="Enter what you want to watch."),400
    n=min(max(int(body.get("n",10)),1),20)
    parsed,source=parse_preferences(prompt,list(engine.title_to_id.keys()))
    df=engine.recommend_for_prompt(prompt,n=n,parsed=parsed).fillna(0)
    top=float(df["score"].max()) if len(df) else 1.0
    if top:
        df["match"]=(df["score"].clip(lower=0)/top).round(3)
    return jsonify(results=df.to_dict("records"),preferences=parsed,source=source)

if __name__=="__main__":
    app.run(debug=True,port=5000)

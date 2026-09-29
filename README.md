# Movie Recommendation System

Python + Pandas + cosine similarity (collaborative filtering and genre matching), with two UIs.

## Run locally
    python -m pip install -r requirements.txt
    python recommender.py      # terminal demo
    python web_app.py          # web app at http://127.0.0.1:5000

Optional Streamlit UI:
    python -m pip install -r requirements-streamlit.txt
    python -m streamlit run app.py

## Deploy (Render)
Build command: pip install -r requirements.txt
Start command: gunicorn web_app:app

## Use real data (MovieLens)
Copy movies.csv and ratings.csv from ml-latest-small into data/. Column names already match.

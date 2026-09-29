"""Simple Streamlit UI.  Run with:  streamlit run app.py"""
import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def get_engine():
    return MovieRecommender()


rec = get_engine()
titles = sorted(rec.movies["title"])

st.title("🎬 Movie Recommendation System")
st.caption("Cosine similarity + collaborative filtering + genre matching")

with st.sidebar:
    n = st.slider("Number of recommendations", 3, 10, 5)
    alpha = st.slider(
        "Ratings vs. genres", 0.0, 1.0, 0.5,
        help="1.0 = only what other users rated similarly, 0.0 = only shared genres",
    )

tab1, tab2 = st.tabs(["Similar to a movie", "Rate movies for me"])

with tab1:
    pick = st.selectbox("Pick a movie you like", titles)
    if st.button("Recommend", key="similar"):
        st.dataframe(rec.similar_movies(pick, n=n, alpha=alpha), hide_index=True, use_container_width=True)

with tab2:
    chosen = st.multiselect("Choose movies you've watched", titles)
    my_ratings = {t: st.slider(t, 1, 5, 4, key=f"r_{t}") for t in chosen}
    if st.button("Get my recommendations", key="personal"):
        if not my_ratings:
            st.warning("Pick at least one movie and rate it.")
        else:
            st.dataframe(rec.recommend_for_ratings(my_ratings, n=n, alpha=alpha),
                         hide_index=True, use_container_width=True)

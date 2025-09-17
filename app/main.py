"""
Streamlit web application for interactive movie recommendations.
Allows users to search for movies, add favorites, and get personalized recommendations using a pre-trained ALS model.
"""

import streamlit as st
from app.utils import MovieRecommendationsEngine


# ---------- CONFIG ----------
MAPPING_CSV = "artifacts/movie_id_index.csv"
FACTORS_NPY = "artifacts/movie_factors.npy"

st.set_page_config(page_title="Movie Recommendations", layout="wide")


# ---------- RECOMMENDATIONS ENGINE ----------
try:
    mrengine = MovieRecommendationsEngine(MAPPING_CSV, FACTORS_NPY)
except Exception as e:
    st.error(
        f"Could not load recommendations engine artifacts. Make sure `{MAPPING_CSV}` and `{FACTORS_NPY}` exist.\n\nError: {e}"
    )
    st.stop()


@st.cache_data(ttl=600)
def _cached_search(query: str, limit: int = 10):
    """Cache mrengine.search_titles results for a given query."""
    if not query or not query.strip():
        return []
    return mrengine.search_titles(query.strip(), limit=limit)


# ---------- SESSION STATE ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "search_results" not in st.session_state:
    st.session_state.search_results = {"query": "", "results": []}
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []


# ---------- HELPERS ----------
def _get_title_for(movieid: int) -> str:
    """Get the movie title for a given movieId."""
    row = mrengine.mapping[mrengine.mapping["movieId"] == movieid]
    if len(row):
        return row["title"].values[0]
    return str(movieid)


# ---------- PAGE HEADER ----------
st.title("🎬 Movie Recommendations")
st.write("Search for movies, add them to your favorites and get personalized suggestions.")

left_col, right_col = st.columns([3, 3], gap="large")


# ------- LEFT PANEL: Search -------
with left_col:
    st.subheader("Search")

    col1, col2 = st.columns([8, 1])
    with col1:
        search_input = st.text_input(
            "Search by title",
            placeholder="Enter a movie (e.g. The Matrix)",
            label_visibility="collapsed",
            key="search_input",
        )
    with col2:
        search_btn = st.button(
            "🔍", help="Search movie", use_container_width=True, key="search_btn"
        )

    # Handle search submission
    if search_btn or search_input:
        query = ""
        results = []

        if search_input and search_input.strip():
            query = search_input.strip()
            with st.spinner("Searching..."):
                results = _cached_search(query, limit=10)

        st.session_state.search_results = {"query": query, "results": results}

    # Display search results
    current_query = st.session_state.search_results["query"]
    current_results = st.session_state.search_results["results"]
    
    if current_query:
        if current_results:
            for i, movie in enumerate(current_results[:10]):
                title = movie.get("title", "")
                movie_id = int(movie.get("movieId"))
                
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(f"**{title}**")
                with col2:
                    if st.button("➕", key=f"add_{movie_id}", help="Add to favorites", disabled=(movie_id in st.session_state.favorites)):
                        st.session_state.favorites.append(movie_id)
                        st.rerun()
        else:
            st.error("No movies found.")


# ------- RIGHT PANEL: Favorites -------
with right_col:
    st.subheader("Favorites")
    
    if st.session_state.favorites:
        for movie_id in st.session_state.favorites:
            title = _get_title_for(movie_id)
            col1, col2 = st.columns([8, 1])
            with col1:
                st.write(f"**{title}**")
            with col2:
                if st.button("❌", key=f"remove_{movie_id}", help="Remove"):
                    st.session_state.favorites.remove(movie_id)
                    st.session_state.last_recs = []  # Clear recommendations
                    st.rerun()
        
        # Clear all button
        if st.button("❌ Clear All", help="Remove all favorites"):
            st.session_state.favorites = []
            st.session_state.last_recs = []
            st.rerun()


# ------- BOTTOM PANEL: Recommendations -------
st.write("---")
st.subheader("Recommendations")

if not st.session_state.favorites:
    st.info("Add some favorite movies to get personalized recommendations.")
else:
    num_recs = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("🪄 Get Recommendations", type="primary"):
        with st.spinner("Computing recommendations..."):
            recs = mrengine.recommend_from_favorites(
                st.session_state.favorites, top_n=num_recs
            )
            st.session_state.last_recs = recs

    # Display recommendations
    if st.session_state.last_recs:
        for title, score in st.session_state.last_recs:
            col1, col2 = st.columns([8, 1])
            with col1:
                st.write(f"**{title}**")
            with col2:
                st.caption(f"{score:.3f}")


# ---------- FOOTER ----------
st.write("---")
st.caption("💡 Tip: Add movies from different genres for better recommendations!")
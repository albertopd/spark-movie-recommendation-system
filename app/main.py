"""
Streamlit web application for interactive movie recommendations.
Allows users to search for movies, add favorites, and get personalized recommendations using a pre-trained ALS model.
"""

import streamlit as st
from app.movie_recom_engine import MovieRecommendationsEngine


# ---------- CONFIG ----------
MOVIE_FACTORS_NPY = "artifacts/movie_factors.npy"
MOVIE_INDEX_MAP_CSV = "artifacts/movie_index_map.csv"

st.set_page_config(page_title="Movie Recommendations", layout="wide")


# ---------- RECOMMENDATIONS ENGINE ----------
@st.cache_data
def _load_recommendations_engine():
    """Load the MovieRecommendationsEngine with caching."""
    return MovieRecommendationsEngine(MOVIE_INDEX_MAP_CSV, MOVIE_FACTORS_NPY)


try:
    _mrengine = _load_recommendations_engine()
except Exception as e:
    st.error(
        f"Could not load recommendations engine artifacts. Make sure `{MOVIE_INDEX_MAP_CSV}` and `{MOVIE_FACTORS_NPY}` exist.\n\nError: {e}"
    )
    st.stop()


# ---------- SESSION STATE ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "search_results" not in st.session_state:
    st.session_state.search_results = {"query": "", "results": []}
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []


# ---------- HELPERS ----------
def _get_title_for(movie_index: int) -> str:
    """Get the movie title for a given movie_index."""
    row = _mrengine.mapping[_mrengine.mapping["index"] == movie_index]
    if len(row):
        return row["title"].values[0]
    return ""


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
                results = _mrengine.search_titles(query.strip(), limit=10)

        st.session_state.search_results["query"] = query
        st.session_state.search_results["results"] = results

    # Display search results
    current_query = st.session_state.search_results["query"]
    current_results = st.session_state.search_results["results"]
    
    if current_query:
        if current_results:
            for i, movie in enumerate(current_results[:10]):
                title = movie.get("title", "")
                movie_index = int(movie.get("index")) # type: ignore
                
                col1, col2 = st.columns([8, 1])
                with col1:
                    st.write(f"**{title}**")
                with col2:
                    if st.button("➕", key=f"add_{movie_index}", help="Add to favorites", disabled=(movie_index in st.session_state.favorites)):
                        st.session_state.favorites.append(movie_index)
                        st.rerun()
        else:
            st.error("No movies found.")


# ------- RIGHT PANEL: Favorites -------
with right_col:
    st.subheader("Favorites")
    
    if st.session_state.favorites:
        for movie_index in st.session_state.favorites:
            title = _get_title_for(movie_index)
            col1, col2 = st.columns([8, 1])
            with col1:
                st.write(f"**{title}**")
            with col2:
                if st.button("❌", key=f"remove_{movie_index}", help="Remove"):
                    st.session_state.favorites.remove(movie_index)
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
            recs = _mrengine.recommend_from_favorites(
                st.session_state.favorites, top_n=num_recs
            )
            st.session_state.last_recs = recs

    # Display recommendations
    if st.session_state.last_recs:
        st.write("")
        
        # Build the ordered list in HTML
        html_list = "<ol>"
        for title, _ in st.session_state.last_recs:
            html_list += f"<li><b>{title}</b></li>"
        html_list += "</ol>"

        # Render it
        st.markdown(html_list, unsafe_allow_html=True)


# ---------- FOOTER ----------
st.write("---")
st.caption("💡 Tip: Add movies from different genres for better recommendations!")
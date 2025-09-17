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



# ---------- LOAD RECOMMENDATIONS ENGINE ----------
try:
    mrengine = MovieRecommendationsEngine(MAPPING_CSV, FACTORS_NPY)
except Exception as e:
    st.error(
        f"Could not load recommendations engine artifacts. Make sure `{MAPPING_CSV}` and `{FACTORS_NPY}` exist.\n\nError: {e}"
    )
    st.stop()


# ---------- HELPERS ----------
def _get_title_for(movieid: int) -> str:
    """
    Get the movie title for a given movieId from the recommendations engine's mapping.

    Args:
        movieid (int): Movie ID to look up.
    Returns:
        str: Movie title if found, else the movieId as string.
    """
    row = mrengine.mapping[mrengine.mapping["movieId"] == movieid]
    if len(row):
        return row["title"].values[0]
    return str(movieid)


def _add_favorite(movieid: int):
    """
    Add a movieId to the favorites list in session state if not already present.

    Args:
        movieid (int): Movie ID to add.
    """
    if movieid not in st.session_state.favorites:
        st.session_state.favorites.append(movieid)


def _remove_favorite_cb(movieid: int):
    """
    Callback to remove a single movieId from favorites in session state.

    Args:
        movieid (int): Movie ID to remove.
    """
    favs = st.session_state.get("favorites", [])
    st.session_state.favorites = [m for m in favs if m != movieid]
    # clear last_recs so UI updates sensibly
    st.session_state.last_recs = []


def _clear_favorites_cb():
    """
    Callback to clear all favorites from session state.
    """
    st.session_state.favorites = []
    st.session_state.last_recs = []


def _render_search_results(results):
    """
    Render a list of search results as selectable movie cards in the UI.

    Args:
        results (list): List of dicts from mrengine.search_titles.
    """
    for row in results:
        title = row.get("title") or row.get("title") or ""
        movieid = int(row.get("movieId"))

        col_a, col_b = st.columns([8, 1])
        with col_a:
            st.markdown(f"**{title}**")
        with col_b:
            if st.button("➕", key=f"add_{movieid}", help="Add to favorites"):
                _add_favorite(movieid)


# ---------- SESSION STATE ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "last_search" not in st.session_state:
    st.session_state.last_search = {"query": "", "results": []}
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []


# ---------- PAGE HEADER ----------
st.title("🎬 Movie Recommendations")
st.write(
    "Search for movies, add them to your favorites list and get personalized suggestions."
)

left_col, right_col = st.columns([3, 3], gap="large")


# ------- LEFT PANEL: Search -------
with left_col:
    st.subheader("Search")

    col1, col2 = st.columns([8, 1])
    with col1:
        q = st.text_input(
            "Search by title",
            value=st.session_state.last_search["query"],
            placeholder="Enter a title (e.g. Matrix)",
            label_visibility="collapsed",
            key="search_input",
        )
    with col2:
        search_btn = st.button(
            "🔍", help="Search movie", use_container_width=True, key="search_btn"
        )

    # Do search when pressing button or when query changed
    safe_q = q if q is not None else ""
    if search_btn or (
        safe_q and safe_q != st.session_state.last_search["query"] and safe_q.strip()
    ):
        hits = mrengine.search_titles(safe_q, limit=10)
        st.session_state.last_search = {"query": safe_q, "results": hits}
    else:
        hits = st.session_state.last_search["results"]

    # Render top 10 results
    _render_search_results(hits[:10])


# ------- RIGHT PANEL: Favorites -------
with right_col:
    st.subheader("Favorites")
    if st.session_state.favorites:
        # Render favorites in a vertical list with a remove button per item
        for mid in list(st.session_state.favorites):
            title = _get_title_for(mid)
            col1, col2 = st.columns([8, 1])
            col1.write(f"**{title}**")
            
            # Use on_click callback; key must be unique and stable:
            col2.button(
                "❌",
                help="Remove from favorites",
                key=f"rm_{mid}",
                on_click=_remove_favorite_cb,
                args=(mid,),
            )

        # Add a remove all button
        st.button(
            "❌ All",
            help="Remove all favorites",
            key="clear_favorites_btn",
            on_click=_clear_favorites_cb,
        )


# ------- BOTTOM PANEL: Recommendations -------
st.write("---")
st.subheader("Recommendations")

if not st.session_state.favorites:
    st.info("No favorites yet — add movies from the search results.")
else:
    n = st.slider(
        "How many recommendations would you like?", min_value=5, max_value=20, value=5
    )
    if st.button("🪄 Recommend", type="primary", key="recommend_btn"):
        with st.spinner("Computing recommendations..."):
            recs = mrengine.recommend_from_favorites(
                st.session_state.favorites, top_n=n
            )
            st.session_state.last_recs = recs

    # Show last computed recommendations
    if st.session_state.last_recs:
        # Display as cards list
        for title, score in st.session_state.last_recs:
            c1, c2 = st.columns([8, 1])
            with c1:
                c1.write(f"**{title}**")
            with c2:
                c2.caption(f"score: {score:.4f}")


# ---------- PAGE FOOTER ----------
st.write("---")
st.caption("Tip: For better matches try several favorites of different genres.")
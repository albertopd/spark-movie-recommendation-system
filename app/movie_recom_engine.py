import numpy as np
import pandas as pd
from typing import List, Tuple


class MovieRecommendationsEngine:
    """
    Engine for movie recommendations using precomputed item factors and a mapping table.

    Attributes:
        mapping (pd.DataFrame): DataFrame with movieId, index, and title columns.
        factors (np.ndarray): Numpy array of item factor vectors.
        title_to_index (dict): Maps lowercase movie titles to row index in factors array.
    """

    def __init__(self, mapping_csv_path: str, factors_npy_path: str):
        """
        Initialize the recommendation engine.

        Args:
            mapping_csv_path (str): Path to CSV file with columns title, index.
            factors_npy_path (str): Path to .npy file with item factor vectors.
        Raises:
            AssertionError: If mapping and factors row counts do not match.
        """
        self.mapping = pd.read_csv(mapping_csv_path)
        self.factors = np.load(factors_npy_path)

        assert (
            len(self.mapping) == self.factors.shape[0]
        ), "Mapping and factors row counts do not match"

        # Build quick lookup
        self.title_to_index = {}
        for _, r in self.mapping.iterrows():
            title = str(r["title"]).strip()
            self.title_to_index.setdefault(title.lower(), []).append(int(r["index"]))

    def search_titles(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search for movie titles containing the query string (case-insensitive).

        Args:
            query (str): Substring to search for in movie titles.
            limit (int): Maximum number of results to return.
        Returns:
            List[dict]: List of matching movie records as dicts.
        """
        q = query.lower().strip()
        if not q:
            return []
        
        candidates = self.mapping[
            self.mapping["title"].str.lower().str.contains(q, na=False)
        ]
        
        return candidates.head(limit).to_dict(orient="records")

    def recommend_from_favorites(
        self, favorite_movie_indexes: List[int], top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recommend movies based on a list of favorite movieIds.

        Args:
            favorite_movie_indexes (List[int]): List of favorite movie indexes.
            top_n (int): Number of recommendations to return.
        Returns:
            List[Tuple[str, float]]: List of (title, score) tuples for recommended movies.
        """
        if not favorite_movie_indexes:
            return []

        # Build synthetic user vector by averaging item factor vectors of favorites
        fav_vectors = self.factors[favorite_movie_indexes]
        user_vec = fav_vectors.mean(axis=0)

        # Compute scores as dot product with all item factors
        scores = self.factors.dot(user_vec)

        # Exclude favorites
        fav_idx_set = set(favorite_movie_indexes)
        ranked = [
            (i, float(scores[i])) for i in np.argsort(-scores) if i not in fav_idx_set
        ]

        results = []
        for idx, score in ranked[:top_n]:
            row = self.mapping[self.mapping["index"] == idx].iloc[0]
            results.append((row["title"], score))

        return results

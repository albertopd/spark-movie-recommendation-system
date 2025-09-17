import numpy as np
import pandas as pd
from typing import List, Tuple


class MovieRecommendationsEngine:
    """
    Engine for movie recommendations using precomputed item factors and a mapping table.

    Attributes:
        mapping (pd.DataFrame): DataFrame with movieId, index, and title columns.
        factors (np.ndarray): Numpy array of item factor vectors.
        title_to_ids (dict): Maps lowercase movie titles to lists of movieIds.
        movieid_to_index (dict): Maps movieId to row index in factors array.
    """
    def __init__(self, mapping_csv_path: str, factors_npy_path: str):
        """
        Initialize the recommendation engine.

        Args:
            mapping_csv_path (str): Path to CSV file with columns movieId, index, title.
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
        self.title_to_ids = {}
        for _, r in self.mapping.iterrows():
            title = str(r["title"]).strip()
            self.title_to_ids.setdefault(title.lower(), []).append(int(r["movieId"]))

        # movieId -> index
        self.movieid_to_index = dict(
            zip(self.mapping["movieId"].astype(int), self.mapping["index"].astype(int))
        )

    def search_titles(self, query: str, limit: int = 20):
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
        return candidates.iloc[:limit].to_dict(orient="records")

    def _index_for_movieid(self, movieid: int):
        """
        Get the index in the factors array for a given movieId.

        Args:
            movieid (int): Movie ID to look up.
        Returns:
            int or None: Index if found, else None.
        """
        return self.movieid_to_index.get(int(movieid), None)

    def recommend_from_favorites(
        self, favorite_movieids: List[int], top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Recommend movies based on a list of favorite movieIds.

        Args:
            favorite_movieids (List[int]): List of favorite movie IDs.
            top_n (int): Number of recommendations to return.
        Returns:
            List[Tuple[str, float]]: List of (title, score) tuples for recommended movies.
        """
        # Build synthetic user vector by averaging item factor vectors of favorites (ignores missing ids)
        idxs = [self._index_for_movieid(mid) for mid in favorite_movieids]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return []
        fav_vectors = self.factors[idxs]
        user_vec = fav_vectors.mean(axis=0)

        # Compute scores as dot product with all item factors
        scores = self.factors.dot(user_vec)

        # Exclude favorites
        fav_idx_set = set(idxs)
        ranked = [
            (i, float(scores[i])) for i in np.argsort(-scores) if i not in fav_idx_set
        ]

        results = []
        for idx, score in ranked[:top_n]:
            row = self.mapping[self.mapping["index"] == idx].iloc[0]
            results.append((row["title"], score))
        return results

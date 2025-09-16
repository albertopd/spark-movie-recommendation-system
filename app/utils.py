import numpy as np
import pandas as pd
from typing import List, Tuple


class MovieRecommendationsEngine:
    def __init__(self, mapping_csv: str, factors_npy: str):
        # mapping_csv: contains movieId,index,original_title
        self.mapping = pd.read_csv(mapping_csv)
        self.factors = np.load(factors_npy)
        # build quick lookup
        self.title_to_ids = {}
        for _, r in self.mapping.iterrows():
            title = str(r["original_title"]).strip()
            self.title_to_ids.setdefault(title.lower(), []).append(int(r["movieId"]))

        # movieId -> index
        self.movieid_to_index = dict(
            zip(self.mapping["movieId"].astype(int), self.mapping["index"].astype(int))
        )

    def search_titles(self, query: str, limit: int = 20):
        q = query.lower().strip()
        if not q:
            return []
        candidates = self.mapping[
            self.mapping["original_title"].str.lower().str.contains(q, na=False)
        ]
        return candidates.iloc[:limit].to_dict(orient="records")

    def _index_for_movieid(self, movieid: int):
        return self.movieid_to_index.get(int(movieid), None)

    def recommend_from_favorites(
        self, favorite_movieids: List[int], top_n: int = 20
    ) -> List[Tuple[str, float]]:
        # Build synthetic user vector by averaging item factor vectors of favorites (ignores missing ids)
        idxs = [self._index_for_movieid(mid) for mid in favorite_movieids]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return []
        fav_vectors = self.factors[idxs]
        user_vec = fav_vectors.mean(axis=0)

        # compute scores as dot product with all item factors
        scores = self.factors.dot(user_vec)

        # exclude favorites
        fav_idx_set = set(idxs)
        ranked = [
            (i, float(scores[i])) for i in np.argsort(-scores) if i not in fav_idx_set
        ]

        results = []
        for idx, score in ranked[:top_n]:
            row = self.mapping[self.mapping["index"] == idx].iloc[0]
            results.append((row["original_title"], score))
        return results

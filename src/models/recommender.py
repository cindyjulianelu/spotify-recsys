"""
Track recommender — Stage 2 of the two-stage recommendation pipeline.

Given the genre probability embeddings produced by the classifier,
builds a nearest-neighbour index over all tracks. At query time,
retrieves the k tracks whose embedding vectors are most similar to
the query track's vector (cosine similarity).

Why cosine similarity over Euclidean distance:
- Probability vectors sum to 1 and sit on a simplex
- Cosine similarity measures the angle between vectors, not their
  magnitude — two tracks can have different total "certainty" but
  still be similar in style
- A track that is 80% jazz reads as more similar to a 70% jazz
  track than to a 40% jazz / 40% blues track, even though all
  three might have similar Euclidean distances to each other

Why this goes beyond a simple genre filter:
- Genre boundaries are hard; a jazz-influenced hip-hop track would
  never appear in a pure genre-filter recommendation for "jazz"
- Probability embeddings are soft: that track's vector will have
  meaningful weight on both jazz and hip-hop, so it surfaces for
  listeners of either
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

DISPLAY_COLS = ["track_name", "artists", "album_name", "track_genre", "popularity"]


class TrackRecommender:
    """
    Content-based recommender using genre probability embeddings.

    Usage
    -----
    rec = TrackRecommender()
    rec.fit(embeddings, cleaned_df)
    rec.recommend_by_name("Clair de Lune", k=10)
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._nn: NearestNeighbors | None = None
        self._embeddings: np.ndarray | None = None
        self._tracks: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, tracks: pd.DataFrame) -> "TrackRecommender":
        """
        Build the nearest-neighbour index.

        Parameters
        ----------
        embeddings : np.ndarray, shape (n_tracks, n_genres)
            Probability matrix from classifier.get_embeddings().
            Must be row-aligned with `tracks`.
        tracks : pd.DataFrame
            Cleaned dataframe — used to retrieve metadata for display.
            Must be row-aligned with `embeddings`.

        Returns
        -------
        self (for chaining)
        """
        self._embeddings = embeddings
        self._tracks = tracks.reset_index(drop=True)
        self._nn = NearestNeighbors(
            # 21 so we can always drop the query itself and still
            # return up to 20 results without a second kneighbors call
            n_neighbors=min(21, len(tracks)),
            metric=self.metric,
            algorithm="brute",
            n_jobs=-1,
        )
        self._nn.fit(embeddings)
        return self

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def recommend(self, idx: int, k: int = 10) -> pd.DataFrame:
        """
        Return the top-k most similar tracks for the track at row `idx`.

        Parameters
        ----------
        idx : int   — row index into the fitted embedding matrix
        k   : int   — number of recommendations to return

        Returns
        -------
        pd.DataFrame with columns: similarity, track_name, artists,
        album_name, track_genre, popularity
        """
        self._check_fitted()
        query = self._embeddings[idx].reshape(1, -1)
        distances, indices = self._nn.kneighbors(query, n_neighbors=min(k + 1, len(self._tracks)))

        pairs = [(i, d) for i, d in zip(indices[0], distances[0]) if i != idx][:k]
        if not pairs:
            return pd.DataFrame(columns=["similarity"] + DISPLAY_COLS)

        rec_idx, rec_dist = zip(*pairs)
        result = self._tracks.iloc[list(rec_idx)][DISPLAY_COLS].copy()
        result.insert(0, "similarity", [round(1.0 - d, 4) for d in rec_dist])
        return result.reset_index(drop=True)

    def recommend_by_id(self, track_id: str, k: int = 10) -> pd.DataFrame:
        """
        Recommend by Spotify track_id string.

        If the same track_id appears under multiple genres (which is
        intentional in this dataset), uses the first occurrence.
        """
        self._check_fitted()
        matches = self._tracks[self._tracks["track_id"] == track_id]
        if matches.empty:
            raise KeyError(f"track_id '{track_id}' not found in index.")
        return self.recommend(int(matches.index[0]), k=k)

    def recommend_by_name(self, track_name: str, k: int = 10) -> pd.DataFrame:
        """
        Recommend by track name (case-insensitive, first match).

        Useful for interactive exploration and demos.
        """
        self._check_fitted()
        matches = self._tracks[
            self._tracks["track_name"].str.lower() == track_name.lower()
        ]
        if matches.empty:
            raise KeyError(f"Track '{track_name}' not found in index.")
        return self.recommend(int(matches.index[0]), k=k)

    def lookup(self, track_name: str) -> pd.DataFrame:
        """
        Show all indexed rows that match a track name (case-insensitive).

        Helpful when the same song appears under multiple genres —
        lets you pick which entry to query from.
        """
        self._check_fitted()
        mask = self._tracks["track_name"].str.lower() == track_name.lower()
        return self._tracks[mask][DISPLAY_COLS].reset_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "recommender.pkl") -> Path:
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / name
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(name: str = "recommender.pkl") -> "TrackRecommender":
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"No saved recommender at {path}. Run train.py first."
            )
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._nn is None:
            raise RuntimeError(
                "Recommender has not been fitted. Call .fit(embeddings, tracks) first."
            )

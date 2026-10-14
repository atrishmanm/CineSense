"""
CineSense Real-Time Online Feature Store & Point-in-Time Joins
==============================================================

Implements:
1. Low-latency online feature store for sub-millisecond feature vector retrieval.
2. Sliding-window user engagement aggregators (recent clicks, dominant genres).
3. Exponentially decaying dynamic user taste vector updates.
4. Point-in-Time joins to prevent temporal feature leakage during training and offline evaluation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class OnlineFeatureStore:
    """
    In-memory low-latency online feature store.
    Stores pre-computed user and item feature representations with sub-millisecond retrieval.
    """

    def __init__(self, decay_rate: float = 0.10) -> None:
        self.decay_rate = decay_rate
        # user_id -> feature dict
        self._user_features: Dict[int, Dict[str, Any]] = defaultdict(dict)
        # item_id -> feature dict
        self._item_features: Dict[int, Dict[str, Any]] = defaultdict(dict)
        # user_id -> deque of (timestamp, movie_id, rating)
        self._user_activity_stream: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))

    def register_item(
        self,
        item_id: int,
        genres_vector: np.ndarray,
        elo_score: float = 1500.0,
        popularity: float = 1.0,
        release_year: int = 2000,
    ) -> None:
        """Register or update an item's feature vector."""
        self._item_features[item_id] = {
            "genres_vector": genres_vector.astype(np.float32),
            "elo_score": float(elo_score),
            "popularity": float(popularity),
            "release_year": int(release_year),
            "last_updated": time.time(),
        }

    def record_user_interaction(
        self,
        user_id: int,
        movie_id: int,
        rating: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record user interaction event and update dynamic real-time features.
        """
        ts = timestamp or time.time()
        self._user_activity_stream[user_id].append((ts, movie_id, rating))

        # Update dynamic taste vector via Exponential Moving Average
        item_data = self._item_features.get(movie_id)
        if item_data is not None and rating >= 4.0:
            item_vec = item_data["genres_vector"]
            curr_user_vec = self._user_features[user_id].get("taste_vector")
            if curr_user_vec is None:
                new_vec = item_vec.copy()
            else:
                # EMA update
                new_vec = (1.0 - self.decay_rate) * curr_user_vec + self.decay_rate * item_vec
            
            norm = np.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm

            self._user_features[user_id]["taste_vector"] = new_vec
            self._user_features[user_id]["last_interaction_time"] = ts

    def get_user_features(self, user_id: int, window_sec: float = 3600.0) -> Dict[str, Any]:
        """
        Retrieve user online features including sliding-window engagement.
        """
        now = time.time()
        cutoff = now - window_sec

        stream = self._user_activity_stream[user_id]
        recent_interactions = [item for item in stream if item[0] >= cutoff]

        user_dict = self._user_features[user_id]
        return {
            "taste_vector": user_dict.get("taste_vector"),
            "interactions_last_window": len(recent_interactions),
            "total_lifetime_interactions": len(stream),
            "has_history": len(stream) > 0,
        }

    def get_item_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        return self._item_features.get(item_id)

    def batch_get_items(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch feature retrieval (simulates Redis MGET pipeline)."""
        return {iid: self._item_features[iid] for iid in item_ids if iid in self._item_features}

    @staticmethod
    def point_in_time_join(
        interactions: List[Tuple[int, int, float, float]],  # (user_id, item_id, rating, timestamp)
        item_features_history: Dict[int, List[Tuple[float, Dict[str, Any]]]],  # item_id -> [(ts, features)]
    ) -> List[Dict[str, Any]]:
        """
        Guarantees ZERO temporal leakage.
        For each interaction at time T, joins only the item features that existed at time <= T.
        """
        joined_records = []
        for uid, iid, rating, ts in interactions:
            history = item_features_history.get(iid, [])
            # Find the most recent feature record prior to or at ts
            valid_feats = None
            for feat_ts, feats in sorted(history, key=lambda x: x[0]):
                if feat_ts <= ts:
                    valid_feats = feats
                else:
                    break

            if valid_feats is not None:
                record = {
                    "user_id": uid,
                    "item_id": iid,
                    "rating": rating,
                    "timestamp": ts,
                    **valid_feats,
                }
                joined_records.append(record)

        return joined_records

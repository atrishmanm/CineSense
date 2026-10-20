"""
Four self-contained recommendation strategies for benchmarking.

Each strategy operates on in-memory MovieLens-100K data -- no MySQL, no
Redis, no external API.  This guarantees reproducibility and lets the
benchmark run on any machine.

Strategies
----------
A. **Popularity** -- rank items by number of positive ratings.
B. **Content** -- cosine similarity between user genre profile and item
   genre vector.
C. **Preference + ELO** -- ELO scores derived from pairwise comparisons
   simulated from explicit ratings, combined with a per-user preference
   accumulator.
D. **Hybrid** -- weighted combination of A, B, and C with configurable
   weights (default 50 / 30 / 20).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set

import numpy as np

from benchmarks.data_loader import Movie, Rating


# -- Base class ------------------------------------------------------

class BaseRecommender:
    """Common interface for all recommenders."""

    def __init__(self, name: str) -> None:
        self.name = name

    # subclass must implement
    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        raise NotImplementedError

    def score_item(self, user_id: int, item_id: int) -> float:
        raise NotImplementedError

    # default top-K selection
    def recommend(self, user_id: int, k: int, candidates: Set[int]) -> List[int]:
        # Iterate candidates in a stable order so equal scores always break
        # ties the same way regardless of PYTHONHASHSEED / set iteration order.
        scored = [(it, self.score_item(user_id, it)) for it in sorted(candidates)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]

    def recommend_all(
        self,
        eval_users: Set[int],
        k: int,
        user_train_items: Dict[int, Set[int]],
        catalog: Set[int],
    ) -> Dict[int, List[int]]:
        recs: Dict[int, List[int]] = {}
        for uid in sorted(eval_users):
            cands = catalog - user_train_items.get(uid, set())
            if cands:
                recs[uid] = self.recommend(uid, k, cands)
        return recs


# -- A. Popularity baseline -----------------------------------------

class PopularityRecommender(BaseRecommender):
    """Rank by count of positive ratings, then by average rating."""

    def __init__(self, relevance_threshold: float = 4.0) -> None:
        super().__init__("Popularity")
        self.threshold = relevance_threshold
        self._scores: Dict[int, tuple] = {}

    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        pos_counts: Counter = Counter()
        all_ratings: Dict[int, list] = defaultdict(list)

        for r in train_ratings:
            all_ratings[r.item_id].append(r.rating)
            if r.rating >= self.threshold:
                pos_counts[r.item_id] += 1

        for iid, rlist in all_ratings.items():
            self._scores[iid] = (pos_counts[iid], float(np.mean(rlist)), len(rlist))

    def score_item(self, user_id: int, item_id: int) -> float:
        s = self._scores.get(item_id, (0, 0.0, 0))
        # encode as a single float for sorting
        return s[0] * 100 + s[1] * 10 + s[2] * 0.001


# -- B. Content-based (genre cosine similarity) ---------------------

class ContentRecommender(BaseRecommender):
    """User profile = weighted-average genre vector of liked items.

    Candidates are ranked by cosine similarity to this profile.
    """

    def __init__(self, relevance_threshold: float = 4.0) -> None:
        super().__init__("Content")
        self.threshold = relevance_threshold
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.item_vectors: Dict[int, np.ndarray] = {}

    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        for mid, m in movies.items():
            self.item_vectors[mid] = m.genres.copy()

        liked: Dict[int, list] = defaultdict(list)
        for r in train_ratings:
            if r.rating >= self.threshold and r.item_id in self.item_vectors:
                liked[r.user_id].append((r.item_id, r.rating))

        for uid, pairs in liked.items():
            vecs = np.array([self.item_vectors[iid] for iid, _ in pairs])
            wts = np.array([rat for _, rat in pairs])
            self.user_profiles[uid] = np.average(vecs, axis=0, weights=wts)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score_item(self, user_id: int, item_id: int) -> float:
        prof = self.user_profiles.get(user_id)
        vec = self.item_vectors.get(item_id)
        if prof is None or vec is None:
            return 0.0
        return self._cosine(prof, vec)


# -- C. Preference + ELO --------------------------------------------

class PreferenceEloRecommender(BaseRecommender):
    """Simulate pairwise comparisons from explicit ratings, maintain
    global ELO scores *and* per-user preference accumulators.

    The ELO update uses the standard Bradley-Terry expected-score
    formula; the preference accumulator simply tracks net "wins" for
    each item per user.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_elo: float = 1500.0,
        relevance_threshold: float = 4.0,
    ) -> None:
        super().__init__("Preference+ELO")
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.threshold = relevance_threshold
        self.global_elo: Dict[int, float] = {}
        self.user_pref: Dict[int, Dict[int, float]] = {}
        self._elo_min = float("inf")
        self._elo_max = float("-inf")

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        items = {r.item_id for r in train_ratings}
        for it in sorted(items):
            self.global_elo[it] = self.initial_elo

        user_ratings: Dict[int, List[Rating]] = defaultdict(list)
        for r in train_ratings:
            user_ratings[r.user_id].append(r)

        for uid, u_rats in user_ratings.items():
            self.user_pref[uid] = {}
            desc = sorted(u_rats, key=lambda r: r.rating, reverse=True)

            # generate pairwise comparisons from consecutive-rating windows
            for i in range(len(desc)):
                for j in range(i + 1, min(i + 5, len(desc))):
                    if desc[i].rating <= desc[j].rating:
                        continue
                    w, l = desc[i].item_id, desc[j].item_id
                    we, le = self.global_elo[w], self.global_elo[l]
                    exp_w = self._expected(we, le)
                    self.global_elo[w] = we + self.k_factor * (1.0 - exp_w)
                    self.global_elo[l] = le + self.k_factor * (0.0 - (1.0 - exp_w))
                    self.user_pref[uid][w] = self.user_pref[uid].get(w, 0.0) + 0.1
                    self.user_pref[uid][l] = self.user_pref[uid].get(l, 0.0) - 0.1

        if self.global_elo:
            self._elo_min = min(self.global_elo.values())
            self._elo_max = max(self.global_elo.values())

    def _norm_elo(self, elo: float) -> float:
        span = self._elo_max - self._elo_min
        if span <= 0:
            return 0.5
        return (elo - self._elo_min) / span

    def score_item(self, user_id: int, item_id: int) -> float:
        # Audit H1.2 disclosure: under the benchmark protocol this preference
        # term is CONSTANT.  ``user_pref`` is populated only from train-rated
        # items (``fit`` above), and ``BaseRecommender.recommend_all`` removes
        # exactly those items from the candidate set before scoring, so
        # ``pref`` is always 0.0 for every scored candidate and the sigmoid
        # below always returns 0.5.  In this class 0.4 * 0.5 = 0.2 is added to
        # every candidate alike (a constant offset, harmless for ranking).
        elo = self._norm_elo(self.global_elo.get(item_id, self.initial_elo))
        pref = self.user_pref.get(user_id, {}).get(item_id, 0.0)
        pref_norm = 1.0 / (1.0 + np.exp(-5.0 * pref))  # sigmoid
        return 0.6 * elo + 0.4 * pref_norm


# -- D. Hybrid (configurable weights) ------------------------------

class HybridRecommender(BaseRecommender):
    """Weighted combination of Content, Preference, and ELO signals.

    Default weights mirror the production configuration:
    content 50 %, preference 30 %, ELO 20 %.
    """

    def __init__(
        self,
        w_content: float = 0.50,
        w_preference: float = 0.30,
        w_elo: float = 0.20,
        relevance_threshold: float = 4.0,
        name: str | None = None,
    ) -> None:
        label = name or f"Hybrid({w_content:.0%}/{w_preference:.0%}/{w_elo:.0%})"
        super().__init__(label)
        self.w_content = w_content
        self.w_preference = w_preference
        self.w_elo = w_elo

        self._content = ContentRecommender(relevance_threshold)
        self._pref_elo = PreferenceEloRecommender(relevance_threshold=relevance_threshold)

    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        self._content.fit(train_ratings, movies)
        self._pref_elo.fit(train_ratings, movies)

        # Audit H1.2 (STALE-CONST preference term): under the benchmark
        # protocol the hybrid's "preference" component is PROVABLY CONSTANT.
        # ``user_pref`` maps each user only to items they rated in train
        # (PreferenceEloRecommender.fit), and ``recommend_all`` excludes each
        # user's train-rated items from the candidate set before scoring --
        # so ``pref_raw`` is 0.0 for every (user, candidate) pair and
        # sigmoid(-5 * 0) == 0.5 deterministically.  The term contributes a
        # fixed w_preference * 0.5 to every candidate score, i.e. it does NOT
        # differentiate candidates.
        self.preference_active = False

        # Surface the dead term loudly so future reruns cannot quietly compare
        # ablation rows that differ only in this constant weight (e.g. the
        # stored "Equal 33/34/33" vs "Pref-heavy 20/60/20" rows are duplicate
        # configurations).  The weight is KEPT for comparability with already
        # stored results; changing/removing it requires a flagged benchmark
        # rerun (see audit H1.2).
        if self.w_preference != 0.0:
            print(
                f"[audit H1.2] WARNING: {self.name}: the preference component "
                "is constant (0.5) under this protocol -- user_pref items are "
                "train-rated and excluded from candidates. Rows differing only "
                "in w_preference are equivalent configurations; "
                "ablations using it need a rerun to be meaningful."
            )

    def score_item(self, user_id: int, item_id: int) -> float:
        c = self._content.score_item(user_id, item_id)
        elo = self._pref_elo._norm_elo(
            self._pref_elo.global_elo.get(item_id, self._pref_elo.initial_elo)
        )
        pref_raw = self._pref_elo.user_pref.get(user_id, {}).get(item_id, 0.0)
        # Constant under this protocol: see fit() / audit H1.2 disclosure.
        pref = 1.0 / (1.0 + np.exp(-5.0 * pref_raw))

        return self.w_content * c + self.w_preference * pref + self.w_elo * elo


# -- E. Hybrid + popularity prior (RRF fusion) ---------------------

class HybridPopularityRecommender(BaseRecommender):
    """Hybrid signals fused with a global popularity prior.

    Uses reciprocal rank fusion (RRF, k=60): the per-user hybrid ranking and
    the global popularity ranking each vote. Tests whether a popularity prior
    closes the ranking gap to the pure popularity baseline without giving up
    personalization entirely (coverage/novelty stay above popularity-only).
    """

    def __init__(
        self,
        w_content: float = 0.50,
        w_preference: float = 0.30,
        w_elo: float = 0.20,
        relevance_threshold: float = 4.0,
        rrf_k: float = 60.0,
        name: str = "Hybrid+Popularity (RRF)",
    ) -> None:
        super().__init__(name)
        self.rrf_k = rrf_k
        self._hybrid = HybridRecommender(w_content, w_preference, w_elo,
                                         relevance_threshold)
        self._pop = PopularityRecommender(relevance_threshold)

    def fit(self, train_ratings: List[Rating], movies: Dict[int, Movie]) -> None:
        self._hybrid.fit(train_ratings, movies)
        self._pop.fit(train_ratings, movies)

    def score_item(self, user_id: int, item_id: int) -> float:
        return self._hybrid.score_item(user_id, item_id)

    def recommend(self, user_id: int, k: int, candidates: Set[int]) -> List[int]:
        cands = sorted(candidates)  # deterministic tie order
        h_rank = {m: r + 1 for r, m in
                  enumerate(self._hybrid.recommend(user_id, len(cands), set(cands)))}
        p_rank = {m: r + 1 for r, m in
                  enumerate(self._pop.recommend(user_id, len(cands), set(cands)))}
        fused = [(m, 1.0 / (self.rrf_k + h_rank[m]) + 1.0 / (self.rrf_k + p_rank[m]))
                 for m in cands]
        # Break score ties by item id for reproducibility across runs.
        fused.sort(key=lambda x: (-x[1], x[0]))
        return [m for m, _ in fused[:k]]

"""
NeuMF Ensemble Scorer — Production inference for CineSense V2 models.

Loads the 13-model mega-ensemble (5 Phase 1 + 8 Phase 2) trained on
MovieLens-100K and provides scoring APIs:

1. `score(user_idx, movie_idx)` — direct CF score for MovieLens IDs
2. `genre_affinity(user_genres, movie_genres)` — transferable genre
   compatibility score usable with *any* movie (TMDB, etc.)

The genre affinity scorer uses learned genre interaction weights that
transfer across datasets (genre semantics are universal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Model Definitions (mirror the notebook exactly) ────────────────

class NeuMF_Genre(nn.Module):
    """Phase 1 model: NeuMF + Genre Interaction + L2 Reg"""

    def __init__(self, n_users, n_movies, movie_genres, user_genre_pref,
                 mf_dim=48, mlp_dim=48, mlp_layers=[96, 48],
                 genre_dim=16, dropout=0.2):
        super().__init__()
        n_genres = movie_genres.shape[1]

        self.register_buffer('movie_genres_buf', torch.FloatTensor(movie_genres))
        self.register_buffer('user_genre_buf', torch.FloatTensor(user_genre_pref))

        self.user_mf = nn.Embedding(n_users, mf_dim)
        self.movie_mf = nn.Embedding(n_movies, mf_dim)
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.movie_mlp = nn.Embedding(n_movies, mlp_dim)

        self.user_genre_proj = nn.Linear(n_genres, genre_dim, bias=False)
        self.movie_genre_proj = nn.Linear(n_genres, genre_dim, bias=False)

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        layers = []
        prev = mlp_dim * 2
        for s in mlp_layers:
            layers.extend([nn.Linear(prev, s), nn.ReLU(), nn.Dropout(dropout)])
            prev = s
        self.mlp_tower = nn.Sequential(*layers)

        self.predict = nn.Linear(mf_dim + mlp_layers[-1] + genre_dim, 1)

    def forward(self, users, movies, return_reg=False):
        u_mf = self.user_mf(users)
        m_mf = self.movie_mf(movies)
        gmf = u_mf * m_mf

        u_mlp = self.user_mlp(users)
        m_mlp = self.movie_mlp(movies)
        mlp = self.mlp_tower(torch.cat([u_mlp, m_mlp], dim=-1))

        ug = self.user_genre_proj(self.user_genre_buf[users])
        mg = self.movie_genre_proj(self.movie_genres_buf[movies])
        genre = ug * mg

        out = self.predict(torch.cat([gmf, mlp, genre], dim=-1)).squeeze(-1)
        pred = out + self.global_bias + self.user_bias(users).squeeze(-1) + self.movie_bias(movies).squeeze(-1)

        if return_reg:
            reg = (u_mf.pow(2).sum() + m_mf.pow(2).sum() +
                   u_mlp.pow(2).sum() + m_mlp.pow(2).sum() +
                   self.user_bias(users).pow(2).sum() +
                   self.movie_bias(movies).pow(2).sum()) / users.shape[0]
            return pred, reg
        return pred


class NeuMF_V2(nn.Module):
    """Phase 2 model: NeuMF + Genre + Demographics + LayerNorm + GELU"""

    def __init__(self, n_users, n_movies, movie_genres, user_genre_pref,
                 u_age, u_gender, u_occ, n_age_bins=7, n_occs=21,
                 mf_dim=48, mlp_dim=48, mlp_layers=[96, 48],
                 genre_dim=16, demo_dim=8, dropout=0.2):
        super().__init__()
        ng = movie_genres.shape[1]

        self.register_buffer('mg_buf', torch.FloatTensor(movie_genres))
        self.register_buffer('ug_buf', torch.FloatTensor(user_genre_pref))
        self.register_buffer('age_buf', torch.LongTensor(u_age))
        self.register_buffer('gen_buf', torch.LongTensor(u_gender))
        self.register_buffer('occ_buf', torch.LongTensor(u_occ))

        self.user_mf = nn.Embedding(n_users, mf_dim)
        self.movie_mf = nn.Embedding(n_movies, mf_dim)
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.movie_mlp = nn.Embedding(n_movies, mlp_dim)

        self.ug_proj = nn.Linear(ng, genre_dim, bias=False)
        self.mg_proj = nn.Linear(ng, genre_dim, bias=False)

        self.age_emb = nn.Embedding(n_age_bins, demo_dim)
        self.gen_emb = nn.Embedding(2, demo_dim)
        self.occ_emb = nn.Embedding(n_occs, demo_dim)
        self.demo_proj = nn.Linear(demo_dim * 3, demo_dim)

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        layers = []
        prev = mlp_dim * 2
        for s in mlp_layers:
            layers += [nn.Linear(prev, s), nn.LayerNorm(s), nn.GELU(), nn.Dropout(dropout)]
            prev = s
        self.mlp_tower = nn.Sequential(*layers)

        self.predict = nn.Linear(mf_dim + mlp_layers[-1] + genre_dim + demo_dim, 1)

    def forward(self, users, movies, return_reg=False):
        u_mf, m_mf = self.user_mf(users), self.movie_mf(movies)
        gmf = u_mf * m_mf

        u_mlp, m_mlp = self.user_mlp(users), self.movie_mlp(movies)
        mlp = self.mlp_tower(torch.cat([u_mlp, m_mlp], -1))

        genre = self.ug_proj(self.ug_buf[users]) * self.mg_proj(self.mg_buf[movies])
        demo = F.relu(self.demo_proj(torch.cat([
            self.age_emb(self.age_buf[users]),
            self.gen_emb(self.gen_buf[users]),
            self.occ_emb(self.occ_buf[users])], -1)))

        out = self.predict(torch.cat([gmf, mlp, genre, demo], -1)).squeeze(-1)
        pred = out + self.global_bias + self.user_bias(users).squeeze(-1) + self.movie_bias(movies).squeeze(-1)

        if return_reg:
            reg = (u_mf.pow(2).sum() + m_mf.pow(2).sum() +
                   u_mlp.pow(2).sum() + m_mlp.pow(2).sum()) / users.shape[0]
            return pred, reg
        return pred


# ── 18 MovieLens-100K genres (fixed ordering, matches training) ────

ML100K_GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Mapping from common TMDB genre names → ML100K index
_TMDB_TO_ML100K = {
    'action': 0, 'adventure': 1, 'animation': 2, 'family': 3,
    'comedy': 4, 'crime': 5, 'documentary': 6, 'drama': 7,
    'fantasy': 8, 'film-noir': 9, 'horror': 10, 'music': 11,
    'mystery': 12, 'romance': 13, 'science fiction': 14, 'sci-fi': 14,
    'thriller': 15, 'war': 16, 'western': 17,
    # Extra TMDB genres mapped to closest ML100K equivalent
    'history': 16, 'tv movie': 7, 'kids': 3, "children's": 3,
    'musical': 11,
}


class EnsembleScorer:
    """
    Production scorer that loads the trained V2 mega-ensemble.

    Provides two scoring modes:

    1. **CF scoring** — `score(user_idx, movie_idx)` for MovieLens-100K
       user/movie indices.  Uses full 13-model ensemble with optimized
       weights.  RMSE = 0.8932.

    2. **Genre affinity** — `genre_affinity_score(user_genre_vec, movie_genre_vec)`
       uses the *learned* genre interaction projection weights to compute
       a compatibility score for *any* movie, including TMDB movies.
       This transfers across datasets because genre semantics are universal.
    """

    def __init__(self, v2_path='cinesense_v2.pt', v1_path='cinesense_model_final.pt',
                 device=None):
        self.device = device or torch.device('cpu')
        self._loaded = False
        self._v2_path = Path(v2_path)
        self._v1_path = Path(v1_path)

        # Populated on load
        self.v2_models = []
        self.v1_models = []
        self.mega_weights = None
        self.stats = None
        self.genre_proj_weights = []  # Transferable genre interaction weights

    def load(self):
        """Load all models from checkpoints. Call once at startup."""
        if self._loaded:
            return True

        try:
            # ── Load Phase 2 (V2) ──
            if not self._v2_path.exists():
                logger.warning(f"V2 checkpoint not found: {self._v2_path}")
                return False

            ckpt2 = torch.load(self._v2_path, map_location=self.device, weights_only=False)
            self.stats = ckpt2['stats']
            genre_data = ckpt2['genre_data']
            demo_data = ckpt2['demo_data']

            for sd, cfg in zip(ckpt2['v2_state_dicts'], ckpt2['v2_configs']):
                m = NeuMF_V2(
                    self.stats['n_users'], self.stats['n_movies'],
                    genre_data['movie_genre_matrix'], genre_data['user_genre_pref'],
                    demo_data['user_age'], demo_data['user_gender'], demo_data['user_occ'],
                    demo_data['n_age_bins'], demo_data['n_occs'], **cfg
                )
                m.load_state_dict(sd)
                m.to(self.device).eval()
                self.v2_models.append(m)

            # ── Load Phase 1 (V1) ──
            if self._v1_path.exists():
                ckpt1 = torch.load(self._v1_path, map_location=self.device, weights_only=False)
                for sd, cfg in zip(ckpt1['ensemble_state_dicts'], ckpt1['model_configs']):
                    m = NeuMF_Genre(
                        self.stats['n_users'], self.stats['n_movies'],
                        genre_data['movie_genre_matrix'], genre_data['user_genre_pref'],
                        **cfg
                    )
                    m.load_state_dict(sd)
                    m.to(self.device).eval()
                    self.v1_models.append(m)
            else:
                logger.warning(f"V1 checkpoint not found: {self._v1_path} — using V2 only")

            # ── Ensemble weights ──
            n_total = len(self.v2_models) + len(self.v1_models)
            if 'mega_weights' in ckpt2 and len(ckpt2['mega_weights']) == n_total:
                self.mega_weights = np.array(ckpt2['mega_weights'], dtype=np.float32)
            else:
                # Uniform fallback
                self.mega_weights = np.ones(n_total, dtype=np.float32) / n_total

            # ── Ridge stacking coefficients ──
            self.ridge_coefs = np.array(ckpt2.get('ridge_coefs', []), dtype=np.float32)
            self.ridge_intercept = float(ckpt2.get('ridge_intercept', 0.0))

            # ── Extract genre projection weights for transfer scoring ──
            self._extract_genre_projections()

            self._loaded = True
            logger.info(
                f"✅ EnsembleScorer loaded: {len(self.v2_models)} V2 + "
                f"{len(self.v1_models)} V1 models | "
                f"mega_weights={self.mega_weights.shape}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}", exc_info=True)
            return False

    def _extract_genre_projections(self):
        """Extract learned genre projection weights for transfer scoring."""
        self.genre_proj_weights = []
        for m in self.v2_models:
            ug_w = m.ug_proj.weight.detach().cpu().numpy()  # (genre_dim, 18)
            mg_w = m.mg_proj.weight.detach().cpu().numpy()  # (genre_dim, 18)
            self.genre_proj_weights.append((ug_w, mg_w))
        for m in self.v1_models:
            ug_w = m.user_genre_proj.weight.detach().cpu().numpy()
            mg_w = m.movie_genre_proj.weight.detach().cpu().numpy()
            self.genre_proj_weights.append((ug_w, mg_w))

    @property
    def is_loaded(self):
        return self._loaded

    # ── CF Scoring (MovieLens-100K indices) ─────────────────────────

    @torch.no_grad()
    def score(self, user_idx, movie_idx):
        """
        Score a (user, movie) pair using the full mega-ensemble.

        Args:
            user_idx: int or array of MovieLens-100K user indices (0-based)
            movie_idx: int or array of MovieLens-100K movie indices (0-based)

        Returns:
            float or ndarray of predicted ratings in [1, 5]
        """
        if not self._loaded:
            raise RuntimeError("Call .load() first")

        scalar = isinstance(user_idx, (int, np.integer))
        u = torch.LongTensor([user_idx] if scalar else user_idx).to(self.device)
        m = torch.LongTensor([movie_idx] if scalar else movie_idx).to(self.device)

        all_preds = []
        for model in self.v2_models + self.v1_models:
            p = model(u, m).cpu().numpy()
            # Denormalize
            p = p * self.stats['global_std'] + self.stats['global_mean']
            all_preds.append(p)

        all_preds = np.array(all_preds)  # (n_models, batch)

        # Weighted combination
        blend = np.tensordot(self.mega_weights, all_preds, axes=([0], [0]))
        blend = np.clip(blend, 1.0, 5.0)

        return float(blend[0]) if scalar else blend

    # ── Genre Affinity Scoring (transferable to any dataset) ────────

    def genre_affinity_score(self, user_genre_vec, movie_genre_vec):
        """
        Compute learned genre compatibility between a user profile and a movie.

        Uses the trained genre interaction projection weights averaged across
        all 13 models.  These weights capture *which genre combinations* predict
        high ratings — this transfers across datasets since genre semantics
        are universal.

        Args:
            user_genre_vec: np.ndarray of shape (18,) — user's genre preferences
                            (e.g. avg of genre vectors of liked movies)
            movie_genre_vec: np.ndarray of shape (18,) — movie's genre binary vector

        Returns:
            float — genre compatibility score (higher = better match)
        """
        if not self._loaded or not self.genre_proj_weights:
            return 0.0

        scores = []
        for ug_w, mg_w in self.genre_proj_weights:
            # Project both through learned weights: (genre_dim, 18) @ (18,) → (genre_dim,)
            u_proj = ug_w @ user_genre_vec
            m_proj = mg_w @ movie_genre_vec
            # Element-wise product then sum = dot product in projected space
            scores.append(float(np.dot(u_proj, m_proj)))

        # Average across all models
        return float(np.mean(scores))

    def tmdb_genre_to_vec(self, genre_string):
        """
        Convert a TMDB genre string (e.g. 'Action, Drama, Thriller')
        to a MovieLens-100K binary genre vector of shape (18,).

        Args:
            genre_string: comma-separated genre names

        Returns:
            np.ndarray of shape (18,) — binary genre vector
        """
        vec = np.zeros(18, dtype=np.float32)
        if not genre_string:
            return vec

        for g in genre_string.split(','):
            g = g.strip().lower()
            if g in _TMDB_TO_ML100K:
                vec[_TMDB_TO_ML100K[g]] = 1.0

        return vec


# ── Module-level singleton ─────────────────────────────────────────

_scorer = None


def get_scorer(v2_path='cinesense_v2.pt', v1_path='cinesense_model_final.pt'):
    """Get or create the global EnsembleScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = EnsembleScorer(v2_path=v2_path, v1_path=v1_path)
        _scorer.load()
    return _scorer

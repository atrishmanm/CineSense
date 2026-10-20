"""
STEP 4-5: Neural Recommendation Models
- NCF (Neural Collaborative Filtering)
- Hybrid Model (NCF + Content Embeddings)
"""

import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    STEP 4: Neural Collaborative Filtering (NCF)
    
    This learns user taste automatically from behavior
    Netflix-style architecture
    
    Architecture:
        User Embedding (64) + Movie Embedding (64)
        → Concatenate (128)
        → Dense Layers
        → Output Score
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        
        # Embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        
        # Deep layers
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user, movie):
        """
        Args:
            user: [batch_size] user indices
            movie: [batch_size] movie indices
        
        Returns:
            scores: [batch_size] predicted ratings
        """
        u = self.user_emb(user)  # [batch, 64]
        m = self.movie_emb(movie)  # [batch, 64]
        
        x = torch.cat([u, m], dim=1)  # [batch, 128]
        
        return self.fc(x).squeeze()  # [batch]


class HybridRecommender(nn.Module):
    """
    STEP 5: HYBRID DEEP LEARNING MODEL
    
    This is what Netflix actually uses!
    
    Combines:
        - User embeddings (collaborative filtering)
        - Movie CF embeddings
        - Movie plot embeddings (from transformers, 768-dim)
    
    Architecture:
        [User Emb (64)] + [Movie Emb (64)] + [Plot Emb (768)]
        → Concatenate (896)
        → Dense Layers
        → Final Score
    
    🔥 This handles cold-start movies — huge win!
    """
    
    def __init__(self, num_users, num_movies, plot_emb_dim=768, user_emb_dim=64):
        super().__init__()
        
        # Collaborative filtering embeddings
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.movie_emb = nn.Embedding(num_movies, user_emb_dim)
        
        # Deep fusion network
        input_dim = user_emb_dim + user_emb_dim + plot_emb_dim  # 64 + 64 + 768 = 896
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user, movie, plot_emb):
        """
        Args:
            user: [batch_size] user indices
            movie: [batch_size] movie indices
            plot_emb: [batch_size, 768] plot embeddings
        
        Returns:
            scores: [batch_size] predicted ratings
        """
        u = self.user_emb(user)  # [batch, 64]
        m = self.movie_emb(movie)  # [batch, 64]
        
        # Concatenate all features
        x = torch.cat([u, m, plot_emb], dim=1)  # [batch, 896]
        
        return self.fc(x).squeeze()  # [batch]


def _batchnorm_param_keys(model: "HybridRecommender") -> set:
    """Parameter/buffer names for every BatchNorm in the hybrid ``fc`` stack.

    The legacy checkpoints predate the BatchNorm layers, so these are the
    only keys legitimately absent from a remapped legacy state dict.
    """
    bn_keys = set()
    for idx, layer in enumerate(model.fc):
        if isinstance(layer, nn.BatchNorm1d):
            for suffix in ("weight", "bias", "running_mean", "running_var",
                           "num_batches_tracked"):
                bn_keys.add(f"fc.{idx}.{suffix}")
    return bn_keys


def load_hybrid_checkpoint(path, map_location=None):
    """Load a ``HybridRecommender`` checkpoint, remapping legacy state-dict keys.

    Checkpoints saved by the pre-BatchNorm variant of ``HybridRecommender``
    (audit F1.1, ``docs/AUDIT-2026-10-19-independent-recheck.md``) use the
    attribute names ``user_embedding.*`` / ``movie_embedding.*`` and a
    Sequential whose Linear layers sit at indices 0/3/6/9 under ``fusion.*``,
    while the current model uses ``user_emb.*`` / ``movie_emb.*`` and Linear
    layers at indices 0/4/8/12 under ``fc.*`` (BatchNorm/ReLU/Dropout occupy
    the interleaved slots; ReLU/Dropout are parameterless so they never show
    up in a state dict). ``load_state_dict`` with the raw legacy keys raises,
    so every consumer crashed.

    This loader:
      * loads with ``torch.load(..., weights_only=False)`` and unwraps
        ``{'state_dict': ...}`` or ``{'model_state_dict': ...}`` payloads,
        reading ``num_users``/``num_movies`` from the top level when present;
      * loads directly when the keys already use the current naming;
      * otherwise remaps by *rank*: the i-th ``fusion`` Linear layer maps to
        the i-th ``fc`` Linear layer (0→0, 3→4, 6→8, 9→12), and
        ``user_embedding.*``/``movie_embedding.*`` become
        ``user_emb.*``/``movie_emb.*``;
      * instantiates ``HybridRecommender`` using the checkpoint dims, loads
        with ``strict=False`` and hard-fails unless the only missing keys are
        BatchNorm parameters (anything else means the remap failed — a silent
        freshly-initialized model would serve garbage).
      * logs :banner: ``Initialized hybrid model ... BN params left at
        defaults`` because the legacy fc had no BatchNorm layers: the running
        stats stay at init (mean 0 / var 1, gamma 1 / beta 0), so in eval mode
        BatchNorm degrades to an identity transform plus eps — a close but
        not exact match to the legacy model. Pending retrain.

    Args:
        path: path to ``model/hybrid_checkpoint.pt`` / ``hybrid_recommender.pt``.
        map_location: passed to ``torch.load`` (default: torch's own default).

    Returns:
        The loaded, eval-ready ``HybridRecommender``.

    Raises:
        KeyError: if the state dict matches neither key convention.
        RuntimeError: if remapping leaves non-BatchNorm keys missing or any
            unexpected keys — the file is not loadable without a retrain.
    """
    import logging

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, nn.Module):
        raise TypeError(
            f"{path} contains a full pickled nn.Module, expected a state dict "
            "or a checkpoint dict with 'model_state_dict'."
        )
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(ckpt)!r}")

    sd = ckpt
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]

    model = HybridRecommender(4, 4)  # any dims; fc layout is dim-independent

    if any(k.startswith("user_emb.") for k in sd):
        # Already matches the current naming — load strictly as-is.
        num_users = ckpt.get("num_users", sd["user_emb.weight"].shape[0])
        num_movies = ckpt.get("num_movies", sd["movie_emb.weight"].shape[0])
        model = HybridRecommender(
            num_users=num_users, num_movies=num_movies, plot_emb_dim=768
        )
        model.load_state_dict(sd)
        return model

    if "user_embedding.weight" not in sd:
        raise KeyError(
            f"{path} matches neither the current ('user_emb.*'/'fc.*') nor the "
            "legacy ('user_embedding.*'/'fusion.*') HybridRecommender key "
            "convention."
        )

    # Rank-based remap: legacy fusion Linear layers (by ascending index) map
    # onto the current fc Linear layers (0, 4, 8, 12).
    legacy_linear_idx = sorted(
        int(k.split(".")[1]) for k in sd if k.startswith("fusion.") and k.endswith("weight")
    )
    fc_linear_idx = [
        i for i, layer in enumerate(model.fc) if isinstance(layer, nn.Linear)
    ][: len(legacy_linear_idx)]
    rank_map = {
        f"fusion.{legacy}.": f"fc.{current}."
        for legacy, current in zip(legacy_linear_idx, fc_linear_idx)
    }

    remapped, consumed = {}, set()
    for k, tensor in sd.items():
        for legacy_prefix, fc_prefix in rank_map.items():
            if k.startswith(legacy_prefix):
                remapped[fc_prefix + k[len(legacy_prefix):]] = tensor
                consumed.add(k)
                break
        else:
            if k.startswith("user_embedding."):
                remapped["user_emb." + k[len("user_embedding."):]] = tensor
                consumed.add(k)
            elif k.startswith("movie_embedding."):
                remapped["movie_emb." + k[len("movie_embedding."):]] = tensor
                consumed.add(k)

    num_users = ckpt.get("num_users", sd["user_embedding.weight"].shape[0])
    num_movies = ckpt.get("num_movies", sd["movie_embedding.weight"].shape[0])
    model = HybridRecommender(
        num_users=num_users, num_movies=num_movies, plot_emb_dim=768
    )

    # strict=False is intentional: the legacy fc had no BatchNorm layers, so
    # the BatchNorm params (fc.{1,5,9}.*) are legitimately missing and stay at
    # their init values. Anything *else* missing or unexpected means the remap
    # failed and a freshly-initialized Linear layer would serve garbage.
    result = model.load_state_dict(remapped, strict=False)
    missing = set(result.missing_keys)
    expected_keys = {
        name for name, _ in model.named_parameters()
    } | {name for name, _ in model.named_buffers()}
    unexpected = [k for k in remapped if k not in expected_keys]

    real_missing = missing - _batchnorm_param_keys(model)
    if real_missing:
        raise RuntimeError(
            f"Remap of {path} left non-BatchNorm keys missing: "
            f"{sorted(real_missing)}. The legacy-key mapping failed — refusing "
            "to return a partially-initialized model (correct the remap or "
            "retrain; serving fresh default weights would be silent garbage)."
        )
    if unexpected:
        raise RuntimeError(
            f"Remap of {path} produced keys not present in HybridRecommender: "
            f"{unexpected}. The legacy-key mapping failed — refusing to "
            "return a partially-initialized model."
        )

    if missing:
        logging.getLogger(__name__).warning(
            "Initialized hybrid model from %s with remapped legacy keys "
            "(%s); BN params left at defaults (pending retrain)",
            path, rank_map,
        )

    return model


def get_model(model_type, num_users, num_movies, plot_emb_dim=768):
    """
    Factory function to get the right model
    
    Args:
        model_type: 'ncf' or 'hybrid'
        num_users: number of users
        num_movies: number of movies
        plot_emb_dim: dimension of plot embeddings (768 for mpnet)
    
    Returns:
        model: NCF or HybridRecommender
    """
    if model_type == 'ncf':
        return NCF(num_users, num_movies)
    elif model_type == 'hybrid':
        return HybridRecommender(num_users, num_movies, plot_emb_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing NCF model...")
    ncf = NCF(num_users=1000, num_movies=5000)
    
    user = torch.randint(0, 1000, (32,))
    movie = torch.randint(0, 5000, (32,))
    
    output = ncf(user, movie)
    print(f"NCF output shape: {output.shape}")
    
    print("\nTesting Hybrid model...")
    hybrid = HybridRecommender(num_users=1000, num_movies=5000)
    
    plot_emb = torch.randn(32, 768)
    output = hybrid(user, movie, plot_emb)
    print(f"Hybrid output shape: {output.shape}")
    
    print("\n✓ Models test passed!")

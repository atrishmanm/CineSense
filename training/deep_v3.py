"""CineSense Deep V3 model definition (import-safe copy).

The canonical training script ``kaggle_runner/train_kaggle_t4x2.py`` executes
training on import (no ``__main__`` guard), so it must only ever be run as a
script or pushed to Kaggle — never imported. This module contains verbatim
copies of its three model classes so checkpoints in ``models/`` can be
loaded, verified, and served locally:

    from training.deep_v3 import CineSenseDeepV3Full32M, load_v3_checkpoint
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetworkV2(nn.Module):
    """Deep & Cross Network v2: Degree-bounded tensor polynomial feature crossing."""
    def __init__(self, in_features: int, num_layers: int = 3):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(in_features, in_features)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(in_features)) for _ in range(num_layers)])
        for w, b in zip(self.weights, self.biases):
            nn.init.xavier_uniform_(w)
            nn.init.zeros_(b)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for w, b in zip(self.weights, self.biases):
            xl = x0 * F.linear(xl, w, b) + xl
        return xl


class FactorizationMachine(nn.Module):
    """Vector Factorization Machine: exact 2nd-order vector inner products in O(kd)."""
    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        sum_embeddings = torch.sum(field_embeddings, dim=1)
        sum_sq_embeddings = torch.sum(field_embeddings ** 2, dim=1)
        sq_sum_embeddings = sum_embeddings ** 2
        return 0.5 * torch.sum(sq_sum_embeddings - sum_sq_embeddings, dim=1, keepdim=True)


class CineSenseDeepV3Full32M(nn.Module):
    def __init__(self, n_users: int, n_items: int, u_dense_mat: np.ndarray, i_dense_mat: np.ndarray, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Register tabular lookup buffers on device
        self.register_buffer("user_dense_lookup", torch.from_numpy(u_dense_mat))
        self.register_buffer("item_dense_lookup", torch.from_numpy(i_dense_mat))

        # Latent Embeddings
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

        # Tabular projections
        self.user_dense_proj = nn.Linear(2, embed_dim)
        self.item_dense_proj = nn.Linear(23, embed_dim)

        # Explicit Biases
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Interaction Engines
        self.fm = FactorizationMachine()
        cross_dim = 4 * embed_dim  # [u_e, i_e, ud_e, id_e]
        self.dcn = CrossNetworkV2(cross_dim, num_layers=3)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Deep Highway MLP with LayerNorm & GELU
        self.mlp = nn.Sequential(
            nn.Linear(cross_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        final_in = 1 + 1 + cross_dim + cross_dim + 32
        self.head = nn.Sequential(
            nn.Linear(final_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, u_ids, i_ids):
        batch_size = u_ids.size(0)

        # Lookup dense features
        u_dense = self.user_dense_lookup[u_ids]
        i_dense = self.item_dense_lookup[i_ids]

        # Linear prior
        linear_part = self.global_bias + self.user_bias(u_ids) + self.item_bias(i_ids)

        # Field embeddings (B, 4, D)
        u_e = self.user_embed(u_ids).unsqueeze(1)
        i_e = self.item_embed(i_ids).unsqueeze(1)
        ud_e = self.user_dense_proj(u_dense).unsqueeze(1)
        id_e = self.item_dense_proj(i_dense).unsqueeze(1)
        fields = torch.cat([u_e, i_e, ud_e, id_e], dim=1)

        # 2nd-order FM interaction
        fm_part = self.fm(fields)

        # Flattened fields for DCN and MLP
        flat_fields = fields.view(batch_size, -1)
        cross_part = self.dcn(flat_fields)

        # Cross Attention
        attn_out, _ = self.cross_attn(fields, fields, fields)
        flat_attn = attn_out.reshape(batch_size, -1)

        # Deep MLP
        deep_part = self.mlp(flat_fields)

        # Combined representation
        combined = torch.cat([linear_part, fm_part, cross_part, flat_attn, deep_part], dim=-1)
        logit = self.head(combined).squeeze(-1)

        pred_norm = torch.sigmoid(logit)
        pred_raw = 0.5 + 4.5 * pred_norm
        return pred_norm, pred_raw


# (n_users, n_items) per bundled checkpoint — verified from state dicts.
KNOWN_SHAPES = {
    "models/kaggle_t4x2_full32m/cinesense_deep_v3_full32m_best.pt": (200948, 84432),
    "models/kaggle_t4x2_3m_subset/cinesense_deep_v3_3m_subset_best.pt": (18915, 42716),
}


def load_v3_checkpoint(path: str | Path, device: str = "cpu"):
    """Instantiate the V3 class at the checkpoint's shapes and load it.

    Dense lookup buffers are zero-filled: weights load exactly, but
    predictions need the real train-split priors for exact reproduction.
    Returns (model, missing, unexpected).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key = path.as_posix()
    if key in KNOWN_SHAPES:
        n_users, n_items = KNOWN_SHAPES[key]
    else:  # infer from embedding tables
        n_users = int(ckpt["user_embed.weight"].shape[0])
        n_items = int(ckpt["item_embed.weight"].shape[0])
    model = CineSenseDeepV3Full32M(
        n_users, n_items,
        np.zeros((n_users, 2), dtype=np.float32),
        np.zeros((n_items, 23), dtype=np.float32),
    )
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, list(missing), list(unexpected)


if __name__ == "__main__":
    for name in KNOWN_SHAPES:
        model, missing, unexpected = load_v3_checkpoint(name)
        print(f"{name}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)

#!/usr/bin/env python3
"""Reproducible ML-100K trainer (CPU, numpy-only).

Trains a biased matrix-factorization model on the bundled MovieLens-100K data
using the same leakage-safe per-user temporal split as the benchmark harness,
then evaluates rating + full-catalog ranking metrics.

Usage:
    python training/train_100k.py [--epochs 15] [--factors 32] [--out model/mf_100k.npz]

Outputs:
    - model/mf_100k.npz               (biases + latent factors + id maps)
    - benchmarks/results/train_100k_metrics.json

This is the reproducible local counterpart to the offline Kaggle torch
checkpoints in models/ (which are NOT reproduced by this script).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.data_loader import (  # noqa: E402
        create_splits, download_and_extract, load_ratings)
from benchmarks.metrics import evaluate_all  # noqa: E402

SEED = 42


class BiasedMF:
    """Biased matrix factorization trained with SGD (deterministic, seed 42)."""

    def __init__(self, n_factors=32, lr=0.015, reg=0.05, n_epochs=15):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, train):
        rng = np.random.RandomState(SEED)
        users = sorted({r.user_id for r in train})
        items = sorted({r.item_id for r in train})
        self.uid = {u: i for i, u in enumerate(users)}
        self.iid = {m: j for j, m in enumerate(items)}
        self.users, self.items = users, items
        n_u, n_i = len(users), len(items)
        self.mu = float(np.mean([r.rating for r in train]))
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors))
        idx = [(self.uid[r.user_id], self.iid[r.item_id], r.rating) for r in train]
        for _ in range(self.n_epochs):
            for u, i, r in idx:  # fixed order => deterministic
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
        return self

    def predict(self, user_id, item_id):
        u = self.uid.get(user_id)
        i = self.iid.get(item_id)
        if u is None or i is None:
            return self.mu
        return float(np.clip(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i], 0.5, 5.0))

    def recommend(self, user_id, candidates, k=10):
        scored = [(c, self.predict(user_id, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:k]]

    def save(self, path):
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi, P=self.P, Q=self.Q,
                 users=np.array(self.users), items=np.array(self.items),
                 n_factors=self.n_factors, lr=self.lr, reg=self.reg,
                 n_epochs=self.n_epochs, seed=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--factors", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="model/mf_100k.npz")
    args = ap.parse_args()

    t0 = time.time()
    download_and_extract()
    splits = create_splits(load_ratings(), seed=SEED)
    train, test = splits["train"], splits["test"]
    eval_users, catalog = splits["eval_users"], splits["catalog"]

    model = BiasedMF(n_factors=args.factors, lr=args.lr, reg=args.reg,
                     n_epochs=args.epochs).fit(train)

    preds = [model.predict(r.user_id, r.item_id) for r in test]
    actual = [r.rating for r in test]
    err = np.array(preds) - np.array(actual)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    user_train = {}
    for r in train:
        user_train.setdefault(r.user_id, set()).add(r.item_id)
    recs = {u: model.recommend(u, catalog - user_train.get(u, set()), args.k)
            for u in eval_users}
    test_rel = {}
    for r in test:
        if r.rating >= 4.0:
            test_rel.setdefault(r.user_id, set()).add(r.item_id)
    n_users = len({r.user_id for r in train})
    counts = {}
    for r in train:
        counts[r.item_id] = counts.get(r.item_id, 0) + 1
    item_pop = {i: c / n_users for i, c in counts.items()}
    from benchmarks.data_loader import load_movies as _lm
    item_feats = {mid: m.genres for mid, m in _lm().items()}
    rank = evaluate_all(recs, test_rel, len(catalog), item_pop, item_feats,
                        args.k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    metrics = {
        "model": "BiasedMF-SGD",
        "dataset": "MovieLens-100K (bundled benchmarks/ml100k_data)",
        "split": "per-user temporal 70/10/20, seed 42 (benchmarks.data_loader)",
        "hyperparams": {"factors": args.factors, "lr": args.lr, "reg": args.reg,
                        "epochs": args.epochs},
        "test_rmse": rmse, "test_mae": mae,
        "ranking": {k2: rank[k2] for k2 in
                    ("precision@10", "recall@10", "ndcg@10", "hit_rate@10",
                     "catalog_coverage", "novelty", "diversity")
                    if k2 in rank},
        "n_eval_users": rank.get("n_eval_users"),
        "train_seconds": round(time.time() - t0, 1),
        "checkpoint": str(out),
    }
    res_path = ROOT / "benchmarks" / "results" / "train_100k_metrics.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(metrics, indent=2))
    print(f"test RMSE={rmse:.4f} MAE={mae:.4f} "
          f"NDCG@10={rank.get('ndcg@10', float('nan')):.4f} "
          f"Hit@10={rank.get('hit_rate@10', float('nan')):.4f} "
          f"({rank.get('n_eval_users')} users, full catalog)")
    print(f"saved {out} + {res_path}")


if __name__ == "__main__":
    main()

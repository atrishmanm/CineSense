# Model Artifacts (gitignored + tracked checkpoints)

Trained artifacts under **`model/`** are **gitignored**
(`model/*.pt`, `model/*.pkl`, `model/*.npy`, `model/*.json`, `model/*.csv`, ...),
so a fresh clone contains **no** legacy checkpoints. The Kaggle Deep V3
checkpoints under **`models/`** (tracked, trained on Kaggle dual T4x2) are the
only models shipped with the repository.

Every loader in `training/`, `inference/` and `serving/` fails loudly with an
actionable message when an artifact is missing (D385/D387); this README
documents how to (re)produce each one.

## Legacy ML-100K stack (`model/`)

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `model/mappings.pkl` | `python -m preprocessing.feature_engineering` | `training/train.py`, `training/advanced_training.py`, `inference/recommend.py` |
| `model/plot_embeddings.npy` (768-d mpnet) | `python -m preprocessing.feature_engineering` | hybrid training + `inference/recommend.py` |
| `model/movie_metadata.csv` | `python -m preprocessing.feature_engineering` | `inference/recommend.py` |
| `model/ncf_recommender.pt` | `python -m training.train` (NCF phase) | `inference/recommend.py` (model_type='ncf') |
| `model/hybrid_recommender.pt` | `python -m training.train` (hybrid phase) | `inference/recommend.py` (model_type='hybrid') |
| `model/{ncf,hybrid}_checkpoint.pt`, `model/*_history.json` | `python -m training.train` (per-epoch checkpoints / history) | resume + training curves |

Pipeline order for the legacy stack:

```text
data/ (MovieLens-32M + TMDB raw csv, see CINESENSE_DATA_DIR in preprocessing)
  -> python -m preprocessing.merge_datasets        # data/merged.csv
  -> python -m preprocessing.feature_engineering   # model/mappings.pkl, plot_embeddings.npy, ...
  -> python -m training.train                      # model/*_recommender.pt
  -> python -m inference.recommend                 # sanity-checks the .pt/.pkl/.npy loads
```

## Kaggle Deep V3 checkpoints (`models/`)

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `models/kaggle_t4x2_full32m/cinesense_deep_v3_full32m_best.pt` (+ `cinesense_deep_v2_best.pt`, `metrics.json`) | `kaggle_runner/train_kaggle_t4x2.py` on Kaggle dual T4x2, full 32M | `serving/model_handler.py` (default candidate), `training/deep_v3.load_v3_checkpoint` |
| `models/kaggle_t4x2_3m_subset/cinesense_deep_v3_3m_subset_best.pt` | same script on the 3M-subset Kaggle run | same as above |
| `models/kaggle_t4x2/cinesense_t4x2_best.pt` | earlier T4x2 run | archive / metrics.json only |

To regenerate the 32M checkpoints: upload `kaggle_runner/train_kaggle_t4x2.py`
to a Kaggle notebook with the MovieLens-32M dataset and a dual-T4 GPU; it
writes `cinesense_deep_v3_full32m_best.pt` + `metrics.json`, which are then
copied under `models/kaggle_t4x2_full32m/`.

## Serving exports

- `serving/model_handler.py` resolves checkpoints via `MODEL_PATH`, the
  TorchServe `model_dir`, `config.CHECKPOINT_DIR/*.pt` (newest first), or the
  tracked `models/` paths above.
- `inference/torchserve_handler.export_model_for_torchserve(model_path)` copies
  the checkpoint and, if needed, writes a default `genre_encoder.json`
  (MovieLens-32M 20-genre order) plus `plot_embeddings.npy` when present.

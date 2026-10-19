"""
MovieLens-100K Data Loader
Downloads, parses, and creates reproducible per-user temporal train/val/test splits.
"""

import ssl
import zipfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Set

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "ml100k_data"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class Rating(NamedTuple):
    user_id: int
    item_id: int
    rating: float
    timestamp: int


class Movie(NamedTuple):
    movie_id: int
    title: str
    genres: np.ndarray  # binary vector of shape (19,)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_and_extract() -> None:
    """Download MovieLens-100K and extract once."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".downloaded"
    if marker.exists():
        return

    zip_path = DATA_DIR / "ml-100k.zip"
    print(f"  Downloading MovieLens-100K from {ML100K_URL} ...")
    try:
        urllib.request.urlretrieve(ML100K_URL, str(zip_path))
    except urllib.error.URLError:
        # Fallback: skip SSL verification (safe for this known URL)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(ML100K_URL, context=ctx) as resp:
            with open(zip_path, "wb") as out:
                out.write(resp.read())

    print("  Extracting ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(DATA_DIR))

    zip_path.unlink()
    marker.touch()
    print("  MovieLens-100K ready.\n")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ratings() -> List[Rating]:
    """Load all ratings from ``u.data`` (tab-separated)."""
    download_and_extract()
    ratings: List[Rating] = []
    data_file = DATA_DIR / "ml-100k" / "u.data"
    with open(data_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                ratings.append(
                    Rating(
                        user_id=int(parts[0]),
                        item_id=int(parts[1]),
                        rating=float(parts[2]),
                        timestamp=int(parts[3]),
                    )
                )
    return ratings


def load_movies() -> Dict[int, Movie]:
    """Load movie metadata from ``u.item`` (pipe-separated, ISO-8859-1)."""
    download_and_extract()
    movies: Dict[int, Movie] = {}
    item_file = DATA_DIR / "ml-100k" / "u.item"
    with open(item_file, "r", encoding="iso-8859-1") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) >= 24:
                movie_id = int(parts[0])
                title = parts[1]
                genres = np.array([int(parts[i]) for i in range(5, 24)], dtype=np.float32)
                movies[movie_id] = Movie(movie_id=movie_id, title=title, genres=genres)
    return movies


# ---------------------------------------------------------------------------
# MovieLens-32M Loaders
# ---------------------------------------------------------------------------

GENRE_NAMES_32M = [
    "(no genres listed)", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_TO_IDX_32M = {g: idx for idx, g in enumerate(GENRE_NAMES_32M)}


def load_movielens_32m_movies(data_dir: Path | str | None = None) -> Dict[int, Movie]:
    """Load MovieLens-32M movies with 20-dimensional binary genre vectors."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "movie-lens_ml-32m"
    data_dir = Path(data_dir)
    movies_file = data_dir / "movies.csv"

    if not movies_file.exists():
        raise FileNotFoundError(f"MovieLens-32M movies.csv not found at {movies_file}")

    movies: Dict[int, Movie] = {}
    import csv
    with open(movies_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # consume header: movieId,title,genres
        for row in reader:
            if len(row) < 3:
                continue
            mid = int(row[0])
            title = row[1]
            genre_str = row[2]
            vec = np.zeros(len(GENRE_NAMES_32M), dtype=np.float32)
            for g in genre_str.split("|"):
                idx = GENRE_TO_IDX_32M.get(g.strip())
                if idx is not None:
                    vec[idx] = 1.0
            movies[mid] = Movie(movie_id=mid, title=title, genres=vec)
    return movies


def load_movielens_32m_ratings(
    data_dir: Path | str | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> List[Rating]:
    """Load MovieLens-32M ratings, optionally taking a uniform random sample.

    When *sample_size* is given, a seeded reservoir sample of exactly
    ``min(sample_size, n_rows)`` ratings is returned.  This is a genuine
    uniform sample over the whole file (not the first N rows), so the
    benchmark is not biased toward the users/movies that happen to appear
    first in ``ratings.csv``.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "movie-lens_ml-32m"
    data_dir = Path(data_dir)
    ratings_file = data_dir / "ratings.csv"

    if not ratings_file.exists():
        raise FileNotFoundError(f"MovieLens-32M ratings.csv not found at {ratings_file}")

    ratings: List[Rating] = []
    import csv

    if sample_size is None:
        with open(ratings_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # userId,movieId,rating,timestamp
            for row in reader:
                if len(row) < 4:
                    continue
                ratings.append(
                    Rating(
                        user_id=int(row[0]),
                        item_id=int(row[1]),
                        rating=float(row[2]),
                        timestamp=int(row[3]),
                    )
                )
        return ratings

    # Seeded reservoir sampling: O(sample_size) memory, uniform over the file.
    rng = np.random.RandomState(seed)
    reservoir: List[Rating] = []
    with open(ratings_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # userId,movieId,rating,timestamp
        for i, row in enumerate(reader):
            if len(row) < 4:
                continue
            rec = Rating(
                user_id=int(row[0]),
                item_id=int(row[1]),
                rating=float(row[2]),
                timestamp=int(row[3]),
            )
            if len(reservoir) < sample_size:
                reservoir.append(rec)
            else:
                j = int(rng.randint(0, i + 1))
                if j < sample_size:
                    reservoir[j] = rec

    return reservoir


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def create_splits(
    ratings: List[Rating],
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    min_interactions: int = 5,
    relevance_threshold: float = 4.0,
    seed: int = 42,
) -> dict:
    """Create *per-user temporal* train / val / test splits.

    For each user with >= *min_interactions* ratings the most recent
    *test_ratio* fraction goes to the test set, the next *val_ratio*
    fraction to validation, and the rest to training.  Users with fewer
    interactions contribute exclusively to training.

    Returns a dict with keys ``train``, ``val``, ``test`` (lists of
    :class:`Rating`), ``eval_users`` (set of user ids that appear in both
    train and test), and ``catalog`` (set of all item ids).
    """
    # group by user and sort chronologically
    user_ratings: Dict[int, List[Rating]] = defaultdict(list)
    for r in ratings:
        user_ratings[r.user_id].append(r)

    for uid in user_ratings:
        user_ratings[uid].sort(key=lambda r: r.timestamp)

    train: List[Rating] = []
    val: List[Rating] = []
    test: List[Rating] = []
    eval_users: Set[int] = set()
    catalog: Set[int] = set()

    for uid, u_ratings in user_ratings.items():
        catalog.update(r.item_id for r in u_ratings)
        n = len(u_ratings)

        if n < min_interactions:
            train.extend(u_ratings)
            continue

        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            train.extend(u_ratings)
            continue

        train.extend(u_ratings[:n_train])
        val.extend(u_ratings[n_train : n_train + n_val])
        test.extend(u_ratings[n_train + n_val :])
        eval_users.add(uid)

    return {
        "train": train,
        "val": val,
        "test": test,
        "eval_users": eval_users,
        "catalog": catalog,
        "relevance_threshold": relevance_threshold,
    }

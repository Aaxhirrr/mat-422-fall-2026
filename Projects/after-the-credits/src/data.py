"""Download MovieLens and keep the user groups separate."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
SEED = 422
BUDGETS = (5, 10, 20)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'ml-1m.zip'
    if not archive.exists():
        temporary = folder / 'ml-1m.download'
        with urllib.request.urlopen(URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        with zipfile.ZipFile(temporary) as z:
            assert z.testzip() is None, 'The download is incomplete.'
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        ratings = pd.read_csv(io.BytesIO(z.read('ml-1m/ratings.dat')),
                              sep='::', engine='python',
                              names=['user', 'movie', 'rating', 'time'])
        movies = pd.read_csv(io.BytesIO(z.read('ml-1m/movies.dat')),
                             sep='::', engine='python', encoding='latin-1',
                             names=['movie', 'title', 'genres']).set_index('movie')
    assert len(ratings) == 1_000_209 and ratings.user.nunique() == 6040
    assert not ratings.isna().any().any()
    assert ratings.rating.between(1, 5).all()
    assert not ratings.duplicated(['user', 'movie']).any()
    return ratings, movies, sha256(archive)


def split_users(ratings, seed=SEED):
    users = np.random.default_rng(seed).permutation(np.sort(ratings.user.unique()))
    train_end = int(len(users) * .70)
    validation_end = train_end + int(len(users) * .15)
    return {
        'train': np.sort(users[:train_end]),
        'validation': np.sort(users[train_end:validation_end]),
        'test': np.sort(users[validation_end:]),
    }


@dataclass
class TrainingData:
    user: np.ndarray
    item: np.ndarray
    rating: np.ndarray
    user_ids: np.ndarray
    movie_ids: np.ndarray
    genres: np.ndarray
    genre_names: list
    titles: list
    popularity: np.ndarray


def training_data(ratings, movies, users):
    frame = ratings[ratings.user.isin(users)]
    user_ids, user = np.unique(frame.user, return_inverse=True)
    movie_ids, item = np.unique(frame.movie, return_inverse=True)
    metadata = movies.loc[movie_ids]
    genre_names = sorted(set('|'.join(metadata.genres).split('|')))
    genres = np.array([[g in value.split('|') for g in genre_names]
                       for value in metadata.genres], dtype=float)
    popularity = np.bincount(item, weights=(frame.rating.to_numpy() >= 4),
                             minlength=len(movie_ids)).astype(float)
    return TrainingData(user, item, frame.rating.to_numpy(dtype=float), user_ids,
                        movie_ids, genres, genre_names, metadata.title.tolist(), popularity)


@dataclass
class Profile:
    user_id: int
    pool: np.ndarray
    ratings: np.ndarray
    positives: np.ndarray
    candidates: np.ndarray


def profiles_for(ratings, users, train):
    # All twenty pool movies stay out of the candidate list, even at budget five.
    frame = ratings[ratings.user.isin(users) & ratings.movie.isin(train.movie_ids)]
    groups = {int(u): rows for u, rows in frame.groupby('user')}
    profiles = []
    counts = {'assigned_users': len(users), 'too_few_ratings': 0, 'no_heldout_positive': 0}
    for user in users:
        rows = groups.get(int(user))
        if rows is None or len(rows) < 30:
            counts['too_few_ratings'] += 1
            continue
        rows = rows.sort_values(['time', 'movie'])
        pool = np.searchsorted(train.movie_ids, rows.movie.iloc[:20].to_numpy())
        heldout = rows.iloc[20:]
        positives = np.searchsorted(train.movie_ids,
                                    heldout.loc[heldout.rating >= 4, 'movie'].to_numpy())
        if not len(positives):
            counts['no_heldout_positive'] += 1
            continue
        candidates = np.ones(len(train.movie_ids), dtype=bool)
        candidates[pool] = False
        profiles.append(Profile(int(user), pool, rows.rating.iloc[:20].to_numpy(dtype=float),
                                positives, candidates))
    counts['evaluated_users'] = len(profiles)
    assert sum(counts[k] for k in ['too_few_ratings', 'no_heldout_positive', 'evaluated_users']) == len(users)
    return profiles, counts

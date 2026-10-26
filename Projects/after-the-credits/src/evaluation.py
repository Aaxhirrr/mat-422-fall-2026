"""Ranking scores use the full eligible catalog, not sampled negatives."""

import numpy as np
import pandas as pd
from .models import genre_scores, percentiles, top_movies


def metrics(recommendations, positives, k=10):
    if len(positives) == 0:
        raise ValueError('Ranking metrics need a held-out positive.')
    hits = np.isin(recommendations[:k], positives).astype(float)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(positives))].sum()
    return float(hits @ discounts[:len(hits)] / ideal), float(hits.sum() / len(positives))


def evaluate(train, profiles, budget, model=None, weights=()):
    rows, covered = {}, {}
    names = ['popularity', 'genre'] if model is None else ['factorization'] + [f'hybrid_{a:g}' for a in weights]
    for name in names:
        rows[name], covered[name] = [], set()
    for p in profiles:
        if model is None:
            options = {'popularity': train.popularity,
                       'genre': genre_scores(train, p.pool[:budget], p.ratings[:budget])}
        else:
            cf = model.scores(p.pool[:budget], p.ratings[:budget])
            options = {'factorization': cf}
            if weights:
                genre = genre_scores(train, p.pool[:budget], p.ratings[:budget])
                cp, gp = percentiles(cf, p.candidates), percentiles(genre, p.candidates)
                options.update({f'hybrid_{a:g}': a * cp + (1 - a) * gp for a in weights})
        for name, scores in options.items():
            top = top_movies(scores, p.candidates, train.movie_ids)
            ndcg, recall = metrics(top, p.positives)
            rows[name].append({'user': p.user_id, 'ndcg10': ndcg, 'recall10': recall})
            covered[name].update(top.tolist())
    details = {name: pd.DataFrame(values) for name, values in rows.items()}
    summary = {name: {'ndcg10': float(frame.ndcg10.mean()),
                      'recall10': float(frame.recall10.mean()),
                      'coverage': len(covered[name]) / len(train.movie_ids),
                      'users': len(frame)} for name, frame in details.items()}
    return summary, details


def paired_interval(left, right, repeats=1000, seed=422):
    joined = left[['user', 'ndcg10']].merge(right[['user', 'ndcg10']], on='user',
                                         suffixes=('_left', '_right'), validate='one_to_one')
    assert len(joined) == len(left) == len(right)
    difference = (joined.ndcg10_left - joined.ndcg10_right).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = difference[rng.integers(0, len(difference), (repeats, len(difference)))].mean(axis=1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return {'mean_difference': float(difference.mean()), 'ci95_low': float(low),
            'ci95_high': float(high), 'users': len(difference), 'resamples': repeats}

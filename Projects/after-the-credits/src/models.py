"""The models are small enough to follow without a recommendation framework."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import rankdata


def groups(index, count):
    order = np.argsort(index, kind='stable')
    return np.split(order, np.cumsum(np.bincount(index, minlength=count))[:-1])


def ridge(design, target, strength):
    return np.linalg.solve(design.T @ design + strength * np.eye(design.shape[1]),
                           design.T @ target)


@dataclass
class FactorModel:
    mean: float
    item_factors: np.ndarray
    item_bias: np.ndarray
    strength: float
    history: list

    def scores(self, items, ratings):
        design = np.column_stack([np.ones(len(items)), self.item_factors[items]])
        target = ratings - self.mean - self.item_bias[items]
        estimate = ridge(design, target, self.strength)
        return self.mean + estimate[0] + self.item_bias + self.item_factors @ estimate[1:]

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, item_factors=self.item_factors,
                            item_bias=self.item_bias, strength=self.strength,
                            history=np.asarray(self.history))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(float(f['mean']), f['item_factors'], f['item_bias'],
                       float(f['strength']), f['history'].tolist())


def fit_factors(train, factors=20, strength=1., epochs=20, seed=422):
    rng = np.random.default_rng(seed)
    nu, ni = len(train.user_ids), len(train.movie_ids)
    users = rng.normal(0, .1, (nu, factors))
    items = rng.normal(0, .1, (ni, factors))
    ub, ib = np.zeros(nu), np.zeros(ni)
    mean = float(train.rating.mean())
    by_user, by_item = groups(train.user, nu), groups(train.item, ni)

    def objective():
        error = 0.
        for start in range(0, len(train.rating), 50_000):
            sl = slice(start, start + 50_000)
            u, i = train.user[sl], train.item[sl]
            pred = mean + ub[u] + ib[i] + np.einsum('ij,ij->i', users[u], items[i])
            error += np.square(train.rating[sl] - pred).sum()
        penalty = strength * (np.square(users).sum() + np.square(items).sum()
                              + ub @ ub + ib @ ib)
        return float(error + penalty)

    history = [objective()]
    for _ in range(epochs):
        design = np.column_stack([np.ones(ni), items])
        for u, idx in enumerate(by_user):
            ids = train.item[idx]
            fit = ridge(design[ids], train.rating[idx] - mean - ib[ids], strength)
            ub[u], users[u] = fit[0], fit[1:]
        design = np.column_stack([np.ones(nu), users])
        for i, idx in enumerate(by_item):
            ids = train.user[idx]
            fit = ridge(design[ids], train.rating[idx] - mean - ub[ids], strength)
            ib[i], items[i] = fit[0], fit[1:]
        history.append(objective())
        if history[-1] > history[-2] + 1e-6 * max(1., history[-2]):
            raise ArithmeticError('The least-squares objective increased.')
        if (history[-2] - history[-1]) / max(history[-2], 1.) < 1e-4:
            break
    return FactorModel(mean, items, ib, strength, history)


def genre_scores(train, items, ratings):
    profile = (ratings - 3.) @ train.genres[items]
    norm = np.linalg.norm(profile)
    if norm == 0:
        return train.popularity.copy()
    denominator = np.linalg.norm(train.genres, axis=1) * norm
    return np.divide(train.genres @ profile, denominator,
                     out=np.zeros(len(train.movie_ids)), where=denominator > 0)


def percentiles(scores, candidates):
    result = np.zeros(len(scores))
    result[candidates] = rankdata(scores[candidates], method='average') / candidates.sum()
    return result


def top_movies(scores, candidates, movie_ids, k=10):
    # Tied scores use movie ID, so rerunning gives the same list.
    eligible = np.flatnonzero(candidates)
    order = np.lexsort((movie_ids[eligible], -scores[eligible]))
    return eligible[order[:k]]

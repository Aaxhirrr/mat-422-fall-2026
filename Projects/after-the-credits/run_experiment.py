"""Run validation first, freeze the choices, then score the final test users."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from src.data import load_data, split_users, training_data, profiles_for, sha256, BUDGETS, SEED
from src.models import fit_factors, FactorModel
from src.evaluation import evaluate, paired_interval

ROOT = Path(__file__).resolve().parent
GRID = [(k, s) for k in [10, 20, 40] for s in [.1, 1., 10.]]
WEIGHTS = (0., .25, .5, .75, 1.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def code_hash():
    files = [Path(__file__), *sorted((ROOT / 'src').glob('*.py'))]
    return hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()


def model_name(k, strength):
    return f'factors_{k}_regularization_{strength:g}'


def make_plots(summary, out):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, frame in summary.groupby('method', sort=False):
        ax.plot(frame.budget, frame.ndcg10, marker='o', label=method)
    ax.set(xlabel='Ratings revealed per user', ylabel='Mean NDCG@10', xticks=BUDGETS,
           title='How much history helps?')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'recommendation_quality.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, frame in summary.groupby('method', sort=False):
        ax.plot(frame.budget, 100 * frame.coverage, marker='o', label=method)
    ax.set(xlabel='Ratings revealed per user', ylabel='Catalog recommended (%)', xticks=BUDGETS,
           title='How much of the catalog gets recommended?')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'catalog_coverage.png', dpi=170)
    plt.close(fig)


def validate(train, ratings, splits, signature, out, epochs):
    if (out / 'test_summary.csv').exists():
        raise RuntimeError('This run already has test results. Use a new output directory for a new study.')
    profiles, cohort = profiles_for(ratings, splits['validation'], train)
    if not profiles:
        raise RuntimeError('No eligible validation users.')
    write_json(out / 'validation_cohort.json', cohort)
    all_rows = []
    for budget in BUDGETS:
        base, _ = evaluate(train, profiles, budget)
        for name, metrics in base.items():
            all_rows.append(dict(budget=budget, method=name, model='none', weight=None, **metrics))
    for k, strength in GRID:
        name = model_name(k, strength)
        cache = out / 'models' / f'{name}.npz'
        cache_info = cache.with_suffix('.json')
        start = time.perf_counter()
        if cache.exists() and cache_info.exists() and json.loads(cache_info.read_text()) == signature:
            model = FactorModel.load(cache)
        else:
            model = fit_factors(train, k, strength, epochs=epochs)
            model.save(cache)
            write_json(cache_info, signature)
        print(f'{name}: fitted {len(model.history)-1} epochs in {time.perf_counter()-start:.1f}s', flush=True)
        write_json(out / f'{name}_training.json', {
            'objective': model.history, 'epochs': len(model.history)-1,
            'regularization': strength, 'factors': k,
        })
        for budget in BUDGETS:
            scored, _ = evaluate(train, profiles, budget, model, WEIGHTS)
            for method, metrics in scored.items():
                weight = float(method.split('_')[1]) if method.startswith('hybrid_') else None
                method = 'hybrid' if weight is not None else 'factorization'
                all_rows.append(dict(budget=budget, method=method, model=name, weight=weight, **metrics))
        pd.DataFrame(all_rows).to_csv(out / 'validation_grid.csv', index=False)
        print(f'{name}: validation complete ({time.perf_counter()-start:.1f}s total)', flush=True)
    grid = pd.DataFrame(all_rows)
    selected = {}
    for budget in BUDGETS:
        selected[str(budget)] = {}
        for method in ['factorization', 'hybrid']:
            options = grid[(grid.budget == budget) & (grid.method == method)]
            # Stable ties follow the predeclared grid order, starting with fewer factors.
            best = options.sort_values('ndcg10', ascending=False, kind='stable').iloc[0]
            selected[str(budget)][method] = {'model': best.model,
                'weight': None if pd.isna(best.weight) else float(best.weight),
                'validation_ndcg10': float(best.ndcg10)}
    frozen = {'signature': signature, 'selected': selected,
              'model_sha256': {p.name: sha256(p) for p in sorted((out/'models').glob('*.npz'))}}
    write_json(out / 'frozen_choices.json', frozen)
    print('Validation choices frozen. The final test users have not been scored.', flush=True)


def test(train, ratings, splits, signature, out):
    frozen_path = out / 'frozen_choices.json'
    if not frozen_path.exists():
        raise RuntimeError('Run validation before the test stage.')
    frozen = json.loads(frozen_path.read_text())
    if frozen['signature'] != signature:
        raise RuntimeError('Data, code, or settings changed after validation. Start a new run.')
    profiles, cohort = profiles_for(ratings, splits['test'], train)
    write_json(out / 'test_cohort.json', cohort)
    summaries, details = [], {}
    for budget in BUDGETS:
        scored, frames = evaluate(train, profiles, budget)
        for method in ['factorization', 'hybrid']:
            choice = frozen['selected'][str(budget)][method]
            path = out / 'models' / (choice['model'] + '.npz')
            assert sha256(path) == frozen['model_sha256'][path.name]
            model = FactorModel.load(path)
            scores, raw = evaluate(train, profiles, budget, model,
                                   () if method == 'factorization' else (choice['weight'],))
            key = method if method == 'factorization' else f"hybrid_{choice['weight']:g}"
            scored[method], frames[method] = scores[key], raw[key]
        for method, values in scored.items():
            summaries.append(dict(budget=budget, method=method, **values))
            frame = frames[method]
            frame.to_csv(out / f'test_{method}_{budget}_users.csv', index=False)
            details[method, budget] = frame
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'test_summary.csv', index=False)
    comparisons = {
        'hybrid_minus_factorization_at_5': paired_interval(details['hybrid',5], details['factorization',5]),
        'factorization_minus_popularity_at_20': paired_interval(details['factorization',20], details['popularity',20]),
        'interval_scope': 'Paired users, conditional on this split and the fitted models; no correction for two comparisons.',
    }
    write_json(out / 'hypothesis_comparisons.json', comparisons)
    write_json(out / 'test_run.json', {'frozen_choices_sha256': sha256(frozen_path),
                                      'signature': signature})
    make_plots(summary, out)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate', 'test'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'models').mkdir(exist_ok=True)
    ratings, movies, archive_hash = load_data(ROOT / 'data')
    splits = split_users(ratings)
    train = training_data(ratings, movies, splits['train'])
    signature = {'data_sha256': archive_hash, 'code_sha256': code_hash(),
                 'seed': SEED, 'epochs': args.epochs,
                 'python': platform.python_version(), 'numpy': np.__version__,
                 'pandas': pd.__version__, 'scipy': scipy.__version__}
    write_json(args.out / 'split_users.json', {k:v.tolist() for k,v in splits.items()})
    write_json(args.out / 'data_manifest.json', {'source': 'MovieLens 1M', 'sha256': archive_hash,
               'ratings': len(ratings), 'users': ratings.user.nunique(),
               'training_ratings': len(train.rating), 'training_catalog': len(train.movie_ids),
               'training_users': len(train.user_ids)})
    print(f'Training: {len(train.rating):,} ratings, {len(train.user_ids):,} users, {len(train.movie_ids):,} movies', flush=True)
    with threadpool_limits(limits=1):
        if args.stage == 'validate':
            validate(train, ratings, splits, signature, args.out, args.epochs)
        else:
            test(train, ratings, splits, signature, args.out)


if __name__ == '__main__':
    main()

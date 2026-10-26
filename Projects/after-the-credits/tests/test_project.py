import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import split_users, training_data, profiles_for
from src.models import fit_factors, FactorModel, genre_scores, percentiles, top_movies
from src.evaluation import metrics, paired_interval


class ProjectChecks(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame([(u, i, 1+(u+i)%5, i)
            for u in range(15) for i in range(40)], columns=['user','movie','rating','time'])
        self.movies = pd.DataFrame({'movie':range(40), 'title':[str(i) for i in range(40)],
            'genres':['Drama|Comedy' if i%2 else 'Action' for i in range(40)]}).set_index('movie')
        self.train = training_data(self.ratings, self.movies, np.arange(8))

    def test_splits_are_disjoint_and_repeatable(self):
        first, second = split_users(self.ratings), split_users(self.ratings)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(set(first['train']) & set(first['validation'])), 0)
        self.assertEqual(len(set(first['train']) & set(first['test'])), 0)
        self.assertEqual(len(set(first['test']) & set(first['validation'])), 0)
        self.assertEqual(sum(map(len, first.values())), 15)

    def test_heldout_users_cannot_change_training(self):
        changed = self.ratings.copy()
        changed.loc[changed.user >= 8, 'rating'] = 1
        other = training_data(changed, self.movies, np.arange(8))
        np.testing.assert_array_equal(other.rating, self.train.rating)
        np.testing.assert_array_equal(other.popularity, self.train.popularity)

    def test_profile_movies_and_targets_do_not_overlap(self):
        profiles, counts = profiles_for(self.ratings, np.arange(8,15), self.train)
        self.assertEqual(counts['evaluated_users'], 7)
        for p in profiles:
            self.assertEqual(len(p.pool), 20)
            self.assertFalse(p.candidates[p.pool].any())
            self.assertTrue(p.candidates[p.positives].all())
            self.assertFalse(np.isin(p.pool,p.positives).any())
            self.assertEqual(p.candidates.sum(),20)

    def test_ranking_metrics_by_hand(self):
        ndcg, recall = metrics(np.array([3,2,1]), np.array([3,1]), k=3)
        self.assertAlmostEqual(ndcg, (1 + .5)/(1 + 1/np.log2(3)))
        self.assertEqual(recall,1.)
        self.assertEqual(metrics(np.array([3,1,2]), np.array([3,1]),k=3),(1.,1.))
        with self.assertRaises(ValueError): metrics(np.array([1]),np.array([]))

    def test_factor_objective_and_roundtrip(self):
        model = fit_factors(self.train,factors=3,strength=1.,epochs=5)
        self.assertTrue(np.all(np.diff(model.history) <= 1e-6))
        before = model.scores(np.arange(5),np.array([1,2,3,4,5]))
        self.assertTrue(np.isfinite(before).all())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.npz'
            model.save(path)
            after=FactorModel.load(path).scores(np.arange(5),np.array([1,2,3,4,5]))
        np.testing.assert_allclose(before,after)

    def test_zero_genre_profile_falls_back(self):
        np.testing.assert_array_equal(genre_scores(self.train,np.arange(5),np.full(5,3.)),self.train.popularity)

    def test_candidates_and_ties(self):
        scores=np.array([9.,3.,3.,2.]); eligible=np.array([False,True,True,True])
        ids=np.array([1,40,20,10])
        np.testing.assert_array_equal(top_movies(scores,eligible,ids,k=2),[2,1])
        ranks=percentiles(scores,eligible)
        self.assertEqual(ranks[1],ranks[2])
        self.assertEqual(ranks[0],0.)

    def test_paired_bootstrap_uses_matching_users(self):
        a=pd.DataFrame({'user':[1,2,3],'ndcg10':[.2,.3,.4]})
        b=pd.DataFrame({'user':[3,1,2],'ndcg10':[.3,.1,.2]})
        result=paired_interval(a,b)
        for field in ['mean_difference','ci95_low','ci95_high']:
            self.assertAlmostEqual(result[field],.1)


if __name__ == '__main__': unittest.main()

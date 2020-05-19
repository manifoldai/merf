"""
MERF Unit Tests

Run with this command for verbose output:
> python tests.py -v

:copyright: 2017 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import pickle
import unittest

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from merf import MERF
from utils import MERFDataGenerator
from viz import plot_merf_training_stats


class DataGenerationTest(unittest.TestCase):
    def test_create_cluster_sizes(self):
        clusters = MERFDataGenerator.create_cluster_sizes_array([1, 2, 3], 1)
        self.assertListEqual(clusters, [1, 2, 3])

        clusters = MERFDataGenerator.create_cluster_sizes_array([30, 20, 7], 3)
        self.assertListEqual(clusters, [30, 30, 30, 20, 20, 20, 7, 7, 7])

    def test_generate_samples(self):
        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        df, ptev, prev = dg.generate_samples([1, 2, 3])
        # check columns
        self.assertListEqual(df.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        # check length
        self.assertEqual(len(df), 6)
        # check cluster sizes
        self.assertEqual(len(df[df["cluster"] == 0]), 1)
        self.assertEqual(len(df[df["cluster"] == 1]), 2)
        self.assertEqual(len(df[df["cluster"] == 2]), 3)

    def test_generate_split_samples(self):
        dg = MERFDataGenerator(m=0.7, sigma_b=2.7, sigma_e=1)
        train, test_known, test_new, training_ids, ptev, prev = dg.generate_split_samples([1, 3], [3, 2], [1, 1])
        # check all have same columns
        self.assertListEqual(train.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        self.assertListEqual(test_known.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])
        self.assertListEqual(test_new.columns.tolist(), ["y", "X_0", "X_1", "X_2", "Z", "cluster"])

        # check length
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test_known), 5)
        self.assertEqual(len(test_new), 2)

        # check cluster sizes
        self.assertEqual(len(train[train["cluster"] == 0]), 1)
        self.assertEqual(len(train[train["cluster"] == 1]), 3)
        self.assertEqual(len(test_known[test_known["cluster"] == 0]), 3)
        self.assertEqual(len(test_known[test_known["cluster"] == 1]), 2)
        self.assertEqual(len(test_new[test_new["cluster"] == 2]), 1)
        self.assertEqual(len(test_new[test_new["cluster"] == 3]), 1)

        # Check training ids
        self.assertListEqual(training_ids.tolist(), [0, 1])

    def test_ohe_clusters(self):
        training_cluster_ids = np.array([0, 1, 2, 3])
        # Training like encoding -- all categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(
            pd.Series([0, 0, 1, 2, 2, 2, 3]), training_cluster_ids=training_cluster_ids
        )
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [2, 1, 3, 1])

        # New encoding -- no categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(pd.Series([4, 4, 5, 6, 6, 7]), training_cluster_ids=training_cluster_ids)
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [0, 0, 0, 0])

        # Mixed encoding -- some categories in matrix
        X_ohe = MERFDataGenerator.ohe_clusters(
            pd.Series([1, 1, 3, 0, 0, 4, 5, 6, 6, 7]), training_cluster_ids=training_cluster_ids
        )
        # check columns and sums
        self.assertListEqual(X_ohe.columns.tolist(), ["cluster_0", "cluster_1", "cluster_2", "cluster_3"])
        self.assertListEqual(X_ohe.sum().tolist(), [2, 2, 0, 1])


class MERFTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(3187)

        dg = MERFDataGenerator(m=0.6, sigma_b=4.5, sigma_e=1)
        train, test_known, test_new, train_cluster_ids, ptev, prev = dg.generate_split_samples([1, 3], [3, 2], [1, 1])

        self.X_train = train[["X_0", "X_1", "X_2"]]
        self.Z_train = train[["Z"]]
        self.clusters_train = train["cluster"]
        self.y_train = train["y"]

        self.X_known = test_known[["X_0", "X_1", "X_2"]]
        self.Z_known = test_known[["Z"]]
        self.clusters_known = test_known["cluster"]
        self.y_known = test_known["y"]

        self.X_new = test_new[["X_0", "X_1", "X_2"]]
        self.Z_new = test_new[["Z"]]
        self.clusters_new = test_new["cluster"]
        self.y_new = test_new["y"]

    def test_not_fitted_error(self):
        m = MERF()
        with self.assertRaises(NotFittedError):
            m.predict(self.X_known, self.Z_known, self.clusters_known)

    def test_fit_and_predict_pandas(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(self.X_train, self.Z_train, self.clusters_train, self.y_train)
        self.assertEqual(len(m.gll_history), 5)
        self.assertEqual(len(m.val_loss_history), 0)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known, self.Z_known, self.clusters_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new, self.Z_new, self.clusters_new)
        self.assertEqual(len(yhat_new), 2)

    def test_fit_and_predict_numpy(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(np.array(self.X_train), np.array(self.Z_train), self.clusters_train, self.y_train)
        self.assertEqual(len(m.val_loss_history), 0)
        # Predict Known Clusters
        yhat_known = m.predict(np.array(self.X_known), np.array(self.Z_known), self.clusters_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(np.array(self.X_new), np.array(self.Z_new), self.clusters_new)
        self.assertEqual(len(yhat_new), 2)

    def test_type_error(self):
        m = MERF(max_iterations=5)
        self.assertRaises(
            TypeError,
            m.fit,
            np.array(self.X_train),
            np.array(self.Z_train),
            np.array(self.clusters_train),
            self.y_train,
        )

    def test_early_stopping(self):
        np.random.seed(3187)
        # Create a MERF model with a high early stopping threshold
        m = MERF(max_iterations=5, gll_early_stop_threshold=0.1)
        # Fit
        m.fit(self.X_train, self.Z_train, self.clusters_train, self.y_train)
        # The number of iterations should be less than max_iterations
        self.assertTrue(len(m.gll_history) < 5)

    def test_pickle(self):
        m = MERF(max_iterations=5)
        # Train
        m.fit(self.X_train, self.Z_train, self.clusters_train, self.y_train)

        # Write to pickle file
        with open("model.pkl", "wb") as fin:
            pickle.dump(m, fin)

        # Read back from pickle file
        with open("model.pkl", "rb") as fout:
            m_pkl = pickle.load(fout)

        # Check that m is not the same object as m_pkl
        self.assertIsNot(m_pkl, m)
        # Predict Known Clusters
        yhat_known_pkl = m_pkl.predict(self.X_known, self.Z_known, self.clusters_known)
        yhat_known = m.predict(self.X_known, self.Z_known, self.clusters_known)
        assert_almost_equal(yhat_known_pkl, yhat_known)
        # Predict New Clusters
        yhat_new_pkl = m_pkl.predict(self.X_new, self.Z_new, self.clusters_new)
        yhat_new = m.predict(self.X_new, self.Z_new, self.clusters_new)
        assert_almost_equal(yhat_new_pkl, yhat_new)

    def test_user_defined_fe_model(self):
        lgbm = LGBMRegressor()
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(self.X_train, self.Z_train, self.clusters_train, self.y_train)
        self.assertEqual(len(m.gll_history), 5)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known, self.Z_known, self.clusters_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new, self.Z_new, self.clusters_new)
        self.assertEqual(len(yhat_new), 2)

    def test_validation(self):
        lgbm = LGBMRegressor()
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(
            self.X_train,
            self.Z_train,
            self.clusters_train,
            self.y_train,
            self.X_known,
            self.Z_known,
            self.clusters_known,
            self.y_known,
        )
        self.assertEqual(len(m.val_loss_history), 5)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known, self.Z_known, self.clusters_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new, self.Z_new, self.clusters_new)
        self.assertEqual(len(yhat_new), 2)

    def test_validation_numpy(self):
        m = MERF(max_iterations=3)
        # Train
        m.fit(
            np.array(self.X_train),
            np.array(self.Z_train),
            self.clusters_train,
            self.y_train,
            np.array(self.X_new),
            np.array(self.Z_new),
            self.clusters_new,
            self.y_new,
        )
        self.assertEqual(len(m.val_loss_history), 3)
        # Predict Known Clusters
        yhat_known = m.predict(self.X_known, self.Z_known, self.clusters_known)
        self.assertEqual(len(yhat_known), 5)
        # Predict New Clusters
        yhat_new = m.predict(self.X_new, self.Z_new, self.clusters_new)
        self.assertEqual(len(yhat_new), 2)

    def test_viz(self):
        lgbm = LGBMRegressor()
        m = MERF(fixed_effects_model=lgbm, max_iterations=5)
        # Train
        m.fit(
            self.X_train,
            self.Z_train,
            self.clusters_train,
            self.y_train,
            self.X_known,
            self.Z_known,
            self.clusters_known,
            self.y_known,
        )
        plot_merf_training_stats(m)


if __name__ == "__main__":
    unittest.main()

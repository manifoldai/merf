"""
Data Generating Process

:copyright: 2017 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MERFDataGenerator(object):

    def __init__(self, m, sigma_b, sigma_e):
        self.m = m
        self.sigma_b = sigma_b
        self.sigma_e = sigma_e
        self.b = None
        self.ptev = None
        self.prev = None

    def generate_split_samples(self, n_training_per_cluster, n_test_known_per_cluster, n_test_new_per_cluster):
        """
        Generate samples split into training and two test sets.
        :return:
        """
        assert(len(n_training_per_cluster) == len(n_test_known_per_cluster))

        # Create global vector to pass to generate_samples function. Add the two known cluster numbers to get the
        # total to get for known clusters. Then append the new cluster numbers.
        n_known_per_cluster = np.array(n_training_per_cluster) + np.array(n_test_known_per_cluster)
        n_samples_per_cluster = np.concatenate((n_known_per_cluster, np.array(n_test_new_per_cluster)))

        # Track known and new cluster ids -- will be used later for splitting.
        num_known_clusters = len(n_known_per_cluster)
        known_cluster_ids = range(0, num_known_clusters)

        # Generate all the data at once.
        merged_df = self.generate_samples(n_samples_per_cluster)

        # Select out new cluster test data (this is easy)
        new_cluster_test_data = merged_df[merged_df['cluster'] >= num_known_clusters]

        # Select out known cluster data, but separate this into training set and test set
        train_dfs = []
        test_dfs = []
        for cluster_id, num_train, num_test in zip(known_cluster_ids, n_training_per_cluster, n_test_known_per_cluster):
            cluster_df = merged_df[merged_df['cluster'] == cluster_id]
            train_cluster_df = cluster_df.iloc[0:num_train]
            test_cluster_df = cluster_df.iloc[num_train:(num_train + num_test)]
            train_dfs.append(train_cluster_df)
            test_dfs.append(test_cluster_df)

        # Turn the list of dataframes into one big dataframe
        training_data = pd.concat(train_dfs)
        known_cluster_test_data = pd.concat(test_dfs)

        return training_data, known_cluster_test_data, new_cluster_test_data

    def generate_samples(self, n_samples_per_cluster):
        """
        Generate test data for the MERF algorithm.
        :param n_samples_per_cluster: array of number representing number of samples to choose from that cluster
        :param m: scale parameter on fixed effect
        :param sigma_b: hyper parameter of random intercept
        :param sigma_e: noise std
        :return: y (response), X (fixed effect features), Z (cluster assignment),
        ptev (proportion of total effect variability), prev (proportion of random effect variability)
        """

        # ~~~~~~~~~ Fixed Effect Generation ~~~~~~~~~~ #
        # draw the fixed effects features for each samples (3 dims independent)
        X = np.random.normal(loc=0, scale=1.0, size=(3, sum(n_samples_per_cluster)))
        X_df = pd.DataFrame(X).T

        # generate the fixed effect response
        g = 2 * X[0] + X[1] ** 2 + 4 * (X[2] > 0) + 2 * np.log(abs(X[0])) * X[2]
        sigma_g = np.std(g)

        # ~~~~~~~~~ Random Effect Generation ~~~~~~~~~~ #
        # Create the number of clusters from vector of samples per cluster
        n_clusters = len(n_samples_per_cluster)

        # Create vector of cluster_id for each sample
        Z = []
        for i in range(0, n_clusters):
            cluster_id = i
            n_samples = n_samples_per_cluster[i]
            zi = cluster_id * np.ones(n_samples, dtype=np.int8)  # want cluster id to be int
            Z.extend(zi)

        # one hot encode it for easier addition to get response
        Z_ohe = pd.get_dummies(Z)

        # create groups partition and random intercept value
        clusters_df = pd.Series(Z)
        Z_df = pd.DataFrame(np.ones(len(Z)))

        # draw the random effect bias for each cluster
        b = np.random.normal(loc=0, scale=self.sigma_b, size=n_clusters)

        # generate the random effect response
        re = Z_ohe.dot(b)

        # ~~~~~~~~~ Noise Generation ~~~~~~~~~~ #
        eps = np.random.normal(loc=0, scale=self.sigma_e, size=sum(n_samples_per_cluster))

        # ~~~~~~~~~ Response Generation ~~~~~~~~~~ #
        # add fixed effect, random effect, and noise to get final response
        y = self.m * g + re + eps
        y_df = pd.Series(y)

        # ~~~~~~~~~ Metrics Generation ~~~~~~~~~~ #
        # compute the ptev and prev
        sigma_fixed = self.m * sigma_g
        ptev = 100 * ((sigma_fixed ** 2 + self.sigma_b ** 2) /
                      (sigma_fixed ** 2 + self.sigma_b ** 2 + self.sigma_e ** 2))
        prev = 100 * (self.sigma_b ** 2 / (sigma_fixed ** 2 + self.sigma_b ** 2))

        logger.info("Drew {} samples from {} clusters.".format(sum(n_samples_per_cluster), n_clusters))
        logger.info("PTEV = {}, PREV = {}.".format(ptev, prev))

        self.ptev = ptev
        self.prev = prev
        self.b = b

        # merge all the separate matrices into one matrix
        merged_df = pd.concat((y_df, X_df, Z_df, clusters_df), axis=1)
        merged_df.columns = ['y', 'X_0', 'X_1', 'X_2', 'Z', 'cluster']
        return merged_df

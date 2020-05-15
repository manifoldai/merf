"""
Synthetic mixed-effects data generator.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MERFDataGenerator(object):
    """
    Synthetic data generator class. It simulates samples y from K clusters according to the following equation.

    .. math::

        y_{ij} = m \cdot g(x_{ij}) + b_i + \epsilon_{ij}

        g(x_{ij}) = 2 x_{1ij} + x_{2ij}^2 + 4(x_{3ij} > 0) + 2 \log |x_{1ij}|x_{3ij}

        b_i \sim N(0, \sigma_b^2)

        \epsilon_{ij} \sim N(0, \sigma_\epsilon^2)

        i = 1, ..., K

        j = 1, ..., n_i

    Args:
        m (float): scale parameter on fixed effect
        sigma_b (float): hyper parameter of random intercept
        sigma_e (float): noise std
    """

    def __init__(self, m, sigma_b, sigma_e):
        self.m = m
        self.sigma_b = sigma_b
        self.sigma_e = sigma_e
        self.b = None
        self.ptev = None
        self.prev = None

    @staticmethod
    def ohe_clusters(clusters, training_cluster_ids):
        """
        Helper function to one hot encode cluster ids based on training cluster ids. Note that for the "new" clusters
        this should encode to a matrix of all zeros.

        Args:
            clusters (np.ndarray): array of cluster labels for data
            training_cluster_ids: array of clusters in training data

        Returns:
            pd.DataFrame: one hot encoded clusters
        """
        clusters_prime = pd.Categorical(clusters, categories=training_cluster_ids)
        X_ohe = pd.get_dummies(clusters_prime, prefix="cluster")
        return X_ohe

    @staticmethod
    def create_X_with_ohe_clusters(X, clusters, training_cluster_ids):
        """
        Helper function to join one hot encoded cluster ids with the feature matrix X.

        Args:
            X (np.ndarray): fixed effects feature matrix
            clusters (np.ndarray): array of cluster labels for data in X
            training_cluster_ids: array of clusters in training data

        Returns:
            pd.DataFrame: X augmented with one hot encoded clusters
        """
        X_ohe = MERFDataGenerator.ohe_clusters(clusters, training_cluster_ids)
        X_w_ohe = pd.merge(X, X_ohe, left_index=True, right_index=True)
        return X_w_ohe

    @staticmethod
    def create_cluster_sizes_array(sizes, num_clusters_per_size):
        """
        Helper function to create an array of cluster sizes.

        Args:
            sizes (np.ndarray): array of sizes
            num_clusters_per_size (np.ndarray): array of the number of clusters to make of each size

        Returns:
            np.ndarray: array of cluster sizes for all clusters
        """
        cluster_sizes = []
        for size in sizes:
            cluster_sizes.extend(size * np.ones(num_clusters_per_size, dtype=np.int8))
        return cluster_sizes

    def generate_split_samples(self, n_training_per_cluster, n_test_known_per_cluster, n_test_new_per_cluster):
        """
        Generate samples split into training and two test sets.

        Args:
            n_training_per_cluster:
            n_test_known_per_cluster:
            n_test_new_per_cluster:

        Returns:
            tuple:
                * training_data
                * known_cluster_test_data
                * new_cluster_test_data
                * training_cluster_ids
                * ptev
                * prev
        """
        assert len(n_training_per_cluster) == len(n_test_known_per_cluster)

        # Create global vector to pass to generate_samples function. Add the two known cluster numbers to get the
        # total to get for known clusters. Then append the new cluster numbers.
        n_known_per_cluster = np.array(n_training_per_cluster) + np.array(n_test_known_per_cluster)
        n_samples_per_cluster = np.concatenate((n_known_per_cluster, np.array(n_test_new_per_cluster)))

        # Track known and new cluster ids -- will be used later for splitting.
        num_known_clusters = len(n_known_per_cluster)
        known_cluster_ids = range(0, num_known_clusters)

        # Generate all the data at once.
        merged_df, ptev, prev = self.generate_samples(n_samples_per_cluster)

        # Select out new cluster test data (this is easy)
        new_cluster_test_data = merged_df[merged_df["cluster"] >= num_known_clusters]

        # Select out known cluster data, but separate this into training set and test set
        train_dfs = []
        test_dfs = []
        for cluster_id, num_train, num_test in zip(known_cluster_ids, n_training_per_cluster, n_test_known_per_cluster):
            cluster_df = merged_df[merged_df["cluster"] == cluster_id]
            train_cluster_df = cluster_df.iloc[0:num_train]
            test_cluster_df = cluster_df.iloc[num_train : (num_train + num_test)]
            train_dfs.append(train_cluster_df)
            test_dfs.append(test_cluster_df)

        # Turn the list of dataframes into one big dataframe
        training_data = pd.concat(train_dfs)
        known_cluster_test_data = pd.concat(test_dfs)

        # Store off the unique labels in the training data
        training_cluster_ids = np.sort(training_data["cluster"].unique())

        return (training_data, known_cluster_test_data, new_cluster_test_data, training_cluster_ids, ptev, prev)

    def generate_samples(self, n_samples_per_cluster):
        """
        Generate test data for the MERF algorithm.

        Args:
            n_samples_per_cluster: array of number representing number of samples to choose from that cluster

        Returns:
            tuple:
                * y (response)
                * X (fixed effect features)
                * Z (cluster assignment)
                * ptev (proportion of total effect variability)
                * prev (proportion of random effect variability)
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
        ptev = 100 * (
            (sigma_fixed ** 2 + self.sigma_b ** 2) / (sigma_fixed ** 2 + self.sigma_b ** 2 + self.sigma_e ** 2)
        )
        prev = 100 * (self.sigma_b ** 2 / (sigma_fixed ** 2 + self.sigma_b ** 2))

        logger.info("Drew {} samples from {} clusters.".format(sum(n_samples_per_cluster), n_clusters))
        logger.info("PTEV = {}, PREV = {}.".format(ptev, prev))

        self.ptev = ptev
        self.prev = prev
        self.b = b

        # merge all the separate matrices into one matrix
        merged_df = pd.concat((y_df, X_df, Z_df, clusters_df), axis=1)
        merged_df.columns = ["y", "X_0", "X_1", "X_2", "Z", "cluster"]
        return merged_df, ptev, prev

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(n_samples_per_cluster, m, sigma_b, sigma_e):
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
    b = np.random.normal(loc=0, scale=sigma_b, size=n_clusters)

    # generate the random effect response
    re = Z_ohe.dot(b)

    # ~~~~~~~~~ Noise Generation ~~~~~~~~~~ #
    eps = np.random.normal(loc=0, scale=sigma_e, size=sum(n_samples_per_cluster))

    # ~~~~~~~~~ Response Generation ~~~~~~~~~~ #
    # add fixed effect, random effect, and noise to get final response
    y = m * g + re + eps
    y_df = pd.Series(y)

    # ~~~~~~~~~ Metrics Generation ~~~~~~~~~~ #
    # compute the ptev and prev
    sigma_fixed = m * sigma_g
    ptev = 100 * ((sigma_fixed ** 2 + sigma_b ** 2) / (sigma_fixed ** 2 + sigma_b ** 2 + sigma_e ** 2))
    prev = 100 * (sigma_b ** 2 / (sigma_fixed ** 2 + sigma_b ** 2))

    logger.info("Drew {} samples from {} clusters.".format(sum(n_samples_per_cluster), n_clusters))
    logger.info("PTEV = {}, PREV = {}.".format(ptev, prev))

    # return relevant vectors and metrics
    return y_df, X_df, Z_df, clusters_df, b, ptev, prev

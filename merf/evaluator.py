"""
Mixed Effects Random Forest Evaluator

:copyright: 2018 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_stats(model):
    """
    Plot training statistics for MERF model
    :param model:
    :return:
    """
    plt.figure(figsize=[15, 10])

    # Plot GLL
    plt.subplot(221)
    plt.plot(model.gll_history)
    plt.grid("on")
    plt.ylabel("GLL")
    plt.title("GLL")

    # Plot trace and determinant of Sigma_b (covariance matrix)
    plt.subplot(222)
    det_sigmaB_history = [np.linalg.det(x) for x in model.D_hat_history]
    trace_sigmaB_history = [np.trace(x) for x in model.D_hat_history]
    plt.plot(det_sigmaB_history, label="det(Sigma_b)")
    plt.plot(trace_sigmaB_history, label="trace(Sigma_b)")
    plt.grid("on")
    plt.legend()
    plt.title("Sigma_b_hat metrics")

    plt.subplot(223)
    plt.plot(model.sigma2_hat_history)
    plt.grid("on")
    plt.ylabel("sigma_e2_hat")
    plt.xlabel("Iteration")

    plt.subplot(224)
    plot_bhat(model, 1)


def plot_bhat(model, cluster_id):
    """
    Plot the bhat for a particular cluster
    :param model:
    :param cluster_id:
    :return:
    """
    # This code does a complicated reshape and re-indexing operation to get the
    # list of dataframes into a multi-indexed dataframe.
    # Step 1 - vertical stack all the arrays at each iteration into a single numpy array
    b_array = np.vstack(model.b_hat_history)

    # Step 2 - Create the multi-index. Note the outer index is iteration. The inner index is cluster.
    iterations = range(len(model.b_hat_history))
    clusters = model.b_hat_history[0].index
    mi = pd.MultiIndex.from_product([iterations, clusters], names=("iteration", "cluster"))

    # Step 3 - Create the multi-indexed dataframe
    b_df = pd.DataFrame(b_array, index=mi)

    # Step 4 - Use the fancy xs function to access all iterations of a single cluster
    b_df.xs(cluster_id, level="cluster").plot()
    plt.grid("on")
    plt.ylabel("b_hat")
    plt.xlabel("Iteration")
    plt.title("Cluster = {}".format(cluster_id))
    return b_df

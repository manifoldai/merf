"""
Mixed Effects Random Forest Evaluator

:copyright: 2018 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
    b_df = np.dstack(model.b_hat_history)
    b_panel = pd.Panel(b_df, items=model.b_hat_history[0].index)
    plt.plot(b_panel.iloc[cluster_id].T)
    plt.grid("on")
    plt.ylabel("b_hat")
    plt.xlabel("Iteration")
    plt.legend()
    plt.title("Cluster = {}".format(model.cluster_counts.index[cluster_id]))
    return b_panel

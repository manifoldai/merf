"""
Mixed Effects Random Forest Evaluator

:copyright: 2018 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_training_stats(model, num_clusters_to_plot=5):
    """
    Plot training statistics for MERF model
    :param model:
    :return:
    """
    fig = plt.figure(tight_layout=True, figsize=[15, 15])
    gs = gridspec.GridSpec(3, 2)

    # Plot GLL
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(model.gll_history)
    ax.grid("on")
    ax.set_ylabel("GLL")
    ax.set_title("GLL")

    # Plot trace and determinant of Sigma_b (covariance matrix)
    ax = fig.add_subplot(gs[0, 1])
    det_sigmaB_history = [np.linalg.det(x) for x in model.D_hat_history]
    trace_sigmaB_history = [np.trace(x) for x in model.D_hat_history]
    ax.plot(det_sigmaB_history, label="det(Sigma_b)")
    ax.plot(trace_sigmaB_history, label="trace(Sigma_b)")
    ax.grid("on")
    ax.legend()
    ax.set_title("Sigma_b_hat metrics")

    # Plot sigma_e across iterations
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(model.sigma2_hat_history)
    ax.grid("on")
    ax.set_ylabel("sigma_e2_hat")
    ax.set_xlabel("Iteration")

    # Plot bi across iterations
    ax = fig.add_subplot(gs[1, 1])
    for cluster_id in model.cluster_counts.index[0:num_clusters_to_plot]:
        ax.plot(model.b_hat_history_df.xs(cluster_id, level="cluster"), label=cluster_id)
    ax.grid("on")
    ax.set_ylabel("b_hat")
    ax.set_xlabel("Iteration")

    # Plot bi distributions
    ax = fig.add_subplot(gs[2, :])
    model.trained_b.hist(bins=15, ax=ax)
    ax.set_xlabel("b_i")
    ax.set_title("Distribution of b_is")

"""
Mixed Effects Random Forest

:copyright: 2017 Manifold, Inc.
:author: Sourav Dey <sdey@manifold.ai>
"""
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(
    format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class MERF(object):

    def __init__(self, n_estimators=300, min_iterations=100, gll_early_stop_threshold=1e-4, max_iterations=200):
        self.n_estimators = n_estimators
        self.min_iterations = min_iterations
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations

        self.trained_rf = None
        self.trained_b = None

        self.b_hat_history = []
        self.sigma2_hat_history = []
        self.D_hat_history = []
        self.gll_history = []

    def predict(self, X, Z, clusters):
        pass

    def fit(self, X, Z, clusters, y):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(len(Z) == len(X))
        assert(len(y) == len(X))
        assert(len(clusters) == len(X))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_clusters = clusters.nunique()
        n_obs = len(y)
        q = Z.shape[1]  # random effects dimension
        Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)
        # p = X.shape[1]  # fixed effects dimension

        # Create a series where cluster_id is the index and n_i is the value
        cluster_counts = clusters.value_counts()

        # Do expensive slicing operations only once
        Z_by_cluster = {}
        y_by_cluster = {}
        n_by_cluster = {}
        I_by_cluster = {}
        indices_by_cluster = {}

        # TODO: Can these be replaced with groupbys? Groupbys are less understandable than brute force.
        for cluster_id in cluster_counts.index:
            # Find the index for all the samples from this cluster in the large vector
            indices_i = (clusters == cluster_id)
            indices_by_cluster[cluster_id] = indices_i

            # Slice those samples from Z and y
            Z_by_cluster[cluster_id] = Z[indices_i]
            y_by_cluster[cluster_id] = y[indices_i]

            # Get the counts for each cluster and create the appropriately sized identity matrix for later computations
            n_by_cluster[cluster_id] = cluster_counts[cluster_id]
            I_by_cluster[cluster_id] = np.eye(cluster_counts[cluster_id])

        # Intialize for EM algorithm
        iteration = 0
        b_hat = np.zeros((n_clusters, q))  # dimension is n_clusters X q
        sigma2_hat = 1
        D_hat = np.eye(q)

        # vectors to hold history
        self.b_hat_history.append(b_hat)
        self.sigma2_hat_history.append(sigma2_hat)
        self.D_hat_history.append(D_hat)

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            logger.debug("Iteration: {}".format(iteration))
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ E-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fill up y_star for all clusters
            y_star = np.zeros(len(y))
            for cluster_id in cluster_counts.index:
                # Get cached cluster slices
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                b_hat_i = b_hat[cluster_id]
                indices_i = indices_by_cluster[cluster_id]

                # Compute y_star for this cluster and put back in right place
                y_star_i = y_i - Z_i.dot(b_hat_i)
                y_star[indices_i] = y_star_i

            # check that still one dimensional
            # TODO: Other checks we want to do?
            assert(len(y_star.shape) == 1)

            # Do the random forest regression with all the fixed effects features
            rf = RandomForestRegressor(n_estimators=self.n_estimators, oob_score=True, n_jobs=-1)
            rf.fit(X, y_star)
            f_hat = rf.oob_prediction_

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sigma2_hat_sum = 0
            D_hat_sum = 0

            for cluster_id in cluster_counts.index:
                # Get cached cluster slices
                indices_i = indices_by_cluster[cluster_id]
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                n_i = n_by_cluster[cluster_id]
                I_i = I_by_cluster[cluster_id]

                # index into f_hat
                f_hat_i = f_hat[indices_i]

                # Compute V_hat_i
                V_hat_i = Z_i.dot(D_hat).dot(Z_i.T) + sigma2_hat * I_i

                # Compute b_hat_i
                V_hat_inv_i = np.linalg.pinv(V_hat_i)
                b_hat_i = D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(y_i - f_hat_i)

                # Compute the total error for this cluster
                eps_hat_i = y_i - f_hat_i - Z_i.dot(b_hat_i)

                logger.debug("------------------------------------------")
                logger.debug("M-step, cluster {}".format(cluster_id))
                logger.debug("error squared for cluster = {}".format(eps_hat_i.T.dot(eps_hat_i)))

                # Store b_hat for cluster
                b_hat[cluster_id] = b_hat_i

                # Update the sums for sigma2_hat and D_hat. We will update after the entire loop over clusters
                sigma2_hat_sum += eps_hat_i.T.dot(eps_hat_i) + sigma2_hat * (n_i - sigma2_hat * np.trace(V_hat_inv_i))
                # sigma2_hat_sum += eps_hat_i.T.dot(eps_hat_i) + sigma2_hat * (n_i - sigma2_hat * np.trace(V_hat_i))
                D_hat_sum += np.outer(b_hat_i, b_hat_i) + \
                             (D_hat - D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(Z_i).dot(D_hat))  # noqa: E127

            # Normalize the sums to get sigma2_hat and D_hat
            sigma2_hat = (1. / n_obs) * sigma2_hat_sum
            D_hat = (1. / n_clusters) * D_hat_sum

            logger.debug("b_hat = {}".format(b_hat))
            logger.debug("sigma2_hat = {}".format(sigma2_hat))
            logger.debug("D_hat = {}".format(D_hat))

            # Store off history so that we can see the evolution of the EM algorithm
            self.b_hat_history.append(b_hat)
            self.sigma2_hat_history.append(sigma2_hat)
            self.D_hat_history.append(D_hat)

            # Generalized Log Likelihood computation to check convergence
            gll = 0
            for cluster_id in cluster_counts.index:
                # Get cached cluster slices
                indices_i = indices_by_cluster[cluster_id]
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                I_i = I_by_cluster[cluster_id]

                # Slice f_hat and get b_hat
                f_hat_i = f_hat[indices_i]
                R_hat_i = sigma2_hat * I_i
                b_hat_i = b_hat[cluster_id]

                gll += (y_i - f_hat_i - Z_i.dot(b_hat_i)).T.\
                           dot(np.linalg.pinv(R_hat_i)).\
                           dot(y_i - f_hat_i - Z_i.dot(b_hat_i)) +\
                       b_hat_i.T.dot(np.linalg.pinv(D_hat)).dot(b_hat_i) + np.log(np.linalg.det(D_hat)) +\
                       np.log(np.linalg.det(R_hat_i))  # noqa: E127

            self.gll_history.append(gll)

        # Store off most recent random forest model and b_hat as the model to be used in the prediction stage
        self.trained_rf = rf
        self.trained_b = b_hat

    def score(self, X):
        pass

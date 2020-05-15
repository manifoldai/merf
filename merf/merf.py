"""
Mixed Effects Random Forest model.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class MERF(object):
    """
    This is the core class to instantiate, train, and predict using a mixed effects random forest model.
    It roughly adheres to the sklearn estimator API.
    Note that the user must pass in an already instantiated fixed_effects_model that adheres to the
    sklearn regression estimator API, i.e. must have a fit() and predict() method defined.

    It assumes a data model of the form:

    .. math::

        y = f(X) + b_i Z + e

    * y is the target variable. The current code only supports regression for now, e.g. continuously varying scalar value
    * X is the fixed effect features. Assume p dimensional
    * f(.) is the nonlinear fixed effects mode, e.g. random forest
    * Z is the random effect features. Assume q dimensional.
    * e is iid noise ~N(0, sigma_e²)
    * i is the cluster index. Assume k clusters in the training.
    * bi is the random effect coefficients. They are different per cluster i but are assumed to be drawn from the same distribution ~N(0, Sigma_b) where Sigma_b is learned from the data.


    Args:
        fixed_effects_model (sklearn.base.RegressorMixin): instantiated model class
        gll_early_stop_threshold (float): early stopping threshold on GLL improvement
        max_iterations (int): maximum number of EM iterations to train
    """

    def __init__(
        self,
        fixed_effects_model=RandomForestRegressor(n_estimators=300, n_jobs=-1),
        gll_early_stop_threshold=None,
        max_iterations=20,
    ):
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations

        self.cluster_counts = None
        # Note fixed_effects_model must already be instantiated when passed in.
        self.fe_model = fixed_effects_model
        self.trained_fe_model = None
        self.trained_b = None

        self.b_hat_history = []
        self.sigma2_hat_history = []
        self.D_hat_history = []
        self.gll_history = []
        self.val_loss_history = []

    def predict(self, X: np.ndarray, Z: np.ndarray, clusters: pd.Series):
        """
        Predict using trained MERF.  For known clusters the trained random effect correction is applied.
        For unknown clusters the pure fixed effect (RF) estimate is used.

        Args:
            X (np.ndarray): fixed effect covariates
            Z (np.ndarray): random effect covariates
            clusters (pd.Series): cluster assignments for samples

        Returns:
            np.ndarray: the predictions y_hat
        """
        if type(clusters) != pd.Series:
            raise TypeError("clusters must be a pandas Series.")

        if self.trained_fe_model is None:
            raise NotFittedError(
                "This MERF instance is not fitted yet. Call 'fit' with appropriate arguments before "
                "using this method"
            )

        Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

        # Apply fixed effects model to all
        y_hat = self.trained_fe_model.predict(X)

        # Apply random effects correction to all known clusters. Note that then, by default, the new clusters get no
        # random effects correction -- which is the desired behavior.
        for cluster_id in self.cluster_counts.index:
            indices_i = clusters == cluster_id

            # If cluster doesn't exist in test data that's ok. Just move on.
            if len(indices_i) == 0:
                continue

            # If cluster does exist, apply the correction.
            b_i = self.trained_b.loc[cluster_id]
            Z_i = Z[indices_i]
            y_hat[indices_i] += Z_i.dot(b_i)

        return y_hat

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        clusters: pd.Series,
        y: np.ndarray,
        X_val: np.ndarray = None,
        Z_val: np.ndarray = None,
        clusters_val: pd.Series = None,
        y_val: np.ndarray = None,
    ):
        """
        Fit MERF using Expectation-Maximization algorithm.

        Args:
            X (np.ndarray): fixed effect covariates
            Z (np.ndarray): random effect covariates
            clusters (pd.Series): cluster assignments for samples
            y (np.ndarray): response/target variable

        Returns:
            MERF: fitted model
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if type(clusters) != pd.Series:
            raise TypeError("clusters must be a pandas Series.")

        assert len(Z) == len(X)
        assert len(y) == len(X)
        assert len(clusters) == len(X)

        if X_val is None:
            assert Z_val is None
            assert clusters_val is None
            assert y_val is None
        else:
            assert len(Z_val) == len(X_val)
            assert len(clusters_val) == len(X_val)
            assert len(y_val) == len(X_val)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_clusters = clusters.nunique()
        n_obs = len(y)
        q = Z.shape[1]  # random effects dimension
        Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

        # Create a series where cluster_id is the index and n_i is the value
        self.cluster_counts = clusters.value_counts()

        # Do expensive slicing operations only once
        Z_by_cluster = {}
        y_by_cluster = {}
        n_by_cluster = {}
        I_by_cluster = {}
        indices_by_cluster = {}

        # TODO: Can these be replaced with groupbys? Groupbys are less understandable than brute force.
        for cluster_id in self.cluster_counts.index:
            # Find the index for all the samples from this cluster in the large vector
            indices_i = clusters == cluster_id
            indices_by_cluster[cluster_id] = indices_i

            # Slice those samples from Z and y
            Z_by_cluster[cluster_id] = Z[indices_i]
            y_by_cluster[cluster_id] = y[indices_i]

            # Get the counts for each cluster and create the appropriately sized identity matrix for later computations
            n_by_cluster[cluster_id] = self.cluster_counts[cluster_id]
            I_by_cluster[cluster_id] = np.eye(self.cluster_counts[cluster_id])

        # Intialize for EM algorithm
        iteration = 0
        # Note we are using a dataframe to hold the b_hat because this is easier to index into by cluster_id
        # Before we were using a simple numpy array -- but we were indexing into that wrong because the cluster_ids
        # are not necessarily in order.
        b_hat_df = pd.DataFrame(np.zeros((n_clusters, q)), index=self.cluster_counts.index)
        sigma2_hat = 1
        D_hat = np.eye(q)

        # vectors to hold history
        self.b_hat_history.append(b_hat_df)
        self.sigma2_hat_history.append(sigma2_hat)
        self.D_hat_history.append(D_hat)

        early_stop_flag = False

        while iteration < self.max_iterations and not early_stop_flag:
            iteration += 1
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            logger.debug("Iteration: {}".format(iteration))
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ E-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fill up y_star for all clusters
            y_star = np.zeros(len(y))
            for cluster_id in self.cluster_counts.index:
                # Get cached cluster slices
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                b_hat_i = b_hat_df.loc[cluster_id]  # used to be ix
                logger.debug("E-step, cluster {}, b_hat = {}".format(cluster_id, b_hat_i))
                indices_i = indices_by_cluster[cluster_id]

                # Compute y_star for this cluster and put back in right place
                y_star_i = y_i - Z_i.dot(b_hat_i)
                y_star[indices_i] = y_star_i

            # check that still one dimensional
            # TODO: Other checks we want to do?
            assert len(y_star.shape) == 1

            # Do the fixed effects regression with all the fixed effects features
            self.fe_model.fit(X, y_star)
            f_hat = self.fe_model.predict(X)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sigma2_hat_sum = 0
            D_hat_sum = 0

            for cluster_id in self.cluster_counts.index:
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
                logger.debug("M-step, pre-update, cluster {}, b_hat = {}".format(cluster_id, b_hat_df.loc[cluster_id]))
                b_hat_i = D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(y_i - f_hat_i)
                logger.debug("M-step, post-update, cluster {}, b_hat = {}".format(cluster_id, b_hat_i))

                # Compute the total error for this cluster
                eps_hat_i = y_i - f_hat_i - Z_i.dot(b_hat_i)

                logger.debug("------------------------------------------")
                logger.debug("M-step, cluster {}".format(cluster_id))
                logger.debug("error squared for cluster = {}".format(eps_hat_i.T.dot(eps_hat_i)))

                # Store b_hat for cluster both in numpy array and in dataframe
                # Note this HAS to be assigned with loc, otw whole df get erroneously assigned and things go to hell
                b_hat_df.loc[cluster_id, :] = b_hat_i
                logger.debug(
                    "M-step, post-update, recalled from db, cluster {}, "
                    "b_hat = {}".format(cluster_id, b_hat_df.loc[cluster_id])
                )

                # Update the sums for sigma2_hat and D_hat. We will update after the entire loop over clusters
                sigma2_hat_sum += eps_hat_i.T.dot(eps_hat_i) + sigma2_hat * (n_i - sigma2_hat * np.trace(V_hat_inv_i))
                D_hat_sum += np.outer(b_hat_i, b_hat_i) + (
                    D_hat - D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(Z_i).dot(D_hat)
                )  # noqa: E127

            # Normalize the sums to get sigma2_hat and D_hat
            sigma2_hat = (1.0 / n_obs) * sigma2_hat_sum
            D_hat = (1.0 / n_clusters) * D_hat_sum

            logger.debug("b_hat = {}".format(b_hat_df))
            logger.debug("sigma2_hat = {}".format(sigma2_hat))
            logger.debug("D_hat = {}".format(D_hat))

            # Store off history so that we can see the evolution of the EM algorithm
            self.b_hat_history.append(b_hat_df.copy())
            self.sigma2_hat_history.append(sigma2_hat)
            self.D_hat_history.append(D_hat)

            # Generalized Log Likelihood computation to check convergence
            gll = 0
            for cluster_id in self.cluster_counts.index:
                # Get cached cluster slices
                indices_i = indices_by_cluster[cluster_id]
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                I_i = I_by_cluster[cluster_id]

                # Slice f_hat and get b_hat
                f_hat_i = f_hat[indices_i]
                R_hat_i = sigma2_hat * I_i
                b_hat_i = b_hat_df.loc[cluster_id]

                # Numerically stable way of computing log(det(A))
                _, logdet_D_hat = np.linalg.slogdet(D_hat)
                _, logdet_R_hat_i = np.linalg.slogdet(R_hat_i)

                gll += (
                    (y_i - f_hat_i - Z_i.dot(b_hat_i))
                    .T.dot(np.linalg.pinv(R_hat_i))
                    .dot(y_i - f_hat_i - Z_i.dot(b_hat_i))
                    + b_hat_i.T.dot(np.linalg.pinv(D_hat)).dot(b_hat_i)
                    + logdet_D_hat
                    + logdet_R_hat_i
                )  # noqa: E127

            logger.info("Training GLL is {} at iteration {}.".format(gll, iteration))
            self.gll_history.append(gll)

            # Save off the most updated fixed effects model and random effects coefficents
            self.trained_fe_model = self.fe_model
            self.trained_b = b_hat_df

            # Early Stopping. This code is entered only if the early stop threshold is specified and
            # if the gll_history array is longer than 1 element, e.g. we are past the first iteration.
            if self.gll_early_stop_threshold is not None and len(self.gll_history) > 1:
                curr_threshold = np.abs((gll - self.gll_history[-2]) / self.gll_history[-2])
                logger.debug("stop threshold = {}".format(curr_threshold))

                if curr_threshold < self.gll_early_stop_threshold:
                    logger.info("Gll {} less than threshold {}, stopping early ...".format(gll, curr_threshold))
                    early_stop_flag = True

            # Compute Validation Loss
            if X_val is not None:
                yhat_val = self.predict(X_val, Z_val, clusters_val)
                val_loss = np.square(np.subtract(y_val, yhat_val)).mean()
                logger.info(f"Validation MSE Loss is {val_loss} at iteration {iteration}.")
                self.val_loss_history.append(val_loss)

        return self

    def score(self, X, Z, clusters, y):
        raise NotImplementedError()

    def get_bhat_history_df(self):
        """
        This function does a complicated reshape and re-indexing operation to get the
        list of dataframes for the b_hat_history into a multi-indexed dataframe.  This
        dataframe is easier to work with in plotting utilities and other downstream
        analyses than the list of dataframes b_hat_history.

        Args:
            b_hat_history (list): list of dataframes of bhat at every iteration

        Returns:
            pd.DataFrame: multi-index dataframe with outer index as iteration, inner index as cluster
        """
        # Step 1 - vertical stack all the arrays at each iteration into a single numpy array
        b_array = np.vstack(self.b_hat_history)

        # Step 2 - Create the multi-index. Note the outer index is iteration. The inner index is cluster.
        iterations = range(len(self.b_hat_history))
        clusters = self.b_hat_history[0].index
        mi = pd.MultiIndex.from_product([iterations, clusters], names=("iteration", "cluster"))

        # Step 3 - Create the multi-indexed dataframe
        b_hat_history_df = pd.DataFrame(b_array, index=mi)
        return b_hat_history_df

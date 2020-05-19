# Mixed Effects Random Forest

![](https://github.com/manifoldai/merf/workflows/CI/badge.svg)

This repository contains a pure Python implementation of a mixed effects random forest (MERF) algorithm. It can be used, out of the box, to fit a MERF model and predict with it.  

* [Sphinx documentation](https://manifoldai.github.io/merf/)
* [Blog post](https://towardsdatascience.com/mixed-effects-random-forests-6ecbb85cb177) 

## MERF Model

The MERF model is:

y_i = f(X_i) + Z_i * b_i + e_i

b_i ~ N(0, D)

e_i ~ N(0, R_i)

for each cluster i out of n total clusters.

In the above:

* y_i -- the (n_i x 1) vector of responses for cluster i. These are given at at training.
* X_i -- the (n_i x p) fixed effects covariates that are associated with the y_i. These are given at training.
* Z_i -- the (n_i x q) random effects covariates that are associated with the y_i. These are given at training.
* e_i -- the (n_i x 1) vector of errors for cluster i. This is unknown.
* i is the cluster_id. This is given at training.

The learned parameters in MERF are:
* f() -- which is a random forest that models the, potentially nonlinear, mapping from the fixed effect covariates to the response. It is common across all clusters.
* D -- which is the covariance of the normal distribution from which each of the b_i are drawn. It is common across all clusters.
* sigma^2 -- which is the variance of e_i, which is assumed to be white.  It is common across all clusters.

Note that one key assumption of the MERF model is that the random effect is *linear*.  Though, this is limiting in some regards, it is still broadly useful for many problems. It is better than not modelling the random effect at all.

The algorithms implemented in this repo were developed by Ahlem Hajjem, Francois Bellavance, and Denis Larocque and published in a paper [here](http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599).  Many thanks to Ahlem and Denis for providing an R reference and aiding in the debugging of this code. Quick note, the published paper has a small typo in the update equation for sigma^2 which is corrected in the source code here.

## Using the Code

The MERF code is modelled after scikit-learn estimators.  To use, you instantiate a MERF object.  As of 1.0, you can pass any non-linear estimator for the fixed effect. By default this is a scikit-learn random forest, but you can pass any model you wish that conforms to the scikit-learn estimator API, e.g. LightGBM, XGBoost, a properly wrapped PyTorch neural net, 

Then you fit the model using training data.  As of 1.0, you can also pass a validation set to see the validation performance on it.  This is meant to feel similar to PyTorch where you can view the validation loss after each epoch of training. After fitting you can predict responses from data, either from known (cluster in training set) or new (cluster not in training set) clusters.

For example:

```
> from merf import MERF
> merf = MERF()
> merf.fit(X_train, Z_train, clusters_train, y_train)
> y_hat = merf.predict(X_test, Z_test, clusters_test)
```

Alternatively: 

```
> from lightgbm import LGBMRegressor
> lgbm = LGBMRegressor()
> mrf_lgbm = MERF(lgbm, max_iterations=15)
> mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train, X_val, Z_val, clusters_val, y_val)
> y_hat = merf.predict(X_test, Z_test, clusters_test)
```

Note that training is slow because the underlying expectation-maximization (EM) algorithm requires many calls to the non-linear fixed effects model, e.g. random forest. That being said, this implemtataion has early stopping which aborts the EM algorithm if the generalized log-likelihood (GLL) stops significantly improving.

## Tour of the Source Code

The `\merf` directory contains all the source code:

* `merf.py` is the key module that contains the MERF class. It is imported at the package level.
* `merf_test.py` contain some simple unit tests.
* `utils.py` contains a class for generating synthetic data that can be used to test the accuracy of MERF.  The process implemented is the same as that in this [paper](http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599).
* `viz.py` contains a plotting function that takes in a trained MERF object and plots various metrics of interest. 

The `\notebooks` directory contains some useful notebooks that show you how to use the code and evaluate MERF performance.  Most of the techniques implemented are the same as those in this [paper](http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599).

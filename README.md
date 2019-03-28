# Mixed Effects Random Forest

[![CircleCI](https://circleci.com/gh/manifoldai/merf.svg?style=svg)](https://circleci.com/gh/manifoldai/merf)

This repository contains a pure Python implementation of a mixed effects random forest (MERF) algorithm. It can be used, out of the box, to fit a MERF model and predict with it.
Read more about MERF in [this](https://towardsdatascience.com/mixed-effects-random-forests-6ecbb85cb177) blogpost. 

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

The MERF code is modelled after scikit-learn estimators.  To use, you instantiate a MERF object (with or without specifying parameters -- the defaults are sensible). Then you fit the model using training data. After fitting you can predict responses from data, either from known (cluster in training set) or new (cluster not in training set) clusters.

For example:

```
> from merf import MERF
> merf = MERF()
> merf.fit(X_train, Z_train, clusters_train, y_train)
> y_hat = merf.predict(X_test, Z_test, clusters_test)
```

Note that training is slow because the underlying expectation-maximization (EM) algorithm requires many calls to the random forest fit method. That being said, this implemtataion has early stopping which aborts the EM algorithm if the generalized log-likelihood (GLL) stops significantly improving.

In its current implementation the fixed effects learner is a random fores, but in theory the EM algorithm can be used with any learner. Our hope is to have future releases that do the same with gradient boosted trees and even deep neural networks.

## Tour of the Source Code

The `\src` directory contains all the source code:

* `merf.py` is the key module that contains the MERF class. It is imported at the package level.
* `tests.py` contain some simple unit tests.
* `utils.py` contains a class for generating synthetic data that can be used to test the accuracy of MERF.  The process implemented is the same as that in this [paper](http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599).

The `\notebooks` directory contains some useful notebooks that show you how to use the code and evaluate MERF performance.  Most of the techniques implemented are the same as those in this [paper](http://www.tandfonline.com/doi/abs/10.1080/00949655.2012.741599)

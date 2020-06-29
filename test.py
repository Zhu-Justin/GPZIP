from typing import Tuple, Optional
import tempfile
import pathlib

import datetime
import io

import tensorflow as tf
import gpflow

from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float

import warnings

warnings.filterwarnings("ignore")

import numpy.random as r
import numpy as np
from simGP import rzip, rzin
import matplotlib.pyplot as plt
N = 1000
X = np.arange(N).reshape(-1, 1)
Yp = rzip(N).reshape(-1, 1)
Yn = rzin(N).reshape(-1, 1)

plt.figure()
plt.scatter(X, Yp)
plt.title('ZIP')
plt.figure()
plt.scatter(X, Yn)
plt.title('ZIN')

M = 50 # Number of inducing locations

kern = gpflow.kernels.RBF()
Z = X[:M, :].copy() # Initialise inducing locations to the first M inputs in the dataset
m = gpflow.models.SVGP(X, Yp, kern, gpflow.likelihoods.Gaussian(), Z)
m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), Z)

m.maximum_log_likelihood_objective((X,Yp))

# Samples
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 10000 # Number of training observations


X = r.rand(N, 1) * 2 - 1 # X values
Y = func(X) + 0.2 * r.randn(N, 1) # Noisy Y values
plt.plot(X, Y, 'x', alpha=0.2)
D = X.shape[1]
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
plt.plot(Xt, Yt, c='k');

M = 50 # Number of inducing locations

kern = gpflow.kernels.RBF(D)
Z = X[:M, :].copy() # Initialise inducing locations to the first M inputs in the dataset
# m = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), Z)
likelihood = gpflow.likelihoods.Gaussian()
m = gpflow.models.SVGP((X,Yp),kern, Z)
m = gpflow.models.SVGP(kern, likelihood, Z)
m.maximum_log_likelihood_objective((X,Y))

evals = [m.maximum_log_likelihood_objective((X, Y)) for _ in range(100)]
plt.hist(evals, label='Minibatch Estimations')


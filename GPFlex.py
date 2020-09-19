import numpy as np
import pandas as pd
import numpy.random as r
import tensorflow as tf
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient

from gpflow.utilities import print_summary

from gpflow import set_trainable
import matplotlib.pyplot as plt

def gaussian1(N=100, pi=0.5, mu1=1, sigma1=1, mu2=-1, sigma2=2):
    X = np.linspace(0, 30, N)[:, None] * 1.0
    y_pi = r.uniform(size=N) < pi
    groups = np.where(y_pi, 0, 1)
    F = r.normal(mu1,sigma1, size=N) * groups + r.normal(mu2,sigma2, size=N) * (1 - groups)
    Y = F[:, None]*1.0  # Noisy data
    return X, Y, groups[:, None]

def gaussian2(N=100, pi=0.5, mu1=1, sigma1=1, mu2=-1, sigma2=2, epsilon=0.001):
    X = np.linspace(0, 30, N)[:, None]*1.0
    y_pi = r.uniform(size=N) < pi
    groups = np.where(y_pi, 0, 1)
    F = r.normal(0,epsilon, size=N) * groups + r.normal(mu1,sigma1, size=N) * (1 - groups)
    Y = F[:, None]*1.0  # Noisy data
    return X, Y, groups[:, None]

def gaussian3(N=100, pi=0.5, mu1=1, sigma1=1, mu2=-1, sigma2=2, epsilon=0.001):
    X = np.linspace(0, 30, N)[:, None]*1.0
    y_pi = r.uniform(size=N) < pi
    groups = np.where(y_pi, 0, 1)
    F = r.normal(0,epsilon, size=N) * groups + r.normal(0,epsilon, size=N) * (1 - groups)
    Y = F[:, None]*1.0  # Noisy data
    return X, Y, groups[:, None]

def zinormal(N=100, pi=0.5, mu1=1, sigma1=1, mu2=-1, sigma2=2, epsilon=0.001):
    X = np.linspace(0, 30, N)[:, None] *1.0
    y_pi = r.uniform(size=N) < pi
    groups = np.where(y_pi, 0, 1)
    F = r.normal(mu1,sigma1, size=N) * groups
    Y = F[:, None]*1.0  # Noisy data
    return X, Y, groups[:, None]

def zipoisson(N=100, pi=0.5, lam=1):
    X = np.linspace(0, 30, N)[:, None]
    y_pi = r.uniform(size=N) < pi
    groups = np.where(y_pi, 0, 1)
    F = r.poisson(lam, size=N) * groups
    Y = F[:, None]*1.0  # Noisy data
    return X, Y, groups[:, None]


def analyze(f, title="Plot", rawplot=True, modelplot=True,summary=True):
    # Obtain randomly generated data
    X, Y, groups = f()
    Y_data = np.hstack([Y, groups])
    # Model construction (notice that num_latent_gps is 1)
    likelihood = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian(variance=1.0),
         gpflow.likelihoods.Gaussian(variance=1.0)]
    )
    natgrad = NaturalGradient(gamma=1.0)
    adam = tf.optimizers.Adam()
    kernel = gpflow.kernels.Matern52(lengthscales=0.5)
    model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)
    for _ in range(ci_niter(1000)):
        natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])

    # Plot of the raw data.
    if rawplot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        _ = ax.plot(X, Y_data, "kx")
        plt.xlabel("Minutes")
        plt.ylabel("Value")
        plt.title(title)
        plt.savefig(title+'.png')

    # Plot of GP model
    if modelplot:
        xx = np.linspace(0, 30, 200)[:, None]
        mu, var = model.predict_f(xx)

        plt.figure(figsize=(12, 6))
        plt.plot(xx, mu, "C0")
        plt.plot(xx, mu + 2 * np.sqrt(var), "C0", lw=0.5)
        plt.plot(xx, mu - 2 * np.sqrt(var), "C0", lw=0.5)
        plt.plot(X, Y, "C1x", mew=2)
        plt.xlabel("Minutes")
        plt.ylabel("Value")
        plt.title(title)
        plt.savefig(title+' GP model.png')

    if summary:
        print_summary(model)

    return model


# Obtain the corresponding GP Models
m1 = analyze(gaussian1, 'N(mu1,sigma1), N(mu2,sigma2)')
m2 = analyze(gaussian2, 'N(0,epsilon), N(mu, sigma)')
m3 = analyze(gaussian3, '2x N(0, epsilon)')
m4 = analyze(zinormal, '0, N(mu,sigma)')
m5 = analyze(zipoisson, '0, Pois(lam)')




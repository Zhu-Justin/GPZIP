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



X1 , Y1 , groups1 = gaussian1()

# X2 , Y2 , groups2 = gaussian2()
# X3 , Y3 , groups3 = gaussian3()
# X4 , Y4 , groups4 = zin()
# X5 , Y5 , groups5 = zip()


def analyze(f, title="Plot"):
    X, Y, groups = f()
    Y_data = np.hstack([Y, groups])
    likelihood = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian(variance=1.0),
         gpflow.likelihoods.Gaussian(variance=1.0)]
    )
    # model construction (notice that num_latent_gps is 1)
    natgrad = NaturalGradient(gamma=1.0)
    adam = tf.optimizers.Adam()
    kernel = gpflow.kernels.Matern52(lengthscales=0.5)
    model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)
    # here's a plot of the raw data.
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    _ = ax.plot(X, Y_data, "kx")
    plt.xlabel("Minutes")
    plt.ylabel("Value")
    plt.title(title)
    plt.savefig(title+'.png')
    for _ in range(ci_niter(1000)):
        natgrad.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
# let's do some plotting!
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

    print_summary(model)
    # print(type(summary))
    # summary.to_markdown(title+'.md')
    # plt.set_xlim(0, 30)
    # _ = ax.plot(xx, 2.5 * np.sin(6 * xx) + np.cos(3 * xx), "C2--")

    # plt.errorbar(
    #     X.squeeze(),
    #     Y.squeeze(),
    #     # yerr=2 * (np.sqrt(NoiseVar)).squeeze(),
    #     marker="x",
    #     lw=0,
    #     elinewidth=1.0,
    #     color="C1",
    # )
    # _ = plt.xlim(-5, 5)
    return

analyze(gaussian1, 'N(mu1,sigma1), N(mu2,sigma2)')
analyze(gaussian2, 'N(0,epsilon), N(mu, sigma)')
analyze(gaussian3, '2x N(0, epsilon)')
analyze(zinormal, '0, N(mu,sigma)')
analyze(zipoisson, '0, Pois(lam)')





import tensorflow as tf
import gpflow as gpf

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

matdata = sio.loadmat('zero-inflated-gp/data/toydata.mat')

Xtrain = matdata['x']
Ytrain = matdata['y']

plt.scatter(Xtrain,Ytrain)
plt.title("Simulated data")
plt.show()

num_iterations = 8000
num_inducing  = 10

kf = gpf.kernels.RBF(1)

kf.lengthscales = 2.
kf.variance = 1.

kg = gpf.kernels.RBF(1)
kg.lengthscales = 2.
kg.variance = 5.

# initialise equally spaced inducing point locations
Zf = np.delete(np.linspace(min(Xtrain),max(Xtrain),num_inducing,endpoint=False),0).transpose().reshape(-1,1)
Zg = np.delete(np.linspace(min(Xtrain),max(Xtrain),num_inducing,endpoint=False),0).transpose().reshape(-1,1)


# TODO: model definition
from onoffgpf import OnOffSVGP2, OnOffLikelihood
m = OnOffSVGP(Xtrain, Ytrain ,kernf=kf,kerng=kg ,likelihood = OnOffLikelihood() ,Zf = Zf,Zg = Zg)
m.optimize(maxiter = num_iterations) #,method= tf.train.AdamOptimizer(learning_rate = 0.01)
m.compute_log_likelihood()


# model plot
from onoffgpf.PlotOnOff1D import PlotOnOff1D
PlotOnOff1D(m)



# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gpflow as gpf
from onoffgpf import OnOffSVGP, OnOffLikelihood
from onoffgpf.PlotOnOff1D import PlotOnOff1D

import gpflow.model

num_iterations = 8000
num_inducing   = 10

# initalize kernel parameters
kf = gpf.kernels.RBF(1)
kf.lengthscales = 2.
kf.variance = 1.

kg = gpf.kernels.RBF(1)
kg.lengthscales = 2.
kg.variance = 5.

# initialise equally spaced inducing point locations
Zf = np.delete(np.linspace(min(Xtrain),max(Xtrain),num_inducing,endpoint=False),0).transpose().reshape(-1,1)
Zg = np.delete(np.linspace(min(Xtrain),max(Xtrain),num_inducing,endpoint=False),0).transpose().reshape(-1,1)

# model definition
m = OnOffSVGP(Xtrain, Ytrain
              ,kernf=kf,kerng=kg
              ,likelihood = OnOffLikelihood()
              ,Zf = Zf,Zg = Zg
             )

# fix the model noise term
m.likelihood.variance = 0.01
m.likelihood.variance.fixed = False

m.optimize(maxiter = num_iterations) #,method= tf.train.AdamOptimizer(learning_rate = 0.01)
m.compute_log_likelihood()
# model plot
PlotOnOff1D(m)



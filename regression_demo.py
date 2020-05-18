import gpflow
import numpy as np
import matplotlib.pyplot as plt

# The lines below are specific to the notebook format
# %matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (12, 6)
# plt = matplotlib.pyplot

data = np.genfromtxt('data/regression_1D.csv', delimiter=',')
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

plt.plot(X, Y, 'kx', mew=2)


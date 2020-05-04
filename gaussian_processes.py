from scipy import stats
import numpy as np
import sys
import os
import platform
import pyreadr
import locale
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(1)
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# loading data from mounted mBox "HeartSteps" folder
def initsys(sysname):
    sysvar = dict()
    if sysname == "Windows":
        sysvar = {'locale' : 'English', 
                'mbox':'Z:/HeartSteps/'}
    elif sysname == "Darwin":
        sysvar = {'locale' : 'en_US', 
                'mbox':'/Volumes/dav/HeartSteps/'}
    elif sysname == "Linux":
        sysvar = {'locale' : 'en_US.UTF-8', 
                'mbox':'~/mbox/HeartSteps/'}

    if sysvar:
        sysvar['mbox.data'] = sysvar['mbox'] + "Tianchen Qian/jbslot_traveldayremoved_90min.RDS"
    return sysvar


sysvar = initsys(platform.system())
# File must exist!
assert(sysvar)

locale.setlocale(locale.LC_ALL, sysvar['locale'])
os.environ['TZ'] = "GMT"

# read from R
rdata = pyreadr.read_r(sysvar['mbox.data'])
# convert to pandas
df = rdata[None]
total_T = 90

df.columns
df.avail == True
df['min.after.decision']

jbslot90 = df[(df['avail'] == True) 
              & (df['min.from.last.decision'] >= total_T)
                & (df['min.after.decision'] < total_T)]


user1 = jbslot90[jbslot90['user'] == 1]
m1 = user1['send']
m2 = user1['steps']
m1 = np.array(m1)
m1 = m1.astype(int)
m2 = np.array(m2)
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
def f(x):
    """The function to predict."""
    return x * np.sin(x)
# Observations
y = f(X).ravel()


gp.fit(X, y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

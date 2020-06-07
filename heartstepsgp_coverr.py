import pandas as pd
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

# Helper functions for plotting
def setcanvas():
    plt.figure()
    plt.xlabel(r'$t$ (minutes from decision point)')
    plt.ylabel(r'$y$ (step counts in a minute)')
    return

def savecanvas(title='Untitled'):
    # title += ''
    plt.title(title)
    plt.savefig(title+'.png')
    return

df = pd.read_pickle('HeartSteps.pkl')

total_T = 90
df.columns
df.avail == True
df['min.after.decision']

jbslot90 = df[(df['avail'] == True) 
              & (df['min.from.last.decision'] >= total_T)
                & (df['min.after.decision'] < total_T)]


user1 = jbslot90[(jbslot90['user'] == 1)]
# Uncomment to set user1 to first 10-15 decision points
user1 = user1[user1['decision.index.nogap'] <= 15]

m1 = user1['send']
m1 = np.array(m1)
m1 = m1.astype(int)

m2r = np.array(user1['steps'])
# Uncomment to take the log of (steps + 0.5)
m2 = user1['steps.log']
m2 = np.array(m2)
# Uncomment to center the steps of y
m2 = np.array(m2) - np.mean(m2)

m3 = np.array(user1['min.after.decision'])

# Observations
obs = user1.shape[0]
X = np.atleast_2d(m3)
X = np.vstack((np.ones(obs), m1, m3)).T
# Uncomment to not have intercepts
# X = np.vstack((m1, m3)).T

xmin = m3.min()
xmax = m3.max()

# Treatment variables for prediction
xtreatment = np.vstack((np.ones(obs), np.ones(obs), np.linspace(xmin, xmax, obs))).T
xnotreatment = np.vstack((np.ones(obs), np.zeros(obs), np.linspace(xmin, xmax, obs))).T


yr = m2r.ravel()
y = m2.ravel()

# kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale = [.1, .1], length_scale_bounds=[(1e-2, 1e2),(1e-2, 1e2)]) \ + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-4))
# # Instantiate a Gaussian Process model
gpr = GaussianProcessRegressor()
gp = GaussianProcessRegressor()
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gpr = GaussianProcessRegressor(kernel=kernel)
# gp = GaussianProcessRegressor(kernel=kernel)

# # # Fit to data using Maximum Likelihood Estimation of the parameters
gpr.fit(X, yr)
gp.fit(X, y)
# TODO check that the prior mean is set to 0

# # # Make the prediction on the meshed x-axis (ask for MSE as well)
y_notreatmentr , sigma0r = gpr.predict(xnotreatment, return_std=True)
y_treatmentr , sigma1r = gpr.predict(xtreatment, return_std=True)
y_notreatment , sigma0 = gp.predict(xnotreatment, return_std=True)
y_treatment , sigma1 = gp.predict(xtreatment, return_std=True)

# # Plot the function, the prediction and the 95% confidence interval based on the MSE
# Figure 1
# setcanvas()
# s = plt.scatter(m3, y, c=m1)
# plt.legend(handles=s.legend_elements()[0], labels=['No Treatment','Treatment'])
# title = 'Centered Log-Step Counts for User 1, DP=15'
# savecanvas(title)

setcanvas()
time = np.linspace(xmin, xmax, obs)
plt.plot(time, y_treatmentr, label='Treatment Prediction')
plt.plot(time, y_notreatmentr, label='No Treatment Prediction')
plt.plot(time, y_treatment, label='Treatment Prediction (Centered-Log)')
plt.plot(time, y_notreatment, label='No Treatment Prediction (Centered-Log)')
plt.legend()
title = 'Predicted Step Counts Using GP for User 1, DP=15'
savecanvas(title)

setcanvas()
s = plt.scatter(m3, yr, c=m1)
time = np.linspace(xmin, xmax, obs)
plt.plot(time, y_treatmentr, label='Treatment Prediction')
plt.plot(time, y_notreatmentr, label='No Treatment Prediction')
plt.legend(handles=s.legend_elements()[0], labels=['No Treatment','Treatment'])
title = 'GP for User 1, DP=15'
savecanvas(title)

setcanvas()
s = plt.scatter(m3, y, c=m1)
time = np.linspace(xmin, xmax, obs)
plt.plot(time, y_treatment, label='Treatment Prediction')
plt.plot(time, y_notreatment, label='No Treatment Prediction')
plt.legend(handles=s.legend_elements()[0], labels=['No Treatment','Treatment'])
title = 'GP for User 1, DP=15 (Centered Log)'
savecanvas(title)

# setcanvas()
# plt.scatter(m3, y, c=m1)
# time = np.linspace(xmin, xmax, obs)
# plt.plot(time, y_treatment, label='Treatment Prediction')
# plt.plot(time, y_notreatment, label='No Treatment Prediction')
# plt.fill(np.concatenate([time, time[::-1]]),
#          np.concatenate([y_treatment - 1.9600 * sigma,
#                         (y_treatment + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# title = 'Treatment Confidence intervals for User 1, DP=15'
# plt.legend()
# savecanvas(title)

# setcanvas()
# plt.scatter(m3, y, c=m1)
# time = np.linspace(xmin, xmax, obs)
# plt.plot(time, y_treatment, label='Treatment Prediction')
# plt.plot(time, y_notreatment, label='No Treatment Prediction')
# plt.fill(np.concatenate([time, time[::-1]]),
#          np.concatenate([y_notreatment - 1.9600 * sigma,
#                         (y_notreatment + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# title = 'Non-Treatment Confidence Intervals for User 1, DP=15'
# plt.legend()
# savecanvas(title)

setcanvas()
plt.scatter(m3, yr, c=m1)
time = np.linspace(xmin, xmax, obs)
plt.plot(time, y_treatmentr, label='Treatment Prediction')
plt.plot(time, y_notreatmentr, label='No Treatment Prediction')
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([y_treatmentr - 1.9600 * sigma1r,
                        (y_treatmentr + 1.9600 *
                         sigma1r)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval (treatment)')
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([y_notreatmentr - 1.9600 * sigma0r,
                         (y_notreatmentr + 1.9600 * sigma0r)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval (non-treatment)')
title = 'CI for User 1, DP=15'
plt.legend()
savecanvas(title)


setcanvas()
plt.scatter(m3, y, c=m1)
time = np.linspace(xmin, xmax, obs)
plt.plot(time, y_treatment, label='Treatment Prediction')
plt.plot(time, y_notreatment, label='No Treatment Prediction')
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([y_treatment - 1.9600 * sigma1,
                        (y_treatment + 1.9600 * sigma1)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval')
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([y_notreatment - 1.9600 * sigma0,
                        (y_notreatment + 1.9600 * sigma0)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
title = 'CI for User 1, DP=15 (Centered Log)'
plt.legend()
savecanvas(title)

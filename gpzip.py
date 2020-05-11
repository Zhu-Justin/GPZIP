from scipy import stats
import numpy as np
import sys
import os
import platform
import pyreadr
import locale
import datetime
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(1)
# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# Helper functions for plotting
def setcanvas():
    plt.figure()
    plt.xlabel(r'$t$ (minutes from decision point)')
    plt.ylabel(r'$y$ (step counts in a minute)')
    return

def savecanvas(title='Untitled'):
    # title += ''
    plt.title(title)
    times = datetime.datetime.now()
    plt.savefig(title+times+'.png')
    return

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

# read from R data
rdata = pyreadr.read_r(sysvar['mbox.data'])
# convert to pandas
df = rdata[None]
total_T = 90
# Uncomment to restrict to 30 as opposed to 90
# total_T = 30

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
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel)
gp = GaussianProcessRegressor(kernel=kernel,normalize_y=True)
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

# load classification and regression results
# do elemen-wise multiplication of these results
# both propabilities and hardcut
# get summaries and save pickle
# save the index of filtered observations
# ****************************************************************
# library import block
# ****************************************************************
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import logging
import time
import sys
from scipy.cluster.vq import kmeans
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.switch_backend('agg')

float_type = tf.float64
jitter_level = 1e-5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(scriptPath):
    tf.reset_default_graph()
    parentDir = '/'.join(os.path.dirname(os.path.realpath(scriptPath)).split('/')[:-1])  #
    subDir = "/" + scriptPath.split("/")[-2].split(".py")[0] + "/"
    sys.path.append(parentDir)

    clf_modelPath = parentDir + subDir + 'results_scgp.pickle'
    reg_modelPath = parentDir + subDir + 'results_svgp.pickle'
    logPath   = parentDir + subDir + 'modelsumm_zi.log'

    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(logPath))

    data = pickle.load(open(parentDir + subDir +"data.pickle","rb"))
    Xtrain = data['Xtrain']
    Ytrain = data['Ytrain']
    Xtest = data['Xtest']
    Ytest = data['Ytest']

    # load results from the clasiifier model
    clf_results = pickle.load(open(clf_modelPath,"rb"))
    reg_results = pickle.load(open(reg_modelPath,"rb"))

    # combined results

    # ****************************************************************
    # model predictions
    # ****************************************************************
    train_clf_prob = clf_results['pred_train']['pfmean']
    test_clf_prob = clf_results['pred_test']['pfmean']
    train_clf_indc = (train_clf_prob > 0.5) * 1.0
    test_clf_indc = (test_clf_prob > 0.5) * 1.0

    pred_train_zi_prob = train_clf_prob * reg_results['pred_train']['fmean']
    pred_test_zi_prob = test_clf_prob * reg_results['pred_test']['fmean']
    pred_train_zi_indc = train_clf_indc * reg_results['pred_train']['fmean']
    pred_test_zi_indc = test_clf_indc * reg_results['pred_test']['fmean']

    def rmse(predict,actual):
        predict = np.maximum(predict,0)
        return np.sqrt(np.mean((actual-predict)**2))

    def mad(predict,actual):
        predict = np.maximum(predict,0)
        return np.mean(np.abs(actual-predict))

    train_zi_prob_reg_rmse = rmse(pred_train_zi_prob,Ytrain)
    logger.info("rmse on train set for zi prob : "+str(train_zi_prob_reg_rmse))
    train_zi_prob_reg_mae = mad(pred_train_zi_prob,Ytrain)
    logger.info("mae on train set for zi prob : "+str(train_zi_prob_reg_mae))

    test_zi_prob_reg_rmse = rmse(pred_test_zi_prob,Ytest)
    logger.info("rmse on test set for zi prob  : "+str(test_zi_prob_reg_rmse))
    test_zi_prob_reg_mae = mad(pred_test_zi_prob,Ytest)
    logger.info("mae on test set for zi prob  : "+str(test_zi_prob_reg_mae))

    train_zi_indc_reg_rmse = rmse(pred_train_zi_indc,Ytrain)
    logger.info("rmse on train set for zi indc : "+str(train_zi_indc_reg_rmse))
    train_zi_indc_reg_mae = mad(pred_train_zi_indc,Ytrain)
    logger.info("mae on train set for zi indc : "+str(train_zi_indc_reg_mae))

    test_zi_indc_reg_rmse = rmse(pred_test_zi_indc,Ytest)
    logger.info("rmse on test set for zi indc  : "+str(test_zi_indc_reg_rmse))
    test_zi_indc_reg_mae = mad(pred_test_zi_indc,Ytest)
    logger.info("mae on test set for zi indc  : "+str(test_zi_indc_reg_mae))

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # ****************************************************************
    # return values
    # ****************************************************************
    results = {
               'pred_train_zi_prob':pred_train_zi_prob,
               'pred_test_zi_prob':pred_test_zi_prob,
               'pred_train_zi_indc':pred_train_zi_indc,
               'pred_test_zi_indc':pred_test_zi_indc,
               'train_zi_prob_reg_rmse':train_zi_prob_reg_rmse,
               'train_zi_prob_reg_mae':train_zi_prob_reg_mae,
               'test_zi_prob_reg_rmse':test_zi_prob_reg_rmse,
               'test_zi_prob_reg_mae':test_zi_prob_reg_mae,
               'train_zi_indc_reg_rmse':train_zi_indc_reg_rmse,
               'train_zi_indc_reg_mae':train_zi_indc_reg_mae,
               'test_zi_indc_reg_rmse':test_zi_indc_reg_rmse,
               'test_zi_indc_reg_mae':test_zi_indc_reg_mae
               }
    pickle.dump(results,open(parentDir+ subDir +"results_zi.pickle","wb"))


if __name__ == "__main__":
    scriptPath = sys.argv[0]
    main(scriptPath)

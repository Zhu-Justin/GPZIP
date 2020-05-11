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

# read from R data
rdata = pyreadr.read_r(sysvar['mbox.data'])
# convert to pandas
df = rdata[None]
df.to_pickle("./HeartSteps.pkl")

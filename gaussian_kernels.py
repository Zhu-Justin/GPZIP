from scipy import stats
import numpy as np
import sys
import os
import platform
import pyreadr
import locale
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
# R has default namespace with non-conflict function names

# os.makedirs("python_exploratory_plots")

# The following code will create thousands of plots (for all available decision points). This can take a while.

# for loop, loop over decision for each user
# work on theta from a single user, single decision point

# def measure(n):
#     "Measurement model, return two coupled measurements."
#     m1 = np.random.normal(size=n)
#     m2 = np.random.normal(scale=0.5, size=n)
#     return m1+m2, m1-m2
# 1. Generate random data    
# m1, m2 = measure(2000)
# xmin = m1.min()
# xmax = m1.max()
# ymin = m2.min()
# ymax = m2.max()

# Perform a kernel density estimate on the data:
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
k = kernel(positions)
k.shape

# Plot the results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()



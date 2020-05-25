from typing import Tuple, Optional
import tempfile
import pathlib

import datetime
import io
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float

import warnings

warnings.filterwarnings("ignore")

# Make tensorboard work in notebook
output_logdir = "/tmp/tensorboard"

!rm -rf "{output_logdir}"
!mkdir "{output_logdir}"

%load_ext tensorboard
# %matplotlib inline


def enumerated_logdir(_logdir_id: int = [0]):
    logdir = pathlib.Path(output_logdir, str(_logdir_id[0]))
    _logdir_id[0] += 1
    return str(logdir)


# Set up random seeds and default float for gpflow tensors
gpflow.config.set_default_float(np.float64)
np.random.seed(0)
tf.random.set_seed(0)

# Loading data using tensorflow datasets
def noisy_sin(x):
    return tf.math.sin(x) + 0.1 * tf.random.normal(x.shape, dtype=default_float())

num_train_data, num_test_data = 100, 500

X = tf.random.uniform((num_train_data, 1), dtype=default_float()) * 10
Xtest = tf.random.uniform((num_test_data, 1), dtype=default_float()) * 10

Y = noisy_sin(X)
Ytest = noisy_sin(Xtest)
data = (X, Y)
plt.plot(X, Y, "xk")
plt.show()

# prefetch size uses tf.data.experimental.AUTOTUNE
# recommended by tensorflow
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
test_dataset = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))

batch_size = 32
num_features = 10
prefetch_size = tf.data.experimental.AUTOTUNE
shuffle_buffer_size = num_train_data // 2
num_batches_per_epoch = num_train_data // batch_size

original_train_dataset = train_dataset
train_dataset = (
    train_dataset.repeat()
    .prefetch(prefetch_size)
    .shuffle(buffer_size=shuffle_buffer_size)
    .batch(batch_size)
)

print(f"prefetch_size={prefetch_size}")
print(f"shuffle_buffer_size={shuffle_buffer_size}")
print(f"num_batches_per_epoch={num_batches_per_epoch}")

# Define a GP model
kernel = gpflow.kernels.SquaredExponential(variance=2.0)
likelihood = gpflow.likelihoods.Gaussian()
inducing_variable = np.linspace(0, 10, num_features).reshape(-1, 1)

model = gpflow.models.SVGP(
    kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable
)

from gpflow import set_trainable

# Set a module to be nontrainable using the auxiliary
# method set_trainable
set_trainable(likelihood, False)
set_trainable(kernel.variance, False)

set_trainable(likelihood, True)
set_trainable(kernel.variance, True)

kernel.lengthscales.assign(0.5)

# print_summary contains all the final model specifications
from gpflow.utilities import print_summary

print_summary(model)  # same as print_summary(model, fmt="fancy_table")

gpflow.config.set_default_summary_fmt("notebook")

# For formatting purposes (hard to see in ipython terminal)
# print_summary(model)  # same as print_summary(model, fmt="notebook")

model

# Training using training_loss
vgp_model = gpflow.models.VGP(data, kernel, likelihood)
optimizer = tf.optimizers.Adam()
optimizer.minimize(
    vgp_model.training_loss, vgp_model.trainable_variables
)  # Note: this does a single step
# In practice, you will need to call minimize() many times, this will be further discussed below.

optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(
    vgp_model.training_loss, vgp_model.trainable_variables, options=dict(maxiter=ci_niter(1000))
)


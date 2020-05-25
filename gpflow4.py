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

# scipy does the full optimization on a single call to
# minimize()
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(
    vgp_model.training_loss, vgp_model.trainable_variables, options=dict(maxiter=ci_niter(1000))
)

vgp_model.training_loss_closure()  # compiled
vgp_model.training_loss_closure(compile=True)  # compiled
vgp_model.training_loss_closure(compile=False)  # uncompiled, same as vgp_model.training_loss

assert isinstance(model, gpflow.models.SVGP)
model.training_loss(data)

# optimizing has a training loss closure that takes the
# data and returns a closure that computes the training
# loss on the data
optimizer = tf.optimizers.Adam()
training_loss = model.training_loss_closure(
    data
)  # We save the compiled closure in a variable so as not to re-compile it each step
optimizer.minimize(training_loss, model.trainable_variables)  # Note that this does a single step

batch_size = 5
batched_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
training_loss = model.training_loss_closure(iter(batched_dataset))

optimizer.minimize(training_loss, model.trainable_variables)  # Note that this does a single step


# Training using gradient tapes
def optimization_step(model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def simple_training_loop(model: gpflow.models.SVGP, epochs: int = 1, logging_epoch_freq: int = 10):
    tf_optimization_step = tf.function(optimization_step)

    batches = iter(train_dataset)
    for epoch in range(epochs):
        for _ in range(ci_niter(num_batches_per_epoch)):
            tf_optimization_step(model, next(batches))

        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(data)}")


# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for the deepof tensorflow models. See documentation for details

"""

from itertools import combinations
from typing import Any, Tuple

from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


# Helper functions and classes
class exponential_learning_rate(tf.keras.callbacks.Callback):
    """Simple class that allows to grow learning rate exponentially during training"""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    # noinspection PyMethodOverriding
    def on_batch_end(self, batch, logs):
        """This callback acts after processing each batch"""

        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_learning_rate(
    model, X, y, epochs=1, batch_size=32, min_rate=10 ** -5, max_rate=10
):
    """Trains the provided model for an epoch with an exponentially increasing learning rate"""

    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = K.exp(K.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = exponential_learning_rate(factor)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):  # pragma: no cover
    """Plots learing rate versus the loss function of the model"""

    plt.plot(rates, losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


@tf.function
def far_away_uniform_initialiser(
    shape: tuple, minval: int = 0, maxval: int = 15, iters: int = 100000
) -> tf.Tensor:
    """
    Returns a uniformly initialised matrix in which the columns are as far as possible

        Parameters:
            - shape (tuple): shape of the object to generate.
            - minval (int): Minimum value of the uniform distribution from which to sample
            - maxval (int): Maximum value of the uniform distribution from which to sample
            - iters (int): the algorithm generates values at random and keeps those runs that
            are the farthest apart. Increasing this parameter will lead to more accurate,
            results while making the function run slowlier.

        Returns:
            - init (tf.Tensor): tensor of the specified shape in which the column vectors
             are as far as possible

    """

    init = tf.random.uniform(shape, minval, maxval)
    init_dist = tf.abs(tf.norm(tf.math.subtract(init[1:], init[:1])))
    i = 0

    while tf.less(i, iters):
        temp = tf.random.uniform(shape, minval, maxval)
        dist = tf.abs(tf.norm(tf.math.subtract(temp[1:], temp[:1])))

        if dist > init_dist:
            init_dist = dist
            init = temp

        i += 1

    return init


def compute_kernel(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """

    Computes the MMD between the two specified vectors using a gaussian kernel.

        Parameters:
            - x (tf.Tensor): left tensor
            - y (tf.Tensor): right tensor

        Returns
            - kernel (tf.Tensor): returns the result of applying the kernel, for
            each training instance

    """

    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    kernel = tf.exp(
        -tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32)
    )
    return kernel


@tf.function
def compute_mmd(tensors: Tuple[Any]) -> tf.Tensor:
    """

    Computes the MMD between the two specified vectors using a gaussian kernel.

        Parameters:
            - tensors (tuple): tuple containing two tf.Tensor objects

        Returns
            - mmd (tf.Tensor): returns the maximum mean discrepancy for each
            training instance

    """

    x = tensors[0]
    y = tensors[1]

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )
    return mmd


# Custom auxiliary classes
class one_cycle_scheduler(tf.keras.callbacks.Callback):
    """

    One cycle learning rate scheduler.
    Based on https://arxiv.org/pdf/1506.01186.pdf

    """

    def __init__(
        self,
        iterations: int,
        max_rate: float,
        start_rate: float = None,
        last_iterations: int = None,
        last_rate: float = None,
    ):
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
        self.history = {}

    def _interpolate(self, iter1: int, iter2: int, rate1: float, rate2: float) -> float:
        return (rate2 - rate1) * (self.iteration - iter1) / (iter2 - iter1) + rate1

    # noinspection PyMethodOverriding,PyTypeChecker
    def on_batch_begin(self, batch: int, logs):
        """ Defines computations to perform for each batch """

        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))

        if self.iteration < self.half_iteration:
            rate = self._interpolate(
                0, self.half_iteration, self.start_rate, self.max_rate
            )
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(
                self.half_iteration,
                2 * self.half_iteration,
                self.max_rate,
                self.start_rate,
            )
        else:
            rate = self._interpolate(
                2 * self.half_iteration,
                self.iterations,
                self.start_rate,
                self.last_rate,
            )
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)


class uncorrelated_features_constraint(Constraint):
    """

    tf.keras.constraints.Constraint subclass that forces a layer to have uncorrelated features.
    Useful, among others, for auto encoder bottleneck layers

    """

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"encoding_dim": self.encoding_dim, "weightage": self.weightage})
        return config

    def get_covariance(self, x):
        """Computes the covariance of the elements of the passed layer"""

        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(
            x_centered.get_shape()[0], tf.float32
        )

        return covariance

    # Constraint penalty
    # noinspection PyUnusedLocal
    def uncorrelated_feature(self, x):
        """Adds a penalty on feature correlation, forcing more independent sets of weights"""

        if self.encoding_dim <= 1:  # pragma: no cover
            return 0.0
        else:
            output = K.sum(
                K.square(
                    self.covariance
                    - tf.math.multiply(self.covariance, tf.eye(self.encoding_dim))
                )
            )
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)


# Custom Layers
class MCDropout(tf.keras.layers.Dropout):
    """Equivalent to tf.keras.layers.Dropout, but with training mode enabled at prediction time.
    Useful for Montecarlo predictions"""

    def call(self, inputs, **kwargs):
        """Overrides the call method of the subclassed function"""
        return super().call(inputs, training=True)


class DenseTranspose(Layer):
    """Mirrors a tf.keras.layers.Dense instance with transposed weights.
    Useful for decoder layers in autoencoders, to force structure and
    decrease the effective number of parameters to train"""

    def __init__(self, dense, output_dim, activation=None, **kwargs):
        self.dense = dense
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update(
            {
                "dense": self.dense,
                "output_dim": self.output_dim,
                "activation": self.activation,
            }
        )
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, batch_input_shape):
        """Updates Layer's build method"""

        self.biases = self.add_weight(
            name="bias",
            shape=self.dense.get_input_at(-1).get_shape().as_list()[1:],
            initializer="zeros",
        )
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        """Updates Layer's call method"""

        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    def compute_output_shape(self, input_shape):  # pragma: no cover
        """Outputs the transposed shape"""

        return input_shape[0], self.output_dim


class KLDivergenceLayer(tfpl.KLDivergenceAddLoss):
    """
    Identity transform layer that adds KL Divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"is_placeholder": self.is_placeholder})
        return config

    def call(self, distribution_a):
        """Updates Layer's call method"""

        kl_batch = self._regularizer(distribution_a)
        self.add_loss(kl_batch, inputs=[distribution_a])
        self.add_metric(
            kl_batch,
            aggregation="mean",
            name="kl_divergence",
        )
        # noinspection PyProtectedMember
        self.add_metric(self._regularizer._weight, aggregation="mean", name="kl_rate")

        return distribution_a


class MMDiscrepancyLayer(Layer):
    """
    Identity transform layer that adds MM Discrepancy
    to the final model loss.
    """

    def __init__(self, batch_size, prior, beta=1.0, *args, **kwargs):
        self.is_placeholder = True
        self.batch_size = batch_size
        self.beta = beta
        self.prior = prior
        super(MMDiscrepancyLayer, self).__init__(*args, **kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"beta": self.beta})
        config.update({"prior": self.prior})
        return config

    def call(self, z, **kwargs):
        """Updates Layer's call method"""

        true_samples = self.prior.sample(self.batch_size)
        # noinspection PyTypeChecker
        mmd_batch = self.beta * compute_mmd((true_samples, z))
        self.add_loss(K.mean(mmd_batch), inputs=z)
        self.add_metric(mmd_batch, aggregation="mean", name="mmd")
        self.add_metric(self.beta, aggregation="mean", name="mmd_rate")

        return z


class Gaussian_mixture_overlap(Layer):
    """
    Identity layer that measures the overlap between the components of the latent Gaussian Mixture
    using a specified metric (MMD, Wasserstein, Fischer-Rao)
    """

    def __init__(self, lat_dims, n_components, loss=False, samples=10, *args, **kwargs):
        self.lat_dims = lat_dims
        self.n_components = n_components
        self.loss = loss
        self.samples = samples
        super(Gaussian_mixture_overlap, self).__init__(*args, **kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"lat_dims": self.lat_dims})
        config.update({"n_components": self.n_components})
        config.update({"loss": self.loss})
        config.update({"samples": self.samples})
        return config

    @tf.function
    def call(self, target, **kwargs):
        """Updates Layer's call method"""

        dists = []
        for k in range(self.n_components):
            locs = (target[..., : self.lat_dims, k],)
            scales = tf.keras.activations.softplus(target[..., self.lat_dims :, k])

            dists.append(
                tfd.BatchReshape(tfd.MultivariateNormalDiag(locs, scales), [-1])
            )

        dists = [tf.transpose(gauss.sample(self.samples), [1, 0, 2]) for gauss in dists]

        # MMD-based overlap #
        intercomponent_mmd = K.mean(
            tf.convert_to_tensor(
                [
                    tf.vectorized_map(compute_mmd, [dists[c[0]], dists[c[1]]])
                    for c in combinations(range(len(dists)), 2)
                ],
                dtype=tf.float32,
            )
        )

        self.add_metric(
            -intercomponent_mmd, aggregation="mean", name="intercomponent_mmd"
        )

        if self.loss:
            self.add_loss(-intercomponent_mmd, inputs=[target])

        return target


class Dead_neuron_control(Layer):
    """
    Identity layer that adds latent space and clustering stats
    to the metrics compiled by the model
    """

    def __init__(self, *args, **kwargs):
        super(Dead_neuron_control, self).__init__(*args, **kwargs)

    # noinspection PyMethodOverriding
    def call(self, target, **kwargs):
        """Updates Layer's call method"""
        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(
            tf.math.zero_fraction(target), aggregation="mean", name="dead_neurons"
        )

        return target


class Entropy_regulariser(Layer):
    """
    Identity layer that adds cluster weight entropy to the loss function
    """

    def __init__(self, weight=1.0, axis=1, *args, **kwargs):
        self.weight = weight
        self.axis = axis
        super(Entropy_regulariser, self).__init__(*args, **kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"weight": self.weight})
        config.update({"axis": self.axis})

    def call(self, z, **kwargs):
        """Updates Layer's call method"""

        # axis=1 increases the entropy of a cluster across instances
        # axis=0 increases the entropy of the assignment for a given instance
        entropy = K.sum(tf.multiply(z + 1e-5, tf.math.log(z) + 1e-5), axis=self.axis)

        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(entropy, aggregation="mean", name="-weight_entropy")

        if self.weight > 0:
            self.add_loss(self.weight * K.sum(entropy), inputs=[z])

        return z

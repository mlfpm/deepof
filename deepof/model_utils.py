# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for the deepof tensorflow models. See documentation for details

"""

from functools import partial
from typing import Any, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

tfd = tfp.distributions
tfpl = tfp.layers


# Helper functions and classes
@tf.function
def compute_shannon_entropy(tensor):
    """Computes Shannon entropy for a given tensor"""
    tensor = tf.cast(tensor, tf.dtypes.int32)
    bins = (
        tf.math.bincount(tensor, dtype=tf.dtypes.float32)
        / tf.cast(tf.shape(tensor), tf.dtypes.float32)[0]
    )
    return -tf.reduce_sum(bins * tf.math.log(bins + 1e-5))


@tf.function
def get_k_nearest_neighbors(tensor, k, index):
    """Retrieve indices of the k nearest neighbors in tensor to the vector with the specified index"""
    query = tf.gather(tensor, index, batch_dims=0)
    distances = tf.norm(tensor - query, axis=1)
    max_distance = tf.sort(distances)[k]
    neighbourhood_mask = distances < max_distance
    return tf.squeeze(tf.where(neighbourhood_mask))


@tf.function
def get_neighbourhood_entropy(index, tensor, clusters, k):
    neighborhood = get_k_nearest_neighbors(tensor, k, index)
    cluster_z = tf.gather(clusters, neighborhood, batch_dims=0)
    neigh_entropy = compute_shannon_entropy(cluster_z)
    return neigh_entropy


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
        log_dir: str = ".",
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
        self.log_dir = log_dir

    def _interpolate(self, iter1: int, iter2: int, rate1: float, rate2: float) -> float:
        return (rate2 - rate1) * (self.iteration - iter1) / (iter2 - iter1) + rate1

    # noinspection PyMethodOverriding,PyTypeChecker
    def on_batch_begin(self, batch: int, logs):
        """Defines computations to perform for each batch"""

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

    def on_epoch_end(self, epoch, logs=None):
        """Logs the learning rate to tensorboard"""

        writer = tf.summary.create_file_writer(self.log_dir)

        with writer.as_default():
            tf.summary.scalar(
                "learning_rate",
                data=self.model.optimizer.lr,
                step=epoch,
            )


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

    def __init__(self, iters, warm_up_iters, annealing_mode, *args, **kwargs):
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        self.is_placeholder = True
        self._iters = iters
        self._warm_up_iters = warm_up_iters
        self._annealing_mode = annealing_mode

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"is_placeholder": self.is_placeholder})
        config.update({"_iters": self._iters})
        config.update({"_warm_up_iters": self._warm_up_iters})
        config.update({"_annealing_mode": self._annealing_mode})
        return config

    def call(self, distribution_a):
        """Updates Layer's call method"""

        # Define and update KL weight for warmup
        if self._warm_up_iters > 0:
            if self._annealing_mode in ["linear", "sigmoid"]:
                kl_weight = tf.cast(
                    K.min([self._iters / self._warm_up_iters, 1.0]), tf.float32
                )
                if self._annealing_mode == "sigmoid":
                    kl_weight = tf.math.sigmoid(
                        (2 * kl_weight - 1) / (kl_weight - kl_weight ** 2)
                    )
            else:
                raise NotImplementedError(
                    "annealing_mode must be one of 'linear' and 'sigmoid'"
                )
        else:
            kl_weight = tf.cast(1.0, tf.float32)

        kl_batch = kl_weight * self._regularizer(distribution_a)

        self.add_loss(kl_batch, inputs=[distribution_a])
        self.add_metric(
            kl_batch,
            aggregation="mean",
            name="kl_divergence",
        )
        # noinspection PyProtectedMember
        self.add_metric(kl_weight, aggregation="mean", name="kl_rate")

        return distribution_a


class MMDiscrepancyLayer(Layer):
    """
    Identity transform layer that adds MM Discrepancy
    to the final model loss.
    """

    def __init__(
        self, batch_size, prior, iters, warm_up_iters, annealing_mode, *args, **kwargs
    ):
        super(MMDiscrepancyLayer, self).__init__(*args, **kwargs)
        self.is_placeholder = True
        self.batch_size = batch_size
        self.prior = prior
        self._iters = iters
        self._warm_up_iters = warm_up_iters
        self._annealing_mode = annealing_mode

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"_iters": self._iters})
        config.update({"_warmup_iters": self._warm_up_iters})
        config.update({"prior": self.prior})
        config.update({"_annealing_mode": self._annealing_mode})
        return config

    def call(self, z, **kwargs):
        """Updates Layer's call method"""

        true_samples = self.prior.sample(self.batch_size)

        # Define and update MMD weight for warmup
        if self._warm_up_iters > 0:
            if self._annealing_mode in ["linear", "sigmoid"]:
                mmd_weight = tf.cast(
                    K.min([self._iters / self._warm_up_iters, 1.0]), tf.float32
                )
                if self._annealing_mode == "sigmoid":
                    mmd_weight = tf.math.sigmoid(
                        (2 * mmd_weight - 1) / (mmd_weight - mmd_weight ** 2)
                    )
            else:
                raise NotImplementedError(
                    "annealing_mode must be one of 'linear' and 'sigmoid'"
                )
        else:
            mmd_weight = tf.cast(1.0, tf.float32)

        mmd_batch = mmd_weight * compute_mmd((true_samples, z))

        self.add_loss(K.mean(mmd_batch), inputs=z)
        self.add_metric(mmd_batch, aggregation="mean", name="mmd")
        self.add_metric(mmd_weight, aggregation="mean", name="mmd_rate")

        return z


class ClusterOverlap(Layer):
    """
    Identity layer that measures the overlap between the components of the latent Gaussian Mixture
    using the the entropy of the nearest neighbourhood. If self.loss_weight > 0, it adds a regularization
    penalty to the loss function
    """

    def __init__(
        self,
        batch_size: int,
        encoding_dim: int,
        k: int = 25,
        loss_weight: float = 0.0,
        *args,
        **kwargs
    ):
        self.batch_size = batch_size
        self.enc = encoding_dim
        self.k = k
        self.loss_weight = loss_weight
        self.min_confidence = 0.25
        super(ClusterOverlap, self).__init__(*args, **kwargs)

    def get_config(self):  # pragma: no cover
        """Updates Constraint metadata"""

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"enc": self.enc})
        config.update({"k": self.k})
        config.update({"loss_weight": self.loss_weight})
        config.update({"min_confidence": self.min_confidence})
        return config

    def call(self, inputs, training=False, **kwargs):
        """Updates Layer's call method"""

        encodings, categorical = inputs[0], inputs[1]

        if training:

            hard_groups = tf.math.argmax(categorical, axis=1)
            max_groups = tf.reduce_max(categorical, axis=1)

            get_local_neighbourhood_entropy = partial(
                get_neighbourhood_entropy,
                tensor=encodings,
                clusters=hard_groups,
                k=self.k,
            )

            purity_vector = tf.map_fn(
                get_local_neighbourhood_entropy,
                tf.constant(list(range(self.batch_size))),
                dtype=tf.dtypes.float32,
            )

            ### CANDIDATE FOR REMOVAL. EXPLORE HOW USEFUL THIS REALLY IS ###
            neighbourhood_entropy = purity_vector * max_groups

            number_of_clusters = tf.cast(
                tf.shape(
                    tf.unique(
                        tf.reshape(
                            tf.gather(
                                tf.cast(hard_groups, tf.dtypes.float32),
                                tf.where(max_groups >= self.min_confidence),
                                batch_dims=0,
                            ),
                            [-1],
                        ),
                    )[0],
                )[0],
                tf.dtypes.float32,
            )

            self.add_metric(
                number_of_clusters,
                name="number_of_populated_clusters",
            )

            self.add_metric(
                max_groups,
                aggregation="mean",
                name="average_confidence_in_selected_cluster",
            )

            self.add_metric(
                neighbourhood_entropy,
                aggregation="mean",
                name="average_neighbourhood_entropy",
            )

            if self.loss_weight:
                # minimize local entropy
                self.add_loss(self.loss_weight * tf.reduce_mean(neighbourhood_entropy))
                # maximize number of clusters
                # self.add_loss(-self.loss_weight * tf.reduce_mean(number_of_clusters))

        return encodings

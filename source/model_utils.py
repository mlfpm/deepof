# @author lucasmiranda42

from itertools import combinations
from tensorflow.keras import backend as K
from sklearn.metrics import silhouette_score
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

# Helper functions
@tf.function
def far_away_uniform_initialiser(shape, minval=0, maxval=15, iters=100000):
    """
    Returns a uniformly initialised matrix in which the columns are as far as possible
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


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(
        -tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32)
    )


@tf.function
def compute_mmd(tensors):

    x = tensors[0]
    y = tensors[1]

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )


# Custom auxiliary classes
class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        iterations,
        max_rate,
        start_rate=None,
        last_iterations=None,
        last_rate=None,
    ):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return (rate2 - rate1) * (self.iteration - iter1) / (iter2 - iter1) + rate1

    def on_batch_begin(self, batch, logs):
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


class UncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {"encoding_dim": self.encoding_dim, "weightage": self.weightage,}
        )
        return config

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(
            x_centered.get_shape()[0], tf.float32
        )

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self):
        if self.encoding_dim <= 1:
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
    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


class KLDivergenceLayer(tfpl.KLDivergenceAddLoss):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"is_placeholder": self.is_placeholder,}
        )
        return config

    def call(self, distribution_a):
        kl_batch = self._regularizer(distribution_a)
        self.add_loss(kl_batch, inputs=[distribution_a])
        self.add_metric(
            kl_batch, aggregation="mean", name="kl_divergence",
        )
        self.add_metric(self._regularizer._weight, aggregation="mean", name="kl_rate")

        return distribution_a


class DenseTranspose(Layer):
    def __init__(self, dense, output_dim, activation=None, **kwargs):
        self.dense = dense
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "dense": self.dense,
                "output_dim": self.output_dim,
                "activation": self.activation,
            }
        )
        return config

    def build(self, batch_input_shape):
        self.biases = self.add_weight(
            name="bias", shape=[self.dense.input_shape[-1]], initializer="zeros"
        )
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


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

    def get_config(self):
        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"beta": self.beta})
        config.update({"prior": self.prior})
        return config

    def call(self, z, **kwargs):
        true_samples = self.prior.sample(self.batch_size)
        mmd_batch = self.beta * compute_mmd([true_samples, z])
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({"lat_dims": self.lat_dims})
        config.update({"n_components": self.n_components})
        config.update({"loss": self.loss})
        config.update({"samples": self.samples})
        return config

    def call(self, target, loss=False):

        dists = []
        for k in range(self.n_components):
            locs = (target[..., : self.lat_dims, k],)
            scales = tf.keras.activations.softplus(target[..., self.lat_dims :, k])

            dists.append(
                tfd.BatchReshape(tfd.MultivariateNormalDiag(locs, scales), [-1])
            )

        dists = [tf.transpose(gauss.sample(self.samples), [1, 0, 2]) for gauss in dists]

        ### MMD-based overlap ###
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

    def call(self, z, z_gauss, z_cat, **kwargs):

        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(
            tf.math.zero_fraction(z_gauss), aggregation="mean", name="dead_neurons"
        )

        return z


class Entropy_regulariser(Layer):
    """
    Identity layer that adds cluster weight entropy to the loss function
    """

    def __init__(self, weight=1.0, *args, **kwargs):
        self.weight = weight
        super(Entropy_regulariser, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"weight": self.weight})

    def call(self, z, **kwargs):

        # axis=1 increases the entropy of a cluster across instances
        # axis=0 increases the entropy of the assignment for a given instance
        entropy = K.sum(tf.multiply(z + 1e-5, tf.math.log(z) + 1e-5), axis=1)

        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(entropy, aggregation="mean", name="-weight_entropy")

        self.add_loss(self.weight * K.sum(entropy), inputs=[z])

        return z

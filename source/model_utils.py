# @author lucasmiranda42

from itertools import combinations
from keras import backend as K
from sklearn.metrics import silhouette_score
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

# Helper functions
def far_away_uniform_initialiser(shape, minval=0, maxval=15, iters=10000000):
    """
    Returns a uniformly initialised matrix in which the columns are as far as possible
    """
    init_dist = 0
    for i in range(iters):
        temp = np.random.uniform(minval, maxval, shape)
        dist = np.abs(np.linalg.norm(np.diff(temp)))

        if dist > init_dist:
            init_dist = dist
            init = temp

    return init.astype(np.float32)


def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
    tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
    return K.exp(
        -tf.reduce_mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32)
    )


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


# Custom layers for efficiency/losses
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
    def uncorrelated_feature(self, x):
        if self.encoding_dim <= 1:
            return 0.0
        else:
            output = K.sum(
                K.square(
                    self.covariance
                    - tf.math.multiply(self.covariance, K.eye(self.encoding_dim))
                )
            )
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)


class KLDivergenceLayer(tfpl.KLDivergenceAddLoss):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, distribution_a):
        kl_batch = self._regularizer(distribution_a)
        self.add_loss(kl_batch, inputs=[distribution_a])
        self.add_metric(
            kl_batch, aggregation="mean", name="kl_divergence",
        )
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


class Latent_space_control(Layer):
    """
    Identity layer that adds latent space and clustering stats
    to the metrics compiled by the model
    """

    def __init__(self, silhouette=False, loss=False, *args, **kwargs):
        self.loss = loss
        self.silhouette = silhouette
        super(Latent_space_control, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"loss": self.loss})
        config.update({"silhouette": self.silhouette})

    def call(self, z, z_gauss, z_cat, **kwargs):

        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(
            tf.math.zero_fraction(z_gauss), aggregation="mean", name="dead_neurons"
        )

        # Adds Silhouette score controlling overlap between clusters
        if self.silhouette:
            hard_labels = tf.math.argmax(z_cat, axis=1)
            silhouette = tf.numpy_function(
                silhouette_score, [z, hard_labels], tf.float32
            )
            self.add_metric(silhouette, aggregation="mean", name="silhouette")

            if self.loss:
                self.add_loss(-K.mean(silhouette), inputs=[z, hard_labels])

        return z


class Entropy_regulariser(Layer):
    """
    Identity layer that adds cluster weight entropy to the loss function
    """

    def __init__(self, weight=1., *args, **kwargs):
        self.weight = weight
        super(Entropy_regulariser, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"weight": self.weight})

    def call(self, z, **kwargs):

        entropy = K.sum(
            tf.multiply(z, tf.where(~tf.math.is_inf(K.log(z)), K.log(z), 0)), axis=0
        )

        # Adds metric that monitors dead neurons in the latent space
        self.add_metric(entropy, aggregation="mean", name="-weight_entropy")

        #self.add_loss(self.weight * K.sum(entropy), inputs=[z])

        return z

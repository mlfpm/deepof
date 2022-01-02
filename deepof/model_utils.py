# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions, layers, regularizers, and general utilities for the unsupervised models within the deepof package.
See deepof.models for details on the currently supported architectures.

"""

from functools import partial
from typing import Any, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_uniform, random_uniform
from tensorflow.keras.layers import Layer, Input, BatchNormalization
from tensorflow.keras.layers import Bidirectional, Dense, Dropout
from tensorflow.keras.layers import GRU, RepeatVector, Reshape
from tensorflow.keras.models import Model

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


# Helper functions and classes
@tf.function
def log_loss(x_true, p_x_q_given_z):
    """

    Computes the negative log likelihood of the data given
    the output distribution

    Args:
        x_true: the true input
        p_x_q_given_z: reconstruction using the output distribution

    Returns:
        the negative log likelihood

    """

    return -tf.reduce_sum(p_x_q_given_z.log_prob(x_true))


@tf.function
def compute_shannon_entropy(tensor):
    """

    Computes Shannon entropy for a given tensor

    Args:
        tensor (tf.Tensor): tensor to compute the entropy for

    Returns:
        tf.Tensor: entropy of the tensor

    """

    tensor = tf.cast(tensor, tf.dtypes.int32)
    bins = (
        tf.math.bincount(tensor, dtype=tf.dtypes.float32)
        / tf.cast(tf.shape(tensor), tf.dtypes.float32)[0]
    )
    return -tf.reduce_sum(bins * tf.math.log(bins + 1e-5))


@tf.function
def get_k_nearest_neighbors(tensor, k, index):
    """

    Retrieve indices of the k nearest neighbors in tensor to the vector with the specified index

    Args:
        tensor (tf.Tensor): tensor to compute the k nearest neighbors for
        k (int): number of nearest neighbors to retrieve
        index (int): index of the vector to compute the k nearest neighbors for

    Returns:
        tf.Tensor: indices of the k nearest neighbors

    """
    query = tf.gather(tensor, index, batch_dims=0)
    distances = tf.norm(tensor - query, axis=1)
    max_distance = tf.sort(distances)[k]
    neighbourhood_mask = distances < max_distance
    return tf.squeeze(tf.where(neighbourhood_mask))


@tf.function
def get_neighbourhood_entropy(index, tensor, clusters, k):
    """

    Computes the neighbourhood entropy for a given vector in a tensor.

    Args:
        index (int): index of the vector to compute the neighbourhood entropy for
        tensor (tf.Tensor): tensor to compute the neighbourhood entropy for
        clusters (tf.Tensor): tensor containing the cluster labels for each vector in the tensor
        k (int): number of nearest neighbours to consider

    Returns:
        tf.Tensor: neighbourhood entropy of the vector with the specified index

    """
    neighborhood = get_k_nearest_neighbors(tensor, k, index)
    cluster_z = tf.gather(clusters, neighborhood, batch_dims=0)
    neigh_entropy = compute_shannon_entropy(cluster_z)
    return neigh_entropy


class exponential_learning_rate(tf.keras.callbacks.Callback):
    """

    Simple class that allows to grow learning rate exponentially during training

    """

    def __init__(self, factor: float):
        """

        Initializes the exponential learning rate callback

        Args:
            factor(float): factor by which to multiply the learning rate

        """
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    # noinspection PyMethodOverriding
    def on_batch_end(self, batch: int, logs: dict):
        """

        This callback acts after processing each batch

        Args:
            batch (int): current batch number
            logs (dict): dictionary containing the loss for the current batch

        """

        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_learning_rate(
    model, X, y, epochs=1, batch_size=32, min_rate=10 ** -5, max_rate=10
):
    """

    Trains the provided model for an epoch with an exponentially increasing learning rate

    Args:
        model (tf.keras.Model): model to train
        X (tf.Tensor): tensor containing the input data
        y (tf.Tensor): tensor containing the target data
        epochs (int): number of epochs to train the model for
        batch_size (int): batch size to use for training
        min_rate (float): minimum learning rate to consider
        max_rate (float): maximum learning rate to consider

    Returns:
        float: learning rate that resulted in the lowest loss

    """

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
    """

    Plots learing rate versus the loss function of the model

    Args:
        rates (np.ndarray): array containing the learning rates to plot in the x-axis
        losses (np.ndarray): array containing the losses to plot in the y-axis

    """

    plt.plot(rates, losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


def compute_kernel(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """

    Computes the MMD between the two specified vectors using a gaussian kernel.

    Args:
        x (tf.Tensor): left tensor
        y (tf.Tensor): right tensor

    Returns:
        kernel (tf.Tensor): returns the result of applying the kernel, for
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

    Args:
        tensors (tuple): tuple containing two tf.Tensor objects

    Returns:
        mmd (tf.Tensor): returns the maximum mean discrepancy for each
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


class GaussianMixtureLatent(tf.keras.models.Model):
    """

    Gaussian Mixture probabilistic latent space model. Used to represent the embedding of motion tracking data in a
    mixture of Gaussians with a specified number of components, with means, covariances and weights specified by
    neural network layers.

    """

    def __init__(
        self,
        input_shape: tuple,
        n_components: int,
        latent_dim: int,
        batch_size: int,
        loss: str = "ELBO",
        kl_warmup: int = 10,
        kl_annealing_mode: str = "linear",
        mc_kl: int = 10,
        mmd_warmup: int = 10,
        mmd_annealing_mode: str = "linear",
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
        overlap_loss: float = 0.0,
        reg_cat_clusters: bool = False,
        reg_cluster_variance: bool = False,
        **kwargs,
    ):
        """

        Initializes the Gaussian Mixture Latent layer.

        Args:
            input_shape (tuple): shape of the input data
            n_components (int): number of components in the Gaussian mixture.
            latent_dim (int): dimensionality of the latent space.
            batch_size (int): batch size for training.
            loss (str): loss function to use for training. Must be one of "ELBO", "MMD", or "ELBO+MMD".
            kl_warmup (int): number of epochs to warm up the KL divergence.
            kl_annealing_mode (str): mode to use for annealing the KL divergence. Must be one of "linear" and "sigmoid".
            mc_kl (int): number of Monte Carlo samples to use for computing the KL divergence.
            mmd_warmup (int): number of epochs to warm up the MMD.
            mmd_annealing_mode (str): mode to use for annealing the MMD. Must be one of "linear" and "sigmoid".
            optimizer (tf.keras.optimizers.Optimizer): optimizer to use for training. The layer needs access to it in
            order to compute the KL and MMD annealing weights.
            overlap_loss (float): weight of the overlap loss as described in deepof.mode_utils.ClusterOverlap
            reg_cat_clusters (bool): whether to use the penalize uneven cluster membership in the latent space.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            **kwargs: keyword arguments passed to the parent class

        """

        super().__init__(**kwargs)
        self.seq_shape = input_shape
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loss = loss
        self.kl_warmup = kl_warmup
        self.kl_annealing_mode = kl_annealing_mode
        self.mc_kl = mc_kl
        self.mmd_warmup = mmd_warmup
        self.mmd_annealing_mode = mmd_annealing_mode
        self.optimizer = optimizer
        self.overlap_loss = overlap_loss
        self.reg_cat_clusters = reg_cat_clusters
        self.reg_cluster_variance = reg_cluster_variance

        # Initialize layers
        self.z_cat = Dense(
            self.n_components,
            name="cluster_assignment",
            activation="softmax",
            activity_regularizer=(
                tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                if self.reg_cat_clusters
                else None
            ),
        )
        self.z_gauss_mean = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim * self.n_components)
            // 2,
            name="cluster_means",
            activation=None,
            activity_regularizer=(tf.keras.regularizers.l1(10e-5)),
            kernel_initializer=he_uniform(),
        )
        self.z_gauss_var = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim * self.n_components)
            // 2,
            name="cluster_variances",
            activation=None,
            kernel_regularizer=(
                tf.keras.regularizers.l2(0.01) if self.reg_cluster_variance else None
            ),
            activity_regularizer=MeanVarianceRegularizer(0.05),
            kernel_initializer=random_uniform(),
        )
        self.latent_distribution = tfpl.DistributionLambda(
            make_distribution_fn=lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(
                    probs=gauss[0],
                ),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., : self.latent_dim, k],
                            scale=1e-3
                            + tf.math.exp(gauss[1][..., self.latent_dim :, k]),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.n_components)
                ],
            ),
            convert_to_tensor_fn="sample",
            name="encoding_distribution",
        )

        # Initialize the Gaussian Mixture prior with the specified number of components
        self.prior = tfd.MixtureSameFamily(
            mixture_distribution=tfd.categorical.Categorical(
                probs=tf.ones(self.n_components) / self.n_components
            ),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.random_uniform_initializer()(
                    [self.n_components, self.latent_dim],
                ),
                scale_diag=tfb.Exp()(
                    tf.ones([self.n_components, self.latent_dim]) / self.n_components
                ),
            ),
        )

    def call(self, inputs):
        """

        Computes the output of the layer.

        """

        z_cat = self.z_cat(inputs)
        z_gauss_mean = self.z_gauss_mean(inputs)
        z_gauss_var = self.z_gauss_var(inputs)
        z_gauss = tf.keras.layers.concatenate([z_gauss_mean, z_gauss_var], axis=1)
        z_gauss = Reshape([2 * self.latent_dim, self.n_components])(z_gauss)

        z = self.latent_distribution([z_cat, z_gauss])

        # Define and control custom loss functions
        if "ELBO" in self.loss:
            kl_warm_up_iters = tf.cast(
                self.kl_warmup * (self.seq_shape[0] // self.batch_size + 1),
                tf.int64,
            )

            # noinspection PyCallingNonCallable
            z = KLDivergenceLayer(
                distribution_b=self.prior,
                test_points_fn=lambda q: q.sample(self.mc_kl),
                test_points_reduce_axis=0,
                iters=self.optimizer.iterations,
                warm_up_iters=kl_warm_up_iters,
                annealing_mode=self.kl_annealing_mode,
            )(z)

        if "MMD" in self.loss:
            mmd_warm_up_iters = tf.cast(
                self.mmd_warmup * (self.seq_shape[0] // self.batch_size + 1),
                tf.int64,
            )

            z = MMDiscrepancyLayer(
                batch_size=self.batch_size,
                prior=self.prior,
                iters=self.optimizer.iterations,
                warm_up_iters=mmd_warm_up_iters,
                annealing_mode=self.mmd_annealing_mode,
            )(z)

        # Tracks clustering metrics and adds a KNN regularizer if self.overlap_loss != 0
        if self.n_components > 1:
            z = ClusterOverlap(
                batch_size=self.batch_size,
                encoding_dim=self.latent_dim,
                k=self.n_components,
                loss_weight=self.overlap_loss,
            )([z, z_cat])

        return z, z_cat

    @property
    def model(self):
        x = Input(self.seq_shape[1:])
        return Model(inputs=x, outputs=self.call(x), name="gaussian_mixture_latent")


class VectorQuantizer(tf.keras.layers.Layer):
    """

    Vector quantizer layer, which quantizes the input vectors into a fixed number of clusters using L2 norm. Based on
    https://arxiv.org/pdf/1509.03700.pdf. Implementation based on https://keras.io/examples/generative/vq_vae/.

    """

    def __init__(self, n_components, embedding_dim, beta, **kwargs):
        """

        Initializes the VQ layer.

        Args:
            n_components (int): number of embeddings to use
            embedding_dim (int): dimensionality of the embeddings
            beta (float): beta value for the loss function
            **kwargs: additional arguments for the parent class

        """

        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta

        # Initialize embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.n_components), dtype="float32"
            ),
            trainable=True,
            name="vqvae_codebook",
        )
        self.embedding_scales = tf.Variable(
            initial_value=tf.ones([self.n_components, self.embedding_dim])
            / self.n_components,
            trainable=False,
            name="codebook_posterior_scale",
        )

    def call(self, x):
        """

        Computes the VQ layer.

        Args:
            x (tf.Tensor): input tensor

        Returns:
                x (tf.Tensor): output tensor

        """

        # Compute input shape and flatten, keeping the embedding dimension intact
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantize input using the codebook
        encoding_indices = tf.cast(self.get_code_indices(flattened), tf.int32)
        encodings = tf.one_hot(encoding_indices, self.n_components)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Update posterior variance
        self.update_posterior_variances()

        # Compute vector quantization loss, and add it to the layer
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator (copy gradients through the undiferentiable layer)
        quantized = x + tf.stop_gradient(quantized - x)

        return quantized

    def get_code_indices(self, flattened_inputs):
        """

        Getter for the code indices at any given time.

        Args:
            flattened_inputs (tf.Tensor): flattened input tensor (encoder output)

        Returns:
            encoding_indices (tf.Tensor): code indices tensor with cluster assignments.

        """
        # Compute L2-norm distance between inputs and codes at a given time
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Return index of the closest code
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def update_posterior_variances(self):
        """

        Updates the posterior variances of the codebook (not used while training, only later as a way
        of sampling the latent space.

        """
        # Compute L2-norm distance amongst codes, to estimate stdev for each cluster
        code_similarity = tf.matmul(tf.transpose(self.embeddings), self.embeddings)
        code_distances = (
            tf.reduce_sum(tf.transpose(self.embeddings) ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * code_similarity
        )

        code_scales = []
        # Compute the standard deviation of each cluster as their distance to
        # their closest embedding / 1.96 (leaver 0.05 probability of overlap assumin
        # an isotropic Gaussian distribution)
        for code in range(self.n_components):
            code_scale = code_distances[code][
                tf.argsort(code_distances[code], axis=0, stable=True)[1]
            ]
            code_scales.append(
                tf.ones([1, self.embedding_dim]) * code_scale / (self.embedding_dim / 2)
            )

        self.embedding_scales.assign(tf.concat(code_scales, axis=0))


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
        """

        Initializes the scheduler.

        Args:
            iterations (int): number of iterations to train for
            max_rate (float): maximum learning rate
            start_rate (float): starting learning rate
            last_iterations (int): number of iterations to train for at the last rate
            last_rate (float): learning rate at the last iteration
            log_dir (str): directory to save the learning rate to

        """
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
    def on_batch_begin(self, batch: int, logs: dict):
        """

        Defines computations to perform for each batch

        Args:
            batch (int): current batch number
            logs (dict): dictionary of logs

        """

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

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """

        Logs the learning rate to tensorboard

        Args:
           epoch (int): current epoch number
           logs (dict): dictionary of logs

        """

        writer = tf.summary.create_file_writer(self.log_dir)

        with writer.as_default():
            tf.summary.scalar(
                "learning_rate",
                data=self.model.optimizer.lr,
                step=epoch,
            )


# Custom Layers
class MCDropout(tf.keras.layers.Dropout):
    """

    Equivalent to tf.keras.layers.Dropout, but with training mode enabled at prediction time.
    Useful for Montecarlo predictions

    """

    def call(self, inputs, **kwargs):
        """

        Overrides the call method of the subclassed function

        """

        return super().call(inputs, training=True)


class DenseTranspose(Layer):
    """

    Mirrors a tf.keras.layers.Dense instance with transposed weights.
    Useful for decoder layers in autoencoders, to force structure and
    decrease the effective number of parameters to train

    """

    def __init__(self, dense, output_dim, activation=None, **kwargs):
        self.dense = dense
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

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
        """

        Updates Layer's build method

        """

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
        """

        Initializes the KL Divergence layer

        Args:
            iters (int): number of training iterations taken so far
            warm_up_iters (int): maximum number of training iterations for warmup
            annealing_mode (str): mode of annealing, either 'linear' or 'sigmoid'
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """

        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        self.is_placeholder = True
        self._iters = iters
        self._warm_up_iters = warm_up_iters
        self._annealing_mode = annealing_mode

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

        config = super().get_config().copy()
        config.update({"is_placeholder": self.is_placeholder})
        config.update({"_iters": self._iters})
        config.update({"_warm_up_iters": self._warm_up_iters})
        config.update({"_annealing_mode": self._annealing_mode})
        return config

    def call(self, distribution_a):
        """

        Updates Layer's call method

        """

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
        self,
        batch_size: int,
        prior: tfd.Distribution,
        iters,
        warm_up_iters,
        annealing_mode,
        *args,
        **kwargs,
    ):
        """

        Initializes the MMDiscrepancy layer

        Args:
            batch_size (int): batch size of the model
            prior (tfd.Distribution): prior distribution of the model
            iters (int): number of training iterations taken so far
            warm_up_iters (int): maximum number of training iterations for warmup
            annealing_mode (str): mode of annealing, either 'linear' or 'sigmoid'
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """

        super(MMDiscrepancyLayer, self).__init__(*args, **kwargs)
        self.is_placeholder = True
        self.batch_size = batch_size
        self.prior = prior
        self._iters = iters
        self._warm_up_iters = warm_up_iters
        self._annealing_mode = annealing_mode

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"_iters": self._iters})
        config.update({"_warmup_iters": self._warm_up_iters})
        config.update({"prior": self.prior})
        config.update({"_annealing_mode": self._annealing_mode})
        return config

    def call(self, z, **kwargs):
        """

        Updates Layer's call method

        """

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
        **kwargs,
    ):
        """

        Initializes the ClusterOverlap layer

        Args:
            batch_size (int): batch size of the model
            encoding_dim (int): dimension of the latent Gaussian Mixture
            k (int): number of nearest components of the latent Gaussian Mixture to consider
            loss_weight (float): weight of the regularization penalty applied to the local entropy of each
            training instance
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """
        super(ClusterOverlap, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.enc = encoding_dim
        self.k = k
        self.loss_weight = loss_weight
        self.min_confidence = 0.25

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"enc": self.enc})
        config.update({"k": self.k})
        config.update({"loss_weight": self.loss_weight})
        config.update({"min_confidence": self.min_confidence})
        return config

    def call(self, inputs, **kwargs):
        """

        Updates Layer's call method

        """

        encodings, categorical = inputs[0], inputs[1]

        hard_groups = tf.math.argmax(categorical, axis=1)
        max_groups = tf.reduce_max(categorical, axis=1)

        get_local_neighbourhood_entropy = partial(
            get_neighbourhood_entropy,
            tensor=encodings,
            clusters=hard_groups,
            k=self.k,
        )

        neighbourhood_entropy = tf.map_fn(
            get_local_neighbourhood_entropy,
            tf.constant(list(range(self.batch_size))),
            dtype=tf.dtypes.float32,
        )

        n_components = tf.cast(
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
            n_components,
            name="number_of_populated_clusters",
        )

        self.add_metric(
            max_groups,
            aggregation="mean",
            name="confidence_in_selected_cluster",
        )

        self.add_metric(
            neighbourhood_entropy,
            aggregation="mean",
            name="local_cluster_entropy",
        )

        if self.loss_weight:
            # minimize local entropy
            self.add_loss(self.loss_weight * tf.reduce_mean(neighbourhood_entropy))
            # maximize number of clusters
            # self.add_loss(-self.loss_weight * tf.reduce_mean(n_components))

        return encodings


@tf.keras.utils.register_keras_serializable(package="Custom", name="var_distance")
class MeanVarianceRegularizer(tf.keras.regularizers.Regularizer):
    """

    Regularizer class that penalizes the variance difference across latent Gaussian Mixture components

    """

    def __init__(self, strength: float = 0.1):
        """

        Initializes MeanVarianceRegularizer

        Args:
            strength (float): strength of the regularization penalty

        """

        self.strength = strength

    def __call__(self, x):
        tensor_mean = tf.reduce_mean(x, axis=1)
        dist_to_mean = tf.math.reduce_euclidean_norm(tf.transpose(x) - tensor_mean)
        return self.strength * dist_to_mean

    def get_config(self):
        return {"strength": self.strength}

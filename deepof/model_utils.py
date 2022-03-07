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
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial.distance import cdist
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_uniform, random_normal
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Nadam

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers


# Helper functions and classes
@tf.function
def compute_shannon_entropy(tensor):  # pragma: no cover
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
def get_k_nearest_neighbors(tensor, k, index):  # pragma: no cover
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
    max_distance = tf.gather(tf.sort(distances), k)
    neighbourhood_mask = distances < max_distance
    return tf.squeeze(tf.where(neighbourhood_mask))


@tf.function
def get_neighbourhood_entropy(index, tensor, clusters, k):  # pragma: no cover
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


def compute_gram_loss(latent_means, weight=1.0, batch_size=64):  # pragma: no cover
    """

    Adds a penalty to the singular values of the Gram matrix of the latent means. It helps disentangle the latent
    space.
    Based on Variational Animal Motion Embedding (VAME) https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

    Args:
        latent_means: tensor containing the means of the latent distribution
        weight: weight of the Gram loss in the total loss function
        batch_size: batch size of the data to compute the Gram loss for.

    Returns:
        tf.Tensor: Gram loss

    """
    gram_matrix = (tf.transpose(latent_means) @ latent_means) / tf.cast(
        batch_size, tf.float32
    )
    s = tf.linalg.svd(gram_matrix, compute_uv=False)
    s = tf.sqrt(tf.maximum(s, 1e-9))
    return weight * tf.reduce_sum(s)


def far_uniform_initializer(shape: tuple, samples: int) -> tf.Tensor:
    """
    Initializes the prior latent means in a spread-out fashion,
    obtained by iteratively picking samples from a uniform distribution
    while maximizing the minimum euclidean distance between means.

    Args:
        shape: shape of the latent space.
        samples: number of initial candidates draw from the uniform distribution.

    Returns:
        tf.Tensor: the initialized latent means.

    """

    init_shape = (samples, shape[1])

    # Initialize latent mean candidates with a uniform distribution
    init_means = tf.keras.initializers.variance_scaling(
        scale=init_shape[0], distribution="uniform"
    )(init_shape)

    # Select the first random candidate as the first cluster mean
    final_samples, init_means = init_means[0][np.newaxis, :], init_means[1:]

    # Iteratively complete the mean set by adding new candidates, which maximize the minimum euclidean distance
    # with all existing means.
    for i in range(shape[0] - 1):
        max_dist = cdist(final_samples, init_means, metric="euclidean")
        max_dist = np.argmax(np.min(max_dist, axis=0))

        final_samples = np.concatenate(
            [final_samples, init_means[max_dist][np.newaxis, :]]
        )
        init_means = np.delete(init_means, max_dist, 0)

    return final_samples


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


def compute_kernel(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:  # pragma: no cover
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
def compute_mmd(tensors: Tuple[Any]) -> tf.Tensor:  # pragma: no cover
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
        latent_loss: str = "ELBO",
        kl_warmup: int = 15,
        kl_annealing_mode: str = "linear",
        mc_kl: int = 1000,
        mmd_warmup: int = 15,
        mmd_annealing_mode: str = "linear",
        n_cluster_loss: float = 0.0,
        reg_gram: float = 0.0,
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
            latent_loss (str): loss function to use for training. Must be one of "ELBO", "MMD", or "ELBO+MMD".
            kl_warmup (int): number of epochs to warm up the KL divergence.
            kl_annealing_mode (str): mode to use for annealing the KL divergence. Must be one of "linear" and "sigmoid".
            mc_kl (int): number of Monte Carlo samples to use for computing the KL divergence.
            mmd_warmup (int): number of epochs to warm up the MMD.
            mmd_annealing_mode (str): mode to use for annealing the MMD. Must be one of "linear" and "sigmoid".
            n_cluster_loss (float): weight of the clustering loss as described in deepof.mode_utils.ClusterControl
            reg_gram (float): weight of the Gram matrix regularization loss.
            reg_cat_clusters (bool): whether to use the penalize uneven cluster membership in the latent space.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            **kwargs: keyword arguments passed to the parent class

        """

        super(GaussianMixtureLatent, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.latent_loss = latent_loss
        self.kl_warmup = kl_warmup
        self.kl_annealing_mode = kl_annealing_mode
        self.mc_kl = mc_kl
        self.mmd_warmup = mmd_warmup
        self.mmd_annealing_mode = mmd_annealing_mode
        self.n_cluster_loss = n_cluster_loss
        self.reg_gram = reg_gram
        self.optimizer = Nadam(learning_rate=1e-4, clipvalue=0.75)
        self.reg_cat_clusters = reg_cat_clusters
        self.reg_cluster_variance = reg_cluster_variance

        # Initialize layers
        self.z_cat = Dense(
            self.n_components,
            name="cluster_assignment",
            activation="softmax",
            activity_regularizer=None,
        )
        self.z_gauss_mean = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim * self.n_components)
            // 2,
            name="cluster_means",
            activation="linear",
            activity_regularizer=None,
            kernel_initializer=he_uniform(),
        )
        self.z_gauss_var = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim * self.n_components)
            // 2,
            name="cluster_variances",
            activation="softplus",
            kernel_regularizer=(
                tf.keras.regularizers.l2(0.01) if self.reg_cluster_variance else None
            ),
            activity_regularizer=MeanVarianceRegularizer(0.1),
            kernel_initializer=random_normal(stddev=0.01),
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
                            scale=1e-3 + gauss[1][..., self.latent_dim :, k],
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
                loc=tf.Variable(
                    far_uniform_initializer(
                        [self.n_components, self.latent_dim],
                        samples=10000,
                    ),
                    name="prior_means",
                    trainable=False,
                ),
                scale_diag=tf.Variable(
                    tf.ones([self.n_components, self.latent_dim])
                    / tf.math.sqrt(tf.cast(self.n_components, dtype=tf.float32) / 2.0),
                    name="prior_scales",
                    trainable=False,
                ),
            ),
        )
        self.cluster_control_layer = ClusterControl(
            batch_size=self.batch_size,
            n_components=self.n_components,
            encoding_dim=self.latent_dim,
            k=self.n_components,
            loss_weight=self.n_cluster_loss,
        )

        # Initialize metric layers
        if "ELBO" in self.latent_loss:
            self.kl_warm_up_iters = tf.cast(
                self.kl_warmup * (self.seq_shape[0] // self.batch_size),
                tf.int64,
            )
            self.kl_layer = KLDivergenceLayer(
                distribution_b=self.prior,
                test_points_fn=lambda q: q.sample(self.mc_kl),
                test_points_reduce_axis=0,
                iters=self.optimizer.iterations,
                warm_up_iters=self.kl_warm_up_iters,
                annealing_mode=self.kl_annealing_mode,
            )
        if "MMD" in self.latent_loss:
            self.mmd_warm_up_iters = tf.cast(
                self.mmd_warmup * (self.seq_shape[0] // self.batch_size),
                tf.int64,
            )
            self.mmd_layer = MMDiscrepancyLayer(
                batch_size=self.batch_size,
                prior=self.prior,
                iters=self.optimizer.iterations,
                warm_up_iters=self.mmd_warm_up_iters,
                annealing_mode=self.mmd_annealing_mode,
            )

    def call(self, inputs):  # pragma: no cover
        """

        Computes the output of the layer.

        """

        z_cat = self.z_cat(inputs)
        z_gauss_mean = self.z_gauss_mean(inputs)
        z_gauss_var = self.z_gauss_var(inputs)
        z_gauss_var = tf.keras.layers.ActivityRegularization(l2=0.01)(z_gauss_var)

        z_gauss = tf.keras.layers.concatenate([z_gauss_mean, z_gauss_var], axis=1)
        z_gauss = Reshape([2 * self.latent_dim, self.n_components])(z_gauss)

        z = self.latent_distribution([z_cat, z_gauss])

        if self.reg_gram:
            gram_loss = compute_gram_loss(
                z, weight=self.reg_gram, batch_size=self.batch_size
            )
            self.add_loss(gram_loss)
            self.add_metric(gram_loss, name="gram_loss")

        # Define and control custom loss functions
        if "ELBO" in self.latent_loss:

            # Update KL weight based on the current iteration
            self.kl_layer._iters = self.optimizer.iterations

            # noinspection PyCallingNonCallable
            z = self.kl_layer(z)

        if "MMD" in self.latent_loss:

            # Update MMD weight based on the current iteration
            self.mmd_layer._iters = self.optimizer.iterations

            z = self.mmd_layer(z)

        # Tracks clustering metrics and adds a KNN regularizer if self.n_cluster_loss != 0
        if self.n_components > 1:
            z = self.cluster_control_layer([z, z_cat])

        return z, z_cat


class VectorQuantizer(tf.keras.models.Model):
    """

    Vector quantizer layer, which quantizes the input vectors into a fixed number of clusters using L2 norm. Based on
    https://arxiv.org/pdf/1509.03700.pdf. Implementation based on https://keras.io/examples/generative/vq_vae/.

    """

    def __init__(
        self, n_components, embedding_dim, beta, reg_gram: float = 0.0, **kwargs
    ):
        """

        Initializes the VQ layer.

        Args:
            n_components (int): number of embeddings to use
            embedding_dim (int): dimensionality of the embeddings
            beta (float): beta value for the loss function
            reg_gram (float): regularization parameter for the Gram matrix
            **kwargs: additional arguments for the parent class

        """

        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta
        self.reg_gram = reg_gram

        # Initialize embedding layer
        self.embedding = tf.keras.layers.Dense(
            self.embedding_dim,
            kernel_initializer="he_uniform",
        )

        # Initialize embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.n_components), dtype="float32"
            ),
            trainable=True,
            name="vqvae_codebook",
        )

    def call(self, x):  # pragma: no cover
        """

        Computes the VQ layer.

        Args:
            x (tf.Tensor): input tensor

        Returns:
                x (tf.Tensor): output tensor

        """

        # Compute input shape and flatten, keeping the embedding dimension intact
        x = self.embedding(x)
        input_shape = tf.shape(x)

        # Add a disentangling penalty to the codebook
        if self.reg_gram:
            gram_loss = compute_gram_loss(
                x, weight=self.reg_gram, batch_size=input_shape[0]
            )
            self.add_loss(gram_loss)
            self.add_metric(gram_loss, name="gram_loss")

        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantize input using the codebook
        encoding_indices = tf.cast(
            self.get_code_indices(flattened, return_soft_counts=False), tf.int32
        )
        soft_counts = self.get_code_indices(flattened, return_soft_counts=True)

        encodings = tf.one_hot(encoding_indices, self.n_components)

        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Compute vector quantization loss, and add it to the layer
        commitment_loss = self.beta * tf.reduce_sum(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_sum((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator (copy gradients through the undiferentiable layer)
        quantized = x + tf.stop_gradient(quantized - x)

        return quantized, soft_counts

    def get_code_indices(
        self, flattened_inputs, return_soft_counts=False
    ):  # pragma: no cover
        """

        Getter for the code indices at any given time.

        Args:
            input_shape (tf.Tensor): input shape
            flattened_inputs (tf.Tensor): flattened input tensor (encoder output)
            return_soft_counts (bool): whether to return soft counts based on the distance to the codes, instead of
            the code indices

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

        if return_soft_counts:
            # Compute soft counts based on the distance to the codes
            similarity = tf.reshape(1 / distances, [-1, self.n_components])
            soft_counts = tf.nn.softmax(similarity, axis=1)
            return soft_counts

        # Return index of the closest code
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


# Custom Layers
class MCDropout(tf.keras.layers.Dropout):
    """

    Equivalent to tf.keras.layers.Dropout, th training mode enabled at prediction time.
    Useful for Montecarlo predictions

    """

    def call(self, inputs):  # pragma: no cover
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

    def call(self, inputs, **kwargs):  # pragma: no cover
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
        self._iters = iters
        self._warm_up_iters = warm_up_iters
        self._annealing_mode = annealing_mode
        self._kl_weight = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="kl_weight"
        )

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

        config = super().get_config().copy()
        config.update({"_iters": self._iters})
        config.update({"_warm_up_iters": self._warm_up_iters})
        config.update({"_annealing_mode": self._annealing_mode})
        config.update({"_kl_weight": self._kl_weight})
        return config

    def call(self, distribution_a):  # pragma: no cover
        """

        Updates Layer's call method

        """

        # Define and update KL weight for warmup
        if self._warm_up_iters > 0:
            if self._annealing_mode in ["linear", "sigmoid"]:
                self._kl_weight = tf.cast(
                    K.min([self._iters / self._warm_up_iters, 1.0]), tf.float32
                )
                if self._annealing_mode == "sigmoid":
                    self._kl_weight = tf.math.sigmoid(
                        (2 * self._kl_weight - 1)
                        / (self._kl_weight - self._kl_weight ** 2)
                    )
            else:
                raise NotImplementedError(
                    "annealing_mode must be one of 'linear' and 'sigmoid'"
                )
        else:
            self._kl_weight = tf.cast(1.0, tf.float32)

        kl_batch = self._kl_weight * self._regularizer(distribution_a)

        self.add_loss(kl_batch, inputs=[distribution_a])
        self.add_metric(
            kl_batch,
            aggregation="mean",
            name="kl_divergence",
        )

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
        self._mmd_weight = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="mmd_weight"
        )

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
        config.update({"_mmd_weight": self._mmd_weight})
        return config

    def call(self, z, **kwargs):  # pragma: no cover
        """

        Updates Layer's call method

        """

        true_samples = self.prior.sample(self.batch_size)

        # Define and update MMD weight for warmup
        if self._warm_up_iters > 0:
            if self._annealing_mode in ["linear", "sigmoid"]:
                self._mmd_weight = tf.cast(
                    K.min([self._iters / self._warm_up_iters, 1.0]), tf.float32
                )
                if self._annealing_mode == "sigmoid":
                    self._mmd_weight = tf.math.sigmoid(
                        (2 * self._mmd_weight - 1)
                        / (self._mmd_weight - self._mmd_weight ** 2)
                    )
            else:
                raise NotImplementedError(
                    "annealing_mode must be one of 'linear' and 'sigmoid'"
                )
        else:
            self._mmd_weight = tf.cast(1.0, tf.float32)

        mmd_batch = self._mmd_weight * compute_mmd((true_samples, z))

        self.add_loss(K.sum(mmd_batch), inputs=z)
        self.add_metric(mmd_batch, aggregation="mean", name="mmd")

        return z


class ClusterControl(Layer):
    """

    Identity layer that evaluates different clustering metrics between the components of the latent Gaussian Mixture
    using the entropy of the nearest neighbourhood. If self.loss_weight > 0, it also adds a regularization
    penalty to the loss function which attempts to maximize the number of populated clusters during training.

    """

    def __init__(
        self,
        batch_size: int,
        n_components: int,
        encoding_dim: int,
        k: int = 15,
        loss_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        """

        Initializes the ClusterControl layer

        Args:
            batch_size (int): batch size of the model
            n_components (int): number of components in the latent Gaussian Mixture
            encoding_dim (int): dimension of the latent Gaussian Mixture
            k (int): number of nearest components of the latent Gaussian Mixture to consider
            loss_weight (float): weight of the regularization penalty applied to the local entropy of each
            training instance
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """
        super(ClusterControl, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.n_components = n_components
        self.enc = encoding_dim
        self.k = k
        self.loss_weight = loss_weight

    def get_config(self):  # pragma: no cover
        """

        Updates Constraint metadata

        """

        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"n_components": self.n_components})
        config.update({"enc": self.enc})
        config.update({"k": self.k})
        config.update({"loss_weight": self.loss_weight})
        return config

    def call(self, inputs, **kwargs):  # pragma: no cover
        """

        Updates Layer's call method

        """

        encodings, categorical = inputs[0], inputs[1]

        hard_groups = tf.math.argmax(categorical, axis=1)
        max_groups = tf.reduce_max(categorical, axis=1)

        # Reduce k if it's too big when compared to the number of instances
        if self.k >= self.batch_size // 4:
            self.k = self.batch_size // 4

        get_local_neighbourhood_entropy = partial(
            get_neighbourhood_entropy,
            tensor=encodings,
            clusters=hard_groups,
            k=self.k,
        )

        neighbourhood_entropy = tf.map_fn(
            get_local_neighbourhood_entropy,
            tf.range(tf.shape(encodings)[0]),
            dtype=tf.dtypes.float32,
        )

        n_components = tf.cast(
            tf.shape(
                tf.unique(
                    tf.reshape(
                        tf.cast(hard_groups, tf.dtypes.float32),
                        [-1],
                    ),
                )[0],
            )[0],
            tf.dtypes.float32,
        )

        # Calculate the number of elements in each cluster, by counting the number of elements in hard_groups
        # that are equal to the corresponding cluster number
        cluster_size_entropy = compute_shannon_entropy(hard_groups)

        self.add_metric(
            cluster_size_entropy,
            name="cluster_size_entropy",
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
            self.add_loss(self.loss_weight * tf.reduce_sum(-n_components))
            self.add_loss(self.loss_weight * tf.reduce_sum(-cluster_size_entropy))

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

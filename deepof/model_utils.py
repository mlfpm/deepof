# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels."""

from datetime import date, datetime
from keras_tuner import BayesianOptimization, Hyperband, Objective
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import (
    Bidirectional,
    GRU,
    LayerNormalization,
    TimeDistributed,
)
from typing import Tuple, Union, Any, List, NewType
import deepof.data
import deepof.hypermodels
import deepof.models
import deepof.post_hoc
import matplotlib.pyplot as plt
import numpy as np
import os
from spektral.layers import CensNetConv
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import tqdm

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

### CONTRASTIVE LEARNING UTILITIES
def select_contrastive_loss(
    history,
    future,
    similarity,
    loss_fn="nce",
    temperature=0.1,
    tau=0.1,
    beta=0.1,
    elimination_topk=0.1,
    attraction=False,
):  # pragma: no cover
    """Select and applies the contrastive loss function to be used in the Contrastive embedding models.

    Args:
        history: Tensor of shape (batch_size, seq_len, embedding_dim).
        future: Tensor of shape (batch_size, seq_len, embedding_dim).
        similarity: Function that computes the similarity between two tensors.
        loss_fn: String indicating the loss function to be used.
        temperature: Float indicating the temperature to be used in the specified loss function.
        tau: Float indicating the tau value to be used if DCL or hard DLC are selected.
        beta: Float indicating the beta value to be used if hard DLC is selected.
        elimination_topk: Float indicating the top-k value to be used if FC is selected.
        attraction: Boolean indicating whether to use attraction in FC.

    """
    similarity_dict = {
        "cosine": _cosine_similarity,
        "dot": _dot_similarity,
        "euclidean": _euclidean_similarity,
        "edit": _edit_similarity,
    }
    similarity = similarity_dict[similarity]

    if loss_fn == "nce":
        loss, pos, neg = nce_loss(history, future, similarity, temperature)
    elif loss_fn == "dcl":
        loss, pos, neg = dcl_loss(
            history, future, similarity, temperature, debiased=True, tau_plus=tau
        )
    elif loss_fn == "fc":
        loss, pos, neg = fc_loss(
            history,
            future,
            similarity,
            temperature,
            elimination_topk=elimination_topk,
        )
    elif loss_fn == "hard_dcl":
        loss, pos, neg = hard_loss(
            history,
            future,
            similarity,
            temperature,
            beta=beta,
            debiased=True,
            tau_plus=tau,
        )

    # noinspection PyUnboundLocalVariable
    return loss, pos, neg


def _cosine_similarity(x, y):  # pragma: no cover
    """Compute the cosine similarity between two tensors."""
    v = tf.keras.losses.CosineSimilarity(
        axis=2, reduction=tf.keras.losses.Reduction.NONE
    )(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return -v


def _dot_similarity(x, y):  # pragma: no cover
    """Compute the dot product between two tensors."""
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)

    return v


def _euclidean_similarity(x, y):  # pragma: no cover
    """Compute the euclidean distance between two tensors."""
    x1 = tf.expand_dims(x, 1)
    y1 = tf.expand_dims(y, 0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s


def _edit_similarity(x, y):  # pragma: no cover
    """Compute the edit distance between two tensors."""
    x1 = tf.expand_dims(x, 1)
    y1 = tf.expand_dims(y, 0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s


def nce_loss(history, future, similarity, temperature=0.1):  # pragma: no cover
    """Compute the NCE loss function, as described in the paper "A Simple Framework for Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2002.05709)."""
    criterion = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    all_sim = K.exp(sim / temperature)

    logits = tf.divide(K.sum(pos_sim), K.sum(all_sim, axis=1))

    lbl = np.ones(history.shape[0])

    # categorical cross entropy
    loss = criterion(y_pred=logits, y_true=lbl)

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def dcl_loss(
    history, future, similarity, temperature=0.1, debiased=True, tau_plus=0.1
):  # pragma: no cover
    """Compute the DCL loss function, as described in the paper "Debiased Contrastive Learning" (https://github.com/chingyaoc/DCL/)."""
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    neg_sim = K.exp(neg / temperature)

    # estimator g()
    if debiased:
        N = N - 1
        Ng = (-tau_plus * N * pos_sim + K.sum(neg_sim, axis=-1)) / (1 - tau_plus)
        # constraint (optional)
        Ng = tf.clip_by_value(
            Ng,
            clip_value_min=N * np.e ** (-1 / temperature),
            clip_value_max=tf.float32.max,
        )
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + Ng)))

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def fc_loss(
    history,
    future,
    similarity,
    temperature=0.1,
    elimination_topk=0.1,
):  # pragma: no cover
    """Compute the FC loss function, as described in the paper "Fully-Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2004.11362)."""
    N = history.shape[0]
    if elimination_topk > 0.5:
        elimination_topk = 0.5
    elimination_topk = np.math.ceil(elimination_topk * N)

    sim = similarity(history, future) / temperature

    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim))

    tri_mask = np.ones(N**2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])

    sorted_sim = tf.sort(neg_sim, axis=1)

    # Top-K cancellation only
    if elimination_topk == 0:
        elimination_topk = 1
    tri_mask = np.ones(N * (N - 1), dtype=np.bool).reshape(N, N - 1)
    tri_mask[:, -elimination_topk:] = False
    neg = tf.reshape(
        tf.boolean_mask(sorted_sim, tri_mask), [N, N - elimination_topk - 1]
    )
    neg_sim = K.sum(K.exp(neg), axis=1)

    # categorical cross entropy
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + neg_sim)))

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim)) * temperature
    mean_neg = K.mean(neg) * temperature
    return loss, mean_sim, mean_neg


def hard_loss(
    history, future, similarity, temperature, beta=0.0, debiased=True, tau_plus=0.1
):  # pragma: no cover
    """Compute the Hard loss function, as described in the paper "Contrastive Learning with Hard Negative Samples" (https://arxiv.org/abs/2011.03343)."""
    N = history.shape[0]

    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    neg_sim = K.exp(neg / temperature)

    reweight = (beta * neg_sim) / tf.reshape(tf.reduce_mean(neg_sim, axis=1), [-1, 1])
    if beta == 0:
        reweight = 1
    # estimator g()
    if debiased:
        N = N - 1

        Ng = (-tau_plus * N * pos_sim + tf.reduce_sum(reweight * neg_sim, axis=-1)) / (
            1 - tau_plus
        )
        # constraint (optional)
        Ng = tf.clip_by_value(
            Ng, clip_value_min=np.e ** (-1 / temperature), clip_value_max=tf.float32.max
        )
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + Ng)))
    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def compute_kmeans_loss(
    latent_means: tf.Tensor, weight: float = 1.0, batch_size: int = 64
):  # pragma: no cover
    """Add a penalty to the singular values of the Gram matrix of the latent means. It helps disentangle the latent space.

    Based on https://arxiv.org/pdf/1610.04794.pdf, and https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

    Args:
        latent_means (tf.Tensor): tensor containing the means of the latent distribution
        weight (float): weight of the Gram loss in the total loss function
        batch_size (int): batch size of the data to compute the kmeans loss for.

    Returns:
        tf.Tensor: kmeans loss

    """
    gram_matrix = (tf.transpose(latent_means) @ latent_means) / tf.cast(
        batch_size, tf.float32
    )
    s = tf.linalg.svd(gram_matrix, compute_uv=False)
    s = tf.sqrt(tf.maximum(s, 1e-9))
    return weight * tf.reduce_mean(s)


@tf.function
def get_k_nearest_neighbors(tensor, k, index):  # pragma: no cover
    """Retrieve indices of the k nearest neighbors in tensor to the vector with the specified index.

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
def compute_shannon_entropy(tensor):  # pragma: no cover
    """Compute Shannon entropy for a given tensor.

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


def plot_lr_vs_loss(rates, losses):  # pragma: no cover
    """Plot learning rate versus the loss function of the model.

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


def get_angles(pos: int, i: int, d_model: int):
    """Auxiliary function for positional encoding computation.

    Args:
        pos (int): position in the sequence.
        i (int): number of sequences.
        d_model (int): dimensionality of the embeddings.

    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_recurrent_block(
    x: tf.Tensor, latent_dim: int, gru_unroll: bool, bidirectional_merge: str
):
    """Build a recurrent embedding block, using a 1D convolution followed by two bidirectional GRU layers.

    Args:
        x (tf.Tensor): Input tensor.
        latent_dim (int): Number of dimensions of the output tensor.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to False.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".

    Returns:
        tf.keras.models.Model object with the specified architecture.

    """
    encoder = TimeDistributed(
        tf.keras.layers.Conv1D(
            filters=2 * latent_dim,
            kernel_size=5,
            strides=1,  # Increased strides yield shorter sequences
            padding="same",
            activation="relu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
    )(x)
    encoder = tf.keras.layers.Masking(mask_value=0.0)(encoder)
    encoder = TimeDistributed(
        Bidirectional(
            GRU(
                2 * latent_dim,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=gru_unroll,
                use_bias=True,
            ),
            merge_mode=bidirectional_merge,
        )
    )(encoder)
    encoder = LayerNormalization()(encoder)
    encoder = TimeDistributed(
        Bidirectional(
            GRU(
                latent_dim,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                unroll=gru_unroll,
                use_bias=True,
            ),
            merge_mode=bidirectional_merge,
        )
    )(encoder)
    encoder = LayerNormalization()(encoder)

    return tf.keras.models.Model(x, encoder)


def positional_encoding(position: int, d_model: int):
    """Compute positional encodings, as in https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

    Args:
        position (int): position in the sequence.
        d_model (int): dimensionality of the embeddings.

    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq: tf.Tensor):
    """Create a padding mask, with zeros where data is missing, and ones where data is available.

    Args:
        seq (tf.Tensor): Sequence to compute the mask on

    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return tf.cast(1 - seq[:, tf.newaxis, tf.newaxis, :], tf.float32)


def create_look_ahead_mask(size: int):
    """Create a triangular matrix containing an increasing amount of ones from left to right on each subsequent row.

    Useful for transformer decoder, which allows it to go through the data in a sequential manner, without taking
    the future into account.

    Args:
        size (int): number of time steps in the sequence

    """
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, tf.float32)


def create_masks(inp: tf.Tensor):
    """Given an input sequence, it creates all necessary masks to pass it through the transformer architecture.

    This includes encoder and decoder padding masks, and a look-ahead mask

    Args:
        inp (tf.Tensor): input sequence to create the masks for.

    """
    # Reduces the dimensionality of the mask to remove the feature dimension
    tar = inp[:, :, 0]
    inp = inp[:, :, 0]

    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def find_learning_rate(
    model, data, epochs=1, batch_size=32, min_rate=10**-8, max_rate=10**-1
):
    """Train the provided model for an epoch with an exponentially increasing learning rate.

    Args:
        model (tf.keras.Model): model to train
        data (tuple): training data
        epochs (int): number of epochs to train the model for
        batch_size (int): batch size to use for training
        min_rate (float): minimum learning rate to consider
        max_rate (float): maximum learning rate to consider

    Returns:
        float: learning rate that resulted in the lowest loss

    """
    init_weights = model.get_weights()
    iterations = len(data)
    factor = K.exp(K.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    model.fit(data, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


@tf.function
def get_hard_counts(soft_counts: tf.Tensor):
    """Compute hard counts per cluster in a differentiable way.

    Args:
        soft_counts (tf.Tensor): soft counts per cluster

    """
    max_per_row = tf.expand_dims(tf.reduce_max(soft_counts, axis=1), axis=1)

    mask = tf.cast(soft_counts == max_per_row, tf.float32)
    mask_forward = tf.multiply(soft_counts, mask)
    mask_complement = tf.multiply(1 / soft_counts, mask)

    binary_counts = tf.multiply(mask_forward, mask_complement)

    return tf.math.reduce_sum(binary_counts, axis=0) + 1


@tf.function
def cluster_frequencies_regularizer(
    soft_counts: tf.Tensor, k: int, n_samples: int = 1000
):
    """Compute the KL divergence between the cluster assignment distribution and a uniform prior across clusters.

    While this assumes an equal distribution between clusters, the prior can be tweaked to reflect domain knowledge.

    Args:
        soft_counts (tf.Tensor): soft counts per cluster
        k (int): number of clusters
        n_samples (int): number of samples to draw from the categorical distribution modeling cluster assignments.

    """
    hard_counts = get_hard_counts(soft_counts)

    dist_a = tfd.Categorical(probs=hard_counts / k)
    dist_b = tfd.Categorical(probs=tf.ones(k))

    z = dist_a.sample(n_samples)

    return tf.reduce_mean(dist_a.log_prob(z) - dist_b.log_prob(z))


def get_callbacks(
    embedding_model: str,
    encoder_type: str,
    kmeans_loss: float = 1.0,
    input_type: str = False,
    cp: bool = False,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """Generate callbacks used for model training.

    Args:
        embedding_model (str): name of the embedding model
        encoder_type (str): Architecture used for the encoder. Must be one of "recurrent", "TCN", and "transformer"
        kmeans_loss (float): Weight of the gram loss
        input_type (str): Input type to use for training
        cp (bool): Whether to use checkpointing or not
        logparam (dict): Dictionary containing the hyperparameters to log in tensorboard
        outpath (str): Path to the output directory
        run (int): Run number to use for checkpointing

    Returns:
        List[Union[Any]]: List of callbacks to be used for training

    """
    run_ID = "{}{}{}{}{}{}{}".format(
        "deepof_unsupervised_{}_{}_encodings".format(embedding_model, encoder_type),
        ("_input_type={}".format(input_type if input_type else "coords")),
        ("_kmeans_loss={}".format(kmeans_loss)),
        ("_encoding={}".format(logparam["latent_dim"]) if logparam is not None else ""),
        ("_k={}".format(logparam["n_components"]) if logparam is not None else ""),
        ("_run={}".format(run) if run else ""),
        ("_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")) if not run else ""),
    )

    log_dir = os.path.abspath(os.path.join(outpath, "fit", run_ID))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch=2
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, cooldown=5, min_lr=1e-8
    )

    callbacks = [run_ID, tensorboard_callback, reduce_lr_callback]

    if cp:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(outpath, "checkpoints", run_ID + "/cp-{epoch:04d}.ckpt"),
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )
        callbacks.append(cp_callback)

    return callbacks


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    """Custom early stopping callback. Prevents the model from stopping before warmup is over."""

    def __init__(self, start_epoch, *args, **kwargs):
        """Initialize the CustomStopper callback.

        Args:
            start_epoch: epoch from which performance will be taken into account when deciding whether to stop training.
            *args: arguments passed to the callback.
            **kwargs: keyword arguments passed to the callback.

        """
        super(CustomStopper, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def get_config(self):  # pragma: no cover
        """Update callback metadata."""
        config = super().get_config().copy()
        config.update({"start_epoch": self.start_epoch})
        return config

    def on_epoch_end(self, epoch, logs=None):
        """Check whether to stop training."""
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    """Simple class that allows to grow learning rate exponentially during training.

    Used to trigger optimal learning rate search in deepof.train_utils.find_learning_rate.

    """

    def __init__(self, factor: float):
        """Initialize the exponential learning rate callback.

        Args:
            factor(float): factor by which to multiply the learning rate

        """
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    # noinspection PyMethodOverriding
    def on_batch_end(self, batch: int, logs: dict):
        """Apply on batch end.

        Args:
            batch: batch number
            logs (dict): dictionary containing the loss for the current batch

        """
        self.rates.append(K.get_value(self.model.optimizer.lr))
        if "total_loss" in logs.keys():
            self.losses.append(logs["total_loss"])
        else:
            self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


class ProbabilisticDecoder(tf.keras.layers.Layer):
    """Map the reconstruction output of a given decoder to a multivariate normal distribution."""

    def __init__(self, input_shape, **kwargs):
        """Initialize the probabilistic decoder."""
        super().__init__(**kwargs)
        self.time_distributer = tf.keras.layers.Dense(
            tfpl.IndependentNormal.params_size(input_shape[1:]) // 2
        )
        self.probabilistic_decoding = tfpl.DistributionLambda(
            make_distribution_fn=lambda decoded: tfd.Masked(
                tfd.Independent(
                    tfd.Normal(
                        loc=decoded[0],
                        scale=tf.ones_like(decoded[0]),
                        validate_args=False,
                        allow_nan_stats=False,
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                validity_mask=decoded[1],
            ),
            convert_to_tensor_fn="mean",
        )
        self.scaled_probabilistic_decoding = tfpl.DistributionLambda(
            make_distribution_fn=lambda decoded: tfd.Masked(
                tfd.TransformedDistribution(
                    decoded[0],
                    tfb.Scale(tf.cast(tf.expand_dims(decoded[1], axis=2), tf.float32)),
                    name="vae_reconstruction",
                ),
                validity_mask=decoded[1],
            ),
            convert_to_tensor_fn="mean",
        )

    def call(self, inputs):  # pragma: no cover
        """Map the reconstruction output of a given decoder to a multivariate normal distribution.

        Args:
            inputs (tuple): tuple containing the reconstruction output and the validity mask

        Returns:
            tf.Tensor: multivariate normal distribution

        """
        hidden, validity_mask = inputs

        hidden = tf.keras.layers.TimeDistributed(self.time_distributer)(hidden)
        prob_decoded = self.probabilistic_decoding([hidden, validity_mask])
        scaled_prob_decoded = self.scaled_probabilistic_decoding(
            [prob_decoded, validity_mask]
        )

        return scaled_prob_decoded


class ClusterControl(tf.keras.layers.Layer):
    """Identity layer.

    Evaluates different clustering metrics between the components of the latent Gaussian Mixture
    using the entropy of the nearest neighbourhood. If self.loss_weight > 0, it also adds a regularization
    penalty to the loss function which attempts to maximize the number of populated clusters during training.

    """

    def __init__(
        self,
        batch_size: int,
        n_components: int,
        encoding_dim: int,
        k: int = 15,
        *args,
        **kwargs,
    ):
        """Initialize the ClusterControl layer.

        Args:
            batch_size (int): batch size of the model
            n_components (int): number of components in the latent Gaussian Mixture
            encoding_dim (int): dimension of the latent Gaussian Mixture
            k (int): number of nearest components of the latent Gaussian Mixture to consider
            loss_weight (float): weight of the regularization penalty applied to the local entropy of each training instance
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """
        super(ClusterControl, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.n_components = n_components
        self.enc = encoding_dim
        self.k = k

    def get_config(self):  # pragma: no cover
        """Update Constraint metadata."""
        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"n_components": self.n_components})
        config.update({"enc": self.enc})
        config.update({"k": self.k})
        config.update({"loss_weight": self.loss_weight})
        return config

    def call(self, inputs):  # pragma: no cover
        """Update Layer's call method."""
        encodings, categorical = inputs[0], inputs[1]

        hard_groups = tf.math.argmax(categorical, axis=1)
        max_groups = tf.reduce_max(categorical, axis=1)

        n_components = tf.cast(
            tf.shape(
                tf.unique(tf.reshape(tf.cast(hard_groups, tf.dtypes.float32), [-1]))[0]
            )[0],
            tf.dtypes.float32,
        )

        self.add_metric(n_components, name="number_of_populated_clusters")
        self.add_metric(
            max_groups, aggregation="mean", name="confidence_in_selected_cluster"
        )

        return encodings


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer encoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer."""

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """Construct the transformer encoder layer.

        Args:
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers
            dff: dimensionality of the embeddings
            rate: dropout rate

        """
        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dff, activation="relu"
                ),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(key_dim),  # (batch_size, seq_len, d_model)
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, return_scores=False):  # pragma: no cover
        """Call the transformer encoder layer."""
        attn_output, attn_scores = self.mha(
            key=x, query=x, value=x, attention_mask=mask, return_attention_scores=True
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        if return_scores:  # pragma: no cover
            return out2, attn_scores

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Transformer decoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer."""

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """Construct the transformer decoder layer.

        Args:
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers
            dff: dimensionality of the embeddings
            rate: dropout rate

        """
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(key_dim),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(
        self, x, enc_output, training, look_ahead_mask, padding_mask
    ):  # pragma: no cover
        """Call the transformer decoder layer."""
        attn1, attn_weights_block1 = self.mha1(
            key=x,
            query=x,
            value=x,
            attention_mask=look_ahead_mask,
            return_attention_scores=True,
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            key=enc_output,
            query=out1,
            value=enc_output,
            attention_mask=padding_mask,
            return_attention_scores=True,
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# noinspection PyCallingNonCallable
class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    """

    def __init__(
        self,
        num_layers,
        seq_dim,
        key_dim,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
    ):
        """Construct the transformer encoder.

        Args:
            num_layers: number of transformer layers to include.
            seq_dim: dimensionality of the sequence embeddings
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers used on the transformer encoder
            dff: dimensionality of the token embeddings
            maximum_position_encoding: maximum time series length
            rate: dropout rate

        """
        super(TransformerEncoder, self).__init__()

        self.rate = rate
        self.key_dim = key_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Conv1D(
            key_dim, kernel_size=1, activation="relu", input_shape=[seq_dim, key_dim]
        )
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.key_dim)
        self.enc_layers = [
            TransformerEncoderLayer(key_dim, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training):  # pragma: no cover
        """Call the transformer encoder."""
        # compute mask on the fly
        mask, _, _ = create_masks(x)
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


# noinspection PyCallingNonCallable
class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    """

    def __init__(
        self,
        num_layers,
        seq_dim,
        key_dim,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
    ):
        """Construct the transformer decoder.

        Args:
            num_layers: number of transformer layers to include.
            seq_dim: dimensionality of the sequence embeddings
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers used on the transformer encoder
            dff: dimensionality of the token embeddings
            maximum_position_encoding: maximum time series length
            rate: dropout rate

        """
        super(TransformerDecoder, self).__init__()

        self.rate = rate
        self.key_dim = key_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Conv1D(
            key_dim, kernel_size=1, activation="relu", input_shape=[seq_dim, key_dim]
        )
        self.pos_encoding = positional_encoding(maximum_position_encoding, key_dim)
        self.dec_layers = [
            TransformerDecoderLayer(key_dim, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(
        self, x, enc_output, training, look_ahead_mask, padding_mask
    ):  # pragma: no cover
        """Call the transformer decoder."""
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )
            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        return x, attention_weights


def log_hyperparameters():
    """Log hyperparameters in tensorboard.

    Blueprint for hyperparameter and metric logging in tensorboard during hyperparameter tuning

    Returns:
        logparams (list): List containing the hyperparameters to log in tensorboard.
        metrics (list): List containing the metrics to log in tensorboard.

    """
    logparams = [
        hp.HParam(
            "latent_dim",
            hp.Discrete([2, 4, 6, 8, 12, 16]),
            display_name="latent_dim",
            description="encoding size dimensionality",
        ),
        hp.HParam(
            "n_components",
            hp.IntInterval(min_value=1, max_value=25),
            display_name="n_components",
            description="latent component number",
        ),
        hp.HParam(
            "kmeans_weight",
            hp.RealInterval(min_value=0.0, max_value=1.0),
            display_name="kmeans_weight",
            description="weight of the kmeans loss",
        ),
    ]

    metrics = [
        hp.Metric(
            "val_number_of_populated_clusters",
            display_name="number of populated clusters",
        ),
        hp.Metric("val_reconstruction_loss", display_name="reconstruction loss"),
        hp.Metric("val_kmeans_loss", display_name="kmeans loss"),
        hp.Metric("val_vq_loss", display_name="vq loss"),
        hp.Metric("val_total_loss", display_name="total loss"),
    ]

    return logparams, metrics


def embedding_model_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    adjacency_matrix: np.ndarray,
    embedding_model: str,
    encoder_type: str,
    batch_size: int,
    latent_dim: int,
    epochs: int,
    log_history: bool,
    log_hparams: bool,
    n_components: int,
    output_path: str,
    kmeans_loss: float,
    pretrained: str,
    save_checkpoints: bool,
    save_weights: bool,
    input_type: str,
    # VaDE Model specific parameters
    kl_annealing_mode: str,
    kl_warmup: int,
    reg_cat_clusters: float,
    recluster: bool,
    # Contrastive Model specific parameters
    temperature: float,
    contrastive_similarity_function: str,
    contrastive_loss_function: str,
    beta: float,
    tau: float,
    interaction_regularization: float,
    run: int = 0,
    **kwargs,
):
    """

    Trains the specified embedding model on the preprocessed data.

    Args:
        coordinates (np.ndarray): Coordinates of the data.
        preprocessed_object (tuple): Tuple containing the preprocessed data.
        adjacency_matrix (np.ndarray): adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), VaDE, and contrastive.
        encoder_type (str): Encoder architecture to use. Must be one of "recurrent", "TCN", and "transformer".
        batch_size (int): Batch size to use for training.
        latent_dim (int): Encoding size to use for training.
        epochs (int): Number of epochs to train the autoencoder for.
        log_history (bool): Whether to log the history of the autoencoder.
        log_hparams (bool): Whether to log the hyperparameters used for training.
        n_components (int): Number of components to fit to the data.
        output_path (str): Path to the output directory.
        kmeans_loss (float): Weight of the gram loss, which adds a regularization term to VQVAE models which penalizes the correlation between the dimensions in the latent space.
        pretrained (str): Path to the pretrained weights to use for the autoencoder.
        save_checkpoints (bool): Whether to save checkpoints during training.
        save_weights (bool): Whether to save the weights of the autoencoder after training.
        input_type (str): Input type of the TableDict objects used for preprocessing. For logging purposes only.
        interaction_regularization (float): Weight of the interaction regularization term (L1 penalization to all features not related to interactions).
        run (int): Run number to use for logging.

        # VaDE Model specific parameters
        kl_annealing_mode (str): Mode to use for KL annealing. Must be one of "linear" (default), or "sigmoid".
        kl_warmup (int): Number of epochs during which KL is annealed.
        reg_cat_clusters (bool): whether to penalize uneven cluster membership in the latent space, by minimizing the KL divergence between cluster membership and a uniform categorical distribution.
        recluster (bool): Whether to recluster the data after each training using a Gaussian Mixture Model.

        # Contrastive Model specific parameters
        temperature (float): temperature parameter for the contrastive loss functions. Higher values put harsher penalties on negative pair similarity.
        contrastive_similarity_function (str): similarity function between positive and negative pairs. Must be one of 'cosine' (default), 'euclidean', 'dot', and 'edit'.
        contrastive_loss_function (str): contrastive loss function. Must be one of 'nce' (default), 'dcl', 'fc', and 'hard_dcl'. See specific documentation for details.
        beta (float): Beta (concentration) parameter for the hard_dcl contrastive loss. Higher values lead to 'harder' negative samples.
        tau (float): Tau parameter for the dcl and hard_dcl contrastive losses, indicating positive class probability.

    Returns:
        List of trained models corresponding to the selected model class. The full trained model is last.

    """
    # Select strategy based on available hardware
    if len(tf.config.list_physical_devices("GPU")) > 1:  # pragma: no cover
        strategy = tf.distribute.MirroredStrategy(
            [dev.name for dev in tf.config.list_physical_devices("GPU")]
        )
    elif len(tf.config.list_physical_devices("GPU")) == 1:
        strategy = tf.distribute.OneDeviceStrategy("gpu")
    else:
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    with tf.device("CPU"):

        # Load data
        try:
            X_train, a_train, y_train, X_val, a_val, y_val = preprocessed_object
        except ValueError:
            X_train, y_train, X_val, y_val = preprocessed_object
            a_train, a_val = np.zeros(X_train.shape), np.zeros(X_val.shape)

        # Make sure that batch_size is not larger than training set
        if batch_size > preprocessed_object[0].shape[0]:
            batch_size = preprocessed_object[0].shape[0]

        # Set options for tf.data.Datasets
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        # Defines hyperparameters to log on tensorboard (useful for keeping track of different models)
        logparam = {
            "latent_dim": latent_dim,
            "n_components": n_components,
            "kmeans_weight": kmeans_loss,
        }

        # Load callbacks
        run_ID, *cbacks = get_callbacks(
            embedding_model=embedding_model,
            encoder_type=encoder_type,
            kmeans_loss=kmeans_loss,
            input_type=input_type,
            cp=save_checkpoints,
            logparam=logparam,
            outpath=output_path,
            run=run,
        )
        if not log_history:
            cbacks = cbacks[1:]

        Xs, ys = X_train, [X_train]
        Xvals, yvals = X_val, [X_val]

        # Cast to float32
        ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
        yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

        # Convert data to tf.data.Dataset objects
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (tf.cast(Xs, tf.float32), tf.cast(a_train, tf.float32), tuple(ys))
            )
            .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
            .shuffle(buffer_size=X_train.shape[0])
            .with_options(options)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (tf.cast(Xvals, tf.float32), tf.cast(a_val, tf.float32), tuple(yvals))
            )
            .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
            .with_options(options)
            .prefetch(tf.data.AUTOTUNE)
        )

    # Build model
    with strategy.scope():

        if embedding_model == "VQVAE":
            ae_full_model = deepof.models.VQVAE(
                input_shape=X_train.shape,
                edge_feature_shape=a_train.shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=len(preprocessed_object) == 6,
                n_components=n_components,
                kmeans_loss=kmeans_loss,
                encoder_type=encoder_type,
                interaction_regularization=interaction_regularization,
            )
            ae_full_model.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=1e-4, clipvalue=0.75
            )

        elif embedding_model == "VaDE":
            ae_full_model = deepof.models.VaDE(
                input_shape=X_train.shape,
                edge_feature_shape=a_train.shape,
                adjacency_matrix=adjacency_matrix,
                batch_size=batch_size,
                latent_dim=latent_dim,
                use_gnn=len(preprocessed_object) == 6,
                kl_annealing_mode=kl_annealing_mode,
                kl_warmup_epochs=kl_warmup,
                montecarlo_kl=100,
                n_components=n_components,
                reg_cat_clusters=reg_cat_clusters,
                encoder_type=encoder_type,
                interaction_regularization=interaction_regularization,
            )

        elif embedding_model == "Contrastive":
            ae_full_model = deepof.models.Contrastive(
                input_shape=X_train.shape,
                edge_feature_shape=a_train.shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=len(preprocessed_object) == 6,
                encoder_type=encoder_type,
                temperature=temperature,
                similarity_function=contrastive_similarity_function,
                loss_function=contrastive_loss_function,
                interaction_regularization=interaction_regularization,
                beta=beta,
                tau=tau,
            )

        else:  # pragma: no cover
            raise ValueError(
                "Invalid embedding model. Select one of 'VQVAE', 'VaDE', and 'Contrastive'"
            )

    callbacks_ = cbacks + [
        CustomStopper(
            monitor="val_total_loss",
            mode="min",
            patience=15,
            restore_best_weights=False,
            start_epoch=15,
        )
    ]

    ae_full_model.compile(
        optimizer=ae_full_model.optimizer,
        run_eagerly=False,
    )

    if embedding_model == "VaDE":
        ae_full_model.pretrain(
            train_dataset,
            embed_x=Xs,
            embed_a=a_train,
            epochs=(np.minimum(10, epochs) if not pretrained else 0),
            **kwargs,
        )
        ae_full_model.optimizer._iterations.assign(0)

    ae_full_model.fit(
        x=train_dataset,
        epochs=(epochs if not pretrained else 0),
        validation_data=val_dataset,
        callbacks=callbacks_,
        **kwargs,
    )

    if embedding_model == "VaDE" and recluster == True:  # pragma: no cover
        ae_full_model.pretrain(
            train_dataset, embed_x=Xs, embed_a=a_train, epochs=0, **kwargs
        )

    if pretrained:  # pragma: no cover
        # If pretrained models are specified, load weights and return
        ae_full_model.build([X_train.shape, a_train.shape])
        ae_full_model.load_weights(pretrained)
        return ae_full_model

    if not os.path.exists(os.path.join(output_path, "trained_weights")):
        os.makedirs(os.path.join(output_path, "trained_weights"))

    if save_weights:
        ae_full_model.save_weights(
            os.path.join(
                "{}".format(output_path),
                "trained_weights",
                "{}_final_weights.h5".format(run_ID),
            )
        )

        # Logs hyperparameters to tensorboard
        if log_hparams:
            logparams, metrics = log_hyperparameters()

            tb_writer = tf.summary.create_file_writer(
                os.path.abspath(os.path.join(output_path, "hparams", run_ID))
            )
            with tb_writer.as_default():
                # Configure hyperparameter logging in tensorboard
                hp.hparams_config(hparams=logparams, metrics=metrics)
                hp.hparams(logparam)  # Log hyperparameters

                # Log metrics
                tf.summary.scalar(
                    "val_total_loss",
                    ae_full_model.history.history["val_total_loss"][-1],
                    step=0,
                )

                if embedding_model != "Contrastive":
                    tf.summary.scalar(
                        "val_reconstruction_loss",
                        ae_full_model.history.history["val_reconstruction_loss"][-1],
                        step=0,
                    )
                    tf.summary.scalar(
                        "val_number_of_populated_clusters",
                        ae_full_model.history.history[
                            "val_number_of_populated_clusters"
                        ][-1],
                        step=0,
                    )
                    tf.summary.scalar(
                        "val_kmeans_loss",
                        ae_full_model.history.history["val_kmeans_loss"][-1],
                        step=0,
                    )

                if embedding_model == "VQVAE":
                    tf.summary.scalar(
                        "val_vq_loss",
                        ae_full_model.history.history["val_vq_loss"][-1],
                        step=0,
                    )

                elif embedding_model == "VaDE":
                    tf.summary.scalar(
                        "val_kl_loss",
                        ae_full_model.history.history["val_kl_divergence"][-1],
                        step=0,
                    )

                elif embedding_model == "Contrastive":
                    tf.summary.scalar(
                        "val_total_loss",
                        ae_full_model.history.history["val_total_loss"][-1],
                        step=0,
                    )

    return ae_full_model


def embedding_per_video(
    coordinates: coordinates,
    to_preprocess: table_dict,
    model: tf.keras.models.Model,
    scale: str = "standard",
    animal_id: str = None,
    ruptures: bool = False,
    global_scaler: Any = None,
    **kwargs,
):  # pragma: no cover
    """Use a previously trained model to produce embeddings, soft_counts and breaks per experiment in table_dict format.

    Args:
        coordinates (coordinates): deepof.Coordinates object for the project at hand.
        to_preprocess (table_dict): dictionary with (merged) features to process.
        scale (str): The type of scaler to use within animals. Defaults to 'standard', but can be changed to 'minmax', 'robust', or False. Use the same that was used when training the original model.
        animal_id (str): if more than one animal is present, provide the ID(s) of the animal(s) to include.
        ruptures (bool): Whether to compute the breaks based on ruptures (with the length of all retrieved chunks per experiment) or not (an all-ones vector per experiment is returned).
        global_scaler (Any): trained global scaler produced when processing the original dataset.
        model (tf.keras.models.Model): trained deepof unsupervised model to run inference with.
        **kwargs: additional arguments to pass to coordinates.get_graph_dataset().

    Returns:
        embeddings (table_dict): embeddings per experiment.
        soft_counts (table_dict): soft_counts per experiment.
        breaks (table_dict): breaks per experiment.

    """
    embeddings = {}
    soft_counts = {}
    breaks = {}

    graph, contrastive = False, False
    try:
        if any([isinstance(i, CensNetConv) for i in model.encoder.layers[2].layers]):
            graph = True
    except AttributeError:
        if any([isinstance(i, CensNetConv) for i in model.encoder.layers]):
            graph, contrastive = True, True

    window_size = model.layers[0].input_shape[0][1]
    for key in tqdm.tqdm(to_preprocess.keys()):

        if graph:
            processed_exp, _, _, _ = coordinates.get_graph_dataset(
                animal_id=animal_id,
                precomputed_tab_dict=to_preprocess.filter_videos([key]),
                preprocess=True,
                scale=scale,
                window_size=window_size,
                window_step=1,
                shuffle=False,
                pretrained_scaler=global_scaler,
            )

        else:

            processed_exp, _ = to_preprocess.filter_videos([key]).preprocess(
                scale=scale,
                window_size=window_size,
                window_step=1,
                shuffle=False,
                pretrained_scaler=global_scaler,
            )

        embeddings[key] = model.encoder([processed_exp[0], processed_exp[1]]).numpy()
        if ruptures:
            breaks[key] = (~np.all(processed_exp[0] == 0, axis=2)).sum(axis=1)
        else:
            breaks[key] = np.ones(embeddings[key].shape[0]).astype(int)

        if not contrastive:
            soft_counts[key] = model.grouper(
                [processed_exp[0], processed_exp[1]]
            ).numpy()

    if contrastive:
        soft_counts = deepof.post_hoc.recluster(coordinates, embeddings, **kwargs)

    return (
        deepof.data.TableDict(
            embeddings,
            typ="unsupervised_embedding",
            exp_conditions=coordinates.get_exp_conditions,
        ),
        deepof.data.TableDict(
            soft_counts,
            typ="unsupervised_counts",
            exp_conditions=coordinates.get_exp_conditions,
        ),
        deepof.data.TableDict(
            breaks,
            typ="unsupervised_breaks",
            exp_conditions=coordinates.get_exp_conditions,
        ),
    )


def tune_search(
    preprocessed_object: tuple,
    adjacency_matrix: np.ndarray,
    encoding_size: int,
    embedding_model: str,
    hypertun_trials: int,
    hpt_type: str,
    k: int,
    project_name: str,
    callbacks: List,
    batch_size: int = 1024,
    n_epochs: int = 30,
    n_replicas: int = 1,
    outpath: str = "unsupervised_tuner_search",
) -> tuple:
    """Define the search space using keras-tuner and hyperband or bayesian optimization.

    Args:
        preprocessed_object (tf.data.Dataset): Dataset object for training and validation.
        adjacency_matrix (np.ndarray): Adjacency matrix for the graph.
        encoding_size (int): Size of the encoding layer.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), VaDE, and Contrastive.
        hypertun_trials (int): Number of hypertuning trials to run.
        hpt_type (str): Type of hypertuning to run. Must be one of "hyperband" or "bayesian".
        k (int): Number of clusters on the latent space.
        kmeans_loss (float): Weight of the kmeans loss, which enforces disentanglement by penalizing the correlation between dimensions in the latent space.
        project_name (str): Name of the project.
        callbacks (List): List of callbacks to use.
        batch_size (int): Batch size to use.
        n_epochs (int): Maximum number of epochs to train for.
        n_replicas (int): Number of replicas to use.
        outpath (str): Path to save the results.

    Returns:
        best_hparams (dict): Dictionary of the best hyperparameters.
        best_run (str): Name of the best run.

    """
    # Load data
    try:
        X_train, a_train, y_train, X_val, a_val, y_val = preprocessed_object
    except ValueError:
        X_train, y_train, X_val, y_val = preprocessed_object
        a_train, a_val = np.zeros(X_train.shape), np.zeros(X_val.shape)

    # Make sure that batch_size is not larger than training set
    if batch_size > preprocessed_object[0].shape[0]:
        batch_size = preprocessed_object[0].shape[0]

    # Set options for tf.data.Datasets
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    # Cast to float32
    ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
    yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(Xs, tf.float32), tf.cast(a_train, tf.float32), tuple(ys))
        )
        .batch(batch_size, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(Xvals, tf.float32), tf.cast(a_val, tf.float32), tuple(yvals))
        )
        .batch(batch_size, drop_remainder=True)
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )

    assert hpt_type in ["bayopt", "hyperband"], (
        "Invalid hyperparameter tuning framework. " "Select one of bayopt and hyperband"
    )

    if embedding_model == "VQVAE":
        hypermodel = deepof.hypermodels.VQVAE(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
            n_components=k,
        )
    elif embedding_model == "VaDE":
        hypermodel = deepof.hypermodels.VaDE(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
            n_components=k,
            batch_size=batch_size,
        )
    elif embedding_model == "Contrastive":
        hypermodel = deepof.hypermodels.Contrastive(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
        )

    tuner_objective = "val_total_loss"

    # noinspection PyUnboundLocalVariable
    hpt_params = {
        "hypermodel": hypermodel,
        "executions_per_trial": n_replicas,
        "objective": Objective(tuner_objective, direction="min"),
        "project_name": project_name,
        "tune_new_entries": True,
    }

    if hpt_type == "hyperband":
        tuner = Hyperband(
            directory=os.path.join(
                outpath, "HyperBandx_VQVAE_{}".format(str(date.today()))
            ),
            max_epochs=n_epochs,
            hyperband_iterations=hypertun_trials,
            factor=3,
            **hpt_params,
        )
    else:
        tuner = BayesianOptimization(
            directory=os.path.join(
                outpath, "BayOpt_VQVAE_{}".format(str(date.today()))
            ),
            max_trials=hypertun_trials,
            **hpt_params,
        )

    print(tuner.search_space_summary())

    # Convert data to tf.data.Dataset objects
    tuner.search(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_run = tuner.hypermodel.build(best_hparams)

    print(tuner.results_summary())

    return best_hparams, best_run

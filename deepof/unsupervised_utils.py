# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels.

"""

import json
import os
from datetime import date, datetime
from functools import partial
from typing import Tuple, Union, Any, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from keras_tuner import BayesianOptimization, Hyperband, Objective
from tensorboard.plugins.hparams import api as hp

import deepof.models
import deepof.hypermodels

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)


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
):
    """



    Args:
        history:
        future:
        similarity:
        loss_fn:
        temperature:
        tau:
        beta:
        elimination_topk:
        attraction:

    Returns:

    """

    if loss_fn == "nce":
        loss, pos, neg = nce_loss_fn(history, future, similarity, temperature)
    elif loss_fn == "dcl":
        loss, pos, neg = dcl_loss_fn(
            history, future, similarity, temperature, debiased=True, tau_plus=tau
        )
    elif loss_fn == "fc":
        loss, pos, neg = fc_loss_fn(
            history,
            future,
            similarity,
            temperature,
            elimination_topk=elimination_topk,
            attraction=attraction,
        )
    elif loss_fn == "hard_dcl":
        loss, pos, neg = hard_loss_fn(
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


def nce_loss_fn(history, future, similarity, temperature=0.1):
    criterion = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
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


def dcl_loss_fn(
    history, future, similarity, temperature=0.1, debiased=True, tau_plus=0.1
):
    # from Debiased Contrastive Learning paper: https://github.com/chingyaoc/DCL/
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
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


def fc_loss_fn(
    history, future, similarity, temperature=0.1, elimination_topk=0.1, attraction=False
):
    N = history.shape[0]
    if elimination_topk > 0.5:
        elimination_topk = 0.5
    elimination_topk = np.math.ceil(elimination_topk * N)

    sim = similarity(history, future) / temperature

    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim))

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
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


def hard_loss_fn(
    history, future, similarity, temperature, beta=0.0, debiased=True, tau_plus=0.1
):
    # from ICLR2021 paper: Contrastive LEarning with Hard Negative Samples https://www.groundai.com/project/contrastive-learning-with-hard-negative-samples
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability
    #
    # reweight = (beta * neg) / neg.mean()
    # Neg = max((-N * tau_plus * pos + reweight * neg).sum() / (1 - tau_plus), e ** (-1 / t))
    # hard_loss = -log(pos.sum() / (pos.sum() + Neg))

    N = history.shape[0]

    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
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


def load_treatments(train_path):
    """

    Loads a dictionary containing the treatments per individual, to be loaded as metadata in the coordinates class.

    Args:
        train_path (str): path to the training data.

    Returns:
        dict: dictionary containing the treatments per individual.

    """

    try:
        with open(
            os.path.join(
                train_path,
                [i for i in os.listdir(train_path) if i.endswith(".json")][0],
            ),
            "r",
        ) as handle:
            treatment_dict = json.load(handle)
    except IndexError:
        treatment_dict = None

    return treatment_dict


def compute_kmeans_loss(latent_means, weight=1.0, batch_size=64):  # pragma: no cover
    """

    Adds a penalty to the singular values of the Gram matrix of the latent means. It helps disentangle the latent
    space.
    Based on https://arxiv.org/pdf/1610.04794.pdf, and https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

    Args:
        latent_means: tensor containing the means of the latent distribution
        weight: weight of the Gram loss in the total loss function
        batch_size: batch size of the data to compute the kmeans loss for.

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


def get_angles(pos: int, i: int, d_model: int):
    """

    Auxiliary function for positional encoding computation.

    Args:
        pos (int): position in the sequence.
        i (int): number of sequences.
        d_model (int): dimensionality of the embeddings.

    """

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int):
    """

    Computes positional encodings, as in
    https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

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
    """

    Creates a padding mask, with zeros where data is missing, and ones where data is available.

    Args:
        seq (tf.Tensor): Sequence to compute the mask on

    """

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return tf.cast(1 - seq[:, tf.newaxis, tf.newaxis, :], tf.float32)


def create_look_ahead_mask(size: int):
    """

    Creates a triangular matrix containing an increasing amount of ones from left to right on each subsequent row.
    Useful for transformer decoder, which allows it to go through the data in a sequential manner, without taking
    the future into account.

    Args:
        size (int): number of time steps in the sequence

    """

    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, tf.float32)


def create_masks(inp: tf.Tensor):
    """

    Given an input sequence, it creates all necessary masks to pass it through the transformer architecture.
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
    model, data, epochs=1, batch_size=32, min_rate=10 ** -8, max_rate=10 ** -1
):
    """

    Trains the provided model for an epoch with an exponentially increasing learning rate

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
    """

    Computes hard counts per cluster in a differentiable way

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
    """

    Computes the KL divergence between the cluster assignment distribution
    and a uniform prior across clusters. While this assumes an equal distribution
    between clusters, the prior can be tweaked to reflect domain knowledge.

    Args:
        soft_counts (tf.Tensor): soft counts per cluster
        k (int): number of clusters
        n_samples (int): number of samples to draw from the categorical distribution
        modeling cluster assignments.

    """

    hard_counts = get_hard_counts(soft_counts)

    dist_a = tfd.Categorical(probs=hard_counts / k)
    dist_b = tfd.Categorical(logits=tf.ones(k))

    z = dist_a.sample(n_samples)

    return tf.reduce_mean(dist_a.log_prob(z) - dist_b.log_prob(z))


def get_callbacks(
    embedding_model: "str",
    kmeans_loss: float = 1.0,
    input_type: str = False,
    cp: bool = False,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """

    Generates callbacks used for model training.

    Args:
        embedding_model (str): name of the embedding model
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
        "deepof_unsupervised_{}_encodings".format(embedding_model),
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
    """

    Custom early stopping callback. Prevents the model from stopping before warmup is over

    """

    def __init__(self, start_epoch, *args, **kwargs):
        """

        Initializes the CustomStopper callback.

        Args:
            start_epoch: epoch from which performance will be taken into account when deciding whether to stop training.
            *args: arguments passed to the callback.
            **kwargs: keyword arguments passed to the callback.

        """
        super(CustomStopper, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def get_config(self):  # pragma: no cover
        """

        Updates callback metadata

        """

        config = super().get_config().copy()
        config.update({"start_epoch": self.start_epoch})
        return config

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    """

    Simple class that allows to grow learning rate exponentially during training.
    Used to trigger optimal learning rate search in deepof.train_utils.find_learning_rate.

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
    """

    Maps the reconstruction output of a given decoder to a multivariate normal distribution.

    """

    def __init__(self, input_shape, **kwargs):
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

    def call(self, inputs):
        """

        Maps the reconstruction output of a given decoder to a multivariate normal distribution.

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
            get_neighbourhood_entropy, tensor=encodings, clusters=hard_groups, k=self.k
        )

        neighbourhood_entropy = tf.map_fn(
            get_local_neighbourhood_entropy,
            tf.range(tf.shape(encodings)[0]),
            dtype=tf.dtypes.float32,
        )

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

        self.add_metric(
            neighbourhood_entropy, aggregation="mean", name="local_cluster_entropy"
        )

        return encodings


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
        self.add_metric(kl_batch, aggregation="mean", name="kl_divergence")

        return distribution_a


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """

    Transformer encoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer

    """

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """

        Constructor for the transformer encoder layer.

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

    def call(self, x, training, mask, return_scores=False):
        attn_output, attn_scores = self.mha(
            key=x, query=x, value=x, attention_mask=mask, return_attention_scores=True
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        if return_scores:
            return out2, attn_scores

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """

    Transformer decoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer

    """

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """

        Constructor for the transformer decoder layer.

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

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

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
    """

    Transformer encoder. Based on https://www.tensorflow.org/text/tutorials/transformer.
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
        """

        Constructor for the transformer encoder.

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

    def call(self, x, training):

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
    """

    Transformer decoder. Based on https://www.tensorflow.org/text/tutorials/transformer.
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
        """

        Constructor for the transformer decoder.

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

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
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
    """

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


def autoencoder_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    embedding_model: str,
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
    # GMVAE Model specific parameters
    kl_annealing_mode: str,
    kl_warmup: int,
    reg_cat_clusters: float,
    run: int = 0,
    strategy: tf.distribute.Strategy = "one_device",
):
    """

    Trains the specified autoencoder on the preprocessed data.

    Args:
        preprocessed_object (tuple): Tuple containing the preprocessed data.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), GMVAE,
        and contrastive.
        batch_size (int): Batch size to use for training.
        latent_dim (int): Encoding size to use for training.
        epochs (int): Number of epochs to train the autoencoder for.
        log_history (bool): Whether to log the history of the autoencoder.
        log_hparams (bool): Whether to log the hyperparameters used for training.
        n_components (int): Number of components to fit to the data.
        output_path (str): Path to the output directory.
        kmeans_loss (float): Weight of the gram loss, which adds a regularization term to VQVAE models which
        penalizes the correlation between the dimensions in the latent space.
        pretrained (str): Path to the pretrained weights to use for the autoencoder.
        save_checkpoints (bool): Whether to save checkpoints during training.
        save_weights (bool): Whether to save the weights of the autoencoder after training.
        input_type (str): Input type of the TableDict objects used for preprocessing. For logging purposes only.
        run (int): Run number to use for logging.
        strategy (tf.distribute.Strategy): Distribution strategy to use for training.

        # GMVAE Model specific parameters
        kl_annealing_mode (str): Mode to use for KL annealing. Must be one of "linear" (default), or "sigmoid".
        kl_warmup (int): Number of epochs during which KL is annealed.
        reg_cat_clusters (bool): whether to use the penalize uneven cluster membership in the latent space, by
        minimizing the KL divergence between cluster membership and a uniform categorical distribution.

    Returns:
        List of trained models corresponding to the selected model class. The full trained model is last.

    """

    # Check if a GPU is available and if not, fall back to CPU
    if strategy == "one_device":
        if len(tf.config.list_physical_devices("GPU")) > 0:
            strategy = tf.distribute.OneDeviceStrategy("gpu")
        else:
            strategy = tf.distribute.OneDeviceStrategy("cpu")

    # Load data
    X_train, y_train, X_val, y_val = preprocessed_object

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
        tf.data.Dataset.from_tensor_slices((tf.cast(Xs, tf.float32), tuple(ys)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.cast(Xvals, tf.float32), tuple(yvals)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build model
    with strategy.scope():

        if embedding_model == "VQVAE":
            ae_full_model = deepof.models.VQVAE(
                input_shape=X_train.shape,
                latent_dim=latent_dim,
                n_components=n_components,
                kmeans_loss=kmeans_loss,
            )
            ae_full_model.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=1e-4, clipvalue=0.75
            )
            encoder, decoder, quantizer, ae = (
                ae_full_model.encoder,
                ae_full_model.decoder,
                ae_full_model.quantizer,
                ae_full_model.vqvae,
            )
            return_list = (encoder, decoder, quantizer, ae)

        elif embedding_model == "GMVAE":
            ae_full_model = deepof.models.GMVAE(
                input_shape=X_train.shape,
                batch_size=batch_size,
                latent_dim=latent_dim,
                kl_annealing_mode=kl_annealing_mode,
                kl_warmup_epochs=kl_warmup,
                montecarlo_kl=1000 * n_components,
                n_components=n_components,
                reg_cat_clusters=reg_cat_clusters,
            )
            encoder, decoder, grouper, ae = (
                ae_full_model.encoder,
                ae_full_model.decoder,
                ae_full_model.grouper,
                ae_full_model.gmvae,
            )
            return_list = (encoder, decoder, grouper, ae)

        elif embedding_model == "contrastive":
            raise NotImplementedError

    if pretrained:
        # If pretrained models are specified, load weights and return
        ae.load_weights(pretrained)
        return return_list

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
        optimizer=ae_full_model.optimizer, run_eagerly=(embedding_model == "GMVAE")
    )
    ae_full_model.fit(
        x=train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks_,
        verbose=1,
    )

    if not os.path.exists(os.path.join(output_path, "trained_weights")):
        os.makedirs(os.path.join(output_path, "trained_weights"))

    if save_weights:
        ae.save_weights(
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
                    "val_number_of_populated_clusters",
                    ae_full_model.history.history["val_number_of_populated_clusters"][
                        -1
                    ],
                    step=0,
                )
                tf.summary.scalar(
                    "val_reconstruction_loss",
                    ae_full_model.history.history["val_reconstruction_loss"][-1],
                    step=0,
                )
                tf.summary.scalar(
                    "val_kmeans_loss",
                    ae_full_model.history.history["val_kmeans_loss"][-1],
                    step=0,
                )
                tf.summary.scalar(
                    "val_total_loss",
                    ae_full_model.history.history["val_total_loss"][-1],
                    step=0,
                )

                if embedding_model == "VQVAE":
                    tf.summary.scalar(
                        "val_vq_loss",
                        ae_full_model.history.history["val_vq_loss"][-1],
                        step=0,
                    )

                elif embedding_model == "GMVAE":
                    tf.summary.scalar(
                        "val_kl_loss",
                        ae_full_model.history.history["val_kl_divergence"][-1],
                        step=0,
                    )

                elif embedding_model == "contrastive":
                    raise NotImplementedError

    return return_list


def tune_search(
    data: tuple,
    encoding_size: int,
    embedding_model: str,
    hypertun_trials: int,
    hpt_type: str,
    k: int,
    project_name: str,
    callbacks: List,
    batch_size: int = 64,
    n_epochs: int = 30,
    n_replicas: int = 1,
    outpath: str = "unsupervised_tuner_search",
) -> tuple:
    """

    Define the search space using keras-tuner and hyperband or bayesian optimization

    Args:
        data (tf.data.Dataset): Dataset object for training and validation.
        encoding_size (int): Size of the encoding layer.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), GMVAE,
        and contrastive.
        hypertun_trials (int): Number of hypertuning trials to run.
        hpt_type (str): Type of hypertuning to run. Must be one of "hyperband" or "bayesian".
        k (int): Number of clusters on the latent space.
        kmeans_loss (float): Weight of the kmeans loss, which enforces disentanglement by penalizing the correlation
        between dimensions in the latent space.
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

    X_train, y_train, X_val, y_val = data

    assert hpt_type in ["bayopt", "hyperband"], (
        "Invalid hyperparameter tuning framework. " "Select one of bayopt and hyperband"
    )

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((Xs, tuple(ys)))
        .batch(batch_size, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((Xvals, tuple(yvals))).batch(
        batch_size, drop_remainder=True
    )

    if embedding_model == "VQVAE":
        hypermodel = deepof.hypermodels.VQVAE(
            input_shape=X_train.shape, latent_dim=encoding_size, n_components=k
        )
    elif embedding_model == "GMVAE":
        hypermodel = deepof.hypermodels.GMVAE(
            input_shape=X_train.shape,
            latent_dim=encoding_size,
            n_components=k,
            batch_size=batch_size,
        )
    elif embedding_model == "contrastive":
        raise NotImplementedError

    tuner_objective = "val_loss"

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

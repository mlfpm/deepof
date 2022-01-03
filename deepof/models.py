# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

deep autoencoder models for unsupervised pose detection. Currently, two main models are available:

- VQ-VAE: a variational autoencoder with a vector quantization latent-space. Based on https://arxiv.org/abs/1711.00937
- GM-VAE: a variational autoencoder with a Gaussian mixture latent-space. Loosely based on https://academic.oup.com/bioinformatics/article-pdf/36/16/4415/33965265/btaa293.pdf

"""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import softplus
from tensorflow.keras.initializers import he_uniform, random_uniform
from tensorflow.keras.layers import BatchNormalization, Bidirectional
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.layers import RepeatVector, Reshape
from tensorflow.keras.optimizers import Nadam
from tensorflow_addons.layers import SpectralNormalization

import deepof.model_utils

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# noinspection PyCallingNonCallable
def get_deepof_encoder(
    input_shape,
    latent_dim,
    conv_filters=64,
    dense_layers=1,
    dense_activation="relu",
    dense_units_1=64,
    gru_units_1=128,
    gru_units_2=64,
    gru_unroll=True,
    bidirectional_merge="concat",
    dropout_rate=0.1,
):
    """

    Returns a deep neural network capable of encoding the motion tracking instances into a vector ready to be fed to
    one of the provided structured latent spaces, such as GMVAE and VQVAE.

    Args:
        input_shape (tuple): shape of the input data
        latent_dim (int): dimensionality of the latent space
        conv_filters (int): number of filters in the first convolutional layer
        dense_layers (int): number of dense layers at the end of the encoder. Defaults to 1.
        dense_activation (str): activation function for the dense layers. Defaults to "relu".
        dense_units_1 (int): number of units in the first dense layer. Defaults to 64.
        gru_units_1 (int): number of units in the first GRU layer. Defaults to 128.
        gru_units_2 (int): number of units in the second GRU layer. Defaults to 64.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to True.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".
        dropout_rate (float): dropout rate for the dropout layers. Defaults to 0.1.

    Returns:
        keras.Model: a keras model that can be trained to encode motion tracking instances into a vector.

    """

    # Define and instantiate encoder
    x = Input(shape=input_shape)
    encoder = tf.keras.layers.Masking(mask_value=0.0)(x)
    encoder = SpectralNormalization(
        tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=5,
            strides=1,  # Increased strides to yield shorter sequences
            padding="valid",
            activation=dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
    )(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Bidirectional(
        GRU(
            gru_units_1,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            unroll=gru_unroll,
            # kernel_constraint=UnitNorm(axis=0),
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Bidirectional(
        GRU(
            gru_units_2,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            unroll=gru_unroll,
            # kernel_constraint=UnitNorm(axis=0),
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = SpectralNormalization(
        Dense(
            dense_units_1,
            activation=dense_activation,
            # kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
    )(encoder)
    encoder = BatchNormalization()(encoder)

    dense_layers = [
        Dense(
            latent_dim,
            activation=dense_activation,
            # kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        for _ in range(dense_layers)
    ]
    encoder_output = []
    for layer in dense_layers:
        encoder_output.append(layer)
        encoder_output.append(BatchNormalization())

    encoder = Sequential(encoder_output)(encoder)
    encoder_output = Dropout(dropout_rate)(encoder)

    return Model(x, encoder_output, name="deepof_encoder")


# noinspection PyCallingNonCallable
def get_deepof_decoder(
    input_shape,
    latent_dim,
    conv_filters=64,
    dense_layers=1,
    dense_activation="relu",
    dense_units_1=64,
    dense_units_2=32,
    gru_units_1=128,
    gru_units_2=64,
    gru_unroll=True,
    bidirectional_merge="concat",
    dropout_rate=0.1,
):

    """

    Returns a deep neural network capable of decoding the structured latent space generated by one of the compatible
    classes, such as GMVAE and VQVAE, into a sequence of motion tracking instances, either reconstructing the original
    input, or generating new data from given clusters.

    Args:
        input_shape (tuple): shape of the input data
        latent_dim (int): dimensionality of the latent space
        conv_filters (int): number of filters in the first convolutional layer
        dense_layers (int): number of dense layers at the start of the decoder. Defaults to 1.
        dense_activation (str): activation function for the dense layers. Defaults to "relu".
        dense_units_1 (int): number of units in the first dense layer. Defaults to 64.
        dense_units_2 (int): number of units in the second dense layer. Defaults to 32.
        gru_units_1 (int): number of units in the first GRU layer. Defaults to 128.
        gru_units_2 (int): number of units in the second GRU layer. Defaults to 64.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to True.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".
        dropout_rate (float): dropout rate for the dropout layers. Defaults to 0.1.

    Returns:
        keras.Model: a keras model that can be trained to decode the latent space into a series of motion tracking
        sequences.

    """

    # Define and instantiate generator
    x = Input(shape=input_shape)  # Encoder input, used to generate an output mask
    g = Input(shape=latent_dim)  # Decoder input, shaped as the latent space

    dense_layers = [
        SpectralNormalization(
            Dense(
                dense_units_2,
                activation=dense_activation,
                kernel_initializer=he_uniform(),
                use_bias=True,
            )
        )
        for _ in range(dense_layers)
    ]
    decoder_input = []
    for layer in dense_layers:
        decoder_input.append(layer)
        decoder_input.append(BatchNormalization())

    generator = Sequential(decoder_input)(g)
    generator = SpectralNormalization(
        Dense(
            dense_units_1,
            activation=dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
    )(generator)
    generator = Dropout(dropout_rate)(generator)
    generator = BatchNormalization()(generator)
    generator = RepeatVector(input_shape[0])(generator)
    generator = Bidirectional(
        GRU(
            gru_units_2,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            unroll=gru_unroll,
            # kernel_constraint=UnitNorm(axis=1),
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(generator)
    generator = BatchNormalization()(generator)
    generator = Bidirectional(
        GRU(
            gru_units_1,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            unroll=gru_unroll,
            # kernel_constraint=UnitNorm(axis=1),
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(generator)
    generator = BatchNormalization()(generator)
    generator = SpectralNormalization(
        tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=5,
            strides=1,
            padding="same",
            activation=dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
    )(generator)
    generator = BatchNormalization()(generator)
    x_decoded_mean = Dense(tfpl.IndependentNormal.params_size(input_shape[1:]) // 2)(
        generator
    )
    x_decoded_var = tf.keras.activations.softplus(
        Dense(tfpl.IndependentNormal.params_size(input_shape[1:]) // 2)(generator)
    )
    x_decoded_var = tf.keras.layers.Lambda(lambda v: 1e-3 + v)(x_decoded_var)
    x_decoded = tfpl.DistributionLambda(
        make_distribution_fn=lambda decoded: tfd.Masked(
            tfd.Independent(
                tfd.Normal(
                    loc=decoded[0],
                    scale=decoded[1],
                ),
                reinterpreted_batch_ndims=1,
            ),
            validity_mask=tf.math.logical_not(tf.reduce_all(decoded[2] == 0.0, axis=2)),
        ),
        convert_to_tensor_fn="mean",
        name="vae_reconstruction",
    )(
        [x_decoded_mean, x_decoded_var, x]
    )  # x is the input to the encoder! That we use to get the mask

    return Model([x, g], x_decoded, name="deepof_decoder")


# noinspection PyCallingNonCallable
def get_vqvae(
    input_shape: tuple,
    latent_dim: int,
    n_components: int,
    beta: float = 0.25,
    conv_filters=64,
    dense_layers=1,
    dense_activation="relu",
    dense_units_1=64,
    dense_units_2=32,
    gru_units_1=128,
    gru_units_2=64,
    gru_unroll=True,
    bidirectional_merge="concat",
    dropout_rate=0.1,
):
    """

    Builds a Vector-Quantization variational autoencoder (VQ-VAE) model, adapted to the DeepOF setting.

    Args:
        input_shape (tuple): shape of the input to the encoder.
        latent_dim (int): dimension of the latent space.
        n_components (int): number of embeddings in the embedding layer.
        beta (float): beta parameter of the VQ loss.
        conv_filters (int): number of filters in the first convolutional layers ib both encoder and decoder.
        dense_layers (int): number of dense layers at the end of the encoder and start of the decoder. Defaults to 1.
        dense_activation (str): activation function for the dense layers in both encoder and decoder. Defaults to "relu".
        dense_units_1 (int): number of units in the first dense layer in both encoder and decoder. Defaults to 64.
        dense_units_2 (int): number of units in the second dense layer in both encoder and decoder. Defaults to 32.
        gru_units_1 (int): number of units in the first GRU layer in both encoder and decoder. Defaults to 128.
        gru_units_2 (int): number of units in the second GRU layer in both encoder and decoder. Defaults to 64.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to True.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".
        dropout_rate (float): dropout rate for the dropout layers in both encoder and decoder. Defaults to 0.1.

    Returns:
        encoder (tf.keras.Model): connected encoder of the VQ-VAE model.
        Outputs a vector of shape (latent_dim,).
        decoder (tf.keras.Model): connected decoder of the VQ-VAE model.
        quantizer (tf.keras.Model): connected embedder layer of the VQ-VAE model.
        Outputs cluster indices of shape (batch_size,).
        vqvae (tf.keras.Model): complete VQ VAE model.

    """
    vq_layer = deepof.model_utils.VectorQuantizer(
        n_components,
        latent_dim,
        beta=beta,
        name="vector_quantizer",
    )
    encoder = get_deepof_encoder(
        input_shape,
        latent_dim,
        conv_filters=conv_filters,
        dense_layers=dense_layers,
        dense_activation=dense_activation,
        dense_units_1=dense_units_1,
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        gru_unroll=gru_unroll,
        bidirectional_merge=bidirectional_merge,
        dropout_rate=dropout_rate,
    )
    decoder = get_deepof_decoder(
        input_shape,
        latent_dim,
        conv_filters=conv_filters,
        dense_layers=dense_layers,
        dense_activation=dense_activation,
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2,
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        gru_unroll=gru_unroll,
        bidirectional_merge=bidirectional_merge,
        dropout_rate=dropout_rate,
    )

    # Connect encoder and quantizer
    inputs = tf.keras.layers.Input(input_shape, name="encoder_input")
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)

    encoder = tf.keras.Model(inputs, encoder_outputs, name="encoder")
    quantizer = tf.keras.Model(inputs, quantized_latents, name="quantizer")
    vqvae = tf.keras.Model(
        quantizer.inputs, decoder([inputs, quantizer.outputs]), name="VQ-VAE"
    )

    return (
        encoder,
        decoder,
        quantizer,
        vqvae,
    )


class VQVAE(tf.keras.models.Model):
    """

    VQ-VAE model adapted to the DeepOF setting.

    """

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int = 32,
        n_components: int = 15,
        beta: float = 0.25,
        architecture_hparams: dict = None,
        **kwargs,
    ):
        """

        Initializes a VQ-VAE model.

        Args:
            input_shape (tuple): Shape of the input to the full model.
            latent_dim (int): Dimensionality of the latent space.
            n_components (int): Number of embeddings (clusters) in the embedding layer.
            beta (float): Beta parameter of the VQ loss.
            architecture_hparams (dict): Dictionary of architecture hyperparameters. Defaults to None.
            **kwargs: Additional keyword arguments.

        """

        super(VQVAE, self).__init__(**kwargs)
        self.seq_shape = input_shape[1:]
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.beta = beta
        self.architecture_hparams = architecture_hparams

        # Define VQ_VAE model
        self.encoder, self.decoder, self.quantizer, self.vqvae = get_vqvae(
            self.seq_shape,
            self.latent_dim,
            self.n_components,
            self.beta,
            conv_filters=self.hparams["conv_filters"],
            dense_layers=self.hparams["dense_layers"],
            dense_activation=self.hparams["dense_activation"],
            dense_units_1=self.hparams["dense_units_1"],
            dense_units_2=self.hparams["dense_units_2"],
            gru_units_1=self.hparams["gru_units_1"],
            gru_units_2=self.hparams["gru_units_2"],
            gru_unroll=self.hparams["gru_unroll"],
            bidirectional_merge=self.hparams["bidirectional_merge"],
            dropout_rate=self.hparams["dropout_rate"],
        )

        # Define metrics to track
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.val_vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @tf.function
    def call(self, inputs):
        return self.vqvae(inputs)

    @property
    def metrics(self):  # pragma: no cover
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_vq_loss_tracker,
        ]

    @property
    def hparams(self):
        hparams = {
            "conv_filters": 64,
            "dense_layers": 1,
            "dense_activation": "relu",
            "dense_units_1": 64,
            "dense_units_2": 32,
            "gru_units_1": 128,
            "gru_units_2": 64,
            "gru_unroll": True,
            "bidirectional_merge": "concat",
            "dropout_rate": 0.1,
        }
        if self.architecture_hparams is not None:
            hparams.update(self.architecture_hparams)

        return hparams

    @tf.function
    def train_step(self, data):  # pragma: no cover
        """

        Performs a training step.

        """

        # Unpack data, repacking labels into a generator
        x, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        with tf.GradientTape() as tape:
            # Get outputs from the full model
            reconstructions = self.vqvae(x)

            # Compute losses
            reconstruction_loss = tf.reduce_mean((next(y) - reconstructions) ** 2)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results (coupled with TensorBoard)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):  # pragma: no cover
        """

        Performs a test step.

        """

        # Unpack data, repacking labels into a generator
        x, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        # Get outputs from the full model
        reconstructions = self.vqvae(x)

        # Compute losses
        reconstruction_loss = tf.reduce_mean((next(y) - reconstructions) ** 2)
        total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Track losses
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results (to couple with TensorBoard in future implementations)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "vq_loss": self.val_vq_loss_tracker.result(),
        }

    def get_vq_posterior(self):
        """

        Returns the posterior distribution of the VQ-VAE.

        """

        n_components = self.vqvae.get_layer("vector_quantizer").n_components
        embeddings = self.vqvae.get_layer("vector_quantizer").embeddings
        embedding_scales = self.vqvae.get_layer("vector_quantizer").embedding_scales

        vq_posterior = tfd.MixtureSameFamily(
            mixture_distribution=tfd.categorical.Categorical(
                probs=tf.ones(n_components) / n_components
            ),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.transpose(embeddings),
                scale_diag=embedding_scales,
            ),
            name="vq_posterior",
        )

        return vq_posterior

    @property
    def posterior(self):
        """

        Returns a mixture of Gaussians fit to the final state of the  VQ-VAE codebook.

        """
        return self.get_vq_posterior()


# noinspection PyCallingNonCallable
def get_gmvae(
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
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Nadam(
        learning_rate=1e-4, clipvalue=0.75
    ),
    overlap_loss: float = 0.0,
    reg_cat_clusters: bool = False,
    reg_cluster_variance: bool = False,
    next_sequence_prediction: bool = False,
    phenotype_prediction: bool = False,
    supervised_prediction: bool = False,
    supervised_features: int = 6,
    conv_filters=64,
    dense_layers=1,
    dense_activation="relu",
    dense_units_1=64,
    dense_units_2=32,
    gru_units_1=128,
    gru_units_2=64,
    gru_unroll=True,
    bidirectional_merge="concat",
    dropout_rate=0.1,
):
    """

    Builds a Gaussian mixture variational autoencoder (GMVAE) model, adapted to the DeepOF setting.

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
            overlap_loss (float): weight of the overlap loss as described in deepof.mode_utils.ClusterOverlap.
            reg_cat_clusters (bool): whether to use the penalize uneven cluster membership in the latent space.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            next_sequence_prediction (bool): whether to add a next sequence prediction loss, which regularizes the
            model by enabling forecasting of the next sequence.
            phenotype_prediction (bool): whether to add a phenotype prediction loss, which regularizes the model by
            including phenotypic information in the latent space.
            supervised_prediction (bool): whether to add a supervised prediction loss, which regularizes the model by
            including supervised annotations (from deepof.data.Coordinates.supervised_annotation())
            information in the latent space.
            supervised_features (int): number of features in the supervised prediction label matrix.
            Ignored if supervised prediction is null.
            conv_filters (int): number of filters in the first convolutional layers ib both encoder and decoder.
            dense_layers (int): number of dense layers at the end of the encoder and start of the decoder. Defaults to 1.
            dense_activation (str): activation function for the dense layers in both encoder and decoder. Defaults to "relu".
            dense_units_1 (int): number of units in the first dense layer in both encoder and decoder. Defaults to 64.
            dense_units_2 (int): number of units in the second dense layer in both encoder and decoder. Defaults to 32.
            gru_units_1 (int): number of units in the first GRU layer in both encoder and decoder. Defaults to 128.
            gru_units_2 (int): number of units in the second GRU layer in both encoder and decoder. Defaults to 64.
            gru_unroll (bool): whether to unroll the GRU layers. Defaults to True.
            bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".
            dropout_rate (float): dropout rate for the dropout layers in both encoder and decoder. Defaults to 0.1.


    Returns:
        encoder (tf.keras.Model): connected encoder of the VQ-VAE model.
        Outputs a vector of shape (latent_dim,).
        decoder (tf.keras.Model): connected decoder of the VQ-VAE model.
        grouper (tf.keras.Model): deep clustering branch of the VQ-VAE model. Outputs a vector of shape (n_components,).
        for each training instance, corresponding to the soft counts for each cluster.
        gmvae (tf.keras.Model): complete GMVAE model

    """

    encoder = get_deepof_encoder(
        input_shape[1:],
        latent_dim,
        conv_filters=conv_filters,
        dense_layers=dense_layers,
        dense_activation=dense_activation,
        dense_units_1=dense_units_1,
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        gru_unroll=gru_unroll,
        bidirectional_merge=bidirectional_merge,
        dropout_rate=dropout_rate,
    )
    latent_space = deepof.model_utils.GaussianMixtureLatent(
        input_shape=[input_shape[0], latent_dim],
        n_components=n_components,
        latent_dim=latent_dim,
        batch_size=batch_size,
        loss=loss,
        kl_warmup=kl_warmup,
        kl_annealing_mode=kl_annealing_mode,
        mc_kl=mc_kl,
        mmd_warmup=mmd_warmup,
        mmd_annealing_mode=mmd_annealing_mode,
        optimizer=optimizer,
        overlap_loss=overlap_loss,
        reg_cat_clusters=reg_cat_clusters,
        reg_cluster_variance=reg_cluster_variance,
        name="gaussian_mixture_latent",
    ).model
    decoder = get_deepof_decoder(
        input_shape[1:],
        latent_dim,
        conv_filters=conv_filters,
        dense_layers=dense_layers,
        dense_activation=dense_activation,
        dense_units_1=dense_units_1,
        dense_units_2=dense_units_2,
        gru_units_1=gru_units_1,
        gru_units_2=gru_units_2,
        gru_unroll=gru_unroll,
        bidirectional_merge=bidirectional_merge,
        dropout_rate=dropout_rate,
    )

    # Connect encoder and latent space
    inputs = Input(input_shape[1:])
    encoder_outputs = encoder(inputs)
    latent, categorical = latent_space(encoder_outputs)
    embedding = tf.keras.Model(inputs, latent, name="encoder")
    grouper = tf.keras.Model(inputs, categorical, name="grouper")

    # Connect decoder
    gmvae_outputs = [decoder([inputs, embedding.outputs])]

    # Add additional (optional) branches departing from the latent space
    if next_sequence_prediction:
        predictor = get_deepof_decoder(input_shape[1:], latent_dim)
        predictor._name = "deepof_predictor"
        gmvae_outputs.append(predictor([inputs, embedding.outputs]))

    if (
        phenotype_prediction
    ):  # Predict from cluster assignments instead of embedding itself!
        pheno_pred = Dense(
            latent_dim, activation="relu", kernel_initializer=he_uniform()
        )(categorical)
        pheno_pred = Dense(tfpl.IndependentBernoulli.params_size(1))(pheno_pred)
        pheno_pred = tfpl.IndependentBernoulli(
            event_shape=1,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name="phenotype_prediction",
        )(pheno_pred)

        gmvae_outputs.append(pheno_pred)

    if supervised_prediction:
        supervised_trait_pred = Dense(
            n_components,
            activation="relu",
            kernel_initializer=he_uniform(),
        )(latent)
        supervised_trait_pred = Dense(
            tfpl.IndependentBernoulli.params_size(supervised_features)
        )(supervised_trait_pred)
        supervised_trait_pred = tfpl.IndependentBernoulli(
            event_shape=supervised_features,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name="supervised_prediction",
        )(supervised_trait_pred)
        gmvae_outputs.append(supervised_trait_pred)

    # Instantiate fully connected model
    gmvae = tf.keras.Model(embedding.inputs, gmvae_outputs, name="GMVAE")

    return (
        embedding,
        decoder,
        grouper,
        gmvae,
    )


# noinspection PyDefaultArgument,PyCallingNonCallable
class GMVAE(tf.keras.models.Model):
    """

    Gaussian Mixture Variational Autoencoder for pose motif elucidation.

    """

    def __init__(
        self,
        input_shape: tuple,
        batch_size: int = 64,
        latent_dim: int = 4,
        kl_annealing_mode: str = "sigmoid",
        kl_warmup_epochs: int = 20,
        loss: str = "ELBO",
        mmd_annealing_mode: str = "sigmoid",
        mmd_warmup_epochs: int = 20,
        montecarlo_kl: int = 10,
        n_components: int = 1,
        overlap_loss: float = 0.0,
        reg_cat_clusters: bool = False,
        reg_cluster_variance: bool = False,
        next_sequence_prediction: float = 0.0,
        phenotype_prediction: float = 0.0,
        supervised_prediction: float = 0.0,
        supervised_features: int = 6,
        architecture_hparams: dict = None,
        **kwargs,
    ):
        """

        Initalizes a GMVAE model.

        Args:
            input_shape (tuple): Shape of the input to the full model.
            batch_size (int): Batch size for training.
            latent_dim (int): Dimensionality of the latent space.
            kl_annealing_mode (str): Annealing mode for KL annealing. Can be one of 'linear' and 'sigmoid'.
            kl_warmup_epochs (int): Number of epochs to warmup KL annealing.
            loss (str): Loss function to use. Can be one of 'ELBO', 'MMD', and 'ELBO+MMD'.
            mmd_annealing_mode (str): Annealing mode for MMD annealing. Can be one of 'linear' and 'sigmoid'.
            mmd_warmup_epochs (int): Number of epochs to warmup MMD annealing.
            montecarlo_kl (int): Number of Monte Carlo samples for KL divergence.
            n_components (int): Number of mixture components in the latent space.
            overlap_loss (float): weight of the overlap loss as described in deepof.mode_utils.ClusterOverlap.
            reg_cat_clusters (bool): whether to use the penalize uneven cluster membership in the latent space.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            next_sequence_prediction (bool): whether to add a next sequence prediction loss, which regularizes the
            model by enabling forecasting of the next sequence.
            phenotype_prediction (bool): whether to add a phenotype prediction loss, which regularizes the model by
            including phenotypic information in the latent space.
            supervised_prediction (bool): whether to add a supervised prediction loss, which regularizes the model by
            including supervised annotations (from deepof.data.Coordinates.supervised_annotation())
            information in the latent space.
            supervised_features (int): number of features in the supervised prediction label matrix.
            Ignored if supervised prediction is null.
            architecture_hparams (dict): dictionary of hyperparameters for the architecture. Defaults to None.
            **kwargs:

        """
        super(GMVAE, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.kl_annealing_mode = kl_annealing_mode
        self.kl_warmup = kl_warmup_epochs
        self.loss = loss
        self.mc_kl = montecarlo_kl
        self.mmd_annealing_mode = mmd_annealing_mode
        self.mmd_warmup = mmd_warmup_epochs
        self.n_components = n_components
        self.optimizer = Nadam(learning_rate=1e-4, clipvalue=0.75)
        self.overlap_loss = overlap_loss
        self.next_sequence_prediction = next_sequence_prediction
        self.phenotype_prediction = phenotype_prediction
        self.supervised_prediction = supervised_prediction
        self.supervised_features = supervised_features
        self.reg_cat_clusters = reg_cat_clusters
        self.reg_cluster_variance = reg_cluster_variance
        self.architecture_hparams = architecture_hparams

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

        # Define GMVAE model
        self.encoder, self.decoder, self.grouper, self.gmvae = get_gmvae(
            self.seq_shape,
            self.n_components,
            self.latent_dim,
            self.batch_size,
            self.loss,
            self.kl_warmup,
            self.kl_annealing_mode,
            self.mc_kl,
            self.mmd_warmup,
            self.mmd_annealing_mode,
            self.optimizer,
            self.overlap_loss,
            self.reg_cat_clusters,
            self.reg_cluster_variance,
            self.next_sequence_prediction,
            self.phenotype_prediction,
            self.supervised_prediction,
            self.supervised_features,
            conv_filters=self.hparams["conv_filters"],
            dense_layers=self.hparams["dense_layers"],
            dense_activation=self.hparams["dense_activation"],
            dense_units_1=self.hparams["dense_units_1"],
            dense_units_2=self.hparams["dense_units_2"],
            gru_units_1=self.hparams["gru_units_1"],
            gru_units_2=self.hparams["gru_units_2"],
            gru_unroll=self.hparams["gru_unroll"],
            bidirectional_merge=self.hparams["bidirectional_merge"],
            dropout_rate=self.hparams["dropout_rate"],
        )

        # Define metrics to track

        # Track all loss function components
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        if "ELBO" in self.loss:
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
            self.kl_loss_weight_tracker = tf.keras.metrics.Mean(name="kl_weight")
            self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
            self.val_kl_loss_weight_tracker = tf.keras.metrics.Mean(name="kl_weight")
        if "MMD" in self.loss:
            self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")
            self.mmd_loss_weight_tracker = tf.keras.metrics.Mean(name="mmd_weight")
            self.val_mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")
            self.val_mmd_loss_weight_tracker = tf.keras.metrics.Mean(name="mmd_weight")
        if self.overlap_loss:
            self.overlap_loss_tracker = tf.keras.metrics.Mean(name="overlap_loss")
            self.val_overlap_loss_tracker = tf.keras.metrics.Mean(name="overlap_loss")
        if self.next_sequence_prediction:
            self.next_sequence_loss_tracker = tf.keras.metrics.Mean(
                name="next_sequence_loss"
            )
            self.val_next_sequence_loss_tracker = tf.keras.metrics.Mean(
                name="next_sequence_loss"
            )
        if self.phenotype_prediction:
            self.phenotype_loss_tracker = tf.keras.metrics.Mean(name="phenotype_loss")
            self.val_phenotype_loss_tracker = tf.keras.metrics.Mean(
                name="phenotype_loss"
            )
        if self.supervised_prediction:
            self.supervised_loss_tracker = tf.keras.metrics.Mean(name="supervised_loss")
            self.val_supervised_loss_tracker = tf.keras.metrics.Mean(
                name="supervised_loss"
            )

        # Track control metrics over the latent space
        self.cluster_population_tracker = tf.keras.metrics.Mean(
            name="cluster_population"
        )
        self.cluster_confidence_tracker = tf.keras.metrics.Mean(
            name="cluster_confidence"
        )
        self.local_cluster_entropy_tracker = tf.keras.metrics.Mean(
            name="local_cluster_entropy"
        )
        self.val_cluster_population_tracker = tf.keras.metrics.Mean(
            name="cluster_population"
        )
        self.val_cluster_confidence_tracker = tf.keras.metrics.Mean(
            name="cluster_confidence"
        )
        self.val_local_cluster_entropy_tracker = tf.keras.metrics.Mean(
            name="local_cluster_entropy"
        )

    @property
    def metrics(self):  # pragma: no cover
        metrics = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
        ]
        if "ELBO" in self.loss:
            metrics += [self.kl_loss_tracker, self.kl_loss_weight_tracker]
            metrics += [self.val_kl_loss_tracker, self.val_kl_loss_weight_tracker]
        if "MMD" in self.loss:
            metrics += [self.mmd_loss_tracker, self.mmd_loss_weight_tracker]
            metrics += [self.val_mmd_loss_tracker, self.val_mmd_loss_weight_tracker]
        if self.overlap_loss:
            metrics += [self.overlap_loss_tracker, self.val_overlap_loss_tracker]
        if self.next_sequence_prediction:
            metrics += [
                self.next_sequence_loss_tracker,
                self.val_next_sequence_loss_tracker,
            ]
        if self.phenotype_prediction:
            metrics += [self.phenotype_loss_tracker, self.val_phenotype_loss_tracker]
        if self.supervised_prediction:
            metrics += [self.supervised_loss_tracker, self.val_supervised_loss_tracker]

        return metrics

    @property
    def hparams(self):
        hparams = {
            "conv_filters": 64,
            "dense_layers": 1,
            "dense_activation": "relu",
            "dense_units_1": 64,
            "dense_units_2": 32,
            "gru_units_1": 128,
            "gru_units_2": 64,
            "gru_unroll": True,
            "bidirectional_merge": "concat",
            "dropout_rate": 0.1,
        }
        if self.architecture_hparams is not None:
            hparams.update(self.architecture_hparams)

        return hparams

    @property
    def prior(self):
        """

        Property to retrieve the current's model prior

        """

        return self.gmvae.get_layer("gaussian_mixture_latent").prior

    @property
    def posterior(self):
        """

        Property to retrieve the current's model posterior

        """

        pass  # TODO: compute posterior on the full dataset and store here

    @tf.function
    def call(self, inputs):
        return self.gmvae(inputs)

    @tf.function
    def train_step(self, data):  # pragma: no cover
        """

        Performs a training step.

        """

        # Unpack data, repacking labels into a generator
        x, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        with tf.GradientTape() as tape:
            # Get outputs from the full model
            outputs = self.gmvae(x)
            if isinstance(outputs, list):
                reconstructions = outputs[0]
            else:
                reconstructions = outputs

            # Compute losses
            reconstruction_loss = tf.reduce_mean((next(y) - reconstructions) ** 2)
            total_loss = reconstruction_loss + sum(self.gmvae.losses)
            if self.next_sequence_prediction:
                next_seq_predictions = [
                    out for out in outputs if "predictor" in out.name
                ][0]
                next_seq_loss = tf.reduce_mean((next(y) - next_seq_predictions) ** 2)
                total_loss += self.next_sequence_prediction * next_seq_loss
            if self.phenotype_prediction:
                pheno_predictions = [out for out in outputs if "phenotype" in out.name][
                    0
                ]
                phenotype_loss = tf.reduce_mean((next(y) - pheno_predictions) ** 2)
                total_loss += self.phenotype_prediction * phenotype_loss
            if self.supervised_prediction:
                sup_predictions = [out for out in outputs if "supervised" in out.name][
                    0
                ]
                supervised_loss = tf.reduce_mean((next(y) - sup_predictions) ** 2)
                total_loss += self.supervised_prediction * supervised_loss

        # Backpropagation
        grads = tape.gradient(total_loss, self.gmvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.gmvae.trainable_variables))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        if "ELBO" in self.loss:
            self.kl_loss_tracker.update_state(
                [
                    metric
                    for metric in self.gmvae.metrics
                    if metric.name == "kl_divergence"
                ][0]
            )
            self.kl_loss_weight_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "kl_rate"][0]
            )
        if "MMD" in self.loss:
            self.mmd_loss_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "mmd"][0]
            )
            self.mmd_loss_weight_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "mmd_rate"][
                    0
                ]
            )
        if self.overlap_loss:
            self.overlap_loss_tracker.update_state(
                [loss for loss in self.gmvae.losses if "cluster_overlap" in loss.name][
                    0
                ]
            )
        if self.next_sequence_prediction:
            self.next_sequence_loss_tracker.update_state(next_seq_loss)
        if self.phenotype_prediction:
            self.phenotype_loss_tracker.update_state(phenotype_loss)
        if self.supervised_prediction:
            self.supervised_loss_tracker.update_state(supervised_loss)

        # Track control metrics over the latent space
        self.cluster_population_tracker.update_state(
            [
                metric
                for metric in self.gmvae.metrics
                if metric.name == "number_of_populated_clusters"
            ][0].result()
        )
        self.cluster_confidence_tracker.update_state(
            [
                metric
                for metric in self.gmvae.metrics
                if metric.name == "confidence_in_selected_cluster"
            ][0].result()
        )
        self.local_cluster_entropy_tracker.update_state(
            [
                metric
                for metric in self.gmvae.metrics
                if metric.name == "local_cluster_entropy"
            ][0].result()
        )

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cluster_population": self.cluster_population_tracker.result(),
            "cluster_confidence": self.cluster_confidence_tracker.result(),
            "local_cluster_entropy": self.local_cluster_entropy_tracker.result(),
        }

        # Add optional metrics to final log dict
        if "ELBO" in self.loss:
            log_dict["kl_loss"] = self.kl_loss_tracker.result()
            log_dict["kl_weight"] = self.kl_loss_weight_tracker.result()
        if "MMD" in self.loss:
            log_dict["mmd_loss"] = self.mmd_loss_tracker.result()
            log_dict["mmd_weight"] = self.mmd_loss_weight_tracker.result()
        if self.overlap_loss:
            log_dict["overlap_loss"] = self.overlap_loss_tracker.result()
        if self.next_sequence_prediction:
            log_dict["next_sequence_loss"] = self.next_sequence_loss_tracker.result()
        if self.phenotype_prediction:
            log_dict["phenotype_loss"] = self.phenotype_loss_tracker.result()
        if self.supervised_prediction:
            log_dict["supervised_loss"] = self.supervised_loss_tracker.result()

        return log_dict

    # noinspection PyUnboundLocalVariable
    @tf.function
    def test_step(self, data):  # pragma: no cover
        """

        Performs a test step.

        """

        # Unpack data, repacking labels into a generator
        x, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        # Get outputs from the full model
        outputs = self.gmvae(x)
        if isinstance(outputs, list):
            reconstructions = outputs[0]
        else:
            reconstructions = outputs

        # Compute losses
        reconstruction_loss = tf.reduce_mean((next(y) - reconstructions) ** 2)
        total_loss = reconstruction_loss + sum(self.gmvae.losses)
        if self.next_sequence_prediction:
            next_seq_predictions = [out for out in outputs if "predictor" in out.name][
                0
            ]
            next_seq_loss = tf.reduce_mean((next(y) - next_seq_predictions) ** 2)
            total_loss += self.next_sequence_prediction * next_seq_loss
        if self.phenotype_prediction:
            pheno_predictions = [out for out in outputs if "phenotype" in out.name][0]
            phenotype_loss = tf.reduce_mean((next(y) - pheno_predictions) ** 2)
            total_loss += self.phenotype_prediction * phenotype_loss
        if self.supervised_prediction:
            sup_predictions = [out for out in outputs if "supervised" in out.name][0]
            supervised_loss = tf.reduce_mean((next(y) - sup_predictions) ** 2)
            total_loss += self.supervised_prediction * supervised_loss

        # Track losses
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        if "ELBO" in self.loss:
            self.val_kl_loss_tracker.update_state(
                [
                    metric
                    for metric in self.gmvae.metrics
                    if metric.name == "kl_divergence"
                ][0]
            )
            self.val_kl_loss_weight_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "kl_rate"][0]
            )
        if "MMD" in self.loss:
            self.val_mmd_loss_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "mmd"][0]
            )
            self.val_mmd_loss_weight_tracker.update_state(
                [metric for metric in self.gmvae.metrics if metric.name == "mmd_rate"][
                    0
                ]
            )
        if self.overlap_loss:
            self.val_overlap_loss_tracker.update_state(
                [loss for loss in self.gmvae.losses if "cluster_overlap" in loss.name][
                    0
                ]
            )
        if self.next_sequence_prediction:
            self.val_next_sequence_loss_tracker.update_state(next_seq_loss)
        if self.phenotype_prediction:
            self.val_phenotype_loss_tracker.update_state(phenotype_loss)
        if self.supervised_prediction:
            self.val_supervised_loss_tracker.update_state(supervised_loss)

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "cluster_population": self.val_cluster_population_tracker.result(),
            "cluster_confidence": self.val_cluster_confidence_tracker.result(),
            "local_cluster_entropy": self.val_local_cluster_entropy_tracker.result(),
        }

        # Add optional metrics to final log dict
        if "ELBO" in self.loss:
            log_dict["kl_loss"] = self.val_kl_loss_tracker.result()
            log_dict["kl_weight"] = self.val_kl_loss_weight_tracker.result()
        if "MMD" in self.loss:
            log_dict["mmd_loss"] = self.val_mmd_loss_tracker.result()
            log_dict["mmd_weight"] = self.val_mmd_loss_weight_tracker.result()
        if self.overlap_loss:
            log_dict["overlap_loss"] = self.val_overlap_loss_tracker.result()
        if self.next_sequence_prediction:
            log_dict[
                "next_sequence_loss"
            ] = self.val_next_sequence_loss_tracker.result()
        if self.phenotype_prediction:
            log_dict["phenotype_loss"] = self.val_phenotype_loss_tracker.result()
        if self.supervised_prediction:
            log_dict["supervised_loss"] = self.val_supervised_loss_tracker.result()

        return log_dict

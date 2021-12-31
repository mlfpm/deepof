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
        dense_layers (int): number of dense layers at the end of the encoder. Defaults to 1.
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
):
    """

    Builds a Vector-Quantization variational autoencoder (VQ-VAE) model, adapted to the DeepOF setting.

    Args:
        input_shape (tuple): shape of the input to the encoder.
        latent_dim (int): dimension of the latent space.
        n_components (int): number of embeddings in the embedding layer.
        beta (float): beta parameter of the VQ loss.

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
    encoder = get_deepof_encoder(input_shape, latent_dim)
    decoder = get_deepof_decoder(input_shape, latent_dim)

    # Connect encoder and quantizer
    inputs = tf.keras.layers.Input(input_shape, name="encoder_input")
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)

    # Connect decoder
    decoder_inputs = tf.keras.layers.Input(
        shape=encoder.output.shape[1:], name="decoder_input"
    )
    reconstructions = decoder([inputs, decoder_inputs])

    encoder = tf.keras.Model(inputs, encoder_outputs, name="encoder")
    quantizer = tf.keras.Model(inputs, quantized_latents, name="quantizer")
    decoder = tf.keras.Model([inputs, decoder_inputs], reconstructions, name="decoder")
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
        train_variance: bool = True,
        latent_dim: int = 32,
        n_components: int = 15,
        beta: float = 0.25,
        **kwargs
    ):
        """

        Initializes a VQ-VAE model.

        Args:
            input_shape (tuple): shape of the input to the full model.
            train_variance (bool): whether to train the variance of the embedding layer.
            latent_dim (int): dimension of the latent space.
            n_components (int): number of embeddings (clusters) in the embedding layer.
            beta (float): beta parameter of the VQ loss.
            **kwargs: additional keyword arguments.

        """

        super(VQVAE, self).__init__(**kwargs)
        self.seq_shape = input_shape[1:]
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.beta = beta

        # Define VQ_VAE model
        self.encoder, self.decoder, self.quantizer, self.vqvae = get_vqvae(
            self.seq_shape,
            self.latent_dim,
            self.n_components,
            self.beta,
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
    def call(self, x):
        return self.vqvae(x)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_vq_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        """

        Performs a training step.

        """
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            # Get outputs from the full model
            reconstructions = self.vqvae(x)

            # Compute losses
            reconstruction_loss = reconstruction_loss = reconstruction_loss = (
                tf.reduce_mean((y - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results (to couple with TensorBoard in future implementations)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        """

        Performs a test step.

        """

        # Unpack data
        x, y = data

        # Get outputs from the full model
        reconstructions = self.vqvae(x)

        # Compute losses
        reconstruction_loss = (
            tf.reduce_mean((y - reconstructions) ** 2) / self.train_variance
        )
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
                probs=tf.ones(n_components) / (n_components)
            ),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.transpose(embeddings),
                scale_diag=embedding_scales,
            ),
            name="vq_posterior",
        )

        return vq_posterior


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
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
    overlap_loss: float = 0.0,
    reg_cat_clusters: bool = False,
    reg_cluster_variance: bool = False,
    next_sequence_prediction: bool = False,
    phenotype_prediction: bool = False,
    supervised_prediction: bool = False,
    supervised_features: int = 6,
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
            overlap_loss (float): weight of the overlap loss as described in deepof.mode_utils.ClusterOverlap
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

    Returns:
        encoder (tf.keras.Model): connected encoder of the VQ-VAE model.
        Outputs a vector of shape (latent_dim,).
        decoder (tf.keras.Model): connected decoder of the VQ-VAE model.
        grouper (tf.keras.Model): deep clustering branch of the VQ-VAE model. Outputs a vector of shape (n_components,).
        for each training instance, corresponding to the soft counts for each cluster.
        gmvae (tf.keras.Model): complete GMVAE model

    """

    encoder = get_deepof_encoder(input_shape[1:], latent_dim)
    latent_space = deepof.model_utils.GaussianMixtureLatent(
        input_shape=input_shape,
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
        name="vector_quantizer",
    )
    decoder = get_deepof_decoder(input_shape[1:], latent_dim)

    # Connect encoder and latent space
    inputs = Input(input_shape[1:])
    encoder_outputs = encoder(inputs)
    latent, categorical = latent_space(encoder_outputs)
    embedding = tf.keras.Model(inputs, latent, name="encoder")
    grouper = tf.keras.Model(inputs, categorical, name="grouper")

    # Connect decoder
    decoder_inputs = tf.keras.layers.Input(
        shape=encoder.output.shape[1:], name="decoder_input"
    )
    reconstructions = decoder([inputs, latent])
    gmvae_outputs = [decoder([inputs, embedding.outputs])]
    generator = tf.keras.Model(
        [inputs, decoder_inputs], reconstructions, name="decoder"
    )

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
        generator,
        grouper,
        gmvae,
    )


# noinspection PyDefaultArgument,PyCallingNonCallable
class GMVAE:
    """Gaussian Mixture Variational Autoencoder for pose motif elucidation."""

    def __init__(
        self,
        architecture_hparams: dict = {},
        batch_size: int = 256,
        compile_model: bool = True,
        encoding: int = 4,
        kl_annealing_mode: str = "sigmoid",
        kl_warmup_epochs: int = 20,
        loss: str = "ELBO",
        mmd_annealing_mode: str = "sigmoid",
        mmd_warmup_epochs: int = 20,
        montecarlo_kl: int = 10,
        number_of_components: int = 1,
        overlap_loss: float = 0.0,
        next_sequence_prediction: float = 0.0,
        phenotype_prediction: float = 0.0,
        supervised_prediction: float = 0.0,
        supervised_features: int = 6,
        reg_cat_clusters: bool = False,
        reg_cluster_variance: bool = False,
    ):
        self.hparams = self.get_hparams(architecture_hparams)
        self.batch_size = batch_size
        self.bidirectional_merge = self.hparams["bidirectional_merge"]
        self.CONV_filters = self.hparams["units_conv"]
        self.DENSE_1 = int(self.hparams["units_gru"] / 2)
        self.DENSE_2 = self.hparams["units_dense2"]
        self.DROPOUT_RATE = self.hparams["dropout_rate"]
        self.ENCODING = encoding
        self.GRU_units_1 = self.hparams["units_gru"]
        self.GRU_units_2 = int(self.hparams["units_gru"] / 2)
        self.clipvalue = self.hparams["clipvalue"]
        self.dense_activation = self.hparams["dense_activation"]
        self.dense_layers_per_branch = self.hparams["dense_layers_per_branch"]
        self.learn_rate = self.hparams["learning_rate"]
        self.gru_unroll = True
        self.compile = compile_model
        self.kl_annealing_mode = kl_annealing_mode
        self.kl_warmup = kl_warmup_epochs
        self.loss = loss
        self.mc_kl = montecarlo_kl
        self.mmd_annealing_mode = mmd_annealing_mode
        self.mmd_warmup = mmd_warmup_epochs
        self.number_of_components = number_of_components
        self.optimizer = Nadam(learning_rate=self.learn_rate, clipvalue=self.clipvalue)
        self.overlap_loss = overlap_loss
        self.next_sequence_prediction = next_sequence_prediction
        self.phenotype_prediction = phenotype_prediction
        self.supervised_prediction = supervised_prediction
        self.supervised_features = supervised_features
        self.prior = "standard_normal"
        self.reg_cat_clusters = reg_cat_clusters
        self.reg_cluster_variance = reg_cluster_variance

        assert (
            "ELBO" in self.loss or "MMD" in self.loss
        ), "loss must be one of ELBO, MMD or ELBO+MMD (default)"

    @property
    def prior(self):
        """Property to set the value of the prior
        once the class is instanciated"""

        return self._prior

    def get_prior(self):
        """Sets the Variational Autoencoder prior distribution"""

        if self.prior == "standard_normal":

            self.prior = tfd.MixtureSameFamily(
                mixture_distribution=tfd.categorical.Categorical(
                    probs=tf.ones(self.number_of_components) / self.number_of_components
                ),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=tf.keras.initializers.orthogonal(gain=1.0)(
                        [self.number_of_components, self.ENCODING],
                    ),
                    scale_diag=tfb.Softplus()(
                        tf.ones([self.number_of_components, self.ENCODING])
                        / self.number_of_components
                    ),
                ),
            )

        else:  # pragma: no cover
            raise NotImplementedError(
                "Gaussian Mixtures are currently the only supported prior"
            )

    @staticmethod
    def get_hparams(params: Dict) -> Dict:
        """Sets the default parameters for the model. Overwritable with a dictionary"""

        defaults = {
            "bidirectional_merge": "concat",
            "clipvalue": 0.75,
            "dense_activation": "relu",
            "dense_layers_per_branch": 1,
            "dropout_rate": 0.1,
            "learning_rate": 1e-4,
            "units_conv": 64,
            "units_dense2": 32,
            "units_gru": 128,
        }

        for k, v in params.items():
            defaults[k] = v

        return defaults

    def get_layers(self, input_shape):
        """Instanciate all layers in the model"""

        ##### Encoder Layers
        Model_E0 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,  # Increased strides to yield shorter sequences
            padding="valid",
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_E1 = Bidirectional(
            GRU(
                self.GRU_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=0),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_E2 = Bidirectional(
            GRU(
                self.GRU_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=0),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_E3 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            # kernel_constraint=UnitNorm(axis=0),
            kernel_initializer=he_uniform(),
            use_bias=True,
        )

        seq_E = [
            Dense(
                self.DENSE_2,
                activation=self.dense_activation,
                # kernel_constraint=UnitNorm(axis=0),
                kernel_initializer=he_uniform(),
                use_bias=True,
            )
            for _ in range(self.dense_layers_per_branch)
        ]
        Model_E4 = []
        for layer in seq_E:
            Model_E4.append(layer)
            Model_E4.append(BatchNormalization())

        ##### Decoder layers
        seq_D = [
            SpectralNormalization(
                Dense(
                    self.DENSE_2,
                    activation=self.dense_activation,
                    kernel_initializer=he_uniform(),
                    use_bias=True,
                )
            )
            for _ in range(self.dense_layers_per_branch)
        ]
        Model_D1 = []
        for layer in seq_D:
            Model_D1.append(layer)
            Model_D1.append(BatchNormalization())

        Model_D2 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_D3 = RepeatVector(input_shape[1])
        Model_D4 = Bidirectional(
            GRU(
                self.GRU_units_2,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_D5 = Bidirectional(
            GRU(
                self.GRU_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_D6 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="same",
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )

        ##### Next-sequence predictor layers
        Model_P1 = Dense(
            self.DENSE_1,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )
        Model_P2 = Bidirectional(
            GRU(
                self.GRU_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_P3 = Bidirectional(
            GRU(
                self.GRU_units_1,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=self.gru_unroll,
                # kernel_constraint=UnitNorm(axis=1),
                use_bias=True,
            ),
            merge_mode=self.bidirectional_merge,
        )
        Model_P4 = tf.keras.layers.Conv1D(
            filters=self.CONV_filters,
            kernel_size=5,
            strides=1,
            padding="same",
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
            use_bias=True,
        )

        ##### Phenotype classification layer
        Model_PC1 = Dense(
            self.number_of_components,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
        )

        ##### Supervised trait classification layer
        Model_RC1 = Dense(
            self.number_of_components,
            activation=self.dense_activation,
            kernel_initializer=he_uniform(),
        )

        return (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
            Model_D6,
            Model_P1,
            Model_P2,
            Model_P3,
            Model_P4,
            Model_PC1,
            Model_RC1,
        )

    def build(self, input_shape: Tuple):
        """Builds the tf.keras model"""

        # Instanciate prior
        self.get_prior()

        # Get model layers
        (
            Model_E0,
            Model_E1,
            Model_E2,
            Model_E3,
            Model_E4,
            Model_D1,
            Model_D2,
            Model_D3,
            Model_D4,
            Model_D5,
            Model_D6,
            Model_P1,
            Model_P2,
            Model_P3,
            Model_P4,
            Model_PC1,
            Model_RC1,
        ) = self.get_layers(input_shape)

        # Define and instantiate encoder
        x = Input(shape=input_shape[1:])
        encoder = tf.keras.layers.Masking(mask_value=0.0)(x)
        encoder = SpectralNormalization(Model_E0)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Model_E1(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Model_E2(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = SpectralNormalization(Model_E3)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(self.DROPOUT_RATE)(encoder)
        encoder = Sequential(Model_E4)(encoder)

        z_cat = Dense(
            self.number_of_components,
            name="cluster_assignment",
            activation="softmax",
            activity_regularizer=(
                tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                if self.reg_cat_clusters
                else None
            ),
        )(encoder)

        z_gauss_mean = Dense(
            tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            )
            // 2,
            name="cluster_means",
            activation=None,
            activity_regularizer=(tf.keras.regularizers.l1(10e-5)),
            kernel_initializer=he_uniform(),
        )(encoder)

        z_gauss_var = Dense(
            tfpl.IndependentNormal.params_size(
                self.ENCODING * self.number_of_components
            )
            // 2,
            name="cluster_variances",
            activation=None,
            kernel_regularizer=(
                tf.keras.regularizers.l2(0.01) if self.reg_cluster_variance else None
            ),
            activity_regularizer=deepof.model_utils.MeanVarianceRegularizer(0.05),
            kernel_initializer=random_uniform(),
        )(encoder)

        z_gauss = tf.keras.layers.concatenate([z_gauss_mean, z_gauss_var], axis=1)

        z_gauss = Reshape([2 * self.ENCODING, self.number_of_components])(z_gauss)

        z = tfpl.DistributionLambda(
            make_distribution_fn=lambda gauss: tfd.mixture.Mixture(
                cat=tfd.categorical.Categorical(
                    probs=gauss[0],
                ),
                components=[
                    tfd.Independent(
                        tfd.Normal(
                            loc=gauss[1][..., : self.ENCODING, k],
                            scale=1e-3
                            + softplus(tf.math.exp(gauss[1][..., self.ENCODING :, k])),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for k in range(self.number_of_components)
                ],
            ),
            convert_to_tensor_fn="sample",
            name="encoding_distribution",
        )([z_cat, z_gauss])

        posterior = Model(x, z, name="SEQ_2_SEQ_trained_distribution")

        # Define and control custom loss functions
        if "ELBO" in self.loss:
            kl_warm_up_iters = tf.cast(
                self.kl_warmup * (input_shape[0] // self.batch_size + 1),
                tf.int64,
            )

            # noinspection PyCallingNonCallable
            z = deepof.model_utils.KLDivergenceLayer(
                distribution_b=self.prior,
                test_points_fn=lambda q: q.sample(self.mc_kl),
                test_points_reduce_axis=0,
                iters=self.optimizer.iterations,
                warm_up_iters=kl_warm_up_iters,
                annealing_mode=self.kl_annealing_mode,
            )(z)

        if "MMD" in self.loss:
            mmd_warm_up_iters = tf.cast(
                self.mmd_warmup * (input_shape[0] // self.batch_size + 1),
                tf.int64,
            )

            z = deepof.model_utils.MMDiscrepancyLayer(
                batch_size=self.batch_size,
                prior=self.prior,
                iters=self.optimizer.iterations,
                warm_up_iters=mmd_warm_up_iters,
                annealing_mode=self.mmd_annealing_mode,
            )(z)

        # Dummy layer with no parameters, to retrieve the previous tensor
        z = tf.keras.layers.Lambda(lambda t: t, name="latent_distribution")(z)

        # Tracks clustering metrics and adds a KNN regularizer if self.overlap_loss != 0
        if self.number_of_components > 1:
            z = deepof.model_utils.ClusterOverlap(
                batch_size=self.batch_size,
                encoding_dim=self.ENCODING,
                k=self.number_of_components,
                loss_weight=self.overlap_loss,
            )([z, z_cat])

        # Define and instantiate generator
        g = Input(shape=self.ENCODING)
        generator = Sequential(Model_D1)(g)
        generator = SpectralNormalization(Model_D2)(generator)
        generator = BatchNormalization()(generator)
        generator = Model_D3(generator)
        generator = Model_D4(generator)
        generator = BatchNormalization()(generator)
        generator = Model_D5(generator)
        generator = BatchNormalization()(generator)
        generator = SpectralNormalization(Model_D6)(generator)
        generator = BatchNormalization()(generator)
        x_decoded_mean = Dense(
            tfpl.IndependentNormal.params_size(input_shape[2:]) // 2
        )(generator)
        x_decoded_var = tf.keras.activations.softplus(
            Dense(tfpl.IndependentNormal.params_size(input_shape[2:]) // 2)(generator)
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
                validity_mask=tf.math.logical_not(
                    tf.reduce_all(decoded[2] == 0.0, axis=2)
                ),
            ),
            convert_to_tensor_fn="mean",
            name="vae_reconstruction",
        )([x_decoded_mean, x_decoded_var, x])

        # define individual branches as models
        encoder = Model(x, z, name="SEQ_2_SEQ_VEncoder")
        generator = Model([g, x], x_decoded, name="vae_reconstruction")

        model_outs = [generator([encoder.outputs, encoder.inputs])]
        model_losses = [deepof.model_utils.log_loss]
        model_metrics = {"vae_reconstruction": ["mae", "mse"]}
        loss_weights = [1.0]

        ##### If requested, instantiate next-sequence-prediction model branch
        if self.next_sequence_prediction > 0:
            # Define and instantiate predictor
            predictor = Dense(
                self.DENSE_2,
                activation=self.dense_activation,
                kernel_initializer=he_uniform(),
            )(z)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P1(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = RepeatVector(input_shape[1])(predictor)
            predictor = Model_P2(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P3(predictor)
            predictor = BatchNormalization()(predictor)
            predictor = Model_P4(predictor)
            x_predicted_mean = Dense(
                tfpl.IndependentNormal.params_size(input_shape[2:]) // 2
            )(predictor)
            x_predicted_var = tf.keras.activations.softplus(
                Dense(tfpl.IndependentNormal.params_size(input_shape[2:]) // 2)(
                    predictor
                )
            )
            x_predicted_var = tf.keras.layers.Lambda(lambda v: 1e-3 + v)(
                x_predicted_var
            )
            x_predicted = tfpl.DistributionLambda(
                make_distribution_fn=lambda predicted: tfd.Masked(
                    tfd.Independent(
                        tfd.Normal(
                            loc=predicted[0],
                            scale=predicted[1],
                        ),
                        reinterpreted_batch_ndims=1,
                    ),
                    validity_mask=tf.math.logical_not(
                        tf.reduce_all(predicted[2] == 0.0, axis=2)
                    ),
                ),
                convert_to_tensor_fn="mean",
                name="vae_prediction",
            )([x_predicted_mean, x_predicted_var, x])

            model_outs.append(x_predicted)
            model_losses.append(log_loss)
            model_metrics["vae_prediction"] = ["mae", "mse"]
            loss_weights.append(self.next_sequence_prediction)

        ##### If requested, instantiate phenotype-prediction model branch
        if self.phenotype_prediction > 0:
            pheno_pred = Model_PC1(z)
            pheno_pred = Dense(tfpl.IndependentBernoulli.params_size(1))(pheno_pred)
            pheno_pred = tfpl.IndependentBernoulli(
                event_shape=1,
                convert_to_tensor_fn=tfp.distributions.Distribution.mean,
                name="phenotype_prediction",
            )(pheno_pred)

            model_outs.append(pheno_pred)
            model_losses.append(log_loss)
            model_metrics["phenotype_prediction"] = ["AUC", "accuracy"]
            loss_weights.append(self.phenotype_prediction)

        ##### If requested, instantiate supervised-annotation-prediction model branch
        if self.supervised_prediction > 0:
            supervised_trait_pred = Model_RC1(z)

            supervised_trait_pred = Dense(
                tfpl.IndependentBernoulli.params_size(self.supervised_features)
            )(supervised_trait_pred)
            supervised_trait_pred = tfpl.IndependentBernoulli(
                event_shape=self.supervised_features,
                convert_to_tensor_fn=tfp.distributions.Distribution.mean,
                name="supervised_prediction",
            )(supervised_trait_pred)

            model_outs.append(supervised_trait_pred)
            model_losses.append(log_loss)
            model_metrics["supervised_prediction"] = [
                "mae",
                "mse",
            ]
            loss_weights.append(self.supervised_prediction)

        # define grouper and end-to-end autoencoder model
        grouper = Model(encoder.inputs, z_cat, name="Deep_Gaussian_Mixture_clustering")
        gmvaep = Model(
            inputs=encoder.inputs,
            outputs=model_outs,
            name="SEQ_2_SEQ_GMVAE",
        )

        if self.compile:
            gmvaep.compile(
                loss=model_losses,
                optimizer=self.optimizer,
                metrics=model_metrics,
                loss_weights=loss_weights,
            )

        gmvaep.build(input_shape)

        return (
            encoder,
            generator,
            grouper,
            gmvaep,
            self.prior,
            posterior,
        )

    @prior.setter
    def prior(self, value):
        self._prior = value

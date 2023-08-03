"""deep autoencoder models for unsupervised pose detection.

- VQ-VAE: a variational autoencoder with a vector quantization latent-space (https://arxiv.org/abs/1711.00937).
- VaDE: a variational autoencoder with a Gaussian mixture latent-space.
- Contrastive: an embedding model consisting of a single encoder, trained using a contrastive loss.

"""
# @author lucasmiranda42
# encoding: utf-8
# module deepof

from sklearn.mixture import GaussianMixture
from spektral.layers import CensNetConv
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Dense, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.layers import LayerNormalization, Bidirectional
from tensorflow.keras.optimizers import Nadam
from typing import Any, NewType
import numpy as np
import tcn
import tensorflow as tf
import tensorflow_probability as tfp

import deepof.model_utils

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# noinspection PyCallingNonCallable
def get_recurrent_encoder(
    input_shape: tuple,
    edge_feature_shape: tuple,
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    use_gnn: bool = True,
    gru_unroll: bool = False,
    bidirectional_merge: str = "concat",
    interaction_regularization: float = 0.0,
):
    """Return a deep recurrent neural encoder.

     Builds a neural network capable of encoding the motion tracking instances into a vector ready to be fed to
    one of the provided structured latent spaces.

    Args:
        input_shape (tuple): shape of the node features for the input data. Should be time x nodes x features.
        edge_feature_shape (tuple): shape of the adjacency matrix to use in the graph attention layers. Should be time x edges x features.
        adjacency_matrix (np.ndarray): adjacency matrix for the mice connectivity graph. Shape should be nodes x nodes.
        latent_dim (int): dimension of the latent space.
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to False.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".
        interaction_regularization (float): Regularization parameter for the interaction features.

    Returns:
        keras.Model: a keras model that can be trained to encode motion tracking instances into a vector.

    """
    # Define feature and adjacency inputs
    x = Input(shape=input_shape)
    a = Input(shape=edge_feature_shape)

    if use_gnn:
        x_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(x),
                [
                    -1,
                    adjacency_matrix.shape[-1],
                    x.shape[1],
                    input_shape[-1] // adjacency_matrix.shape[-1],
                ][::-1],
            )
        )
        a_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(a),
                [
                    -1,
                    edge_feature_shape[-1],
                    a.shape[1],
                    1,
                ][::-1],
            )
        )

    else:
        x_reshaped = tf.expand_dims(x, axis=1)

    # Instantiate temporal RNN block
    encoder = deepof.model_utils.get_recurrent_block(
        x_reshaped, latent_dim, gru_unroll, bidirectional_merge
    )(x_reshaped)

    # Instantiate spatial graph block
    if use_gnn:

        # Embed edge features too
        a_encoder = deepof.model_utils.get_recurrent_block(
            a_reshaped, latent_dim, gru_unroll, bidirectional_merge
        )(a_reshaped)

        spatial_block = CensNetConv(
            node_channels=latent_dim,
            edge_channels=latent_dim,
            activation="relu",
            node_regularizer=tf.keras.regularizers.l1(interaction_regularization),
        )

        # Process adjacency matrix
        laplacian, edge_laplacian, incidence = spatial_block.preprocess(
            adjacency_matrix
        )

        # Get and concatenate node and edge embeddings
        x_nodes, x_edges = spatial_block(
            [encoder, (laplacian, edge_laplacian, incidence), a_encoder], mask=None
        )

        x_nodes = tf.reshape(
            x_nodes,
            [-1, adjacency_matrix.shape[-1] * latent_dim],
        )

        x_edges = tf.reshape(
            x_edges,
            [-1, edge_feature_shape[-1] * latent_dim],
        )

        encoder = tf.concat([x_nodes, x_edges], axis=-1)

    else:
        encoder = tf.squeeze(encoder, axis=1)

    encoder_output = tf.keras.layers.Dense(latent_dim, kernel_initializer="he_uniform")(
        encoder
    )

    return Model([x, a], encoder_output, name="recurrent_encoder")


# noinspection PyCallingNonCallable
def get_recurrent_decoder(
    input_shape: tuple,
    latent_dim: int,
    gru_unroll: bool = False,
    bidirectional_merge: str = "concat",
):
    """Return a recurrent neural decoder.

    Builds a deep neural network capable of decoding the structured latent space generated by one of the compatible
    classes into a sequence of motion tracking instances, either reconstructing the original
    input, or generating new data from given clusters.

    Args:
        input_shape (tuple): shape of the input data
        latent_dim (int): dimensionality of the latent space
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to False.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".

    Returns:
        keras.Model: a keras model that can be trained to decode the latent space into a series of motion tracking
        sequences.

    """
    # Define and instantiate generator
    g = Input(shape=latent_dim)  # Decoder input, shaped as the latent space
    x = Input(shape=input_shape)  # Encoder input, used to generate an output mask
    validity_mask = tf.math.logical_not(tf.reduce_all(x == 0.0, axis=2))

    generator = RepeatVector(input_shape[0])(g)
    generator = Bidirectional(
        GRU(
            latent_dim,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            unroll=gru_unroll,
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(generator, mask=validity_mask)
    generator = LayerNormalization()(generator)
    generator = Bidirectional(
        GRU(
            2 * latent_dim,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            unroll=gru_unroll,
            use_bias=True,
        ),
        merge_mode=bidirectional_merge,
    )(generator)
    generator = LayerNormalization()(generator)
    generator = tf.keras.layers.Conv1D(
        filters=2 * latent_dim,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=he_uniform(),
        use_bias=False,
    )(generator)
    generator = LayerNormalization()(generator)

    x_decoded = deepof.model_utils.ProbabilisticDecoder(input_shape)(
        [generator, validity_mask]
    )

    return Model([g, x], x_decoded, name="recurrent_decoder")


def get_TCN_encoder(
    input_shape: tuple,
    edge_feature_shape: tuple,
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    use_gnn: bool = True,
    conv_filters: int = 32,
    kernel_size: int = 4,
    conv_stacks: int = 2,
    conv_dilations: tuple = (1, 2, 4, 8),
    padding: str = "causal",
    use_skip_connections: bool = True,
    dropout_rate: int = 0,
    activation: str = "relu",
    interaction_regularization: float = 0.0,
):
    """Return a Temporal Convolutional Network (TCN) encoder.

    Builds a neural network that can be used to encode motion tracking instances into a
    vector. Each layer contains a residual block with a convolutional layer and a skip connection. See the following
    paper for more details: https://arxiv.org/pdf/1803.01271.pdf

    Args:
        input_shape: shape of the input data
        edge_feature_shape (tuple): shape of the adjacency matrix to use in the graph attention layers. Should be time x edges x features.
        adjacency_matrix (np.ndarray): adjacency matrix for the mice connectivity graph. Shape should be nodes x nodes.
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
        latent_dim: dimensionality of the latent space
        conv_filters: number of filters in the TCN layers
        kernel_size: size of the convolutional kernels
        conv_stacks: number of TCN layers
        conv_dilations: list of dilation factors for each TCN layer
        padding: padding mode for the TCN layers
        use_skip_connections: whether to use skip connections between TCN layers
        dropout_rate: dropout rate for the TCN layers
        activation: activation function for the TCN layers
        interaction_regularization (float): Regularization parameter for the interaction features

    Returns:
        keras.Model: a keras model that can be trained to encode a sequence of motion tracking instances into a latent
        space using temporal convolutional networks.

    """
    # Define feature and adjacency inputs
    x = Input(shape=input_shape)
    a = Input(shape=edge_feature_shape)

    if use_gnn:
        x_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(x),
                [
                    -1,
                    adjacency_matrix.shape[-1],
                    x.shape[1],
                    input_shape[-1] // adjacency_matrix.shape[-1],
                ][::-1],
            )
        )
        a_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(a),
                [
                    -1,
                    edge_feature_shape[-1],
                    a.shape[1],
                    1,
                ][::-1],
            )
        )

    else:
        x_reshaped = tf.expand_dims(x, axis=1)

    encoder = TimeDistributed(
        tcn.TCN(
            conv_filters,
            kernel_size,
            conv_stacks,
            conv_dilations,
            padding,
            use_skip_connections,
            dropout_rate,
            return_sequences=False,
            activation=activation,
            kernel_initializer="random_normal",
            use_batch_norm=True,
        )
    )(x_reshaped)

    # Instantiate spatial graph block
    if use_gnn:

        # Embed edge features too
        a_encoder = TimeDistributed(
            tcn.TCN(
                conv_filters,
                kernel_size,
                conv_stacks,
                conv_dilations,
                padding,
                use_skip_connections,
                dropout_rate,
                return_sequences=False,
                activation=activation,
                kernel_initializer="random_normal",
                use_batch_norm=True,
            )
        )(a_reshaped)

        spatial_block = CensNetConv(
            node_channels=latent_dim,
            edge_channels=latent_dim,
            activation="relu",
            node_regularizer=tf.keras.regularizers.l1(interaction_regularization),
        )

        # Process adjacency matrix
        laplacian, edge_laplacian, incidence = spatial_block.preprocess(
            adjacency_matrix
        )

        # Get and concatenate node and edge embeddings
        x_nodes, x_edges = spatial_block(
            [encoder, (laplacian, edge_laplacian, incidence), a_encoder], mask=None
        )

        x_nodes = tf.reshape(
            x_nodes,
            [-1, adjacency_matrix.shape[-1] * latent_dim],
        )

        x_edges = tf.reshape(
            x_edges,
            [-1, edge_feature_shape[-1] * latent_dim],
        )

        encoder = tf.concat([x_nodes, x_edges], axis=-1)

    else:
        encoder = tf.squeeze(encoder, axis=1)

    encoder = tf.keras.layers.Dense(2 * latent_dim, activation="relu")(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = Dense(latent_dim, activation="relu")(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Dense(latent_dim)(encoder)

    return Model([x, a], encoder, name="TCN_encoder")


def get_TCN_decoder(
    input_shape: tuple,
    latent_dim: int,
    conv_filters: int = 64,
    kernel_size: int = 4,
    conv_stacks: int = 1,
    conv_dilations: tuple = (8, 4, 2, 1),
    padding: str = "causal",
    use_skip_connections: bool = True,
    dropout_rate: int = 0,
    activation: str = "relu",
):
    """Return a Temporal Convolutional Network (TCN) decoder.

    Builds a neural network that can be used to decode a latent space into a sequence of
    motion tracking instances. Each layer contains a residual block with a convolutional layer and a skip connection. See
    the following paper for more details: https://arxiv.org/pdf/1803.01271.pdf,

    Args:
        input_shape: shape of the input data
        latent_dim: dimensionality of the latent space
        conv_filters: number of filters in the TCN layers
        kernel_size: size of the convolutional kernels
        conv_stacks: number of TCN layers
        conv_dilations: list of dilation factors for each TCN layer
        padding: padding mode for the TCN layers
        use_skip_connections: whether to use skip connections between TCN layers
        dropout_rate: dropout rate for the TCN layers
        activation: activation function for the TCN layers

    Returns:
        keras.Model: a keras model that can be trained to decode a latent space into a sequence of motion tracking
        instances using temporal convolutional networks.

    """
    # Define and instantiate generator
    g = Input(shape=latent_dim)  # Decoder input, shaped as the latent space
    x = Input(shape=input_shape)  # Encoder input, used to generate an output mask
    validity_mask = tf.math.logical_not(tf.reduce_all(x == 0.0, axis=2))

    generator = tf.keras.layers.Dense(latent_dim)(g)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.Dense(2 * latent_dim, activation="relu")(generator)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.Dense(4 * latent_dim, activation="relu")(generator)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.RepeatVector(input_shape[0])(generator)

    generator = tcn.TCN(
        conv_filters,
        kernel_size,
        conv_stacks,
        conv_dilations,
        padding,
        use_skip_connections,
        dropout_rate,
        return_sequences=True,
        activation=activation,
        kernel_initializer="random_normal",
        use_batch_norm=True,
    )(generator)

    x_decoded = deepof.model_utils.ProbabilisticDecoder(input_shape)(
        [generator, validity_mask]
    )

    return Model([g, x], x_decoded, name="TCN_decoder")


# noinspection PyCallingNonCallable
def get_transformer_encoder(
    input_shape: tuple,
    edge_feature_shape: tuple,
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    use_gnn: bool = True,
    num_layers: int = 4,
    num_heads: int = 64,
    dff: int = 128,
    dropout_rate: float = 0.1,
    interaction_regularization: float = 0.0,
):
    """Build a Transformer encoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    Args:
        input_shape (tuple): shape of the input data
        edge_feature_shape (tuple): shape of the adjacency matrix to use in the graph attention layers. Should be time x edges x features.
        adjacency_matrix (np.ndarray): adjacency matrix for the mice connectivity graph. Shape should be nodes x nodes.
        latent_dim (int): dimensionality of the latent space
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
        num_layers (int): number of transformer layers to include
        num_heads (int): number of heads of the multi-head-attention layers used on the transformer encoder
        dff (int): dimensionality of the token embeddings
        dropout_rate (float): dropout rate
        interaction_regularization (float): regularization parameter for the interaction features

    """
    # Define feature and adjacency inputs
    x = Input(shape=input_shape)
    a = Input(shape=edge_feature_shape)

    if use_gnn:
        x_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(x),
                [
                    -1,
                    adjacency_matrix.shape[-1],
                    x.shape[1],
                    input_shape[-1] // adjacency_matrix.shape[-1],
                ][::-1],
            )
        )
        a_reshaped = tf.transpose(
            tf.reshape(
                tf.transpose(a),
                [
                    -1,
                    edge_feature_shape[-1],
                    a.shape[1],
                    1,
                ][::-1],
            )
        )

    else:
        x_reshaped = tf.expand_dims(x, axis=1)

    transformer_embedding = TimeDistributed(
        deepof.model_utils.TransformerEncoder(
            num_layers=num_layers,
            seq_dim=input_shape[-1],
            key_dim=input_shape[-1],
            num_heads=num_heads,
            dff=dff,
            maximum_position_encoding=input_shape[0],
            rate=dropout_rate,
        )
    )(x_reshaped, training=False)
    transformer_embedding = tf.reshape(
        transformer_embedding,
        [
            -1,
            (adjacency_matrix.shape[0] if x_reshaped.shape[1] != 1 else 1),
            input_shape[0] * input_shape[1],
        ],
    )

    if use_gnn:

        # Embed edge features too
        transformer_a_embedding = TimeDistributed(
            deepof.model_utils.TransformerEncoder(
                num_layers=num_layers,
                seq_dim=input_shape[-1],
                key_dim=input_shape[-1],
                num_heads=num_heads,
                dff=dff,
                maximum_position_encoding=input_shape[0],
                rate=dropout_rate,
            )
        )(a_reshaped, training=False)

        transformer_a_embedding = tf.reshape(
            transformer_a_embedding,
            [-1, adjacency_matrix.shape[0], input_shape[0] * input_shape[1]],
        )

        spatial_block = CensNetConv(
            node_channels=latent_dim,
            edge_channels=latent_dim,
            activation="relu",
            node_regularizer=tf.keras.regularizers.l1(interaction_regularization),
        )

        # Process adjacency matrix
        laplacian, edge_laplacian, incidence = spatial_block.preprocess(
            adjacency_matrix
        )

        # Get and concatenate node and edge embeddings
        x_nodes, x_edges = spatial_block(
            [
                transformer_embedding,
                (laplacian, edge_laplacian, incidence),
                transformer_a_embedding,
            ],
            mask=None,
        )

        x_nodes = tf.reshape(
            x_nodes,
            [-1, adjacency_matrix.shape[-1] * latent_dim],
        )

        x_edges = tf.reshape(
            x_edges,
            [-1, edge_feature_shape[-1] * latent_dim],
        )

        transformer_embedding = tf.concat([x_nodes, x_edges], axis=-1)

    else:
        transformer_embedding = tf.squeeze(transformer_embedding, axis=1)

    encoder = tf.keras.layers.Dense(2 * latent_dim, activation="relu")(
        transformer_embedding
    )
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Dense(latent_dim, activation="relu")(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Dense(latent_dim)(encoder)

    return tf.keras.models.Model([x, a], encoder, name="transformer_encoder")


def get_transformer_decoder(
    input_shape, latent_dim, num_layers=2, num_heads=8, dff=128, dropout_rate=0.1
):
    """Build a Transformer decoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    Args:
        input_shape (tuple): shape of the input data
        latent_dim (int): dimensionality of the latent space
        num_layers (int): number of transformer layers to include
        num_heads (int): number of heads of the multi-head-attention layers used on the transformer encoder
        dff (int): dimensionality of the token embeddings
        dropout_rate (float): dropout rate

    """
    x = tf.keras.layers.Input(input_shape)
    g = tf.keras.layers.Input([latent_dim])
    validity_mask = tf.math.logical_not(tf.reduce_all(x == 0.0, axis=2))

    generator = tf.keras.layers.Dense(latent_dim)(g)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.Dense(2 * latent_dim, activation="relu")(generator)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.Dense(4 * latent_dim, activation="relu")(generator)
    generator = tf.keras.layers.BatchNormalization()(generator)
    generator = tf.keras.layers.RepeatVector(input_shape[0])(generator)

    # Get masks for generated output
    _, look_ahead_mask, padding_mask = deepof.model_utils.create_masks(generator)

    (transformer_embedding, attention_weights,) = deepof.model_utils.TransformerDecoder(
        num_layers=num_layers,
        seq_dim=input_shape[-1],
        key_dim=input_shape[-1],
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=input_shape[0],
        rate=dropout_rate,
    )(
        x,
        generator,
        training=False,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )

    x_decoded = deepof.model_utils.ProbabilisticDecoder(input_shape)(
        [transformer_embedding, validity_mask]
    )

    return tf.keras.models.Model(
        [g, x], [x_decoded, attention_weights], name="transformer_decoder"
    )


class VectorQuantizer(tf.keras.models.Model):
    """Vector quantizer layer.

    Quantizes the input vectors into a fixed number of clusters using L2 norm. Based on
    https://arxiv.org/pdf/1509.03700.pdf, and adapted for clustering using https://arxiv.org/abs/1806.02199.
    Implementation based on https://keras.io/examples/generative/vq_vae/.

    """

    def __init__(
        self, n_components, embedding_dim, beta, kmeans_loss: float = 0.0, **kwargs
    ):
        """Initialize the VQ layer.

        Args:
            n_components (int): number of embeddings to use
            embedding_dim (int): dimensionality of the embeddings
            beta (float): beta value for the loss function
            kmeans_loss (float): regularization parameter for the Gram matrix
            **kwargs: additional arguments for the parent class

        """
        super(VectorQuantizer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta
        self.kmeans = kmeans_loss

        # Initialize the VQ codebook
        w_init = tf.random_uniform_initializer()
        self.codebook = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.n_components), dtype="float32"
            ),
            trainable=True,
            name="vqvae_codebook",
        )

    def call(self, x):  # pragma: no cover
        """Compute the VQ layer.

        Args:
            x (tf.Tensor): input tensor

        Returns:
                x (tf.Tensor): output tensor
        """
        # Compute input shape and flatten, keeping the embedding dimension intact
        input_shape = tf.shape(x)

        # Add a disentangling penalty to the embeddings
        if self.kmeans:
            kmeans_loss = deepof.model_utils.compute_kmeans_loss(
                x, weight=self.kmeans, batch_size=input_shape[0]
            )
            self.add_loss(kmeans_loss)
            self.add_metric(kmeans_loss, name="kmeans_loss")

        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantize input using the codebook
        encoding_indices = tf.cast(
            self.get_code_indices(flattened, return_soft_counts=False), tf.int32
        )
        soft_counts = self.get_code_indices(flattened, return_soft_counts=True)

        encodings = tf.one_hot(encoding_indices, self.n_components)

        quantized = tf.matmul(encodings, self.codebook, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Compute vector quantization loss, and add it to the layer
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator (copy gradients through the undiferentiable layer)
        # This approach has been reported to have issues for clustering, so we use add an extra
        # reconstruction loss to ensure that the gradients can flow through the encoder.
        # quantized = x + tf.stop_gradient(quantized - x)

        return quantized, soft_counts

    # noinspection PyTypeChecker
    def get_code_indices(
        self, flattened_inputs, return_soft_counts=False
    ):  # pragma: no cover
        """Getter for the code indices at any given time.

        Args:
            flattened_inputs (tf.Tensor): flattened input tensor (encoder output)
            return_soft_counts (bool): whether to return soft counts based on the distance to the codes, instead of the code indices

        Returns:
            encoding_indices (tf.Tensor): code indices tensor with cluster assignments.
        """
        # Compute L2-norm distance between inputs and codes at a given time
        similarity = tf.matmul(flattened_inputs, self.codebook)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.codebook**2, axis=0)
            - 2 * similarity
        )

        if return_soft_counts:
            # Compute soft counts based on the distance to the codes
            similarity = (1 / distances) ** 2
            soft_counts = similarity / tf.expand_dims(
                tf.reduce_sum(similarity, axis=1), axis=1
            )
            return soft_counts

        # Return index of the closest code
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


# noinspection PyCallingNonCallable
def get_vqvae(
    input_shape: tuple,
    edge_feature_shape: tuple,
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    use_gnn: bool,
    n_components: int,
    beta: float = 1.0,
    kmeans_loss: float = 0.0,
    encoder_type: str = "recurrent",
    interaction_regularization: float = 0.0,
):
    """Build a Vector-Quantization variational autoencoder (VQ-VAE) model, adapted to the DeepOF setting.

    Args:
        input_shape (tuple): shape of the input to the encoder.
        edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
        adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
        latent_dim (int): dimension of the latent space.
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
        n_components (int): number of embeddings in the embedding layer.
        beta (float): beta parameter of the VQ loss.
        kmeans_loss (float): regularization parameter for the Gram matrix.
        encoder_type (str): type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
        interaction_regularization (float): Regularization parameter for the interaction features.

    Returns:
        encoder (tf.keras.Model): connected encoder of the VQ-VAE model. Outputs a vector of shape (latent_dim,).
        decoder (tf.keras.Model): connected decoder of the VQ-VAE model.
        grouper (tf.keras.Model): connected embedder layer of the VQ-VAE model. Outputs cluster indices of shape (batch_size,).
        vqvae (tf.keras.Model): complete VQ VAE model.

    """
    vq_layer = VectorQuantizer(
        n_components,
        latent_dim,
        beta=beta,
        kmeans_loss=kmeans_loss,
        name="vector_quantizer",
    )

    if encoder_type == "recurrent":
        encoder = get_recurrent_encoder(
            input_shape=input_shape[1:],
            edge_feature_shape=edge_feature_shape[1:],
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_recurrent_decoder(
            input_shape=input_shape[1:], latent_dim=latent_dim
        )

    elif encoder_type == "TCN":
        encoder = get_TCN_encoder(
            input_shape=input_shape[1:],
            edge_feature_shape=edge_feature_shape[1:],
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_TCN_decoder(input_shape=input_shape[1:], latent_dim=latent_dim)

    elif encoder_type == "transformer":
        encoder = get_transformer_encoder(
            input_shape[1:],
            edge_feature_shape=edge_feature_shape[1:],
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_transformer_decoder(input_shape[1:], latent_dim=latent_dim)

    # Connect encoder and quantizer
    inputs = tf.keras.layers.Input(input_shape[1:], name="encoder_input")
    a = tf.keras.layers.Input(edge_feature_shape[1:], name="encoder_edge_features")
    encoder_outputs = encoder([inputs, a])
    quantized_latents, soft_counts = vq_layer(encoder_outputs)

    # Connect full models
    encoder = tf.keras.Model([inputs, a], encoder_outputs, name="encoder")
    grouper = tf.keras.Model([inputs, a], quantized_latents, name="grouper")
    soft_grouper = tf.keras.Model([inputs, a], soft_counts, name="soft_grouper")
    vqvae = tf.keras.Model(
        grouper.inputs, decoder([grouper.outputs, inputs]), name="VQ-VAE"
    )

    models = [encoder, decoder, grouper, soft_grouper, vqvae]

    return models


class VQVAE(tf.keras.models.Model):
    """VQ-VAE model adapted to the DeepOF setting."""

    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray = None,
        latent_dim: int = 8,
        n_components: int = 15,
        beta: float = 1.0,
        kmeans_loss: float = 0.0,
        use_gnn: bool = True,
        encoder_type: str = "recurrent",
        interaction_regularization: float = 0.0,
        **kwargs,
    ):
        """Initialize a VQ-VAE model.

        Args:
            input_shape (tuple): Shape of the input to the full model.
            edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
            adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
            latent_dim (int): Dimensionality of the latent space.
            n_components (int): Number of embeddings (clusters) in the embedding layer.
            beta (float): Beta parameter of the VQ loss, as described in the original VQVAE paper.
            kmeans_loss (float): Regularization parameter for the Gram matrix.
            encoder_type (str): Type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
            interaction_regularization (float): Regularization parameter for the interaction features.
            **kwargs: Additional keyword arguments.

        """
        super(VQVAE, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.edge_feature_shape = edge_feature_shape
        self.adjacency_matrix = adjacency_matrix
        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        self.n_components = n_components
        self.beta = beta
        self.kmeans = kmeans_loss
        self.encoder_type = encoder_type
        self.interaction_regularization = interaction_regularization

        # Define VQ_VAE model
        (
            self.encoder,
            self.decoder,
            self.grouper,
            self.soft_grouper,
            self.vqvae,
        ) = get_vqvae(
            self.seq_shape,
            self.edge_feature_shape,
            self.adjacency_matrix,
            self.latent_dim,
            self.use_gnn,
            self.n_components,
            self.beta,
            self.kmeans,
            self.encoder_type,
            self.interaction_regularization,
        )

        # Define metrics to track
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.encoding_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="encoding_reconstruction_loss"
        )
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.cluster_population = tf.keras.metrics.Mean(
            name="number_of_populated_clusters"
        )
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        self.val_encoding_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_encoding_reconstruction_loss"
        )
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_vq_loss_tracker = tf.keras.metrics.Mean(name="val_vq_loss")
        self.val_cluster_population = tf.keras.metrics.Mean(
            name="val_number_of_populated_clusters"
        )

    @tf.function
    def call(self, inputs, **kwargs):
        """Call the VQVAE model."""
        return self.vqvae(inputs, **kwargs)

    @property
    def metrics(self):  # pragma: no cover
        """Initialize VQVAE tracked metrics."""
        metrics = [
            self.total_loss_tracker,
            self.encoding_reconstruction_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.cluster_population,
            self.val_total_loss_tracker,
            self.val_encoding_reconstruction_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_vq_loss_tracker,
            self.val_cluster_population,
        ]

        return metrics

    @tf.function
    def train_step(self, data):  # pragma: no cover
        """Perform a training step."""
        # Unpack data, repacking labels into a generator
        x, a, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        with tf.GradientTape() as tape:
            # Get outputs from the full model
            encoding_reconstructions = self.vqvae([x, a], training=True)
            reconstructions = self.decoder(
                [self.encoder([x, a], training=True), x], training=True
            )

            # Get rid of the attention scores that the transformer decoder outputs
            if self.encoder_type == "transformer":
                encoding_reconstructions = encoding_reconstructions[0]
                reconstructions = reconstructions[0]

            # Compute losses
            reconstruction_labels = next(y)
            encoding_reconstruction_loss = -tf.reduce_mean(
                encoding_reconstructions.log_prob(reconstruction_labels)
            )
            reconstruction_loss = -tf.reduce_mean(
                reconstructions.log_prob(reconstruction_labels)
            )

            total_loss = (
                encoding_reconstruction_loss
                + reconstruction_loss
                + sum(self.vqvae.losses)
            )

        # Backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Compute populated clusters
        unique_indices = tf.unique(
            tf.reshape(tf.argmax(self.soft_grouper([x, a]), axis=1), [-1])
        ).y
        populated_clusters = tf.shape(unique_indices)[0]

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.encoding_reconstruction_loss_tracker.update_state(
            encoding_reconstruction_loss
        )
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.cluster_population.update_state(populated_clusters)

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.total_loss_tracker.result(),
            "encoding_reconstruction_loss": self.encoding_reconstruction_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
            "number_of_populated_clusters": self.cluster_population.result(),
        }

        return {
            **log_dict,
            **{met.name: met.result() for met in self.vqvae.metrics},
        }

    @tf.function
    def test_step(self, data):  # pragma: no cover
        """Performs a test step."""
        # Unpack data, repacking labels into a generator
        x, a, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        # Get outputs from the full model
        encoding_reconstructions = self.vqvae([x, a], training=False)
        reconstructions = self.decoder(
            [self.encoder([x, a], training=False), x], training=False
        )

        # Get rid of the attention scores that the transformer decoder outputs
        if self.encoder_type == "transformer":
            encoding_reconstructions = encoding_reconstructions[0]
            reconstructions = reconstructions[0]

        # Compute losses
        reconstruction_labels = next(y)
        encoding_reconstruction_loss = -tf.reduce_mean(
            encoding_reconstructions.log_prob(reconstruction_labels)
        )
        reconstruction_loss = -tf.reduce_mean(
            reconstructions.log_prob(reconstruction_labels)
        )
        total_loss = (
            encoding_reconstruction_loss + reconstruction_loss + sum(self.vqvae.losses)
        )

        # Compute populated clusters
        unique_indices = tf.unique(
            tf.reshape(tf.argmax(self.soft_grouper([x, a]), axis=1), [-1])
        ).y
        populated_clusters = tf.shape(unique_indices)[0]

        # Track losses
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_encoding_reconstruction_loss_tracker.update_state(
            encoding_reconstruction_loss
        )
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.val_cluster_population.update_state(populated_clusters)

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.val_total_loss_tracker.result(),
            "encoding_reconstruction_loss": self.val_encoding_reconstruction_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "vq_loss": self.val_vq_loss_tracker.result(),
            "number_of_populated_clusters": self.val_cluster_population.result(),
        }

        return {
            **log_dict,
            **{met.name: met.result() for met in self.vqvae.metrics},
        }


class GaussianMixtureLatent(tf.keras.models.Model):
    """Gaussian Mixture probabilistic latent space model.

    Used to represent the embedding of motion tracking data in a mixture of Gaussians
    with a provided number of components, with means, covariances and weights.
    Implementation based on VaDE (https://arxiv.org/abs/1611.05148)
    and VaDE-SC (https://openreview.net/forum?id=RQ428ZptQfU).

    """

    def __init__(
        self,
        input_shape: tuple,
        n_components: int,
        latent_dim: int,
        batch_size: int,
        kl_warmup: int = 5,
        kl_annealing_mode: str = "linear",
        mc_kl: int = 100,
        mmd_warmup: int = 15,
        mmd_annealing_mode: str = "linear",
        kmeans_loss: float = 0.0,
        reg_cluster_variance: bool = False,
        **kwargs,
    ):
        """Initialize the Gaussian Mixture Latent layer.

        Args:
            input_shape (tuple): shape of the input data
            n_components (int): number of components in the Gaussian mixture.
            latent_dim (int): dimensionality of the latent space.
            batch_size (int): batch size for training.
            kl_warmup (int): number of epochs to warm up the KL divergence.
            kl_annealing_mode (str): mode to use for annealing the KL divergence. Must be one of "linear" and "sigmoid".
            mc_kl (int): number of Monte Carlo samples to use for computing the KL divergence.
            mmd_warmup (int): number of epochs to warm up the MMD.
            mmd_annealing_mode (str): mode to use for annealing the MMD. Must be one of "linear" and "sigmoid".
            kmeans_loss (float): weight of the Gram matrix regularization loss.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            **kwargs: keyword arguments passed to the parent class

        """
        super(GaussianMixtureLatent, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.kl_warmup = kl_warmup
        self.kl_annealing_mode = kl_annealing_mode
        self.mc_kl = mc_kl
        self.mmd_warmup = mmd_warmup
        self.mmd_annealing_mode = mmd_annealing_mode
        self.kmeans = kmeans_loss
        self.optimizer = Nadam(learning_rate=1e-3, clipvalue=0.75)
        self.reg_cluster_variance = reg_cluster_variance
        self.pretrain = tf.Variable(0.0, name="pretrain", trainable=False)

        # Initialize GM parameters
        self.c_mu = tf.Variable(
            tf.initializers.GlorotNormal()(shape=[self.n_components, self.latent_dim]),
            name="mu_c",
        )
        self.log_c_sigma = tf.Variable(
            tf.initializers.GlorotNormal()([self.n_components, self.latent_dim]),
            name="log_sigma_c",
        )

        # Initialize the Gaussian Mixture prior with the specified number of components
        self.prior = tf.constant(tf.ones([self.n_components]) * (1 / self.n_components))

        # Initialize layers
        self.z_gauss_mean = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim) // 2,
            name="cluster_means",
            activation="linear",
            kernel_initializer="glorot_uniform",
            activity_regularizer=None,
        )
        self.z_gauss_var = Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim) // 2,
            name="cluster_variances",
            activation="softplus",
            kernel_initializer="glorot_uniform",
            activity_regularizer=tf.keras.regularizers.l1(0.1),
        )

        self.cluster_control_layer = deepof.model_utils.ClusterControl(
            batch_size=self.batch_size,
            n_components=self.n_components,
            encoding_dim=self.latent_dim,
            k=self.n_components,
        )

        # control KL weight
        self.kl_warm_up_iters = tf.cast(
            self.kl_warmup * (self.seq_shape // self.batch_size), tf.int64
        )
        self._kl_weight = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name="kl_weight"
        )

    def call(self, inputs, training=False):  # pragma: no cover
        """Compute the output of the layer."""
        z_gauss_mean = self.z_gauss_mean(inputs)
        z_gauss_var = self.z_gauss_var(inputs)

        z = tfd.MultivariateNormalDiag(
            loc=z_gauss_mean, scale_diag=tf.math.sqrt(tf.math.exp(z_gauss_var))
        )
        z_sample = tf.squeeze(z.sample())

        # Compute embedding probabilities given each cluster
        p_z_c = tf.stack(
            [
                tfd.MultivariateNormalDiag(
                    loc=self.c_mu[i, :],
                    scale_diag=tf.math.exp(self.log_c_sigma)[i, :],
                ).log_prob((z_sample if training else z_gauss_mean))
                + 1e-6
                for i in range(self.n_components)
            ],
            axis=-1,
        )

        # Update prior
        prior = self.prior

        # Compute cluster probabilitie given embedding
        z_cat = tf.math.log(prior + 1e-6) + p_z_c
        z_cat = tf.nn.log_softmax(z_cat, axis=-1)
        z_cat = tf.math.exp(z_cat)

        # Add clustering loss
        loss_clustering = -tf.reduce_sum(
            tf.multiply(z_cat, tf.math.softmax(p_z_c, axis=-1)), axis=-1
        ) * (1.0 - tf.cast(self.pretrain, tf.float32))
        loss_prior = -tf.math.reduce_sum(
            tf.math.xlogy(z_cat, 1e-6 + prior), axis=-1
        ) * (1.0 - tf.cast(self.pretrain, tf.float32))

        self.add_metric(loss_clustering, name="clustering_loss", aggregation="mean")
        self.add_metric(loss_prior, name="prior_loss", aggregation="mean")

        # Update KL weight based on the current iteration
        if self.kl_warm_up_iters > 0:
            if self.kl_annealing_mode in ["linear", "sigmoid"]:
                self._kl_weight = tf.cast(
                    tf.keras.backend.min(
                        [self.optimizer.iterations / self.kl_warm_up_iters, 1.0]
                    ),
                    tf.float32,
                )
                if self.kl_annealing_mode == "sigmoid":
                    self._kl_weight = tf.math.sigmoid(
                        (2 * self._kl_weight - 1)
                        / (self._kl_weight - self._kl_weight**2)
                    )
            else:
                raise NotImplementedError(
                    "annealing_mode must be one of 'linear' and 'sigmoid'"
                )
        else:
            self._kl_weight = tf.cast(1.0, tf.float32)

        loss_variational_1 = -1 / 2 * tf.reduce_sum(z_gauss_var + 1, axis=-1)
        loss_variational_2 = tf.math.reduce_sum(
            tf.math.xlogy(z_cat, 1e-6 + z_cat), axis=-1
        )
        kl = loss_variational_1 + loss_variational_2 * (
            1.0 - tf.cast(self.pretrain, tf.float32)
        )
        kl_batch = self._kl_weight * kl

        self.add_metric(self._kl_weight, aggregation="mean", name="kl_weight")
        self.add_metric(kl, aggregation="mean", name="kl_divergence")

        self.add_loss(tf.math.reduce_mean(loss_clustering))
        self.add_loss(tf.math.reduce_mean(loss_prior))
        self.add_loss(tf.math.reduce_mean(kl_batch))

        if training:
            z = z_sample
        else:
            # Select corresponding mean
            z = z_gauss_mean

        # Tracks clustering metrics
        if self.n_components > 1:
            z = self.cluster_control_layer([z, z_cat])

        if self.kmeans:
            kmeans_loss = deepof.model_utils.compute_kmeans_loss(
                z, weight=self.kmeans, batch_size=self.batch_size
            )
            self.add_loss(kmeans_loss)
            self.add_metric(kmeans_loss, name="kmeans_loss")

        return z, z_cat


# noinspection PyCallingNonCallable
def get_vade(
    input_shape: tuple,
    edge_feature_shape: tuple,
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    use_gnn: bool,
    n_components: int,
    batch_size: int = 64,
    kl_warmup: int = 15,
    kl_annealing_mode: str = "sigmoid",
    mc_kl: int = 100,
    kmeans_loss: float = 1.0,
    reg_cluster_variance: bool = False,
    encoder_type: str = "recurrent",
    interaction_regularization: float = 0.0,
):
    """Build a Gaussian mixture variational autoencoder (VaDE) model, adapted to the DeepOF setting.

    Args:
        input_shape (tuple): shape of the input data.
        edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
        adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
        latent_dim (int): dimensionality of the latent space.
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
        n_components (int): number of components in the Gaussian mixture.
        batch_size (int): batch size for training.
        kl_warmup (int): Number of iterations during which to warm up the KL divergence.
        kl_annealing_mode (str): mode to use for annealing the KL divergence. Must be one of "linear" and "sigmoid".
        mc_kl (int): number of Monte Carlo samples to use for computing the KL divergence.
        kmeans_loss (float): weight of the Gram matrix loss as described in deepof.model_utils.compute_kmeans_loss.
        reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
        encoder_type (str): type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
        interaction_regularization (float): weight of the interaction regularization term.

    Returns:
        encoder (tf.keras.Model): connected encoder of the VQ-VAE model. Outputs a vector of shape (latent_dim,).
        decoder (tf.keras.Model): connected decoder of the VQ-VAE model.
        grouper (tf.keras.Model): deep clustering branch of the VQ-VAE model. Outputs a vector of shape (n_components,) for each training instance, corresponding to the soft counts for each cluster.
        vade (tf.keras.Model): complete VaDE model

    """
    if encoder_type == "recurrent":
        encoder = get_recurrent_encoder(
            input_shape=input_shape[1:],
            adjacency_matrix=adjacency_matrix,
            edge_feature_shape=edge_feature_shape[1:],
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_recurrent_decoder(
            input_shape=input_shape[1:], latent_dim=latent_dim
        )

    elif encoder_type == "TCN":
        encoder = get_TCN_encoder(
            input_shape=input_shape[1:],
            adjacency_matrix=adjacency_matrix,
            edge_feature_shape=edge_feature_shape[1:],
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_TCN_decoder(input_shape=input_shape[1:], latent_dim=latent_dim)

    elif encoder_type == "transformer":
        encoder = get_transformer_encoder(
            input_shape[1:],
            edge_feature_shape=edge_feature_shape[1:],
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = get_transformer_decoder(input_shape[1:], latent_dim=latent_dim)

    latent_space = GaussianMixtureLatent(
        input_shape=input_shape[0],
        n_components=n_components,
        latent_dim=latent_dim,
        batch_size=batch_size,
        kl_warmup=kl_warmup,
        kl_annealing_mode=kl_annealing_mode,
        mc_kl=mc_kl,
        kmeans_loss=kmeans_loss,
        reg_cluster_variance=reg_cluster_variance,
        name="gaussian_mixture_latent",
    )

    # Connect encoder and latent space
    inputs = Input(input_shape[1:])
    a = tf.keras.layers.Input(edge_feature_shape[1:], name="encoder_edge_features")
    encoder_outputs = encoder([inputs, a])
    latent, categorical = latent_space(encoder_outputs)
    embedding = tf.keras.Model([inputs, a], latent, name="encoder")
    grouper = tf.keras.Model([inputs, a], categorical, name="grouper")

    # Connect decoder
    vade_outputs = decoder([embedding.outputs, inputs])

    # Instantiate fully connected model
    vade = tf.keras.Model(embedding.inputs, vade_outputs, name="VaDE")

    return embedding, decoder, grouper, vade


class Classifier(tf.keras.Model):
    """Classifier for supervised pose motif elucidation."""

    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray = None,
        use_gnn: bool = True,
        batch_size: int = 2048,
        bias_initializer: float = 0.0,
        encoder_type: str = "recurrent",
        **kwargs,
    ):
        """Initialize a classifier model.

        Args:
            input_shape (tuple): shape of the input data.
            edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
            adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
            use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
            batch_size (int): batch size for training.
            encoder_type (str): type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
            bias_initializer (float): value to initialize the bias of the last layer to (default: 0.0).

        """
        super().__init__(**kwargs)

        if encoder_type == "recurrent":
            self.encoder = get_recurrent_encoder(
                input_shape=input_shape[1:],
                adjacency_matrix=adjacency_matrix,
                edge_feature_shape=edge_feature_shape[1:],
                latent_dim=1,
                use_gnn=use_gnn,
            )
        elif encoder_type == "TCN":
            self.encoder = get_TCN_encoder(
                input_shape=input_shape[1:],
                adjacency_matrix=adjacency_matrix,
                edge_feature_shape=edge_feature_shape[1:],
                latent_dim=1,
                use_gnn=use_gnn,
            )
        elif encoder_type == "transformer":
            self.encoder = get_transformer_encoder(
                input_shape[1:],
                edge_feature_shape=edge_feature_shape[1:],
                adjacency_matrix=adjacency_matrix,
                latent_dim=1,
                use_gnn=use_gnn,
            )

        self.dense = tf.keras.layers.Dense(16, activation="relu", name="classifier")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.bias_initializer = tf.keras.initializers.Constant(bias_initializer)
        self.clf = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="classifier",
            bias_initializer=self.bias_initializer,
        )

    def call(self, inputs, training=None, mask=None):
        """Apply a forward pass of the classifier.

        Args:
            - inputs (tf.Tensor): input data.
            - training (bool): whether the model is in training mode.
            - mask (tf.Tensor): mask for the input data.
        """
        x = self.encoder(inputs)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.clf(x)

        return x


# noinspection PyDefaultArgument,PyCallingNonCallable
class VaDE(tf.keras.models.Model):
    """Gaussian Mixture Variational Autoencoder for pose motif elucidation."""

    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray = None,
        latent_dim: int = 8,
        use_gnn: bool = True,
        n_components: int = 15,
        batch_size: int = 64,
        kl_annealing_mode: str = "linear",
        kl_warmup_epochs: int = 15,
        montecarlo_kl: int = 100,
        kmeans_loss: float = 1.0,
        reg_cat_clusters: float = 1.0,
        reg_cluster_variance: bool = False,
        encoder_type: str = "recurrent",
        interaction_regularization: float = 0.0,
        **kwargs,
    ):
        """Init a VaDE model.

        Args:
            input_shape (tuple): Shape of the input to the full model.
            edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
            adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
            batch_size (int): Batch size for training.
            latent_dim (int): Dimensionality of the latent space.
            use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
            kl_annealing_mode (str): Annealing mode for KL annealing. Can be one of 'linear' and 'sigmoid'.
            kl_warmup_epochs (int): Number of epochs to warmup KL annealing.
            montecarlo_kl (int): Number of Monte Carlo samples for KL divergence.
            n_components (int): Number of mixture components in the latent space.
            kmeans_loss (float): weight of the gram matrix regularization loss.
            reg_cat_clusters (bool): whether to use the penalized uneven cluster membership in the latent space, by minimizing the KL divergence between cluster membership and a uniform categorical distribution.
            reg_cluster_variance (bool): whether to penalize uneven cluster variances in the latent space.
            encoder_type (str): type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
            interaction_regularization (float): Regularization parameter for the interaction features.
            **kwargs: Additional keyword arguments.

        """
        super(VaDE, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.edge_feature_shape = edge_feature_shape
        self.adjacency_matrix = adjacency_matrix
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        self.kl_annealing_mode = kl_annealing_mode
        self.kl_warmup = kl_warmup_epochs
        self.mc_kl = montecarlo_kl
        self.n_components = n_components
        self.optimizer = Nadam(learning_rate=1e-3, clipvalue=0.75)
        self.kmeans = kmeans_loss
        self.reg_cat_clusters = reg_cat_clusters
        self.reg_cluster_variance = reg_cluster_variance
        self.encoder_type = encoder_type
        self.interaction_regularization = interaction_regularization

        # Define VaDE model
        self.encoder, self.decoder, self.grouper, self.vade = get_vade(
            input_shape=self.seq_shape,
            edge_feature_shape=self.edge_feature_shape,
            adjacency_matrix=self.adjacency_matrix,
            n_components=self.n_components,
            latent_dim=self.latent_dim,
            use_gnn=use_gnn,
            batch_size=self.batch_size,
            kl_warmup=self.kl_warmup,
            kl_annealing_mode=self.kl_annealing_mode,
            mc_kl=self.mc_kl,
            kmeans_loss=self.kmeans,
            reg_cluster_variance=self.reg_cluster_variance,
            encoder_type=self.encoder_type,
            interaction_regularization=self.interaction_regularization,
        )

        # Propagate the optimizer to all relevant sub-models, to enable metric annealing
        self.vade.optimizer = self.optimizer
        self.vade.get_layer("gaussian_mixture_latent").optimizer = self.optimizer

        # Define metrics to track

        # Track all loss function components
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_reconstruction_loss"
        )

        if self.reg_cat_clusters:
            self.cat_cluster_loss_tracker = tf.keras.metrics.Mean(
                name="cat_cluster_loss"
            )
            self.val_cat_cluster_loss_tracker = tf.keras.metrics.Mean(
                name="val_cat_cluster_loss"
            )

    @property
    def metrics(self):  # pragma: no cover
        """Initializes tracked metrics of VaDE model."""
        metrics = [
            self.total_loss_tracker,
            self.val_total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.val_reconstruction_loss_tracker,
        ]

        if self.reg_cat_clusters:
            metrics += [
                self.cat_cluster_loss_tracker,
                self.val_cat_cluster_loss_tracker,
            ]

        return metrics

    @property
    def get_gmm_params(self):
        """Return the GMM parameters of the model."""
        # Get GMM parameters
        return {
            "means": self.grouper.get_layer("gaussian_mixture_latent").c_mu,
            "sigmas": tf.math.exp(
                self.grouper.get_layer("gaussian_mixture_latent").log_c_sigma
            ),
            "weights": tf.math.softmax(
                self.grouper.get_layer("gaussian_mixture_latent").prior
            ),
        }

    def set_pretrain_mode(self, switch):
        """Set the pretrain mode of the model."""
        self.grouper.get_layer("gaussian_mixture_latent").pretrain.assign(switch)

    def pretrain(
        self,
        data,
        embed_x,
        embed_a,
        epochs=10,
        samples=10000,
        gmm_initialize=True,
        **kwargs,
    ):
        """Run a GMM directed pretraining of the encoder, to minimize the likelihood of getting stuck in a local minimum."""
        # Turn on pretrain mode
        self.set_pretrain_mode(1.0)

        # pre-train
        self.fit(
            data,
            epochs=epochs,
            **kwargs,
        )

        # Turn off pretrain mode
        self.set_pretrain_mode(0.0)

        if gmm_initialize:

            with tf.device("CPU"):
                # Get embedding samples
                emb_idx = np.random.choice(range(embed_x.shape[0]), samples)

                # map to latent
                z = self.encoder([embed_x[emb_idx], embed_a[emb_idx]])
                # fit GMM
                gmm = GaussianMixture(
                    n_components=self.n_components,
                    covariance_type="diag",
                    reg_covar=1e-04,
                    **kwargs,
                ).fit(z)
                # get GMM parameters
                mu = gmm.means_
                sigma2 = gmm.covariances_

            # initialize mixture components
            self.grouper.get_layer("gaussian_mixture_latent").c_mu.assign(
                tf.convert_to_tensor(value=mu, dtype=tf.float32)
            )
            self.grouper.get_layer("gaussian_mixture_latent").log_c_sigma.assign(
                tf.math.log(
                    tf.math.sqrt(tf.convert_to_tensor(value=sigma2, dtype=tf.float32))
                )
            )

    @tf.function
    def call(self, inputs, **kwargs):
        """Call the VaDE model."""
        return self.vade(inputs, **kwargs)

    def train_step(self, data):  # pragma: no cover
        """Perform a training step."""
        # Unpack data, repacking labels into a generator
        x, a, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        with tf.GradientTape() as tape:

            # Get outputs from the full model
            outputs = self.vade([x, a], training=True)

            # Get rid of the attention scores that the transformer decoder outputs
            if self.encoder_type == "transformer":
                outputs = outputs[0]

            if isinstance(outputs, list):
                reconstructions = outputs[0]
            else:
                reconstructions = outputs

            # Regularize embeddings
            # groups = self.grouper(x, training=True)

            # Compute losses
            seq_inputs = next(y)
            total_loss = sum(self.vade.losses)

            # Add a regularization term to the soft_counts, to prevent the embedding layer from
            # collapsing into a few clusters.
            if self.reg_cat_clusters:

                soft_counts = self.grouper([x, a], training=True)
                soft_counts_regulrization = (
                    self.reg_cat_clusters
                    * deepof.model_utils.cluster_frequencies_regularizer(
                        soft_counts=soft_counts, k=self.n_components
                    )
                )
                total_loss += soft_counts_regulrization

            # Compute reconstruction loss
            reconstruction_loss = -tf.reduce_mean(reconstructions.log_prob(seq_inputs))
            total_loss += reconstruction_loss

        # Backpropagation
        grads = tape.gradient(total_loss, self.vade.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vade.trainable_variables))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }

        if self.reg_cat_clusters:
            self.cat_cluster_loss_tracker.update_state(soft_counts_regulrization)
            log_dict["cat_cluster_loss"] = self.cat_cluster_loss_tracker.result()

        # Log to TensorBoard, both explicitly and implicitly (within model) tracked metrics
        return {**log_dict, **{met.name: met.result() for met in self.vade.metrics}}

    # noinspection PyUnboundLocalVariable
    @tf.function
    def test_step(self, data):  # pragma: no cover
        """Performs a test step."""
        # Unpack data, repacking labels into a generator
        x, a, y = data
        if not isinstance(y, tuple):
            y = [y]
        y = (labels for labels in y)

        # Get outputs from the full model
        outputs = self.vade([x, a], training=False)

        # Get rid of the attention scores that the transformer decoder outputs
        if self.encoder_type == "transformer":
            outputs = outputs[0]

        if isinstance(outputs, list):
            reconstructions = outputs[0]
        else:
            reconstructions = outputs

        # Compute losses
        seq_inputs = next(y)
        total_loss = sum(self.vade.losses)

        # Add a regularization term to the soft_counts, to prevent the embedding layer from
        # collapsing into a few clusters.
        if self.reg_cat_clusters:
            soft_counts = self.grouper([x, a], training=False)
            soft_counts_regulrization = (
                self.reg_cat_clusters
                * deepof.model_utils.cluster_frequencies_regularizer(
                    soft_counts=soft_counts, k=self.n_components
                )
            )
            total_loss += soft_counts_regulrization

        # Compute reconstruction loss
        reconstruction_loss = -tf.reduce_mean(reconstructions.log_prob(seq_inputs))
        total_loss += reconstruction_loss

        # Track losses
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)

        # Log results (coupled with TensorBoard)
        log_dict = {
            "total_loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
        }

        if self.reg_cat_clusters:
            self.val_cat_cluster_loss_tracker.update_state(soft_counts_regulrization)
            log_dict["cat_cluster_loss"] = self.val_cat_cluster_loss_tracker.result()

        return {**log_dict, **{met.name: met.result() for met in self.vade.metrics}}


# noinspection PyDefaultArgument,PyCallingNonCallable
class Contrastive(tf.keras.models.Model):
    """Self-supervised contrastive embeddings."""

    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray = None,
        encoder_type: str = "TCN",
        latent_dim: int = 8,
        use_gnn: bool = True,
        temperature: float = 0.1,
        similarity_function: str = "cosine",
        loss_function: str = "nce",
        beta: float = 0.1,
        tau: float = 0.1,
        interaction_regularization: float = 0.0,
        **kwargs,
    ):
        """Init a self-supervised Contrastive embedding model.

        Args:
            input_shape (tuple): Shape of the input to the full model.
            edge_feature_shape (tuple): shape of the edge feature matrix used for graph representations.
            adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
            encoder_type (str): type of encoder to use. Can be set to "recurrent" (default), "TCN", or "transformer".
            latent_dim (int): Dimensionality of the latent space.
            use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
            temperature (float):
            similarity_function (str):
            loss_function (str):
            beta (float):
            tau (float):
            interaction_regularization (float): Regularization parameter for the interaction features.
            **kwargs: Additional keyword arguments.

        """
        super(Contrastive, self).__init__(**kwargs)
        self.seq_shape = input_shape
        self.edge_feature_shape = edge_feature_shape
        self.adjacency_matrix = adjacency_matrix
        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        self.window_length = self.seq_shape[1] // 2
        self.temperature = temperature
        self.similarity_function = similarity_function
        self.loss_function = loss_function
        self.beta = beta
        self.tau = tau
        self.optimizer = Nadam(learning_rate=1e-3, clipvalue=0.75)
        self.encoder_type = encoder_type
        self.interaction_regularization = interaction_regularization

        # Define Contrastive model
        if encoder_type == "recurrent":

            self.encoder = get_recurrent_encoder(
                input_shape=(self.window_length, input_shape[-1]),
                edge_feature_shape=(
                    self.window_length,
                    self.edge_feature_shape[2],
                ),
                adjacency_matrix=self.adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )

        elif encoder_type == "TCN":
            self.encoder = get_TCN_encoder(
                input_shape=(self.window_length, input_shape[-1]),
                edge_feature_shape=(
                    self.window_length,
                    self.edge_feature_shape[2],
                ),
                adjacency_matrix=self.adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )

        elif encoder_type == "transformer":

            self.encoder = get_transformer_encoder(
                (self.window_length, input_shape[-1]),
                edge_feature_shape=(
                    self.window_length,
                    self.edge_feature_shape[2],
                ),
                adjacency_matrix=self.adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )

        # Define metrics to track

        # Track all loss function components
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        self.mean_sim_tracker = tf.keras.metrics.Mean(name="pos_similarity")
        self.val_mean_sim_tracker = tf.keras.metrics.Mean(name="val_pos_similarity")
        self.neg_sim_tracker = tf.keras.metrics.Mean(name="neg_similarity")
        self.val_neg_sim_tracker = tf.keras.metrics.Mean(name="val_neg_similarity")

    @property
    def metrics(self):  # pragma: no cover
        """Initializes tracked metrics of the contrastive model."""
        metrics = [
            self.total_loss_tracker,
            self.val_total_loss_tracker,
            self.mean_sim_tracker,
            self.val_mean_sim_tracker,
            self.neg_sim_tracker,
            self.val_neg_sim_tracker,
        ]

        return metrics

    @tf.function
    def call(self, inputs, **kwargs):
        """Call the contrastive model."""
        return self.encoder(inputs, **kwargs)

    def train_step(self, data):  # pragma: no cover
        """Perform a training step."""
        # Unpack data
        x, a, y = data
        if not isinstance(y, tuple):
            y = [
                y
            ]  # Labels won't be used for now, but may come handy if exploring regularizers in the future

        with tf.GradientTape() as tape:

            # Get positive and negative pairs
            def ts_samples(mbatch, win):
                x = mbatch[:, 1 : win + 1]
                y = mbatch[:, -win:]

                return x, y

            pos, neg = ts_samples(x, self.window_length)
            pos_a, neg_a = ts_samples(a, self.window_length)

            # Compute contrastive loss
            enc_pos = self.encoder([pos, pos_a], training=True)
            enc_neg = self.encoder([neg, neg_a], training=True)

            # normalize projection feature vectors
            enc_pos = tf.math.l2_normalize(enc_pos, axis=1)
            enc_neg = tf.math.l2_normalize(enc_neg, axis=1)

            # loss, mean_sim = ls.dcl_loss_fn(zis, zjs, temperature, lfn)
            (
                contrastive_loss,
                mean_sim,
                neg_sim,
            ) = deepof.model_utils.select_contrastive_loss(
                enc_pos,
                enc_neg,
                similarity=self.similarity_function,
                loss_fn=self.loss_function,
                temperature=self.temperature,
                tau=self.tau,
                beta=self.beta,
                elimination_topk=0.1,
                attraction=False,
            )

            total_loss = contrastive_loss

        # Backpropagation
        grads = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        # Track losses
        self.total_loss_tracker.update_state(total_loss)
        self.mean_sim_tracker.update_state(mean_sim)
        self.neg_sim_tracker.update_state(neg_sim)

        # Log to TensorBoard, both explicitly and implicitly (within model) tracked metrics
        return {met.name: met.result() for met in self.metrics if "val" not in met.name}

    # noinspection PyUnboundLocalVariable
    @tf.function
    def test_step(self, data):  # pragma: no cover
        """Performs a test step."""
        # Unpack data
        x, a, y = data
        if not isinstance(y, tuple):
            y = [
                y
            ]  # Labels won't be used for now, but may come handy if exploring regularizers in the future

        # Get positive and negative pairs
        def ts_samples(mbatch, win):
            x = mbatch[:, 1 : win + 1]
            y = mbatch[:, -win:]

            return x, y

        pos, neg = ts_samples(x, self.window_length)
        pos_a, neg_a = ts_samples(a, self.window_length)

        # Compute contrastive loss
        enc_pos = self.encoder([pos, pos_a], training=False)
        enc_neg = self.encoder([neg, neg_a], training=False)

        # normalize projection feature vectors
        enc_pos = tf.math.l2_normalize(enc_pos, axis=1)
        enc_neg = tf.math.l2_normalize(enc_neg, axis=1)

        # loss, mean_sim = ls.dcl_loss_fn(zis, zjs, temperature, lfn)
        (
            contrastive_loss,
            mean_sim,
            neg_sim,
        ) = deepof.model_utils.select_contrastive_loss(
            enc_pos,
            enc_neg,
            similarity=self.similarity_function,
            loss_fn=self.loss_function,
            temperature=self.temperature,
            tau=self.tau,
            beta=self.beta,
            elimination_topk=0.1,
            attraction=False,
        )

        total_loss = contrastive_loss

        # Track losses
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_mean_sim_tracker.update_state(mean_sim)
        self.val_neg_sim_tracker.update_state(neg_sim)

        # Log to TensorBoard, both explicitly and implicitly (within model) tracked metrics
        return {
            met.name.replace("val_", ""): met.result()
            for met in self.metrics
            if "val" in met.name
        }

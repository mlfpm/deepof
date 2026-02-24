"""deep autoencoder models for unsupervised pose detection.

- VQ-VAE: a variational autoencoder with a vector quantization latent-space (https://arxiv.org/abs/1711.00937).
- VaDE: a variational autoencoder with a Gaussian mixture latent-space.
- Contrastive: an embedding model consisting of a single encoder, trained using a contrastive loss.

"""
# @author lucasmiranda42
# encoding: utf-8
# module deepof

from typing import Any, NewType, Iterable, Tuple, Dict, Optional, Mapping, List, Callable

import os
import numpy as np
import math
import tcn
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from spektral.layers import CensNetConv
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    Dense,
    LayerNormalization,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Nadam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Distribution, TransformedDistribution, Normal
from torch.distributions.transforms import AffineTransform

import deepof.model_utils
import deepof.clustering.model_utils_new
from deepof.clustering.censNetConv_pt import CensNetConvPT
import deepof.utils
from deepof.data_loading import get_dt
import warnings
from deepof.clustering.model_utils_new import ProbabilisticDecoderPT


tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


class RecurrentEncoderPT(nn.Module):
    """
    PyTorch translation of the TF recurrent encoder.

    Expected shapes:
      - use_gnn=True:
          x: (B, T, N_nodes, F_per_node)
          a: (B, T, E_edges, F_per_edge)

          Internally:
            - TF-style grouping reshape -> (B, N_nodes, T, F_per_node) / (B, E_edges, T, F_per_edge)
            - Recurrent blocks -> (B, N_nodes/E_edges, 2*latent_dim)
            - CensNetConvPT -> (B, N_nodes/E_edges, latent_dim)
            - ReLU (to match TF activation='relu')
            - Flatten+concat -> Linear(latent_dim)

      - use_gnn=False:
          x: (B, T, N_nodes, F_per_node) -> flatten to (B, 1, T, N_nodes*F_per_node)
          recurrent block -> (B, 1, 2*latent_dim) -> squeeze -> Linear(latent_dim)
    """

    def __init__(
        self,
        input_shape: tuple,            # (T, N_nodes, F_per_node)
        edge_feature_shape: tuple,     # (T, E_edges, F_per_edge)
        adjacency_matrix: np.ndarray,  # (N_nodes, N_nodes)
        latent_dim: int,
        use_gnn: bool = True,
        interaction_regularization: float = 0.0,  # not used in PT, kept for API parity
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.latent_dim = latent_dim

        self.num_nodes = int(adjacency_matrix.shape[0])
        self.num_edges = int(edge_feature_shape[1]) if use_gnn else 0

        if self.use_gnn:
            # Node stream
            node_feat_per_node = int(input_shape[-1])
            self.node_recurrent_block = deepof.clustering.model_utils_new.RecurrentBlockPT(
                input_features=node_feat_per_node,
                latent_dim=latent_dim,
            )

            # Edge stream
            edge_feat_per_edge = int(edge_feature_shape[-1])
            self.edge_recurrent_block = deepof.clustering.model_utils_new.RecurrentBlockPT(
                input_features=edge_feat_per_edge,
                latent_dim=latent_dim,
            )

            # GNN block
            self.spatial_gnn_block = CensNetConvPT(
                node_channels=latent_dim,
                edge_channels=latent_dim,
                activation='relu',
            )
            # Build with expected feature dims (2*latent on each stream)
            self.spatial_gnn_block._build(
                node_features_shape=[None, None, 2 * latent_dim],
                edge_features_shape=[None, None, 2 * latent_dim],
            )

            # Preprocess graph operators once and buffer them
            lap, edge_lap, inc = self.spatial_gnn_block.preprocess(torch.tensor(adjacency_matrix))
            self.register_buffer("laplacian", lap.float())
            self.register_buffer("edge_laplacian", edge_lap.float())
            self.register_buffer("incidence", inc.float())

            # Final projection after concatenating node+edge embeddings
            final_in = (self.num_nodes * latent_dim) + (self.num_edges * latent_dim)
            self.final_dense = nn.Linear(final_in, latent_dim)

        else:
            # Non-GNN: single recurrent block over a single "group" dimension
            in_features = int(input_shape[1]) * int(input_shape[2])  # N_nodes * F_per_node
            self.recurrent_block = deepof.clustering.model_utils_new.RecurrentBlockPT(
                input_features=in_features,
                latent_dim=latent_dim,
            )
            self.final_dense = nn.Linear(2 * latent_dim, latent_dim)

    @staticmethod
    def tf_style_group_reshape(x: torch.Tensor, groups: int, feat_per_group: int) -> torch.Tensor:
        """
        Exact TF mapping used in the encoder for the GNN path:
          x: (B, T, groups, feat_per_group)
          -> (B, groups, T, feat_per_group)
        Derived from the TF sequence: transpose -> reshape -> transpose.
        """
        B, T, G, F = x.shape
        assert G == groups and F == feat_per_group
        # (B, T, G*F)
        flat = x.reshape(B, T, G * F)
        # (G*F, T, B)
        tmp = flat.permute(2, 1, 0)
        # (F, T, G, B)
        tmp = tmp.reshape(F, T, G, B)
        # (B, G, T, F)
        out = tmp.permute(3, 2, 1, 0).contiguous()
        return out

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_nodes, F_per_node)
        a: (B, T, E_edges, F_per_edge)
        """
        B, T, N_nodes, F_per_node = x.shape
        _, _, E_edges, F_per_edge = a.shape

        if self.use_gnn:
            # TF-style reshape into (B, groups, T, features)
            x_reshaped = self.tf_style_group_reshape(x, self.num_nodes, F_per_node)  # (B, N, T, F)
            a_reshaped = self.tf_style_group_reshape(a, self.num_edges, F_per_edge)  # (B, E, T, F)

            # Recurrent blocks -> (B, N/E, 2*latent)
            node_output = self.node_recurrent_block(x_reshaped)
            edge_output = self.edge_recurrent_block(a_reshaped)

            # GNN
            adj_tuple = (self.laplacian, self.edge_laplacian, self.incidence)
            x_nodes, x_edges = self.spatial_gnn_block(
                [node_output, adj_tuple, edge_output]
            )  # (B, N/E, latent)

            # Activation to match Spektral's activation='relu'
            # Apply here only if not already applied inside CensNetConvPT.
            if getattr(self.spatial_gnn_block, "activation", None) == "relu":
                x_nodes = F.relu(x_nodes)
                x_edges = F.relu(x_edges)

            # Flatten and concat
            x_nodes_flat = x_nodes.view(B, -1)
            x_edges_flat = x_edges.view(B, -1)
            encoder = torch.cat([x_nodes_flat, x_edges_flat], dim=-1)

            if torch.isnan(encoder).any():
                print("z issues!")

        else:
            # Flatten nodes/features into a single feature dim and keep a single group
            x_flat = x.view(B, T, N_nodes * F_per_node)  # (B, T, N*F)
            x_grouped = x_flat.unsqueeze(1)              # (B, 1, T, N*F)
            encoder = self.recurrent_block(x_grouped).squeeze(1)  # (B, 2*latent)

        # Final projection
        return self.final_dense(encoder)
        

# noinspection PyCallingNonCallable
@deepof.data_loading._suppress_warning(
    warn_messages=[
        "The initializer GlorotUniform is unseeded and being called multiple times, which will return identical values each time"
    ]
)
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
        x_flat = tf.reshape(x, [-1, input_shape[0], input_shape[1] * input_shape[2]])
        x_reshaped = tf.expand_dims(x_flat, axis=1)

    # Instantiate temporal RNN block
    encoder = deepof.clustering.model_utils_new.get_recurrent_block(
        x_reshaped, latent_dim, gru_unroll, bidirectional_merge
    )(x_reshaped)


    # Instantiate spatial graph block
    if use_gnn:

        # Embed edge features too
        a_encoder = deepof.clustering.model_utils_new.get_recurrent_block(
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


class RecurrentDecoderPT(nn.Module):
    """
    A full PyTorch implementation of the recurrent decoder.
    """
    def __init__(self, output_shape: tuple, latent_dim: int, bidirectional_merge: str = "concat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        if bidirectional_merge != "concat":
            warnings.warn("Bidirectional merge mode is fixed to 'concat' to correspond with original TensorFlow implementation.")

        # First Bi-GRU layer
        self.gru1 = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm1 = nn.LayerNorm(2 * latent_dim, eps=1e-3)

        # Second Bi-GRU layer
        self.gru2 = nn.GRU(
            input_size=2 * latent_dim, # Input from first Bi-GRU
            hidden_size=2 * latent_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm2 = nn.LayerNorm(4 * latent_dim, eps=1e-3) # Output of second Bi-GRU is 2 * (2*latent_dim)

        # Convolutional Layer
        self.conv1d = nn.Conv1d(
            in_channels=4 * latent_dim, # Input from second norm layer
            out_channels=2 * latent_dim,
            kernel_size=5,
            padding="same",
            bias=False
        )
        self.norm3 = nn.LayerNorm(2 * latent_dim, eps=1e-3) # Output of Conv1D

        # Probabilistic Layer 
        self.prob_decoder = ProbabilisticDecoderPT(
            hidden_dim=2 * latent_dim, # Input from third norm layer
            data_dim=output_shape[1]
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> TransformedDistribution:
        B, T, _ = x.shape

        # Validity mask and lengths
        validity_mask = ~torch.all(x == 0.0, dim=2)
        lengths_cpu = validity_mask.sum(dim=1).to(torch.int64).cpu()
        valid_idx_cpu = torch.where(lengths_cpu > 0)[0]

        # Force data onto this module's device
        dev = self.gru1.weight_ih_l0.device
        g = g.to(dev, non_blocking=True)
        x = x.to(dev, non_blocking=True)
        valid_idx = valid_idx_cpu.to(dev)

        # 1) RepeatVector
        generator = g.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

        # 2) First Bi-GRU with masking
        gru1_out_full = torch.zeros(B, T, 2 * self.latent_dim, device=generator.device, dtype=generator.dtype)
        if valid_idx.numel() > 0:
            packed_input_1 = pack_padded_sequence(
                generator[valid_idx], lengths_cpu[valid_idx_cpu], batch_first=True, enforce_sorted=False
            )
            packed_output_1, _ = self.gru1(packed_input_1)
            unpacked_output_1, _ = pad_packed_sequence(packed_output_1, batch_first=True, total_length=T)
            gru1_out_full[valid_idx] = unpacked_output_1.to(gru1_out_full.dtype)
        norm1_out = self.norm1(gru1_out_full)

        # 3) Second Bi-GRU with masking
        gru2_out_full = torch.zeros(B, T, 4 * self.latent_dim, device=norm1_out.device, dtype=norm1_out.dtype)
        if valid_idx.numel() > 0:
            packed_input_2 = pack_padded_sequence(
                norm1_out[valid_idx], lengths_cpu[valid_idx_cpu], batch_first=True, enforce_sorted=False
            )
            packed_output_2, _ = self.gru2(packed_input_2)
            unpacked_output_2, _ = pad_packed_sequence(packed_output_2, batch_first=True, total_length=T)
            gru2_out_full[valid_idx] = unpacked_output_2.to(gru2_out_full.dtype)
        norm2_out = self.norm2(gru2_out_full)

        # 4) Conv + Norm
        conv_in = norm2_out.permute(0, 2, 1)  # (B, C, T)
        conv_out = F.relu(self.conv1d(conv_in))
        norm3_in = conv_out.permute(0, 2, 1)
        norm3_out = self.norm3(norm3_in)

        # 5) Probabilistic output
        final_dist = self.prob_decoder(norm3_out, validity_mask.to(dev))
        return final_dist
    

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


def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name in {"linear", "identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class TemporalBlockPT(nn.Module):
    """
    Residual TCN block compatible with keras-tcn:
      - Conv1d -> BN(eps=1e-3) -> Act -> Drop
      - Conv1d -> BN(eps=1e-3) -> Act -> Drop
      - Residual add (with 1x1 projection if channels differ) -> Act
    Returns:
      out: post-residual activation
      skip: post-second-conv activation (summed across blocks when skip connections are used)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: str = "causal",
        dropout_rate: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        conv_init_std: float = 0.05,
    ):
        super().__init__()
        assert padding in {"causal", "same"}
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding
        self.act = _act(activation)
        self.use_batch_norm = use_batch_norm

        pad = lambda: ((self.kernel_size - 1) * self.dilation) // 2 if padding == "same" else 0

        self.conv1 = nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=self.dilation, padding=pad(), bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-3) if use_batch_norm else nn.Identity()
        self.drop1 = nn.Dropout(float(dropout_rate)) if dropout_rate else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, self.kernel_size, dilation=self.dilation, padding=pad(), bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-3) if use_batch_norm else nn.Identity()
        self.drop2 = nn.Dropout(float(dropout_rate)) if dropout_rate else nn.Identity()

        # 1x1 residual projection if channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True) if in_channels != out_channels else None

        # Init similar to keras random_normal
        nn.init.normal_(self.conv1.weight, mean=0.0, std=conv_init_std); nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=conv_init_std); nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=conv_init_std); nn.init.zeros_(self.downsample.bias)

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        return F.pad(x, (pad, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C_in, T)
        y = self._causal_pad(x) if self.padding_mode == "causal" else x
        y = self.drop1(self.act(self.bn1(self.conv1(y))))

        y = self._causal_pad(y) if self.padding_mode == "causal" else y
        y = self.drop2(self.act(self.bn2(self.conv2(y))))

        skip = y  # per-block skip is the post-second-activation output

        res = x if self.downsample is None else self.downsample(x)
        out = self.act(y + res)
        return out, skip  # both (B, C_out, T)


class TCN1DPT(nn.Module):
    """
    Temporal Convolutional Network over sequences (B, T, C_in).
    - When use_skip_connections=True: sum per-block skip outputs, then apply a final activation.
    - Otherwise: use the last block’s residual output.
    - return_sequences=False: returns last timestep features (B, C_out).
    """
    def __init__(
        self,
        in_channels: int,
        conv_filters: int = 32,
        kernel_size: int = 4,
        conv_stacks: int = 2,
        conv_dilations: Iterable[int] = (1, 2, 4, 8),
        padding: str = "causal",
        use_skip_connections: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        return_sequences: bool = False,
    ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.final_act = _act(activation)

        blocks = []
        c_in = in_channels
        for _ in range(int(conv_stacks)):
            for d in tuple(conv_dilations):
                blocks.append(
                    TemporalBlockPT(
                        in_channels=c_in,
                        out_channels=conv_filters,
                        kernel_size=kernel_size,
                        dilation=int(d),
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        use_batch_norm=use_batch_norm,
                    )
                )
                c_in = conv_filters
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in) -> Conv1d expects (B, C_in, T)
        dtype_in = x.dtype
        y = x.transpose(1, 2).float() # force fp32 compute
        skip_sum, last_out = None, None

        for blk in self.blocks:
            y, skip = blk(y)
            last_out = y
            if self.use_skip_connections:
                skip_sum = skip if skip_sum is None else (skip_sum + skip)

        out = skip_sum if self.use_skip_connections else last_out  # (B, C, T)
        out = self.final_act(out).transpose(1, 2)
        return out.to(dtype_in) if self.return_sequences else out[:, -1, :].to(dtype_in)

class BatchNorm1dKerasFP32(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True):
        # momentum=0.01 here matches Keras momentum=0.99 semantics
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute BN in float32 for stability; cast back to input dtype
        y = super().forward(x.float())
        return y.to(dtype=x.dtype)
    
class TCNEncoderPT(nn.Module):
    """
    PyTorch port of the TF get_TCN_encoder with matching behavior:
      - Inputs:
          x: (B, W, N, NF)   node features
          a: (B, W, E, EF)   edge features
      - use_gnn=True:
          TimeDistributed(TCN) over nodes/edges -> (B, N, C) and (B, E, C)
          CensNetConvPT([node, (lap, edge_lap, inc), edge]) -> (B, N, latent), (B, E, latent)
          Flatten and MLP head
      - use_gnn=False:
          Flatten nodes+features -> TCN -> MLP head

      Parity details:
        - keras-tcn-compatible skip semantics and activation placement
        - BN eps=1e-3 everywhere
        - 'causal' and 'same' paddings supported
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],        # (W, N, NF)
        edge_feature_shape: Tuple[int, int, int], # (W, E, EF)
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        use_gnn: bool = True,
        conv_filters: int = 32,
        kernel_size: int = 4,
        conv_stacks: int = 2,
        conv_dilations: Iterable[int] = (1, 2, 4, 8),
        padding: str = "causal",
        use_skip_connections: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        interaction_regularization: float = 0.0,  # not used explicitly in PT
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.latent_dim = int(latent_dim)
        self.conv_filters = int(conv_filters)

        W, N, F_node = input_shape
        _, E, F_edge = edge_feature_shape
        assert adjacency_matrix.shape[0] == N == adjacency_matrix.shape[1], "Adjacency must be NxN and match input nodes."

        tcn_cfg = dict(
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            conv_stacks=conv_stacks,
            conv_dilations=tuple(conv_dilations),
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_rate=float(dropout_rate),
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_sequences=False,
        )

        if use_gnn:
            # Per-node and per-edge TCNs
            self.node_tcn = TCN1DPT(in_channels=F_node, **tcn_cfg)
            self.edge_tcn = TCN1DPT(in_channels=F_edge, **tcn_cfg)

            # Graph block and buffers
            self.spatial_gnn_block = CensNetConvPT(node_channels=latent_dim, edge_channels=latent_dim, activation="relu")
            lap, edge_lap, inc = self.spatial_gnn_block.preprocess(torch.tensor(adjacency_matrix))
            self.register_buffer("laplacian", lap.float())
            self.register_buffer("edge_laplacian", edge_lap.float())
            self.register_buffer("incidence", inc.float())

            final_in = (N * latent_dim) + (E * latent_dim)
        else:
            # Single TCN over flattened node features
            self.flat_tcn = TCN1DPT(in_channels=N * F_node, **tcn_cfg)
            final_in = conv_filters

        # Head MLP: Dense(2*latent) -> BN -> Dense(latent) -> BN -> Dense(latent)
        self.head = nn.Sequential(
            nn.Linear(final_in, 2 * latent_dim),
            nn.ReLU(),
            BatchNorm1dKerasFP32(2 * latent_dim, eps=1e-3),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            BatchNorm1dKerasFP32(latent_dim, eps=1e-3),
            nn.Linear(latent_dim, latent_dim),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, N, NF)  a: (B, W, E, EF)  -> returns (B, latent_dim)
        """
        B, W, N, F_node = x.shape
        _, _, E, F_edge = a.shape

        if self.use_gnn:
            # Nodes: TF-style reshape pipeline to match memory layout exactly
            x_3d = x.view(B, W, N * F_node)          # (B, W, N*F)
            x_t = x_3d.permute(2, 1, 0)              # (N*F, W, B)
            x_reshaped_t = x_t.reshape(F_node, W, N, B)
            x_nodes = x_reshaped_t.permute(3, 2, 1, 0)  # (B, N, W, F)

            node_in = x_nodes.reshape(B * N, W, F_node)
            node_out = self.node_tcn(node_in).view(B, N, self.conv_filters)  # (B, N, C)

            # Edges: TF-style reshape pipeline to match memory layout exactly
            a_3d = a.view(B, W, E * F_edge)          # (B, W, E*F_edge)
            a_t = a_3d.permute(2, 1, 0)              # (E*F_edge, W, B)
            a_reshaped_t = a_t.reshape(F_edge, W, E, B)
            a_edges = a_reshaped_t.permute(3, 2, 1, 0)  # (B, E, W, F_edge)

            edge_in = a_edges.reshape(B * E, W, F_edge)
            edge_out = self.edge_tcn(edge_in).view(B, E, self.conv_filters)  # (B, E, C)

            # Graph block
            adj_tuple = (self.laplacian, self.edge_laplacian, self.incidence)
            x_nodes_g, x_edges_g = self.spatial_gnn_block([node_out, adj_tuple, edge_out])
            x_nodes_g = F.relu(x_nodes_g)
            x_edges_g = F.relu(x_edges_g)

            enc = torch.cat([x_nodes_g.reshape(B, -1), x_edges_g.reshape(B, -1)], dim=-1)
        else:
            # Non-GNN unchanged
            x_flat = x.view(B, W, N * F_node)        # (B, W, N*NF)
            enc = self.flat_tcn(x_flat)              # (B, C)

        head_in = enc.float()
        
        # Per-sample RMS normalization to control scale
        rms = head_in.pow(2).mean(dim=1, keepdim=True).sqrt()  # (B, 1)
        head_in = head_in / rms.clamp(min=1.0)  # avoid division by tiny values

        # Clamp extreme outliers to keep BN/Linear in a sane range
        head_in = head_in.clamp(min=-1e4, max=1e4)

        # As a final guard: replace any NaN/Inf just in case
        head_in = torch.nan_to_num(head_in, nan=0.0, posinf=1e4, neginf=-1e4)
        head_out = self.head(head_in)
        return head_out


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
        latent_dim: dimensionality of the latent space
        use_gnn (bool): If True, the encoder uses a graph representation of the input, with coordinates and speeds as node attributes, and distances as edge attributes. If False, a regular 3D tensor is used as input.
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


class TCNDecoderPT(nn.Module):
    """
    PyTorch port of TF get_TCN_decoder:
      - g: (B, latent_dim)
      - x: (B, W, NNF) or (B, W, N, NF) for mask computation
      Pipeline:
        Dense(latent) -> BN ->
        Dense(2*latent, relu) -> BN ->
        Dense(4*latent, relu) -> BN ->
        RepeatVector(W) ->
        TCN(return_sequences=True) ->
        ProbabilisticDecoderPT(hidden_dim=conv_filters, data_dim=NNF)
      Returns: a distribution whose .mean is (B, W, NNF)
    """
    def __init__(
        self,
        output_shape: Tuple[int, int],   # (W, NNF)
        latent_dim: int,
        conv_filters: int = 64,
        kernel_size: int = 4,
        conv_stacks: int = 1,
        conv_dilations: Iterable[int] = (8, 4, 2, 1),
        padding: str = "causal",
        use_skip_connections: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.W, self.data_dim = int(output_shape[0]), int(output_shape[1])
        self.latent_dim = int(latent_dim)

        # Front MLP: Dense -> BN -> Dense(relu) -> BN -> Dense(relu) -> BN
        self.fc0 = nn.Linear(latent_dim, latent_dim)
        self.bn0 = BatchNorm1dKerasFP32(latent_dim) # Keras like batch norm to help prevent extreme values 

        self.fc1 = nn.Linear(latent_dim, 2 * latent_dim)
        self.act1 = _act(activation)
        self.bn1 = BatchNorm1dKerasFP32(2 * latent_dim)

        self.fc2 = nn.Linear(2 * latent_dim, 4 * latent_dim)
        self.act2 = _act(activation)
        self.bn2 = BatchNorm1dKerasFP32(4 * latent_dim)

        # TCN over repeated latent sequence
        self.tcn = TCN1DPT(
            in_channels=4 * latent_dim,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            conv_stacks=conv_stacks,
            conv_dilations=conv_dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_rate=float(dropout_rate),
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_sequences=True,
        )
        # Probabilistic reconstruction head
        self.prob_decoder = ProbabilisticDecoderPT(hidden_dim=conv_filters, data_dim=self.data_dim)

        # Init linear layers (BN stats copied by transfer)
        for m in [self.fc0, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _stabilize_latent(self, g: torch.Tensor) -> torch.Tensor:
        # Training-only guard for extreme latent magnitudes
        g = g.float()
        #if self.training:
        # Per-sample RMS normalization
        rms = g.pow(2).mean(dim=1, keepdim=True).sqrt()  # (B,1)
        # Scale down to roughly unit RMS; avoid tiny denominators
        g = g / rms.clamp(min=1.0)
        # Clamp outliers to keep Dense/BN numerically safe
        g = g.clamp(min=-1e4, max=1e4)
        # Replace any remaining NaN/Inf (belt-and-suspenders)
        g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
        return g

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        """
        g: (B, latent_dim)
        x: (B, W, NNF) or (B, W, N, NF)  -> used only to compute validity mask
        returns: distribution with .mean of shape (B, W, NNF)
        """
        B = g.shape[0]
        # validity mask code unchanged...
        if x.dim() == 4:
            x_flat = x.view(x.size(0), x.size(1), -1)
        else:
            x_flat = x
        validity_mask = ~torch.all(x_flat == 0.0, dim=-1)

        # Stabilize latent and run front MLP in float32
        g32 = self._stabilize_latent(g)
        with torch.amp.autocast(device_type=g32.device.type , enabled=False):
            z = self.bn0(self.fc0(g32))
            z = self.bn1(self.act1(self.fc1(z)))
            z = self.bn2(self.act2(self.fc2(z)))

        # Repeat, TCN, and prob head as before
        z_rep = z.unsqueeze(1).repeat(1, self.W, 1)  # (B, W, 4*latent)
        # If you already compute TCN in fp32, keep it; otherwise:
        hidden_seq = self.tcn(z_rep)                 # (B, W, conv_filters)
        return self.prob_decoder(hidden_seq, validity_mask)
    

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

# --------- NaN debug helpers (shared) ---------
def _dbg_report_nan(name: str, t: torch.Tensor, sample_elems: int = 8):
    if t is None or not torch.is_floating_point(t):
        return
    with torch.no_grad():
        nan_mask = torch.isnan(t)
        if not nan_mask.any():
            return
        device = str(t.device)
        dtype = str(t.dtype)
        shape = tuple(t.shape)
        num_nan = int(nan_mask.sum().item())
        numel = t.numel()
        inf_mask = torch.isinf(t)
        num_inf = int(inf_mask.sum().item())
        finite_mask = torch.isfinite(t)
        finite_count = int(finite_mask.sum().item())
        stats = ""
        if finite_count > 0:
            finite_vals = t[finite_mask]
            try:
                stats = f"min={finite_vals.min().item():.4e}, max={finite_vals.max().item():.4e}, mean={finite_vals.float().mean().item():.4e}"
            except Exception:
                stats = "min/max/mean unavailable"
        idx = torch.nonzero(nan_mask, as_tuple=False)
        idx_sample = idx[:sample_elems].cpu().numpy() if idx.numel() > 0 else []
        print(f"[NaN DETECTED] {name}: shape={shape}, dtype={dtype}, device={device}, NaNs={num_nan}/{numel}, Infs={num_inf}, {stats}, nan_idx_sample={idx_sample}")

def _has_nonfinite(t: torch.Tensor) -> bool:
    if t is None or not torch.is_floating_point(t):
        return False
    with torch.no_grad():
        return not torch.isfinite(t).all()

def _safe_pointwise_conv1d(conv1x1: nn.Conv1d, x_bct: torch.Tensor, out_dtype: torch.dtype, name_prefix: str) -> torch.Tensor:
    """
    FIX: run pointwise Conv1d in float32 with autocast disabled.
    Also sanitize inputs if non-finite values are detected (no-op otherwise).
    x_bct: (B, C_in, T)
    """
    _dbg_report_nan(f"{name_prefix}.input_bct", x_bct)
    # Sanitize only if needed
    if _has_nonfinite(x_bct):
        with torch.no_grad():
            print(f"[SANITIZE] Non-finite detected at {name_prefix}.input_bct -> applying nan_to_num")
        x_bct = torch.nan_to_num(x_bct, nan=0.0, posinf=1e4, neginf=-1e4)

    # Check weights/bias before use
    _dbg_report_nan(f"{name_prefix}.weight", conv1x1.weight)
    if conv1x1.bias is not None:
        _dbg_report_nan(f"{name_prefix}.bias", conv1x1.bias)

    # Compute in float32 (AMP off) to avoid fp16 overflows
    with torch.amp.autocast(device_type=x_bct.device.type, enabled=False):
        y = conv1x1(x_bct.float())
    _dbg_report_nan(f"{name_prefix}.out_fp32", y)

    y = y.to(out_dtype)
    _dbg_report_nan(f"{name_prefix}.out_cast", y)
    return y
# ----------------------------------------------

def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if name == "leaky_relu": return nn.LeakyReLU(0.2)
    if name in {"linear", "identity", "none"}: return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def sinusoidal_positional_encoding(max_len: int, d_model: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Generate sinusoidal positional encodings."""
    pe = torch.zeros(max_len, d_model, dtype=dtype, device=device)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    n_odd = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term)[:, :n_odd]
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class BatchNorm1dKerasFP32(nn.BatchNorm1d):
    """Keras-like BatchNorm with eps=1e-3 and momentum=0.01."""
    def __init__(self, num_features, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x.float())
        return y.to(dtype=x.dtype)


class MultiHeadAttentionPT(nn.Module):
    """Multi-head attention using PyTorch's optimized scaled_dot_product_attention."""
    def __init__(self, in_dim: int, num_heads: int, key_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.inner_dim = self.num_heads * self.key_dim
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(self.in_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.in_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.in_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, self.in_dim, bias=False)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Project to Q, K, V and reshape for multi-head
        q = self.q_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)  # (B, H, T, K)
        k = self.k_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.key_dim).transpose(1, 2)

        # Convert boolean mask to attention mask format if needed
        # scaled_dot_product_attention expects: True = masked out (for attn_mask with is_causal=False)
        if attn_mask is not None:
            # Input mask: True = pad/invalid -> we want to mask these out
            # SDPA with attn_mask: positions with -inf are masked out
            if attn_mask.dtype == torch.bool:
                # Expand mask for broadcast: (B, T) -> (B, 1, 1, T)
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(B, self.num_heads, T, T)
                attn_mask = torch.where(attn_mask, torch.tensor(float('-inf'), device=x.device), torch.tensor(0.0, device=x.device))

        # Use PyTorch's optimized attention (FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        out = self.out_proj(attn_out)
        return out


class TransformerEncoderLayerPT(nn.Module):
    """Transformer encoder layer with post-normalization."""
    def __init__(self, key_dim: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttentionPT(in_dim=key_dim, num_heads=num_heads, key_dim=key_dim // num_heads, dropout=rate)
        self.dropout1 = nn.Dropout(rate)
        self.norm1 = nn.LayerNorm(key_dim, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(key_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, key_dim),
        )
        self.dropout2 = nn.Dropout(rate)
        self.norm2 = nn.LayerNorm(key_dim, eps=1e-6)

        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.mha(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerCorePT(nn.Module):
    """Core transformer: Linear embedding -> positional encoding -> transformer layers."""
    def __init__(
        self,
        in_channels: int,
        key_dim: int,
        num_layers: int,
        num_heads: int,
        dff: int,
        max_pos: int,
        rate: float = 0.1,
        return_sequences: bool = True,
    ):
        super().__init__()
        self.key_dim = int(key_dim)
        self.max_pos = int(max_pos)
        self.return_sequences = return_sequences
        self.dropout = nn.Dropout(rate)

        # Input projection
        self.embed = nn.Linear(in_channels, self.key_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayerPT(key_dim=self.key_dim, num_heads=num_heads, dff=dff, rate=rate) 
            for _ in range(int(num_layers))
        ])

        # Positional encoding buffer
        pe = sinusoidal_positional_encoding(self.max_pos, self.key_dim)
        self.register_buffer("pos_encoding", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_channels)
        Returns:
            (B, T, key_dim) if return_sequences else (B, key_dim)
        """
        B, T, _ = x.shape
        
        # Compute padding mask (True = pad/invalid)
        with torch.no_grad():
            mask = torch.all(x == 0.0, dim=-1)  # (B, T)

        # Embed
        y = self.embed(x)
        y = F.relu(y)
        y = y * (self.key_dim ** 0.5)

        # Add positional encoding
        if T > self.pos_encoding.size(1):
            self.pos_encoding = sinusoidal_positional_encoding(T, self.key_dim, device=x.device, dtype=x.dtype)
        y = y + self.pos_encoding[:, :T, :].to(y.dtype)
        y = self.dropout(y)

        # Apply transformer layers
        for layer in self.layers:
            y = layer(y, attn_mask=mask)

        if self.return_sequences:
            return y  # (B, T, key_dim)
        else:
            return y[:, -1, :]  # (B, key_dim) - last timestep

class TFMEncoderPT(nn.Module):
    """
    PyTorch Transformer Encoder with optional GNN.
    Mirrors the TCN encoder structure for consistency.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],        # (W, N, NF)
        edge_feature_shape: Tuple[int, int, int], # (W, E, EF)
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        use_gnn: bool = True,
        num_layers: int = 2,      # Reduced from 4 for speed
        num_heads: int = 4,       # Reduced from 8 for speed
        dff: int = 128,
        dropout_rate: float = 0.1,
        key_dim: int = None,      # Allow explicit key_dim
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.latent_dim = int(latent_dim)
        self.W, self.N, self.NF = input_shape
        _, self.E, self.EF = edge_feature_shape
        
        assert adjacency_matrix.shape[0] == self.N == adjacency_matrix.shape[1], \
            "Adjacency must be NxN and match input nodes."

        # Use reasonable key_dim (must be divisible by num_heads)
        if key_dim is None:
            key_dim = min(64, self.N * self.NF)
            # Ensure divisible by num_heads
            key_dim = (key_dim // num_heads) * num_heads
            key_dim = max(key_dim, num_heads)  # At least num_heads
        self.key_dim = int(key_dim)

        if use_gnn:
            # Node transformer: processes each node's temporal sequence
            self.node_tf = TransformerCorePT(
                in_channels=self.NF,
                key_dim=self.key_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dff=dff,
                max_pos=self.W,
                rate=dropout_rate,
                return_sequences=False,  # Only need last timestep
            )
            
            # Edge transformer: processes each edge's temporal sequence
            self.edge_tf = TransformerCorePT(
                in_channels=self.EF,
                key_dim=self.key_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dff=dff,
                max_pos=self.W,
                rate=dropout_rate,
                return_sequences=False,  # Only need last timestep
            )

            # Spatial GNN block
            self.spatial_gnn_block = CensNetConvPT(
                node_channels=self.latent_dim,
                edge_channels=self.latent_dim,
                activation="relu"
            )
            
            # Precompute and register graph matrices
            lap, edge_lap, inc = self.spatial_gnn_block.preprocess(torch.tensor(adjacency_matrix))
            self.register_buffer("laplacian", lap.float())
            self.register_buffer("edge_laplacian", edge_lap.float())
            self.register_buffer("incidence", inc.float())

            # Input to head: node features + edge features after GNN
            final_in = (self.N * self.latent_dim) + (self.E * self.latent_dim)
            
        else:
            # Single transformer for flattened input
            self.flat_tf = TransformerCorePT(
                in_channels=self.N * self.NF,
                key_dim=self.key_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dff=dff,
                max_pos=self.W,
                rate=dropout_rate,
                return_sequences=False,
            )
            final_in = self.key_dim

        # MLP head: matches TCN encoder structure
        self.head = nn.Sequential(
            nn.Linear(final_in, 2 * self.latent_dim),
            nn.ReLU(),
            BatchNorm1dKerasFP32(2 * self.latent_dim, eps=1e-3),
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.ReLU(),
            BatchNorm1dKerasFP32(self.latent_dim, eps=1e-3),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        # Initialize head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, N, NF) - node features over time
            a: (B, W, E, EF) - edge features over time
        Returns:
            (B, latent_dim) - encoded representation
        """
        B, W, N, NF = x.shape
        _, _, E, EF = a.shape
        
        assert (W, N, NF) == (self.W, self.N, self.NF), \
            f"Input shape mismatch: got ({W}, {N}, {NF}), expected ({self.W}, {self.N}, {self.NF})"

        if self.use_gnn:
            # === Process Nodes ===
            # Reshape to process each node's time series independently
            # Following the same memory layout as TCN encoder
            x_flat = x.view(B, W, N * NF)
            x_transposed = x_flat.permute(2, 1, 0)           # (N*NF, W, B)
            x_reshaped = x_transposed.reshape(NF, W, N, B)   # (NF, W, N, B)
            x_nodes = x_reshaped.permute(3, 2, 1, 0)         # (B, N, W, NF)
            
            node_in = x_nodes.reshape(B * N, W, NF)          # (B*N, W, NF)
            node_out = self.node_tf(node_in)                 # (B*N, key_dim)
            nodes_encoded = node_out.view(B, N, self.key_dim)  # (B, N, key_dim)

            # === Process Edges ===
            # Reshape to process each edge's time series independently
            # Same pattern as TCN encoder
            a_flat = a.view(B, W, E * EF)
            a_transposed = a_flat.permute(2, 1, 0)           # (E*EF, W, B)
            a_reshaped = a_transposed.reshape(EF, W, E, B)   # (EF, W, E, B)
            a_edges = a_reshaped.permute(3, 2, 1, 0)         # (B, E, W, EF)
            
            edge_in = a_edges.reshape(B * E, W, EF)          # (B*E, W, EF)
            edge_out = self.edge_tf(edge_in)                 # (B*E, key_dim)
            edges_encoded = edge_out.view(B, E, self.key_dim)  # (B, E, key_dim)

            # === Apply Spatial GNN ===
            adj_tuple = (self.laplacian, self.edge_laplacian, self.incidence)
            x_nodes_g, x_edges_g = self.spatial_gnn_block([
                nodes_encoded,
                adj_tuple,
                edges_encoded
            ])
            x_nodes_g = F.relu(x_nodes_g)
            x_edges_g = F.relu(x_edges_g)

            # Concatenate flattened node and edge features
            enc = torch.cat([
                x_nodes_g.reshape(B, -1),
                x_edges_g.reshape(B, -1)
            ], dim=-1)
            
        else:
            # === Non-GNN path ===
            x_flat = x.view(B, W, N * NF)  # (B, W, N*NF)
            enc = self.flat_tf(x_flat)     # (B, key_dim)

        # === Apply MLP head ===
        # Stabilize input similar to TCN encoder
        head_in = enc.float()
        rms = head_in.pow(2).mean(dim=1, keepdim=True).sqrt()
        head_in = head_in / rms.clamp(min=1.0)
        head_in = head_in.clamp(min=-1e4, max=1e4)
        head_in = torch.nan_to_num(head_in, nan=0.0, posinf=1e4, neginf=-1e4)
        
        out = self.head(head_in)

        # Force diversity during training by batch standardization
        if self.training and out.size(0) > 1:
            out = (out - out.mean(dim=0, keepdim=True)) / (out.std(dim=0, keepdim=True).clamp(min=0.1))

        return out
    

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
        deepof.clustering.model_utils_new.TransformerEncoder(
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
            deepof.clustering.model_utils_new.TransformerEncoder(
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


def create_look_ahead_mask_pt(size: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    PyTorch replica of TF create_look_ahead_mask (KEEP mask).
    Returns lower-triangular True (keep), False above diagonal.
    Shape: (T, T) boolean.
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def create_masks_pt(inp_3d: torch.Tensor):
    """
    PyTorch replica of TF create_masks for the decoder (KEEP semantics).
    """
    device = inp_3d.device
    B, T, _ = inp_3d.shape

    tar = inp_3d[:, :, 0]  # (B, T)
    dec_padding_keep = (tar != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
    la_keep = create_look_ahead_mask_pt(T, device=device, dtype=torch.bool)  # (T, T)
    combined_keep = (dec_padding_keep | la_keep.unsqueeze(0).unsqueeze(0))  # (B, 1, T, T)
    return combined_keep, dec_padding_keep


class MultiHeadAttentionGeneralPT(nn.Module):
    """Multi-head cross-attention using PyTorch 2.0+ optimized SDPA."""
    
    def __init__(self, q_in_dim: int, kv_in_dim: int, num_heads: int, key_dim: int, dropout: float = 0.0):
        super().__init__()
        self.q_in = int(q_in_dim)
        self.kv_in = int(kv_in_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.inner_dim = self.num_heads * self.key_dim
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(self.q_in, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(self.kv_in, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.kv_in, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, self.q_in, bias=True)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        query: torch.Tensor,   # (B, Tq, q_in)
        key: torch.Tensor,     # (B, Tk, kv_in)
        value: torch.Tensor,   # (B, Tk, kv_in)
        attn_mask: torch.Tensor = None,   # keep mask
        return_attention_scores: bool = True,
    ):
        B, Tq, _ = query.shape
        Tk = key.shape[1]

        q = self.q_proj(query).reshape(B, Tq, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, Tk, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Tk, self.num_heads, self.key_dim).transpose(1, 2)

        # Convert keep-mask to SDPA format
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Your masks use True=keep, SDPA uses True=masked for boolean
                # So we need to invert
                if attn_mask.dim() == 2:  # (B, Tk)
                    sdpa_mask = (~attn_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Tk)
                elif attn_mask.dim() == 4:  # (B, 1, Tq, Tk)
                    sdpa_mask = ~attn_mask

        ctx = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        ctx = ctx.transpose(1, 2).reshape(B, Tq, self.inner_dim)
        out = self.out_proj(ctx)

        if return_attention_scores:
            # Compute attention weights separately only if needed (slower path)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.key_dim ** 0.5)
                if sdpa_mask is not None:
                    scores = scores.masked_fill(sdpa_mask, float('-inf'))
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            return out, attn_weights
        
        return out


class TransformerDecoderLayerPT(nn.Module):
    def __init__(self, model_dim: int, memory_dim: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha1 = MultiHeadAttentionGeneralPT(q_in_dim=model_dim, kv_in_dim=model_dim, num_heads=num_heads, key_dim=model_dim, dropout=rate)
        self.dropout1 = nn.Dropout(rate)
        self.norm1 = nn.LayerNorm(model_dim, eps=1e-6)

        self.mha2 = MultiHeadAttentionGeneralPT(q_in_dim=model_dim, kv_in_dim=memory_dim, num_heads=num_heads, key_dim=model_dim, dropout=rate)
        self.dropout2 = nn.Dropout(rate)
        self.norm2 = nn.LayerNorm(model_dim, eps=1e-6)

        self.ffn1 = nn.Linear(model_dim, dff)
        self.act = nn.ReLU()
        self.ffn2 = nn.Linear(dff, model_dim)
        self.dropout3 = nn.Dropout(rate)
        self.norm3 = nn.LayerNorm(model_dim, eps=1e-6)

        for m in [self.ffn1, self.ffn2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                # (B, T, model_dim)
        memory: torch.Tensor,           # (B, T, memory_dim)
        look_ahead_mask_3d: torch.Tensor,  # (B, Tq, Tk) True=masked-out
        padding_mask_2d: torch.Tensor,     # (B, Tk) True=masked-out
        training: bool = False,
    ):
        _dbg_report_nan("Decoder.DecLayer.input_x", x)
        _dbg_report_nan("Decoder.DecLayer.input_memory", memory)
        # Self-attention
        attn1, w1 = self.mha1(query=x, key=x, value=x, attn_mask=look_ahead_mask_3d, return_attention_scores=True)
        _dbg_report_nan("Decoder.DecLayer.attn1_out", attn1)
        x = self.norm1(x + self.dropout1(attn1))
        _dbg_report_nan("Decoder.DecLayer.after_norm1", x)

        # Cross-attention
        attn2, w2 = self.mha2(query=x, key=memory, value=memory, attn_mask=padding_mask_2d, return_attention_scores=True)
        _dbg_report_nan("Decoder.DecLayer.attn2_out", attn2)
        x = self.norm2(x + self.dropout2(attn2))
        _dbg_report_nan("Decoder.DecLayer.after_norm2", x)

        # FFN
        ffn_out = self.ffn2(self.act(self.ffn1(x)))
        _dbg_report_nan("Decoder.DecLayer.ffn_out", ffn_out)
        x = self.norm3(x + self.dropout3(ffn_out))
        _dbg_report_nan("Decoder.DecLayer.after_norm3", x)
        return x, w1, w2


class DecoderCorePT(nn.Module):
    def __init__(self, model_dim: int, memory_dim: int, num_layers: int, num_heads: int, dff: int, max_pos: int, rate: float = 0.1):
        super().__init__()
        self.model_dim = int(model_dim)
        self.memory_dim = int(memory_dim)
        self.max_pos = int(max_pos)
        self.dropout = nn.Dropout(rate)

        self.embed = nn.Conv1d(self.model_dim, self.model_dim, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        self.layers = nn.ModuleList([
            TransformerDecoderLayerPT(model_dim=self.model_dim, memory_dim=self.memory_dim, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(int(num_layers))
        ])

        pe = sinusoidal_positional_encoding(self.max_pos, self.model_dim)
        self.register_buffer("pos_encoding", pe, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                # (B, T, model_dim)
        memory: torch.Tensor,           # (B, T, memory_dim)
        look_ahead_mask_3d: torch.Tensor,  # (B, T, T) True=masked-out
        padding_mask_2d: torch.Tensor,     # (B, T) True=masked-out
        training: bool = False,
    ):
        B, T, _ = x.shape
        _dbg_report_nan("Decoder.Core.input_x", x)
        _dbg_report_nan("Decoder.Core.input_memory", memory)

        # FIX: safe 1x1 conv in fp32 with optional sanitization
        x_bct = x.transpose(1, 2)  # (B, C_in, T)
        y_bct = _safe_pointwise_conv1d(self.embed, x_bct, out_dtype=x.dtype, name_prefix="Decoder.Core.embed")
        y = y_bct.transpose(1, 2)
        _dbg_report_nan("Decoder.Core.after_embed", y)

        y = torch.relu(y)
        _dbg_report_nan("Decoder.Core.after_relu", y)
        y = y * (self.model_dim ** 0.5)
        _dbg_report_nan("Decoder.Core.after_scale", y)

        if T > self.pos_encoding.size(1):
            self.pos_encoding = sinusoidal_positional_encoding(T, self.model_dim, device=x.device).to(self.pos_encoding.dtype)
        y = y + self.pos_encoding[:, :T, :].to(y.dtype)
        _dbg_report_nan("Decoder.Core.after_posenc", y)
        y = self.dropout(y)

        attention_weights = {}
        out = y
        for i, layer in enumerate(self.layers, start=1):
            out, w1, w2 = layer(out, memory, look_ahead_mask_3d, padding_mask_2d, training=training)
            _dbg_report_nan(f"Decoder.Core.layer[{i}].out", out)
            attention_weights[f"decoder_layer{i}_block1"] = w1
            attention_weights[f"decoder_layer{i}_block2"] = w2

        return out, attention_weights


class TFMDecoderPT(nn.Module):
    """
    Transformer decoder that FORCES latent usage by concatenating 
    latent to every timestep, not using cross-attention.
    """
    def __init__(
        self,
        output_shape: Tuple[int, int],  # (W, D_in)
        latent_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.1,
        # Legacy params (ignored but kept for compatibility)
        teacher_forcing_mode: str = "zeros",
        input_dropout_p: float = 0.0,
        self_attn_diag_only: bool = False,
    ):
        super().__init__()
        self.W, self.D_in = output_shape
        self.latent_dim = int(latent_dim)
        
        # Expand latent dimension
        self.expanded_latent_dim = 4 * self.latent_dim
        
        # Latent expansion MLP
        self.latent_expand = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, 2 * self.latent_dim),
            nn.GELU(),
            nn.Linear(2 * self.latent_dim, self.expanded_latent_dim),
            nn.GELU(),
        )
        
        # Model dimension = expanded latent (latent is THE input, not cross-attended)
        self.model_dim = self.expanded_latent_dim
        
        # Positional encoding
        pe = sinusoidal_positional_encoding(self.W, self.model_dim)
        self.register_buffer("pos_encoding", pe, persistent=False)
        
        # Causal self-attention layers (NO cross-attention)
        self.layers = nn.ModuleList([
            CausalSelfAttentionLayer(
                d_model=self.model_dim,
                num_heads=num_heads,
                dff=dff,
                dropout=dropout_rate,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection to data dimension
        self.output_proj = nn.Linear(self.model_dim, self.D_in)
        
        # Probabilistic output head
        self.prob_decoder = ProbabilisticDecoderPT(hidden_dim=self.D_in, data_dim=self.D_in)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor, x_target: torch.Tensor, training: bool = None):
        """
        Args:
            g: (B, latent_dim) - latent code from encoder
            x_target: (B, W, D_in) - target sequence (only used for validity mask)
        """
        B, W, D = x_target.shape
        assert (W, D) == (self.W, self.D_in), f"Shape mismatch: got ({W}, {D}), expected ({self.W}, {self.D_in})"
        
        # Compute validity mask from target
        with torch.no_grad():
            validity_mask = ~torch.all(x_target == 0.0, dim=-1)  # (B, W)
        
        # === KEY CHANGE: Latent is THE input, not cross-attended ===
        # Expand latent
        g_expanded = self.latent_expand(g)  # (B, expanded_latent_dim)
        
        # Repeat latent for each timestep - this IS the decoder input
        h = g_expanded.unsqueeze(1).expand(-1, self.W, -1).contiguous()  # (B, W, model_dim)
        
        # Add positional encoding to differentiate timesteps
        if self.W > self.pos_encoding.size(1):
            self.pos_encoding = sinusoidal_positional_encoding(
                self.W, self.model_dim, device=g.device, dtype=g.dtype
            )
        h = h + self.pos_encoding[:, :self.W, :].to(h.dtype)
        
        # Causal self-attention layers
        for layer in self.layers:
            h = layer(h)
        
        # Project to output dimension
        h = self.output_proj(h)  # (B, W, D_in)
        
        return self.prob_decoder(h, validity_mask)


class CausalSelfAttentionLayer(nn.Module):
    """Causal self-attention layer (no cross-attention)."""
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Causal attention with Flash Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.dropout(self.out_proj(attn_out))
        
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x
    

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
  

class VectorQuantizerPT(nn.Module):
    """PyTorch Vector quantizer layer."""

    def __init__(
        self,
        n_components: int,
        embedding_dim: int,
        beta: float,
        kmeans_loss: float = 0.0,
    ):
        super(VectorQuantizerPT, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta
        self.kmeans = kmeans_loss

        self.codebook = nn.Parameter(
            torch.empty(self.embedding_dim, self.n_components).uniform_(0, 1)
        )
        
        # Store individual loss components for inspection (but not for summing)
        self.last_commitment_loss = None
        self.last_codebook_loss = None
        self.last_kmeans_loss = None
        self.last_vq_loss = None

    def forward(self, x: torch.Tensor, return_losses: bool = True):
        """
        Args:
            x: Input tensor of shape (..., embedding_dim)
               Typically (batch_size, embedding_dim) from encoder
        """
        input_shape = x.shape
        
        losses = {}

        # Flatten to 2D
        flattened = x.reshape(-1, self.embedding_dim)

        # Kmeans loss on flattened 2D tensor
        if self.kmeans and return_losses:
            kmeans_loss_val = deepof.clustering.model_utils_new.compute_kmeans_loss_pt(flattened, self.kmeans)
            losses['kmeans_loss'] = kmeans_loss_val
            self.last_kmeans_loss = kmeans_loss_val.item()

        # Get encodings
        encoding_indices = self.get_code_indices(
            flattened, return_soft_counts=False
        ).long()
        soft_counts = self.get_code_indices(flattened, return_soft_counts=True)

        encodings = F.one_hot(encoding_indices, self.n_components).float()
        quantized = torch.matmul(encodings, self.codebook.T)
        quantized = quantized.reshape(input_shape)

        if return_losses:
            commitment_loss = self.beta * torch.mean(
                (quantized.detach() - x) ** 2
            )
            codebook_loss = torch.mean((quantized - x.detach()) ** 2)
            vq_loss = commitment_loss + codebook_loss
            
            # Store the COMBINED vq_loss in the losses dict (matching TF behavior)
            losses['vq_loss'] = vq_loss
            
            # Store individual components for inspection/logging only
            self.last_commitment_loss = commitment_loss.item()
            self.last_codebook_loss = codebook_loss.item()
            self.last_vq_loss = vq_loss.item()

        if return_losses:
            return quantized, soft_counts, losses
        else:
            return quantized, soft_counts

    def get_code_indices(
        self, flattened_inputs: torch.Tensor, return_soft_counts: bool = False
    ) -> torch.Tensor:
        similarity = torch.matmul(flattened_inputs, self.codebook)
        distances = (
            torch.sum(flattened_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=0)
            - 2 * similarity
        )

        if return_soft_counts:
            similarity = (1 / distances) ** 2
            soft_counts = similarity / torch.sum(similarity, dim=1, keepdim=True)
            return soft_counts

        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer Layer for VQ-VAE.
    - Inherits from torch.nn.Module.
    - Manages a codebook of embeddings.
    - Computes quantization, losses, and returns results.

    Based on
    https://arxiv.org/pdf/1509.03700.pdf, and adapted for clustering using https://arxiv.org/abs/1806.02199.
    Implementation based on https://keras.io/examples/generative/vq_vae/.

    """

    def __init__(self, n_components: int, embedding_dim: int, beta: float, kmeans_loss_weight: float = 0.0):
        """
        Initialize the VQ layer.

        Args:
            n_components (int): Number of vectors in the codebook (K).
            embedding_dim (int): Dimensionality of each embedding vector (D).
            beta (float): Weight for the commitment loss.
            kmeans_loss_weight (float): Weight for the k-means-like disentanglement loss.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta
        self.kmeans_loss_weight = kmeans_loss_weight

        # Initialize the codebook. 
        self.codebook = nn.Parameter(torch.rand(n_components, embedding_dim))


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the VQ layer.

        Args:
            x (torch.Tensor): The input tensor from the encoder.
                              Shape: [batch_size, ..., embedding_dim].

        Returns:
            A tuple containing:
            - quantized_st (torch.Tensor): The quantized output tensor (with straight-through
                                           gradient), same shape as input x.
            - vq_loss (torch.Tensor): The vector quantization loss (codebook + commitment).
            - kmeans_loss (torch.Tensor): The k-means disentanglement loss.
            - soft_counts (torch.Tensor): Soft assignments to clusters.
                                          Shape: [batch_size, ..., n_components].
            - encoding_indices (torch.Tensor): The hard indices of the chosen codes.
                                               Shape: [batch_size, ...].
        """
        input_shape = x.shape
        device = x.device
        
        # Flatten input tensor while keeping the embedding dimension.
        # Shape: [B * ..., D] -> [N, D]
        flattened = x.reshape(-1, self.embedding_dim)

        # --- K-Means Loss ---
        kmeans_loss = torch.tensor(0.0, device=device)
        if self.kmeans_loss_weight > 0:
            kmeans_loss = deepof.clustering.model_utils_new.compute_kmeans_loss(flattened, weight=self.kmeans_loss_weight)

        # --- Distance Calculation (Compute only once) ---
        # Calculate squared L2 distance between each input and each codebook vector.
        # distances = (a-b)^2 = a^2 - 2ab + b^2
        sum_sq_inputs = torch.sum(flattened**2, dim=1, keepdim=True)
        sum_sq_codes = torch.sum(self.codebook**2, dim=1)
        dot_product = torch.matmul(flattened, self.codebook.T)
        distances = sum_sq_inputs + sum_sq_codes - 2 * dot_product  # Shape: [N, K]

        # --- Hard and Soft Assignments ---
        # 1. Get hard assignments (indices of the closest codebook vectors)
        encoding_indices = torch.argmin(distances, dim=1) # Shape: [N]
        
        # 2. Get soft assignments (based on the original paper's logic)
        # Add a small epsilon for numerical stability to avoid division by zero.
        # The original logic is (1/d^2)^2, which is 1/d^4.
        similarity = (1.0 / (distances + 1e-9)) ** 2
        soft_counts = similarity / torch.sum(similarity, dim=1, keepdim=True) # Shape: [N, K]

        # --- Quantization using hard assignments ---
        quantized = F.embedding(encoding_indices, self.codebook)
        quantized = quantized.view(input_shape) # Reshape back to original input shape
        
        # --- VQ Loss Calculation ---
        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # --- Straight-Through Estimator ---
        # This allows gradients to flow from the decoder back to the encoder
        # through the non-differentiable quantization step.
        quantized_st = x + (quantized - x).detach()
        
        # --- Reshape outputs to match original spatial dimensions ---
        # Reshape soft_counts to [batch_size, ..., n_components]
        soft_counts_reshaped = soft_counts.view(*input_shape[:-1], self.n_components)
        # Reshape indices to [batch_size, ...]
        encoding_indices_reshaped = encoding_indices.view(input_shape[:-1])

        return(
           quantized_st,
           vq_loss,
           kmeans_loss,
           soft_counts_reshaped,
           encoding_indices_reshaped
        )


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
            kmeans_loss = deepof.clustering.model_utils_new.compute_kmeans_loss(
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


class VQVAEPT(nn.Module):
    """
    PyTorch implementation of the VQ-VAE model adapted to the DeepOF setting.
    
    Note: This version handles the actual DeepOF input format where:
    - x: (B, T, node_features) - flattened node features
    - a: (B, T, edge_features) - flattened edge features
    """
    
    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        n_components: int,
        encoder_type: str = "recurrent", 
        use_gnn: bool = True, 
        kmeans_loss: float = 0.0,
        interaction_regularization: float = 0.0,      
        beta: float = 1.0,
    ):
        """Initialize a VQ-VAE model.

        Args:
            input_shape (tuple): Shape of the input (time_steps, node_features).
            edge_feature_shape (tuple): Shape of edge features (time_steps, edge_features).
            adjacency_matrix (np.ndarray): Adjacency matrix for GNN.
            latent_dim (int): Dimensionality of the latent space.
            n_components (int): Number of embeddings (clusters) in the codebook.
            beta (float): Beta parameter of the VQ loss.
            kmeans_loss (float): Regularization parameter for the Gram matrix.
            use_gnn (bool): Whether to use GNN in encoder.
            encoder_type (str): Type of encoder ("recurrent", "TCN", or "transformer").
            interaction_regularization (float): Regularization parameter for interactions.
        """
        super().__init__()
        
        time_steps, n_nodes, n_features_per_node = input_shape
        self.input_n_nodes = n_nodes
        self.input_n_features_per_node = n_features_per_node
        self.window_size = time_steps
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.encoder_type = encoder_type
        
        # Initialize encoder based on type
        if encoder_type == "recurrent":
            self.encoder = deepof.clustering.models_new.RecurrentEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )
            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = deepof.clustering.models_new.RecurrentDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
            )
        elif encoder_type == "TCN":
            self.encoder = deepof.clustering.models_new.TCNEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )
            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = deepof.clustering.models_new.TCNDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
            ) 
        elif encoder_type == "transformer":
            self.encoder = deepof.clustering.models_new.TFMEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
            )
            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = deepof.clustering.models_new.TFMDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
                num_layers=2,
                num_heads=8,
                dff=128,
                dropout_rate=0.2,
                teacher_forcing_mode="zeros",
                input_dropout_p=0.5,
                self_attn_diag_only=False,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Initialize Vector Quantizer
        self.vq_layer = VectorQuantizerPT(
            n_components=n_components,
            embedding_dim=latent_dim,
            beta=beta,
            kmeans_loss=kmeans_loss,
        )

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        return_losses: bool = True,
        return_all_outputs: bool = False,
    ):
        """
        Forward pass through the VQ-VAE model.
        
        Args:
            x: Input node features (B, T, N, F)
            a: Input edge features (B, T, E, F_edge)
            return_losses: Whether to compute and return VQ losses
            return_all_outputs: Whether to return all intermediate outputs
        """
        # Encode
        encoder_output = self.encoder(x, a)  # Shape: (B, latent_dim)
        
        # Vector Quantization
        if return_losses:
            quantized_latents, soft_counts, vq_losses = self.vq_layer(
                encoder_output, return_losses=True
            )
        else:
            quantized_latents, soft_counts = self.vq_layer(
                encoder_output, return_losses=False
            )
            vq_losses = {}
        
        # Prepare input for decoder (flatten spatial dimensions for teacher forcing)
        B, T, N, F = x.shape
        x_for_decoder = x.view(B, T, N * F)  # Flatten to (B, T, node_features)
        
        # Decode from QUANTIZED latents (main path)
        encoding_reconstruction_dist = self.decoder(quantized_latents, x_for_decoder)
        
        # Decode from ORIGINAL encoder output (bypass path for gradient flow)
        reconstruction_dist = self.decoder(encoder_output, x_for_decoder)
        
        # Handle transformer decoder outputs (which return attention weights)
        if self.encoder_type == "transformer":
            if isinstance(encoding_reconstruction_dist, tuple):
                encoding_reconstruction_dist = encoding_reconstruction_dist[0]
            if isinstance(reconstruction_dist, tuple):
                reconstruction_dist = reconstruction_dist[0]
        
        if return_all_outputs:
            return (
                encoding_reconstruction_dist,
                reconstruction_dist,
                quantized_latents,
                soft_counts,
                encoder_output,
                vq_losses if return_losses else None,
            )
        else:
            if return_losses:
                return encoding_reconstruction_dist, reconstruction_dist, vq_losses
            else:
                return encoding_reconstruction_dist, reconstruction_dist

    @torch.no_grad()
    def encode(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Inference-only: Get encoder output. Equivalent to TF 'encoder' model."""
        return self.encoder(x, a)

    @torch.no_grad()
    def group(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Inference-only: Get quantized latents. Equivalent to TF 'grouper' model."""
        encoder_output = self.encoder(x, a)
        quantized, _ = self.vq_layer(encoder_output, return_losses=False)
        return quantized

    @torch.no_grad()
    def soft_group(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Inference-only: Get soft cluster assignments. Equivalent to TF 'soft_grouper' model."""
        encoder_output = self.encoder(x, a)
        _, soft_counts = self.vq_layer(encoder_output, return_losses=False)
        return soft_counts

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Full reconstruction from input through VQ-VAE."""
        encoding_recon_dist, _ = self.forward(x, a, return_losses=False)
        return encoding_recon_dist.mean
    
    def get_codebook_usage(self, data_loader, max_samples: int = 10000):
        """Compute codebook usage statistics over a dataset."""
        self.eval()
        all_indices = []
        all_soft_counts = []
        samples_seen = 0
        
        with torch.no_grad():
            for x, a, *_ in data_loader:
                x = x.to(next(self.parameters()).device)
                a = a.to(next(self.parameters()).device)
                
                encoder_output = self.encoder(x, a)
                flattened = encoder_output.reshape(-1, self.latent_dim)
                
                indices = self.vq_layer.get_code_indices(
                    flattened, return_soft_counts=False
                )
                soft_counts = self.vq_layer.get_code_indices(
                    flattened, return_soft_counts=True
                )
                
                all_indices.append(indices.cpu())
                all_soft_counts.append(soft_counts.cpu())
                
                samples_seen += x.size(0)
                if samples_seen >= max_samples:
                    break
        
        all_indices = torch.cat(all_indices)
        all_soft_counts = torch.cat(all_soft_counts)
        
        usage_counts = torch.bincount(all_indices, minlength=self.n_components)
        
        # Compute perplexity
        avg_probs = all_soft_counts.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return usage_counts, perplexity.item()
    

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


class GaussianMixtureLatentPT(nn.Module):
    """
    PyTorch implementation of the Gaussian Mixture probabilistic latent space model.
    It embeds data into a latent space and models that space as a mixture of Gaussians.
    """
    def __init__(
        self,
        input_dim: int,
        n_components: int,
        latent_dim: int,
        kmeans: float,
        lens_enabled: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.kmeans_weight = kmeans
        self.lens_enabled = False # lens_enabled
        self.mixture_dim = (24 if self.lens_enabled else self.latent_dim)

        # --- Trainable Parameters for the GMM components ---
        self.gmm_means = nn.Parameter(torch.empty(n_components, self.mixture_dim))
        self.gmm_log_vars = nn.Parameter(torch.empty(n_components, self.mixture_dim))
        nn.init.xavier_normal_(self.gmm_means)
        nn.init.xavier_normal_(self.gmm_log_vars)

        # --- Encoder Layers to produce the latent distribution ---
        self.encoder_mean = nn.Linear(self.input_dim, self.latent_dim)
        self.encoder_log_var = nn.Linear(self.input_dim, self.latent_dim)

        # --- Non-trainable Buffers ---
        self.register_buffer('prior', torch.ones(n_components) / n_components)
        self.register_buffer('pretrain', torch.tensor(0.0))
        
        # --- Helper Layers ---
        self.cluster_control = deepof.clustering.model_utils_new.ClusterControlPT()

        # --- Focus layer ---
        self.lens = nn.Linear(self.latent_dim,self.mixture_dim)


    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input into mean and log-variance of the latent distribution."""
        z_mean = self.encoder_mean(x)
        z_log_var_pre = self.encoder_log_var(x) # Note: softplus is applied in the forward pass
        return z_mean, z_log_var_pre

    def _reparameterize(
        self, mean: torch.Tensor, log_var: torch.Tensor, epsilon: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs reparameterization.
        MODIFIED to exactly replicate the original TF model's non-standard scale calculation.
        """
        # Original TF logic: scale = sqrt(exp(variance))
        # The 'var' input here is the direct output of the softplus activation.
        scale = torch.exp(0.5 * log_var)  # sqrt(exp(log_var))
        if epsilon is None:
            epsilon = torch.randn_like(scale)
        return mean + scale * epsilon        
        

    def _calculate_posterior(self, z: torch.Tensor) -> torch.Tensor:
        """Calculates the posterior probability p(c|z) for each sample."""
        # MODIFIED: The GMM parameters from TF are log-std-dev, not log-variance.
        # So we just exponentiate them to get the scale.
        gmm_std = torch.exp(0.5 * self.gmm_log_vars).clamp(min=1e-3)

        gmm_dist = Normal(
            loc=self.gmm_means.unsqueeze(0),
            scale=gmm_std.unsqueeze(0)
        )
        log_p_z_given_c = gmm_dist.log_prob(z.unsqueeze(1)).sum(dim=-1)
        
        log_p_c_given_z = torch.log(self.prior + 1e-9) + log_p_z_given_c
        
        return F.softmax(log_p_c_given_z, dim=-1)

    def forward(
        self, x: torch.Tensor, epsilon: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        z_mean, z_log_var_pre = self._encode(x)
        z_log_var = F.softplus(z_log_var_pre) # Apply activation

        # Pass z_var directly, not z_log_var
        z_sample = self._reparameterize(z_mean, z_log_var, epsilon)
        z_for_downstream = z_sample if self.training else z_mean

        # Compute lens-space posterior parameters (μ_h, log_var_h) for proper MC-KL  ###HERE!
        if self.lens_enabled: 
            W = self.lens.weight  # (d_lens, d_latent) 
            h_mean = F.linear(z_mean, W, bias=None)  # z_mean @ W.T 
            var_z = torch.exp(z_log_var)  # (B, d_latent) 
            var_h = torch.clamp(var_z @ (W.pow(2)).t(), min=1e-8)  # (B, d_lens) 
            h_log_var = torch.log(var_h)  # (B, d_lens)  
        else:  
            h_mean = z_mean  # (B, latent_dim) 
            h_log_var = z_log_var  # (B, latent_dim) 

        # Focus to lower dimension, if lens is enabled
        if self.lens_enabled:
            z_for_gaussian = self.lens(z_for_downstream)
        else:
            z_for_gaussian = z_for_downstream

        if torch.isnan(z_for_downstream).any():
            print("z issues!")
        
        # get probabilities from Gaussians
        z_cat = self._calculate_posterior(z_for_gaussian)
        z_final, metrics = self.cluster_control(z_for_downstream, z_cat)
        kmeans_loss = torch.tensor(0.0, device=x.device)
        if self.kmeans_weight > 0:
            kmeans_loss = deepof.clustering.model_utils_new.compute_kmeans_loss_pt(z_final, weight=self.kmeans_weight)

        return (z_final, z_cat, metrics["number_of_populated_clusters"], metrics["confidence_in_selected_cluster"], kmeans_loss, h_mean, h_log_var, z_for_gaussian)

    @torch.no_grad()
    def set_lens_weights(self, W: torch.Tensor) -> None: 
        """Set the projection (lens) weights from an external initializer (e.g., PCA/LDA).""" 
        if W.shape != self.lens.weight.shape:  
            raise ValueError(f"Lens weight shape {W.shape} does not match {self.lens.weight.shape}")  
        self.lens.weight.data.copy_(W.to(self.lens.weight.device, dtype=self.lens.weight.dtype)) 
        #self.freeze_lens(True)

    def freeze_lens(self, freeze: bool = True) -> None:  
        """Freeze/unfreeze the lens parameters."""  
        if freeze:
            print("Freezing lense")
        else:
            print("Unfreezing lense")
        for p in self.lens.parameters():       
            p.requires_grad = not freeze 












class GaussianMixtureLatentPT_new(nn.Module):
    def __init__(self, input_dim: int, n_components: int, latent_dim: int, kmeans: float, responsibility_temp: float = 1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.kmeans_weight = kmeans
        self.responsibility_temp = responsibility_temp

        # GMM params (trainable)
        self.gmm_means = nn.Parameter(torch.empty(n_components, latent_dim))
        self.gmm_log_vars = nn.Parameter(torch.empty(n_components, latent_dim))
        nn.init.xavier_normal_(self.gmm_means)
        nn.init.xavier_normal_(self.gmm_log_vars)

        # Encoder layers q(z|x)
        self.encoder_mean = nn.Linear(self.input_dim, self.latent_dim)
        self.encoder_log_var = nn.Linear(self.input_dim, self.latent_dim)  # keep this name for checkpoint compat

        # Buffers
        self.register_buffer("prior", torch.ones(n_components) / n_components)
        self.register_buffer("pretrain", torch.tensor(0.0))

        self.cluster_control = deepof.clustering.model_utils_new.ClusterControlPT()

    def _encode(self, x: torch.Tensor):
        z_mean = self.encoder_mean(x)
        # Output raw log-variance (can be negative). Clamp for stability if needed.
        z_log_var = self.encoder_log_var(x)
        z_log_var = torch.clamp(z_log_var, min=-10.0, max=10.0)
        return z_mean, z_log_var

    def _reparameterize(self, mean: torch.Tensor, z_log_var: torch.Tensor, epsilon: torch.Tensor = None):
        scale = torch.exp(0.5 * z_log_var)
        if epsilon is None:
            epsilon = torch.randn_like(scale)
        return mean + scale * epsilon

    def _calculate_posterior(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure everything is on the same device and dtype
        dev = self.gmm_means.device
        dtype = self.gmm_means.dtype

        z = z.to(dev, dtype=dtype, non_blocking=True)
        # Defensive guard to avoid hard crashes (remove once stable)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

        gmm_means = self.gmm_means                               # (C, D) on dev
        # Clamp to keep scales reasonable
        gmm_log_vars = torch.clamp(self.gmm_log_vars, min=-10.0, max=10.0)
        gmm_scale = torch.exp(0.5 * gmm_log_vars).clamp(min=1e-6)



        gmm_dist = Normal(gmm_means.unsqueeze(0), gmm_scale.unsqueeze(0))  # (1, C, D)
        log_p_z_given_c = gmm_dist.log_prob(z.unsqueeze(1)).sum(dim=-1)    # (B, C)

        prior = self.prior.to(dev, dtype=dtype)
        log_p_c_given_z = torch.log(prior + 1e-9).unsqueeze(0) + log_p_z_given_c
        #return F.softmax(log_p_c_given_z, dim=-1)  # (B, C)

        T_resp = float(getattr(self, "responsibility_temp", 1.0))
        return torch.softmax(log_p_c_given_z / max(1e-6, T_resp), dim=-1)  # (B, C)

    def forward(self, x: torch.Tensor, epsilon: torch.Tensor = None):
        z_mean, z_log_var = self._encode(x)
        z_sample = self._reparameterize(z_mean, z_log_var, epsilon)
        # Use z_mean for responsibilities to reduce noise
        z_cat = self._calculate_posterior(z_mean)

        # For the decoder, keep reparameterized sample during training; mean at eval
        z_for_downstream = z_sample if self.training else z_mean

        z_final, metrics = self.cluster_control(z_for_downstream, z_cat)

        kmeans_loss = torch.tensor(0.0, device=x.device, dtype=z_final.dtype)
        if self.kmeans_weight > 0:
            with torch.autocast(device_type=z_final.device.type, enabled=False):
                km32 = deepof.clustering.model_utils_new.compute_kmeans_loss_pt(
                    z_final.float(), weight=self.kmeans_weight
                )
            kmeans_loss = torch.nan_to_num(km32, nan=0.0, posinf=1e6, neginf=0.0).to(z_final.dtype)

        return (
            z_final,
            z_cat,
            metrics["number_of_populated_clusters"],
            metrics["confidence_in_selected_cluster"],
            kmeans_loss,
            z_mean,
            z_log_var,
        )
    
    @torch.no_grad()
    def log_emissions(self, z: torch.Tensor) -> torch.Tensor:
        gmm_log_vars = torch.clamp(self.gmm_log_vars, min=-10.0, max=10.0)
        scale = torch.exp(0.5 * gmm_log_vars).clamp(min=1e-6)
        dist = Normal(self.gmm_means.unsqueeze(0), scale.unsqueeze(0))
        return dist.log_prob(z.unsqueeze(1)).sum(dim=-1) # [T, C]



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

    def call(self, inputs, training=False, epsilon=None, return_all_outputs_for_testing=False): # pragma: no cover
        """Compute the output of the layer."""
        z_gauss_mean = self.z_gauss_mean(inputs)
        z_gauss_var = self.z_gauss_var(inputs)

        if epsilon is not None:
            # Use deterministic reparameterization for testing
            z_sample = z_gauss_mean + tf.math.sqrt(tf.math.exp(z_gauss_var)) * epsilon
        else:
            # Original stochastic sampling for production
            z_dist = tfd.MultivariateNormalDiag(
                loc=z_gauss_mean, scale_diag=tf.math.sqrt(tf.math.exp(z_gauss_var))
            )
            z_sample = tf.squeeze(z_dist.sample())

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

        #self.add_metric(loss_clustering, name="clustering_loss", aggregation="mean")
        #self.add_metric(loss_prior, name="prior_loss", aggregation="mean")

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

        #self.add_metric(self._kl_weight, aggregation="mean", name="kl_weight")
        #self.add_metric(kl, aggregation="mean", name="kl_divergence")

        #self.add_loss(tf.math.reduce_mean(loss_clustering))
        #self.add_loss(tf.math.reduce_mean(loss_prior))
        #self.add_loss(tf.math.reduce_mean(kl_batch))


        # Calculate metrics for potential return
        hard_groups = tf.math.argmax(z_cat, axis=1)
        max_groups = tf.reduce_max(z_cat, axis=1)
        n_populated = tf.cast(tf.shape(tf.unique(tf.reshape(hard_groups, [-1]))[0])[0], tf.float32)
        confidence = tf.reduce_mean(max_groups)

        z = z_sample if training else z_gauss_mean

        if self.n_components > 1:
            z = self.cluster_control_layer([z, z_cat])

        k_loss = 0.0
        if self.kmeans:
            k_loss = deepof.model_utils.compute_kmeans_loss(z, weight=self.kmeans, batch_size=self.batch_size)
            #self.add_loss(k_loss)
            #self.add_metric(k_loss, name="kmeans_loss")

        # MODIFIED: Added a switch for the return value to be bale to test this block
        if return_all_outputs_for_testing:
            # In test mode, return all computed values for direct comparison
            return z, z_cat, n_populated, confidence, k_loss
        else:
            # Otherwise, use side effects (add_loss/add_metric) and return the original signature
            loss_clustering = -tf.reduce_sum(tf.multiply(z_cat, tf.math.softmax(p_z_c, axis=-1)), axis=-1) * (1.0 - tf.cast(self.pretrain, tf.float32))
            loss_prior = -tf.reduce_sum(tf.math.xlogy(z_cat, 1e-6 + self.prior), axis=-1) * (1.0 - tf.cast(self.pretrain, tf.float32))
            self.add_metric(loss_clustering, name="clustering_loss", aggregation="mean")
            self.add_metric(loss_prior, name="prior_loss", aggregation="mean")

            self.add_metric(self._kl_weight, aggregation="mean", name="kl_weight")
            self.add_metric(kl, aggregation="mean", name="kl_divergence")

            self.add_loss(tf.math.reduce_mean(loss_clustering))
            self.add_loss(tf.math.reduce_mean(loss_prior))
            self.add_loss(tf.math.reduce_mean(kl_batch))

            if self.kmeans:
                self.add_loss(k_loss)
                self.add_metric(k_loss, name="kmeans_loss")

            return z, z_cat



def vade_loss_function(reconstruction_dist, original_data, model_internal_losses, categorical_probs, reg_cat_clusters_weight):
    # Reconstruction Loss (Negative Log-Likelihood)
    recon_loss = -reconstruction_dist.log_prob(original_data).mean()

    # Model Internal Losses (KL + kmeans from VaDEPT)
    # The model already calculates these per-batch, we just sum them.
    internal_loss = model_internal_losses.mean() # Assuming losses are per-item in batch

    # Categorical Regularization (if enabled)
    cat_reg_loss = torch.tensor(0.0, device=original_data.device)
    if reg_cat_clusters_weight > 0:
        # KL divergence between categorical_probs and a uniform distribution
        uniform_dist = torch.full_like(categorical_probs, 1.0 / categorical_probs.size(1))
        # F.kl_div requires log-probabilities as input
        cat_reg_loss = F.kl_div(
            categorical_probs.log(), uniform_dist, reduction='batchmean'
        )
        cat_reg_loss *= reg_cat_clusters_weight

    total_loss = recon_loss + internal_loss + cat_reg_loss
    return total_loss, recon_loss, internal_loss, cat_reg_loss


class VaDEPT(nn.Module):
    """
    A self-contained PyTorch implementation of the VaDE model.
    """
    def __init__(
        self,
        input_shape: tuple,
        edge_feature_shape: tuple,
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        n_components: int,
        encoder_type: str = "recurrent",
        use_gnn: bool = True,
        kmeans_loss: float = 1.0,
        interaction_regularization: float = 0.0,
        lens_enabled = False,
    ):
        super().__init__()
        
        time_steps, n_nodes, n_features_per_node = input_shape
        self.input_n_nodes = n_nodes
        self.input_n_features_per_node = n_features_per_node
        self.window_size = time_steps #important for modal usage later
        self.lens_enabled=lens_enabled

        if encoder_type == "recurrent":
            self.encoder = RecurrentEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )

            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = RecurrentDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
            )
        elif encoder_type == "TCN":
            self.encoder = TCNEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )

            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = TCNDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
            ) 
        elif encoder_type == "transformer":
            self.encoder = TFMEncoderPT(
                input_shape=input_shape,
                edge_feature_shape=edge_feature_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                #interaction_regularization=interaction_regularization,
            )

            decoder_output_features = n_nodes * n_features_per_node
            self.decoder = TFMDecoderPT(
                output_shape=(time_steps, decoder_output_features),
                latent_dim=latent_dim,
                num_layers=2,
                num_heads=8,
                dff=128,
                dropout_rate=0.2,               # a bit more dropout helps
                teacher_forcing_mode="zeros", # try "zeros" if collapse persists
                input_dropout_p=0.5,            # drop 50% of time steps during training
                self_attn_diag_only=False,      # set True to further reduce copying
            ) 
        else:
            raise NotImplementedError("invalid encoder type, try \"recurrent\", \"TCN\" or \"transformer\" ")          

        self.latent_space = GaussianMixtureLatentPT(
            input_dim=latent_dim,
            n_components=n_components,
            latent_dim=latent_dim,
            kmeans=kmeans_loss,
            lens_enabled=self.lens_enabled,
        )



    def forward(
        self, x: torch.Tensor, a: torch.Tensor, return_gmm_params: bool = False
    ):
        """
        Returns:
            - reconstruction_dist
            - latent
            - categorical
            - kmeans_loss
            - z_mean
            - z_log_var
            - gmm_params (dict) [if return_gmm_params=True]
        """
        encoder_output = self.encoder(x, a)
        z_mean_head = self.latent_space.encoder_mean(encoder_output)               
        z_var_param = torch.nn.functional.softplus(self.latent_space.encoder_log_var(encoder_output))  
        (
            latent,
            categorical,
            _n_populated,
            _confidence,
            kmeans_loss,
            h_mean,
            h_log_var,
            z_for_gaussian,
        ) = self.latent_space(encoder_output)

        if torch.isnan(z_mean_head).any() or torch.isnan(z_var_param).any():
            print("z issues!")

        B, T, _, _ = x.shape
        x_for_decoder = x.view(B, T, self.input_n_nodes * self.input_n_features_per_node)
        reconstruction_dist = self.decoder(latent, x_for_decoder)

        if return_gmm_params:
            gmm_params = {
                "means": self.latent_space.gmm_means,
                "log_vars": self.latent_space.gmm_log_vars,
                "prior": self.latent_space.prior,
            }
            return (
                reconstruction_dist,
                z_for_gaussian,
                categorical,
                kmeans_loss,
                h_mean,
                h_log_var,
                gmm_params,
            )
        else:
            return reconstruction_dist, z_for_gaussian, categorical, kmeans_loss

    @property
    def get_gmm_params(self) -> dict:
        """Returns the GMM parameters from the latent space."""
        with torch.no_grad():
            means = self.latent_space.gmm_means
            log_vars = self.latent_space.gmm_log_vars
            stds = torch.exp(0.5 * log_vars)
            weights = self.latent_space.prior
        return {"means": means, "log_vars": log_vars, "sigmas": stds, "weights": weights}

    def set_pretrain_mode(self, pretrain_on: bool):
        """Sets the pretrain flag in the latent space."""
        self.latent_space.pretrain.fill_(1.0 if pretrain_on else 0.0)

    def initialize_gmm_from_data(self, data_loader, n_samples=10000):
        """
        Runs the autoencoder part of the model over the data to get embeddings,
        then fits a scikit-learn GMM to initialize the latent space.
        """
        print("Initializing GMM from data embeddings...")
        self.eval()
        
        all_embeddings = []
        samples_gathered = 0
        dev = next(self.parameters()).device
        with torch.no_grad():
            for x, a, *_ in data_loader:
                x, a = x.to(dev), a.to(dev)
                enc = self.encoder(x, a)
                z_mean, _ = self.latent_space._encode(enc)
                if getattr(self.latent_space, "lens_enabled", False):          
                    z_mean = self.latent_space.lens(z_mean)  
                all_embeddings.append(z_mean.cpu())
                samples_gathered += z_mean.size(0)
                if samples_gathered >= n_samples:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        if all_embeddings.shape[0] > n_samples:
            all_embeddings = all_embeddings[:n_samples]

        from sklearn.mixture import GaussianMixture
        print(f"Fitting scikit-learn GMM on {all_embeddings.shape[0]} samples...")
        gmm = GaussianMixture(
            n_components=self.latent_space.n_components,
            covariance_type="diag",
            reg_covar=1e-04,
        ).fit(all_embeddings)

        print("Assigning learned GMM parameters to the model.")
        self.latent_space.gmm_means.data = torch.from_numpy(gmm.means_).float().to(dev)
        # Store log-variances (not log-sigmas)
        self.latent_space.gmm_log_vars.data = torch.from_numpy(np.log(gmm.covariances_)).float().to(dev)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Inference-only method to get the latent embedding. Equivalent to the 'embedding' Keras model.

        Args:
            x (torch.Tensor): Input node features tensor.
            a (torch.Tensor): Input edge features tensor.

        Returns:
            torch.Tensor: The latent representation `z`.
        """
        encoder_output = self.encoder(x, a)
        latent, _, _, _, _, _, _ = self.latent_space(encoder_output)
        return latent

    @torch.no_grad()
    def group(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Inference-only method to get cluster probabilities. Equivalent to the 'grouper' Keras model.

        Args:
            x (torch.Tensor): Input node features tensor.
            a (torch.Tensor): Input edge features tensor.

        Returns:
            torch.Tensor: The soft cluster assignments (categorical probabilities).
        """
        encoder_output = self.encoder(x, a)
        _, categorical, _, _, _, _, _ = self.latent_space(encoder_output)
        return categorical


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
                em_x=get_dt(embed_x, 'embed_x')
                em_a=get_dt(embed_a, 'embed_a')

                emb_idx = np.random.choice(range(em_x.shape[0]), samples)

                # map to latent
                z = self.encoder([em_x[emb_idx], em_a[emb_idx]])
                
                del em_x
                del em_a
                del emb_idx

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


class ContrastivePT(nn.Module):
    """
    PyTorch port of the TF Contrastive model.

    Inputs:
      x: (B, T, N, F)
      a: (B, T, E, F_edge)

    Behavior:
      - Builds an encoder for sequences of length T//2.
      - forward(x_half, a_half) returns embeddings (B, D) for a given half-window.
      - compute_loss(x_full, a_full) slices pos/neg windows and returns (loss, pos_mean, neg_mean, debug).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],         # (T, N, F)
        edge_feature_shape: Tuple[int, int, int],  # (T, E, F_edge)
        adjacency_matrix,
        latent_dim: int = 8,
        encoder_type: str = "TCN",
        use_gnn: bool = True,
        temperature: float = 0.1,
        similarity_function: str = "cosine",
        loss_function: str = "nce",
        beta: float = 0.1,
        tau: float = 0.1,
        interaction_regularization: float = 0.0,
    ):
        super().__init__()

        T, N, F_in = input_shape
        Te, E, Fe = edge_feature_shape
        
        if T != Te:
            raise ValueError(f"Node and edge time dims must match: T={T}, Te={Te}")
        if T < 2 or (T % 2) != 0:
            raise ValueError(
                f"ContrastivePT requires an even sequence length T>=2. Got T={T}. "
                "Please pre-trim or pad your sequences (e.g., use T=24 if original T=25)."
            )

        self.full_time_steps = T
        self.window_size = T // 2
        self.input_shape = input_shape
        self.edge_feature_shape = edge_feature_shape
        self.adjacency_matrix = adjacency_matrix

        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        self.encoder_type = encoder_type

        self.temperature = temperature
        self.similarity_function = similarity_function
        self.loss_function = loss_function
        self.beta = beta
        self.tau = tau
        self.interaction_regularization = interaction_regularization

        if encoder_type == "recurrent":
            self.encoder = deepof.clustering.models_new.RecurrentEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )
        elif encoder_type == "TCN":
            self.encoder = deepof.clustering.models_new.TCNEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )
        elif encoder_type == "transformer":
            self.encoder = deepof.clustering.models_new.TFMEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Debug cache
        self._last_debug: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Encode a half-window:
          x: (B, T_half, N, F), a: (B, T_half, E, Fe) -> (B, D)
        """
        return self.encoder(x, a)

    @staticmethod
    def _ts_samples(x: torch.Tensor, win: int):
        # TF parity: pos = x[:, 1:win+1], neg = x[:, -win:]
        pos = x[:, 1 : win + 1]
        neg = x[:, -win:]
        return pos, neg

    def compute_loss(
        self,
        x: torch.Tensor,  # (B, T, N, F)
        a: torch.Tensor,  # (B, T, E, Fe)
        return_debug: bool = False,
    ):
        B, T, N, F_in = x.shape
        if T != self.full_time_steps:
            raise ValueError(f"Input time dim T={T} does not match model T={self.full_time_steps}")

        # Slice windows exactly like TF
        pos_x, neg_x = self._ts_samples(x, self.window_size)
        pos_a, neg_a = self._ts_samples(a, self.window_size)

        # Encode and normalize
        z_pos = self.encoder(pos_x, pos_a)  # (B, D)
        z_neg = self.encoder(neg_x, neg_a)  # (B, D)
        z_pos = deepof.clustering.model_utils_new._l2_normalize(z_pos, dim=1, eps=1e-12)
        z_neg = deepof.clustering.model_utils_new._l2_normalize(z_neg, dim=1, eps=1e-12)

        # Compute loss
        loss, pos_mean, neg_mean = deepof.clustering.model_utils_new.select_contrastive_loss_pt(
            z_pos,
            z_neg,
            similarity=self.similarity_function,
            loss_fn=self.loss_function,
            temperature=self.temperature,
            tau=self.tau,
            beta=self.beta,
            elimination_topk=0.1,  # same default as TF snippet
        )

        debug = None
        if return_debug:
            # Build a minimal debug pack for parity troubleshooting
            sim_fn = deepof.clustering.model_utils_new._SIMILARITIES[self.similarity_function]
            with torch.no_grad():
                sim = sim_fn(z_pos, z_neg)  # (B, B)
                diag = torch.diag(sim)
                offdiag = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]
                offdiag = offdiag.view(B, B - 1) if B > 1 else offdiag.view(B, 0)

                debug = {
                    "z_pos_shape": torch.tensor(z_pos.shape),
                    "z_neg_shape": torch.tensor(z_neg.shape),
                    "z_pos_norm_mean": torch.norm(z_pos, dim=1).mean().cpu(),
                    "z_neg_norm_mean": torch.norm(z_neg, dim=1).mean().cpu(),
                    "sim_diag_mean": diag.mean().cpu(),
                    "sim_offdiag_mean": offdiag.mean().cpu() if offdiag.numel() > 0 else torch.tensor(0.0),
                    "loss": loss.detach().cpu(),
                    "pos_mean": pos_mean.detach().cpu(),
                    "neg_mean": neg_mean.detach().cpu(),
                }
            self._last_debug = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in debug.items()}

        return loss, pos_mean, neg_mean, debug

    def get_last_debug(self) -> Dict[str, Any]:
        return self._last_debug


def training_step_contrastive_pt(
    model: ContrastivePT,
    x: torch.Tensor,
    a: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_value: float = 0.75,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, pos_mean, neg_mean, _ = model.compute_loss(x, a, return_debug=False)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
    optimizer.step()
    return loss.item(), float(pos_mean.item()), float(neg_mean.item())


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
    

#########################################################
# Intermediary function stash for presentation
#########################################################

from dataclasses import dataclass
from types import SimpleNamespace
from deepof.clustering.dataset import BatchDictDataset
from deepof.clustering.model_utils_new import select_contrastive_loss_pt, save_model_info, CommonFitCfg, TurtleTeacherCfg, VaDECfg, ContrastiveCfg 
from sklearn.decomposition import IncrementalPCA
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def unwrap_dp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def move_to(x, device):
    if isinstance(x, (list, tuple)):
        return type(x)(move_to(xx, device) for xx in x)
    if isinstance(x, Mapping):
        return {k: move_to(v, device) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x

@torch.no_grad()
def _compute_diagnostics(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    max_batches: int = 4,
) -> Dict[str, float]:
    """
    Fast diagnostics:
      - mean entropy of q(c|z) over samples
      - entropy of marginal q̄(c) across the inspected set
      - GMM logvar min/max and prior entropy
      - teacher τ*: marginal entropy, mean confidence, mean distillation weight
    """
    base = unwrap_dp(model)
    base.eval()

    # q(c|z) stats on a few validation batches
    total_samples = 0
    sum_ent = 0.0
    sum_q = None
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x, a, *rest = move_to(batch, device)
        outputs = base(x, a, return_gmm_params=True)
        q = outputs[2].float().clamp_min(1e-8)  # (B,C)
        ent = -(q * q.log()).sum(dim=-1)        # (B,)
        sum_ent += float(ent.sum().item())
        total_samples += q.size(0)
        sum_q = q.sum(dim=0) if sum_q is None else (sum_q + q.sum(dim=0))

    if total_samples > 0:
        q_marginal = (sum_q / total_samples).clamp_min(1e-9)   # (C,) torch
        mean_q_entropy = sum_ent / total_samples
        q_marginal_entropy = float(-(q_marginal * q_marginal.log()).sum().item())
    else:
        q_marginal = None
        mean_q_entropy = float("nan")
        q_marginal_entropy = float("nan")

    # GMM stats
    gmm_log_vars = unwrap_dp(model).latent_space.gmm_log_vars.detach()
    logvar_min = float(gmm_log_vars.min().item())
    logvar_max = float(gmm_log_vars.max().item())

    prior = unwrap_dp(model).latent_space.prior
    prior = prior.detach() if torch.is_tensor(prior) else torch.as_tensor(prior)
    prior = prior.clamp_min(1e-9)
    prior_entropy = float(-(prior * prior.log()).sum().item())

    # Teacher τ* stats
    teacher_marginal_entropy, teacher_conf_mean, teacher_w_mean = float("nan"), float("nan"), float("nan")
    kl_marg_q_to_tau = float("nan")  # NEW
    if criterion is not None and getattr(criterion, "tau_star", None) is not None:
        tau = criterion.tau_star.detach()
        T = float(getattr(criterion, "distill_sharpen_T", 0.5))
        tau_sharp = torch.softmax((tau.clamp_min(1e-8)).log() / T, dim=-1) if T > 0.0 else tau
        conf = tau_sharp.max(dim=1).values
        teacher_conf_mean = float(conf.mean().item())
        if bool(getattr(criterion, "distill_conf_weight", False)):
            thr = float(getattr(criterion, "distill_conf_thresh", 0.55))
            w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(min=0.0, max=1.0)
            teacher_w_mean = float(w.mean().item())
        else:
            teacher_w_mean = 1.0

        tau_marg = tau.mean(dim=0).clamp_min(1e-9)  # (C,) torch
        teacher_marginal_entropy = float(-(tau_marg * tau_marg.log()).sum().item())

        # NEW: KL(student marginal || teacher marginal)
        if q_marginal is not None:
            kl = (q_marginal * (q_marginal.log() - tau_marg.log())).sum().item()
            kl_marg_q_to_tau = float(max(0.0, kl))

    return {
        "diag/q_mean_entropy": mean_q_entropy,
        "diag/q_marginal_entropy": q_marginal_entropy,
        "diag/gmm_logvar_min": logvar_min,
        "diag/gmm_logvar_max": logvar_max,
        "diag/prior_entropy": prior_entropy,
        "diag/teacher_marginal_entropy": teacher_marginal_entropy,
        "diag/teacher_conf_mean": teacher_conf_mean,
        "diag/teacher_weight_mean": teacher_w_mean,
        "diag/kl_marg_q_to_tau": kl_marg_q_to_tau, 
    }

################
# Turtle teacher
################


def soft_cross_entropy_logits(logits, soft_targets, eps=1e-8, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    soft_targets = torch.clamp(soft_targets, min=eps, max=1.0)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class TurtleHeads(nn.Module):
    def __init__(self, feature_dims: List[int], n_components: int,
                 inner_lr: float = 0.1, M: int = 100,
                 weight_decay: float = 1e-4,
                 normalize_feats: bool = True,
                 temperature: float = 0.7,
                 supervised_index: Optional[int] = None, 
                 mlp_hidden: int = 32,                    
                 mlp_dropout: float = 0.0                 
                 ):
        super().__init__()
        self.M = M
        self.normalize_feats = normalize_feats
        self.temperature = temperature
        self.supervised_index = supervised_index

        self.heads = nn.ModuleList()
        self._optims = []

        for i, d in enumerate(feature_dims):
            if False: #i != supervised_index:
                head = MLPHead(d, n_components,
                               hidden_dim=mlp_hidden,
                               dropout=mlp_dropout)
            else:
                head = nn.Linear(d, n_components)
            self.heads.append(head)
            self._optims.append(
                torch.optim.SGD(head.parameters(), lr=inner_lr, weight_decay=weight_decay)
            )

    @torch.no_grad()
    def reset_parameters(self):
        for h in self.heads:
            # Check if module has reset_parameters (Linear/MLP)
            if hasattr(h, 'reset_parameters'):
                h.reset_parameters()
            elif hasattr(h, 'net'):
                for m in h.net:
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()

    def _maybe_normalize(self, feats_list):
        if not self.normalize_feats:
            return feats_list
        out = []
        for i, f in enumerate(feats_list):
            if hasattr(self, "supervised_index") and (i == self.supervised_index):
                out.append(f)  # don't normalize labels
            else:
                out.append(F.normalize(f, dim=-1))
        return out

    def inner_fit(self, feats_list, soft_targets):
        feats_list = self._maybe_normalize([f.detach().float() for f in feats_list])
        soft_targets = soft_targets.detach().float()
        
        for _ in range(self.M):
            for head, opt, feats in zip(self.heads, self._optims, feats_list):
                logits = head(feats) / self.temperature
                loss = soft_cross_entropy_logits(logits, soft_targets)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def logits_list(self, feats_list):
        feats_list = self._maybe_normalize([f.detach().float() for f in feats_list])
        return [(head(feats) / self.temperature).detach() for head, feats in zip(self.heads, feats_list)]

class TaskEncoder(nn.Module):
    def __init__(self, feature_dims: List[int], n_components: int, 
                 temperature: float = 1.0,
                 supervised_index: Optional[int] = None,  
                 mlp_hidden: int = 32,                  
                 mlp_dropout: float = 0.0                
                 ):
        super().__init__()
        self.temperature = temperature
        self.supervised_index = supervised_index

        self.projs = nn.ModuleList()
        for i, d in enumerate(feature_dims):
            if False: #i != supervised_index:
                proj = MLPHead(d, n_components,
                               hidden_dim=mlp_hidden,
                               dropout=mlp_dropout)
            else:
                proj = nn.Linear(d, n_components)
            self.projs.append(proj)

    def forward(self, feats_list):
        logits = None
        for proj, feats in zip(self.projs, feats_list):
            # Ensure float for stability, detaching is handled by optimizer zero_grad mostly
            # but usually we want gradients through this.
            out = proj(feats.float()) / self.temperature
            logits = out if logits is None else (logits + out)
        
        logits = logits / max(len(self.projs), 1)
        return F.softmax(logits, dim=-1)
    

class TurtleTeacher(nn.Module):
    def __init__(self, feature_dims: List[int], n_components: int,
                 gamma: float = 10.0,
                 alpha_sample_entropy: float = 0.1,
                 inner_lr: float = 0.1,
                 inner_steps: int = 100,
                 head_wd: float = 1e-4,
                 head_temp: float = 0.5,
                 task_temp: float = 0.5,
                 normalize_feats: bool = True,
                 lr_theta: float = 5e-3,
                 delta_death_barrier: float = 40.0,
                 supervised_index: Optional[int] = None,  
                 mlp_hidden: int = 32,                    
                 mlp_dropout: float = 0.0                  
                 ):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.delta = delta_death_barrier
        self.alpha = alpha_sample_entropy
        self.supervised_index = supervised_index

        self.heads = TurtleHeads(
            feature_dims, n_components,
            inner_lr=inner_lr, M=inner_steps,
            weight_decay=head_wd,
            normalize_feats=normalize_feats,
            temperature=head_temp,
            supervised_index=supervised_index,
            mlp_hidden=mlp_hidden,
            mlp_dropout=mlp_dropout, 
        )
        self.task_encoder = TaskEncoder(
            feature_dims, n_components,
            temperature=task_temp,
            supervised_index=supervised_index,
            mlp_hidden=mlp_hidden,
            mlp_dropout=mlp_dropout,
        )
        self.opt_theta = torch.optim.Adam(self.task_encoder.parameters(), lr=lr_theta)
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device
        self.heads.to(device)
        self.task_encoder.to(device)
        return self

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    # --- NEW: Helper for full dataset prediction (Sequential) ---
    @torch.no_grad()
    def predict(self, loader) -> torch.Tensor:
        """
        Runs a sequential pass over the data to compute assignments for the whole dataset.
        Used to generate the final tau_star without loading everything into GPU RAM at once.
        """
        self.task_encoder.eval()
        all_taus = []
        for batch_views in loader:
            # Move batch to GPU
            batch_views = [b.to(self.device) for b in batch_views]
            tau = self.task_encoder(batch_views)
            all_taus.append(tau.cpu())
        self.task_encoder.train()
        return torch.cat(all_taus, dim=0)

    # Fits on batches from a DataLoader ---
    def fit(self, loader, outer_steps: int = 200,
            rho: float = 0.04, verbose: bool = True):
        
        # Create an infinite iterator over the loader
        iterator = iter(loader)
        
        for step in range(outer_steps):
            # Fetch next batch
            try:
                feats_list = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                feats_list = next(iterator)
            
            # Move batch to GPU (non_blocking helps if pin_memory=True)
            feats_list = [f.to(self.device, non_blocking=True) for f in feats_list]

            # Extract labels if present in the batch
            labels = None
            if self.supervised_index is not None and 0 <= self.supervised_index < len(feats_list):
                labels = feats_list[self.supervised_index]

            # --- Forward Pass ---
            tau = self.task_encoder(feats_list)  # [Batch, K]

            # --- Inner Loop: Update Heads ---
            # Fit linear heads to current batch assignments
            self.heads.inner_fit(feats_list, tau)

            # --- Compute Loss Components ---
            logits_k = self.heads.logits_list(feats_list)
            ce_term = 0.0
            feats_weight = np.ones(len(logits_k))
            
            for k, logits in enumerate(logits_k):
                ce_term = ce_term + soft_cross_entropy_logits(logits, tau) * feats_weight[k]
            ce_term = ce_term / max(len(logits_k), 1)

            sample_entropy = self._entropy(tau).mean()
            
            # Estimate Marginal Entropy (Batch Approximation)
            marginal = tau.mean(dim=0)
            H_marginal = self._entropy(marginal.unsqueeze(0)).mean()

            logK = float(np.log(self.n_components))
            H_target = torch.tensor(1 * logK, device=H_marginal.device)
            marg_gap = torch.relu(H_target - H_marginal)
            
            gamma_t = float(self.gamma) * (1.0 - float(step) / float(max(1, outer_steps)))
            dead_floor = max(1e-4, 0.1 / self.n_components)

            # Dead penalty
            tau_clamp = tau.clamp_min(1e-8)
            gamma_pow = 2.0
            usage = (tau_clamp ** gamma_pow).mean(dim=0)
            dead_pen = torch.relu(dead_floor - usage).sum() / (dead_floor * self.n_components)

            delta_t = self.delta * max(0.5, 0.6 + 0.4 * (1.0 - step / (float(max(1, outer_steps)))))
            
            # Active count metric (for logging)
            K_dim = int(marginal.numel())
            tau_act = 0.02
            active_soft = torch.sigmoid((marginal - dead_floor) / tau_act)
            active_count = active_soft.sum()

            lambda_purity, H_y_weighted, beta, L_size = 0, 0, 0, 0
            
            # Supervised Regularization (Batch Approximation)
            if labels is not None:
                labels_bin = (labels > 0.05).to(torch.int64)
                
                # Count unique labels in this batch
                unique_rows, inverse_idx, counts = torch.unique(
                    labels_bin, dim=0, return_inverse=True, return_counts=True
                )
                freqs = counts.float() / labels_bin.size(0)
                freqs_sorted, _ = torch.sort(freqs, descending=True)
                
                K_keep = min(len(freqs_sorted)-1, int(self.n_components/2))
                
                if K_keep < len(freqs_sorted):
                    thresh = freqs_sorted[K_keep]
                    keep = freqs > thresh
                else:
                    keep = torch.ones_like(freqs, dtype=torch.bool)

                kept_codes = torch.nonzero(keep, as_tuple=False).squeeze(1)
                K_combos = kept_codes.numel()

                if K_combos > 0:
                    mapping = -torch.ones(len(unique_rows), dtype=torch.long, device=labels.device)
                    mapping[kept_codes] = torch.arange(K_combos, device=labels.device)
                    cls_idx = mapping[inverse_idx]

                    Y_one_hot = torch.zeros(labels_bin.size(0), K_combos, device=labels.device)
                    mask = cls_idx >= 0
                    Y_one_hot[mask, cls_idx[mask]] = 1.0

                    mass_c = tau_clamp.sum(dim=0) + 1e-8
                    p_c = mass_c / mass_c.sum()
                    counts_c = tau_clamp.t() @ Y_one_hot
                    p_y_given_c = (counts_c + 1e-8) / (mass_c.unsqueeze(1) + 1e-8)
                    H_y_c = -(p_y_given_c * p_y_given_c.log()).sum(dim=1)

                    alpha_w = 1.0
                    w = (p_c + 1e-8).pow(-alpha_w)
                    w = w / w.sum()
                    H_y_weighted = (w * H_y_c).sum()

                    beta = 0.5
                    purity = 1.0 - H_y_c / H_y_c.max().clamp_min(1e-8)
                    psi = 1.0 / (p_c + 1e-8)
                    L_size = ((1.0 - purity) * psi).sum()
                    lambda_purity = 20.0

            loss = (
                ce_term
                + self.alpha * sample_entropy
                + gamma_t * marg_gap
                + delta_t * dead_pen
                + lambda_purity * H_y_weighted
                + beta * L_size
            )

            # Temporal Smoothing (Batch-local)
            if (step % 2) != 0 and rho > 0.0:
                # Calculates smoothness between consecutive samples in the batch.
                # Note: if batch is shuffled, this regularization is noisy/weak, 
                # but prevents collapse during training.
                diff = tau[1:] - tau[:-1]
                smooth = (diff.abs().sum(dim=-1)).mean()
                loss = loss + rho * smooth

            # --- Optimization ---
            self.opt_theta.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_theta.step()

            if verbose and (step % 20 == 0 or step == outer_steps - 1):
                with torch.no_grad():
                    mean_max_p = tau.max(dim=1).values.mean().item()
                    print(f"[Teacher] step {step:03d} | loss {loss.item():.4f} | CE {ce_term.item():.4f} | "
                          f"E[H(τ)] {sample_entropy.item():.4f} | H(marg) {H_marginal.item():.4f} | "
                          f"mean max_p {mean_max_p:.3f} | dead_pen {dead_pen.item():.3f} | "
                          f"H(y|c) {lambda_purity * H_y_weighted:.3f} | "
                          f"active≈{active_count.item():.2f}/{K_dim}")





@torch.no_grad()
def extract_latents(model: nn.Module, dataset: BatchDictDataset, device: torch.device,
                    batch_size: int = 512, num_workers: int = 0) -> torch.Tensor:
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )
    base = unwrap_dp(model)                                                      
    base.eval()
    zs = []
    for batch in loader:
        x, a, *rest = move_to(batch, device)
        # Use encoder + posterior head to get z_mean in encoder space (latent_dim) 
        enc = base.encoder(x, a)                                                 
        z_mean, _ = base.latent_space._encode(enc)                             
        zs.append(z_mean.detach().cpu())
    z_all = torch.cat(zs, dim=0)
    return z_all  # CPU


def initialize_gmm_from_teacher(model: nn.Module,
                                z_all: torch.Tensor,        # [N, D] z_mean for train set (CPU or device)
                                tau_star: torch.Tensor,     # [N, C] teacher labels (on same device you ran teacher)
                                min_var: float = 1e-4,
                                min_mass: float = 1e-6) -> None:
    """
    Compute GMM parameters from teacher assignments:
      μ_c = sum_i τ*_ic u_i / sum_i τ*_ic
      σ^2_c = sum_i τ*_ic (u_i - μ_c)^2 / sum_i τ*_ic
      π_c = sum_i τ*_ic / N

    Where u_i = lens(z_i) if lens_enabled, else z_i.

    Writes directly into model.latent_space.{gmm_means, gmm_log_vars, prior}.
    """
    base = unwrap_dp(model)
    base.eval()

    device = next(base.parameters()).device
    z = z_all.to(device=device, dtype=base.latent_space.gmm_means.dtype)
    tau = tau_star.to(device=device, dtype=z.dtype)

    # Project to mixture space only if z is encoder/posterior-dim (not already lens-dim)
    if getattr(base.latent_space, "lens_enabled", False):                                           
        in_feats = getattr(base.latent_space.lens, "in_features", base.latent_space.latent_dim)     
        if z.shape[1] == int(in_feats):                                                           
            with torch.no_grad():                                                                   
                z = base.latent_space.lens(z)                                                       
        # else: z already in lens-space; skip projection  
        
    N, D_mix = z.shape
    C = tau.shape[1]

    # Cluster masses and priors
    mass = tau.sum(dim=0) + min_mass
    prior = (mass / mass.sum()).clamp(min=1e-8, max=1.0)

    # Means: (C, D_mix)
    means = (tau.T @ z) / mass.unsqueeze(1)

    # Variances: weighted per-cluster diag variance (C, D_mix)
    diffs = z.unsqueeze(1) - means.unsqueeze(0)                       # [N, C, D_mix]
    vars_ = (tau.unsqueeze(-1) * (diffs ** 2)).sum(dim=0) / mass.unsqueeze(-1)  # [C, D_mix]
    vars_ = vars_.clamp(min=min_var)
    log_vars = vars_.log()

    # Edge-case guard
    tiny = (mass <= 1e-4)
    if tiny.any():
        global_mean = z.mean(dim=0)
        global_var = z.var(dim=0, unbiased=False).clamp(min=min_var)
        means[tiny] = global_mean
        log_vars[tiny] = global_var.log()

    # Write into model buffers/params
    base.latent_space.gmm_means.data.copy_(means)
    base.latent_space.gmm_log_vars.data.copy_(log_vars)
    if hasattr(base.latent_space, "prior"):
        if isinstance(base.latent_space.prior, torch.nn.Parameter):
            base.latent_space.prior.data.copy_(prior)
        else:
            base.latent_space.prior.copy_(prior)

    print("Initialized GMM from teacher τ*: "
          f"mean |μ|={means.norm(dim=1).mean():.3f}, "
          f"mean σ²={vars_.mean():.5f}, "
          f"entropy(π)={-(prior*prior.clamp_min(1e-9).log()).sum().item():.3f}")
    

###########
# PCA stuff
###########

@torch.no_grad()
def fit_nodes_pca(
    dataset: BatchDictDataset,
    n_components_pos: int = 32,
    n_components_spd: int = 32,
    batch_size: int = 4096,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Fits two IncrementalPCAs:
      - one on positions (x,y) only
      - one on speeds (the 3rd channel per node)

    Assumes dataset returns x with shape [B, T, N, F] where F>=3 and
    channel 0,1 are (x,y), channel 2 is speed.

    Returns:
      ipca_pos, feats_pos_all  # [N, n_components_pos]
      ipca_spd, feats_spd_all  # [N, n_components_spd]
    """

    shuffle=False
    if seed is not None:
        shuffle=True

    # ---- Pass 1: partial_fit ----
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=shuffle,
        permute_within_block=False,
        seed=seed,
    )

    ipca_pos = IncrementalPCA(n_components=n_components_pos)
    ipca_spd = IncrementalPCA(n_components=n_components_spd)

    seen = 0
    for batch in loader:
        x, a, *rest = batch
        B, T, N, F = x.shape

        if F < 3:
            raise ValueError(f"Expected at least 3 channels (x,y,speed); got F={F}")

        pos = x[..., :2]   # [B,T,N,2]
        spd = x[..., 2:3]  # [B,T,N,1]

        X_pos = pos.reshape(B, -1).float().numpy()  # (B, T*N*2)
        X_spd = spd.view(B, -1).float().numpy()  # (B, T*N*1)

        if (max_samples is not None) and (seen >= max_samples):
            break
        if (max_samples is not None) and (seen + B > max_samples):
            cut = max(1, max_samples - seen)
            X_pos = X_pos[:cut]
            X_spd = X_spd[:cut]
            B = cut

        ipca_pos.partial_fit(X_pos)
        ipca_spd.partial_fit(X_spd)
        seen += B

    # ---- Pass 2: transform all ----
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=shuffle,
        permute_within_block=False,
        seed=seed,
    )

    feats_pos, feats_spd = [], []
    for batch in loader:
        x, a, *rest = batch
        B, T, N, F = x.shape

        pos = x[..., :2]   # [B,T,N,2]
        spd = x[..., 2:3]  # [B,T,N,1]

        X_pos = pos.reshape(B, -1).float().numpy()
        X_spd = spd.view(B, -1).float().numpy()

        Z_pos = ipca_pos.transform(X_pos)  # (B, n_components_pos)
        Z_spd = ipca_spd.transform(X_spd)  # (B, n_components_spd)

        feats_pos.append(torch.from_numpy(Z_pos).float())
        feats_spd.append(torch.from_numpy(Z_spd).float())

    feats_pos_all = torch.cat(feats_pos, dim=0)  # [N, n_components_pos]
    feats_spd_all = torch.cat(feats_spd, dim=0)  # [N, n_components_spd]
    return ipca_pos, feats_pos_all, ipca_spd, feats_spd_all


@torch.no_grad()
def fit_angles_pca(
    dataset_with_angles: BatchDictDataset,
    n_components: int = 32,
    batch_size: int = 8192,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> Tuple[IncrementalPCA, torch.Tensor]:
    """
    Fits IncrementalPCA on angle tensors and returns both the fitted ipca and features.
    Mirrors fit_nodes_pca but for angle data.
    """
    assert getattr(dataset_with_angles, "return_angles", False), \
        "fit_angles_pca expects a dataset created with return_angles=True."

    shuffle=False
    if seed is not None:
        shuffle=True

    # Pass 1: partial_fit
    ipca = IncrementalPCA(n_components=n_components)
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=shuffle, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
        seed=seed,
    )
    for batch in loader:
        # Expected: (x, a, ang, vid) when return_angles=True
        if len(batch) >= 3:
            ang = batch[2]  # angles tensor
        else:
            raise RuntimeError(f"Angles loader must yield at least 3 elements, got {len(batch)}")
        # Flatten: (B, T, K) -> (B, T*K)
        X = ang.view(ang.size(0), -1).float().cpu().numpy()
        ipca.partial_fit(X)

    # Pass 2: transform all
    feats_all = []
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=shuffle, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
        seed=seed,
    )
    for batch in loader:
        ang = batch[2]
        X = ang.view(ang.size(0), -1).float().cpu().numpy()
        Z = ipca.transform(X)
        feats_all.append(torch.from_numpy(Z).float())

    return ipca, torch.cat(feats_all, dim=0)


@torch.no_grad()
def extract_pca_angles_view(
    dataset_with_angles: BatchDictDataset,
    n_components: int = 32,
    batch_size: int = 8192,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Builds an IncrementalPCA view from precomputed angles in the dataset.
    Requires dataset_with_angles to be constructed with return_angles=True.
    Returns [N_samples, n_components] in dataset order (shuffle=False).
    """
    assert getattr(dataset_with_angles, "return_angles", False), \
        "extract_pca_angles_view expects a dataset created with return_angles=True."

    shuffle=False
    if seed is not None:
        shuffle=True

    # Pass 1: partial_fit
    ipca = IncrementalPCA(n_components=n_components)
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=shuffle, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
        seed=seed,
    )
    for batch in loader:
        # expected: x, a, ang, vid
        if len(batch) == 4:
            _, _, ang, _, _ = batch
        else:
            raise RuntimeError("Angles loader must yield (x, a, ang, vid)")
        X = ang.view(ang.size(0), -1).cpu().numpy()  # flatten (T*K*1)
        ipca.partial_fit(X)

    # Pass 2: transform
    feats_all = []
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=shuffle, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
        seed=seed,
    )
    for batch in loader:
        _, _, ang, _, _ = batch
        X = ang.view(ang.size(0), -1).cpu().numpy()
        Z = ipca.transform(X)
        feats_all.append(torch.from_numpy(Z).float())

    return torch.cat(feats_all, dim=0)  # [N, n_components]

@torch.no_grad()
def extract_pca_edges_view(dataset: BatchDictDataset,
                           n_components: int = 16,
                           batch_size: int = 8192,
                           num_workers: int = 0,
                           max_samples: Optional[int] = None,
                           seed: Optional[int] = None) -> torch.Tensor:
    """
    Returns PCA features [N, n_components] for all samples' edge tensor 'a' (T, E, F_edge),
    in order (shuffle=False), using two passes: partial_fit, then transform.

    Notes:
    - This keeps the node PCA (Cell 3) as-is and adds a separate view for edges.
    - If edges have very different scale across datasets, consider pre-standardizing.
    """

    shuffle=False
    if seed is not None:
        shuffle=True

    # 1) Pass 1: fit IncrementalPCA on flattened edges
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=shuffle,
        permute_within_block=False,
        seed=seed,
    )

    ipca = IncrementalPCA(n_components=n_components)
    seen = 0
    for batch in loader:
        x, a, *rest = batch  # keep on CPU for PCA
        B, T, E, F_edge = a.shape
        A = a.view(B, T * E * F_edge).float().numpy()  # (B, D_edges)
        if (max_samples is not None) and (seen >= max_samples):
            break
        if (max_samples is not None) and (seen + B > max_samples):
            A = A[:max(1, max_samples - seen), :]
            B = A.shape[0]
        ipca.partial_fit(A)
        seen += B

    # 2) Pass 2: transform all edges -> PCA
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=shuffle,
        permute_within_block=False,
        seed=seed,
    )
    feats = []
    for batch in loader:
        x, a, *rest = batch
        B, T, E, F_edge = a.shape
        A = a.view(B, T * E * F_edge).float().numpy()
        Z = ipca.transform(A)  # (B, n_components)
        feats.append(torch.from_numpy(Z).float())
    feats_all = torch.cat(feats, dim=0)  # CPU [N, n_components]
    return feats_all

@torch.no_grad()
def extract_supervised_labels_view(
    dataset: BatchDictDataset,
    batch_size: int = 8192,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Extracts the supervised labels 'y' from the dataset for use as a teacher view.
    Returns [N, n_behaviors] tensor.
    """
    if dataset.supervised_dict is None:
        raise ValueError("Dataset does not contain supervised labels.")

    shuffle=False
    if seed is not None:
        shuffle=True
    loader = dataset.make_loader(
        batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=shuffle, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
        seed=seed,
    )
    
    labels_all = []
    # Loader yields (x, a, [ang], [y], vid). We need to find y.
    # The structure depends on return_angles and supervised_dict.
    
    for batch in loader:
        # Unpack based on known structure from BatchDictDataset.__getitem__
        # Basic is (x, a). Optionals follow.
        # We know y is the last optional before vid.
        
        # Easier approach: check tuple length
        # (x, a, vid) -> 3
        # (x, a, ang, vid) -> 4
        # (x, a, y, vid) -> 4
        # (x, a, ang, y, vid) -> 5
        
        has_y = dataset.supervised_dict is not None
        has_angles = dataset.has_angles
        
        if not has_y:
             raise ValueError("Loader did not yield labels.")
             
            # (x, a, ang, y, vid)
        y_batch = batch[2]

            
        # y_batch shape might be (B, 1, F) or (B, F). Squeeze if needed.
        if y_batch.ndim == 3 and y_batch.shape[1] == 1:
            y_batch = y_batch.squeeze(1)
        if y_batch.ndim == 1:
            y_batch = y_batch[:, None]
            
        labels_all.append(y_batch.cpu())

    return torch.cat(labels_all, dim=0)


def run_turtle_teacher_on_views(views_dict: dict,
                                n_components: int,
                                gamma: float = 6.0,
                                alpha_sample_entropy: float = 1.0,
                                outer_steps: int = 200,
                                inner_steps: int = 200,
                                normalize_feats: bool = True,
                                verbose: bool = True,
                                device: Optional[torch.device] = None,
                                head_temp: float = 0.3,
                                task_temp: float = 0.3,
                                batch_size: int = 2048) -> Tuple[Any, torch.Tensor]: # Added batch_size
    
    device = device or torch.device("cpu")
    
    # 1. Filter active views and keep them on CPU
    # active_items = [(k, v) for k, v in views_dict.items() if v is not None]
    # We need to maintain the order carefully to identify the supervised index
    keys = []
    tensors = []
    for k, v in views_dict.items():
        if v is not None:
            keys.append(k)
            tensors.append(v.cpu()) # Ensure CPU
            
    assert len(tensors) > 0, "No active views found."
    
    # 2. Determine supervised index based on the list order
    try:
        supervised_index = keys.index("supervised_labels")
    except ValueError:
        supervised_index = None

    # 3. Create DataLoader (Shuffling is generally good for the teacher)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True)

    # 4. Initialize Teacher
    feature_dims = [t.shape[1] for t in tensors]
    
    teacher = TurtleTeacher(
        feature_dims=feature_dims,
        n_components=n_components,
        gamma=gamma,
        alpha_sample_entropy=alpha_sample_entropy,
        inner_lr=0.1,
        inner_steps=inner_steps,
        head_wd=1e-4,
        head_temp=head_temp,
        task_temp=task_temp,
        normalize_feats=normalize_feats,
        lr_theta=1e-3,
        supervised_index=supervised_index,
        mlp_hidden=32,
        mlp_dropout=0.0,
    ).to(device)

    # 5. Fit (Batch-wise)
    print(f"--- Fitting TurtleTeacher (batch_size={batch_size}) ---")
    teacher.fit(loader, outer_steps=outer_steps, rho=0.04, verbose=verbose)
    
    # 6. Predict full tau_star (Sequential pass)
    # We use a sequential loader to ensure output matches dataset order
    print("--- Computing final τ* assignments ---")
    seq_loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    tau_star = teacher.predict(seq_loader)
    
    return teacher, tau_star


def cache_pca_angles(
    dataset_with_angles: BatchDictDataset,
    cache_path: str,
    n_components: int = 32,
    batch_size: int = 8192,
    num_workers: int = 0,
) -> torch.Tensor:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        return torch.from_numpy(np.array(arr)).float()
    Z = extract_pca_angles_view(dataset_with_angles, n_components, batch_size, num_workers)
    np.save(cache_path, Z.numpy())
    return Z


class DiscriminativeHead(nn.Module):
    """
    Simple linear head on top of z (latent) to predict C clusters (logits).
    """
    def __init__(self, latent_dim: int, n_components: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_components)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)  # logits
    

def maybe_build_turtle_teacher(  
    *,
    teacher_cfg: TurtleTeacherCfg,  
    common_cfg: CommonFitCfg,
    train_dataset: "BatchDictDataset",  
    preprocessed_train: dict,  
    data_path: str, 
    device: torch.device,
    latent_view: Optional[torch.Tensor] = None, #get's calculated externally from the model

) -> Tuple[Optional[Any], Optional[torch.Tensor], Dict[str, Any]]:
    """
    Returns:
      teacher: the TURTLE teacher object (or None)
      tau_star: [N,K] soft assignments (or None)
      views: dict with computed views + PCA objects (for reuse/refresh/inference)
    """

    # Early return in case of no teacher being required
    if not teacher_cfg.use_turtle_teacher:
        return None, None, {}

    views: Dict[str, Any] = {
        "z": None,
        "pca_pos": None,
        "pca_spd": None,
        "pca_edges": None,
        "pca_angles": None,
        "supervised_labels": None,
    }

    # --- Latent view (Mainly for VaDE): include only if explicitly requested ---
    if teacher_cfg.include_latent_view:
        if latent_view is None:
            raise ValueError("include_latent_view=True but latent_view=None")
        views["z"]=latent_view.to(device)

    # --- Nodes view (PCA pos + PCA spd) ---
    if teacher_cfg.include_nodes_view:
        print("\n--- Building PCA views for teacher (nodes) ---")
        _, pca_pos, _, pca_spd = fit_nodes_pca(
            train_dataset,
            n_components_pos=teacher_cfg.pca_nodes_dim,
            n_components_spd=teacher_cfg.pca_nodes_dim,
            batch_size=4096,
            num_workers=0,
        )
        views["pca_pos"] = pca_pos
        views["pca_spd"] = pca_spd

    # --- Edges view (distances between nodes) ---
    if teacher_cfg.include_edges_view:
        print("\n--- Building PCA views for teacher (edges) ---")
        pca_edges = extract_pca_edges_view(
            train_dataset, n_components=teacher_cfg.pca_edges_dim, batch_size=8192, num_workers=0
        )
        views["pca_edges"] = pca_edges

    # --- Angles view (standardized to fit_angles_pca everywhere) ---
    if teacher_cfg.include_angles_view:
        print("\n--- Building PCA views for teacher (angles) ---")
        angles_train_dataset = BatchDictDataset(
            preprocessed_train, data_path, "train_",
            force_rebuild=False, h5_chunk_len=common_cfg.batch_size, return_angles=True
        )
        _, pca_angles_train = fit_angles_pca(
            angles_train_dataset, n_components=teacher_cfg.pca_angles_dim, batch_size=8192, num_workers=0 
        )
        views["pca_angles"] = pca_angles_train

    # --- Supervised view (raw, no thresholding) ---
    if teacher_cfg.include_supervised_view:
        print("\n--- Adding Supervised Labels view for teacher ---")
        if train_dataset.supervised_dict is None:
            raise ValueError("include_supervised_view=True but dataset has no supervised labels.")
        supervised_labels = extract_supervised_labels_view(train_dataset, batch_size=8192, num_workers=0)
        views["supervised_labels"] = supervised_labels

    print("\n--- Running TURTLE teacher on views ---")
    teacher, tau_star = run_turtle_teacher_on_views(
        views_dict=views, n_components=common_cfg.n_components, gamma=teacher_cfg.teacher_gamma,
        alpha_sample_entropy=teacher_cfg.teacher_alpha_sample_entropy, outer_steps=teacher_cfg.teacher_outer_steps,
        inner_steps=teacher_cfg.teacher_inner_steps, normalize_feats=teacher_cfg.teacher_normalize_feats,
        verbose=True, device=device, head_temp=teacher_cfg.teacher_head_temp, task_temp=teacher_cfg.teacher_task_temp,
    )

    return teacher, tau_star.detach(), views


########
# Losses
########

class Dynamic_weight_manager:
    """handles KL and lambda weights over epochs"""
    def __init__(
        self,
        n_batches_per_epoch: int,
        mode: str = "sigmoid",
        warmup_epochs: int = 15,
        max_weight: float = 1.0,
        at_max_epochs: int = 0,
        cooldown_epochs: int = 15,
        end_weight: float = 1.0,
    ):
        self.mode = mode
        self.warmup_iters = max(1, warmup_epochs * n_batches_per_epoch)
        self.at_max_iters = max(0, at_max_epochs * n_batches_per_epoch)
        self.cooldown_iters = max(0, cooldown_epochs * n_batches_per_epoch)
        self.total_iters = self.warmup_iters + self.at_max_iters + self.cooldown_iters
        self.current_iteration = 0
        self.max_weight = float(max_weight)
        self.end_weight = float(end_weight)

    def _shape(self, p: float) -> float:
        p = float(max(0.0, min(1.0, p)))
        if self.mode == "linear":
            return p
        elif self.mode == "sigmoid":
            return 1.0 / (1.0 + math.exp(-12.0 * (p - 0.5)))
        elif self.mode == "tf_sigmoid":
            eps = 1e-2
            denom = max(eps, p - p * p)
            arg = (2.0 * p - 1.0) / denom
            return 1.0 / (1.0 + math.exp(-arg))
        else:
            return p

    def get_weight(self) -> float:
        t = self.current_iteration

        # If at or beyond max schedule, return end weight directly
        if t >= self.total_iters:
            return self.end_weight
        
        # If at max schedule, return max
        if self.at_max_iters > 0 and t >= self.warmup_iters and t < self.warmup_iters + self.at_max_iters:
            return self.max_weight

        # Warmup: 0 -> max_weight
        if t <= self.warmup_iters:
            p = t / self.warmup_iters  # [0,1]
            return self.max_weight * self._shape(p)

        # Cooldown: max_weight -> end_weight
        if self.cooldown_iters <= 0:
            return self.max_weight

        tc = t - (self.warmup_iters + self.at_max_iters)
        pc = tc / self.cooldown_iters  # [0,1]
        # Linear blend 
        return (1.0 - pc) * self.max_weight + pc * self.end_weight

    def step(self):
        self.current_iteration += 1


def cluster_frequencies_regularizer(soft_counts: torch.Tensor) -> torch.Tensor:
    mean_freq = torch.mean(soft_counts, dim=0)
    n_components = soft_counts.shape[1]
    uniform_target = torch.ones(n_components, device=soft_counts.device) / n_components
    kld_loss = nn.KLDivLoss(reduction="batchmean")
    return kld_loss(torch.log(mean_freq + 1e-9), uniform_target)



class VaDELossTFExact(nn.Module):
    """
    TF-exact VaDE loss:
      - Reconstruction NLL
      - KL surrogate (TF): -0.5 * sum(z_log_var + 1) + sum(q log q) [entropy off in pretrain], weighted by kl_weight
      - Clustering agreement: -E[q * softmax(log p(z|c))] (no prior in target)
      - Prior matching: -E[q log pi], pi uniform and fixed
      - KMeans/Gram compactness: passed from model outputs
      - Activity L1 on z_var (Dense 'cluster_variances' output in TF): l1_activity * mean_batch(|z_var|)
    """
    def __init__(self, n_components: int, latent_dim: int,
                l1_activity_weight: float = 0.1,
                kl_scheduler: Optional["Dynamic_weight_manager"] = None,
                reg_cat_clusters=0.0,
                kl_weight=1.0,
                prior_loss_weight=0.0,
                mc_kl_samples=100,          # kept for API, unused in TF-analytic mode #### CHANGES SECTION #####
                kl_clamp_min=0.0,
                lambda_distill: float = 0.0,
                tau_star: Optional[torch.Tensor] = None,
                distill_sharpen_T: float = 0.5,
                distill_conf_weight: bool = False,
                distill_conf_thresh: float = 0.55,
                tf_cluster_weight: float = 0.0,             # weight for TF-style clustering term
                reg_cluster_var_weight: float = 0.0,       # variance equalization weight
                distill_use_class_reweight: bool = True,
                distill_class_reweight_beta: float = 1.0,
                distill_class_reweight_cap: Optional[float] = 3.0,
                nonempty_weight: float = 2e-2,
                nonempty_floor: float = 0.003,
                nonempty_p: float = 2,
                kmeans_loss_weight: float = 0.0
                ):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.l1_activity_weight = float(l1_activity_weight)
        self.kl_scheduler = kl_scheduler 
        self.pretrain_mode = False
        self.kmeans_loss_weight = kmeans_loss_weight

        # Distillation
        self.lambda_distill = float(lambda_distill)
        self.tau_star = tau_star
        self.distill_sharpen_T = float(distill_sharpen_T)
        self.distill_conf_weight = bool(distill_conf_weight)
        self.distill_conf_thresh = float(distill_conf_thresh)

        # TF-style clustering and variance equalization
        self.tf_cluster_weight = float(tf_cluster_weight)
        self.reg_cat_clusters_weight = float(reg_cat_clusters)
        self.reg_cluster_var_weight = float(reg_cluster_var_weight)

        self.distill_use_class_reweight = bool(distill_use_class_reweight)
        self.distill_class_reweight_beta = float(distill_class_reweight_beta)
        self.distill_class_reweight_cap = (None if distill_class_reweight_cap is None
                                           else float(distill_class_reweight_cap))
        self.class_weight: Optional[torch.Tensor] = None  # set in set_teacher()
        self.nonempty_weight = nonempty_weight
        self.nonempty_floor = nonempty_floor
        self.nonempty_p = nonempty_p

        self.mc_kl_samples = 100
        self.kl_clamp_min = 0.0
        self.gmm_logvar_clamp = (-8.0, 8.0)


    def set_teacher(self, tau_star: torch.Tensor, lambda_distill: float = 1.0):
        self.tau_star = tau_star
        self.lambda_distill = float(lambda_distill)

        # NEW: compute inverse-marginal class weights from teacher τ*
        self.class_weight = None
        if self.distill_use_class_reweight and (tau_star is not None):
            with torch.no_grad():
                eps = 1e-8
                pi = tau_star.mean(dim=0).clamp_min(eps)             # (C,)
                w = pi.pow(-self.distill_class_reweight_beta)        # inverse-marginal
                w = w / w.mean()                                     # normalize to mean 1
                if self.distill_class_reweight_cap is not None:
                    w = w.clamp_max(self.distill_class_reweight_cap) # cap to avoid extreme weights
            self.class_weight = w  # move to device at use-time

        if tau_star is not None:
            with torch.no_grad():
                eps = 1e-8
                self.teacher_marginal = tau_star.mean(dim=0).clamp_min(eps)  # (C,)

    @staticmethod
    def _log_normal_diag(x, mean, log_var):
        LOG_2PI = math.log(2.0 * math.pi)
        return -0.5 * (torch.sum(LOG_2PI + log_var + (x - mean) ** 2 * torch.exp(-log_var), dim=-1))

    def _log_mog(self, z, gmm_means, gmm_log_vars, prior, eps=1e-8):
        S, B, D = z.shape
        C = gmm_means.shape[0]
        gmm_log_vars = torch.clamp(gmm_log_vars, min=self.gmm_logvar_clamp[0], max=self.gmm_logvar_clamp[1])
        log_prior = torch.log(torch.clamp(prior, min=eps))
        z_exp = z.unsqueeze(2)               # (S, B, 1, D)
        means = gmm_means.view(1, 1, C, D)   # (1, 1, C, D)
        log_vars = gmm_log_vars.view(1, 1, C, D)
        log_p_z_given_c = self._log_normal_diag(z_exp, means, log_vars)  # (S,B,C)
        log_mix = log_prior.view(1, 1, C)
        log_pz = torch.logsumexp(log_mix + log_p_z_given_c, dim=-1)      # (S,B)
        return log_pz

    def _log_p_z_given_c(self, z: torch.Tensor, gmm_means: torch.Tensor, gmm_log_vars: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z|c) for each class c (diagonal Gaussian).
        z: (B, D), means: (C, D), log_vars: (C, D) -> returns (B, C)
        """
        # clamp for stability
        gmm_log_vars = torch.clamp(gmm_log_vars, min=self.gmm_logvar_clamp[0], max=self.gmm_logvar_clamp[1])
        scale = torch.exp(0.5 * gmm_log_vars).clamp(min=1e-3)
        dist = Normal(gmm_means.unsqueeze(0), scale.unsqueeze(0))  # (1,C,D)
        logp = dist.log_prob(z.unsqueeze(1)).sum(dim=-1)           # (B,C)
        return logp

    def _monte_carlo_kl(self, z_mean, z_log_var, gmm_means, gmm_log_vars, prior):
        # kept for compatibility, no longer used in TF-analytic branch #### CHANGES SECTION #####
        z_log_var = torch.clamp(z_log_var, min=-4.0, max=4.0)
        B, D = z_mean.shape
        S = self.mc_kl_samples
        scale_q = torch.exp(0.5 * z_log_var)
        eps = torch.randn(S, B, D, device=z_mean.device, dtype=z_mean.dtype)
        z_samples = z_mean.unsqueeze(0) + eps * scale_q.unsqueeze(0)  # (S,B,D)
        log_q = self._log_normal_diag(z_samples, z_mean.unsqueeze(0), z_log_var.unsqueeze(0))
        log_p = self._log_mog(z_samples, gmm_means, gmm_log_vars, prior)
        kl = (log_q - log_p).mean()
        if self.kl_clamp_min is not None:
            kl = torch.clamp(kl, min=self.kl_clamp_min)
        return kl

    def forward(self, model_outputs, x_original, batch_indices: Optional[torch.Tensor] = None):
        (recon_dist, latent_z, q, kmeans_loss, z_mean, z_log_var, gmm_params) = model_outputs
        device = z_mean.device
        B, T, N, F = x_original.shape
        x_flat = x_original.view(B, T, N * F)

        # Reconstruction NLL
        with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
            rec_nll = -(recon_dist.log_prob(x_flat.float())).mean()

        # Ensure q is normalized
        if q is not None:
            eps = 1e-8
            q = q.clamp_min(eps)
            q = q / q.sum(dim=-1, keepdim=True)


        # Activity regularizer: average over batch (matches TF effective scaling)
        activity_l1 = self.l1_activity_weight * torch.sum(torch.abs(z_log_var), dim=-1).mean()  #### CHANGES SECTION #####
        # (TF regularizes the 'cluster_variances' Dense output; here we use z_log_var as proxy.)

        # KL weight from scheduler (TF-style)
        klw = 0.0
        if getattr(self, "kl_scheduler", None) is not None:
            klw = float(self.kl_scheduler.get_weight())

        # ---- TF-analytic KL surrogate ---- #### CHANGES SECTION #####
        # loss_variational_1 = -0.5 * sum(z_log_var + 1)
        # loss_variational_2 = sum(q log q)    (entropy term; gated by pretrain in TF via (1 - pretrain))
        z_log_var32 = z_log_var.float()
        if q is not None:
            q32 = q.float()
        else:
            # If q is None, we cannot include the entropy term; fall back to loss_variational_1 only.
            q32 = None

        if self.pretrain_mode:
            # Pretrain: only loss_variational_1 proxy (no q log q), as in TF (they multiply entropy by (1 - pretrain))
            v_max = 2.0 * math.log(2.0) 
            z_var_eff = z_log_var32.clamp_max(v_max) 
            loss_var1 = -1 * (z_var_eff + 1.0).sum(dim=-1).mean()/z_log_var32.shape[-1]  # [B]
            kl_vec = loss_var1
        else:
            loss_var1 = -1 * (z_log_var32 + 1.0).sum(dim=-1)  # [B]
            if q32 is not None:
                loss_var2 = (q32 * q32.clamp_min(1e-8).log()).sum(dim=-1)  # Σ q log q
            else:
                loss_var2 = torch.zeros_like(loss_var1)
            kl_vec = loss_var1 + loss_var2  # [B]

        kl_batch = klw * kl_vec.mean()  # this can be negative, as in TF

        # ----------------------------------------------------------------

        # Init other losses
        tf_cluster = torch.tensor(0.0, device=device, dtype=rec_nll.dtype)
        prior_loss = torch.tensor(0.0, device=device, dtype=rec_nll.dtype)
        cat_cluster_loss = torch.tensor(0.0, device=device, dtype=rec_nll.dtype)
        scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)
        repel_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)
        distill_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)
        temporal_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)
        nonempty_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)

        if not self.pretrain_mode and q is not None:
            z_for = latent_z
            logp = self._log_p_z_given_c(
                z_for.float(),
                gmm_params["means"].float(),
                gmm_params["log_vars"].float(),
            )
            post_like = torch.softmax(logp, dim=-1)
            tf_cluster = -(q * post_like).sum(dim=-1).mean() * self.tf_cluster_weight

            # Prior match (uniform)
            C = self.n_components
            log_pi = math.log(1.0 / max(1, C))
            prior_loss = -(q * log_pi).sum(dim=-1).mean()

            # Cat balance (optional)
            if self.reg_cat_clusters_weight > 0 and q is not None:
                cat_cluster_loss = self.reg_cat_clusters_weight * cluster_frequencies_regularizer(q)

            # Temporal cohesion on q(c|z)
            rho = float(getattr(self, "temporal_cohesion_weight", 0.0))
            if (rho > 0.0) and (q is not None) and (q.size(0) > 1):
                diffs = (q[1:] - q[:-1]).abs().sum(dim=-1).mean()
                temporal_loss = (rho * diffs).to(rec_nll.dtype)

            # Non-empty floor on batch marginal q̄(c)
            nonempty_w = float(getattr(self, "nonempty_weight", 0.0))
            if (nonempty_w > 0.0) and (q is not None):
                p = int(getattr(self, "nonempty_p", 2))
                q_marg = q.mean(dim=0)  # (C,)

                base_floor = float(getattr(self, "nonempty_floor", max(1e-4, 0.05 / max(1, self.n_components))))
                if getattr(self, "teacher_marginal", None) is not None:
                    pi_t = self.teacher_marginal.to(q_marg.device)  # (C,)
                    alpha = 0.9
                    floor_c = torch.max(base_floor * torch.ones_like(pi_t),
                                        alpha * pi_t)
                else:
                    floor_c = base_floor * torch.ones_like(q_marg)

                underuse = (floor_c - q_marg).clamp_min(0.0)
                pen = underuse.pow(p).sum()
                nonempty_loss = (nonempty_w * pen).to(rec_nll.dtype)

            eta = float(getattr(self, "reg_scatter_weight", 3e-2))
            beta = float(getattr(self, "reg_scatter_beta", 1.0))
            if eta > 0.0 and (q is not None):
                qf = q.float()                        # (B,C)
                z = z_mean.float()                    # (B,D)
                pi_b = qf.sum(dim=0).clamp_min(1e-8)  # (C,)
                mu = (qf.t() @ z) / pi_b.unsqueeze(1) # (C,D)
                diff = z.unsqueeze(1) - mu.unsqueeze(0)              # (B,C,D)
                scat_c = (qf.unsqueeze(-1) * diff.pow(2)).sum(dim=0) / pi_b.unsqueeze(1)  # (C,D)
                w = ((pi_b / pi_b.mean()).pow(-beta)).unsqueeze(1)   # (C,1)
                scatter_loss = eta * (w * scat_c).mean()
            else:
                scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)

            # Center repulsion between GMM means
            repel_weight = float(getattr(self, "repel_weight", 8e-2))
            repel_length_scale = float(getattr(self, "repel_length_scale", 1.0))
            if repel_weight > 0.0 and gmm_params["means"].requires_grad:
                with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
                    means = gmm_params["means"].float()  # (C,D)
                    C = means.size(0)
                    diffs = means.unsqueeze(1) - means.unsqueeze(0)      # (C,C,D)
                    D2 = (diffs * diffs).sum(dim=-1)                     # (C,C)
                    Kmat = torch.exp(-D2 / max(1e-9, 2.0 * (repel_length_scale ** 2)))
                    Kmat = Kmat - torch.diag(torch.diag(Kmat))
                    denom = float(max(1, C * C - C))
                    repel_loss = (repel_weight * (Kmat.sum() / denom)).to(rec_nll.dtype)
            else:
                repel_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)

        # Distillation (unchanged)
        if (self.lambda_distill > 0.0) and (self.tau_star is not None) and (batch_indices is not None):
            eps = 1e-8
            tau_batch = self.tau_star[batch_indices]  # (B,C)

            if self.distill_sharpen_T is not None and self.distill_sharpen_T > 0.0:
                logits_t = (tau_batch.clamp_min(eps)).log() / float(self.distill_sharpen_T)
                tau_batch = torch.softmax(logits_t, dim=-1)

            per_sample_ce = -(tau_batch * (q.clamp_min(eps).log())).sum(dim=-1)  # (B,)

            if self.distill_conf_weight:
                conf = tau_batch.max(dim=1).values
                thr = float(self.distill_conf_thresh)
                w_conf = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(min=0.0, max=1.0).detach()
            else:
                w_conf = None

            w_total = None
            if self.distill_use_class_reweight and (self.class_weight is not None):
                w_class = torch.matmul(tau_batch, self.class_weight.to(tau_batch.device))  # (B,)
                w_class = (w_class / w_class.mean().clamp_min(1e-8)).detach()
                w_total = w_class if w_conf is None else (w_class * w_conf)
            else:
                w_total = w_conf

            if w_total is not None:
                distill_loss = (w_total * per_sample_ce).mean()
            else:
                distill_loss = per_sample_ce.mean()

            distill_loss = (self.lambda_distill * distill_loss).to(rec_nll.dtype)
        else:
            distill_loss = torch.tensor(0.0, device=x_original.device, dtype=rec_nll.dtype)

        # KMeans loss
        if not torch.is_tensor(kmeans_loss):
            kmeans_loss = torch.as_tensor(kmeans_loss, device=device, dtype=rec_nll.dtype)
        if not self.pretrain_mode and q is not None:
            kmeans_loss = (self.kmeans_loss_weight * kmeans_loss).to(rec_nll.dtype)
        elif self.pretrain_mode and q is not None:
            kmeans_loss = (1.0 * kmeans_loss).to(rec_nll.dtype)

        # Total loss
        total = (
            rec_nll
            + kl_batch
            + cat_cluster_loss
            + temporal_loss
            + nonempty_loss
            + tf_cluster
            + prior_loss
            + kmeans_loss
            + activity_l1
            + scatter_loss
            + repel_loss
            + distill_loss
        )

        return {
            "total_loss": total,
            "reconstruction_loss": rec_nll,
            "kl_surrogate": kl_batch,                               #### CHANGES SECTION #####
            "kl_divergence": kl_vec.mean(),                         #### CHANGES SECTION #####
            "kl_weight": torch.tensor(klw, device=device, dtype=rec_nll.dtype),
            "tf_cluster_loss": tf_cluster,
            "prior_loss": prior_loss,
            "kmeans_loss": kmeans_loss,
            "activity_l1": activity_l1,
            "cat_cluster_loss": cat_cluster_loss,
            "reg_cluster_var_loss": None,
            "distill_loss": distill_loss,
            "nonempty_loss": nonempty_loss,
            "temporal_loss": temporal_loss,
            "scatter_loss": scatter_loss,
            "repel_loss": repel_loss,
            "student_clustering_loss": None,
        }
    

def _print_losses(model_name:str, epoch: int, n_epochs: int, train_logs: dict, val_logs: dict, klw: int = 0, lambda_d: int = 0,):
    
    if model_name == "vade":
        print(f"Epoch {epoch+1}/{n_epochs} | KLw={klw:.3f}   | λ_distill={lambda_d:.3f}")
        print(f"  Train: total={train_logs.get('total_loss', np.nan):.4f}   | rec={train_logs.get('reconstruction_loss', np.nan):.4f}    | "
                f"kl={train_logs.get('kl_divergence', np.nan):.4f}      | cat={train_logs.get('cat_cluster_loss', np.nan):.4f}    | "
                f"kmeans={train_logs.get('kmeans_loss', np.nan):.4f}      | distill={train_logs.get('distill_loss', np.nan):.4f} |\n"
                f"         temporal={train_logs.get('temporal_loss', np.nan):.4f} | scatter={train_logs.get('scatter_loss', np.nan):.4f} |"
                f" nonempty={train_logs.get('nonempty_loss', np.nan):.4f} | repel={train_logs.get('repel_loss', np.nan):.4f}  |"
                f" clustering={train_logs.get('tf_cluster_loss', np.nan):.4f}")
        print(f"  Val  : total={val_logs.get('total_loss', np.nan):.4f}   | rec={val_logs.get('reconstruction_loss', np.nan):.4f}    | "
                f"kl={val_logs.get('kl_divergence', np.nan):.4f}      | cat={val_logs.get('cat_cluster_loss', np.nan):.4f}    | "
                f"kmeans={val_logs.get('kmeans_loss', np.nan):.4f}      | distill={val_logs.get('distill_loss', np.nan):.4f} |\n"
                f"         temporal={val_logs.get('temporal_loss', np.nan):.4f} | scatter={val_logs.get('scatter_loss', np.nan):.4f} |"
                f" nonempty={val_logs.get('nonempty_loss', np.nan):.4f} | repel={val_logs.get('repel_loss', np.nan):.4f}  |"
                f" clustering={val_logs.get('tf_cluster_loss', np.nan):.4f}")
        print(f"  Align: conf={train_logs.get('conf_norm',np.nan):.3f} | bal={train_logs.get('bal_norm',np.nan):.3f} | score={train_logs.get('alignment_score'):.3f}")
    elif model_name == "Contrastive":
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train: total={train_logs.get('total_loss', np.nan):.4f} | rec={train_logs.get('reconstruction_loss', np.nan):.4f} | "
                f"kl={train_logs.get('kl_divergence', np.nan):.4f} | cat={train_logs.get('cat_cluster_loss', np.nan):.4f} | "
                f"kmeans={train_logs.get('kmeans_loss', np.nan):.4f} | distill={train_logs.get('distill_loss', np.nan):.4f} |\n"
                f"         temporal={train_logs.get('temporal_loss', np.nan):.4f} | scatter={train_logs.get('scatter_loss', np.nan):.4f} |"
                f" nonempty={train_logs.get('nonempty_loss', np.nan):.4f} | repel={train_logs.get('repel_loss', np.nan):.4f}  |"
                f" clustering={train_logs.get('tf_cluster_loss', np.nan):.4f}")
        print(f"  Val  : total={val_logs.get('total_loss', np.nan):.4f} | rec={val_logs.get('reconstruction_loss', np.nan):.4f} | "
                f"kl={val_logs.get('kl_divergence', np.nan):.4f} | cat={val_logs.get('cat_cluster_loss', np.nan):.4f} | "
                f"kmeans={val_logs.get('kmeans_loss', np.nan):.4f} | distill={val_logs.get('distill_loss', np.nan):.4f} |\n"
                f"         temporal={val_logs.get('temporal_loss', np.nan):.4f} | scatter={val_logs.get('scatter_loss', np.nan):.4f} |"
                f" nonempty={val_logs.get('nonempty_loss', np.nan):.4f} | repel={val_logs.get('repel_loss', np.nan):.4f}  |"
                f" clustering={val_logs.get('tf_cluster_loss', np.nan):.4f}")
        print(f"  Align: conf={train_logs.get('conf_norm',np.nan):.3f} | bal={train_logs.get('bal_norm',np.nan):.3f} | score={train_logs.get('alignment_score'):.3f}")
    else:
        print("")


##############
# Optimization
##############

def build_optimizer_generic(
    model: nn.Module,
    distill_head: Optional[nn.Module] = None,
    base_lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    params = list(unwrap_dp(model).parameters())
    if distill_head is not None:
        params += list(distill_head.parameters())
    return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)


def build_optimizer(
    model: nn.Module,
    base_lr: float = 3e-4,
    gmm_lr: float = 1e-4,    #3e-4 * 0.33       
) -> torch.optim.Optimizer:
    m = unwrap_dp(model)
    gmm_params = [m.latent_space.gmm_means, m.latent_space.gmm_log_vars]
    if hasattr(m.latent_space, "prior") and isinstance(m.latent_space.prior, torch.nn.Parameter):
        gmm_params.append(m.latent_space.prior)
    gmm_ids = {id(p) for p in gmm_params}
    base_params = [p for p in m.parameters() if id(p) not in gmm_ids]
    return torch.optim.Adam(
        [
            {"params": base_params, "lr": base_lr},
            {"params": gmm_params, "lr": gmm_lr},
        ]
    )


##########
# Training
##########

@dataclass
class StepResult:
    loss: torch.Tensor
    logs: Dict[str, float]

def move_to(x, device):
    if isinstance(x, (list, tuple)):
        return type(x)(move_to(xx, device) for xx in x)
    if isinstance(x, Mapping):
        return {k: move_to(v, device) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x

def average_logs(logs_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, Tuple[float, int]] = {}
    for logs in logs_list:
        for k, v in logs.items():
            s, n = out.get(k, (0.0, 0))
            out[k] = (s + float(v), n + 1)
    return {k: s / max(n, 1) for k, (s, n) in out.items()}

def _progress_wrap(dataloader, show: bool, desc: str, leave=False):
    return tqdm(dataloader, desc=desc, mininterval=0.5, leave=leave) if show else dataloader

def _format_postfix(logs: Dict[str, float], max_items: int = 4) -> Dict[str, str]:
    priority = [
        "total_loss",
        "reconstruction_loss", "enc_rec_loss",
        "kl_divergence",
        "vq_loss", "kmeans_loss",
        "pos_similarity", "neg_similarity",
    ]
    out: Dict[str, str] = {}
    for k in priority:
        if k in logs:
            out[k] = f"{logs[k]:.4f}"
            if len(out) >= max_items:
                return out
    for k, v in logs.items():
        if k not in out:
            out[k] = f"{v:.4f}"
            if len(out) >= max_items:
                break
    return out


def apply_decoder_schedule(model: nn.Module, epoch: int):
    base = unwrap_dp(model)
    dec = getattr(base, "decoder", None)
    if dec is None:
        return
    if hasattr(dec, "teacher_forcing_mode"):
        if epoch < 3:
            dec.teacher_forcing_mode = "zeros"
            if hasattr(dec, "input_dropout_p"):
                dec.input_dropout_p = 0.0
        else:
            dec.teacher_forcing_mode = "dropout"
            if hasattr(dec, "input_dropout_p"):
                dec.input_dropout_p = 0.5


def train_one_epoch_indexed(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[nn.Module, Any, SimpleNamespace], StepResult],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    grad_clip_value: Optional[float] = 0.75,
    ctx: Optional[SimpleNamespace] = None,
    show_progress: bool = True,
    leave: bool=False,
) -> Dict[str, float]:
    model.train()
    if ctx and getattr(ctx, "criterion", None) is not None:
        ctx.criterion.train()

    logs_accum = []
    iterator = _progress_wrap(dataloader, show_progress, desc=f"Train {epoch+1}/{num_epochs}", leave=leave)

    seen = 0
    mean_kl_weight = 0.0
    mean_lambda_weight = 0.0
    for step, batch in enumerate(iterator):
        batch = move_to(batch, device)
        x, a, _, idx, _ = batch  # assume dataset returns (x, a)
        B = x.size(0)
        #idx = torch.arange(seen, seen + B, device=x.device, dtype=torch.long)
        seen += B

        batch_idx = (x, a, idx)

        if scaler is not None and use_amp and device.type == "cuda":
            with autocast(device_type=device.type, dtype=torch.float16):
                res = step_fn(model, batch_idx, SimpleNamespace(
                    train=True, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
                ))
            scaler.scale(res.loss).backward()
            print(scaler.get_scale())
            
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            if torch.isnan(model.encoder.node_recurrent_block.conv1d.weight).any():
                print("z issues!")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            res = step_fn(model, batch_idx, SimpleNamespace(
                train=True, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
            ))
            optimizer.zero_grad(set_to_none=True)
            res.loss.backward()
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            optimizer.step()

        if ctx is not None:
            if hasattr(ctx, "kl_scheduler") and ctx.kl_scheduler is not None:
                ctx.kl_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_kl_weight=ctx.kl_scheduler.get_weight()
            if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
                ctx.lambda_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_lambda_weight=ctx.lambda_scheduler.get_weight()

        logs_accum.append(res.logs)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(_format_postfix(res.logs), refresh=False)        

    return average_logs(logs_accum), mean_kl_weight, mean_lambda_weight


@torch.no_grad()
def validate_one_epoch_indexed(
    model: nn.Module,
    dataloader: DataLoader,
    step_fn: Callable[[nn.Module, Any, SimpleNamespace], StepResult],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    ctx: Optional[SimpleNamespace] = None,
    show_progress: bool = True,
    leave: bool=False,
) -> Dict[str, float]:
    model.eval()
    if ctx and getattr(ctx, "criterion", None) is not None:
        ctx.criterion.eval()

    logs_accum = []
    iterator = _progress_wrap(dataloader, show_progress, desc=f"Val {epoch+1}/{num_epochs}", leave=leave)

    seen = 0
    for batch in iterator:
        batch = move_to(batch, device)
        x, a, _, idx, _ = batch
        B = x.size(0)
        #idx = torch.arange(seen, seen + B, device=x.device, dtype=torch.long)
        seen += B

        res = step_fn(model, (x, a, idx), SimpleNamespace(
            train=False, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
        ))
        logs_accum.append(res.logs)

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(_format_postfix(res.logs), refresh=False)

    return average_logs(logs_accum)


def build_model_and_step(
    input_shape: Tuple[int, int, int],
    edge_feature_shape: Tuple[int, int, int],
    adjacency_matrix: np.ndarray,
    latent_dim: int,
    n_components: int,
    encoder_type: str,
    use_gnn: bool,
    interaction_regularization: float,
    kmeans_loss: float,
    device: torch.device,
) -> Tuple[nn.Module, Callable]:
    model = deepof.clustering.models_new.VaDEPT(
        input_shape=input_shape,
        edge_feature_shape=edge_feature_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=latent_dim,
        n_components=n_components,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        kmeans_loss=kmeans_loss,
        interaction_regularization=interaction_regularization,
    ).to(device, non_blocking=True)
    return model, step_vade


def step_vade(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],  # (x, a, idx or None)
    ctx: SimpleNamespace,
) -> StepResult:
    x, a, idx = batch


    base_m = unwrap_dp(model)
    T_high = float(getattr(ctx, "resp_temp_start", 1.8))
    T_low  = 1.0
    T_resp = T_high
    if hasattr(ctx, "kl_scheduler") and (ctx.kl_scheduler is not None):
        w = float(ctx.kl_scheduler.get_weight())
        w_max = float(getattr(ctx.kl_scheduler, "max_weight", 1.0))
        prog = 0.0 if w_max <= 0 else min(1.0, max(0.0, w / w_max))
        T_resp = T_low + (T_high - T_low) * (1.0 - prog)
    # Clamp and set
    T_resp = float(max(1.0, min(T_high, T_resp)))
    if hasattr(base_m, "latent_space"):
        setattr(base_m.latent_space, "responsibility_temp", T_resp)


    outputs = model(x, a, return_gmm_params=True)

    if hasattr(ctx, "criterion") and ctx.criterion is not None:
        apply_distill = getattr(ctx, "apply_distill", True)
        batch_indices = idx if apply_distill else None

        if hasattr(ctx, "kl_scheduler") and ctx.kl_scheduler is not None:
            ctx.criterion.kl_weight = float(ctx.kl_scheduler.get_weight())
        elif hasattr(ctx, "kl_weight"):
            ctx.criterion.kl_weight = float(ctx.kl_weight)
        if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
            ctx.criterion.lambda_distill = float(ctx.lambda_scheduler.get_weight())

        loss_dict = ctx.criterion(outputs, x, batch_indices=batch_indices)
        total = loss_dict["total_loss"]
        

        # EMA update of π from current batch responsibilities 
        with torch.no_grad():
            beta = float(getattr(ctx, "pi_ema_beta", 0.0))
            if beta > 0.0 and outputs[2] is not None:
                q_batch = outputs[2].detach()                 # (B,C)
                m = q_batch.mean(dim=0)                       # (C,)
                base = unwrap_dp(model)
                if hasattr(base.latent_space, "prior"):
                    pi = base.latent_space.prior
                    if isinstance(pi, torch.nn.Parameter):
                        pi_data = pi.data
                        pi_data.mul_(1.0 - beta).add_(beta * m)
                        pi_data.clamp_(min=1e-6)
                        pi_data.div_(pi_data.sum())
                    else:
                        pi.mul_(1.0 - beta).add_(beta * m)
                        pi.clamp_(min=1e-6)
                        pi.div_(pi.sum())

        logs = {
            "total_loss": float(total.detach().item()),
            "reconstruction_loss": float(loss_dict["reconstruction_loss"].detach().item()),
            "kl_divergence": float(loss_dict["kl_divergence"].detach().item()),
            "cat_cluster_loss": float(loss_dict["cat_cluster_loss"].detach().item()),
            "kmeans_loss": float(loss_dict["kmeans_loss"].detach().item()) if torch.is_tensor(loss_dict["kmeans_loss"]) else float(loss_dict["kmeans_loss"]),
            "prior_loss": float(loss_dict["prior_loss"].detach().item()),
            "distill_loss": float(loss_dict["distill_loss"].detach().item()),
            "student_clustering_loss": 0.0, #float(loss_dict["student_clustering_loss"].detach().item()),
            "tf_cluster_loss" : float(loss_dict["tf_cluster_loss"].detach().item()),
            "nonempty_loss": float(loss_dict["nonempty_loss"].detach().item()),
            "temporal_loss": float(loss_dict["temporal_loss"].detach().item()),
            "scatter_loss": float(loss_dict["scatter_loss"].detach().item()),
            "repel_loss": float(loss_dict["repel_loss"].detach().item()),
        }
        return StepResult(loss=total, logs=logs)
    else:
        raise RuntimeError("VaDE step requires ctx.criterion (VaDELoss) to be provided.")


def step_vqvae_distill(
    model: nn.Module,  # VQVAEPT or DataParallel(VQVAEPT)
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, a, idx)
    ctx: SimpleNamespace,
) -> StepResult:
    x, a, idx = batch
    device = x.device
    apply_distill = getattr(ctx, "apply_distill", True)

    # Forward (ask for all to get z_e)
    encoding_recon, recon, quant, soft_counts, encoder_output, vq_losses = model(
        x, a, return_losses=True, return_all_outputs=True
    )

    # Flatten targets to match decoder distribution event shape
    B, T, N, F = x.shape
    x_flat = x.view(B, T, N * F)

    with torch.amp.autocast(device_type=device.type, enabled=False):
        enc_rec_loss = -(encoding_recon.log_prob(x_flat.float())).mean()
        rec_loss     = -(recon.log_prob(x_flat.float())).mean()

    vq_loss     = float(vq_losses.get("vq_loss", 0.0))
    kmeans_loss = float(vq_losses.get("kmeans_loss", 0.0))
    vq_total_t  = torch.as_tensor(vq_loss + kmeans_loss, device=device, dtype=enc_rec_loss.dtype)

    base_total = enc_rec_loss + rec_loss + vq_total_t
    distill_loss = torch.tensor(0.0, device=device, dtype=enc_rec_loss.dtype)

    # Distillation on z_e
    if apply_distill and hasattr(ctx, "distill_head") and (getattr(ctx, "lambda_distill", 0.0) > 0.0):
        z_e = encoder_output  # pre-quantization latent
        logits = ctx.distill_head(z_e)  # [B, C]

        idx = batch[-2].to(device).long() 
        tau_b = ctx.tau_star[idx]  # (B, C)
        eps = 1e-8
        T = float(getattr(ctx, "distill_sharpen_T", 0.5))
        if T > 0.0:
            logits_t = (tau_b.clamp_min(eps)).log() / T
            tau_b = torch.softmax(logits_t, dim=-1)

        if getattr(ctx, "distill_conf_weight", False):
            conf = tau_b.max(dim=1).values
            thr = float(getattr(ctx, "distill_conf_thresh", 0.6))
            w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(0.0, 1.0).detach()
            per_sample = _soft_ce_logits(logits, tau_b, reduction="none")
            distill_loss = (w * per_sample).mean()
        else:
            distill_loss = _soft_ce_logits(logits, tau_b, reduction="mean")
        lambda_distill = 0.0
        if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
            lambda_distill = float(ctx.lambda_scheduler.get_weight())
        
        distill_loss = lambda_distill * distill_loss
        total = base_total + distill_loss
    else:
        total = base_total

    # Populated clusters (codes used)
    try:
        populated = soft_counts.argmax(dim=-1).unique().numel()
        populated_f = float(populated)
    except Exception:
        populated_f = float("nan")

    logs = {
        "total_loss": float(total.detach().item()),
        "enc_rec_loss": float(enc_rec_loss.detach().item()),
        "reconstruction_loss": float(rec_loss.detach().item()),
        "vq_loss": vq_loss,
        "kmeans_loss": kmeans_loss,
        "number_of_populated_clusters": populated_f,
        "distill_loss": float(distill_loss.detach().item()) if torch.is_tensor(distill_loss) else float(distill_loss),
    }
    return StepResult(loss=total, logs=logs)


def _soft_ce_logits(logits: torch.Tensor, soft_targets: torch.Tensor, eps: float = 1e-8, reduction: str = "mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    soft_targets = torch.clamp(soft_targets, min=eps, max=1.0)
    per_sample = -(soft_targets * log_probs).sum(dim=-1)
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample


def step_contrastive_distill(
    model: nn.Module,  # ContrastivePT or DataParallel(ContrastivePT)
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, a, idx)
    ctx: SimpleNamespace,
) -> StepResult:
    x, a, idx = batch
    base = unwrap_dp(model)
    device = x.device
    win = base.window_size
    apply_distill = getattr(ctx, "apply_distill", True)

    def ts_samples(mb, w):
        pos = mb[:, 1:w + 1]
        neg = mb[:, -w:]
        return pos, neg

    pos_x, neg_x = ts_samples(x, win)
    pos_a, neg_a = ts_samples(a, win)

    # Encode via forward for DP compatibility
    z_pos = model(pos_x, pos_a)
    z_neg = model(neg_x, neg_a)

    # Normalize row-wise
    z_pos = torch.nn.functional.normalize(z_pos, dim=1)
    z_neg = torch.nn.functional.normalize(z_neg, dim=1)

    # Base contrastive loss
    loss, pos_mean, neg_mean = select_contrastive_loss_pt(
        z_pos, z_neg,
        similarity=base.similarity_function,
        loss_fn=base.loss_function,
        temperature=base.temperature,
        tau=base.tau,
        beta=base.beta,
        elimination_topk=0.1,
    )

    distill_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)

    # Distillation on the main window embedding
    if apply_distill and hasattr(ctx, "distill_head") and (getattr(ctx, "lambda_distill", 0.0) > 0.0):
        z_main = model(x, a)  # [B, D]
        logits = ctx.distill_head(z_main)  # [B, C]

        idx = batch[-2].to(device).long() 
        tau_b = ctx.tau_star[idx]  # (B, C)
        eps = 1e-8
        T = float(getattr(ctx, "distill_sharpen_T", 0.5))
        if T > 0.0:
            logits_t = (tau_b.clamp_min(eps)).log() / T
            tau_b = torch.softmax(logits_t, dim=-1)

        if getattr(ctx, "distill_conf_weight", False):
            conf = tau_b.max(dim=1).values
            thr = float(getattr(ctx, "distill_conf_thresh", 0.6))
            w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(0.0, 1.0).detach()
            per_sample = _soft_ce_logits(logits, tau_b, reduction="none")
            distill_loss = (w * per_sample).mean()
        else:
            distill_loss = _soft_ce_logits(logits, tau_b, reduction="mean")
        lambda_distill = 0.0
        if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
            lambda_distill = float(ctx.lambda_scheduler.get_weight())

        distill_loss = lambda_distill * distill_loss
        total = loss + distill_loss
    else:
        total = loss

    logs = {
        "total_loss": float(total.detach().item()),
        "pos_similarity": float(pos_mean),
        "neg_similarity": float(neg_mean),
        "distill_loss": float(distill_loss.detach().item()) if torch.is_tensor(distill_loss) else float(distill_loss),
    }
    return StepResult(loss=total, logs=logs)


###########
# Main call
###########


def embedding_model_fittingPT(
    preprocessed_object: Tuple[dict, dict],
    adjacency_matrix: np.ndarray,
    encoder_type: str,
    batch_size: int,
    latent_dim: int,
    epochs: int,
    n_components: int,
    # Logging/IO
    output_path: str,
    log_history: bool = True,
    data_path: str = ".",
    pretrained: Optional[str] = None,
    save_weights: bool = True,
    run: int = 0,
    # VaDE-specific
    kl_annealing_mode: str = "sigmoid",
    reg_cat_clusters: float = 0.0,
    recluster: bool = False,
    freeze_gmm_epochs: int = 0,
    freeze_decoder_epochs: int = 0,
    prior_loss_weight: float = 0.0,
    # Regularization knobs
    interaction_regularization: float = 0.0,
    kmeans_loss: float = 0.0,
    # System
    num_workers: int = 0,
    prefetch_factor: int = 0,
    use_amp: bool = False,
    # TURTLE teacher + distillation (VaDE)
    use_turtle_teacher: bool = True,
    teacher_gamma: float = 8.0,
    teacher_outer_steps: int = 500,
    teacher_inner_steps: int = 100,
    teacher_normalize_feats: bool = True,
    lambda_distill: float = 4.0,
    lambda_decay_start: int = 10,
    lambda_end_weight: float = 0.2,
    lambda_cooldown: int = 10,
    teacher_refresh_every: Optional[int] = False,
    teacher_freeze_at: Optional[int] = 10,
    teacher_head_temp: float = 0.5,
    teacher_task_temp: float = 0.5,
    teacher_alpha_sample_entropy: float = 2.0,
    # KL cap
    kl_max_weight: float = 1,
    kl_warmup: int = 5,
    kl_end_weight: float = 0.2,
    kl_cooldown: int = 5,
    reg_scatter_weight: float = 0,
    temporal_cohesion_weight: float = 0,
    reg_scatter_beta: float = 1.0,
    repel_weight: float = 0,
    repel_length_scale: float = 1.0,
    # TF-style cluster term
    tf_cluster_weight: float = 0.0,
    nonempty_weight: float = 2e-2,
    # Distillation weighting (VaDE)
    distill_conf_weight: bool = False,
    distill_conf_thresh: float = 0.3,
    distill_sharpen_T: float = 0.5,
    # Views for teacher
    include_edges_view: bool = False,
    include_nodes_view: bool = True,
    pca_nodes_dim: int = 32,
    pca_edges_dim: int = 32,
    include_angles_view: bool = True,
    pca_angles_dim: int = 32,
    include_supervised_view: bool = True,
    reinit_gmm_on_refresh: bool = False,
    # Diagnostics
    diag_max_batches: int = 4,
    # Model type
    model_name: str = "VaDE",   # "VaDE" (default), "VQVAE", "Contrastive"
    # Distill head (for VQVAE/Contrastive)
    generic_lambda_distill: float = 2.0,
    generic_distill_sharpen_T: float = 0.5,
    generic_distill_conf_weight: bool = True,
    generic_distill_conf_thresh: float = 0.6,
    generic_distill_warmup_epochs: int = 1,
    distill_class_reweight_beta: float = 1,
    distill_class_reweight_cap: float = 3,
    # Contrastive opts
    temperature: float = 0.1,
    contrastive_similarity_function: str = "cosine",
    contrastive_loss_function: str = "nce",
    beta: float = 0.1,
    tau: float = 0.1,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    
    # Verify if various model inputs have valid values (TO DO)
    #_check_model_inputs()

    # Create configs for different models to avoid gigantic function signaturs
    common_cfg = CommonFitCfg(
        model_name=model_name.lower(),
        encoder_type=encoder_type,
        batch_size=batch_size,
        latent_dim=latent_dim,
        epochs=epochs,
        n_components=n_components,
        output_path=output_path,
        data_path=data_path,
        log_history=log_history,
        pretrained=pretrained,
        save_weights=save_weights,
        run=run,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        use_amp=use_amp,
        interaction_regularization=interaction_regularization,
        kmeans_loss=kmeans_loss,
        diag_max_batches=diag_max_batches,
        kl_annealing_mode=kl_annealing_mode,
        kl_max_weight=kl_max_weight,
        kl_warmup=kl_warmup,
        kl_end_weight=kl_end_weight,
        kl_cooldown=kl_cooldown,
        seed=0, 
    )

    teacher_cfg = TurtleTeacherCfg(
        use_turtle_teacher=use_turtle_teacher,
        teacher_gamma=teacher_gamma,
        teacher_outer_steps=teacher_outer_steps,
        teacher_inner_steps=teacher_inner_steps,
        teacher_normalize_feats=teacher_normalize_feats,
        teacher_head_temp=teacher_head_temp,
        teacher_task_temp=teacher_task_temp,
        teacher_alpha_sample_entropy=teacher_alpha_sample_entropy,

        lambda_distill=lambda_distill,
        lambda_decay_start=lambda_decay_start,
        lambda_end_weight=lambda_end_weight,
        lambda_cooldown=lambda_cooldown,
        distill_sharpen_T=distill_sharpen_T,
        distill_conf_weight=distill_conf_weight,
        distill_conf_thresh=distill_conf_thresh,

        generic_lambda_distill=generic_lambda_distill,
        generic_distill_sharpen_T=generic_distill_sharpen_T,
        generic_distill_conf_weight=generic_distill_conf_weight,
        generic_distill_conf_thresh=generic_distill_conf_thresh,
        generic_distill_warmup_epochs=generic_distill_warmup_epochs,
        distill_class_reweight_beta=distill_class_reweight_beta,
        distill_class_reweight_cap=distill_class_reweight_cap,

        include_edges_view=include_edges_view,
        include_nodes_view=include_nodes_view,
        include_angles_view=include_angles_view,
        include_supervised_view=include_supervised_view,
        pca_nodes_dim=pca_nodes_dim,
        pca_edges_dim=pca_edges_dim,
        pca_angles_dim=pca_angles_dim,

        # normalize "False" -> None 
        teacher_refresh_every=(None if teacher_refresh_every is False else teacher_refresh_every),
        teacher_freeze_at=teacher_freeze_at,
        reinit_gmm_on_refresh=reinit_gmm_on_refresh,
    )

    vade_cfg = VaDECfg(
        reg_cat_clusters=reg_cat_clusters,
        recluster=recluster,
        freeze_gmm_epochs=freeze_gmm_epochs,
        freeze_decoder_epochs=freeze_decoder_epochs,
        prior_loss_weight=prior_loss_weight,

        reg_scatter_weight=reg_scatter_weight,
        temporal_cohesion_weight=temporal_cohesion_weight,
        reg_scatter_beta=reg_scatter_beta,
        repel_weight=repel_weight,
        repel_length_scale=repel_length_scale,

        tf_cluster_weight=tf_cluster_weight,
        nonempty_weight=nonempty_weight,
    )

    contrastive_cfg = ContrastiveCfg(
        temperature=temperature,
        contrastive_similarity_function=contrastive_similarity_function,
        contrastive_loss_function=contrastive_loss_function,
        beta=beta,
        tau=tau,
    )

    return embedding_model_fitting(preprocessed_object, adjacency_matrix, common_cfg=common_cfg, teacher_cfg=teacher_cfg, vade_cfg=vade_cfg, contrastive_cfg=contrastive_cfg)


# Unified checkpoint paths per model/run
def _ckpt_paths(model_name: str, common_cfg : CommonFitCfg):
    ckpt_dir = os.path.join(common_cfg.output_path, "models", model_name.lower(), f"run_{common_cfg.run}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_path = os.path.join(ckpt_dir, "best_model_val.pth")
    best_score_path = os.path.join(ckpt_dir, "best_model_score.pth")
    teacher_init_path = os.path.join(ckpt_dir, "model_teacher_init.pth")
    return ckpt_dir, best_val_path, best_score_path, teacher_init_path
    

def embedding_model_fitting(
    preprocessed_object: Tuple[dict, dict],
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    vade_cfg: VaDECfg,
    contrastive_cfg: ContrastiveCfg,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:


    # ----------------------------------------------------
    # Prepare device and data
    # ----------------------------------------------------
    model_name = common_cfg.model_name # Name defaults to "vade"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(common_cfg.seed)
    np.random.seed(common_cfg.seed)

    data_path = os.path.join(common_cfg.output_path, "Datasets")
    preprocessed_train, preprocessed_val, supervised_train, supervised_val = preprocessed_object
    train_dataset = BatchDictDataset(
        preprocessed_train, data_path, "train_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=supervised_train
    )
    val_dataset = BatchDictDataset(
        preprocessed_val, data_path, "val_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=supervised_val
    )

    train_loader = train_dataset.make_loader(
        batch_size=common_cfg.batch_size, shuffle=True, num_workers=common_cfg.num_workers, drop_last=False,
        iterable_for_h5=True, pin_memory=(device.type == 'cuda'), prefetch_factor=common_cfg.prefetch_factor,
        persistent_workers=(common_cfg.num_workers > 0), block_shuffle=True, permute_within_block=False, seed=common_cfg.seed,
    )
    val_loader = val_dataset.make_loader(
        batch_size=common_cfg.batch_size, shuffle=False, num_workers=common_cfg.num_workers, drop_last=False,
        iterable_for_h5=True, pin_memory=(device.type == 'cuda'), prefetch_factor=common_cfg.prefetch_factor,
        persistent_workers=(common_cfg.num_workers > 0), block_shuffle=False, permute_within_block=False,
    )

    writer = None
    if common_cfg.log_history:
        log_dir = os.path.join(common_cfg.output_path, "logs", f"{model_name}_run_{common_cfg.run}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs -> {log_dir}")


    # ----------------------------------------------------
    # Branch 1: VQVAE
    # ----------------------------------------------------
    if model_name == "vqvae":
        return fit_VQVAE(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            common_cfg,
            teacher_cfg,
            writer,
        )

    # ----------------------------------------------------
    # Branch 2: Contrastive
    # ----------------------------------------------------
    elif model_name == "contrastive":
        return fit_contrastive(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            common_cfg,
            teacher_cfg,
            contrastive_cfg,
            writer,
        )

    # ----------------------------------------------------
    # Branch 3: VaDE
    # ----------------------------------------------------
    elif model_name == "vade":
        return fit_VADE(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            common_cfg,
            teacher_cfg,
            vade_cfg,
            writer,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")



def fit_VQVAE(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    # Create model
    model = VQVAEPT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        n_components=common_cfg.n_components,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        interaction_regularization=common_cfg.interaction_regularization,
        kmeans_loss=common_cfg.kmeans_loss,
    ).to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Create teacher
    teacher_cfg.include_latent_view=False
    teacher, tau_star, teacher_views = maybe_build_turtle_teacher(
        teacher_cfg=teacher_cfg,
        common_cfg=common_cfg,
        train_dataset=train_loader.dataset,
        preprocessed_train=preprocessed_train,
        data_path=data_path,
        device=device,
        latent_view=None,
    )

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler=None
    if not apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0, at_max_epochs=teacher_cfg.lambda_decay_start, max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown, end_weight=teacher_cfg.lambda_end_weight
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = _ckpt_paths("vqvae", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    score_value = 0

    print(f"\n--- Training VQVAE for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=(tau_star.to(device) if tau_star is not None else None),
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, dataloader=train_loader, optimizer=optimizer, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=True,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        # To do: calculate score

        # Print training progress
        print(f"Epoch {epoch+1}/{common_cfg.epochs} | Train total={train_logs.get('total_loss', np.nan):.4f} | "
                f"Val total={v_total:.4f} | λ_distill={lam:.2f}")              
        _print_losses(model_name="vqvae", epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)

        # Write training progress
        if writer:
            for k, v in train_logs.items(): writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_logs.items(): writer.add_scalar(f"Val/{k}", v, epoch)
            writer.add_scalar("Distill/lambda", lam, epoch)

        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_val)
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=None,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        if score_value > best_score:
            best_score = score_value
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_score)
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")
        

    # Load states of best val and score models
    model_score = deepcopy(model)
    if common_cfg.save_weights and os.path.exists(best_path_val):
        unwrap_dp(model).load_state_dict(torch.load(best_path_val, map_location=device))
    if common_cfg.save_weights and os.path.exists(best_path_score):
        unwrap_dp(model_score).load_state_dict(torch.load(best_path_score, map_location=device))

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model), unwrap_dp(model_score), None


def fit_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    contrastive_cfg: ContrastiveCfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    # Create model
    model = deepof.clustering.models_new.ContrastivePT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        similarity_function=contrastive_cfg.contrastive_similarity_function,
        loss_function=contrastive_cfg.contrastive_loss_function,
        temperature=contrastive_cfg.temperature,
        beta=contrastive_cfg.beta,
        tau=contrastive_cfg.tau,
    ).to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Create teacher
    teacher_cfg.include_latent_view=False
    teacher, tau_star, teacher_views = maybe_build_turtle_teacher(
        teacher_cfg=teacher_cfg,
        common_cfg=common_cfg,
        train_dataset=train_loader.dataset,
        preprocessed_train=preprocessed_train,
        data_path=data_path,
        device=device,
        latent_view=None,
    )

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler=None
    if not apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0, at_max_epochs=teacher_cfg.lambda_decay_start, max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown, end_weight=teacher_cfg.lambda_end_weight
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = _ckpt_paths("contrastive", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    score_value = 0

    print(f"\n--- Training Contrastive for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=(tau_star.to(device) if tau_star is not None else None),
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, dataloader=train_loader, optimizer=optimizer, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=True,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        # To do: calculate score

        # Print training progress
        print(f"Epoch {epoch+1}/{common_cfg.epochs} | Train total={train_logs.get('total_loss', np.nan):.4f} | "
                f"Val total={v_total:.4f} | λ_distill={lam:.2f}")
        _print_losses(model_name="Contrastive", epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)

        # Write training progress
        if writer:
            for k, v in train_logs.items(): writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_logs.items(): writer.add_scalar(f"Val/{k}", v, epoch)
            writer.add_scalar("Distill/lambda", lam, epoch)
   
        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_val)
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=None,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    contrastive_cfg=contrastive_cfg,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        if score_value > best_score:
            best_score = score_value
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_score)
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    contrastive_cfg=contrastive_cfg,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")


    # Load states of best val and score models
    model_score = deepcopy(model)
    if common_cfg.save_weights and os.path.exists(best_path_val):
        unwrap_dp(model).load_state_dict(torch.load(best_path_val, map_location=device))
    if common_cfg.save_weights and os.path.exists(best_path_score):
        unwrap_dp(model_score).load_state_dict(torch.load(best_path_score, map_location=device))

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model), unwrap_dp(model_score), None    


def fit_VADE(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    vade_cfg: VaDECfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    # Create model and step function
    model, step_fn = build_model_and_step(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        n_components=common_cfg.n_components,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        interaction_regularization=common_cfg.interaction_regularization,
        kmeans_loss=1.0,
        device=device,
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # More setup
    optimizer = build_optimizer(model=model, base_lr=1e-3, gmm_lr=1e-3)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))
    n_batches_per_epoch = len(train_loader)
    # Set up fixed KL weight schedule (determines KL weight for each epoch)
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
        warmup_epochs=15, max_weight=1.0, cooldown_epochs=1, end_weight=1.0
    )

    # Load pretrained model if available
    if common_cfg.pretrained:
        print(f"Loading pretrained weights from {common_cfg.pretrained}")
        unwrap_dp(model).load_state_dict(torch.load(common_cfg.pretrained, map_location=device))
        if writer:
            writer.flush(); writer.close()
        return unwrap_dp(model), unwrap_dp(model), None

    # Set up loss function
    criterion = VaDELossTFExact(
        n_components=common_cfg.n_components,
        latent_dim=common_cfg.latent_dim,
        reg_cat_clusters=vade_cfg.reg_cat_clusters,
        kl_scheduler=kl_scheduler,
        prior_loss_weight=vade_cfg.prior_loss_weight,
        mc_kl_samples=32,
        kl_clamp_min=0.0,
        lambda_distill=0.0,
        distill_sharpen_T=teacher_cfg.distill_sharpen_T,
        distill_conf_weight=teacher_cfg.distill_conf_weight,
        distill_conf_thresh=teacher_cfg.distill_conf_thresh,
        tf_cluster_weight=vade_cfg.tf_cluster_weight,
        reg_cluster_var_weight=0.001,
        distill_use_class_reweight=True,
        distill_class_reweight_beta=teacher_cfg.distill_class_reweight_beta,
        distill_class_reweight_cap=teacher_cfg.distill_class_reweight_cap,
        kmeans_loss_weight=common_cfg.kmeans_loss,
        nonempty_weight=vade_cfg.nonempty_weight,
    ).to(device)
    if hasattr(criterion, "gmm_logvar_clamp"):
        criterion.gmm_logvar_clamp = (-8.0, 8.0)
    criterion.temporal_cohesion_weight = vade_cfg.temporal_cohesion_weight
    criterion.reg_scatter_weight = vade_cfg.reg_scatter_weight
    criterion.reg_scatter_beta = vade_cfg.reg_scatter_beta
    criterion.repel_weight = vade_cfg.repel_weight
    criterion.repel_length_scale = vade_cfg.repel_length_scale
    criterion.nonempty_floor = max(1e-4, 0.05 / common_cfg.n_components)
    criterion.nonempty_p = 1

    # Pretraining
    print("\n--- Pretraining (reconstruction and setting up the latent space) ---")
    unwrap_dp(model).set_pretrain_mode(True)
    criterion.pretrain_mode = True
    pre_epochs = min(10, max(1, common_cfg.epochs))
    ctx = SimpleNamespace(kl_scheduler=kl_scheduler, criterion=criterion, scheduler=None, scheduler_per_batch=True)

    leave = False 
    for ep in range(pre_epochs):
        
        # Leave loading bar in last step (for optics)
        if ep == len(range(pre_epochs))-1:
            leave=True
        apply_decoder_schedule(model, ep)
        pre_logs, _, _ = train_one_epoch_indexed(
            model=model, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=ep, num_epochs=pre_epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=leave
        )
        if writer:
            writer.add_scalar("Pretrain/total_loss", pre_logs["total_loss"], ep)

    # Finish pretraining, reset KL schedule
    unwrap_dp(model).set_pretrain_mode(False)
    criterion.pretrain_mode = False
    criterion.kl_scheduler.current_iteration = 0

    # New user determiend KL schedule
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
        warmup_epochs=common_cfg.kl_warmup, max_weight=common_cfg.kl_max_weight,
        cooldown_epochs=common_cfg.kl_cooldown, end_weight=common_cfg.kl_end_weight
    )
    lambda_scheduler = None
    
    optimizer = build_optimizer(model=model, base_lr=1e-3, gmm_lr=1e-3)

    # VaDE unified checkpoint paths
    _, best_path_val, best_path_score, teacher_init_path = _ckpt_paths("vade", common_cfg=common_cfg)

    tau_star = None
    teacher_init_model = None  # returned as 3rd output
    # cached views for refresh
    pca_pos = pca_spd = pca_edges = pca_angles_train = supervised_labels = None

    if teacher_cfg.use_turtle_teacher:
        print("\n--- Extracting latents for teacher ---")
        z_all = extract_latents(
            unwrap_dp(model), train_loader.dataset, device=device,
            batch_size=2048, num_workers=common_cfg.num_workers
        ).cpu()

        # Set up lambda schedule (distillation weight)
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0, at_max_epochs=teacher_cfg.lambda_decay_start, max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown, end_weight=teacher_cfg.lambda_end_weight
        )

        # Build teacher for VADE
        teacher_cfg.include_latent_view=True # Vade has a free and useful latent view due to pretraining
        teacher, tau_star, teacher_views = maybe_build_turtle_teacher(
            teacher_cfg=teacher_cfg,    
            common_cfg=common_cfg,        
            train_dataset=train_loader.dataset,
            preprocessed_train=preprocessed_train,
            data_path=data_path,
            device=device,
            latent_view=z_all.to(device),
        )

        # Get views for teacher
        pca_pos = teacher_views.get("pca_pos", None)
        pca_spd = teacher_views.get("pca_spd", None)
        pca_edges = teacher_views.get("pca_edges", None)
        pca_angles_train = teacher_views.get("pca_angles", None)
        supervised_labels = teacher_views.get("supervised_labels", None)

        print("\n--- Initializing GMM from teacher τ* ---")
        initialize_gmm_from_teacher(model, z_all, tau_star, min_var=0.01)
        criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill)

        # Save teacher-init checkpoint + info (VaDE only)
        teacher_init_model = deepcopy(unwrap_dp(model))
        if common_cfg.save_weights:
            torch.save(teacher_init_model.state_dict(), teacher_init_path)
            save_model_info(
                teacher_init_path,
                stage="teacher_init",
                epoch=pre_epochs - 1,
                train_steps=pre_epochs * len(train_loader),
                extra={"note": "after pretrain + teacher + GMM init, before main training"},
                common_cfg=common_cfg,
                teacher_cfg=teacher_cfg,
                vade_cfg=vade_cfg,
            )
            print(f"  Saved teacher-init model -> {teacher_init_path}")

    else:
        # If there is no teacher, init GMM directly with train_loader
        print("\n--- Initializing GMM from embeddings (sklearn) ---")
        unwrap_dp(model).initialize_gmm_from_data(train_loader)

    # Inits for training
    best_align = -float("inf")
    best_val = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 50
    early_stop_min_delta = 0.0
    early_stop_warmup = max(5, vade_cfg.freeze_gmm_epochs)

    # Epoch dependent weights
    lambda_d = 0.0
    klw = 0.0

    print(f"\n--- Training for {common_cfg.epochs} epochs ---")
    print("NOTE: some losses are intentionally defined negative and function as rewards.")
    for epoch in range(common_cfg.epochs):
               
        apply_decoder_schedule(model, epoch)
        base = unwrap_dp(model)

        # Freezing/unfreezing of model parts based on user inputs
        if epoch == 0 and vade_cfg.freeze_gmm_epochs > 0:
            print(f"Freezing GMM for {vade_cfg.freeze_gmm_epochs} epoch(s)")
            base.latent_space.gmm_means.requires_grad = False
            base.latent_space.gmm_log_vars.requires_grad = False
        if epoch == vade_cfg.freeze_gmm_epochs:
            new_base_lr = 5e-4
            new_gmm_lr = 2e-4
            for i, group in enumerate(optimizer.param_groups):
                if i == 0: group["lr"] = new_base_lr
                elif i == 1: group["lr"] = new_gmm_lr
            print("Unfreezing GMM")
            base.latent_space.gmm_means.requires_grad = True
            base.latent_space.gmm_log_vars.requires_grad = True
            #base.latent_space.freeze_lens(False)
        if epoch == 0 and vade_cfg.freeze_decoder_epochs > 0:
            print(f"Freezing decoder for {vade_cfg.freeze_decoder_epochs} epoch(s)")
            for p in base.decoder.parameters():
                p.requires_grad = False
        if epoch == vade_cfg.freeze_decoder_epochs:
            print("Unfreezing decoder")
            for p in base.decoder.parameters():
                p.requires_grad = True

        # Refresh Turtle teacher (i.e. retrain on current model and potentially also reinit GMM)
        if (
            epoch > 0 and teacher_cfg.use_turtle_teacher and (teacher_cfg.teacher_refresh_every is not None) and (teacher_cfg.teacher_refresh_every > 0)
            and (epoch) % teacher_cfg.teacher_refresh_every == 0 and (teacher_cfg.teacher_freeze_at is None or (epoch) <= teacher_cfg.teacher_freeze_at)
        ):
            print(f"\n--- Refresh TURTLE teacher at epoch {epoch+1} ---")
            z_curr = extract_latents(base, train_loader.dataset, device=device, batch_size=2048, num_workers=common_cfg.num_workers).cpu()

            # Get current views
            views_dict = {"z": z_curr.to(device)}
            if teacher_cfg.include_nodes_view and (pca_pos is not None) and (pca_spd is not None):
                views_dict["pca_pos"] = pca_pos.to(device)
                views_dict["pca_spd"] = pca_spd.to(device)
            if teacher_cfg.include_edges_view and (pca_edges is not None):
                views_dict["pca_edges"] = pca_edges.to(device)
            if teacher_cfg.include_angles_view and (pca_angles_train is not None):
                views_dict["pca_angles"] = pca_angles_train.to(device)
            if teacher_cfg.include_supervised_view and (supervised_labels is not None):
                views_dict["supervised_labels"] = supervised_labels.to(device)

            # Rerun teacher
            teacher, tau_star = run_turtle_teacher_on_views(
                views_dict=views_dict, n_components=common_cfg.n_components, gamma=teacher_cfg.teacher_gamma,
                alpha_sample_entropy=teacher_cfg.teacher_alpha_sample_entropy,
                outer_steps=max(200, teacher_cfg.teacher_outer_steps), inner_steps=teacher_cfg.teacher_inner_steps,
                normalize_feats=teacher_cfg.teacher_normalize_feats, verbose=True, device=device,
                head_temp=teacher_cfg.teacher_head_temp, task_temp=teacher_cfg.teacher_task_temp,
            )
            tau_star = tau_star.detach()
            criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill)

            # Optionally reinit GMM
            if teacher_cfg.reinit_gmm_on_refresh:
                initialize_gmm_from_teacher(model, z_curr, tau_star, min_var=1e-4)
                print("  Reinitialized GMM from refreshed τ*.")

        # Actual training of the main model
        ctx = SimpleNamespace(kl_scheduler=kl_scheduler, lambda_scheduler=lambda_scheduler, criterion=criterion, scheduler=None, scheduler_per_batch=True)
        # klw is updated in every training step, getting the weight for printing at the middle of the epochs
        train_logs, klw, lambda_d = train_one_epoch_indexed(
            model=model, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_fn, device=device,
            epoch=epoch, num_epochs=common_cfg.epochs, ctx=SimpleNamespace(criterion=criterion, apply_distill=False),
            show_progress=True,
        )

        # A ton of diagnostics for printing training progress
        diag = _compute_diagnostics(model, val_loader, criterion, device, max_batches=common_cfg.diag_max_batches)

        logK = math.log(common_cfg.n_components)
        qH = float(diag.get("diag/q_mean_entropy", float("nan")))
        mKL = float(diag.get("diag/kl_marg_q_to_tau", float("nan")))
        train_logs["conf_norm"] = max(0.0, min(1.0, 1.0 - qH / max(1e-9, logK)))
        train_logs["bal_norm"]  = max(0.0, min(1.0, 1.0 - mKL / max(1e-9, logK)))
        if not math.isfinite(mKL):
            mH = float(diag.get("diag/q_marginal_entropy", float("nan")))
            train_logs["bal_norm"] = max(0.0, min(1.0, mH / max(1e-9, logK)))
        train_logs["alignment_score"] = train_logs["conf_norm"] * train_logs["bal_norm"]
        val_total = float(val_logs.get("total_loss", float("inf")))

        _print_losses(model_name="vade", epoch=epoch, n_epochs=common_cfg.epochs, klw=klw, lambda_d=lambda_d, train_logs=train_logs, val_logs=val_logs)

        # Write training progress
        if writer:
            for k, v in train_logs.items(): writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_logs.items(): writer.add_scalar(f"Val/{k}", v, epoch)
            writer.add_scalar("Align/score", train_logs["alignment_score"], epoch)

        # Deterimine if validation loss and / or balance + certainty score has improved
        improved_score = (train_logs["alignment_score"] > best_align + 1e-6) or (
            abs(train_logs["alignment_score"] - best_align) <= 1e-6 and val_total < (best_val - early_stop_min_delta)
        )
        improved_val = (epoch > 5) and (val_total < best_val)

        # Save best model based on total validation loss
        if improved_val:
            best_val = val_total
            epochs_no_improve = 0
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_val)
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                )
                print(f"Saved best VAL model -> {best_path_val} (val={best_val:.4f})")
        #else:
        #    epochs_no_improve += 1

        # Save best model based on model balance and certainty score
        if improved_score:
            best_align = train_logs["alignment_score"]
            epochs_no_improve = 0
            if common_cfg.save_weights:
                torch.save(unwrap_dp(model).state_dict(), best_path_score)
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    score_value=train_logs["alignment_score"],
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                )
                print(f"Saved best SCORE model -> {best_path_score} (align={best_align:.4f})")
        


        #if epoch >= early_stop_warmup and epochs_no_improve >= early_stop_patience:
        #    print(f"[EarlyStopping] No val improvement for {early_stop_patience} epoch(s). Stop.")
        #    break


    # Load states of best val and score models
    model_score = deepcopy(model)
    if common_cfg.save_weights and os.path.exists(best_path_val):
        unwrap_dp(model).load_state_dict(torch.load(best_path_val, map_location=device))
    if common_cfg.save_weights and os.path.exists(best_path_score):
        unwrap_dp(model_score).load_state_dict(torch.load(best_path_score, map_location=device))

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model), unwrap_dp(model_score), teacher_init_model    
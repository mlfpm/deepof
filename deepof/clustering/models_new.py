"""deep autoencoder models for unsupervised pose detection.

- VQ-VAE: a variational autoencoder with a vector quantization latent-space (https://arxiv.org/abs/1711.00937).
- VaDE: a variational autoencoder with a Gaussian mixture latent-space.
- Contrastive: an embedding model consisting of a single encoder, trained using a contrastive loss.

Models were translated from original tensorflow implementations to Pytorch using LLMs.

"""
# @author lucasmiranda42 and NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Iterable, Tuple, Dict, Optional, Mapping, List, Callable
from types import SimpleNamespace
from dataclasses import dataclass
from functools import partial

import os
import numpy as np
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import TransformedDistribution, Normal


import deepof.clustering.model_utils_new
from deepof.clustering.model_utils_new import ProbabilisticDecoderPT, select_contrastive_loss_pt, save_model_info, CommonFitCfg, TurtleTeacherCfg, VaDECfg, ContrastiveCfg 
from deepof.clustering.censNetConv_pt import CensNetConvPT
import deepof.utils
from deepof.data_loading import get_dt
import warnings
from deepof.clustering.dataset import BatchDictDataset
from sklearn.decomposition import IncrementalPCA
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
from torch.amp import GradScaler


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

        else:
            # Flatten nodes/features into a single feature dim and keep a single group
            x_flat = x.view(B, T, N_nodes * F_per_node)  # (B, T, N*F)
            x_grouped = x_flat.unsqueeze(1)              # (B, 1, T, N*F)
            encoder = self.recurrent_block(x_grouped).squeeze(1)  # (B, 2*latent)

        # Final projection
        return self.final_dense(encoder)
        

class RecurrentDecoderPT(nn.Module):
    """
    A full PyTorch implementation of the recurrent decoder.
    """
    def __init__(self, output_shape: tuple, latent_dim: int, bidirectional_merge: str = "concat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

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
    

#def _act(name: str) -> nn.Module:
#    name = (name or "relu").lower()
#    if name == "relu":
#        return nn.ReLU()
#    if name == "gelu":
#        return nn.GELU()
#    if name == "tanh":
#        return nn.Tanh()
#    if name == "leaky_relu":
#        return nn.LeakyReLU(0.2)
#    if name in {"linear", "identity", "none"}:
#        return nn.Identity()
#    raise ValueError(f"Unsupported activation: {name}")


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
    Builds a neural network that can be used to encode motion tracking instances into a
    vector. Each layer contains a residual block with a convolutional layer and a skip connection. See the following
    paper for more details: https://arxiv.org/pdf/1803.01271.pdf
      - Inputs:
          x: (B, W, N, NF)   node features
          a: (B, W, E, EF)   edge features
      - use_gnn=True:
          TimeDistributed(TCN) over nodes/edges -> (B, N, C) and (B, E, C)
          CensNetConvPT([node, (lap, edge_lap, inc), edge]) -> (B, N, latent), (B, E, latent)
          Flatten and MLP head
      - use_gnn=False:
          Flatten nodes+features -> TCN -> MLP head

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
            x_3d = x.reshape(B, W, N * F_node)          # (B, W, N*F)
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


class TCNDecoderPT(nn.Module):
    """
    Builds a neural network that can be used to decode a latent space into a sequence of
    motion tracking instances. Each layer contains a residual block with a convolutional layer and a skip connection. See
    the following paper for more details: https://arxiv.org/pdf/1803.01271.pdf
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
    # Sanitize only if needed
    if _has_nonfinite(x_bct):
        with torch.no_grad():
            print(f"[SANITIZE] Non-finite detected at {name_prefix}.input_bct -> applying nan_to_num")
        x_bct = torch.nan_to_num(x_bct, nan=0.0, posinf=1e4, neginf=-1e4)

    # Compute in float32 (AMP off) to avoid fp16 overflows
    with torch.amp.autocast(device_type=x_bct.device.type, enabled=False):
        y = conv1x1(x_bct.float())

    y = y.to(out_dtype)
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
    """Compute positional encodings, as in https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf."""
    pe = torch.zeros(max_len, d_model, dtype=dtype, device=device)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    n_odd = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term)[:, :n_odd]
    return pe.unsqueeze(0)  # (1, max_len, d_model)


#class BatchNorm1dKerasFP32(nn.BatchNorm1d):
#    """Keras-like BatchNorm with eps=1e-3 and momentum=0.01."""
#    def __init__(self, num_features, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True):
#        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
#    
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        y = super().forward(x.float())
#        return y.to(dtype=x.dtype)


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
    """Transformer encoder layer with post-normalization. Based on https://www.tensorflow.org/text/tutorials/transformer."""
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
    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377
    and https://arxiv.org/abs/1711.03905.
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
            x_flat = x.reshape(B, W, N * NF)
            x_transposed = x_flat.permute(2, 1, 0)           # (N*NF, W, B)
            x_reshaped = x_transposed.reshape(NF, W, N, B)   # (NF, W, N, B)
            x_nodes = x_reshaped.permute(3, 2, 1, 0)         # (B, N, W, NF)
            
            node_in = x_nodes.reshape(B * N, W, NF)          # (B*N, W, NF)
            node_out = self.node_tf(node_in)                 # (B*N, key_dim)
            nodes_encoded = node_out.view(B, N, self.key_dim)  # (B, N, key_dim)

            # === Process Edges ===
            # Reshape to process each edge's time series independently
            # Same pattern as TCN encoder
            a_flat = a.reshape(B, W, E * EF)
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
            x_flat = x.reshape(B, W, N * NF)  # (B, W, N*NF)
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
    

'''def create_look_ahead_mask_pt(size: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    PyTorch replica of TF create_look_ahead_mask (KEEP mask).
    Returns lower-triangular True (keep), False above diagonal.
    Shape: (T, T) boolean.
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))'''


'''def create_masks_pt(inp_3d: torch.Tensor):
    """
    PyTorch replica of TF create_masks for the decoder (KEEP semantics).
    """
    device = inp_3d.device
    B, T, _ = inp_3d.shape

    tar = inp_3d[:, :, 0]  # (B, T)
    dec_padding_keep = (tar != 0).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
    la_keep = create_look_ahead_mask_pt(T, device=device, dtype=torch.bool)  # (T, T)
    combined_keep = (dec_padding_keep | la_keep.unsqueeze(0).unsqueeze(0))  # (B, 1, T, T)
    return combined_keep, dec_padding_keep'''


'''class MultiHeadAttentionGeneralPT(nn.Module):
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
        
        return out'''


'''class TransformerDecoderLayerPT(nn.Module):
    """Transformer decoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer."""
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
        # Self-attention
        attn1, w1 = self.mha1(query=x, key=x, value=x, attn_mask=look_ahead_mask_3d, return_attention_scores=True)
        x = self.norm1(x + self.dropout1(attn1))

        # Cross-attention
        attn2, w2 = self.mha2(query=x, key=memory, value=memory, attn_mask=padding_mask_2d, return_attention_scores=True)
        x = self.norm2(x + self.dropout2(attn2))

        # FFN
        ffn_out = self.ffn2(self.act(self.ffn1(x)))
        x = self.norm3(x + self.dropout3(ffn_out))
        return x, w1, w2'''


'''class DecoderCorePT(nn.Module):
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

        # FIX: safe 1x1 conv in fp32 with optional sanitization
        x_bct = x.transpose(1, 2)  # (B, C_in, T)
        y_bct = _safe_pointwise_conv1d(self.embed, x_bct, out_dtype=x.dtype, name_prefix="Decoder.Core.embed")
        y = y_bct.transpose(1, 2)

        y = torch.relu(y)
        y = y * (self.model_dim ** 0.5)

        if T > self.pos_encoding.size(1):
            self.pos_encoding = sinusoidal_positional_encoding(T, self.model_dim, device=x.device).to(self.pos_encoding.dtype)
        y = y + self.pos_encoding[:, :T, :].to(y.dtype)
        y = self.dropout(y)

        attention_weights = {}
        out = y
        for i, layer in enumerate(self.layers, start=1):
            out, w1, w2 = layer(out, memory, look_ahead_mask_3d, padding_mask_2d, training=training)
            attention_weights[f"decoder_layer{i}_block1"] = w1
            attention_weights[f"decoder_layer{i}_block2"] = w2

        return out, attention_weights'''


class TFMDecoderPT(nn.Module):
    """
    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

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
    

class VectorQuantizerPT(nn.Module):
    """Quantizes the input vectors into a fixed number of clusters using L2 norm. Based on
    https://arxiv.org/pdf/1509.03700.pdf, and adapted for clustering using https://arxiv.org/abs/1806.02199.
    Implementation based on https://keras.io/examples/generative/vq_vae/."""

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
        x_for_decoder = x.reshape(B, T, N * F)  # Flatten to (B, T, node_features)
        
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
        """encoder_output = self.encoder(x, a)
        quantized, _ = self.vq_layer(encoder_output, return_losses=False)
        return quantized"""

    @torch.no_grad()
    def soft_group(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Inference-only: Get soft cluster assignments. Equivalent to TF 'soft_grouper' model."""
        """encoder_output = self.encoder(x, a)
        _, soft_counts = self.vq_layer(encoder_output, return_losses=False)
        return soft_counts"""

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Full reconstruction from input through VQ-VAE."""
        """encoding_recon_dist, _ = self.forward(x, a, return_losses=False)
        return encoding_recon_dist.mean"""
    
    def get_codebook_usage(self, data_loader, max_samples: int = 10000):
        """Compute codebook usage statistics over a dataset."""
        """self.eval()
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
        
        return usage_counts, perplexity.item()"""
    

class GaussianMixtureLatentPT(nn.Module):
    """
    PyTorch implementation of the Gaussian Mixture probabilistic latent space model.
    It embeds data into a latent space and models that space as a mixture of Gaussians.
    Implementation based on VaDE (https://arxiv.org/abs/1611.05148)
    and VaDE-SC (https://openreview.net/forum?id=RQ428ZptQfU)
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
        #if self.lens_enabled: 
        #    W = self.lens.weight  # (d_lens, d_latent) 
        #    h_mean = F.linear(z_mean, W, bias=None)  # z_mean @ W.T 
        #    var_z = torch.exp(z_log_var)  # (B, d_latent) 
        #    var_h = torch.clamp(var_z @ (W.pow(2)).t(), min=1e-8)  # (B, d_lens) 
        #    h_log_var = torch.log(var_h)  # (B, d_lens)  
        #else:  
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
        """if W.shape != self.lens.weight.shape:  
            raise ValueError(f"Lens weight shape {W.shape} does not match {self.lens.weight.shape}")  
        self.lens.weight.data.copy_(W.to(self.lens.weight.device, dtype=self.lens.weight.dtype)) 
        #self.freeze_lens(True)"""

    def freeze_lens(self, freeze: bool = True) -> None:  
        """Freeze/unfreeze the lens parameters."""  
        """if freeze:
            print("Freezing lense")
        else:
            print("Unfreezing lense")
        for p in self.lens.parameters():       
            p.requires_grad = not freeze""" 


def vade_loss_function(reconstruction_dist, original_data, model_internal_losses, categorical_probs, reg_cat_clusters_weight):
    # Reconstruction Loss (Negative Log-Likelihood)
    """recon_loss = -reconstruction_dist.log_prob(original_data).mean()

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
    return total_loss, recon_loss, internal_loss, cat_reg_loss"""


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
        else: # pragma: no cover
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

        #if torch.isnan(z_mean_head).any() or torch.isnan(z_var_param).any():
        #    print("z issues!")

        B, T, _, _ = x.shape
        x_for_decoder = x.reshape(B, T, self.input_n_nodes * self.input_n_features_per_node)
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
    def get_gmm_params(self) -> dict: # pragma: no cover
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
        Inference-only method to get the latent embedding.

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
        Inference-only method to get cluster probabilities.

        Args:
            x (torch.Tensor): Input node features tensor.
            a (torch.Tensor): Input edge features tensor.

        Returns:
            torch.Tensor: The soft cluster assignments (categorical probabilities).
        """
        encoder_output = self.encoder(x, a)
        _, categorical, _, _, _, _, _ = self.latent_space(encoder_output)
        return categorical


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
        
        if T != Te: # pragma: no cover
            raise ValueError(f"Node and edge time dims must match: T={T}, Te={Te}")
        #if T < 2 or (T % 2) != 0:
        #    raise ValueError(
        #        f"ContrastivePT requires an even sequence length T>=2. Got T={T}. "
        #        "Please pre-trim or pad your sequences (e.g., use T=24 if original T=25)."
        #    )

        self.full_time_steps = T
        self.window_size = T // 2 # To enable length shift augmentation
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
        else: # pragma: no cover
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Debug cache
        self._last_debug: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Encode a half-window:
          x: (B, T_half, N, F), a: (B, T_half, E, Fe) -> (B, D)
        """
        return self.encoder(x, a)

    #@staticmethod
    #def _ts_samples(x: torch.Tensor, win: int):
    #    # TF parity: pos = x[:, 1:win+1], neg = x[:, -win:]
    #    pos = x[:, 1 : win + 1]
    #    neg = x[:, -win:]
    #    return pos, neg

    '''def compute_loss(
        self,
        x: torch.Tensor,  # (B, T, N, F)
        a: torch.Tensor,  # (B, T, E, Fe)
        x_augmented: torch.Tensor,  # (B, T, N, F)
        a_augmented: torch.Tensor,  # (B, T, E, Fe)
        return_debug: bool = False,
    ):
        B, T, N, F_in = x.shape
        if T != self.full_time_steps:
            raise ValueError(f"Input time dim T={T} does not match model T={self.full_time_steps}")

        # Slice windows exactly like TF
        #x, x_augmented = self._ts_samples(x, self.window_size) just commented out for now
        #a, a_augmented = self._ts_samples(a, self.window_size)

        # Encode and normalize
        z = self.encoder(x, a)  # (B, D)
        z_augmented = self.encoder(x_augmented, a_augmented)  # (B, D)
        z = deepof.clustering.model_utils_new.l2_normalize(z, dim=1, eps=1e-12)
        z_augmented = deepof.clustering.model_utils_new.l2_normalize(z_augmented, dim=1, eps=1e-12)

        # Compute loss
        loss, pos_mean, neg_mean = deepof.clustering.model_utils_new.select_contrastive_loss_pt(
            z,
            z_augmented,
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
                sim = sim_fn(z, z_augmented)  # (B, B)
                diag = torch.diag(sim)
                offdiag = sim[~torch.eye(B, dtype=torch.bool, device=sim.device)]
                offdiag = offdiag.view(B, B - 1) if B > 1 else offdiag.view(B, 0)

                debug = {
                    "z_pos_shape": torch.tensor(z.shape),
                    "z_neg_shape": torch.tensor(z_augmented.shape),
                    "z_pos_norm_mean": torch.norm(z, dim=1).mean().cpu(),
                    "z_neg_norm_mean": torch.norm(z_augmented, dim=1).mean().cpu(),
                    "sim_diag_mean": diag.mean().cpu(),
                    "sim_offdiag_mean": offdiag.mean().cpu() if offdiag.numel() > 0 else torch.tensor(0.0),
                    "loss": loss.detach().cpu(),
                    "pos_mean": pos_mean.detach().cpu(),
                    "neg_mean": neg_mean.detach().cpu(),
                }
            self._last_debug = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in debug.items()}

        return loss, pos_mean, neg_mean, debug

    def get_last_debug(self) -> Dict[str, Any]:
        return self._last_debug'''
 

#########################################################
# Intermediary function stash for presentation
#########################################################


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

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

@torch.no_grad()
def _get_q_vade(
    model: nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments q(c|z) from VaDE.

    Assumes model(x, a, return_gmm_params=True) returns a tuple whose third
    element is the cluster responsibility matrix of shape [B, K].
    """
    outputs = model(x, a, return_gmm_params=True)
    q = outputs[2]

    if not torch.is_tensor(q) or q.ndim != 2: # pragma: no cover
        raise RuntimeError(
            f"Expected VaDE responsibilities at outputs[2] with shape [B, K], got {type(q)}"
        )

    q = q.float().clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def _get_q_vqvae(
    model: nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
    *,
    distill_head: nn.Module,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments for VQVAE by applying the distillation head
    to the pre-quantization encoder output.
    """
    _, _, _, _, encoder_output, _ = model(
        x, a, return_losses=True, return_all_outputs=True
    )

    logits = distill_head(encoder_output.float())
    q = F.softmax(logits, dim=-1).clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def _get_q_contrastive(
    model: nn.Module,
    x_full: torch.Tensor,
    a_full: torch.Tensor,
    *,
    distill_head: nn.Module,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments for the contrastive model by reproducing
    the main-window embedding path used for distillation during training.
    """
    del a_full  # Not used; edges are recomputed from x_full exactly as in training.

    a_full = _recompute_edges(x_full, edge_index)

    half_len = x_full.shape[1] // 2
    starts = torch.full(
        (x_full.shape[0],),
        fill_value=half_len // 2,
        device=x_full.device,
        dtype=torch.long,
    )

    x = deepof.clustering.model_utils_new._slice_time_per_sample(x_full, starts, half_len)
    a = deepof.clustering.model_utils_new._slice_time_per_sample(a_full, starts, half_len)

    z = model(x, a)
    z = F.normalize(z, dim=1)

    logits = distill_head(z.float())
    q = F.softmax(logits, dim=-1).clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def _compute_vade_specific_diagnostics(model: nn.Module) -> Dict[str, float]:
    """
    Computes VaDE-specific diagnostics related to the latent GMM.

    Returns an empty dict for models without a latent_space GMM.
    """
    base = unwrap_dp(model)
    latent_space = getattr(base, "latent_space", None)
    if latent_space is None: # pragma: no cover
        return {}

    out: Dict[str, float] = {}

    if hasattr(latent_space, "gmm_log_vars"):
        gmm_log_vars = latent_space.gmm_log_vars.detach()
        out["diag/gmm_logvar_min"] = float(gmm_log_vars.min().item())
        out["diag/gmm_logvar_max"] = float(gmm_log_vars.max().item())

    if hasattr(latent_space, "prior"):
        prior = latent_space.prior
        prior = prior.detach() if torch.is_tensor(prior) else torch.as_tensor(prior)
        prior = prior.clamp_min(1e-9)
        out["diag/prior_entropy"] = float(-(prior * prior.log()).sum().item())

    return out


@torch.no_grad()
def _compute_diagnostics(
    model: nn.Module,
    dataloader: DataLoader,
    q_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    n_components: int,
    tau_star: Optional[torch.Tensor] = None,
    distill_sharpen_T: float = 0.5,
    distill_conf_weight: bool = False,
    distill_conf_thresh: float = 0.55,
    max_batches: int = 4,
    extra_stats_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Computes clustering diagnostics and alignment score from model soft assignments.

    The function is model-agnostic: it only requires a q_fn that extracts soft
    assignments q of shape [B, K] from a batch.

    Args:
    model (nn.Module): Model to evaluate.
    dataloader (DataLoader): Validation or analysis loader.
    q_fn (Callable): Function that maps (model, x, a) -> q of shape [B, K].
    device (torch.device): Device for inference.
    n_components (int): Number of clusters/components K.
    tau_star (Optional[torch.Tensor]): Teacher assignments of shape [N, K]. If provided,
        the balance term is based on KL(q_marginal || tau_marginal). If None, the balance
        term falls back to normalized marginal entropy of q.
    distill_sharpen_T (float): Temperature used when computing teacher-confidence diagnostics.
    distill_conf_weight (bool): Whether teacher confidence weighting is enabled.
    distill_conf_thresh (float): Threshold used for teacher confidence weighting diagnostics.
    max_batches (int): Maximum number of dataloader batches to inspect.
    extra_stats_fn (Optional[Callable]): Optional model-specific diagnostics helper.

    Returns:
    Dict[str, float]: Diagnostics dictionary including conf_norm, bal_norm, and alignment_score.
    """
    if n_components < 2: # pragma: no cover
        raise ValueError(f"n_components must be >= 2, got {n_components}")

    was_training = model.training
    model.eval()

    total_samples = 0
    sum_ent = 0.0
    sum_max_p = 0.0
    sum_q = None

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        batch = move_to(batch, device)
        x, a = batch[0], batch[1]

        q = q_fn(model, x, a)

        if not torch.is_tensor(q) or q.ndim != 2 or q.size(1) != n_components: # pragma: no cover
            raise RuntimeError(
                f"q_fn must return a [B, {n_components}] tensor, got {tuple(q.shape)}"
            )

        q = q.float().clamp_min(1e-8)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        ent = -(q * q.log()).sum(dim=-1)  # [B]
        sum_ent += float(ent.sum().item())
        sum_max_p += float(q.max(dim=-1).values.sum().item())
        total_samples += q.size(0)

        q_sum = q.sum(dim=0)  # [K]
        sum_q = q_sum if sum_q is None else (sum_q + q_sum)

    out: Dict[str, float] = {
        "diag/q_mean_entropy": float("nan"),
        "diag/q_marginal_entropy": float("nan"),
        "diag/q_mean_max_prob": float("nan"),
        "diag/teacher_marginal_entropy": float("nan"),
        "diag/teacher_conf_mean": float("nan"),
        "diag/teacher_weight_mean": float("nan"),
        "diag/kl_marg_q_to_tau": float("nan"),
        "conf_norm": float("nan"),
        "bal_norm": float("nan"),
        "alignment_score": float("nan"),
    }

    if total_samples > 0:
        q_marginal = (sum_q / total_samples).clamp_min(1e-9)
        mean_q_entropy = sum_ent / total_samples
        q_marginal_entropy = float(-(q_marginal * q_marginal.log()).sum().item())
        mean_q_max_prob = sum_max_p / total_samples

        logK = math.log(float(n_components))
        conf_norm = _clip01(1.0 - mean_q_entropy / max(1e-9, logK))

        out["diag/q_mean_entropy"] = mean_q_entropy
        out["diag/q_marginal_entropy"] = q_marginal_entropy
        out["diag/q_mean_max_prob"] = mean_q_max_prob

        if tau_star is not None:
            tau = tau_star.detach().to(device=q_marginal.device, dtype=q_marginal.dtype)
            if tau.ndim != 2 or tau.size(1) != n_components: # pragma: no cover
                raise RuntimeError(
                    f"tau_star must have shape [N, {n_components}], got {tuple(tau.shape)}"
                )

            tau_marg = tau.mean(dim=0).clamp_min(1e-9)
            teacher_marginal_entropy = float(-(tau_marg * tau_marg.log()).sum().item())

            kl_marg_q_to_tau = float(
                (q_marginal * (q_marginal.log() - tau_marg.log())).sum().item()
            )
            kl_marg_q_to_tau = max(0.0, kl_marg_q_to_tau)

            bal_norm = _clip01(1.0 - kl_marg_q_to_tau / max(1e-9, logK))

            T = float(distill_sharpen_T)
            if T > 0.0:
                tau_sharp = torch.softmax(tau.clamp_min(1e-8).log() / T, dim=-1)
            else:
                tau_sharp = tau

            conf = tau_sharp.max(dim=1).values
            teacher_conf_mean = float(conf.mean().item())

            if distill_conf_weight:
                thr = float(distill_conf_thresh)
                w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(min=0.0, max=1.0)
                teacher_weight_mean = float(w.mean().item())
            else:
                teacher_weight_mean = 1.0

            out["diag/teacher_marginal_entropy"] = teacher_marginal_entropy
            out["diag/teacher_conf_mean"] = teacher_conf_mean
            out["diag/teacher_weight_mean"] = teacher_weight_mean
            out["diag/kl_marg_q_to_tau"] = kl_marg_q_to_tau

        else:
            bal_norm = _clip01(q_marginal_entropy / max(1e-9, logK))

        alignment_score = conf_norm * bal_norm

        out["conf_norm"] = conf_norm
        out["bal_norm"] = bal_norm
        out["alignment_score"] = alignment_score

    if extra_stats_fn is not None:
        out.update(extra_stats_fn(model))

    if was_training:
        model.train()

    return out


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
   

class TurtleHeads(nn.Module):
    """Trains each linear head for each view to resemble the target tau (the lienar fit based on all views)
    i.e. tries to get as close as possible to prediciting the separation within the full dataset (at teh current batch) based on each view of the dataset.
    
    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        inner_lr (float): Learning rate used for the per-head inner optimization. Defaults to 0.1.
        M (int): Number of inner optimization steps performed for the heads at each outer step. Defaults to 100.
        weight_decay (float): Weight decay used for the head optimizers. Defaults to 1e-4.
        normalize_feats (bool): If True, L2-normalizes each input feature vector before fitting or inference. Defaults to True.
        temperature (float): Temperature used to scale the logits produced by each head. Defaults to 0.7.
    """
    def __init__(self, feature_dims: List[int], n_components: int,
                 inner_lr: float = 0.1, M: int = 100,
                 weight_decay: float = 1e-4,
                 normalize_feats: bool = True,
                 temperature: float = 0.7,
                 ):
        super().__init__()
        self.M = M
        self.normalize_feats = normalize_feats
        self.temperature = temperature
        self.heads = nn.ModuleList()
        self._optims = []

        for d in feature_dims:
            head = nn.Linear(d, n_components)
            self.heads.append(head)
            self._optims.append(
                torch.optim.SGD(head.parameters(), lr=inner_lr, weight_decay=weight_decay)
            )

    def _maybe_normalize(self, feats_list):
        if not self.normalize_feats:
            return feats_list
        out = []
        for i, f in enumerate(feats_list):
            out.append(F.normalize(f, dim=-1))
        return out

    def inner_fit(self, feats_list, soft_targets):
        """
        Trains each linear head for each view to resemble the target tau (the lienar fit based on all views)

        Args:
            feats_list (List[torch.Tensor]): List of feature tensors, one per view, each of shape [B, D_i].
            soft_targets (torch.Tensor): Soft target assignments of shape [B, K], typically produced by the task encoder.

        Returns:
            None
        """
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
        """Computes detached logits for each view-specific head."""
        feats_list = self._maybe_normalize([f.detach().float() for f in feats_list])
        return [(head(feats) / self.temperature).detach() for head, feats in zip(self.heads, feats_list)]

class TaskEncoder(nn.Module):
    """
    Applies linear projections (fully connected layers) to each individual view and sums up the result
    
    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        temperature (float): Temperature used to scale the per-view logits before averaging and softmax. Defaults to 1.0.
    """
    def __init__(self, feature_dims: List[int], n_components: int, 
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.temperature = temperature

        self.projs = nn.ModuleList()
        for d in feature_dims:
            proj = nn.Linear(d, n_components)
            self.projs.append(proj)

    def forward(self, feats_list):
        """
        Computes soft cluster assignments from the available views.

        Args:
            feats_list (List[torch.Tensor]): List of feature tensors, one per view, each of shape [B, D_i], D_i may vary based on view type.

        Returns:
            torch.Tensor: Soft assignments of shape [B, n_clusters], obtained by averaging projected logits across views and applying softmax.
        """
        logits = None
        for proj, feats in zip(self.projs, feats_list):
            # Ensure float for stability
            out = proj(feats.float()) / self.temperature
            logits = out if logits is None else (logits + out)
        
        logits = logits / max(len(self.projs), 1)
        return F.softmax(logits, dim=-1)
    

class TurtleTeacher(nn.Module):
    """ 
    Teacher model that learns soft cluster assignments τ which are easy to predict
    from each individual view using linear heads, while regularizers encourage
    confident assignments and balanced cluster usage.

    Based on "Let Go of Your Labels with Unsupervised Transfer" by Artyom Gadetsky et al., see https://arxiv.org/abs/2406.07236  

    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        gamma (float): Strength of the marginal entropy penalty that encourages balanced cluster usage. Defaults to 10.0.
        alpha_sample_entropy (float): Weight of the per-sample entropy term that encourages confident assignments. Defaults to 0.1.
        inner_lr (float): Learning rate for the per-view heads during inner optimization. Defaults to 0.1.
        inner_steps (int): Number of inner optimization steps used to fit the per-view heads at each outer step. Defaults to 100.
        head_wd (float): Weight decay used for the per-view head optimizers. Defaults to 1e-4.
        head_temp (float): Temperature used for the per-view head logits. Defaults to 0.5.
        task_temp (float): Temperature used for the task encoder logits. Defaults to 0.5.
        normalize_feats (bool): If True, L2-normalizes features before passing them to the per-view heads. Defaults to True.
        lr_theta (float): Learning rate for the task encoder optimizer. Defaults to 5e-3.
        delta_death_barrier (float): Strength of the penalty discouraging dead or weakly used clusters. Defaults to 40.0.
        device (str): Device on which the teacher should operate. Defaults to "cpu".
    """
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
                 device: str = "cpu",                 
                 ):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.delta = delta_death_barrier
        self.alpha = alpha_sample_entropy

        # to predict general labeling using individual views
        self.heads = TurtleHeads(
            feature_dims, n_components,
            inner_lr=inner_lr, M=inner_steps,
            weight_decay=head_wd,
            normalize_feats=normalize_feats,
            temperature=head_temp,
        )

        # To generate general labeling using full dataset
        self.task_encoder = TaskEncoder(
            feature_dims, n_components,
            temperature=task_temp,
        )
        self.opt_theta = torch.optim.Adam(self.task_encoder.parameters(), lr=lr_theta)
        self.device = torch.device(device)

    def to(self, device: torch.device):
        self.device = device
        self.heads.to(device)
        self.task_encoder.to(device)
        return self

    # Helper for full dataset prediction
    @torch.no_grad()
    def predict(self, loader) -> torch.Tensor:
        """
        Runs a sequential pass over the data to compute assignments for the whole dataset.
        Used to generate the final tau_star without loading everything into GPU RAM at once.
        Args:
            loader (DataLoader): DataLoader yielding batches of views.

        Returns:
            torch.Tensor: Tensor of shape [N, K] containing the soft assignments for the full dataset.
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
        """
        Fits the teacher by alternating between per-view head updates and task encoder updates on mini-batches.

        At each outer step, the task encoder produces a batch of soft assignments tau.
        The per-view heads are then fitted to match tau, after which the task encoder is updated so that tau becomes easier 
        to predict from each view while remaining confident and well-distributed across clusters.

        Args:
        loader (DataLoader): DataLoader yielding batches of feature tensors, one tensor per active view.
        outer_steps (int): Number of outer optimization steps for the task encoder. Defaults to 200.
        rho (float): Weight of the optional batch-local smoothness regularizer between neighboring rows in a batch. Defaults to 0.04.
        verbose (bool): If True, prints optimization statistics during fitting. Defaults to True.

        Returns:
        None
        """
        
        def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
            p = p.clamp_min(eps)
            return -(p * p.log()).sum(dim=-1)
        
        # Create an infinite iterator over the loader
        iterator = iter(loader)
        
        # Outer fitting loop 
        for step in range(outer_steps):
            # Fetch next batch
            try:
                feats_list = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                feats_list = next(iterator)
            
            # Move batch to GPU (non_blocking helps if pin_memory=True)
            feats_list = [f.to(self.device, non_blocking=True) for f in feats_list]

            # Generate labeling based on linear layers (will start out as random gibberish)
            tau = self.task_encoder(feats_list)  # [Batch, K]

            # Inner Loop: Update Heads
            # Train linear heads to reflect current labeling based only on their individual views
            self.heads.inner_fit(feats_list, tau)

            # Compute Loss Components
            logits_k = self.heads.logits_list(feats_list)
            # Sum of cross entropy losses all heads have when trying to predicting the current labeling
            # (Minimizing this pushes τ toward labels that each view can predict linearly.)
            ce_term = 0.0           
            for k, logits in enumerate(logits_k):
                ce_term = ce_term + soft_cross_entropy_logits(logits, tau)
            ce_term = ce_term / max(len(logits_k), 1)

            # sample entropy of labelling accross clusters
            # (Less uniform labelling i.e. in one sample containing values for n clusters, one cluster is more prominent than others, 
            # i.e. lower sample entropy, is better)
            sample_entropy = _entropy(tau).mean()
            
            # Estimate Marginal Entropy (Batch Approximation)
            # (Measure for usage of clusters, even usuage of all clusters i.e. higher H(marg) is better)
            marginal = tau.mean(dim=0)
            H_marginal = _entropy(marginal.unsqueeze(0)).mean()
            logK = float(np.log(self.n_components))
            H_target = torch.tensor(1 * logK, device=H_marginal.device) #option for weighting for not compeltely even distribution, currently not applied
            marg_gap = torch.relu(H_target - H_marginal)          
            gamma_t = float(self.gamma) * (1.0 - float(step) / float(max(1, outer_steps)))

            # Dead penalty
            # (punishes dead i.e. empty or nearly empty clusters)
            dead_floor = max(1e-4, 0.1 / self.n_components)
            tau_clamp = tau.clamp_min(1e-8)
            # Using tau**2 makes diffuse assignments count less than confident ones
            gamma_pow = 2.0
            usage = (tau_clamp ** gamma_pow).mean(dim=0)
            dead_pen = torch.relu(dead_floor - usage).sum() / (dead_floor * self.n_components)
            delta_t = self.delta * max(0.5, 0.6 + 0.4 * (1.0 - step / (float(max(1, outer_steps)))))
            
            # Active count metric (for logging)
            K_dim = int(marginal.numel())
            tau_act = 0.02
            active_soft = torch.sigmoid((marginal - dead_floor) / tau_act)
            active_count = active_soft.sum()
            
            loss = (
                ce_term
                + self.alpha * sample_entropy
                + gamma_t * marg_gap
                + delta_t * dead_pen
            )

            # Optional adjacency penalty between neighboring rows in the current batch.
            # If the loader is shuffled, this is not true temporal smoothing;
            # it acts only as a weak local consistency / stability regularizer.
            if (step % 2) != 0 and rho > 0.0:
                diff = tau[1:] - tau[:-1]
                smooth = (diff.abs().sum(dim=-1)).mean()
                loss = loss + rho * smooth

            # Optimization
            self.opt_theta.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_theta.step()

            if verbose and (step % 20 == 0 or step == outer_steps - 1):
                with torch.no_grad():
                    mean_max_p = tau.max(dim=1).values.mean().item()
                    print(f"[Teacher] step {step:03d} | loss {loss.item():.4f} | CE {ce_term.item():.4f} | "
                          f"E[H(τ)] {sample_entropy.item():.4f} | H(marg) {H_marginal.item():.4f} | "
                          f"mean max_p {mean_max_p:.3f} | dead_pen {dead_pen.item():.3f} | "
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
    """
        Extracts latent mean vectors using the model encoder.

        Args:
            model (nn.Module): Model providing an encoder and latent-space encoder head.
            dataset (BatchDictDataset): Dataset from which latent representations should be extracted.
            device (torch.device): Device on which the model inference should be performed.
            batch_size (int): Batch size used for latent extraction. Defaults to 512.
            num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

        Returns:
            torch.Tensor: Tensor of shape [N_samples, latent_dim] containing the extracted latent mean vectors on CPU.
    """
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

    Writes directly into model.latent_space.{gmm_means, gmm_log_vars, prior}.

    Args:
        model (nn.Module): Model whose latent-space GMM parameters should be initialized.
        z_all (torch.Tensor): Latent representations of shape [N_samples, D], typically extracted from the training set.
        tau_star (torch.Tensor): Teacher soft assignments of shape [N_samples, C], where C is the number of mixture components.
        min_var (float): Minimum variance value used to clamp estimated cluster variances for numerical stability. Defaults to 1e-4.
        min_mass (float): Small constant added to cluster masses to avoid division by zero during estimation. Defaults to 1e-6.

    Returns:
        None
    """
    base = unwrap_dp(model)
    base.eval()

    device = next(base.parameters()).device
    z = z_all.to(device=device, dtype=base.latent_space.gmm_means.dtype)
    tau = tau_star.to(device=device, dtype=z.dtype)
        
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
):
    """
    Fits two IncrementalPCAs:
      - one on positions (x,y) only
      - one on speeds (the 3rd channel per node)

    Assumes dataset returns x with shape [B, T, N, F] where F>=3 and
    channel 0,1 are (x,y), channel 2 is speed.

    Args:
        dataset (BatchDictDataset): Dataset providing node features and adjacency information.
        n_components_pos (int): Number of PCA components to retain for flattened position features. Defaults to 32.
        n_components_spd (int): Number of PCA components to retain for flattened speed features. Defaults to 32.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 4096.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.
        max_samples (Optional[int]): Maximum number of samples to use during PCA fitting. If None, uses all samples. Defaults to None.

    Returns:
        Tuple[IncrementalPCA, torch.Tensor, IncrementalPCA, torch.Tensor]:
        ipca_pos: Fitted IncrementalPCA object for flattened position features.
        feats_pos_all: Tensor of shape [N_samples, n_components_pos] containing PCA-transformed position features.
        ipca_spd: Fitted IncrementalPCA object for flattened speed features.
        feats_spd_all: Tensor of shape [N_samples, n_components_spd] containing PCA-transformed speed features.
    """

    # ---- Pass 1: partial_fit ----
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )

    ipca_pos = IncrementalPCA(n_components=n_components_pos)
    ipca_spd = IncrementalPCA(n_components=n_components_spd)

    seen = 0
    for batch in loader:
        x, a, *rest = batch
        B, T, N, F = x.shape

        if F < 3: # pragma: no cover
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
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
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
) -> Tuple[IncrementalPCA, torch.Tensor]:
    """
    Fits IncrementalPCA on angle tensors and returns both the fitted ipca and features.
    Mirrors fit_nodes_pca but for angle data.

    Args:
        dataset_with_angles (BatchDictDataset): Dataset providing precomputed angle tensors.
        n_components (int): Number of PCA components to retain for the flattened angle features. Defaults to 32.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

    Returns:
        Tuple[IncrementalPCA, torch.Tensor]:
        ipca: Fitted IncrementalPCA object for the angle features.
        feats_all: Tensor of shape [N_samples, n_components] containing PCA-transformed angle features.
    """
    assert getattr(dataset_with_angles, "return_angles", False), \
        "fit_angles_pca expects a dataset created with return_angles=True."

    # Pass 1: partial_fit
    ipca = IncrementalPCA(n_components=n_components)
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
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
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
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
) -> torch.Tensor:
    """
    Builds an IncrementalPCA view from precomputed angles in the dataset.
    Requires dataset_with_angles to be constructed with return_angles=True.
    
    Args:
        dataset_with_angles (BatchDictDataset): Dataset providing precomputed angle tensors.
        n_components (int): Number of PCA components to retain for the flattened angle features. Defaults to 32.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

    Returns:
        torch.Tensor: Tensor of shape [N_samples, n_components] containing PCA-transformed angle features.
    """
    assert getattr(dataset_with_angles, "return_angles", False), \
        "extract_pca_angles_view expects a dataset created with return_angles=True."

    # Pass 1: partial_fit
    ipca = IncrementalPCA(n_components=n_components)
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
    )
    for batch in loader:
        # expected: x, a, ang, vid
        if len(batch) == 4:
            _, _, ang, _, = batch
        else:
            raise RuntimeError("Angles loader must yield (x, a, ang, vid)")
        X = ang.view(ang.size(0), -1).cpu().numpy()  # flatten (T*K*1)
        ipca.partial_fit(X)

    # Pass 2: transform
    feats_all = []
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
    )
    for batch in loader:
        _, _, ang, _ = batch
        X = ang.view(ang.size(0), -1).cpu().numpy()
        Z = ipca.transform(X)
        feats_all.append(torch.from_numpy(Z).float())

    return torch.cat(feats_all, dim=0)  # [N, n_components]

@torch.no_grad()
def extract_pca_edges_view(dataset: BatchDictDataset,
                           n_components: int = 16,
                           batch_size: int = 8192,
                           num_workers: int = 0,
                           max_samples: Optional[int] = None) -> torch.Tensor:
    """
    Returns PCA features [N, n_components] for all samples' edge tensor 'a' (T, E, F_edge),
    in order (shuffle=False), using two passes: partial_fit, then transform.
   
    Args:
        dataset (BatchDictDataset): Dataset providing node and edge features.
        n_components (int): Number of PCA components to retain for the flattened edge features. Defaults to 16.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.
        max_samples (Optional[int]): Maximum number of samples to use during PCA fitting. If None, uses all samples. Defaults to None.

    Returns:
       torch.Tensor: Tensor of shape [N_samples, n_components] containing PCA-transformed edge features.
    """

    # 1) Pass 1: fit IncrementalPCA on flattened edges
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
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
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
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
                                batch_size: int = 2048) -> Tuple[Any, torch.Tensor]:
    """
    Fits a TURTLE teacher on a set of precomputed views and returns the final soft assignments.

    The input views are packed into a TensorDataset, trained in shuffled mini-batches, 
    and then evaluated sequentially to produce tau_star.

    Args:
        views_dict (dict): Dictionary mapping view names to tensors of shape [N, D]. Entries with value None are ignored.
        n_components (int): Number of output components or clusters.
        gamma (float): Strength of the marginal entropy penalty encouraging balanced cluster usage. Defaults to 6.0.
        alpha_sample_entropy (float): Weight of the per-sample entropy term encouraging confident assignments. Defaults to 1.0.
        outer_steps (int): Number of outer optimization steps for the task encoder. Defaults to 200.
        inner_steps (int): Number of inner optimization steps for the per-view heads at each outer step. Defaults to 200.
        normalize_feats (bool): If True, L2-normalizes features before passing them to the per-view heads. Defaults to True.
        verbose (bool): If True, prints training progress during fitting. Defaults to True.
        device (Optional[torch.device]): Device on which the teacher should be trained. Defaults to CPU if None.
        head_temp (float): Temperature used for the per-view head logits. Defaults to 0.3.
        task_temp (float): Temperature used for the task encoder logits. Defaults to 0.3.
        batch_size (int): Batch size used for training and prediction. Defaults to 2048.

    Returns:
        Tuple[Any, torch.Tensor]:
        teacher: Fitted TURTLE teacher object.
        tau_star: Tensor of shape [N, K] containing the final soft assignments in dataset order.
    """
    
    device = device or torch.device("cpu")
    
    # Move data to cpu
    keys = []
    tensors = []
    for k, v in views_dict.items():
        if v is not None:
            keys.append(k)
            tensors.append(v.cpu()) # Ensure CPU
            
    assert len(tensors) > 0, "No active views found."
    
    # Create DataLoader (Shuffling is generally good for the teacher)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True)

    # Initialize Teacher
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
        device=device.type,
    ).to(device)

    # Fit teacher batch wise
    print(f"--- Fitting TurtleTeacher (batch_size={batch_size}) ---")
    teacher.fit(loader, outer_steps=outer_steps, rho=0.04, verbose=verbose)
    
    # Predict tau star ("ideal" labeling) using the trained teacher
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
    """
    Loads a cached PCA angle view from disk if available, otherwise compute-store-returns.
    Used to avoid repeated computation of angle view
    
    Args:
        dataset_with_angles (BatchDictDataset): Dataset providing precomputed angle tensors.
        cache_path (str): File path where the PCA-transformed angle features should be stored or loaded from.
        n_components (int): Number of PCA components to retain for the flattened angle features. Defaults to 32.
        batch_size (int): Batch size used when computing the PCA angle view. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

    Returns:
        torch.Tensor: Tensor of shape [N_samples, n_components] containing PCA-transformed angle features.
    """
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

    Args:
        latent_dim (int): Dimensionality of the latent input vectors.
        n_components (int): Number of output components or clusters.
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
    Builds the requested teacher views, fits the TURTLE teacher if enabled, and returns the resulting teacher together with final soft assignments.

    Depending on the configuration, this function can include a latent view, PCA-based node, edge, and angle views, and a supervised label view.
    All constructed views are returned so they can be reused later.

    Args:
        teacher_cfg (TurtleTeacherCfg): Configuration controlling which views are used and how the teacher is trained.
        common_cfg (CommonFitCfg): Common training configuration, including the number of components.
        train_dataset (BatchDictDataset): Training dataset used to extract or derive teacher views.
        preprocessed_train (dict): Preprocessed training data used when rebuilding angle-based datasets.
        data_path (str): Path to the underlying dataset files.
        device (torch.device): Device on which the teacher should be trained.
        latent_view (Optional[torch.Tensor]): Optional latent representation computed externally, for example by another model. Required if include_latent_view is enabled.

    Returns:
        Tuple[Optional[Any], Optional[torch.Tensor], Dict[str, Any]]:
        teacher: Fitted TURTLE teacher object, or None if the teacher is disabled.
        tau_star: Final soft assignments of shape [N, K], or None if the teacher is disabled.
        views: Dictionary containing the constructed views and any fitted PCA objects for reuse.
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

    # Latent view (only for VaDE): include only if explicitly requested
    if teacher_cfg.include_latent_view:
        if latent_view is None:
            raise ValueError("include_latent_view=True but latent_view=None")
        views["z"]=latent_view.to(device)

    # Nodes view (PCA pos + PCA spd) ---
    if teacher_cfg.include_nodes_view:
        print("\n--- Building PCA views for teacher (nodes) ---")
        _, pca_pos, _, pca_spd = fit_nodes_pca(
            train_dataset,
            n_components_pos=teacher_cfg.pca_nodes_dim,
            n_components_spd=teacher_cfg.pca_nodes_dim,
            batch_size=teacher_cfg.batch_size_nodes,
            num_workers=0,
        )
        views["pca_pos"] = pca_pos
        views["pca_spd"] = pca_spd

    # Edges view (distances between nodes)
    if teacher_cfg.include_edges_view:
        print("\n--- Building PCA views for teacher (edges) ---")
        pca_edges = extract_pca_edges_view(
            train_dataset, n_components=teacher_cfg.pca_edges_dim, batch_size=teacher_cfg.batch_size_edges, num_workers=0
        )
        views["pca_edges"] = pca_edges

    # Angles view (standardized to fit_angles_pca everywhere)
    if teacher_cfg.include_angles_view:
        print("\n--- Building PCA views for teacher (angles) ---")
        angles_train_dataset = BatchDictDataset(
            preprocessed_train, data_path, "train_",
            force_rebuild=False, h5_chunk_len=common_cfg.batch_size, return_angles=True
        )
        _, pca_angles_train = fit_angles_pca(
            angles_train_dataset, n_components=teacher_cfg.pca_angles_dim, batch_size=teacher_cfg.batch_size_angles, num_workers=0 
        )
        views["pca_angles"] = pca_angles_train

    print("\n--- Running TURTLE teacher on views ---")
    teacher, tau_star = run_turtle_teacher_on_views(
        views_dict=views, n_components=common_cfg.n_components, gamma=teacher_cfg.teacher_gamma,
        alpha_sample_entropy=teacher_cfg.teacher_alpha_sample_entropy, outer_steps=teacher_cfg.teacher_outer_steps,
        inner_steps=teacher_cfg.teacher_inner_steps, normalize_feats=teacher_cfg.teacher_normalize_feats,
        verbose=True, device=device, head_temp=teacher_cfg.teacher_head_temp, task_temp=teacher_cfg.teacher_task_temp,
        batch_size = teacher_cfg.teacher_batch_size,
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



class VadeLoss(nn.Module):
    """
    VaDE loss function combining reconstruction, KL divergence, clustering, and optional teacher distillation terms.

    Supports two training phases (pretrain and main) with separate parameter sets for phase-dependent
    regularization terms. Call set_pretrain_mode() to switch between them.

    Phase-dependent parameters (stored separately for pretrain and main):
    - repel_weight, repel_length_scale
    - nonempty_weight, nonempty_floor, nonempty_p

    Phase-independent parameters are set once at construction and do not change between phases.

    Args:
        common_cfg (CommonFitCfg): Common training configuration providing n_components, latent_dim, and kmeans_loss.
        vade_cfg (VaDECfg): VaDE-specific configuration including both pretrain and main parameter values.
        teacher_cfg (TurtleTeacherCfg): Teacher distillation configuration.
        kl_scheduler (Optional[Dynamic_weight_manager]): KL weight scheduler. Can be replaced later via set_kl_scheduler().
        l1_activity_weight (float): L1 regularization weight on z_log_var. Defaults to 0.1.
        gmm_logvar_clamp (Tuple[float, float]): Min and max clamp values for GMM log-variances. Defaults to (-8.0, 8.0).
    """
    def __init__(
            self,
            common_cfg: "CommonFitCfg",
            vade_cfg: "VaDECfg",
            teacher_cfg: "TurtleTeacherCfg",
            kl_scheduler: Optional["Dynamic_weight_manager"] = None,
            l1_activity_weight: float = 0.1,
            gmm_logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
        ):
        super().__init__()
        # Core dimensions
        self.n_components = common_cfg.n_components
        self.latent_dim = common_cfg.latent_dim

        # Phase-independent loss parameters
        self.l1_activity_weight = float(l1_activity_weight)
        self.tf_cluster_weight = float(vade_cfg.tf_cluster_weight)
        self.reg_cat_clusters_weight = float(vade_cfg.reg_cat_clusters)
        self.temporal_cohesion_weight = float(vade_cfg.temporal_cohesion_weight)
        self.reg_scatter_weight = float(vade_cfg.reg_scatter_weight)
        self.reg_scatter_beta = float(vade_cfg.reg_scatter_beta)
        self.gmm_logvar_clamp = gmm_logvar_clamp

        # Distillation parameters (teacher and lambda set later via set_teacher())
        self.lambda_distill = 0.0
        self.lambda_scheduler = None
        self.tau_star = None
        self.teacher_marginal = None
        self.class_weight = None
        self.distill_sharpen_T = float(teacher_cfg.distill_sharpen_T)
        self.distill_conf_weight = bool(teacher_cfg.distill_conf_weight)
        self.distill_conf_thresh = float(teacher_cfg.distill_conf_thresh)
        self.distill_use_class_reweight = True
        self.distill_class_reweight_beta = float(teacher_cfg.distill_class_reweight_beta)
        self.distill_class_reweight_cap = (
            None if teacher_cfg.distill_class_reweight_cap is None
            else float(teacher_cfg.distill_class_reweight_cap)
        )

        # KL scheduling
        self.kl_scheduler = kl_scheduler

        # Mode-dependent parameters: two complete sets, applied via set_pretrain_mode()
        self.mode_params = {
            "pretrain": {
                "kmeans_loss" : float(vade_cfg.kmeans_loss_pretrain),
                "repel_weight": float(vade_cfg.repel_weight_pretrain),
                "repel_length_scale": float(vade_cfg.repel_length_scale_pretrain),
                "nonempty_weight": float(vade_cfg.nonempty_weight_pretrain),
                "nonempty_floor": max(1e-4, vade_cfg.nonempty_floor_percent_pretrain / common_cfg.n_components),
                "nonempty_p": int(vade_cfg.nonempty_p_pretrain),
            },
            "main": {
                "kmeans_loss" : float(common_cfg.kmeans_loss),
                "repel_weight": float(vade_cfg.repel_weight),
                "repel_length_scale": float(vade_cfg.repel_length_scale),
                "nonempty_weight": float(vade_cfg.nonempty_weight),
                "nonempty_floor": max(1e-4, vade_cfg.nonempty_floor_percent / common_cfg.n_components),
                "nonempty_p": int(vade_cfg.nonempty_p),
            },
        }

        # Start in pretrain mode
        self.set_mode("pretrain")

    def set_mode(self, mode: str):
        """Copies the parameter set for the given phase into the active attributes."""
        self.pretrain_mode = (mode == "pretrain")
        params = self.mode_params[mode]
        self.kmeans_loss_weight = params["kmeans_loss"]
        self.repel_weight = params["repel_weight"]
        self.repel_length_scale = params["repel_length_scale"]
        self.nonempty_weight = params["nonempty_weight"]
        self.nonempty_floor = params["nonempty_floor"]
        self.nonempty_p = params["nonempty_p"]


    def set_teacher(self, tau_star: torch.Tensor, lambda_distill: float = 1.0, lambda_scheduler: Optional["Dynamic_weight_manager"] = None,):
        """
        Sets teacher assignments and distillation parameters.

        Computes inverse-marginal class reweighting from the teacher distribution
        if distill_use_class_reweight is enabled.

        Args:
            tau_star (torch.Tensor): Teacher soft assignments of shape [N, K].
            lambda_distill (float): Distillation loss weight. Defaults to 1.0.
            lambda_scheduler (Optional[Dynamic_weight_manager]): Optional scheduler for lambda_distill.
        """
        self.tau_star = tau_star
        self.lambda_distill = float(lambda_distill)
        self.lambda_scheduler = lambda_scheduler

        #Compute inverse-marginal class weights from teacher τ*
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

    def set_kl_scheduler(self, kl_scheduler: Optional["Dynamic_weight_manager"] = None):
        """Replaces the KL weight scheduler and resets its iteration counter."""
        self.kl_scheduler = kl_scheduler 
        self.kl_scheduler.current_iteration = 0

    @staticmethod
    def _log_normal_diag(x, mean, log_var):
        """Log-probability under a diagonal Gaussian."""
        LOG_2PI = math.log(2.0 * math.pi)
        return -0.5 * torch.sum(
            LOG_2PI + log_var + (x - mean) ** 2 * torch.exp(-log_var), dim=-1
        )

    def _log_mog(self, z, gmm_means, gmm_log_vars, prior, eps=1e-8):
        """Log-probability under a mixture of diagonal Gaussians."""
        S, B, D = z.shape
        C = gmm_means.shape[0]
        gmm_log_vars = torch.clamp(
            gmm_log_vars,
            min=self.gmm_logvar_clamp[0],
            max=self.gmm_logvar_clamp[1],
        )
        log_prior = torch.log(torch.clamp(prior, min=eps))

        z_exp = z.unsqueeze(2)                          # (S, B, 1, D)
        means = gmm_means.view(1, 1, C, D)
        log_vars = gmm_log_vars.view(1, 1, C, D)

        log_p_z_given_c = self._log_normal_diag(z_exp, means, log_vars)  # (S, B, C)
        log_mix = log_prior.view(1, 1, C)
        return torch.logsumexp(log_mix + log_p_z_given_c, dim=-1)        # (S, B)

    def _monte_carlo_kl(self, z_mean, z_log_var, gmm_means, gmm_log_vars, prior,
                        n_samples: int = 32):
        """
        Monte Carlo estimate of KL(q(z|x) || p(z)) where p(z) is the GMM prior.

        This is the key loss term that pulls latent representations toward cluster centers.
        """
        z_log_var = torch.clamp(z_log_var, min=-4.0, max=4.0)
        B, D = z_mean.shape
        scale_q = torch.exp(0.5 * z_log_var)

        eps = torch.randn(n_samples, B, D, device=z_mean.device, dtype=z_mean.dtype)
        z_samples = z_mean.unsqueeze(0) + eps * scale_q.unsqueeze(0)  # (S, B, D)

        log_q = self._log_normal_diag(
            z_samples, z_mean.unsqueeze(0), z_log_var.unsqueeze(0)
        )
        log_p = self._log_mog(z_samples, gmm_means, gmm_log_vars, prior)

        kl = (log_q - log_p).mean()
        return torch.clamp(kl, min=0.0)
    
    def _log_p_z_given_c(self, z: torch.Tensor, gmm_means: torch.Tensor, gmm_log_vars: torch.Tensor) -> torch.Tensor:
        """
        Computes log p(z|c) for each component c using diagonal Gaussians.

        Args:
            z (torch.Tensor): Latent samples of shape [B, D].
            gmm_means (torch.Tensor): Component means of shape [C, D].
            gmm_log_vars (torch.Tensor): Component log-variances of shape [C, D].

        Returns:
            torch.Tensor: Log-likelihoods of shape [B, C].
        """
        # clamp for stability
        gmm_log_vars = torch.clamp(gmm_log_vars, min=self.gmm_logvar_clamp[0], max=self.gmm_logvar_clamp[1])
        scale = torch.exp(0.5 * gmm_log_vars).clamp(min=1e-3)
        dist = Normal(gmm_means.unsqueeze(0), scale.unsqueeze(0))  # (1,C,D)
        logp = dist.log_prob(z.unsqueeze(1)).sum(dim=-1)           # (B,C)
        return logp
    

    def forward(self, model_outputs, x_original, batch_indices: Optional[torch.Tensor] = None):
        """
        Computes the full VaDE loss from model outputs and original inputs.

        Args:
            model_outputs: Tuple of (recon_dist, latent_z, q, kmeans_loss, z_mean, z_log_var, gmm_params).
            x_original (torch.Tensor): Original input tensor of shape [B, T, N, F].
            batch_indices (Optional[torch.Tensor]): Sample indices into tau_star for distillation.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of all individual loss terms and the total loss.
        """
        (recon_dist, latent_z, q, kmeans_loss, z_mean, z_log_var, gmm_params) = model_outputs
        device = z_mean.device
        B, T, N, F = x_original.shape
        x_flat = x_original.view(B, T, N * F)

        # Reconstruction loss
        with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
            reconstruction_loss = -(recon_dist.log_prob(x_flat.float())).mean()

        # Ensure q is normalized
        if q is not None:
            eps = 1e-8
            q = q.clamp_min(eps)
            q = q / q.sum(dim=-1, keepdim=True)

        # Activity regularizer: average over batch (matches TF effective scaling)
        activity_l1 = self.l1_activity_weight * torch.sum(torch.abs(z_log_var), dim=-1).mean() 

        # KL weight from scheduler
        klw = 0.0
        if self.kl_scheduler is not None:
            klw = float(self.kl_scheduler.get_weight())

        # KL terms
        z_mean32 = z_mean.float()
        z_log_var32 = z_log_var.float().clamp(min=-4.0, max=2.0)

        if self.pretrain_mode:
            # Pretrain: weak standard-VAE KL against N(0, I).
            # This keeps the latent space compact and well-behaved
            # without imposing mixture structure too early.
            kl_vec = 0.5 * (
                z_mean32.pow(2) + z_log_var32.exp() - 1.0 - z_log_var32
            ).sum(dim=-1) / z_log_var32.shape[-1]
            kl_batch = klw * kl_vec.mean()
        else:
            # Main training: proper MC-KL against the GMM prior.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                kl_vec = self._monte_carlo_kl(
                    z_mean32,
                    z_log_var32,
                    gmm_params["means"].float(),
                    gmm_params["log_vars"].float(),
                    gmm_params["prior"].float(),
                )
                kl_batch = klw * kl_vec

        # Init Losses
        tf_cluster = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        prior_loss = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        cat_cluster_loss = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        repel_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        distill_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        temporal_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        nonempty_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)


        # ----------------------------------------------------------------
        # Losses used in all modes

        if not torch.is_tensor(kmeans_loss):
            kmeans_loss = torch.as_tensor(kmeans_loss, device=device, dtype=reconstruction_loss.dtype)
        if q is not None:
            kmeans_loss = (self.kmeans_loss_weight * kmeans_loss).to(reconstruction_loss.dtype)

        repel_weight = self.repel_weight
        repel_length_scale = self.repel_length_scale          
        if repel_weight > 0.0:
            with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
                # Use data-derived centroids instead of (possibly frozen) GMM means
                if q is not None and latent_z is not None:
                    qf = q.float().detach()  # (B, C)
                    zf = latent_z.float()    # (B, D)
                    pi_b = qf.sum(dim=0).clamp_min(1e-8)  # (C,)
                    means = (qf.t() @ zf) / pi_b.unsqueeze(1)  # (C, D) — soft centroids
                else:
                    means = gmm_params["means"].float()
                
                C = means.size(0)
                diffs = means.unsqueeze(1) - means.unsqueeze(0)
                D2 = (diffs * diffs).sum(dim=-1)
                Kmat = torch.exp(-D2 / max(1e-9, 2.0 * (repel_length_scale ** 2)))
                Kmat = Kmat - torch.diag(torch.diag(Kmat))
                denom = float(max(1, C * C - C))
                repel_loss = (repel_weight * (Kmat.sum() / denom)).to(reconstruction_loss.dtype)


        # Non-empty floor on batch marginal q̄(c)
        nonempty_w = self.nonempty_weight
        if (nonempty_w > 0.0) and (q is not None):
            p = self.nonempty_p
            q_marg = q.mean(dim=0)  # (C,)

            base_floor = float(self.nonempty_floor)
            if self.teacher_marginal is not None:
                pi_t = self.teacher_marginal.to(q_marg.device)  # (C,)
                alpha = 0.9
                floor_c = torch.max(base_floor * torch.ones_like(pi_t),
                                    alpha * pi_t)
            else:
                floor_c = base_floor * torch.ones_like(q_marg)

            underuse = (floor_c - q_marg).clamp_min(0.0)
            pen = underuse.pow(p).sum()
            nonempty_loss = (nonempty_w * pen).to(reconstruction_loss.dtype)

        # ----------------------------------------------------------------
        # Losses only used in main training

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
            rho = self.temporal_cohesion_weight
            if (rho > 0.0) and (q is not None) and (q.size(0) > 1):
                diffs = (q[1:] - q[:-1]).abs().sum(dim=-1).mean()
                temporal_loss = (rho * diffs).to(reconstruction_loss.dtype)

            eta = self.reg_scatter_weight
            beta = self.reg_scatter_beta
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
                scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)

        # ----------------------------------------------------------------
        # Distillation
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

            distill_loss = (self.lambda_distill * distill_loss).to(reconstruction_loss.dtype)
        else:
            distill_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)


        # Total loss
        total = (
            reconstruction_loss
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
            "reconstruct_loss": reconstruction_loss,
            "kl_surrogate": kl_batch,                               
            "kl_div": kl_batch,                        
            "kl_weight": torch.tensor(klw, device=device, dtype=reconstruction_loss.dtype),
            "tf_clust_loss": tf_cluster,
            "prior_loss": prior_loss,
            "kmeans_loss": kmeans_loss,
            "activity_l1": activity_l1,
            "cat_clust_loss": cat_cluster_loss,
            "distill_loss": distill_loss,
            "nonempty_loss": nonempty_loss,
            "temporal_loss": temporal_loss,
            "scatter_loss": scatter_loss,
            "repel_loss": repel_loss,
        }
    

def _init_log_summary():
    """Initialize distionary structure of losses to collect"""

    log_summary={}
    for data_type in ['train','val']:
        log_summary[data_type]={}
        log_summary[data_type]['total_loss']=[]
        log_summary[data_type]['reconstruction_loss']=[]
        log_summary[data_type]['kl_divergence']=[]
        log_summary[data_type]['cat_cluster_loss']=[]
        log_summary[data_type]['kmeans_loss']=[]
        log_summary[data_type]['distill_loss']=[]
        log_summary[data_type]['temporal_loss']=[]
        log_summary[data_type]['scatter_loss']=[]
        log_summary[data_type]['nonempty_loss']=[]
        log_summary[data_type]['repel_loss']=[]
        log_summary[data_type]['tf_cluster_loss']=[]
        log_summary[data_type]['prior_loss']=[]
        log_summary[data_type]['activity_l1']=[]
        log_summary[data_type]['pos_similarity']=[]
        log_summary[data_type]['neg_similarity']=[]
        log_summary[data_type]['conf_norm']=[]
        log_summary[data_type]['bal_norm']=[]
        log_summary[data_type]['alignment_score']=[]


    return log_summary


def _update_log_summary(log_summary: dict, train_logs: dict, val_logs: dict):
    """Append the current losses from train and val logs to the log summary."""
    
    for data_type, logs in zip(['train', 'val'], [train_logs, val_logs]):
        # Append scores directly
        if data_type=="train":
            for key in log_summary.keys():
                if key=="train" or key=="val" or key=="test":
                    continue
                value = logs.get(key, np.nan)
                log_summary[key].append(value)
        # Append losses to training / testing dicts
        for key in log_summary[data_type].keys():
            value = logs.get(key, np.nan)
            log_summary[data_type][key].append(value)
    return log_summary


def _print_losses(model_name: str,
                  log_summary: dict,
                  epoch: int,
                  n_epochs: int,
                  train_logs: dict,
                  val_logs: dict,
                  klw: float = 0.0,
                  lambda_d: float = 0.0):
    """Print losses neatly aligned and append them to the log summary."""

    # Define consistent field width for alignment
    col_width = 10

    def _fmt_loss(name: str, logs: dict, width: int = col_width, precision: int = 4):
        val = logs.get(name, float("nan"))
        # right-aligned, fixed width, fixed decimals
        return f"{val:<{width}.{precision}f}"

    loss_names = [
        "total_loss", "pos_similarity", "neg_similarity", "reconstruct_loss", "prior_loss", "kl_div",
        "kmeans_loss", "cat_clust_loss", "distill_loss", "temporal_loss", 
        "scatter_loss", "repel_loss", "nonempty_loss", "tf_clust_loss",
    ]

    # Print header line
    header = f"Epoch {epoch+1}/{n_epochs}"
    if model_name == "vade":
        header += f" | KLw={klw:.3f}"
    header += f" | λ_distill={lambda_d:.3f}"
 
    print(header)
    print("Losses:")

    # Helper for print losses oderly
    def _print_phase(phase: str, logs: dict):
        line = f"  {phase:<7}:"
        z=0
        for name in loss_names:
            # Skip non-relevant losses
            if np.isnan(logs.get(name, float("nan"))):
                continue
            key_label = name.replace("_loss", "").replace("_", "")
            key_label = key_label.replace("reconstruct", "recon")
            key_label = key_label.replace("similarity", "-sim")
            line += f" {key_label[:9]:<9}: {_fmt_loss(name, logs)} |"
            if (z + 1) % 5 == 0 and z != len(loss_names) - 1:  # line wrap for long outputs
                line += "\n          "
            z+=1
        print(line.rstrip("|"))

    # Print train and validation sections
    _print_phase("Train", train_logs)
    _print_phase("Val", val_logs)

    print("Scores:")

    # Alignment metrics
    footer = f"  Train total={train_logs.get('total_loss', np.nan):.4f} | Val total={val_logs.get('total_loss', np.nan):.4f}"
    footer += (f" | Align: conf={val_logs.get('conf_norm', np.nan):.3f} | " +
        f"bal={val_logs.get('bal_norm', np.nan):.3f} | " +
        f"score={val_logs.get('alignment_score', np.nan):.3f}")
    print(footer)
        

    # Update summary
    log_summary = _update_log_summary(log_summary, train_logs, val_logs)
    return log_summary


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
        "reconstruct_loss", "enc_rec_loss",
        "kl_div",
        "vq_loss", "kmeans_loss",
        "repel_loss", "nonempty_loss", 
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
    model_name: str,
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
        x, a = batch[0], batch[1]              # <----
        idx = batch[-2]   # assume dataset returns (x, a)
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
            
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            #if torch.isnan(model.encoder.node_recurrent_block.conv1d.weight).any():
            #    print("z issues!")
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

        if ctx is not None and model_name=="vade":
            if hasattr(ctx.criterion, "kl_scheduler") and ctx.criterion.kl_scheduler is not None:
                ctx.criterion.kl_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_kl_weight=ctx.criterion.kl_scheduler.get_weight()
            if hasattr(ctx.criterion, "lambda_scheduler") and ctx.criterion.lambda_scheduler is not None:
                ctx.criterion.lambda_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_lambda_weight=ctx.criterion.lambda_scheduler.get_weight()
        elif ctx is not None:
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
        x, a = batch[0], batch[1]
        idx = batch[-2]
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
    if hasattr(ctx.criterion, "kl_scheduler") and (ctx.criterion.kl_scheduler is not None):
        w = float(ctx.criterion.kl_scheduler.get_weight())
        w_max = float(getattr(ctx.criterion.kl_scheduler, "max_weight", 1.0))
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

        if hasattr(ctx.criterion, "kl_scheduler") and ctx.criterion.kl_scheduler is not None:
            ctx.criterion.kl_weight = float(ctx.criterion.kl_scheduler.get_weight())
        elif hasattr(ctx.criterion, "kl_weight"):
            ctx.criterion.kl_weight = float(ctx.criterion.kl_weight)
        if hasattr(ctx, "lambda_scheduler") and ctx.criterion.lambda_scheduler is not None:
            ctx.criterion.lambda_distill = float(ctx.criterion.lambda_scheduler.get_weight())

        loss_dict = ctx.criterion(outputs, x, batch_indices=batch_indices)
        total = loss_dict["total_loss"]
        
        if loss_dict["reconstruct_loss"] > 50000:
            print("Halt!")

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
            "reconstruct_loss": float(loss_dict["reconstruct_loss"].detach().item()),
            "kl_div": float(loss_dict["kl_div"].detach().item()),
            "cat_clust_loss": float(loss_dict["cat_clust_loss"].detach().item()),
            "kmeans_loss": float(loss_dict["kmeans_loss"].detach().item()) if torch.is_tensor(loss_dict["kmeans_loss"]) else float(loss_dict["kmeans_loss"]),
            "activity_l1": float(loss_dict["activity_l1"].detach().item()),
            "prior_loss": float(loss_dict["prior_loss"].detach().item()),
            "distill_loss": float(loss_dict["distill_loss"].detach().item()),
            "tf_clust_loss" : float(loss_dict["tf_clust_loss"].detach().item()),
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
    lambda_distill = 0.0
    if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
        lambda_distill = float(ctx.lambda_scheduler.get_weight())

    if apply_distill and hasattr(ctx, "distill_head") and lambda_distill > 0.0:
        z_e = encoder_output
        logits = ctx.distill_head(z_e)

        idx = idx.to(device).long()
        tau_b = ctx.tau_star[idx]

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
        "reconstruct_loss": float(rec_loss.detach().item()),
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


def _recompute_edges(
    x: torch.Tensor,           # (B, T, N, 3) with [x,y,speed] per node
    edge_index: torch.Tensor,  # indices pairs of nodes to connect
) -> torch.Tensor:
    """
    Recompute edge distances from node coordinates.

    Returns:
        a: (B, T, E, 1) where a[..., e, 0] is the Euclidean distance between the
           two nodes specified by edge_index[e].
    """
    # vvvvv NEW block vvvvv
    if x.ndim != 4 or x.size(-1) < 2:
        raise ValueError(f"x must have shape (B,T,N,>=2). Got {tuple(x.shape)}")
    if edge_index.ndim != 2 or edge_index.size(-1) != 2:
        raise ValueError(f"edge_index must have shape (E,2). Got {tuple(edge_index.shape)}")

    coords = x[..., 0:2]  # (B,T,N,2)

    # Ensure edge_index on same device
    if edge_index.device != x.device:
        edge_index = edge_index.to(x.device)

    i = edge_index[:, 0].long()  # (E,)
    j = edge_index[:, 1].long()  # (E,)

    pi = coords.index_select(dim=2, index=i)  # (B,T,E,2)
    pj = coords.index_select(dim=2, index=j)  # (B,T,E,2)

    d2 = (pi - pj).pow(2).sum(dim=-1)         # (B,T,E)
    d = torch.sqrt(torch.clamp(d2, min=1e-12))  # (B,T,E)

    return d.unsqueeze(-1)  # (B,T,E,1)


def step_contrastive_distill(
    model: nn.Module,  # ContrastivePT or DataParallel(ContrastivePT)
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, a, idx)
    ctx: SimpleNamespace,
) -> StepResult:
    x_full, a_full, idx = batch
    base = unwrap_dp(model)
    device = x_full.device
    apply_distill = getattr(ctx, "apply_distill", True)
    edge_index = getattr(ctx, "edge_index", None)  
    if edge_index is None:
        raise RuntimeError("ctx.edge_index is required for contrastive augmentation!")
    
    contrastive_cfg=getattr(ctx, "contrastive_cfg", None)
    
    a_full = _recompute_edges(x_full, edge_index)
    rot_precomp = getattr(ctx, "rot_precomp", None)
    if rot_precomp is None:
        raise RuntimeError("ctx.rot_precomp is required (build it once in fit_contrastive).")

    x_aug, a_aug = _make_augmented_view(
        x_full, a_full, edge_index, rot_precomp,
        min_shift = contrastive_cfg.aug_min_shift,
        max_shift = contrastive_cfg.aug_max_shift,
        p_shift = contrastive_cfg.aug_p_shift,
        noise_sigma = contrastive_cfg.aug_noise_sigma,  
        p_noise = contrastive_cfg.aug_p_noise,           
        max_interp = contrastive_cfg.aug_max_interp,
        min_interp = contrastive_cfg.aug_min_interp,         
        p_interp = contrastive_cfg.aug_p_interp, 
        max_rot = contrastive_cfg.aug_max_rot, 
        n_rot = contrastive_cfg.aug_n_rot,
        p_rot = contrastive_cfg.aug_p_rot,         
    )

    # Cut middle section from tensor
    half_len = x_full.shape[1] // 2
    starts=(torch.ones([x_full.shape[0]],device=x_full.device)*half_len // 2).int()

    x = deepof.clustering.model_utils_new._slice_time_per_sample(x_full, starts, half_len)
    a = deepof.clustering.model_utils_new._slice_time_per_sample(a_full, starts, half_len)
        
    # Encode via forward for DP compatibility
    z = model(x, a)
    z_aug = model(x_aug, a_aug)

    # Normalize row-wise
    z = torch.nn.functional.normalize(z, dim=1)
    z_aug = torch.nn.functional.normalize(z_aug, dim=1)

    # Base contrastive loss
    loss, pos_mean, neg_mean = select_contrastive_loss_pt(
        z, z_aug,
        similarity=base.similarity_function,
        loss_fn=base.loss_function,
        temperature=base.temperature,
        tau=base.tau,
        beta=base.beta,
        elimination_topk=0.1,
    )

    distill_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)

    # Distillation on the main window embedding
    lambda_distill = 0.0
    if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
        lambda_distill = float(ctx.lambda_scheduler.get_weight())

    if apply_distill and hasattr(ctx, "distill_head") and lambda_distill > 0.0:
        z_main = z
        logits = ctx.distill_head(z_main)

        idx = idx.to(device).long()
        tau_b = ctx.tau_star[idx]

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

        distill_loss = lambda_distill * distill_loss
        total = loss + distill_loss
    else:
        total = loss

    logs = {
        "total_loss": float(total.detach().item()),
        "pos_similarity": float(pos_mean.detach().item()),
        "neg_similarity": float(neg_mean.detach().item()),
        "distill_loss": float(distill_loss.detach().item()) if torch.is_tensor(distill_loss) else float(distill_loss),
    }
    return StepResult(loss=total, logs=logs)


###########
# Main call
###########

def _check_model_inputs(
    model_name: Optional[str] = None,
    encoder_type: Optional[str] = None,
    kl_annealing_mode: Optional[str] = None,
    contrastive_similarity_function: Optional[str] = None,
    contrastive_loss_function: Optional[str] = None, 
):  # pragma: no cover
    """
    Checks and validates enum-like input parameters for various plot functions.

    This function acts as a centralized guard to ensure that all categorical
    and list-based inputs are valid before being used in downstream logic.

    Args:
        model_name (str): Name of the model
        encoder_type (str): Type of encode-decoder pair being used
        kl_annealing_mode (str): Which function should be used to increase and decrease KL
        contrastive_similarity_function (str): Which function should be used to calculate similarity between sampels for the contrastive model
        contrastive_loss_function (str): Which function should be used to calculate the loss for the contrastive model
    """    

    # =========================================================================
    # 1. GENERATE LISTS OF VALID OPTIONS
    # =========================================================================
    
    # --- Statically defined options ---
    model_opts = ["VaDE", "VQVAE", "Contrastive"]
    encoder_opts = ["recurrent", "TCN", "transformer"]
    kl_annealing_mode_opts = ["linear","sigmoid","tf_sigmoid"]
    contrastive_similarity_function_opts = ["cosine","dot","euclidean","edit"]
    contrastive_loss_function_ops=["nce","fc", "dlc", "hard_dcl"]

    # =========================================================================
    # 3. CONFIGURE AND RUN VALIDATION CHECKS
    # Format: (param_name, param_value, valid_options, is_list, custom_error)
    # =========================================================================
    validation_checks = [
        ("model_name", model_name, model_opts, False, None, True, False),
        ("encoder_type", encoder_type, encoder_opts, False, None, True, False),
        ("kl_annealing_mode", kl_annealing_mode, kl_annealing_mode_opts, False, None, True, False),
        ("contrastive_similarity_function", contrastive_similarity_function, contrastive_similarity_function_opts, False, None, True, False),
        ("contrastive_loss_function", contrastive_loss_function, contrastive_loss_function_ops, False, None, True, False),
    ]

    for name, value, options, is_list, error_msg, only_one_of_many, can_be_dict in validation_checks:
        deepof.utils.validate_parameter(name, value, options, is_list, error_msg, only_one_of_many, can_be_dict)


def embedding_model_fittingPT(
    preprocessed_object: Tuple[dict, dict],
    adjacency_matrix: np.ndarray,
    meta_info: dict,
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
    kl_annealing_mode: str = "tf_sigmoid",
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
    teacher_batch_size: int = 2048,
    # Vade pretrain
    kmeans_loss_pretrain: float = 1.0,
    repel_weight_pretrain: float = 0.5,
    repel_length_scale_pretrain: float = 0.5,
    nonempty_weight_pretrain: float = 2e-2,
    nonempty_p_pretrain: float = 2.0,
    nonempty_floor_percent_pretrain: float = 0.05,
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
    nonempty_floor_percent: float = 0.05,
    nonempty_p: float = 2.0,
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
    # Contrastive augmentations
    aug_min_shift: int = 1,
    aug_max_shift: int = 3,
    aug_p_shift: int = 0.4,
    aug_max_rot: int = 30, 
    aug_n_rot: int = 3, 
    aug_p_rot: int = 0.8,
    aug_max_interp: int = 8,
    aug_min_interp: int = 3,         
    aug_p_interp: float = 0.4, 
    aug_noise_sigma: float = 0.03,  
    aug_p_noise: float = 0.4, 
    # Dataset management 
    h5_dataset_folder: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    
    # Verify if various model inputs have valid values (TO DO)
    _check_model_inputs()

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
        teacher_batch_size=teacher_batch_size,
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
        nonempty_floor_percent=nonempty_floor_percent,
        nonempty_p=nonempty_p,

        kmeans_loss_pretrain = kmeans_loss_pretrain,
        repel_weight_pretrain = repel_weight_pretrain,
        repel_length_scale_pretrain = repel_length_scale_pretrain,
        nonempty_weight_pretrain = nonempty_weight_pretrain,
        nonempty_p_pretrain = nonempty_p_pretrain,
        nonempty_floor_percent_pretrain = nonempty_floor_percent_pretrain,
    )

    contrastive_cfg = ContrastiveCfg(
        temperature=temperature,
        contrastive_similarity_function=contrastive_similarity_function,
        contrastive_loss_function=contrastive_loss_function,
        beta=beta,
        tau=tau,
        aug_min_shift=aug_min_shift,
        aug_max_shift=aug_max_shift,
        aug_p_shift=aug_p_shift,
        aug_noise_sigma=aug_noise_sigma,
        aug_p_noise=aug_p_noise,
        aug_min_interp=aug_min_interp,
        aug_max_interp=aug_max_interp,
        aug_p_interp=aug_p_interp,
        aug_max_rot=aug_max_rot,
        aug_n_rot=aug_n_rot,
        aug_p_rot=aug_p_rot,
    )

    return embedding_model_fitting(
        preprocessed_object, 
        adjacency_matrix,
        meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg, 
        vade_cfg=vade_cfg,
        contrastive_cfg=contrastive_cfg,
        h5_dataset_folder=h5_dataset_folder,
    )


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
    meta_info: dict,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    vade_cfg: VaDECfg,
    contrastive_cfg: ContrastiveCfg,
    h5_dataset_folder: str = None,
    shuffle: bool = True,
    device: str = None,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:


    # ----------------------------------------------------
    # Prepare device and data
    # ----------------------------------------------------
    model_name = common_cfg.model_name # Name defaults to "vade"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str) and (device=="cpu" or device=="gpu"):
        device = torch.device(device)
    else:
        raise ValueError("If a device is given, it needs to be either cpu or gpu!")
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(common_cfg.seed)
    np.random.seed(common_cfg.seed)

    if h5_dataset_folder is None:
        data_path = os.path.join(common_cfg.output_path, "Datasets")
    else:
        data_path = h5_dataset_folder
    preprocessed_train, preprocessed_val = preprocessed_object
    train_dataset = BatchDictDataset(
        preprocessed_train, data_path, "train_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=None
    )
    val_dataset = BatchDictDataset(
        preprocessed_val, data_path, "val_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=None
    )

    train_loader = train_dataset.make_loader(
        batch_size=common_cfg.batch_size, shuffle=shuffle, num_workers=common_cfg.num_workers, drop_last=False,
        iterable_for_h5=True, pin_memory=(device.type == 'cuda'), prefetch_factor=common_cfg.prefetch_factor,
        persistent_workers=(common_cfg.num_workers > 0), block_shuffle=shuffle, permute_within_block=False, seed=common_cfg.seed,
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
            meta_info,
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


######################move to utils later

def _load_best_checkpoints(
    model: nn.Module,
    best_path_val: str,
    best_path_score: str,
    device: torch.device,
    save_weights: bool,
) -> Tuple[nn.Module, nn.Module]:
    """
    Loads the best-val and best-score checkpoints into two separate model copies.

    Returns the best-val model and best-score model, both unwrapped from DataParallel.
    If a checkpoint does not exist, the corresponding model retains its current weights.

    Args:
        model (nn.Module): Current model (possibly DataParallel-wrapped).
        best_path_val (str): Path to the best-validation checkpoint.
        best_path_score (str): Path to the best-score checkpoint.
        device (torch.device): Device for loading weights.
        save_weights (bool): Whether weight saving was enabled during training.

    Returns:
    Tuple[nn.Module, nn.Module]: (best_val_model, best_score_model), both unwrapped.
    """
    model_score = deepcopy(model)

    if save_weights and os.path.exists(best_path_val):
        ckpt = torch.load(best_path_val, map_location=device, weights_only=False)
        unwrap_dp(model).load_state_dict(ckpt["state_dict"], strict=False)

    if save_weights and os.path.exists(best_path_score):
        ckpt = torch.load(best_path_score, map_location=device, weights_only=False)
        unwrap_dp(model_score).load_state_dict(ckpt["state_dict"], strict=False)

    return unwrap_dp(model), unwrap_dp(model_score)


def _log_epoch_to_tensorboard(
    writer: Optional[SummaryWriter],
    train_logs: Dict[str, float],
    val_logs: Dict[str, float],
    epoch: int,
    score_value: float = float("nan"),
    lambda_d: float = 0.0,
):
    """
    Writes per-epoch training and validation metrics to TensorBoard.

    Args:
        writer (Optional[SummaryWriter]): TensorBoard writer. If None, this is a no-op.
        train_logs (Dict[str, float]): Training metrics for the current epoch.
        val_logs (Dict[str, float]): Validation metrics for the current epoch.
        epoch (int): Current epoch index.
        score_value (float): Current alignment score. Only logged if finite.
        lambda_d (float): Current distillation lambda weight.
    """
    if writer is None:
        return

    for k, v in train_logs.items():
        writer.add_scalar(f"Train/{k}", v, epoch)
    for k, v in val_logs.items():
        writer.add_scalar(f"Val/{k}", v, epoch)
    writer.add_scalar("Distill/lambda", lambda_d, epoch)

    if math.isfinite(score_value):
        writer.add_scalar("Val/alignment_score", score_value, epoch)
        writer.add_scalar("Val/conf_norm", val_logs.get("conf_norm", float("nan")), epoch)
        writer.add_scalar("Val/bal_norm", val_logs.get("bal_norm", float("nan")), epoch)

######################move to utils later

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

    model_name = "vqvae"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "interaction_regularization": common_cfg.interaction_regularization,
    }

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
    if tau_star is not None:
        tau_star = tau_star.to(device)

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler = None
    if apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch,
            mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0,
            at_max_epochs=teacher_cfg.lambda_decay_start,
            max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown,
            end_weight=teacher_cfg.lambda_end_weight,
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = _ckpt_paths("vqvae", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    score_tol = 0.01
    log_summary=_init_log_summary()

    print(f"\n--- Training VQVAE for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=False,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float("nan")

        if apply_distill:
            diag = _compute_diagnostics(
                model=model,
                dataloader=val_loader,
                q_fn=partial(_get_q_vqvae, distill_head=distill_head),
                device=device,
                n_components=common_cfg.n_components,
                tau_star=tau_star,
                distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
                distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
                distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
                max_batches=common_cfg.diag_max_batches,
            )
            val_logs.update(diag)
            score_value = float(val_logs["alignment_score"])
        else:
            val_logs["alignment_score"] = float("nan")
            val_logs["conf_norm"] = float("nan")
            val_logs["bal_norm"] = float("nan")

        # Print training progress            
        log_summary = _print_losses(model_name="vqvae", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)
        _log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value, lam)

        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=None,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        improved_score = (
            apply_distill and math.isfinite(score_value) and
            (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and v_total < best_score_val)
            )
        )

        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")
        

    # Load states of best val and score models
    model_val, model_score = _load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model_val), unwrap_dp(model_score), None, log_summary


def fit_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    meta_info: dict,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    contrastive_cfg: ContrastiveCfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    model_name = "contrastive"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "interaction_regularization": common_cfg.interaction_regularization,
    }

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

    n_nodes = train_loader.dataset.x_shape[1]
    edge_index_global, edge_index_local, _ = _build_edge_from_metainfo(
    meta_info=meta_info,
    device=device,
    n_nodes=n_nodes,
    return_local=True,
    )
    rot_precomp = build_rotation_precomp(edge_index=edge_index_local, n_nodes=n_nodes, device=device)

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
    if tau_star is not None:
        tau_star = tau_star.to(device)

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler = None
    if apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch,
            mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0,
            at_max_epochs=teacher_cfg.lambda_decay_start,
            max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown,
            end_weight=teacher_cfg.lambda_end_weight,
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = _ckpt_paths("contrastive", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    score_tol = 0.01
    log_summary=_init_log_summary()

    print(f"\n--- Training Contrastive for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
            edge_index=edge_index_global,
            edge_index_local=edge_index_local,
            contrastive_cfg=contrastive_cfg,
            rot_precomp=rot_precomp,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=False, 
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False,edge_index=edge_index_global,edge_index_local=edge_index_local,contrastive_cfg=contrastive_cfg, rot_precomp=rot_precomp), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float("nan")

        if apply_distill:
            diag = _compute_diagnostics(
                model=model,
                dataloader=val_loader,
                q_fn=partial(
                    _get_q_contrastive,
                    distill_head=distill_head,
                    edge_index=edge_index_global,
                ),
                device=device,
                n_components=common_cfg.n_components,
                tau_star=tau_star,
                distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
                distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
                distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
                max_batches=common_cfg.diag_max_batches,
            )
            val_logs.update(diag)
            score_value = float(val_logs["alignment_score"])
        else:
            val_logs["alignment_score"] = float("nan")
            val_logs["conf_norm"] = float("nan")
            val_logs["bal_norm"] = float("nan")

        # Print training progress
        log_summary = _print_losses(model_name="Contrastive", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)
        _log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value, lam)

   
        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
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
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        improved_score = (
            apply_distill and math.isfinite(score_value) and
            (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and v_total < best_score_val)
            )
        )

        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = v_total
            if common_cfg.save_weights:
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
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")


    # Load states of best val and score models
    model_val, model_score = _load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model_val), unwrap_dp(model_score), None, log_summary   


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
    

    ###############
    # Set up
    ###############

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    # Create model and step function
    model = deepof.clustering.models_new.VaDEPT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        n_components=common_cfg.n_components,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        kmeans_loss=vade_cfg.kmeans_loss_pretrain,
        interaction_regularization=common_cfg.interaction_regularization,
    ).to(device, non_blocking=True)
    step_fn = step_vade

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # More setup
    optimizer = build_optimizer(model=model, base_lr=1e-3, gmm_lr=1e-3)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))
    n_batches_per_epoch = len(train_loader)

    model_name = "vade"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "kmeans_loss": common_cfg.kmeans_loss,
        "interaction_regularization": common_cfg.interaction_regularization,
        "lens_enabled": False,
    }

    # Load pretrained model if available + early return
    if common_cfg.pretrained:
        print(f"Loading pretrained weights from {common_cfg.pretrained}")
        unwrap_dp(model).load_state_dict(torch.load(common_cfg.pretrained, map_location=device, weights_only=False))
        if writer:
            writer.flush(); writer.close()
        return unwrap_dp(model), unwrap_dp(model), None


    ###############
    # Define function, will start in pretrain mode
    ###############

    criterion = VadeLoss(
        common_cfg=common_cfg,
        vade_cfg=vade_cfg,
        teacher_cfg=teacher_cfg,
    ).to(device)


    ###############
    # Pretraining
    ###############

    print("\n--- Pretraining (reconstruction and setting up the latent space) ---")
    unwrap_dp(model).set_pretrain_mode(True)
    pre_epochs = min(10, max(1, common_cfg.epochs))
    ctx = SimpleNamespace(criterion=criterion, scheduler=None, scheduler_per_batch=True)
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
        warmup_epochs=15, max_weight=0.2, cooldown_epochs=10, end_weight=0.2
    )
    criterion.set_kl_scheduler(kl_scheduler)

    leave = False 
    for ep in range(pre_epochs):
        
        # Leave loading bar in last step (for optics)
        if ep == len(range(pre_epochs))-1:
            leave=True
        apply_decoder_schedule(model, ep)
        pre_logs, _, _ = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=ep, num_epochs=pre_epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=leave
        )
        if writer:
            writer.add_scalar("Pretrain/total_loss", pre_logs["total_loss"], ep)

    ###############
    # Main training
    ###############

    # Finish pretraining, reset KL schedule
    unwrap_dp(model).set_pretrain_mode(False)
    criterion.set_mode("main")
    # New KL schedule
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
        warmup_epochs=common_cfg.kl_warmup, max_weight=common_cfg.kl_max_weight,
        cooldown_epochs=common_cfg.kl_cooldown, end_weight=common_cfg.kl_end_weight
    )
    criterion.set_kl_scheduler(kl_scheduler)
    
    optimizer = build_optimizer(model=model, base_lr=1e-3, gmm_lr=1e-3)

    # VaDE unified checkpoint paths
    _, best_path_val, best_path_score, teacher_init_path = _ckpt_paths("vade", common_cfg=common_cfg)

    tau_star = None
    teacher_init_model = None  # returned as 3rd output
    # cached views for refresh
    pca_pos = pca_spd = pca_edges = pca_angles_train = supervised_labels = None

    log_summary=_init_log_summary()
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
        criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill, lambda_scheduler=lambda_scheduler)

        # Save teacher-init checkpoint + info (VaDE only)
        teacher_init_model = deepcopy(unwrap_dp(model))
        if common_cfg.save_weights:
            save_model_info(
                teacher_init_path,
                stage="teacher_init",
                epoch=pre_epochs - 1,
                train_steps=pre_epochs * len(train_loader),
                extra={"note": "after pretrain + teacher + GMM init, before main training"},
                common_cfg=common_cfg,
                teacher_cfg=teacher_cfg,
                vade_cfg=vade_cfg,
                model=unwrap_dp(model),
                log_summary=log_summary,
                rebuild_spec=rebuild_spec,
                save_weights=common_cfg.save_weights,
                save_bundle=True,
            )
            print(f"  Saved teacher-init model -> {teacher_init_path}")

    else:
        # If there is no teacher, init GMM directly with train_loader
        print("\n--- Initializing GMM from embeddings (sklearn) ---")
        unwrap_dp(model).initialize_gmm_from_data(train_loader)

    # Inits for training
    best_val = -float("inf") #start negative, as Vade first is expected to get worse validation wise, then top out and get better
    best_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_tol = 0.01
    val_tol = 0.01
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    val_top_reached=False

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
                batch_size = teacher_cfg.teacher_batch_size,
            )
            tau_star = tau_star.detach()
            criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill, lambda_scheduler=lambda_scheduler)

            # Optionally reinit GMM
            if teacher_cfg.reinit_gmm_on_refresh:
                initialize_gmm_from_teacher(model, z_curr, tau_star, min_var=1e-4)
                print("  Reinitialized GMM from refreshed τ*.")

        # Actual training of the main model
        ctx = SimpleNamespace(criterion=criterion, scheduler_per_batch=True)
        # klw is updated in every training step, getting the weight for printing at the middle of the epochs
        train_logs, klw, lambda_d = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_fn, device=device,
            epoch=epoch, num_epochs=common_cfg.epochs, ctx=SimpleNamespace(criterion=criterion, apply_distill=False),
            show_progress=True,
        )

        # A ton of diagnostics for printing training progress
        diag = _compute_diagnostics(
            model=model,
            dataloader=val_loader,
            q_fn=_get_q_vade,
            device=device,
            n_components=common_cfg.n_components,
            tau_star=getattr(criterion, "tau_star", None),
            distill_sharpen_T=float(getattr(criterion, "distill_sharpen_T", 0.5)),
            distill_conf_weight=bool(getattr(criterion, "distill_conf_weight", False)),
            distill_conf_thresh=float(getattr(criterion, "distill_conf_thresh", 0.55)),
            max_batches=common_cfg.diag_max_batches,
            extra_stats_fn=_compute_vade_specific_diagnostics,
        )
        
        val_logs.update(diag)
        val_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float(val_logs["alignment_score"])

        log_summary = _print_losses(model_name="vade", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, klw=klw, lambda_d=lambda_d, train_logs=train_logs, val_logs=val_logs)
        _log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value)

        # Deterimine if validation loss and / or balance + certainty score has improved
        improved_val = ((val_total + val_tol) < best_val)

        if not improved_val and not val_top_reached:
            best_val = val_total

        improved_score = (
            math.isfinite(score_value) and (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and val_total < best_score_val)
            )
        )

        # Save best model based on total validation loss
        if improved_val:
            val_top_reached=True
            best_val = val_total
            val_tol=0.0
            if common_cfg.save_weights:
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"Saved best VAL model -> {best_path_val} (val={best_val:.4f})")


        # Save best model based on model balance and certainty score
        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = val_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                    save_bundle=True,
                )
                print(f"Saved best SCORE model -> {best_path_score} (score={best_score:.4f})")


    # Load states of best val and score models
    model_val, model_score = _load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model_val), unwrap_dp(model_score), teacher_init_model, log_summary    


def _build_edge_from_metainfo(
    meta_info: dict,
    device: torch.device,
    n_nodes: int,
    return_local: bool = True,
    return_node_names: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Builds:
      - edge_index_global: (E,2) long on device, includes cross-mouse edges (e.g. B_Nose--W_Nose)
      - edge_index_local:  (E_local,2) long on device, excludes cross-mouse edges (within-mouse only)
      - node_names: list[str] length n_nodes in node-axis order (optional)

    Assumes node names are like 'B_Nose', 'W_Tail_base', etc.
    """
    if "node_columns" not in meta_info or "edge_columns" not in meta_info:
        raise RuntimeError("meta_info must contain 'node_columns' and 'edge_columns'.")

    node_cols = list(meta_info["node_columns"])
    edge_cols = list(meta_info["edge_columns"])

    # 1) Infer node order from the first n_nodes entries of form (bp, 'x')
    node_names: List[str] = []
    for c in node_cols:
        if isinstance(c, tuple) and len(c) == 2 and c[1] == "x":
            node_names.append(c[0])
            if len(node_names) == n_nodes:
                break

    if len(node_names) != n_nodes:
        raise RuntimeError(
            f"Failed to infer {n_nodes} node names from meta_info['node_columns']. Got {len(node_names)}."
        )

    node_to_idx = {name: i for i, name in enumerate(node_names)}

    # 2) Build global edge index 
    pairs = []
    for (u_name, v_name) in edge_cols:
        if u_name not in node_to_idx or v_name not in node_to_idx:
            raise RuntimeError(
                f"Edge ({u_name},{v_name}) contains node(s) not found in inferred node list."
            )
        pairs.append((node_to_idx[u_name], node_to_idx[v_name]))

    edge_index_global = torch.tensor(pairs, dtype=torch.long, device=device)

    if not return_local:
        return edge_index_global, None, (node_names if return_node_names else None)

    # 3) Build local/within-mouse edge index by filtering on name prefix ('B_' vs 'W_')
    def animal_key(name: str) -> str:
        # 'B_Right_bhip' -> 'B', 'W_Nose' -> 'W'
        # (split on first underscore)
        if "_" not in name:
            return("")
        return name.split("_", 1)[0]

    keys = [animal_key(nm) for nm in node_names]  # len N
    # map each node index -> animal id (e.g., B->0, W->1)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    animal_id = torch.tensor([uniq[k] for k in keys], dtype=torch.long, device=device)  # (N,)

    u = edge_index_global[:, 0]
    v = edge_index_global[:, 1]
    same_animal = (animal_id[u] == animal_id[v])
    edge_index_local = edge_index_global[same_animal]

    return edge_index_global, edge_index_local, (node_names if return_node_names else None)


def _plot_augmentation(x_in: torch.Tensor, x_aug: torch.Tensor):
    """
    Plots one random batch element as a row of skeletons over time (top: original, bottom: augmented).
    Uses _plot_augmentation._edge_index if available (set by augmentation functions).
    """
    import matplotlib.pyplot as plt

    edge_index = getattr(_plot_augmentation, "_edge_index", None)

    b = int(torch.randint(0, x_in.size(0), (1,)).item())
    xin = x_in[b, ::4, :, 0:2].detach().cpu()   # (T,N,2)
    xau = x_aug[b, ::4, :, 0:2].detach().cpu()  # (T,N,2)
    T = xin.size(0)

    dx = 2.5  # horizontal offset per frame
    fig, ax = plt.subplots(2, 1, figsize=(min(28, 1.2 * T), 6), sharey=True)

    def draw_row(ax_, X, title):
        for t in range(T):
            off = t * dx
            pts = X[t]  # (N,2)
            xs = pts[:, 0].numpy() + off
            ys = pts[:, 1].numpy()
            ax_.plot(xs, ys, "kx", ms=3)

            if edge_index is not None:
                ei = edge_index.detach().cpu().numpy()
                for (i, j) in ei:
                    ax_.plot([xs[i], xs[j]], [ys[i], ys[j]], "k-", lw=0.8, alpha=0.7)

        ax_.set_title(title)
        ax_.axis("off")

    draw_row(ax[0], xin, "original")
    draw_row(ax[1], xau, "augmented")
    range_min=np.min([plt.xlim(),plt.ylim()])
    range_max=np.max([plt.xlim(),plt.ylim()])
    plt.xlim([range_min,range_max])
    plt.ylim([range_min,range_max])
    plt.tight_layout()
    plt.show()


def _valid_triplets_from_edge_index(ei: torch.Tensor, n_nodes: int):
    adj = [[] for _ in range(n_nodes)]
    for u, v in ei.detach().cpu().tolist():
        adj[u].append(v)
        adj[v].append(u)

    triplets = []
    for b in range(n_nodes):
        nb = adj[b]
        if len(nb) < 2:
            continue
        # all unordered neighbor pairs define (a,b,c)
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a_ = nb[i]
                c_ = nb[j]
                triplets.append((a_, b, c_))
    return triplets, adj


@dataclass
class RotationPrecomp:
    # Triplets (a,b,c) in a tensor for cheap access
    triplets: torch.Tensor            # (M,3) long, on device
    centers: torch.Tensor             # (M,) long, on device

    # Variable-length node lists, stored as list-of-tensors (each on device)
    branches_a: List[torch.Tensor]    # len M, each (Ka,) long
    branches_c: List[torch.Tensor]    # len M, each (Kc,) long

    # 0 => prefer a-branch, 1 => prefer c-branch, 2 => tie (random at runtime)
    prefer_side: torch.Tensor         # (M,) uint8 on device


def build_rotation_precomp(edge_index: torch.Tensor, n_nodes: int, device: torch.device) -> RotationPrecomp:
    """
    Build triplets and per-triplet branch node sets ONCE.

    This runs Python/CPU graph logic once, but stores results as CUDA tensors (if device is CUDA)
    so the augmentation step doesn't do BFS anymore.
    """
    ei_cpu = edge_index.detach().cpu().tolist()

    # adjacency list on CPU
    adj = [[] for _ in range(n_nodes)]
    for u, v in ei_cpu:
        adj[u].append(v)
        adj[v].append(u)

    # triplets (a,b,c): for each center b, all unordered neighbor pairs (a,c)
    triplets_py: List[Tuple[int,int,int]] = []
    for b in range(n_nodes):
        nb = adj[b]
        if len(nb) < 2:
            continue
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                triplets_py.append((nb[i], b, nb[j]))

    # branch BFS (still CPU, but one-time)
    def branch_nodes(center: int, side: int) -> List[int]:
        seen = {side}
        stack = [side]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v == center or v in seen:
                    continue
                seen.add(v)
                stack.append(v)
        return list(seen)  # excludes center by construction

    branches_a: List[torch.Tensor] = []
    branches_c: List[torch.Tensor] = []
    prefer: List[int] = []

    for (a, b, c) in triplets_py:
        ba = branch_nodes(b, a)
        bc = branch_nodes(b, c)

        branches_a.append(torch.tensor(ba, dtype=torch.long, device=device))
        branches_c.append(torch.tensor(bc, dtype=torch.long, device=device))

        prefer.append(2)

        #if len(ba) < len(bc):
        #    prefer.append(0)
        #elif len(bc) < len(ba):
        #    prefer.append(1)
        #else:
        #    prefer.append(2)  # tie

    triplets = torch.tensor(triplets_py, dtype=torch.long, device=device)
    centers = triplets[:, 1].contiguous() if triplets.numel() else torch.empty((0,), dtype=torch.long, device=device)
    prefer_side = torch.tensor(prefer, dtype=torch.uint8, device=device)

    return RotationPrecomp(
        triplets=triplets,
        centers=centers,
        branches_a=branches_a,
        branches_c=branches_c,
        prefer_side=prefer_side,
    )


def _augment_time_shift(
    x: torch.Tensor,             # (B,T_full,N,3)
    edge_index: torch.Tensor,    # (E,2) (not used here, kept for signature consistency / plotting)
    min_shift: int = 1,
    max_shift: int = 3,
    p: float = 0.8,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a half-window slice (T_full//2) from the middle of the full window.
    If triggered, shifts the slice start by +/- U[min_shift, max_shift] (per sample).
    Shift is consistent across the whole window (no frame-to-frame jitter).
    """
    B, T = x.shape[0], x.shape[1]
    half_len = T // 2
    base = (T - half_len) // 2  # == T//4 when T even

    # vvvvv sample shifts per sample vvvvv
    apply = (torch.rand(B, device=x.device) < p)

    mag = torch.randint(min_shift, max_shift + 1, (B,), device=x.device)     # (B,)
    sgn = (torch.randint(0, 2, (B,), device=x.device) * 2 - 1)               # (B,) in {-1,+1}
    shift = mag * sgn
    shift = shift * apply.long()  # (B,) zero if not applied

    start = base + shift
    start = start.clamp(0, T - half_len)  # keep valid
    # <^^^^ sample shifts <^^^^

    x_cut = deepof.clustering.model_utils_new._slice_time_per_sample(x, start, half_len)

    if plot:
        # show what changed (note: this plots only the cut windows)
        _plot_augmentation._edge_index = edge_index
        _plot_augmentation(deepof.clustering.model_utils_new._slice_time_per_sample(x, (torch.ones([B],device=x.device)*(T - half_len) // 2).int(), half_len), x_cut)

    return x_cut


def _augment_full_rotation(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # only used for plotting
    max_rot: float = 30.0,
    p: float = 0.5,
    plot: bool = False,
) -> torch.Tensor:
    """
    Rotate the entire graph (all nodes) by a single angle per sample (constant over time).
    Rotation is applied around the per-frame centroid (keeps the skeleton "in place").
    """
    if max_rot <= 0.0 or p <= 0.0:
        return x

    B = x.size(0)
    x_aug = x.clone()

    apply = (torch.rand(B, device=x.device) < p).to(x.dtype)  # (B,)
    theta = (torch.rand(B, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * (max_rot * math.pi / 180.0)
    theta = theta * apply

    cos_t = torch.cos(theta).view(B, 1, 1, 1)  # (B,1,1,1)
    sin_t = torch.sin(theta).view(B, 1, 1, 1)

    coords = x_aug[..., 0:2]                    # (B,T,N,2)
    pivot = coords.mean(dim=2, keepdim=True)    # (B,T,1,2)
    rel = coords - pivot                        # (B,T,N,2)

    rx = rel[..., 0:1] * cos_t - rel[..., 1:2] * sin_t
    ry = rel[..., 0:1] * sin_t + rel[..., 1:2] * cos_t
    coords = torch.cat([rx, ry], dim=-1) + pivot

    x_aug[..., 0:2] = coords

    if plot:
        _plot_augmentation._edge_index = edge_index
        _plot_augmentation(x, x_aug)

    return x_aug


def _augment_angle_rotations(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    rot_precomp: RotationPrecomp,
    n_rot: int = 3,
    max_rot: float = 30.0,
    p: float = 0.5,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly apply up to n_rot joint-like rotations (consistent across time per sample).
    Branch selection is done by a graph-cut rule:
      - For a center b and neighbor side s, rotate the connected component reachable from s
        when traversal through b is forbidden.
    This yields more realistic articulated rotations (e.g., center rotating head/tail as a unit).

    Notes on speed:
      - n_rot is small (<=3), so small Python loops are fine.
      - Rotations are vectorized across (B,T,...) with Torch ops.
    """
    B, T, N, _ = x.shape
    M = int(rot_precomp.triplets.shape[0])
    if n_rot <= 0 or max_rot <= 0.0 or p <= 0.0 or M == 0:
        return x

    x_aug = x.clone()
    coords = x_aug[..., 0:2]  # (B,T,N,2)

    apply = (torch.rand(B, device=x.device) < p).to(x.dtype)  # (B,)
    max_rad = float(max_rot) * math.pi / 180.0

    # choose up to n_rot triplets with unique centers (still a tiny Python loop)
    perm = torch.randperm(M, device=x.device)
    chosen_idx = []
    center_count = torch.zeros(N, dtype=torch.int, device=x.device)
    
    # NOTE: this .tolist() is small (M is small for N=11), but you could avoid it later if needed
    for k in perm.tolist():
        b0 = int(rot_precomp.centers[k].item())
        if center_count[b0] >= 2:
            continue
        center_count[b0] += 1
        chosen_idx.append(k)
        if len(chosen_idx) >= n_rot:
            break

    for k in chosen_idx:
        a0, b0, c0 = rot_precomp.triplets[k].tolist()

        pref = int(rot_precomp.prefer_side[k].item())
        if pref == 0:
            rot_nodes = rot_precomp.branches_a[k]
        elif pref == 1:
            rot_nodes = rot_precomp.branches_c[k]
        else:
            # tie
            rot_nodes = rot_precomp.branches_a[k] if torch.rand((), device=x.device) < 0.5 else rot_precomp.branches_c[k]

        if rot_nodes.numel() == 0:
            continue

        theta = (torch.rand(B, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * max_rad
        theta = theta * apply  # (B,)

        cos_t = torch.cos(theta).view(B, 1, 1)
        sin_t = torch.sin(theta).view(B, 1, 1)

        pivot = coords[:, :, b0, :].unsqueeze(2)                  # (B,T,1,2)
        pts = coords.index_select(dim=2, index=rot_nodes)         # (B,T,K,2)
        rel = pts - pivot                                         # (B,T,K,2)

        rx = rel[..., 0] * cos_t - rel[..., 1] * sin_t
        ry = rel[..., 0] * sin_t + rel[..., 1] * cos_t
        new_pts = torch.stack([rx, ry], dim=-1) + pivot           # (B,T,K,2)

        coords[:, :, rot_nodes, :] = new_pts

    x_aug[..., 0:2] = coords

    if plot:
        _plot_augmentation._edge_index = edge_index
        _plot_augmentation(x, x_aug)

    return x_aug


def _augment_noise_xys(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    sigma: float = 0.03,
    p: float = 0.5,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise per bodypart (per sample), consistent across the window:
      - For each (sample, node), choose axis in {x,y} and add a random offset to that axis.
      - Also add random offset to speed channel.
    Recomputes a from x after augmentation.
    """
    if sigma <= 0.0 or p <= 0.0:
        return x

    B, T, N, F = x.shape
    x_aug = x.clone()

    # vvvvv VECTORIZED per-node offsets (no python loop) vvvvv
    apply = (torch.rand(B, device=x.device) < p).to(x.dtype)          # (B,)
    apply_bn = apply.view(B, 1).expand(B, N)                           # (B,N)

    # For each (B,N) choose whether to perturb x or y
    axis = torch.randint(0, 2, (B, N), device=x.device)                # (B,N), 0=x 1=y

    # One offset per (B,N), constant over time
    offset_xy = sigma * torch.randn((B, N), device=x.device, dtype=x.dtype) * apply_bn  # (B,N)
    dx = offset_xy * (axis == 0).to(x.dtype)                            # (B,N)
    dy = offset_xy * (axis == 1).to(x.dtype)                            # (B,N)

    # Speed offsets per (B,N)
    ds = sigma * torch.randn((B, N), device=x.device, dtype=x.dtype) * apply_bn         # (B,N)

    # Broadcast across time
    x_aug[:, :, :, 0] = x_aug[:, :, :, 0] + dx.view(B, 1, N)            # <----
    x_aug[:, :, :, 1] = x_aug[:, :, :, 1] + dy.view(B, 1, N)            # <----
    x_aug[:, :, :, 2] = x_aug[:, :, :, 2] + ds.view(B, 1, N)            # <----

    if plot:
        _plot_augmentation._edge_index = edge_index  # <----
        _plot_augmentation(x, x_aug)                 # <----

    return x_aug


def _augment_linear_interpolate_segments(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    min_len: int = 5,
    max_len: int = 15,
    p: float = 0.3,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace one random contiguous segment (length <= max_len) with linear interpolation, per sample with prob p.
    Applies to all node channels (x,y,speed). Recomputes a from x after augmentation.
    """
    if max_len <= 0 or p <= 0.0:
        return x

    B, T = x.size(0), x.size(1)
    if T < 3:
        return x

    x_aug = x.clone()

    # vvvvv VECTORIZED interpolation (no python loops over bs / frames) vvvvv
    device = x.device
    dtype = x_aug.dtype

    apply = (torch.rand(B, device=device) < p)  # (B,)

    # Sample L per sample (even for non-applied; we mask later)
    L = torch.randint(min_len, max_len + 1, (B,), device=device)  # (B,)

    # Need endpoints at (t0-1) and (t0+L) within [0, T-1]
    # -> t0 in [1, T-L-1]  (inclusive) => randint(1, T-L) (exclusive high)
    t0_max = (T - L).clamp_min(2)  # ensure high>=2 so randint(1, high) is valid
    # torch.randint doesn't support per-element highs directly; sample uniform and mod safely:
    # We'll sample from a generous range then clamp.
    t0 = torch.randint(1, T - 1, (B,), device=device)  # provisional
    t0 = torch.minimum(t0, (T - L - 1).clamp_min(1))   # enforce t0 <= T-L-1

    t_start = t0 - 1                 # (B,)
    t_end = t0 + L                   # (B,)

    # Gather endpoints: (B,N,3)
    b_idx = torch.arange(B, device=device)
    start = x_aug[b_idx, t_start]    # (B,N,3)
    end   = x_aug[b_idx, t_end]      # (B,N,3)

    # Build mask over time: frames to replace are t0..t0+L-1
    tt = torch.arange(T, device=device).view(1, T)                    # (1,T)
    t0v = t0.view(B, 1)                                                # (B,1)
    Lv  = L.view(B, 1)                                                 # (B,1)
    mask = (tt >= t0v) & (tt < (t0v + Lv)) & apply.view(B, 1)          # (B,T)

    # Alpha for each t in the segment: alpha = (t - (t0-1)) / (L+1)
    denom = (Lv + 1).to(dtype)                                         # (B,1)
    alpha = ((tt.to(dtype) - (t0v.to(dtype) - 1.0)) / denom)           # (B,T)
    alpha = alpha.clamp(0.0, 1.0)

    # Interpolated frames: (B,T,N,3)
    start_e = start.unsqueeze(1)                                       # (B,1,N,3)
    end_e   = end.unsqueeze(1)                                         # (B,1,N,3)
    alpha_e = alpha.unsqueeze(-1).unsqueeze(-1)                        # (B,T,1,1)
    interp  = (1.0 - alpha_e) * start_e + alpha_e * end_e              # (B,T,N,3)

    # Apply only on masked frames
    x_aug = torch.where(mask.unsqueeze(-1).unsqueeze(-1), interp, x_aug)
    # <^^^^ VECTORIZED interpolation <^^^^

    if plot:
        _plot_augmentation._edge_index = edge_index  # <----
        _plot_augmentation(x, x_aug)                 # <----

    return x_aug


def _make_augmented_view(
    x: torch.Tensor,   # (B,T,N,3)
    a: torch.Tensor,   # (B,T,E,1)
    edge_index: torch.Tensor,
    rot_precomp: RotationPrecomp,
    min_shift: int = 1,
    max_shift: int = 6,
    p_shift: float = 0.8,
    n_rot: int =3,
    max_rot: int = 30, 
    p_rot: float = 0.7,
    max_interp: int = 6,
    min_interp: int = 5,
    p_interp: float = 0.6,
    noise_sigma: float = 0.02,
    p_noise: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce augmented (x_aug, a_aug). a_aug is recomputed from x_aug, then affine-matched to a.
    """
    x_aug_raw = x

    x_aug = _augment_time_shift(x_aug_raw, edge_index, min_shift=min_shift, max_shift=max_shift, p=p_shift, plot=False)
    #x_aug = _augment_full_rotation(x_aug, edge_index, max_rot=180, p=0.5, plot=False)
    x_aug = _augment_angle_rotations(x_aug, edge_index, rot_precomp, n_rot=n_rot, max_rot=max_rot, p=p_rot, plot=False)
    x_aug = _augment_linear_interpolate_segments(x_aug, edge_index, min_len=min_interp, max_len=max_interp, p=p_interp, plot=False)
    x_aug = _augment_noise_xys(x_aug, edge_index, sigma=noise_sigma, p=p_noise, plot=False)

    a_aug = _recompute_edges(x_aug, edge_index) 

    return x_aug, a_aug
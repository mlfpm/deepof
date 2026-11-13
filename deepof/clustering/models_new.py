"""deep autoencoder models for unsupervised pose detection.

- VQ-VAE: a variational autoencoder with a vector quantization latent-space (https://arxiv.org/abs/1711.00937).
- VaDE: a variational autoencoder with a Gaussian mixture latent-space.
- Contrastive: an embedding model consisting of a single encoder, trained using a contrastive loss.

Models were translated from original tensorflow implementations to Pytorch using LLMs.

"""
# @author lucasmiranda42 (of original Tensorflow implementations) and NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Iterable, Tuple, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import TransformedDistribution, Normal


import deepof.clustering.losses
from deepof.clustering.censNetConv_pt import CensNetConvPT
import warnings
from torch.distributions.transforms import AffineTransform



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def _encoder_head(in_features: int, latent_dim: int, dropout_rate: float = 0.0) -> nn.Sequential:
    """Projection head of the encoders: Dense(2*latent) -> LayerNorm -> Dense(latent) -> LayerNorm -> Dense(latent)."""
    head = nn.Sequential(
        nn.Linear(in_features, 2 * latent_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.LayerNorm(2 * latent_dim),
        nn.Linear(2 * latent_dim, latent_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, latent_dim),
    )
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return head


def _normalize_backbone(enc: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS normalization of the backbone output (the embedding used for clustering), clamped and NaN-free."""
    head_in = enc.float()
    rms = (head_in.pow(2).mean(dim=1, keepdim=True) + 1e-12).sqrt()  # (B, 1), eps: finite gradient at 0
    head_in = head_in / rms.clamp(min=1.0)  # avoid division by tiny values
    head_in = head_in.clamp(min=-1e4, max=1e4)
    return torch.nan_to_num(head_in, nan=0.0, posinf=1e4, neginf=-1e4)


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
            self.node_recurrent_block = RecurrentBlockPT(
                input_features=node_feat_per_node,
                latent_dim=latent_dim,
            )

            # Edge stream
            edge_feat_per_edge = int(edge_feature_shape[-1])
            self.edge_recurrent_block = RecurrentBlockPT(
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

            final_in = (self.num_nodes * latent_dim) + (self.num_edges * latent_dim)

        else:
            # Non-GNN: single recurrent block over a single "group" dimension
            in_features = int(input_shape[1]) * int(input_shape[2])  # N_nodes * F_per_node
            self.recurrent_block = RecurrentBlockPT(
                input_features=in_features,
                latent_dim=latent_dim,
            )
            final_in = 2 * latent_dim

        # Head MLP after concatenating node+edge embeddings, as in the TCN encoder
        self.head = _encoder_head(final_in, latent_dim)

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

    def forward(self, x: torch.Tensor, a: torch.Tensor, cluster_mode=False, return_backbone=False) -> torch.Tensor:
        """
        x: (B, T, N_nodes, F_per_node)
        a: (B, T, E_edges, F_per_edge)
        return_backbone: return (projection, backbone output) instead of one of them
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

        head_in = _normalize_backbone(encoder)
        if return_backbone:
            return self.head(head_in), head_in
        return self.head(head_in) if not cluster_mode else head_in
        

class RecurrentBlockPT(nn.Module):
    def __init__(self, input_features: int, latent_dim: int, bidirectional_merge: str = "concat", internal_dim: int = 32):
        super().__init__()
        self.internal_dim = int(internal_dim)  # 32: the width of the TCN (tcn_conv_filters)
        self.latent_dim = latent_dim
        if bidirectional_merge != "concat":
            warnings.warn("Bidirectional merge mode defaulting to 'concat'.")

        self.conv1d = nn.Conv1d(
            in_channels=input_features,
            out_channels=2 * self.internal_dim,
            kernel_size=5,
            padding="same",
            bias=False,
        )
        self.gru1 = nn.GRU(
            input_size=2 * self.internal_dim,
            hidden_size=2 * self.internal_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm1 = nn.LayerNorm(4 * self.internal_dim, eps=1e-3)
        self.gru2 = nn.GRU(
            input_size=4 * self.internal_dim,
            hidden_size=self.internal_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm2 = nn.LayerNorm(2 * self.internal_dim, eps=1e-3)
        self.projection = nn.Linear(in_features=self.internal_dim*2, out_features=latent_dim*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, T, N, F)
        B, T, N, _ = x.shape

        # Force input onto the same device as this block's parameters
        dev = self.conv1d.weight.device
        x = x.to(dev, non_blocking=True)

        # Stage 1: Conv (TimeDistributed over T)
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            x32 = x.float()    
            conv_in = x32.reshape(B * T, N, -1).permute(0, 2, 1)  # (B*T, F, N)
            conv_out = F.relu(self.conv1d(conv_in))             # (B*T, 2*latent, N)
            gru1_in = conv_out.permute(0, 2, 1)                 # (B*T, N, 2*internal)

            # Stage 2: First GRU over the full sequences (windows have no padding; zeros are valid values)
            gru1_out, _ = self.gru1(gru1_in)                    # (B*T, N, 4*internal)

            # Stage 3: First LayerNorm
            norm1_out = self.norm1(gru1_out)

            # Stage 4: Second GRU, final states of both directions
            _, h_n_2 = self.gru2(norm1_out)                     # (2, B*T, internal)
            gru2_final_state = h_n_2.permute(1, 0, 2).reshape(B * T, -1)

            # Stage 5: Second LayerNorm
            norm2_out = self.norm2(gru2_final_state)  # (B*T, 2*internal)

            final_output = norm2_out.reshape(B, T, -1)  # (B, T, 2*internal)
            if self.internal_dim != self.latent_dim:
                final_output=self.projection(final_output) # restore latent space dependent output shape
        return final_output.to(x.dtype)


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
    

class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over C for (B, C, T). Per-timestep, causal-safe."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)
    

def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind in (None, "none", "identity"):
        return nn.Identity()
    if kind in ("bn", "batch", "batchnorm"):
        return nn.BatchNorm1d(channels, eps=1e-3)
    if kind in ("ln", "layer", "layernorm"):
        return ChannelLayerNorm1d(channels, eps=1e-5)
    if kind in ("gn", "group", "groupnorm"):
        groups = 8 if channels >= 8 and channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels, eps=1e-5)
    raise ValueError(f"unknown norm_type={kind}")


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
        use_batch_norm: bool = False,
        conv_init_std: float = 0.05,
        legacy_tcn: bool = False,
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
        if legacy_tcn:
            self.n1 = nn.BatchNorm1d(out_channels, eps=1e-3) if use_batch_norm else nn.Identity()
        else:
            self.n1 = _make_norm("layer" if not use_batch_norm else "batch", out_channels)
        self.drop1 = nn.Dropout(float(dropout_rate)) if dropout_rate else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, self.kernel_size, dilation=self.dilation, padding=pad(), bias=True)
        if legacy_tcn:
            self.n2 = nn.BatchNorm1d(out_channels, eps=1e-3) if use_batch_norm else nn.Identity()
        else:
            self.n2 = _make_norm("layer" if not use_batch_norm else "batch", out_channels)
        self.drop2 = nn.Dropout(float(dropout_rate)) if dropout_rate else nn.Identity()

        # 1x1 residual projection if channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True) if in_channels != out_channels else None

        # Init similar to keras random_normal
        nn.init.normal_(self.conv1.weight, mean=0.0, std=conv_init_std)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=conv_init_std)
        nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=conv_init_std)
            nn.init.zeros_(self.downsample.bias)

    def _causal_pad(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) * self.dilation
        return F.pad(x, (pad, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C_in, T)
        y = self._causal_pad(x) if self.padding_mode == "causal" else x
        y = self.drop1(self.act(self.n1(self.conv1(y))))

        y = self._causal_pad(y) if self.padding_mode == "causal" else y
        y = self.drop2(self.act(self.n2(self.conv2(y))))

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
        use_batch_norm: bool = False,
        return_sequences: bool = False,
        legacy_tcn: bool = False,
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
                        legacy_tcn=legacy_tcn,
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
        conv_filters: int = 32, #16,
        kernel_size: int = 4,
        conv_stacks: int = 2, #1,
        conv_dilations: Iterable[int] = (1, 2, 4, 8), #(1,2,4),
        padding: str = "causal",
        use_skip_connections: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        interaction_regularization: float = 0.0,  # not used explicitly in PT
        use_batch_norm: bool = False,
        legacy_tcn: bool = False,
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
            dropout_rate=float(0.0), #no dropout for TCN blocks
            activation=activation,
            use_batch_norm=use_batch_norm,
            return_sequences=False,
        )

        if use_gnn:
            # Per-node and per-edge TCNs
            self.node_tcn = TCN1DPT(in_channels=F_node, legacy_tcn=legacy_tcn, **tcn_cfg)
            self.edge_tcn = TCN1DPT(in_channels=F_edge, legacy_tcn=legacy_tcn, **tcn_cfg)

            # Graph block and buffers
            self.spatial_gnn_block = CensNetConvPT(node_channels=latent_dim, edge_channels=latent_dim, activation="relu")
            lap, edge_lap, inc = self.spatial_gnn_block.preprocess(torch.tensor(adjacency_matrix))
            self.register_buffer("laplacian", lap.float())
            self.register_buffer("edge_laplacian", edge_lap.float())
            self.register_buffer("incidence", inc.float())

            final_in = (N * latent_dim) + (E * latent_dim)
        else:
            # Single TCN over flattened node features
            self.flat_tcn = TCN1DPT(in_channels=N * F_node, legacy_tcn=legacy_tcn, **tcn_cfg)
            final_in = conv_filters

        # Head MLP: Dense(2*latent) -> BN -> Dense(latent) -> BN -> Dense(latent)
        if legacy_tcn:
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
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.head = _encoder_head(final_in, latent_dim, dropout_rate)

    def forward(self, x: torch.Tensor, a: torch.Tensor, cluster_mode=False, return_backbone=False) -> torch.Tensor:
        """
        x: (B, W, N, NF)  a: (B, W, E, EF)  -> returns (B, latent_dim)
        return_backbone: return (projection, backbone output) instead of one of them
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
            # Non-GNN 
            x_flat = x.view(B, W, N * F_node)        # (B, W, N*NF)
            enc = self.flat_tcn(x_flat)              # (B, C)

        head_in = _normalize_backbone(enc)
        head_out = self.head(head_in)
        if return_backbone:
            return head_out, head_in
        return head_out if not cluster_mode else head_in


class AffineTransformedDistribution(TransformedDistribution):
    """
    A specific TransformedDistribution for Affine transforms that implements .mean.
    """
    def __init__(self, base_distribution, transform):
        super().__init__(base_distribution, transform)

    @property
    def mean(self):
        """
        Computes the mean of the transformed distribution.
        E[loc + scale * X] = loc + scale * E[X]
        """
        # The transform itself is callable and applies the affine transformation.
        return self.transforms[0](self.base_dist.mean)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log-likelihood per frame; frames with scale 0 (all-zero padding frames, see the validity mask of the
        decoders) contribute 0 instead of the NaN of the affine transform. Valid frames have scale 1."""
        valid = self.transforms[0].scale.squeeze(-1) > 0
        return torch.where(valid, self.base_dist.log_prob(value), torch.zeros((), device=value.device))


class ProbabilisticDecoderPT(nn.Module):
    """
    PyTorch translation of the ProbabilisticDecoder, including scaling transform.
    AMP-safe version: do distribution math in float32, sanitize NaNs/Infs.
    """
    def __init__(self, hidden_dim: int, data_dim: int):
        super().__init__()
        self.loc_projection = nn.Linear(in_features=hidden_dim, out_features=data_dim)

    def forward(self, hidden: torch.Tensor, validity_mask: torch.Tensor) -> AffineTransformedDistribution:
        B, T, _ = hidden.shape

        # Linear projection in float32 for numerical stability under AMP
        hidden_2d = hidden.reshape(B * T, -1).float()
        loc_params = self.loc_projection(hidden_2d).reshape(B, T, -1)

        # Sanitize to avoid NaN/Inf in distribution parameters
        loc_params = torch.nan_to_num(loc_params, nan=0.0, posinf=1e6, neginf=-1e6)

        # Build the distribution in float32 (disable autocast to avoid fp16 validator issues)
        with torch.amp.autocast(device_type=loc_params.device.type, enabled=False):
            loc32 = loc_params  # already float32
            scale32 = torch.ones_like(loc32)  # unit variance

            base_dist = torch.distributions.Normal(loc=loc32, scale=scale32, validate_args=False)
            independent_dist = torch.distributions.Independent(base_dist, 1)

            # Keep transform dtype consistent with distribution dtype
            scale_transform = validity_mask.unsqueeze(-1).to(dtype=loc32.dtype, device=loc32.device)
            transform = AffineTransform(loc=0.0, scale=scale_transform)

            final_dist = AffineTransformedDistribution(independent_dist, transform)

        return final_dist
    

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
        use_batch_norm: bool = False,
        legacy_tcn: bool = False,
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
            legacy_tcn=legacy_tcn,
        )
        # Probabilistic reconstruction head
        self.prob_decoder = ProbabilisticDecoderPT(hidden_dim=conv_filters, data_dim=self.data_dim)

        # Init linear layers (BN stats copied by transfer)
        for m in [self.fc0, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _stabilize_latent(self, g: torch.Tensor) -> torch.Tensor:
        # Training-only guard for extreme latent magnitudes
        g = g.float()
        #if self.training:
        # Per-sample RMS normalization
        rms = (g.pow(2).mean(dim=1, keepdim=True) + 1e-12).sqrt()  # (B,1), eps: finite gradient for a zero latent
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
        # validity mask
        if x.dim() == 4: # potentially unnecessary, making a note here
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
        hidden_seq = self.tcn(z_rep)                 # (B, W, conv_filters)
        return self.prob_decoder(hidden_seq, validity_mask)
   

def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if name == "leaky_relu": return nn.LeakyReLU(0.2)
    if name in {"linear", "identity", "none"}: return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}") # pragma: no cover


def sinusoidal_positional_encoding(max_len: int, d_model: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Compute positional encodings, as in https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf."""
    pe = torch.zeros(max_len, d_model, dtype=dtype, device=device)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    n_odd = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term)[:, :n_odd]
    return pe.unsqueeze(0)  # (1, max_len, d_model)


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
    ):
        super().__init__()
        self.key_dim = int(key_dim)
        self.max_pos = int(max_pos)
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

        # Embed (no padding mask: windows have a fixed length and zeros are valid values; a node with only zeros
        # would mask all keys and give NaN)
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
            y = layer(y)

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
        num_layers: int = 4,      # about the parameters of the TCN (32 filters, 2 stacks, dilations 1-8)
        num_heads: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.0,  # as the TCN (the augmentations regularize; dropout would differ between views)
        key_dim: int = 32,        # the width of the TCN; None: min(64, nodes * features)
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
            )
            final_in = self.key_dim

        # MLP head as in the TCN encoder (LayerNorm: BatchNorm statistics would be a cue across the batch)
        self.head = _encoder_head(final_in, self.latent_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor, cluster_mode=False, return_backbone=False) -> torch.Tensor:
        """
        Args:
            x: (B, W, N, NF) - node features over time
            a: (B, W, E, EF) - edge features over time
            return_backbone: return (projection, backbone output) instead of one of them
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

        # === Apply MLP head === (as the TCN encoder; no batch standardization of the output, which would be a cue
        # across the batch and differ between training and inference)
        head_in = _normalize_backbone(enc)
        if return_backbone:
            return self.head(head_in), head_in
        return self.head(head_in) if not cluster_mode else head_in


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
        ema_decay: float = 0.0,
    ):
        super(VectorQuantizerPT, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.beta = beta
        self.kmeans = kmeans_loss
        # ema_decay > 0: the codes are running centroids of their encoder outputs (EMA per training batch) and get no
        # gradient; 0: the codebook is trained by the codebook loss and by the reconstruction from the codes
        self.ema_decay = float(ema_decay)
        # Codes whose running usage share (ema_count) falls below restart_below are moved to an encoder output of the
        # batch (EMA mode, set per epoch by fit_VQVAE; 0 = off); n_restarts counts them
        self.restart_below = 0.0
        self.n_restarts = 0

        # Centred random start; fit_VQVAE replaces it by k-means centroids of the encoder outputs after the warm-up
        self.codebook = nn.Parameter(0.1 * torch.randn(self.embedding_dim, self.n_components))
        # Running mean of the squared distance to the assigned code per latent dimension (training batches); sets the
        # sharpness of the soft counts
        self.register_buffer("code_variance", torch.ones(()))
        # EMA mode: running share of the batch windows per code and running sum of their encoder outputs / batch size
        self.register_buffer("ema_count", torch.full((n_components,), 1.0 / n_components))
        self.register_buffer("ema_sum", self.codebook.detach().clone() / n_components)

    @torch.no_grad()
    def init_from_centroids(self, centroids: torch.Tensor, variance: float, usage: torch.Tensor = None) -> None:
        """Sets the codebook to centroids [K, D], the soft-count variance (per dimension) and the EMA statistics (usage:
        share of the windows per code, default uniform)."""
        self.codebook.data.copy_(centroids.T.to(self.codebook.dtype))
        self.code_variance.fill_(max(float(variance), 1e-8))
        share = (torch.full((self.n_components,), 1.0 / self.n_components) if usage is None else usage.float())
        self.ema_count.copy_(share.to(self.ema_count.device))
        self.ema_sum.copy_(self.codebook.detach().float() * self.ema_count[None, :])

    @torch.no_grad()
    def sync_ema(self) -> None:
        """Makes the EMA sums consistent with the current codebook (e.g. a loaded checkpoint)."""
        self.ema_sum.copy_(self.codebook.detach().float() * self.ema_count[None, :])

    @torch.no_grad()
    def _restart_dead_codes(self, flattened: torch.Tensor) -> None:
        """Moves the codes with a running usage share below restart_below to random encoder outputs of the batch
        (under DDP those of rank 0; the shares are averaged over the ranks, so all ranks find the same codes)."""
        dead = self.ema_count < self.restart_below
        if not dead.any():
            return
        pick = torch.randint(flattened.shape[0], (int(dead.sum()),), device=flattened.device)
        new = flattened[pick].detach().float().T.contiguous()  # [D, n_dead]
        if dist.is_available() and dist.is_initialized():  # pragma: no cover (DDP, not run in CI)
            dist.broadcast(new, src=0)
        share = 1.0 / self.n_components
        self.codebook.data[:, dead] = new.to(self.codebook.dtype)
        self.ema_count[dead] = share
        self.ema_sum[:, dead] = new * share
        self.n_restarts += int(dead.sum())

    @torch.no_grad()
    def _ema_update(self, flattened: torch.Tensor, encodings: torch.Tensor) -> None:
        """EMA of the share of the windows and of the sum of their encoder outputs per code (averaged over DDP ranks);
        codes without windows for a long time keep their position."""
        n = encodings.sum(0) / flattened.shape[0]
        s = flattened.detach().float().T @ encodings / flattened.shape[0]
        if dist.is_available() and dist.is_initialized():  # pragma: no cover (DDP, not run in CI)
            stats = torch.cat([n[None, :], s])
            dist.all_reduce(stats)
            stats /= dist.get_world_size()
            n, s = stats[0], stats[1:]
        self.ema_count.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * n)
        self.ema_sum.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * s)
        alive = self.ema_count > 1e-12
        self.codebook.data[:, alive] = (self.ema_sum[:, alive] / self.ema_count[alive]).to(self.codebook.dtype)

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
            losses['kmeans_loss'] = deepof.clustering.losses.compute_kmeans_loss_pt(flattened, self.kmeans)

        ema = self.ema_decay > 0
        if ema and self.training and self.restart_below > 0:
            self._restart_dead_codes(flattened)

        # Get encodings
        distances = self._distances(flattened)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_components).float()
        if self.training:
            with torch.no_grad():
                d_min = distances.gather(1, encoding_indices[:, None]).clamp_min(0).mean() / self.embedding_dim
                self.code_variance.mul_(0.99).add_(0.01 * d_min)
            if ema:
                self._ema_update(flattened, encodings)
        soft_counts = self._soft_counts(distances)

        # EMA mode: the decoder reconstructs from the codes without moving them (no gradient into the codebook)
        quantized = torch.matmul(encodings, (self.codebook.detach() if ema else self.codebook).T)
        quantized = quantized.reshape(input_shape)

        if return_losses:
            commitment_loss = self.beta * torch.mean(
                (quantized.detach() - x) ** 2
            )
            losses['vq_loss'] = commitment_loss if ema else commitment_loss + torch.mean((quantized - x.detach()) ** 2)
            return quantized, soft_counts, losses
        return quantized, soft_counts

    def _distances(self, flattened_inputs: torch.Tensor) -> torch.Tensor:
        """Squared L2 distances [B, K] to the codes."""
        return (
            torch.sum(flattened_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=0)
            - 2 * torch.matmul(flattened_inputs, self.codebook)
        )

    def _soft_counts(self, distances: torch.Tensor) -> torch.Tensor:
        """Posterior of an isotropic Gaussian mixture with equal weights around the codes, with the running
        per-dimension variance of the assignments: softmax(-d / (2 * variance))."""
        return F.softmax(-distances.clamp_min(0) / (2.0 * self.code_variance.clamp_min(1e-8)), dim=1)
    

def init_encoder_decoder(
    *,
    encoder_type: str,
    input_shape,
    edge_feature_shape,
    adjacency_matrix,
    latent_dim: int,
    use_gnn: bool,
    interaction_regularization: float,
    n_nodes: int,
    n_features_per_node: int,
    time_steps: int,
    tcn_conv_filters: int = 32,
    tcn_kernel_size: int = 4,
    tcn_conv_stacks: int = 2,
    tcn_conv_dilations: Iterable[int] = (1, 2, 4, 8), 
    legacy_tcn: bool = False,
):
    """
    Initialize encoder/decoder modules based on encoder type.

    Args:
        encoder_type: can be either "recurrent", "TCN", or "transformer".
        input_shape, edge_feature_shape, adjacency_matrix: forwarded to encoders.
        latent_dim, use_gnn, interaction_regularization: forwarded to encoders.
        n_nodes, n_features_per_node, time_steps: used to compute decoder output shape.

    Returns:
        (encoder, decoder)
    """
    decoder_output_features = n_nodes * n_features_per_node

    if encoder_type.lower() == "recurrent":
        encoder = RecurrentEncoderPT(
            input_shape=input_shape,
            edge_feature_shape=edge_feature_shape,
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
        )
        decoder = RecurrentDecoderPT(
            output_shape=(time_steps, decoder_output_features),
            latent_dim=latent_dim,
        )

    elif encoder_type.lower() == "tcn":
        encoder = TCNEncoderPT(
            input_shape=input_shape,
            edge_feature_shape=edge_feature_shape,
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
            conv_filters=tcn_conv_filters,
            kernel_size=tcn_kernel_size,
            conv_stacks=tcn_conv_stacks,
            conv_dilations=tcn_conv_dilations,
            legacy_tcn=legacy_tcn,
        )
        decoder = TCNDecoderPT(
            output_shape=(time_steps, decoder_output_features),
            latent_dim=latent_dim,
            legacy_tcn=legacy_tcn,
        )

    elif encoder_type.lower() == "transformer":
        encoder = TFMEncoderPT(
            input_shape=input_shape,
            edge_feature_shape=edge_feature_shape,
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            # interaction_regularization=interaction_regularization,
        )
        decoder = TFMDecoderPT(
            output_shape=(time_steps, decoder_output_features),
            latent_dim=latent_dim,
            num_layers=2,
            num_heads=8,
            dff=128,
            dropout_rate=0.2,
        )

    else:  # pragma: no cover
        raise NotImplementedError(
            'invalid encoder type, try "recurrent", "TCN" or "transformer"'
        )

    return encoder, decoder


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
        beta: float = 0.25,
        ema_decay: float = 0.0,
        tcn_conv_filters: int = 32,
        tcn_kernel_size: int = 4,
        tcn_conv_stacks: int = 2,
        tcn_conv_dilations: Iterable[int] = (1, 2, 4, 8),
        legacy_tcn: bool = False,
    ):
        """Initialize a VQ-VAE model.

        Args:
            input_shape (tuple): Shape of the input (time_steps, node_features).
            edge_feature_shape (tuple): Shape of edge features (time_steps, edge_features).
            adjacency_matrix (np.ndarray): Adjacency matrix for GNN.
            latent_dim (int): Dimensionality of the latent space.
            n_components (int): Number of embeddings (clusters) in the codebook.
            beta (float): Weight of the commitment loss (pulls the encoder outputs to their codes).
            ema_decay (float): > 0: codes as running centroids of their encoder outputs (EMA per batch), see
                VectorQuantizerPT; 0: codebook trained by gradient.
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
        self.encoder, self.decoder = init_encoder_decoder(
            encoder_type=encoder_type,
            input_shape=input_shape,
            edge_feature_shape=edge_feature_shape,
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
            n_nodes=n_nodes,
            n_features_per_node=n_features_per_node,
            time_steps=time_steps,
            tcn_conv_filters=tcn_conv_filters,
            tcn_kernel_size=tcn_kernel_size,
            tcn_conv_stacks=tcn_conv_stacks,
            tcn_conv_dilations=tcn_conv_dilations,
            legacy_tcn=legacy_tcn,
        )
        
        # Initialize Vector Quantizer
        self.vq_layer = VectorQuantizerPT(
            n_components=n_components,
            embedding_dim=latent_dim,
            beta=beta,
            kmeans_loss=kmeans_loss,
            ema_decay=ema_decay,
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
        if self.encoder_type.lower() == "transformer":
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
    

class ClusterControlPT(nn.Module):
    """
    Calculates clustering metrics. This is a pass-through layer for the main
    latent vector `z`, returning it unmodified alongside a dictionary of metrics.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, z: torch.Tensor, z_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates metrics and passes the latent vector `z` through.

        Args:
            z: The latent vector (batch_size, latent_dim).
            z_cat: Cluster probabilities (batch_size, n_components).

        Returns:
            A tuple containing the unmodified `z` and a dictionary of metrics.
        """
        confidence, hard_groups = torch.max(z_cat, dim=1)
        
        # Calculate the number of unique clusters populated in the batch
        num_populated = torch.unique(hard_groups).numel()
        
        metrics = {
            "number_of_populated_clusters": torch.tensor(
                float(num_populated), device=z.device
            ),
            "confidence_in_selected_cluster": torch.mean(confidence),
        }
        
        return z, metrics
    

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
        self.cluster_control = ClusterControlPT()

        # --- Focus layer ---
        self.lens = nn.Linear(self.latent_dim,self.mixture_dim)


    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input into mean and log-variance of the latent distribution."""
        z_mean = self.encoder_mean(x)
        # Log-variance bounded smoothly to (-8, 8): beyond a hard clamp the gradient is zero, and pretraining (KL weight
        # ~0) drives most dimensions below the bound, where they would stay frozen in the main phase
        z_log_var = 8.0 * torch.tanh(self.encoder_log_var(x) / 8.0)
        return z_mean, z_log_var

    def _reparameterize(
        self, mean: torch.Tensor, log_var: torch.Tensor, epsilon: torch.Tensor = None
    ) -> torch.Tensor:
        """Reparameterization z = mean + exp(log_var / 2) * epsilon."""
        scale = torch.exp(0.5 * log_var)
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
        
        z_mean, z_log_var = self._encode(x)
        z_sample = self._reparameterize(z_mean, z_log_var, epsilon)
        z_for_downstream = z_sample if self.training else z_mean

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
            kmeans_loss = deepof.clustering.losses.compute_kmeans_loss_pt(z_final, weight=self.kmeans_weight)

        return (z_final, z_cat, metrics["number_of_populated_clusters"], metrics["confidence_in_selected_cluster"], kmeans_loss, h_mean, h_log_var, z_for_gaussian)


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
        tcn_conv_filters: int = 32,
        tcn_kernel_size: int = 4,
        tcn_conv_stacks: int = 2,
        tcn_conv_dilations: Iterable[int] = (1, 2, 4, 8), 
        legacy_tcn: bool = False,
    ):
        super().__init__()
        
        time_steps, n_nodes, n_features_per_node = input_shape
        self.input_n_nodes = n_nodes
        self.input_n_features_per_node = n_features_per_node
        self.window_size = time_steps #important for modal usage later
        self.lens_enabled=lens_enabled

        self.encoder, self.decoder = init_encoder_decoder(
            encoder_type=encoder_type,
            input_shape=input_shape,
            edge_feature_shape=edge_feature_shape,
            adjacency_matrix=adjacency_matrix,
            latent_dim=latent_dim,
            use_gnn=use_gnn,
            interaction_regularization=interaction_regularization,
            n_nodes=n_nodes,
            n_features_per_node=n_features_per_node,
            time_steps=time_steps,
            tcn_conv_filters=tcn_conv_filters,
            tcn_kernel_size=tcn_kernel_size,
            tcn_conv_stacks=tcn_conv_stacks,
            tcn_conv_dilations=tcn_conv_dilations,
            legacy_tcn=legacy_tcn,
        )        

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

    def initialize_gmm_from_data(self, data_loader, n_samples=50000):
        """
        Runs the autoencoder part of the model over the data to get embeddings,
        then fits a scikit-learn GMM to initialize the latent space. Uses the first n_samples windows of the loader,
        a random sample across videos with the globally shuffled training data.
        """
        print("Initializing GMM from data embeddings...")
        self.eval()
        
        all_embeddings, posterior_vars = [], []
        samples_gathered = 0
        dev = next(self.parameters()).device
        with torch.no_grad():
            for x, a, *_ in data_loader:
                x, a = x.to(dev), a.to(dev)
                enc = self.encoder(x, a)
                z_mean, z_log_var = self.latent_space._encode(enc)
                all_embeddings.append(z_mean.cpu())
                posterior_vars.append(z_log_var.exp().cpu())
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
        # Log-variances of the posterior samples the KL compares with the GMM: spread of the means plus the mean
        # posterior variance (with the means only they can be far below the posterior noise, e.g. 1e-4 vs ~1, and the
        # KL explodes); within the clamp of the loss (VadeLoss.gmm_logvar_clamp), else they get no gradient
        posterior_var = torch.cat(posterior_vars, dim=0)[: all_embeddings.shape[0]].mean(dim=0).numpy()
        self.latent_space.gmm_log_vars.data = torch.from_numpy(
            np.clip(np.log(gmm.covariances_ + posterior_var[None, :]), -8.0, 8.0)).float().to(dev)

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
        return self.latent_space(self.encoder(x, a))[0]

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
        return self.latent_space(self.encoder(x, a))[1]


class ContrastivePT(nn.Module):
    """
    PyTorch port of the TF Contrastive model.

    Inputs:
      x: (B, T, N, F)
      a: (B, T, E, F_edge)

    Behavior:
      - Builds an encoder for sequences of length window_len (default T//2).
      - forward(x_win, a_win) returns embeddings (B, D) for a window of that length (the backbone output with
        cluster_mode=True).
      - The model has no loss of its own: views and loss are built in training.step_contrastive_distill.
      - social_nodes > 0 (per-animal datasets with social context, get_graph_dataset(per_animal=True)): the last
        social_nodes nodes of x are the context nodes of the social_partners other animals (partner-major). The encoder
        only sees the body parts of the animal itself; the context nodes of every partner go through one shared TCN,
        are summed over the partners (any number and order) and mapped to social_dim values, which are appended to the
        backbone output: [own behavior | social context]. The social part is not normalized per sample, so that a
        partner far away (inputs near 0) gives a nearly constant vector. The projection head works on both parts.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],         # (T, N, F)
        edge_feature_shape: Tuple[int, int, int],  # (T, E, F_edge)
        adjacency_matrix,
        window_len: int = None,
        latent_dim: int = 8,
        encoder_type: str = "TCN",
        use_gnn: bool = True,
        temperature: float = 0.1,
        similarity_function: str = "cosine",
        loss_function: str = "nce",
        beta: float = 0.1,
        tau: float = 0.1,
        interaction_regularization: float = 0.0,
        tcn_conv_filters: int = 32,
        tcn_kernel_size: int = 4,
        tcn_conv_stacks: int = 2,
        tcn_conv_dilations: Iterable[int] = (1, 2, 4, 8), 
        legacy_tcn: bool = False,
        cluster_mode: bool = False,
        social_nodes: int = 0,
        social_partners: int = 1,
        social_dim: int = 32,
    ):
        super().__init__()

        T, N, F_in = input_shape
        Te, E, Fe = edge_feature_shape
        self.social_nodes, self.social_partners, self.social_dim = int(social_nodes), int(social_partners), int(social_dim)
        if self.social_nodes and (self.social_nodes % self.social_partners or not 0 < self.social_nodes < N):
            raise ValueError(f"social_nodes={social_nodes} must be a multiple of social_partners={social_partners} "
                             f"and smaller than the number of nodes ({N})")
        N = N - self.social_nodes  # body parts of the animal itself
        
        if T != Te: # pragma: no cover
            raise ValueError(f"Node and edge time dims must match: T={T}, Te={Te}")
        
        self.cluster_mode=cluster_mode

        self.full_time_steps = T
        if window_len is None:
            window_len = T // 2
        self.window_size = window_len # To enable length shift augmentation
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

        if encoder_type.lower() == "recurrent":
            self.encoder = RecurrentEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
            )
        elif encoder_type.lower() == "tcn":
            self.encoder = TCNEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
                interaction_regularization=interaction_regularization,
                conv_filters=tcn_conv_filters,
                kernel_size=tcn_kernel_size,
                conv_stacks=tcn_conv_stacks,
                conv_dilations=tcn_conv_dilations,
                legacy_tcn=legacy_tcn,
            )
        elif encoder_type.lower() == "transformer":
            self.encoder = TFMEncoderPT(
                input_shape=(self.window_size, N, F_in),
                edge_feature_shape=(self.window_size, E, Fe),
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=use_gnn,
            )
        else: # pragma: no cover
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        if self.social_nodes:
            per_partner = self.social_nodes // self.social_partners
            # Without normalization layers (legacy_tcn with use_batch_norm=False): a LayerNorm would scale the near-zero
            # inputs of a partner far away up to unit size, so that its movements there would shape the embedding
            self.social_tcn = TCN1DPT(
                in_channels=per_partner * F_in, conv_filters=tcn_conv_filters, kernel_size=tcn_kernel_size,
                conv_stacks=tcn_conv_stacks, conv_dilations=tuple(tcn_conv_dilations), return_sequences=False,
                use_batch_norm=False, legacy_tcn=True,
            )
            for module in self.social_tcn.modules():  # without normalization the small default init would mute it
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            self.social_out = nn.Linear(tcn_conv_filters, self.social_dim)
            # One projection head on [own | social] instead of the encoder's own one
            self.head = _encoder_head(self.encoder.head[0].in_features + self.social_dim, latent_dim)
            self.encoder.head = nn.Identity()

        # Debug cache
        self._last_debug: Dict[str, Any] = {}

    def split_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """(body parts of the animal, social context nodes or None) of node features (B, T, N, F)."""
        if not self.social_nodes:
            return x, None
        n = x.shape[2] - self.social_nodes
        return x[:, :, :n], x[:, :, n:]

    def forward(self, x: torch.Tensor, a: torch.Tensor, return_backbone: bool = False) -> torch.Tensor:
        """
        Encode a half-window:
          x: (B, T_half, N, F), a: (B, T_half, E, Fe) -> (B, D)
        With return_backbone, returns (projection, backbone output) (used by the regression teacher).
        """
        if not self.social_nodes:
            return self.encoder(x, a, self.cluster_mode, return_backbone=return_backbone)
        x_self, x_social = self.split_nodes(x)
        _, own = self.encoder(x_self, a, False, return_backbone=True)
        B, T, _, F_in = x_social.shape
        partners = x_social.reshape(B, T, self.social_partners, -1).permute(0, 2, 1, 3).reshape(
            B * self.social_partners, T, -1)
        social = self.social_out(self.social_tcn(partners.float()).view(B, self.social_partners, -1).sum(dim=1))
        backbone = torch.cat([own, social.to(own.dtype)], dim=1)
        projection = self.head(backbone)
        if return_backbone:
            return projection, backbone
        return backbone if self.cluster_mode else projection
 
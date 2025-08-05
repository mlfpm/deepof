"""Translation of the CensNetConv module (from spektral,pip install spektral version 1.3.1) into Pytorch

Original module can be imported with 

from spektral.layers import CensNetConv

if the library "spektral" is installed. Based on the paper 

"Co-embedding of Nodes and Edges with Graph Neural Networks" by Xiaodong Jiang, Ronghang Zhu, Pengsheng Ji, and Sheng Li 2020
arXiv:2010.13242v1 [cs.LG] 25 Oct 2020"""

# @author NoCreativeIdeForGoodUsername
# encoding: utf-8
# module deepof.clustering

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

###############
#  MAIN  MODULE
###############

class CensNetConvPT(nn.Module):
    r"""
    A PyTorch implementation of the CensNet convolutional layer.
    """
    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        activation: str = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.use_bias = use_bias

        # Activation function (none is default in tensorflow version)
        if activation == 'relu':
            self.activation = F.relu
        elif activation is None:
            self.activation = lambda x: x
        else:
            # You can add more activations here if needed
            raise NotImplementedError(f"Activation '{activation}' not implemented.")
            
        # Weights are difined as None here. They will be created in the first
        # forward pass, mimicking Keras's `build` method. This makes the layer
        # input-shape-agnostic until it's first used.
        self.node_kernel = None
        self.edge_kernel = None
        self.node_weights = None
        self.edge_weights = None
        self.node_bias = None
        self.edge_bias = None

    def _build(self, node_features_shape, edge_features_shape):
        """
        Mimics Keras's build method to initialize weights based on input shapes.
        """
        num_input_node_features = node_features_shape[-1]
        num_input_edge_features = edge_features_shape[-1]
        
        # Using Glorot/Xavier uniform initialization, the default in Keras
        # and a good standard choice in PyTorch.
        
        self.node_kernel = nn.Parameter(torch.empty(num_input_node_features, self.node_channels))
        nn.init.xavier_uniform_(self.node_kernel)

        self.edge_kernel = nn.Parameter(torch.empty(num_input_edge_features, self.edge_channels))
        nn.init.xavier_uniform_(self.edge_kernel)
        
        # These are P_n and P_e in the paper.
        self.node_weights = nn.Parameter(torch.empty(num_input_node_features, 1))
        nn.init.xavier_uniform_(self.node_weights)
        
        self.edge_weights = nn.Parameter(torch.empty(num_input_edge_features, 1))
        nn.init.xavier_uniform_(self.edge_weights)

        if self.use_bias:
            self.node_bias = nn.Parameter(torch.empty(self.node_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.node_kernel)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.node_bias, -bound, bound) # Keras default bias init
            
            self.edge_bias = nn.Parameter(torch.empty(self.edge_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.edge_kernel)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.edge_bias, -bound, bound) # Keras default bias init
            
    def _propagate_nodes(self, inputs):
        """Performs the node feature propagation step."""
        node_features, (laplacian, _, incidence), edge_features = inputs

        # weighted_edge_features = diag(I^T * P_e)
        weighted_edge_features = modal_dot_pt(edge_features, self.edge_weights)
        weighted_edge_features = weighted_edge_features.squeeze(-1)
        # tf.linalg.diag is batch-wise, torch.diag is not. Need to handle batch.
        if weighted_edge_features.ndim == 2: # Batch dimension exists
            weighted_edge_features_diag = torch.diag_embed(weighted_edge_features)
        else: # No batch dimension
            weighted_edge_features_diag = torch.diag(weighted_edge_features)

        # node_adjacency = (I * diag(E * P_e) * I^T) * L_tilde
        # First part: I * diag(...)
        temp = modal_dot_pt(incidence, weighted_edge_features_diag)
        # Second part: (I * diag(...)) * I^T
        weighted_incidence = modal_dot_pt(temp, incidence, transpose_b=True)
        
        node_adjacency = weighted_incidence * laplacian
        
        # H_n' = (node_adjacency @ H_n) @ W_n
        output = modal_dot_pt(node_adjacency, node_features)
        output = modal_dot_pt(output, self.node_kernel)

        # Apply bias and activation
        if self.use_bias:
            output = output + self.node_bias
        return self.activation(output)

    def _propagate_edges(self, inputs):
        """Performs the edge feature propagation step."""
        node_features, (laplacian, line_laplacian, incidence), edge_features = inputs
        
        # weighted_node_features = diag(H_n * P_n)
        weighted_node_features = modal_dot_pt(node_features, self.node_weights)
        weighted_node_features = weighted_node_features.squeeze(-1)
        if weighted_node_features.ndim == 2: # Batch
             weighted_node_features_diag = torch.diag_embed(weighted_node_features)
        else: # No Batch
            weighted_node_features_diag = torch.diag(weighted_node_features)
        
        # edge_adjacency = (I^T * diag(H_n * P_n) * I) * L_line_tilde
        # First part: I^T * diag(...)
        temp = modal_dot_pt(incidence, weighted_node_features_diag, transpose_a=True)
        # Second part: (I^T * diag(...)) * I
        weighted_line_graph = modal_dot_pt(temp, incidence)
        
        edge_adjacency = weighted_line_graph * line_laplacian
        
        # H_e' = (edge_adjacency @ H_e) @ W_e
        output = modal_dot_pt(edge_adjacency, edge_features)
        output = modal_dot_pt(output, self.edge_kernel)
        
        # Apply bias and activation
        if self.use_bias:
            output = output + self.edge_bias
        return self.activation(output)

    def forward(self, inputs):
        """
        The forward pass of the CensNet layer.
        
        Args:
            inputs: A tuple containing:
                - node_features (torch.Tensor): Shape [B, N, F_n]
                - graph_ops (tuple): (laplacian, edge_laplacian, incidence)
                - edge_features (torch.Tensor): Shape [B, E, F_e]
        """
        node_features, graph_ops, edge_features = inputs

        # Build weights on first pass
        if self.node_kernel is None:
            # Move parameters to the same device as input tensors
            self._build(node_features.shape, edge_features.shape)
            self.to(node_features.device)

        propagated_nodes = self._propagate_nodes((node_features, graph_ops, edge_features))
        propagated_edges = self._propagate_edges((node_features, graph_ops, edge_features))
        
        return propagated_nodes, propagated_edges

    @staticmethod
    def preprocess(adjacency: torch.Tensor):
        """
        Computes the graph operators needed for the forward pass.
        
        Args:
            adjacency (torch.Tensor): Adjacency matrix, shape [B, N, N] or [N, N].

        Returns:
            A tuple: (laplacian, edge_laplacian, incidence)
        """
        laplacian = gcn_filter_pt(adjacency)
        incidence = incidence_matrix_pt(adjacency)
        edge_laplacian = gcn_filter_pt(line_graph_pt(incidence))

        return laplacian, edge_laplacian, incidence


###############
# PREPROCESSING
###############

def degree_power_pt(A: torch.Tensor, k: float) -> torch.Tensor:
    """
    Computes D^k from the given adjacency matrix A.

    Args:
        A (torch.Tensor): A dense adjacency matrix of shape [N, N].
        k (float): The exponent.

    Returns:
        torch.Tensor: A diagonal matrix D^k of shape [N, N].
    """
    # Sum over columns to get degrees
    degrees = torch.sum(A, dim=1)
    # Handle potential 0 degrees to avoid inf/nan
    degrees[degrees == 0] = 1.0
    
    powered_degrees = torch.pow(degrees, k)
    
    # Create a diagonal matrix from the computed degrees
    D_k = torch.diag(powered_degrees)
    
    return D_k

def normalized_adjacency_pt(A: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    """
    Normalizes the given adjacency matrix.

    Args:
        A (torch.Tensor): A dense adjacency matrix of shape [N, N].
        symmetric (bool): If True, computes D^-0.5 * A * D^-0.5.
                          Otherwise, computes D^-1 * A.

    Returns:
        torch.Tensor: The normalized adjacency matrix.
    """
    if symmetric:
        # D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = degree_power_pt(A, -0.5)
        return D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # D^(-1) * A
        D_inv = degree_power_pt(A, -1.0)
        return D_inv @ A

def gcn_filter_pt(A: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    """
    Computes the GCN filter (I + D^-0.5 * A * D^-0.5).

    Args:
        A (torch.Tensor): A dense adjacency matrix or a batch of them.
                          Shape [N, N] or [B, N, N].
        symmetric (bool): Whether to use symmetric normalization.

    Returns:
        torch.Tensor: The GCN-normalized adjacency matrix.
    """
    if A.ndim == 2:
        # Single graph case
        # Add self-loops: A_hat = A + I
        A_hat = A + torch.eye(A.shape[0], device=A.device)
        # Normalize
        return normalized_adjacency_pt(A_hat, symmetric=symmetric)
    elif A.ndim == 3:
        # Batch of graphs case
        A_hat_list = []
        for i in range(A.shape[0]):
            A_i = A[i]
            # Add self-loops: A_hat = A + I
            A_hat_i = A_i + torch.eye(A_i.shape[0], device=A.device)
            # Normalize and append
            A_hat_list.append(normalized_adjacency_pt(A_hat_i, symmetric=symmetric))
        return torch.stack(A_hat_list)
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions.")
    

def line_graph_pt(incidence: torch.Tensor) -> torch.Tensor:
    """
    Creates the line graph adjacency matrix.

    Args:
        incidence (torch.Tensor): The incidence matrix of shape [..., N, E].

    Returns:
        torch.Tensor: The line graph adjacency matrix of shape [..., E, E].
    """
    # Transpose the last two dimensions: [..., N, E] -> [..., E, N]
    incidence_t = incidence.transpose(-2, -1)
    
    # Compute L = I^T * I
    line_adj = incidence_t @ incidence
    
    # The line graph has an edge between e_i and e_j if they share a node.
    # The diagonal of L gives the degree of each node (which is always 2 for
    # an edge). L_ij is 1 if they share a node, 0 otherwise.
    # The formula L - 2*I gives the desired adjacency matrix.
    identity = torch.eye(line_adj.shape[-1], device=line_adj.device)
    return line_adj - 2 * identity


def _triangular_adjacency_pt(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Gets the upper triangle of the adjacency matrix.
    
    Args:
        adjacency (torch.Tensor): Full adjacency matrix, shape [..., N, N].

    Returns:
        torch.Tensor: Upper triangle of the adjacency matrix.
    """
    # torch.triu is the direct equivalent of tf.linalg.band_part(..., 0, -1)
    return torch.triu(adjacency)


def _incidence_matrix_single_pt(triangular_adjacency: torch.Tensor, num_edges: int) -> torch.Tensor:
    """
    Creates the incidence matrix for a single graph.

    Args:
        triangular_adjacency (torch.Tensor): Upper triangular adjacency matrix [N, N].
        num_edges (int): The number of edge columns for the output matrix.

    Returns:
        torch.Tensor: The computed incidence matrix [N, n_edges].
    """
    # Find the coordinates of the edges (non-zero elements)
    connected_node_indices = triangular_adjacency.nonzero(as_tuple=False)
    
    num_nodes = triangular_adjacency.shape[0]
    
    # Create an empty incidence matrix
    output = torch.zeros((num_nodes, num_edges), dtype=torch.float32, device=triangular_adjacency.device)
    
    # If there are no edges, return the zero matrix
    if connected_node_indices.shape[0] == 0:
        return output
        
    # Get row indices (the nodes involved in each edge)
    # connected_node_indices has shape [num_actual_edges, 2]
    rows_node_1 = connected_node_indices[:, 0]
    rows_node_2 = connected_node_indices[:, 1]
    
    # Get column indices (a unique index for each edge)
    cols_edge = torch.arange(connected_node_indices.shape[0], device=triangular_adjacency.device)

    # Populate the incidence matrix. For each edge (column), set a 1 at the
    # two rows corresponding to the nodes it connects.
    output[rows_node_1, cols_edge] = 1
    output[rows_node_2, cols_edge] = 1
    
    return output


def incidence_matrix_pt(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Creates incidence matrices for a batch of graphs.

    Args:
        adjacency (torch.Tensor): Adjacency matrix, shape [N, N] or [B, N, N].

    Returns:
        torch.Tensor: Incidence matrices, shape [N, E] or [B, N, E].
    """
    # Ensure input is a tensor and handle batch dimension
    added_batch_dim = False
    if adjacency.ndim == 2:
        adjacency = adjacency.unsqueeze(0)
        added_batch_dim = True
        
    # Get upper triangular part to count unique edges
    adjacency_upper = _triangular_adjacency_pt(adjacency)
    
    # Count number of edges in each graph to find the max for padding
    num_edges_per_graph = torch.count_nonzero(adjacency_upper, dim=(1, 2))
    max_num_edges = int(torch.max(num_edges_per_graph).item())
    
    # Compute incidence matrix for each graph in the batch
    incidence_matrices = []
    for adj_single in adjacency_upper:
        inc = _incidence_matrix_single_pt(adj_single, num_edges=max_num_edges)
        incidence_matrices.append(inc)
    
    output = torch.stack(incidence_matrices)
    
    # Remove the batch dimension if it wasn't there originally
    if added_batch_dim:
        output = output.squeeze(0)
        
    return output


###############
# OPS UTILITIES
###############

def transpose_pt(a: torch.Tensor, perm=None) -> torch.Tensor:
    """
    Transposes a tensor, handling sparsity.
    """
    if perm is None:
        # Default behavior for 2D tensor
        perm = (1, 0)

    if a.is_sparse:
        # NOTE: torch.sparse.transpose is deprecated. The standard transpose works.
        # It only supports 2D sparse tensors.
        if a.ndim != 2:
            raise NotImplementedError("PyTorch sparse transpose only supports 2D tensors.")
        return a.transpose(perm[0], perm[1])
    else:
        return a.permute(*perm)

def reshape_pt(a: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Reshapes a tensor, handling sparsity.
    """
    if a.is_sparse:
        # Sparse reshape is not directly supported in PyTorch in the same way.
        # This is a major difference from TF. For CensNet's usage, this is
        # unlikely to be called on a sparse tensor. If it is, this will need
        # a more complex implementation (coo -> values/indices -> reshape -> new coo).
        raise NotImplementedError("Sparse reshape is not directly supported in PyTorch.")
    else:
        return torch.reshape(a, shape)

def dot_pt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes a @ b, handling sparsity.
    torch.matmul is a powerful replacement that handles most cases:
    - dense @ dense (rank 2 and 3/batch)
    - sparse @ dense
    - dense @ sparse
    """
    return torch.matmul(a, b)

def mixed_mode_dot_pt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes einsum('ij,bjk->bik', a, b), handling sparsity.
    PyTorch equivalent of ops.mixed_mode_dot.
    """
    # torch.einsum is the most direct and often most efficient way to do this.
    # It works correctly even if 'a' is a sparse tensor.
    return torch.einsum('ij,bjk->bik', a, b)


def modal_dot_pt(a: torch.Tensor, b: torch.Tensor, transpose_a=False, transpose_b=False):
    """
    Computes matrix multiplication, handling different data modes automatically.
    PyTorch equivalent of ops.modal_dot.
    """
    a_ndim = a.ndim
    b_ndim = b.ndim
    
    if transpose_a:
        perm_a = tuple(range(a_ndim - 2)) + (a_ndim - 1, a_ndim - 2)
        a = transpose_pt(a, perm_a)

    if transpose_b:
        perm_b = tuple(range(b_ndim - 2)) + (b_ndim - 1, b_ndim - 2)
        b = transpose_pt(b, perm_b)
        
    if a_ndim == b_ndim:
        return dot_pt(a, b)
    elif a_ndim == 2 and b_ndim == 3:
        return mixed_mode_dot_pt(a, b)
    elif a_ndim == 3 and b_ndim == 2:
        # We can use einsum here as well for a clean implementation.
        return torch.einsum('bij,jk->bik', a, b)
    else:
        raise ValueError(f"Unsupported combination of ranks: {a_ndim} and {b_ndim}")
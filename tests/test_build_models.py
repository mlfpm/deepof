# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import networkx as nx
import numpy as np

import deepof.model_utils
import deepof.models


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_VaDE_build(use_gnn, encoder_type):
    vade = deepof.models.VaDE(
        input_shape=(1000, 15, 33),
        edge_feature_shape=(1000, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
        encoder_type=encoder_type,
        n_components=10,
        latent_dim=8,
        batch_size=64,
    )
    vade.build([(1000, 15, 33), (1000, 15, 11)])
    vade.compile()


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_VQVAE_build(use_gnn, encoder_type):
    vqvae = deepof.models.VQVAE(
        input_shape=(1000, 15, 33),
        edge_feature_shape=(1000, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
        encoder_type=encoder_type,
        n_components=10,
        latent_dim=8,
    )
    vqvae.build([(1000, 15, 33), (1000, 15, 11)])
    vqvae.compile()


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_Contrastive_build(use_gnn, encoder_type):
    contrasts = deepof.models.Contrastive(
        input_shape=(1000, 15, 33),
        edge_feature_shape=(1000, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
        encoder_type=encoder_type,
        latent_dim=8,
    )
    contrasts.build([(1000, 7, 33), (1000, 7, 11)])
    contrasts.compile()

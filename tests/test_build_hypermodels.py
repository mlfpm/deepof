# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.hypermodels. Checks that all hyperparameter
tuning models are building properly in all possible configurations

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from keras_tuner import HyperParameters

import networkx as nx

import deepof.hypermodels


@settings(deadline=None, max_examples=10)
@given(
    use_gnn=st.booleans(),
)
def test_VaDE_hypermodel_build(use_gnn):
    deepof.hypermodels.VaDE(
        latent_dim=8,
        input_shape=(100, 15, 11),
        edge_feature_shape=(100, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        n_components=10,
        batch_size=64,
        use_gnn=use_gnn,
    ).build(hp=HyperParameters())


@settings(deadline=None, max_examples=10)
@given(
    use_gnn=st.booleans(),
)
def test_VQVAE_hypermodel_build(use_gnn):
    deepof.hypermodels.VQVAE(
        latent_dim=8,
        input_shape=(100, 15, 11),
        edge_feature_shape=(100, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        n_components=10,
        use_gnn=use_gnn,
    ).build(hp=HyperParameters())


@settings(deadline=None, max_examples=10)
@given(
    use_gnn=st.booleans(),
)
def test_Contrastive_hypermodel_build(use_gnn):
    deepof.hypermodels.Contrastive(
        latent_dim=8,
        input_shape=(100, 15, 11),
        edge_feature_shape=(100, 15, 11),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
    ).build(hp=HyperParameters())

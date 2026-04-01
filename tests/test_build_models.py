# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

import torch
import numpy as np
from types import SimpleNamespace

import deepof.model_utils
import deepof.models
import deepof.clustering.models_new


###################
# VADE TESTS
###################


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    mode=st.sampled_from(["main","pretrain"]),
    latent_dim=st.sampled_from([4,8]),
    n_components=st.sampled_from([4,8]),
)
def test_vade_backward_step(use_gnn,encoder_type,mode,latent_dim,n_components):
    device = torch.device("cpu")

    B, T, N, F_node = 2, 24, 11, 3
    E, F_edge = 11, 1

    adjacency = np.eye(N, dtype=np.float32)

    model = deepof.clustering.models_new.VaDEPT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        n_components=n_components,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        interaction_regularization=0.0,
        kmeans_loss=1.0,
    ).to(device, non_blocking=True)
    step_fn = deepof.clustering.models_new.step_vade

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg(
        latent_dim = latent_dim,
        n_components = n_components,
        kmeans_loss=1.0,
    )
    vade_cfg = deepof.clustering.model_utils_new.VaDECfg(
        reg_cat_clusters=1.0,
        tf_cluster_weight=1.0,
        nonempty_weight=1.0,
        temporal_cohesion_weight=1.0,
        reg_scatter_weight=1.0,
        repel_weight=1.0,
        kmeans_loss_pretrain = 1.0,
        repel_weight_pretrain = 0.5,
        repel_length_scale_pretrain = 0.5,
        nonempty_weight_pretrain = 2e-2,
        nonempty_p_pretrain = 2.0,
        nonempty_floor_percent_pretrain = 0.05,
    )
    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg(
    )

    criterion = deepof.clustering.models_new.VadeLoss(
        common_cfg=common_cfg,
        vade_cfg=vade_cfg,
        teacher_cfg=teacher_cfg,
    ).to(device)
    criterion.set_mode(mode)

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    result = step_fn(model, (x, a, idx), SimpleNamespace(criterion=criterion, apply_distill=False))

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    assert torch.isfinite(result.loss)
    losses = ['total_loss','reconstruct_loss','kl_div', 'cat_clust_loss', 'kmeans_loss', 'activity_l1', 'distill_loss','tf_clust_loss', 'nonempty_loss', 'temporal_loss', 'scatter_loss','repel_loss']
    main_train_losses=['cat_clust_loss', 'distill_loss','tf_clust_loss', 'temporal_loss', 'scatter_loss']
    # Check if all partial losses are present
    assert all(loss in result.logs for loss in losses)
    # losses summed up need to be equal to total loss
    assert np.isclose(
        result.logs["total_loss"],
        sum(v for k, v in result.logs.items() if k != "total_loss"),
        atol=1e-5,
        rtol=1e-5,
    )
    # Make sure some losses do not trigger in pretrain mode
    if mode=="pretrain":
        assert np.isclose(0.0,np.sum(np.abs([v for k, v in result.logs.items() if k in main_train_losses])))


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    latent_dim=st.sampled_from([4, 8]),
    n_components=st.sampled_from([4, 8]),
)
def test_vade_backward_step_with_teacher(use_gnn, encoder_type, latent_dim, n_components):
    device = torch.device("cpu")

    B, T, N, F_node = 2, 24, 11, 3
    E, F_edge = 11, 1
    adjacency = np.eye(N, dtype=np.float32)

    model = deepof.clustering.models_new.VaDEPT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        n_components=n_components,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        interaction_regularization=0.0,
        kmeans_loss=1.0,
    ).to(device, non_blocking=True)
    step_fn = deepof.clustering.models_new.step_vade

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg(
        latent_dim=latent_dim,
        n_components=n_components,
        kmeans_loss=1.0,
    )
    vade_cfg = deepof.clustering.model_utils_new.VaDECfg(
        tf_cluster_weight=1.0,
        reg_cat_clusters=1.0,
        nonempty_weight=1.0,
        temporal_cohesion_weight=1.0,
        reg_scatter_weight=1.0,
        repel_weight=1.0,
    )
    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg(
        lambda_distill=1.0,
        distill_sharpen_T=0.5,
        distill_conf_weight=False,
        distill_conf_thresh=0.6,
    )

    criterion = deepof.clustering.models_new.VadeLoss(
        common_cfg=common_cfg,
        vade_cfg=vade_cfg,
        teacher_cfg=teacher_cfg,
    ).to(device)
    criterion.set_mode("main")

    # Fake teacher assignments [N_dataset, K]
    tau_star = torch.softmax(torch.randn(B, n_components, device=device), dim=-1)
    criterion.set_teacher(tau_star=tau_star, lambda_distill=1.0)

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    result = step_fn(
        model,
        (x, a, idx),
        SimpleNamespace(criterion=criterion, apply_distill=True),
    )

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()
    
    # Check that teacher loss was generated
    assert torch.isfinite(result.loss)
    assert "distill_loss" in result.logs
    assert np.isfinite(result.logs["distill_loss"])
    assert result.logs["distill_loss"] > 0.0


###################
# VQVAE TESTS
###################


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    latent_dim=st.sampled_from([4, 8]),
    n_components=st.sampled_from([4, 8]),
)
def test_vqvae_backward_step(use_gnn, encoder_type, latent_dim, n_components):
    device = torch.device("cpu")

    B, T, N, F_node = 4, 24, 11, 3
    E, F_edge = 11, 1

    adjacency = np.eye(N, dtype=np.float32)

    model = deepof.clustering.models_new.VQVAEPT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        n_components=n_components,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        interaction_regularization=0.0,
        kmeans_loss=1.0,
    ).to(device, non_blocking=True)

    step_fn = deepof.clustering.models_new.step_vqvae_distill
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    result = step_fn(
        model,
        (x, a, idx),
        SimpleNamespace(apply_distill=False),
    )

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    assert torch.isfinite(result.loss)

    losses = [
        "total_loss",
        "enc_rec_loss",
        "reconstruct_loss",
        "vq_loss",
        "kmeans_loss",
        "number_of_populated_clusters",
        "distill_loss",
    ]
    assert all(loss in result.logs for loss in losses)

    assert np.isfinite(result.logs["total_loss"])
    assert np.isfinite(result.logs["enc_rec_loss"])
    assert np.isfinite(result.logs["reconstruct_loss"])
    assert np.isfinite(result.logs["vq_loss"])
    assert np.isfinite(result.logs["kmeans_loss"])
    assert np.isfinite(result.logs["number_of_populated_clusters"])
    assert np.isfinite(result.logs["distill_loss"])

    # Distillation is disabled in this test
    assert np.isclose(result.logs["distill_loss"], 0.0, atol=1e-8)

    # Logged total should equal the sum of its components
    total_expected = (
        result.logs["enc_rec_loss"]
        + result.logs["reconstruct_loss"]
        + result.logs["vq_loss"]
        + result.logs["kmeans_loss"]
        + result.logs["distill_loss"]
    )
    assert np.isclose(result.logs["total_loss"], total_expected, atol=1e-5, rtol=1e-5)

    # Populated code count should be valid
    assert 1 <= result.logs["number_of_populated_clusters"] <= n_components


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    latent_dim=st.sampled_from([4, 8]),
    n_components=st.sampled_from([4, 8]),
    wm_mode=st.sampled_from(["linear", "sigmoid", "tf_sigmoid"]),
)
def test_vqvae_backward_step_with_distillation(
    use_gnn,
    encoder_type,
    latent_dim,
    n_components,
    wm_mode,
):
    device = torch.device("cpu")

    B, T, N, F_node = 4, 24, 11, 3
    E, F_edge = 11, 1

    adjacency = np.eye(N, dtype=np.float32)

    model = deepof.clustering.models_new.VQVAEPT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        n_components=n_components,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        interaction_regularization=0.0,
        kmeans_loss=1.0,
    ).to(device, non_blocking=True)

    step_fn = deepof.clustering.models_new.step_vqvae_distill
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    distill_head = deepof.clustering.models_new.DiscriminativeHead(
        latent_dim=latent_dim,
        n_components=n_components,
    ).to(device)

    lambda_scheduler = deepof.clustering.models_new.Dynamic_weight_manager(
        n_batches_per_epoch=1,
        mode=wm_mode,
        warmup_epochs=0,
        max_weight=1.0,
        at_max_epochs=0,
        cooldown_epochs=0,
        end_weight=1.0,
    )
    lambda_scheduler.step()  # ensure positive weight

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    # Fake teacher assignments indexed by idx
    tau_star = torch.softmax(torch.randn(B, n_components, device=device), dim=-1)

    result = step_fn(
        model,
        (x, a, idx),
        SimpleNamespace(
            apply_distill=True,
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=0.5,
            distill_conf_weight=False,   # keeps the test robust
            distill_conf_thresh=0.6,
        ),
    )

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    assert torch.isfinite(result.loss)

    losses = [
        "total_loss",
        "enc_rec_loss",
        "reconstruct_loss",
        "vq_loss",
        "kmeans_loss",
        "number_of_populated_clusters",
        "distill_loss",
    ]
    assert all(loss in result.logs for loss in losses)

    assert np.isfinite(result.logs["total_loss"])
    assert np.isfinite(result.logs["enc_rec_loss"])
    assert np.isfinite(result.logs["reconstruct_loss"])
    assert np.isfinite(result.logs["vq_loss"])
    assert np.isfinite(result.logs["kmeans_loss"])
    assert np.isfinite(result.logs["number_of_populated_clusters"])
    assert np.isfinite(result.logs["distill_loss"])

    # Distillation should be active and contribute positively
    assert result.logs["distill_loss"] > 0.0

    # Logged total should equal the sum of its components
    total_expected = (
        result.logs["enc_rec_loss"]
        + result.logs["reconstruct_loss"]
        + result.logs["vq_loss"]
        + result.logs["kmeans_loss"]
        + result.logs["distill_loss"]
    )
    assert np.isclose(result.logs["total_loss"], total_expected, atol=1e-5, rtol=1e-5)

    # Populated code count should be valid
    assert 1 <= result.logs["number_of_populated_clusters"] <= n_components


###################
# CONTRASTIVE TESTS
###################

@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    latent_dim=st.sampled_from([4,8]),
    similarity_function=st.sampled_from(["cosine","dot","euclidean","edit"]),
    loss_function=st.sampled_from(["nce","dcl","fc","hard_dcl"]),
)
def test_contrastive_backward_step(use_gnn,encoder_type,latent_dim,similarity_function,loss_function):
    device = torch.device("cpu")

    B, T, N, F_node = 2, 24, 11, 3
    E, F_edge = 11, 1

    adjacency = np.eye(N, dtype=np.float32)

    # Create model
    model = deepof.clustering.models_new.ContrastivePT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        similarity_function=similarity_function,
        loss_function=loss_function,
        temperature=0.1,
        beta=0.1,
        tau=0.1,
    ).to(device, non_blocking=True)
    step_fn = deepof.clustering.models_new.step_contrastive_distill

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg(
        contrastive_similarity_function=similarity_function,
        contrastive_loss_function=loss_function,
        aug_min_shift = 1,
        aug_max_shift = 6,
        aug_p_shift = 0.5,
        aug_max_rot = 30, 
        aug_n_rot = 4, 
        aug_p_rot = 0.5,
        aug_max_interp = 8,
        aug_min_interp = 3,         
        aug_p_interp = 0.5, 
        aug_noise_sigma = 0.03,  
        aug_p_noise = 0.5, 
    )

    meta_info={}
    meta_info["node_columns"]=[('A','x'),('B','x'),('C','x'),('D','x'),('E','x'),('F','x'),('G','x'),('H','x'),('I','x'),('J','x'),('K','x')]
    meta_info["edge_columns"]=[('A','B'),('B','C'),('C','D'),('D','E'),('E','F'),('F','G'),('F','H'),('F','I'),('H','J'),('J','K'),('K','H')]

    edge_index_global, edge_index_local, _ = deepof.clustering.models_new._build_edge_from_metainfo(
    meta_info=meta_info,
    device=device,
    n_nodes=N,
    return_local=True,
    )
    rot_precomp = deepof.clustering.models_new.build_rotation_precomp(edge_index=edge_index_local, n_nodes=N, device=device)

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    result = step_fn(model, (x, a, idx), SimpleNamespace(edge_index=edge_index_global, edge_index_local=edge_index_local, contrastive_cfg=contrastive_cfg, rot_precomp=rot_precomp, apply_distill=False))

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    assert torch.isfinite(result.loss)
    
    losses = ['total_loss','pos_similarity','neg_similarity', 'distill_loss']
    # Check if all losses are present
    assert all(loss in result.logs for loss in losses)
    # positive and negative similarity are within specific ranges based on similarity function
    if similarity_function in {"cosine", "dot"}:
        assert -1.0001 <= result.logs["pos_similarity"] <= 1.0001
        assert -1.0001 <= result.logs["neg_similarity"] <= 1.0001
    elif similarity_function in {"euclidean", "edit"}:
        assert (1.0 / 3.0) - 1e-4 <= result.logs["pos_similarity"] <= 1.0001
        if loss_function in {"fc"}:
            assert 0.0 <= result.logs["neg_similarity"] <= 1.0001
        else:
            assert (1.0 / 3.0) - 1e-4 <= result.logs["neg_similarity"] <= 1.0001
    # Distill needs to be 0 (as it is disabled in this test) 
    assert result.logs['distill_loss'] == 0


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    latent_dim=st.sampled_from([4, 8]),
    n_components=st.sampled_from([4, 8]),
    wm_mode=st.sampled_from(["linear", "sigmoid","tf_sigmoid"]),
    similarity_function=st.sampled_from(["cosine", "dot", "euclidean", "edit"]),
    loss_function=st.sampled_from(["nce", "dcl", "fc", "hard_dcl"]),
)
def test_contrastive_backward_step_with_distillation(
    use_gnn,
    encoder_type,
    latent_dim,
    n_components,
    wm_mode,
    similarity_function,
    loss_function,
):
    device = torch.device("cpu")

    B, T, N, F_node = 4, 24, 11, 3
    E, F_edge = 11, 1

    adjacency = np.eye(N, dtype=np.float32)

    model = deepof.clustering.models_new.ContrastivePT(
        input_shape=(T, N, F_node),
        edge_feature_shape=(T, E, F_edge),
        adjacency_matrix=adjacency,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        use_gnn=use_gnn,
        similarity_function=similarity_function,
        loss_function=loss_function,
        temperature=0.1,
        beta=0.1,
        tau=0.1,
    ).to(device, non_blocking=True)

    step_fn = deepof.clustering.models_new.step_contrastive_distill
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg(
        contrastive_similarity_function=similarity_function,
        contrastive_loss_function=loss_function,
        aug_min_shift=1,
        aug_max_shift=6,
        aug_p_shift=0.5,
        aug_max_rot=30,
        aug_n_rot=4,
        aug_p_rot=0.5,
        aug_max_interp=8,
        aug_min_interp=3,
        aug_p_interp=0.5,
        aug_noise_sigma=0.03,
        aug_p_noise=0.5,
    )

    meta_info = {}
    meta_info["node_columns"] = [
        ("A", "x"), ("B", "x"), ("C", "x"), ("D", "x"), ("E", "x"),
        ("F", "x"), ("G", "x"), ("H", "x"), ("I", "x"), ("J", "x"), ("K", "x"),
    ]
    meta_info["edge_columns"] = [
        ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"),
        ("F", "G"), ("F", "H"), ("F", "I"), ("H", "J"), ("J", "K"), ("K", "H"),
    ]

    edge_index_global, edge_index_local, _ = deepof.clustering.models_new._build_edge_from_metainfo(
        meta_info=meta_info,
        device=device,
        n_nodes=N,
        return_local=True,
    )
    rot_precomp = deepof.clustering.models_new.build_rotation_precomp(
        edge_index=edge_index_local,
        n_nodes=N,
        device=device,
    )

    distill_head = deepof.clustering.models_new.DiscriminativeHead(
        latent_dim=latent_dim,
        n_components=n_components,
    ).to(device)

    # Use the real scheduler
    lambda_scheduler = deepof.clustering.models_new.Dynamic_weight_manager(
        n_batches_per_epoch=1,
        mode=wm_mode,
        warmup_epochs=0,
        max_weight=1.0,
        at_max_epochs=0,
        cooldown_epochs=0,
        end_weight=1.0,
    )
    lambda_scheduler.step()  # ensure get_weight() is 1.0

    x = torch.randn(B, T, N, F_node, device=device)
    a = torch.randn(B, T, E, F_edge, device=device)
    idx = torch.arange(B, device=device)

    tau_star = torch.softmax(torch.randn(B, n_components, device=device), dim=-1)

    result = step_fn(
        model,
        (x, a, idx),
        SimpleNamespace(
            edge_index=edge_index_global,
            edge_index_local=edge_index_local,
            contrastive_cfg=contrastive_cfg,
            rot_precomp=rot_precomp,
            apply_distill=True,
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=0.5,
            distill_conf_weight=False,
            distill_conf_thresh=0.6,
        ),
    )

    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    assert torch.isfinite(result.loss)

    losses = ["total_loss", "pos_similarity", "neg_similarity", "distill_loss"]
    assert all(loss in result.logs for loss in losses)

    assert np.isfinite(result.logs["pos_similarity"])
    assert np.isfinite(result.logs["neg_similarity"])
    assert np.isfinite(result.logs["distill_loss"])

    # Check if Distillation is active and contributes positively
    assert result.logs["distill_loss"] > 0.0

    if similarity_function in {"cosine", "dot"}:
        assert -1.0001 <= result.logs["pos_similarity"] <= 1.0001
        assert -1.0001 <= result.logs["neg_similarity"] <= 1.0001
    elif similarity_function in {"euclidean", "edit"}:
        assert (1.0 / 3.0) - 1e-4 <= result.logs["pos_similarity"] <= 1.0001
        if loss_function == "fc":
            assert 0.0 <= result.logs["neg_similarity"] <= 1.0001
        else:
            assert (1.0 / 3.0) - 1e-4 <= result.logs["neg_similarity"] <= 1.0001


#
#
#
##
#
####
#
###################
# Old models
###################
#
####
#
##
#
#
#


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_VQVAE_build(use_gnn, encoder_type):
    vqvae = deepof.clustering.models_new.VQVAEPT(
        input_shape=(15, 11, 3),
        edge_feature_shape=(15, 11, 1),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
        encoder_type=encoder_type,
        n_components=10,
        latent_dim=8,
    )
    vqvae.compile()


@settings(deadline=None)
@given(
    use_gnn=st.booleans(),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_Contrastive_build(use_gnn, encoder_type):
    contrasts = deepof.clustering.models_new.ContrastivePT(
        input_shape=(15, 11, 3),
        edge_feature_shape=(15, 11, 1),
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(11, 11)
        ).todense(),
        use_gnn=use_gnn,
        encoder_type=encoder_type,
        latent_dim=8,
    )
    contrasts.compile()

test_Contrastive_build()





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

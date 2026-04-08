# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

import os
import random
from shutil import rmtree

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Optional

import deepof.model_utils
import deepof.models
import deepof.clustering
import deepof.clustering.models_new
import deepof.clustering.model_utils_new


###################
# HELPER OBJECTS
###################


class TinyIndexedDataset(Dataset):
    def __init__(self, n=4, T=24, N=11, F_node=3, F_edge=1):
        self.X = torch.randn(n, T, N, F_node)
        self.A = torch.randn(n, T, N, F_edge)
        self.x_shape = (T, N, F_node)
        self.a_shape = (T, N, F_edge)
        self.n_videos = n

    def make_loader(
        self,
        batch_size: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        iterable_for_h5: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 0,
        persistent_workers: Optional[bool] = None,
        seed: Optional[int] = None,
        block_shuffle: bool = False,
        permute_within_block: bool = False,
    ) -> DataLoader:
        """Minimal loader compatible with extract_latents / fit_nodes_pca."""
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))

        # In-memory dataset → no HDF5 iterable needed
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            generator=gen,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx_t = torch.tensor(idx, dtype=torch.long)
        # Keep idx and vid equal here because validate_one_epoch_indexed currently
        # reads the 4th returned element as idx
        return self.X[idx], self.A[idx], idx_t, idx_t


class DummyWriter:
    def __init__(self):
        self.flushed = False
        self.closed = False

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True

    def __getattr__(self, _):
        return lambda *args, **kwargs: None


def _tiny_setup():
    ds = TinyIndexedDataset()
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    N = ds.x_shape[1]
    names = [chr(65 + i) for i in range(N)]

    adjacency = np.zeros((N, N), dtype=np.float32)
    edge_columns = []
    for i in range(N - 1):
        edge_columns.append((names[i], names[i + 1]))
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    edge_columns.append(("A", "C"))
    adjacency[0, 2] = 1
    adjacency[2, 0] = 1

    meta_info = {
        "node_columns": [(name, "x") for name in names],
        "edge_columns": edge_columns,
    }
    return loader, loader, adjacency, meta_info


###################
# VADE TESTS
###################


@settings(deadline=None, max_examples=2)
@given(use_teacher=st.booleans())
def test_fit_vade_smoke(use_teacher):
    out_path = os.path.join(".", "tests", "test_examples", "test_data", "fit_contrastive_smoke")

    if os.path.exists(out_path):
        rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    train_loader, val_loader, adjacency, _ = _tiny_setup()
    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg()
    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg()
    vade_cfg = deepof.clustering.model_utils_new.VaDECfg()
    writer = DummyWriter()

    common_cfg.output_path = out_path
    common_cfg.epochs = 1
    common_cfg.save_weights = False
    common_cfg.latent_dim = 4
    common_cfg.n_components = 4
    common_cfg.diag_max_batches = 1
    teacher_cfg.use_turtle_teacher=use_teacher
    teacher_cfg.teacher_outer_steps=10
    teacher_cfg.teacher_inner_steps=10
    teacher_cfg.pca_nodes_dim=4
    teacher_cfg.teacher_batch_size=2


    seen_apply_distill = []

    orig_step = deepof.clustering.models_new.step_vade

    def wrapped_step(model, batch, ctx):
        seen_apply_distill.append(bool(getattr(ctx.criterion, "lambda_scheduler", False)))
        return orig_step(model, batch, ctx)

    deepof.clustering.models_new.step_vade = wrapped_step

    model_val, model_score, _, _ = deepof.clustering.models_new.fit_VADE(
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessed_train={},
        adjacency_matrix=adjacency,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        vade_cfg=vade_cfg,
        writer=writer,
    )

    assert isinstance(model_val, deepof.clustering.models_new.VaDEPT)
    assert isinstance(model_score, deepof.clustering.models_new.VaDEPT)
    assert writer.flushed and writer.closed
    assert seen_apply_distill
    assert False in seen_apply_distill
    # no teacher in pretraining
    if use_teacher:
        assert np.sum(seen_apply_distill)/len(seen_apply_distill) == 4/6
    else:
        assert not any(seen_apply_distill)
    assert (True in seen_apply_distill) == use_teacher


    deepof.clustering.models_new.step_vqvae_distill = orig_step
    if os.path.exists(out_path):
        rmtree(out_path)


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


@settings(deadline=None, max_examples=2)
@given(use_teacher=st.booleans())
def test_fit_vqvae_smoke(use_teacher):
    out_path = os.path.join(".", "tests", "test_examples", "test_data", "fit_contrastive_smoke")

    if os.path.exists(out_path):
        rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    train_loader, val_loader, adjacency, _ = _tiny_setup()
    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg()
    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg()
    writer = DummyWriter()

    common_cfg.output_path = out_path
    common_cfg.epochs = 1
    common_cfg.save_weights = False
    common_cfg.latent_dim = 4
    common_cfg.n_components = 4
    common_cfg.diag_max_batches = 1


    seen_apply_distill = []
    diag_calls = {"n": 0}

    orig_step = deepof.clustering.models_new.step_vqvae_distill
    orig_teacher = deepof.clustering.models_new.maybe_build_turtle_teacher
    orig_diag = deepof.clustering.models_new._compute_diagnostics

    def wrapped_step(model, batch, ctx):
        seen_apply_distill.append(bool(getattr(ctx, "apply_distill", False)))
        return orig_step(model, batch, ctx)

    def fake_teacher(**kwargs):
        if not use_teacher:
            return None, None, None
        n = kwargs["train_dataset"].n_videos
        k = kwargs["common_cfg"].n_components
        tau_star = torch.softmax(torch.randn(n, k), dim=-1)
        return object(), tau_star, None

    def fake_diag(**kwargs):
        diag_calls["n"] += 1
        return {"alignment_score": 0.7, "conf_norm": 0.4, "bal_norm": 0.6}

    deepof.clustering.models_new.step_vqvae_distill = wrapped_step
    deepof.clustering.models_new.maybe_build_turtle_teacher = fake_teacher
    deepof.clustering.models_new._compute_diagnostics = fake_diag

    model_val, model_score, _, _ = deepof.clustering.models_new.fit_VQVAE(
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessed_train={},
        adjacency_matrix=adjacency,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        writer=writer,
    )

    assert isinstance(model_val, deepof.clustering.models_new.VQVAEPT)
    assert isinstance(model_score, deepof.clustering.models_new.VQVAEPT)
    assert writer.flushed and writer.closed
    assert seen_apply_distill
    assert False in seen_apply_distill
    assert (True in seen_apply_distill) == use_teacher
    assert diag_calls["n"] == int(use_teacher)


    deepof.clustering.models_new.step_vqvae_distill = orig_step
    deepof.clustering.models_new.maybe_build_turtle_teacher = orig_teacher
    deepof.clustering.models_new._compute_diagnostics = orig_diag
    if os.path.exists(out_path):
        rmtree(out_path)


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


@settings(deadline=None, max_examples=2)
@given(use_teacher=st.booleans())
def test_fit_contrastive_smoke(use_teacher):
    out_path = os.path.join(".", "tests", "test_examples", "test_data", "fit_contrastive_smoke")

    if os.path.exists(out_path):
        rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    train_loader, val_loader, adjacency, meta_info = _tiny_setup()
    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg()
    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg()
    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg()
    writer = DummyWriter()

    common_cfg.output_path = out_path
    common_cfg.epochs = 1
    common_cfg.save_weights = False
    common_cfg.latent_dim = 4
    common_cfg.n_components = 4
    common_cfg.diag_max_batches = 1

    seen_apply_distill = []
    diag_calls = {"n": 0}

    orig_step = deepof.clustering.models_new.step_contrastive_distill
    orig_teacher = deepof.clustering.models_new.maybe_build_turtle_teacher
    orig_diag = deepof.clustering.models_new._compute_diagnostics

    def wrapped_step(model, batch, ctx):
        seen_apply_distill.append(bool(getattr(ctx, "apply_distill", False)))
        return orig_step(model, batch, ctx)

    def fake_teacher(**kwargs):
        if not use_teacher:
            return None, None, None
        n = kwargs["train_dataset"].n_videos
        k = kwargs["common_cfg"].n_components
        tau_star = torch.softmax(torch.randn(n, k), dim=-1)
        return object(), tau_star, None

    def fake_diag(**kwargs):
        diag_calls["n"] += 1
        return {"alignment_score": 0.7, "conf_norm": 0.4, "bal_norm": 0.6}

    deepof.clustering.models_new.step_contrastive_distill = wrapped_step
    deepof.clustering.models_new.maybe_build_turtle_teacher = fake_teacher
    deepof.clustering.models_new._compute_diagnostics = fake_diag

    model_val, model_score, _, _ = deepof.clustering.models_new.fit_contrastive(
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessed_train={},
        adjacency_matrix=adjacency,
        meta_info=meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        contrastive_cfg=contrastive_cfg,
        writer=writer,
    )

    assert isinstance(model_val, deepof.clustering.models_new.ContrastivePT)
    assert isinstance(model_score, deepof.clustering.models_new.ContrastivePT)
    assert writer.flushed and writer.closed
    assert seen_apply_distill
    assert False in seen_apply_distill
    assert (True in seen_apply_distill) == use_teacher
    assert diag_calls["n"] == int(use_teacher)


    deepof.clustering.models_new.step_contrastive_distill = orig_step
    deepof.clustering.models_new.maybe_build_turtle_teacher = orig_teacher
    deepof.clustering.models_new._compute_diagnostics = orig_diag
    if os.path.exists(out_path):
        rmtree(out_path)


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

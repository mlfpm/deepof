# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Data structures and functions for analyzing supervised and unsupervised model results.

"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_embedding(model, data, verbose=False):
    """
    Get the embedding of the data.

    Args:
        model: The trained model to use.
        data: The data to use.
        verbose: Whether to print the progress.

    """

    embeddings = model.encoder.predict(data, verbose=verbose)
    cluster_labels = model.soft_quantizer.predict(data, verbose=verbose)

    return embeddings, np.argmax(cluster_labels, axis=1)


def split_results_in_time_bins(embeddings, bin_size, cluster_labels=None, ruptures=None):
    """
    Splits all inputs into bins of equal size.

    Args:
        embeddings: The data to split. Can be a set of supervised annotations or unsupervised embeddings.
        bin_size: The bin size to use.
        cluster_labels: The cluster labels to use. If included, the function will return a list of lists, where each
        list contains the cluster labels for the corresponding bin.
        ruptures: The ruptures to use. If included, the function will return a list of lists, where each list contains
        the ruptures for the corresponding bin.

    """

    pass


def get_aggregated_embedding(
    embeddings,
    exp_labels,
    cluster_labels=None,
    verbose=False,
    aggregation_mode="cluster_population",
):
    """
    Get the embedding of the data.

    Args:
        embeddings: Non-grouped embedding, with one entry per changepoint detection rupture.
        exp_labels: The labels to use.
        cluster_labels: The cluster labels to use.
        verbose: Whether to print the progress.
        aggregation_mode: Controls how the embedding is aggregated to generate per-video mappings to the latent space.
        If "mean", embeddings for all ruptures are averaged. If "cluster_population", the embedding for each rupture is
        summed for each cluster independently, generating a vector of length equal to the number of clusters, where each
        entry is the proportion of time spent on the corresponding cluster.

    """

    pass


def get_growing_distance_between_conditions(
    model,
    exp_labels,
    data,
    min_time_scale,
    max_time_scale,
    verbose=False,
):
    """
    Get the growing distance between conditions.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        min_time_scale: The minimum time scale to use.
        max_time_scale: The maximum time scale to use.
        verbose: Whether to print the progress.

    """

    pass


def compare_cluster_enrichment(model, exp_labels, data, verbose=False):
    """
    Compare the cluster enrichment of the data.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        verbose: Whether to print the progress.

    """

    pass


def compute_markov_stationary_distribution(cluster_labels, n_clusters):
    """
    Compute the markov stationary distribution from the model's transition matrix.

    Args:
        cluster_labels: The cluster labels to use.
        n_clusters: The number of clusters to use.
    """

    pass


def compare_cluster_markov_dynamics(
    model, exp_labels, data, verbose=False
):
    """
    Compare the cluster dynamics of the data.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        verbose: Whether to print the progress.

    """

    pass

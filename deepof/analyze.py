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


def get_embedding(model, data, exp_labels, batch_size=32, verbose=False):
    """
    Get the embedding of the data.

    Args:
        model: The model to use.
        data: The data to use.
        batch_size: The batch size to use.
        verbose: Whether to print the progress.

    """
    embedding = []
    for i in range(0, len(data), batch_size):
        if verbose:
            print("Embedding batch {}/{}".format(i, len(data)))
        embedding.append(model.predict(data[i : i + batch_size]))
    return np.concatenate(embedding)


def get_aggregated_embedding(model, exp_labels, data, batch_size=32, verbose=False, aggregation_mode="cluster_population"):
    """
    Get the embedding of the data.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        batch_size: The batch size to use.
        verbose: Whether to print the progress.
        aggregation_mode: Controls how the embedding is aggregated to generate per-video mappings to the latent space.
        If "mean", embeddings for all ruptures are averaged. If "cluster_population", the embedding for each rupture is
        summed for each cluster independently, generating a vector of length equal to the number of clusters, where each
        entry is the proportion of time spent on the corresponding cluster.

    """
    embedding = []
    for i in range(0, len(data), batch_size):
        if verbose:
            print("Embedding batch {}/{}".format(i, len(data)))
        embedding.append(model.predict(data[i : i + batch_size]))
    embedding = np.concatenate(embedding)
    embedding = pd.DataFrame(embedding, index=exp_labels)
    if aggregation_mode == "mean":
        return embedding.groupby(level=0).mean()
    elif aggregation_mode == "cluster_population":
        return embedding.groupby(level=0).sum()
    else:
        raise ValueError("Invalid aggregation mode")


def get_growing_distance_between_conditions(
    model, exp_labels, data, min_time_scale, max_time_scale, batch_size=32, verbose=False,
):
    """
    Get the growing distance between conditions.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        batch_size: The batch size to use.
        verbose: Whether to print the progress.

    """
    embedding = []
    for i in range(0, len(data), batch_size):
        if verbose:
            print("Embedding batch {}/{}".format(i, len(data)))
        embedding.append(model.predict(data[i : i + batch_size]))
    embedding = np.concatenate(embedding)
    embedding = pd.DataFrame(embedding, index=exp_labels)
    return embedding.groupby(level=0).apply(lambda x: x.diff().max())


def compare_cluster_enrichment(model, exp_labels, data, batch_size=32, verbose=False):
    """
    Compare the cluster enrichment of the data.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        batch_size: The batch size to use.
        verbose: Whether to print the progress.

    """

    pass


def compare_cluster_markov_dynamics(
    model, exp_labels, data, batch_size=32, verbose=False
):
    """
    Compare the cluster dynamics of the data.

    Args:
        model: The model to use.
        exp_labels: The labels to use.
        data: The data to use.
        batch_size: The batch size to use.
        verbose: Whether to print the progress.

    """

    pass

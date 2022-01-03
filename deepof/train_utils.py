# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels.

"""

import json
import os
from datetime import date, datetime
from typing import Tuple, Union, Any, List

import numpy as np
import tensorflow as tf
from keras_tuner import BayesianOptimization, Hyperband, Objective
from tensorboard.plugins.hparams import api as hp

import deepof.hypermodels
import deepof.model_utils

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    """

    Custom early stopping callback. Prevents the model from stopping before warmup is over

    """

    def __init__(self, start_epoch, *args, **kwargs):
        """

        Initializes the CustomStopper callback.

        Args:
            start_epoch: epoch from which performance will be taken into account when deciding whether to stop training.
            *args: arguments passed to the callback.
            **kwargs: keyword arguments passed to the callback.

        """
        super(CustomStopper, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def get_config(self):  # pragma: no cover
        """

        Updates callback metadata

        """

        config = super().get_config().copy()
        config.update({"start_epoch": self.start_epoch})
        return config

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def load_treatments(train_path):
    """

    Loads a dictionary containing the treatments per individual, to be loaded as metadata in the coordinates class.

    Args:
        train_path (str): path to the training data.

    Returns:
        dict: dictionary containing the treatments per individual.

    """

    try:
        with open(
            os.path.join(
                train_path,
                [i for i in os.listdir(train_path) if i.endswith(".json")][0],
            ),
            "r",
        ) as handle:
            treatment_dict = json.load(handle)
    except IndexError:
        treatment_dict = None

    return treatment_dict


def get_callbacks(
    X_train: np.array,
    batch_size: int,
    embedding_model: str,
    phenotype_prediction: float = 0.0,
    next_sequence_prediction: float = 0.0,
    supervised_prediction: float = 0.0,
    overlap_loss: float = 0.0,
    loss: str = "ELBO",
    loss_warmup: int = 0,
    warmup_mode: str = "none",
    input_type: str = False,
    cp: bool = False,
    reg_cat_clusters: bool = False,
    reg_cluster_variance: bool = False,
    entropy_knn: int = 100,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """

    Generates callbacks used for model training.

    Args:
        X_train (np.array): Training data
        batch_size (int): Batch size
        embedding_model (str): Embedding model used for training. Must be "VQVAE" or "GMVAE".
        phenotype_prediction (float): Weight of the phenotype prediction loss.
        next_sequence_prediction (float): Weight of the next sequence prediction loss.
        supervised_prediction (float): Weight of the supervised prediction loss
        overlap_loss (float): Weight of the overlap loss
        loss (str): Loss function to use for training
        loss_warmup (int): Number of epochs to warmup the loss function
        warmup_mode (str): Warmup mode to use for training
        input_type (str): Input type to use for training
        cp (bool): Whether to use checkpointing or not
        reg_cat_clusters (bool): Whether to use regularization on categorical clusters
        reg_cluster_variance (bool): Whether to use regularization on cluster variance
        entropy_knn (int): Number of nearest neighbors to use for entropy regularization
        logparam (dict): Dictionary containing the hyperparameters to log in tensorboard
        outpath (str): Path to the output directory
        run (int): Run number to use for checkpointing

    Returns:
        List[Union[Any]]: List of callbacks to be used for training

    """

    latreg = "none"
    if reg_cat_clusters and not reg_cluster_variance:
        latreg = "categorical"
    elif reg_cluster_variance and not reg_cat_clusters:
        latreg = "variance"
    elif reg_cat_clusters and reg_cluster_variance:
        latreg = "categorical+variance"

    run_ID = "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
        "deepof_unsupervised_{}".format(embedding_model),
        ("_input_type={}".format(input_type) if input_type else "coords"),
        (
            ("_NSPred={}".format(next_sequence_prediction))
            if embedding_model == "GMVAE"
            else ""
        ),
        (
            ("_PPred={}".format(phenotype_prediction))
            if embedding_model == "GMVAE"
            else ""
        ),
        (
            ("_SupPred={}".format(supervised_prediction))
            if embedding_model == "GMVAE"
            else ""
        ),
        (("_loss={}".format(loss)) if embedding_model == "GMVAE" else ""),
        (
            ("_overlap_loss={}".format(overlap_loss))
            if embedding_model == "GMVAE"
            else ""
        ),
        (("_loss_warmup={}".format(loss_warmup)) if embedding_model == "GMVAE" else ""),
        (("_warmup_mode={}".format(warmup_mode)) if embedding_model == "GMVAE" else ""),
        ("_encoding={}".format(logparam["encoding"]) if logparam is not None else ""),
        ("_k={}".format(logparam["k"]) if logparam is not None else ""),
        (("_latreg={}".format(latreg)) if embedding_model == "GMVAE" else ""),
        (("_entknn={}".format(entropy_knn)) if embedding_model == "GMVAE" else ""),
        ("_run={}".format(run) if run else ""),
        ("_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")) if not run else ""),
    )

    log_dir = os.path.abspath(os.path.join(outpath, "fit", run_ID))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch=2,
    )

    onecycle = deepof.model_utils.one_cycle_scheduler(
        X_train.shape[0] // batch_size * 250,
        max_rate=0.005,
        log_dir=os.path.join(outpath, "metrics", run_ID),
    )

    callbacks = [run_ID, tensorboard_callback, onecycle]

    if cp:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(outpath, "checkpoints", run_ID + "/cp-{epoch:04d}.ckpt"),
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )
        callbacks.append(cp_callback)

    return callbacks


def log_hyperparameters(phenotype_class: float, rec: str):
    """

    Blueprint for hyperparameter and metric logging in tensorboard during hyperparameter tuning

    Args:
        phenotype_class (float): Phenotype class to use for training
        rec (str): Record to use for training

    Returns:
        logparams (list): List containing the hyperparameters to log in tensorboard.
        metrics (list): List containing the metrics to log in tensorboard.

    """

    logparams = [
        hp.HParam(
            "encoding",
            hp.Discrete([2, 4, 6, 8, 12, 16]),
            display_name="encoding",
            description="encoding size dimensionality",
        ),
        hp.HParam(
            "k",
            hp.IntInterval(min_value=1, max_value=25),
            display_name="k",
            description="cluster_number",
        ),
        hp.HParam(
            "loss",
            hp.Discrete(["ELBO", "MMD", "ELBO+MMD"]),
            display_name="loss function",
            description="loss function",
        ),
    ]

    metrics = [
        hp.Metric("val_{}mae".format(rec), display_name="val_{}mae".format(rec)),
        hp.Metric("val_{}mse".format(rec), display_name="val_{}mse".format(rec)),
    ]
    if phenotype_class:
        logparams.append(
            hp.HParam(
                "pheno_weight",
                hp.RealInterval(min_value=0.0, max_value=1000.0),
                display_name="pheno weight",
                description="weight applied to phenotypic classifier from the latent space",
            )
        )
        metrics += [
            hp.Metric(
                "phenotype_prediction_accuracy",
                display_name="phenotype_prediction_accuracy",
            ),
            hp.Metric(
                "phenotype_prediction_auc",
                display_name="phenotype_prediction_auc",
            ),
        ]

    return logparams, metrics


def autoencoder_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    embedding_model: str,
    batch_size: int,
    encoding_size: int,
    epochs: int,
    hparams: dict,
    kl_annealing_mode: str,
    kl_warmup: int,
    log_history: bool,
    log_hparams: bool,
    loss: str,
    mmd_annealing_mode: str,
    mmd_warmup: int,
    montecarlo_kl: int,
    n_components: int,
    output_path: str,
    overlap_loss: float,
    next_sequence_prediction: float,
    phenotype_prediction: float,
    supervised_prediction: float,
    pretrained: str,
    save_checkpoints: bool,
    save_weights: bool,
    reg_cat_clusters: bool,
    reg_cluster_variance: bool,
    entropy_knn: int,
    input_type: str,
    run: int = 0,
    strategy: tf.distribute.Strategy = "one_device",
):
    """

    Implementation function for deepof.data.coordinates.deep_unsupervised_embedding. Trains the specified autoencoder
    on the preprocessed data.

    Args:
        preprocessed_object (tuple): Tuple containing the preprocessed data.
        embedding_model (str): Name of the embedding model to use. Must be one of "VQVAE" or "GMVAE".
        batch_size (int): Batch size to use for training.
        encoding_size (int): Encoding size to use for training.
        epochs (int): Number of epochs to train the autoencoder for.
        hparams (dict): Dictionary containing the hyperparameters to use for training.
        kl_annealing_mode (str): Annealing mode to use for KL annealing. Must be one of "linear" or "sigmoid". Only used
        if embedding_model is "GMVAE".
        kl_warmup (int): Number of epochs to warmup KL annealing. Only used if embedding_model is "GMVAE".
        log_history (bool): Whether to log the history of the autoencoder.
        log_hparams (bool): Whether to log the hyperparameters used for training.
        loss (str): Loss function to use for training. Must be one of "ELBO", "MMD", or "ELBO+MMD". Only used if
        embedding_model is "GMVAE".
        mmd_annealing_mode (str): Annealing mode to use for MMD annealing. Must be one of "linear" or "sigmoid". Only used
        if embedding_model is "GMVAE".
        mmd_warmup (int): Number of epochs to warmup MMD annealing. Only used if embedding_model is "GMVAE".
        montecarlo_kl (int): Number of Monte Carlo samples to use for KL annealing. Only used if embedding_model is
        "GMVAE".
        n_components (int): Number of components to use for the GMVAE.
        output_path (str): Path to the output directory.
        overlap_loss (float): Weight to use for the overlap loss. Only used if embedding_model is "GMVAE".
        next_sequence_prediction (float): Weight to use for the next sequence prediction loss. Only used if embedding_model
        is "GMVAE".
        phenotype_prediction (float): Weight to use for the phenotype prediction loss. Only used if embedding_model is
        "GMVAE".
        supervised_prediction (float): Weight to use for the supervised prediction loss. Only used if embedding_model is
        "GMVAE".
        pretrained (str): Path to the pretrained weights to use for the autoencoder.
        save_checkpoints (bool): Whether to save checkpoints during training.
        save_weights (bool): Whether to save the weights of the autoencoder after training.
        reg_cat_clusters (bool): Whether to use the categorical cluster regularization loss. Only used if embedding_model
        is "GMVAE".
        reg_cluster_variance (bool): Whether to use the cluster variance regularization loss. Only used if embedding_model
        is "GMVAE".
        entropy_knn (int): Number of nearest neighbors to use for the entropy regularization loss. Only used if embedding_model
        is "GMVAE".
        input_type (str): Input type of the TableDict objects used for preprocessing. For logging purposes only.
        run (int): Run number to use for logging.
        strategy (tf.distribute.Strategy): Distribution strategy to use for training.

    Returns:
        List of trained models (encoder, decoder, grouper and full autoencoders.

    """

    # Check if a GPU is available and if not, fall back to CPU
    if strategy == "one_device":
        if len(tf.config.list_physical_devices("GPU")) > 0:
            strategy = tf.distribute.OneDeviceStrategy("gpu")
        else:
            strategy = tf.distribute.OneDeviceStrategy("cpu")

    # Load data
    X_train, y_train, X_val, y_val = preprocessed_object

    # Make sure that batch_size is not larger than training set
    if batch_size > preprocessed_object[0].shape[0]:
        batch_size = preprocessed_object[0].shape[0]

    # To avoid stability issues
    tf.keras.backend.clear_session()

    # Set options for tf.data.Datasets
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    # Defines what to log on tensorboard (useful for keeping track of different models)
    logparam = {
        "encoding": encoding_size,
        "k": n_components,
        "loss": loss,
    }
    if phenotype_prediction:
        logparam["pheno_weight"] = phenotype_prediction

    # Load callbacks
    run_ID, *cbacks = get_callbacks(
        X_train=X_train,
        batch_size=batch_size,
        embedding_model=embedding_model,
        phenotype_prediction=phenotype_prediction,
        next_sequence_prediction=next_sequence_prediction,
        supervised_prediction=supervised_prediction,
        loss=loss,
        loss_warmup=kl_warmup,
        overlap_loss=overlap_loss,
        warmup_mode=kl_annealing_mode,
        input_type=input_type,
        cp=save_checkpoints,
        reg_cat_clusters=reg_cat_clusters,
        reg_cluster_variance=reg_cluster_variance,
        entropy_knn=entropy_knn,
        logparam=logparam,
        outpath=output_path,
        run=run,
    )
    if not log_history:
        cbacks = cbacks[1:]

    # Logs hyperparameters to tensorboard
    rec = "reconstruction_" if phenotype_prediction else ""
    if log_hparams:
        logparams, metrics = log_hyperparameters(phenotype_prediction, rec)

        with tf.summary.create_file_writer(
            os.path.join(output_path, "hparams", run_ID)
        ).as_default():
            hp.hparams_config(
                hparams=logparams,
                metrics=metrics,
            )

    # Gets the number of supervised features
    try:
        supervised_features = (
            y_train.shape[1] if not phenotype_prediction else y_train.shape[1] - 1
        )
    except IndexError:
        supervised_features = 0

    # Build model
    with strategy.scope():
        if embedding_model == "VQVAE":
            ae_models = deepof.models.VQVAE(
                architecture_hparams=hparams,
                input_shape=X_train.shape,
                latent_dim=encoding_size,
                n_components=n_components,
            )
            encoder, decoder, quantizer, ae = (
                ae_models.encoder,
                ae_models.decoder,
                ae_models.quantizer,
                ae_models.vqvae,
            )
            return_list = (encoder, decoder, quantizer, ae)

        elif embedding_model == "GMVAE":
            ae_models = deepof.models.GMVAE(
                architecture_hparams=hparams,
                input_shape=X_train.shape,
                batch_size=batch_size,
                latent_dim=encoding_size,
                kl_annealing_mode=kl_annealing_mode,
                kl_warmup_epochs=kl_warmup,
                loss=loss,
                mmd_annealing_mode=mmd_annealing_mode,
                mmd_warmup_epochs=mmd_warmup,
                montecarlo_kl=montecarlo_kl,
                n_components=n_components,
                overlap_loss=overlap_loss,
                next_sequence_prediction=next_sequence_prediction,
                phenotype_prediction=phenotype_prediction,
                supervised_prediction=supervised_prediction,
                supervised_features=supervised_features,
                reg_cat_clusters=reg_cat_clusters,
                reg_cluster_variance=reg_cluster_variance,
            )
            encoder, decoder, grouper, ae = (
                ae_models.encoder,
                ae_models.decoder,
                ae_models.grouper,
                ae_models.gmvae,
            )
            return_list = (encoder, decoder, grouper, ae)

    if pretrained:
        # If pretrained models are specified, load weights and return
        ae.load_weights(pretrained)
        return return_list

    callbacks_ = cbacks + [
        CustomStopper(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            start_epoch=max(kl_warmup, mmd_warmup),
        ),
    ]

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    if next_sequence_prediction > 0.0:
        Xs, ys = X_train[:-1], [X_train[:-1], X_train[1:]]
        Xvals, yvals = X_val[:-1], [X_val[:-1], X_val[1:]]

    if phenotype_prediction > 0.0:
        ys += [y_train[-Xs.shape[0] :, 0][:, np.newaxis]]
        yvals += [y_val[-Xvals.shape[0] :, 0][:, np.newaxis]]

        # Remove the used column (phenotype) from both y arrays
        y_train = y_train[:, 1:]
        y_val = y_val[:, 1:]

    if supervised_prediction > 0.0:
        ys += [y_train[-Xs.shape[0] :]]
        yvals += [y_val[-Xvals.shape[0] :]]

    # Cast to float32
    ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
    yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.cast(Xs, tf.float32), tuple(ys)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
        .with_options(options)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.cast(Xvals, tf.float32), tuple(yvals)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .with_options(options)
    )

    ae.compile()
    ae.fit(
        x=train_dataset,
        epochs=epochs,
        verbose=1,
        validation_data=val_dataset,
        callbacks=callbacks_,
    )

    if not os.path.exists(os.path.join(output_path, "trained_weights")):
        os.makedirs(os.path.join(output_path, "trained_weights"))

    if save_weights:
        ae.save_weights(
            os.path.join(
                "{}".format(output_path),
                "trained_weights",
                "{}_final_weights.h5".format(run_ID),
            )
        )

    return return_list


def tune_search(
    data: tf.data.Dataset,
    encoding_size: int,
    embedding_model: str,
    hypertun_trials: int,
    hpt_type: str,
    k: int,
    kl_warmup_epochs: int,
    loss: str,
    mmd_warmup_epochs: int,
    overlap_loss: float,
    next_sequence_prediction: float,
    phenotype_prediction: float,
    supervised_prediction: float,
    project_name: str,
    callbacks: List,
    batch_size: int = 64,
    n_epochs: int = 30,
    n_replicas: int = 1,
    outpath: str = "unsupervised_tuner_search",
) -> tuple:
    """

    Define the search space using keras-tuner and hyperband or bayesian optimization

    Args:
        data (tf.data.Dataset): Dataset object for training and validation.
        encoding_size (int): Size of the encoding layer.
        embedding_model (str): Embedding model to use. Must be one of "VQVAE" or "GMVAE".
        hypertun_trials (int): Number of hypertuning trials to run.
        hpt_type (str): Type of hypertuning to run. Must be one of "hyperband" or "bayesian".
        k (int): Number of clusters on the latent space.
        kl_warmup_epochs (int): Number of epochs to warmup KL loss. Only used if embedding_model is "GMVAE".
        loss (str): Loss function to use. Must be one of "mmd", "kl", or "overlap". Only used if embedding_model is "GMVAE".
        mmd_warmup_epochs (int): Number of epochs to warmup MMD loss. Only used if embedding_model is "GMVAE"
        overlap_loss (float): Weight of the overlap loss. Only used if embedding_model is "GMVAE".
        next_sequence_prediction (float): Weight of the next sequence prediction loss. Only used if embedding_model is "GMVAE".
        phenotype_prediction (float): Weight of the phenotype prediction loss. Only used if embedding_model is "GMVAE".
        supervised_prediction (float): Weight of the supervised prediction loss. Only used if embedding_model is "GMVAE".
        project_name (str): Name of the project.
        callbacks (List): List of callbacks to use.
        batch_size (int): Batch size to use.
        n_epochs (int): Maximum number of epochs to train for.
        n_replicas (int): Number of replicas to use.
        outpath (str): Path to save the results.

    Returns:
        best_hparams (dict): Dictionary of the best hyperparameters.
        best_run (str): Name of the best run.

    """

    X_train, y_train, X_val, y_val = data

    assert hpt_type in ["bayopt", "hyperband"], (
        "Invalid hyperparameter tuning framework. " "Select one of bayopt and hyperband"
    )

    if embedding_model == "VQVAE":
        hypermodel = deepof.hypermodels.VQVAE(
            input_shape=X_train.shape,
            latent_dim=encoding_size,
            n_components=k,
        )

    elif embedding_model == "GMVAE":
        hypermodel = deepof.hypermodels.GMVAE(
            input_shape=X_train.shape,
            batch_size=batch_size,
            latent_dim=encoding_size,
            kl_warmup_epochs=kl_warmup_epochs,
            loss=loss,
            mmd_warmup_epochs=mmd_warmup_epochs,
            n_components=k,
            overlap_loss=overlap_loss,
            next_sequence_prediction=next_sequence_prediction,
            phenotype_prediction=phenotype_prediction,
            supervised_prediction=supervised_prediction,
            supervised_features=(
                y_train.shape[1] if not phenotype_prediction else y_train.shape[1] - 1
            ),
        )

    tuner_objective = "val_loss"

    # noinspection PyUnboundLocalVariable
    hpt_params = {
        "hypermodel": hypermodel,
        "executions_per_trial": n_replicas,
        "objective": Objective(tuner_objective, direction="min"),
        "project_name": project_name,
        "tune_new_entries": True,
    }

    if hpt_type == "hyperband":
        tuner = Hyperband(
            directory=os.path.join(
                outpath, "HyperBandx_{}_{}".format(loss, str(date.today()))
            ),
            max_epochs=n_epochs,
            hyperband_iterations=hypertun_trials,
            factor=3,
            **hpt_params
        )
    else:
        tuner = BayesianOptimization(
            directory=os.path.join(
                outpath, "BayOpt_{}_{}".format(loss, str(date.today()))
            ),
            max_trials=hypertun_trials,
            **hpt_params
        )

    print(tuner.search_space_summary())

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    if next_sequence_prediction > 0.0:
        Xs, ys = X_train[:-1], [X_train[:-1], X_train[1:]]
        Xvals, yvals = X_val[:-1], [X_val[:-1], X_val[1:]]

    if phenotype_prediction > 0.0:
        ys += [y_train[-Xs.shape[0] :, 0]]
        yvals += [y_val[-Xvals.shape[0] :, 0]]

        # Remove the used column (phenotype) from both y arrays
        y_train = y_train[:, 1:]
        y_val = y_val[:, 1:]

    if supervised_prediction > 0.0:
        ys += [y_train[-Xs.shape[0] :]]
        yvals += [y_val[-Xvals.shape[0] :]]

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((Xs, tuple(ys)))
        .batch(batch_size, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((Xvals, tuple(yvals))).batch(
        batch_size, drop_remainder=True
    )

    # Convert data to tf.data.Dataset objects
    tuner.search(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_run = tuner.hypermodel.build(best_hparams)

    print(tuner.results_summary())

    return best_hparams, best_run

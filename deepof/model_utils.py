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
import tensorflow.keras.backend as K
from keras_tuner import BayesianOptimization, Hyperband, Objective
from scipy.spatial.distance import cdist
from tensorboard.plugins.hparams import api as hp

import deepof.hypermodels

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)


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


def compute_gram_loss(latent_means, weight=1.0, batch_size=64):  # pragma: no cover
    """

    Adds a penalty to the singular values of the Gram matrix of the latent means. It helps disentangle the latent
    space.
    Based on Variational Animal Motion Embedding (VAME) https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

    Args:
        latent_means: tensor containing the means of the latent distribution
        weight: weight of the Gram loss in the total loss function
        batch_size: batch size of the data to compute the Gram loss for.

    Returns:
        tf.Tensor: Gram loss

    """
    gram_matrix = (tf.transpose(latent_means) @ latent_means) / tf.cast(
        batch_size, tf.float32
    )
    s = tf.linalg.svd(gram_matrix, compute_uv=False)
    s = tf.sqrt(tf.maximum(s, 1e-9))
    return weight * tf.reduce_sum(s)


def far_uniform_initializer(shape: tuple, samples: int) -> tf.Tensor:
    """
    Initializes the prior latent means in a spread-out fashion,
    obtained by iteratively picking samples from a uniform distribution
    while maximizing the minimum euclidean distance between means.

    Args:
        shape: shape of the latent space.
        samples: number of initial candidates draw from the uniform distribution.

    Returns:
        tf.Tensor: the initialized latent means.

    """

    init_shape = (samples, shape[1])

    # Initialize latent mean candidates with a uniform distribution
    init_means = tf.keras.initializers.variance_scaling(
        scale=init_shape[0], distribution="uniform"
    )(init_shape)

    # Select the first random candidate as the first cluster mean
    final_samples, init_means = init_means[0][np.newaxis, :], init_means[1:]

    # Iteratively complete the mean set by adding new candidates, which maximize the minimum euclidean distance
    # with all existing means.
    for i in range(shape[0] - 1):
        max_dist = cdist(final_samples, init_means, metric="euclidean")
        max_dist = np.argmax(np.min(max_dist, axis=0))

        final_samples = np.concatenate(
            [final_samples, init_means[max_dist][np.newaxis, :]]
        )
        init_means = np.delete(init_means, max_dist, 0)

    return final_samples


def plot_lr_vs_loss(rates, losses):  # pragma: no cover
    """

    Plots learing rate versus the loss function of the model

    Args:
        rates (np.ndarray): array containing the learning rates to plot in the x-axis
        losses (np.ndarray): array containing the losses to plot in the y-axis

    """

    plt.plot(rates, losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


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


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    """

    Simple class that allows to grow learning rate exponentially during training.
    Used to trigger optimal learning rate search in deepof.train_utils.find_learning_rate.

    """

    def __init__(self, factor: float):
        """

        Initializes the exponential learning rate callback

        Args:
            factor(float): factor by which to multiply the learning rate

        """
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    # noinspection PyMethodOverriding
    def on_batch_end(self, batch: int, logs: dict):
        """

        This callback acts after processing each batch

        Args:
            batch: batch number
            logs (dict): dictionary containing the loss for the current batch

        """

        self.rates.append(K.get_value(self.model.optimizer.lr))
        if "total_loss" in logs.keys():
            self.losses.append(logs["total_loss"])
        else:
            self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_learning_rate(
    model, data, epochs=1, batch_size=32, min_rate=10 ** -8, max_rate=10 ** -1
):
    """

    Trains the provided model for an epoch with an exponentially increasing learning rate

    Args:
        model (tf.keras.Model): model to train
        data (tuple): training data
        epochs (int): number of epochs to train the model for
        batch_size (int): batch size to use for training
        min_rate (float): minimum learning rate to consider
        max_rate (float): maximum learning rate to consider

    Returns:
        float: learning rate that resulted in the lowest loss

    """

    init_weights = model.get_weights()
    iterations = len(data)
    factor = K.exp(K.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    model.fit(data, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def get_callbacks(
    gram_loss: float = 1.0,
    input_type: str = False,
    cp: bool = False,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """

    Generates callbacks used for model training.

    Args:
        gram_loss (float): Weight of the gram loss
        input_type (str): Input type to use for training
        cp (bool): Whether to use checkpointing or not
        logparam (dict): Dictionary containing the hyperparameters to log in tensorboard
        outpath (str): Path to the output directory
        run (int): Run number to use for checkpointing

    Returns:
        List[Union[Any]]: List of callbacks to be used for training

    """

    run_ID = "{}{}{}{}{}{}{}".format(
        "deepof_unsupervised_VQVAE_encodings",
        ("_input_type={}".format(input_type if input_type else "coords")),
        ("_gram_loss={}".format(gram_loss)),
        ("_encoding={}".format(logparam["latent_dim"]) if logparam is not None else ""),
        ("_k={}".format(logparam["n_components"]) if logparam is not None else ""),
        ("_run={}".format(run) if run else ""),
        ("_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")) if not run else ""),
    )

    log_dir = os.path.abspath(os.path.join(outpath, "fit", run_ID))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch=2,
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-8
    )

    callbacks = [run_ID, tensorboard_callback, reduce_lr_callback]

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


def log_hyperparameters():
    """

    Blueprint for hyperparameter and metric logging in tensorboard during hyperparameter tuning

    Returns:
        logparams (list): List containing the hyperparameters to log in tensorboard.
        metrics (list): List containing the metrics to log in tensorboard.

    """

    logparams = [
        hp.HParam(
            "latent_dim",
            hp.Discrete([2, 4, 6, 8, 12, 16]),
            display_name="latent_dim",
            description="encoding size dimensionality",
        ),
        hp.HParam(
            "n_components",
            hp.IntInterval(min_value=1, max_value=25),
            display_name="n_components",
            description="latent component number",
        ),
        hp.HParam(
            "gram_weight",
            hp.RealInterval(min_value=0.0, max_value=1.0),
            display_name="gram_weight",
            description="weight of the gram loss",
        ),
    ]

    metrics = [
        hp.Metric(
            "val_number_of_populated_clusters",
            display_name="number of populated clusters",
        ),
        hp.Metric(
            "val_reconstruction_loss",
            display_name="reconstruction loss",
        ),
        hp.Metric(
            "val_gram_loss",
            display_name="gram loss",
        ),
        hp.Metric(
            "val_vq_loss",
            display_name="vq loss",
        ),
        hp.Metric(
            "val_total_loss",
            display_name="total loss",
        ),
    ]

    return logparams, metrics


def autoencoder_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    latent_dim: int,
    epochs: int,
    hparams: dict,
    log_history: bool,
    log_hparams: bool,
    n_components: int,
    output_path: str,
    gram_loss: float,
    pretrained: str,
    save_checkpoints: bool,
    save_weights: bool,
    input_type: str,
    run: int = 0,
    strategy: tf.distribute.Strategy = "one_device",
):
    """

    Trains the specified autoencoder on the preprocessed data.

    Args:
        preprocessed_object (tuple): Tuple containing the preprocessed data.
        batch_size (int): Batch size to use for training.
        latent_dim (int): Encoding size to use for training.
        epochs (int): Number of epochs to train the autoencoder for.
        hparams (dict): Dictionary containing the hyperparameters to use for training.
        log_history (bool): Whether to log the history of the autoencoder.
        log_hparams (bool): Whether to log the hyperparameters used for training.
        n_components (int): Number of components to use for the VQVAE.
        output_path (str): Path to the output directory.
        gram_loss (float): Weight of the gram loss, which adds a regularization term to VQVAE models which
        penalizes the correlation between the dimensions in the latent space.
        pretrained (str): Path to the pretrained weights to use for the autoencoder.
        save_checkpoints (bool): Whether to save checkpoints during training.
        save_weights (bool): Whether to save the weights of the autoencoder after training.
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

    # Set options for tf.data.Datasets
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    # Defines hyperparameters to log on tensorboard (useful for keeping track of different models)
    logparam = {
        "latent_dim": latent_dim,
        "n_components": n_components,
        "gram_weight": gram_loss,
    }

    # Load callbacks
    run_ID, *cbacks = get_callbacks(
        gram_loss=gram_loss,
        input_type=input_type,
        cp=save_checkpoints,
        logparam=logparam,
        outpath=output_path,
        run=run,
    )
    if not log_history:
        cbacks = cbacks[1:]

    # Build model
    with strategy.scope():
        ae_full_model = deepof.models.VQVAE(
            architecture_hparams=hparams,
            input_shape=X_train.shape,
            latent_dim=latent_dim,
            n_components=n_components,
            reg_gram=gram_loss,
        )
        ae_full_model.optimizer = tf.keras.optimizers.Nadam(
            learning_rate=1e-4, clipvalue=0.75
        )
        encoder, decoder, quantizer, ae = (
            ae_full_model.encoder,
            ae_full_model.decoder,
            ae_full_model.quantizer,
            ae_full_model.vqvae,
        )
        return_list = (encoder, decoder, quantizer, ae)

    if pretrained:
        # If pretrained models are specified, load weights and return
        ae.load_weights(pretrained)
        return return_list

    callbacks_ = cbacks + [
        CustomStopper(
            monitor="val_total_loss",
            mode="min",
            patience=15,
            restore_best_weights=True,
            start_epoch=15,
        ),
    ]

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    # Cast to float32
    ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
    yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.cast(Xs, tf.float32), tuple(ys)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.cast(Xvals, tf.float32), tuple(yvals)))
        .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )

    ae_full_model.compile(optimizer=ae_full_model.optimizer)
    ae_full_model.fit(
        x=train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks_,
        verbose=1,
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

        # Logs hyperparameters to tensorboard
        if log_hparams:
            logparams, metrics = log_hyperparameters()

            tb_writer = tf.summary.create_file_writer(
                os.path.abspath(os.path.join(output_path, "hparams", run_ID))
            )
            with tb_writer.as_default():
                # Configure hyperparameter logging in tensorboard
                hp.hparams_config(
                    hparams=logparams,
                    metrics=metrics,
                )
                hp.hparams(logparam)  # Log hyperparameters
                # Log metrics
                tf.summary.scalar(
                    "val_number_of_populated_clusters",
                    ae_full_model.history.history["val_number_of_populated_clusters"][
                        -1
                    ],
                    step=0,
                )
                tf.summary.scalar(
                    "val_reconstruction_loss",
                    ae_full_model.history.history["val_reconstruction_loss"][-1],
                    step=0,
                )
                tf.summary.scalar(
                    "val_gram_loss",
                    ae_full_model.history.history["val_gram_loss"][-1],
                    step=0,
                )
                tf.summary.scalar(
                    "val_vq_loss",
                    ae_full_model.history.history["val_vq_loss"][-1],
                    step=0,
                )
                tf.summary.scalar(
                    "val_total_loss",
                    ae_full_model.history.history["val_total_loss"][-1],
                    step=0,
                )

    return return_list


def tune_search(
    data: tf.data.Dataset,
    encoding_size: int,
    hypertun_trials: int,
    hpt_type: str,
    k: int,
    gram_loss: float,
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
        hypertun_trials (int): Number of hypertuning trials to run.
        hpt_type (str): Type of hypertuning to run. Must be one of "hyperband" or "bayesian".
        k (int): Number of clusters on the latent space.
        gram_loss (float): Weight of the gram loss, which enforces disentanglement by penalizing the correlation
        between dimensions in the latent space.
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

    hypermodel = deepof.hypermodels.VQVAE(
        input_shape=X_train.shape,
        latent_dim=encoding_size,
        n_components=k,
        reg_gram=gram_loss,
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
                outpath, "HyperBandx_VQVAE_{}".format(str(date.today()))
            ),
            max_epochs=n_epochs,
            hyperband_iterations=hypertun_trials,
            factor=3,
            **hpt_params
        )
    else:
        tuner = BayesianOptimization(
            directory=os.path.join(
                outpath, "BayOpt_VQVAE_{}".format(str(date.today()))
            ),
            max_trials=hypertun_trials,
            **hpt_params
        )

    print(tuner.search_space_summary())

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

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

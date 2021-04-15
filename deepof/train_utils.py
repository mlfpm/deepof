# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Simple utility functions used in deepof example scripts. These are not part of the main package

"""

import json
import os
from datetime import date, datetime
from typing import Tuple, Union, Any, List

import numpy as np
import tensorflow as tf
from kerastuner import BayesianOptimization, Hyperband
from kerastuner_tensorboard_logger import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from tensorboard.plugins.hparams import api as hp

import deepof.hypermodels
import deepof.model_utils

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    """ Custom early stopping callback. Prevents the model from stopping before warmup is over """

    def __init__(self, start_epoch, *args, **kwargs):
        super(CustomStopper, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def get_config(self):  # pragma: no cover
        """Updates callback metadata"""

        config = super().get_config().copy()
        config.update({"start_epoch": self.start_epoch})
        return config

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def load_treatments(train_path):
    """Loads a dictionary containing the treatments per individual,
    to be loaded as metadata in the coordinates class"""
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
    variational: bool,
    phenotype_prediction: float,
    next_sequence_prediction: float,
    rule_based_prediction: float,
    loss: str,
    X_val: np.array = None,
    input_type: str = False,
    cp: bool = False,
    reg_cat_clusters: bool = False,
    reg_cluster_variance: bool = False,
    entropy_samples: int = 15000,
    entropy_knn: int = 100,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """Generates callbacks for model training, including:
    - run_ID: run name, with coarse parameter details;
    - tensorboard_callback: for real-time visualization;
    - cp_callback: for checkpoint saving;
    - onecycle: for learning rate scheduling;
    - entropy: neighborhood entropy in the latent space;
    """

    latreg = "none"
    if reg_cat_clusters and not reg_cluster_variance:
        latreg = "categorical"
    elif reg_cluster_variance and not reg_cat_clusters:
        latreg = "variance"
    elif reg_cat_clusters and reg_cluster_variance:
        latreg = "categorical+variance"

    run_ID = "{}{}{}{}{}{}{}_{}".format(
        ("GMVAE" if variational else "AE"),
        ("_input_type={}".format(input_type) if input_type else "coords"),
        ("_NextSeqPred={}".format(next_sequence_prediction) if variational else ""),
        ("_PhenoPred={}".format(phenotype_prediction) if variational else ""),
        ("_RuleBasedPred={}".format(rule_based_prediction) if variational else ""),
        ("_loss={}".format(loss) if variational else ""),
        ("_encoding={}".format(logparam["encoding"]) if logparam is not None else ""),
        ("_k={}".format(logparam["k"]) if logparam is not None else ""),
        ("_latreg={}".format(latreg)),
        ("_entknn={}".format(entropy_knn)),
        ("_run={}".format(run) if run else ""),
        ("_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")) if not run else ""),
    )

    log_dir = os.path.abspath(os.path.join(outpath, "fit", run_ID))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        profile_batch=2,
    )

    entropy = deepof.model_utils.neighbor_latent_entropy(
        encoding_dim=logparam["encoding"],
        k=entropy_knn,
        samples=entropy_samples,
        validation_data=X_val,
        log_dir=os.path.join(outpath, "metrics", run_ID),
        variational=variational,
    )

    onecycle = deepof.model_utils.one_cycle_scheduler(
        X_train.shape[0] // batch_size * 250,
        max_rate=0.005,
        log_dir=os.path.join(outpath, "metrics", run_ID),
    )

    callbacks = [run_ID, tensorboard_callback, entropy, onecycle]

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
    """Blueprint for hyperparameter and metric logging in tensorboard during hyperparameter tuning"""

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


# noinspection PyUnboundLocalVariable
def tensorboard_metric_logging(
    run_dir: str,
    hpms: Any,
    ae: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    next_sequence_prediction: float,
    phenotype_prediction: float,
    rule_based_prediction: float,
    rec: str,
):
    """Autoencoder metric logging in tensorboard"""

    outputs = ae.predict(X_val)
    idx_generator = (idx for idx in range(len(outputs)))

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hpms)  # record the values used in this trial
        idx = next(idx_generator)

        val_mae = tf.reduce_mean(
            tf.keras.metrics.mean_absolute_error(y_val[idx], outputs[idx])
        )
        val_mse = tf.reduce_mean(
            tf.keras.metrics.mean_squared_error(y_val[idx], outputs[idx])
        )
        tf.summary.scalar("val_{}mae".format(rec), val_mae, step=1)
        tf.summary.scalar("val_{}mse".format(rec), val_mse, step=1)

        if next_sequence_prediction:
            idx = next(idx_generator)
            pred_mae = tf.reduce_mean(
                tf.keras.metrics.mean_absolute_error(y_val[idx], outputs[idx])
            )
            pred_mse = tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(y_val[idx], outputs[idx])
            )
            tf.summary.scalar(
                "val_next_sequence_prediction_mae".format(rec), pred_mae, step=1
            )
            tf.summary.scalar(
                "val_next_sequence_prediction_mse".format(rec), pred_mse, step=1
            )

        if phenotype_prediction:
            idx = next(idx_generator)
            pheno_acc = tf.keras.metrics.binary_accuracy(
                y_val[idx], tf.squeeze(outputs[idx])
            )
            pheno_auc = tf.keras.metrics.AUC()
            pheno_auc.update_state(y_val[idx], outputs[idx])
            pheno_auc = pheno_auc.result().numpy()

            tf.summary.scalar("phenotype_prediction_accuracy", pheno_acc, step=1)
            tf.summary.scalar("phenotype_prediction_auc", pheno_auc, step=1)

        if rule_based_prediction:
            idx = next(idx_generator)
            rules_mae = tf.reduce_mean(
                tf.keras.metrics.mean_absolute_error(y_val[idx], outputs[idx])
            )
            rules_mse = tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(y_val[idx], outputs[idx])
            )
            tf.summary.scalar("val_prediction_mae".format(rec), rules_mae, step=1)
            tf.summary.scalar("val_prediction_mse".format(rec), rules_mse, step=1)


def autoencoder_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    encoding_size: int,
    epochs: int,
    hparams: dict,
    kl_warmup: int,
    log_history: bool,
    log_hparams: bool,
    loss: str,
    mmd_warmup: int,
    montecarlo_kl: int,
    n_components: int,
    output_path: str,
    next_sequence_prediction: float,
    phenotype_prediction: float,
    rule_based_prediction: float,
    pretrained: str,
    save_checkpoints: bool,
    save_weights: bool,
    variational: bool,
    reg_cat_clusters: bool,
    reg_cluster_variance: bool,
    entropy_samples: int,
    entropy_knn: int,
    input_type: str,
    run: int = 0,
):
    """Implementation function for deepof.data.coordinates.deep_unsupervised_embedding"""

    # Load data
    X_train, y_train, X_val, y_val = preprocessed_object

    # To avoid stability issues
    tf.keras.backend.clear_session()

    # Defines what to log on tensorboard (useful for trying out different models)
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
        variational=variational,
        phenotype_prediction=phenotype_prediction,
        next_sequence_prediction=next_sequence_prediction,
        rule_based_prediction=rule_based_prediction,
        loss=loss,
        input_type=input_type,
        X_val=(X_val if X_val.shape != (0,) else None),
        cp=save_checkpoints,
        reg_cat_clusters=reg_cat_clusters,
        reg_cluster_variance=reg_cluster_variance,
        entropy_samples=entropy_samples,
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

    # Gets the number of rule-based features
    try:
        rule_based_features = (
            y_train.shape[1] if not phenotype_prediction else y_train.shape[1] - 1
        )
    except IndexError:
        rule_based_features = 0

    # Build models
    if not variational:
        encoder, decoder, ae = deepof.models.SEQ_2_SEQ_AE(
            ({} if hparams is None else hparams)
        ).build(X_train.shape)
        return_list = (encoder, decoder, ae)

    else:
        (
            encoder,
            generator,
            grouper,
            ae,
            prior,
            posterior,
        ) = deepof.models.SEQ_2_SEQ_GMVAE(
            architecture_hparams=({} if hparams is None else hparams),
            batch_size=batch_size,
            compile_model=True,
            encoding=encoding_size,
            kl_warmup_epochs=kl_warmup,
            loss=loss,
            mmd_warmup_epochs=mmd_warmup,
            montecarlo_kl=montecarlo_kl,
            neuron_control=False,
            number_of_components=n_components,
            overlap_loss=False,
            next_sequence_prediction=next_sequence_prediction,
            phenotype_prediction=phenotype_prediction,
            rule_based_prediction=rule_based_prediction,
            rule_based_features=rule_based_features,
            reg_cat_clusters=reg_cat_clusters,
            reg_cluster_variance=reg_cluster_variance,
        ).build(
            X_train.shape
        )
        return_list = (encoder, generator, grouper, ae)

    if pretrained:
        # If pretrained models are specified, load weights and return
        ae.load_weights(pretrained)
        return return_list

    else:
        if not variational:

            ae.fit(
                x=X_train,
                y=X_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_val, X_val),
                callbacks=cbacks
                + [
                    CustomStopper(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                        start_epoch=max(kl_warmup, mmd_warmup),
                    ),
                ],
            )

            if save_weights:
                ae.save_weights("{}_final_weights.h5".format(run_ID))

        else:

            callbacks_ = cbacks + [
                CustomStopper(
                    monitor="val_loss",
                    patience=5,
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
                ys += [y_train[-Xs.shape[0] :, 0]]
                yvals += [y_val[-Xs.shape[0] :, 0]]

                # Remove the used column (phenotype) from both y arrays
                y_train = y_train[:, 1:]
                y_val = y_val[:, 1:]

            if rule_based_prediction > 0.0:
                ys += [y_train[-Xs.shape[0] :]]
                yvals += [y_val[-Xs.shape[0] :]]

            ae.fit(
                x=Xs,
                y=ys,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=(
                    Xvals,
                    yvals,
                ),
                callbacks=callbacks_,
            )

            if not os.path.exists(os.path.join(output_path, "trained_weights")):
                os.makedirs("trained_weights")

            if save_weights:
                ae.save_weights(
                    os.path.join(
                        "{}".format(output_path),
                        "trained_weights",
                        "{}_final_weights.h5".format(run_ID),
                    )
                )

            if log_hparams:
                # Logparams to tensorboard
                tensorboard_metric_logging(
                    run_dir=os.path.join(output_path, "hparams", run_ID),
                    hpms=logparam,
                    ae=ae,
                    X_val=Xvals,
                    y_val=yvals,
                    next_sequence_prediction=next_sequence_prediction,
                    phenotype_prediction=phenotype_prediction,
                    rule_based_prediction=rule_based_prediction,
                    rec=rec,
                )

    return return_list


def tune_search(
    data: List[np.array],
    encoding_size: int,
    hypertun_trials: int,
    hpt_type: str,
    hypermodel: str,
    k: int,
    kl_warmup_epochs: int,
    loss: str,
    mmd_warmup_epochs: int,
    overlap_loss: float,
    next_sequence_prediction: float,
    phenotype_prediction: float,
    rule_based_prediction: float,
    project_name: str,
    callbacks: List,
    n_epochs: int = 30,
    n_replicas: int = 1,
    outpath: str = ".",
) -> Union[bool, Tuple[Any, Any]]:
    """Define the search space using keras-tuner and bayesian optimization

    Parameters:
        - train (np.array): dataset to train the model on
        - test (np.array): dataset to validate the model on
        - hypertun_trials (int): number of Bayesian optimization iterations to run
        - hpt_type (str): specify one of Bayesian Optimization (bayopt) and Hyperband (hyperband)
        - hypermodel (str): hypermodel to load. Must be one of S2SAE (plain autoencoder)
        or S2SGMVAE (Gaussian Mixture Variational autoencoder).
        - k (int) number of components of the Gaussian Mixture
        - loss (str): one of [ELBO, MMD, ELBO+MMD]
        - overlap_loss (float): assigns as weight to an extra loss term which
        penalizes overlap between GM components
        - phenotype_class (float): adds an extra regularizing neural network to the model,
        which tries to predict the phenotype of the animal from which the sequence comes
        - predictor (float): adds an extra regularizing neural network to the model,
        which tries to predict the next frame from the current one
        - project_name (str): ID of the current run
        - callbacks (list): list of callbacks for the training loop
        - n_epochs (int): optional. Number of epochs to train each run for
        - n_replicas (int): optional. Number of replicas per parameter set. Higher values
         will yield more robust results, but will affect performance severely

    Returns:
        - best_hparams (dict): dictionary with the best retrieved hyperparameters
        - best_run (tf.keras.Model): trained instance of the best model found

    """

    X_train, y_train, X_val, y_val = data

    assert hpt_type in ["bayopt", "hyperband"], (
        "Invalid hyperparameter tuning framework. " "Select one of bayopt and hyperband"
    )

    if hypermodel == "S2SAE":  # pragma: no cover
        assert (
            next_sequence_prediction == 0.0 and phenotype_prediction == 0.0
        ), "Prediction branches are only available for variational models. See documentation for more details"
        batch_size = 1
        hypermodel = deepof.hypermodels.SEQ_2_SEQ_AE(input_shape=X_train.shape)

    elif hypermodel == "S2SGMVAE":
        batch_size = 64
        hypermodel = deepof.hypermodels.SEQ_2_SEQ_GMVAE(
            input_shape=X_train.shape,
            encoding=encoding_size,
            kl_warmup_epochs=kl_warmup_epochs,
            loss=loss,
            mmd_warmup_epochs=mmd_warmup_epochs,
            number_of_components=k,
            overlap_loss=overlap_loss,
            next_sequence_prediction=next_sequence_prediction,
            phenotype_prediction=phenotype_prediction,
            rule_based_prediction=rule_based_prediction,
            rule_based_features=(
                y_train.shape[1] if not phenotype_prediction else y_train.shape[1] - 1
            ),
        )

    else:
        return False

    hpt_params = {
        "hypermodel": hypermodel,
        "executions_per_trial": n_replicas,
        "logger": TensorBoardLogger(
            metrics=["val_mae"], logdir=os.path.join(outpath, "logged_hparams")
        ),
        "objective": "val_mae",
        "project_name": project_name,
        "tune_new_entries": True,
    }

    if hpt_type == "hyperband":
        tuner = Hyperband(
            directory=os.path.join(
                outpath, "HyperBandx_{}_{}".format(loss, str(date.today()))
            ),
            max_epochs=35,
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

    Xs, ys = [X_train], [X_train]
    Xvals, yvals = [X_val], [X_val]

    if next_sequence_prediction > 0.0:
        Xs, ys = X_train[:-1], [X_train[:-1], X_train[1:]]
        Xvals, yvals = X_val[:-1], [X_val[:-1], X_val[1:]]

    if phenotype_prediction > 0.0:
        ys += [y_train[:, 0]]
        yvals += [y_val[:, 0]]

        # Remove the used column (phenotype) from both y arrays
        y_train = y_train[:, 1:]
        y_val = y_val[:, 1:]

    if rule_based_prediction > 0.0:
        ys += [y_train]
        yvals += [y_val]

    tuner.search(
        Xs,
        ys,
        epochs=n_epochs,
        validation_data=(Xvals, yvals),
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_run = tuner.hypermodel.build(best_hparams)

    print(tuner.results_summary())

    return best_hparams, best_run

# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Simple utility functions used in deepof example scripts. These are not part of the main package

"""
from datetime import datetime

from kerastuner import BayesianOptimization
from typing import Tuple, Union, Any, List
import deepof.hypermodels
import deepof.model_utils
import keras
import numpy as np
import os
import pickle
import tensorflow as tf


def load_hparams(hparams):
    """Loads hyperparameters from a custom dictionary pickled on disc.
    Thought to be used with the output of hyperparameter_tuning.py"""

    if hparams is not None:
        with open(hparams, "rb") as handle:
            hparams = pickle.load(handle)
    else:
        hparams = {
            "units_conv": 256,
            "units_lstm": 256,
            "units_dense2": 64,
            "dropout_rate": 0.25,
            "encoding": 16,
            "learning_rate": 1e-3,
        }
    return hparams


def load_treatments(train_path):
    """Loads a dictionary containing the treatments per individual,
    to be loaded as metadata in the coordinates class"""
    try:
        with open(
            os.path.join(
                train_path,
                [i for i in os.listdir(train_path) if i.endswith(".pkl")][0],
            ),
            "rb",
        ) as handle:
            treatment_dict = pickle.load(handle)
    except IndexError:
        treatment_dict = None

    return treatment_dict


def get_callbacks(
    X_train: np.array,
    batch_size: int,
    variational: bool,
    predictor: float,
    k: int,
    loss: str,
    kl_wu: int,
    mmd_wu: int,
) -> Tuple:
    """Generates callbacks for model training, including:
        - run_ID: run name, with coarse parameter details;
        - tensorboard_callback: for real-time visualization;
        - cp_callback: for checkpoint saving,
        - onecycle: for learning rate scheduling"""

    run_ID = "{}{}{}{}{}{}_{}".format(
        ("GMVAE" if variational else "AE"),
        ("P" if predictor > 0 and variational else ""),
        ("_components={}".format(k) if variational else ""),
        ("_loss={}".format(loss) if variational else ""),
        ("_kl_warmup={}".format(kl_wu) if variational else ""),
        ("_mmd_warmup={}".format(mmd_wu) if variational else ""),
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    log_dir = os.path.abspath("logs/fit/{}".format(run_ID))
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch=2,
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        "./logs/checkpoints/" + run_ID + "/cp-{epoch:04d}.ckpt",
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        save_freq="epoch",
    )

    onecycle = deepof.model_utils.one_cycle_scheduler(
        X_train.shape[0] // batch_size * 250, max_rate=0.005,
    )

    return run_ID, tensorboard_callback, cp_callback, onecycle


def tune_search(
    train: np.array,
    test: np.array,
    bayopt_trials: int,
    hypermodel: str,
    k: int,
    kl_wu: int,
    loss: str,
    mmd_wu: int,
    overlap_loss: float,
    predictor: float,
    project_name: str,
    callbacks: List,
) -> Union[bool, Tuple[Any, Any]]:
    """Define the search space using keras-tuner and bayesian optimization

        Parameters:
            - train (np.array): dataset to train the model on
            - test (np.array): dataset to validate the model on
            - bayopt_trials (int): number of Bayesian optimization iterations to run
            - hypermodel (str): hypermodel to load. Must be one of S2SAE (plain autoencoder) or
            S2SGMVAE (Gaussian Mixture Variational autoencoder).
            - k (int) number of components of the Gaussian Mixture
            - kl_wu (int): number of epochs for KL divergence warm up
            - loss (str): one of [ELBO, MMD, ELBO+MMD]
            - mmd_wu (int): number of epochs for MMD warm up
            - overlap_loss (float): assigns as weight to an extra loss term which
            penalizes overlap between GM components
            - predictor (float): adds an extra regularizing neural network to the model,
            which tries to predict the next frame from the current one
            - project_name (str): ID of the current run
            - callbacks (list): list of callbacks for the training loop

        Returns:
            - best_hparams (dict): dictionary with the best retrieved hyperparameters
            - best_run (tf.keras.Model): trained instance of the best model found

    """

    tensorboard_callback, cp_callback, onecycle = callbacks

    if hypermodel == "S2SAE":
        hypermodel = deepof.hypermodels.SEQ_2_SEQ_AE(input_shape=train.shape)

    elif hypermodel == "S2SGMVAE":
        hypermodel = deepof.hypermodels.SEQ_2_SEQ_GMVAE(
            input_shape=train.shape,
            loss=loss,
            number_of_components=k,
            kl_warmup_epochs=kl_wu,
            mmd_warmup_epochs=mmd_wu,
            predictor=predictor,
            overlap_loss=overlap_loss,
        )
    else:
        return False

    tuner = BayesianOptimization(
        hypermodel,
        max_trials=bayopt_trials,
        executions_per_trial=1,
        objective="val_mae",
        seed=42,
        directory="BayesianOptx",
        project_name=project_name,
    )

    print(tuner.search_space_summary())

    tuner.search(
        train,
        train,
        epochs=30,
        validation_data=(test, test),
        verbose=1,
        batch_size=256,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping("val_mae", patience=5),
        ],
    )

    print(tuner.results_summary())
    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_run = tuner.hypermodel.build(best_hparams)

    return best_hparams, best_run


# TODO:
#    - load_treatments should be part of the main data module. If available in the main directory,
#    a table (preferrable in csv) should be loaded as metadata of the coordinates automatically.
#    This becomes particularly important por the supervised models that include phenotype classification
#    alongside the encoding.

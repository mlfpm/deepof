# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Simple utility functions used in deepof example scripts. These are not part of the main package

"""

import os
import pickle


def load_hparams(hparams, encoding):
    """Loads hyperparameters from a custom dictionary pickled on disc.
    Thought to be used with the output of hyperparameter_tuning.py"""

    if hparams is not None:
        with open(hparams, "rb") as handle:
            hparams = pickle.load(handle)
        hparams["encoding"] = encoding
    else:
        hparams = {
            "units_conv": 256,
            "units_lstm": 256,
            "units_dense2": 64,
            "dropout_rate": 0.25,
            "encoding": encoding,
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
                [i for i in os.listdir(train_path) if i.endswith(".pickle")][0],
            ),
            "rb",
        ) as handle:
            treatment_dict = pickle.load(handle)
    except IndexError:
        treatment_dict = None

    return treatment_dict


# TODO:
#    - load_treatments should be part of the main data module. If available in the main directory,
#    a table (preferrable in csv) should be loaded as metadata of the coordinates automatically.
#    This becomes particularly important por the supervised models that include phenotype classification
#    alongside the encoding.

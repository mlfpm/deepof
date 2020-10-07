# @author lucasmiranda42

from datetime import datetime
from deepof.data import *
from deepof.hypermodels import *
from .example_utils import *
from kerastuner import BayesianOptimization
from tensorflow import keras
import argparse
import os, pickle

parser = argparse.ArgumentParser(
    description="hyperparameter tuning for DeepOF autoencoder models"
)

parser.add_argument("--train_path", "-tp", help="set training set path", type=str)
parser.add_argument(
    "--components",
    "-k",
    help="set the number of components for the MMVAE(P) model. Defaults to 1",
    type=int,
    default=1,
)
parser.add_argument(
    "--input-type",
    "-d",
    help="Select an input type for the autoencoder hypermodels. \
    It must be one of coords, dists, angles, coords+dist, coords+angle or coords+dist+angle",
    type=str,
    default="coords",
)
parser.add_argument(
    "--bayopt",
    "-n",
    help="sets the number of Bayesian optimization iterations to run. Default is 25",
    default=25,
    type=int,
)
parser.add_argument(
    "--hypermodel",
    "-m",
    help="Selects which hypermodel to use. It must be one of S2SAE, S2SVAE, S2SVAE-ELBO, S2SVAE-MMD, "
    "S2SVAEP, S2SVAEP-ELBO and S2SVAEP-MMD. Please refer to the documentation for details on each option.",
    default="S2SVAE",
    type=str,
)

args = parser.parse_args()
train_path = os.path.abspath(args.train_path)
val_path = os.path.abspath(args.val_path)
input_type = args.input_type
bayopt_trials = args.bayopt
hyp = args.hypermodel
k = args.components

if not train_path:
    raise ValueError("Set a valid data path for the training to run")
if not val_path:
    raise ValueError("Set a valid data path for the validation to run")
assert input_type in [
    "coords",
    "dists",
    "angles",
    "coords+dist",
    "coords+angle",
    "coords+dist+angle",
], "Invalid input type. Type python hyperparameter_tuning.py -h for help."
assert hyp in [
    "S2SAE",
    "S2SGMVAE",
], "Invalid hypermodel. Type python hyperparameter_tuning.py -h for help."

log_dir = os.path.abspath(
    "logs/fit/{}_{}".format(hyp, datetime.now().strftime("%Y%m%d-%H%M%S"))
)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

treatment_dict = load_treatments(train_path)

project_coords = project(
    path=train_path,  # Path where to find the required files
    smooth_alpha=0.85,  # Alpha value for exponentially weighted smoothing
    arena="circular",  # Type of arena used in the experiments
    arena_dims=tuple([380]),  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
    exp_conditions=treatment_dict,
).run(verbose=True)

# Coordinates for training data
coords = project_coords.get_coords(center="Center", align="Spine_1", align_inplace=True)
distances = project_coords.get_distances()
angles = project_coords.get_angles()
coords_distances = merge_tables(coords, distances)
coords_angles = merge_tables(coords, angles)
dists_angles = merge_tables(distances, angles)
coords_dist_angles = merge_tables(coords, distances, angles)


def batch_preprocess(tab_dict):
    """Returns a preprocessed instance of the input table_dict object"""

    return tab_dict.preprocess(
        window_size=window_size,
        window_step=window_step,
        scale="standard",
        conv_filter=gaussian_filter,
        sigma=1,
        test_videos=val_num,
        shuffle=True,
    )


input_dict_train = {
    "coords": coords,
    "dists": distances,
    "angles": angles,
    "coords+dist": coords_distances,
    "coords+angle": coords_angles,
    "dists+angle": dists_angles,
    "coords+dist+angle": coords_dist_angles,
}

print("Preprocessing data...")
for key, value in input_dict_train.items():
    input_dict_train[key] = batch_preprocess(value)
print("Done!")


def tune_search(train, test, project_name, hyp):
    """Define the search space using keras-tuner and bayesian optimization"""
    if hyp == "S2SAE":
        hypermodel = SEQ_2_SEQ_AE(input_shape=train.shape)
    elif hyp == "S2SGMVAE":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape,
            loss="ELBO+MMD",
            predictor=False,
            number_of_components=k,
        ).build()
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
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hyperparameters)

    return best_hyperparameters, best_model


# Runs hyperparameter tuning with the specified parameters and saves the results
best_hyperparameters, best_model = tune_search(
    input_dict_train[input_type],
    input_dict_val[input_type],
    "{}-based_{}_BAYESIAN_OPT".format(input_type, hyp),
    hyp=hyp,
)

# Saves a compiled, untrained version of the best model
best_model.build(input_dict_train[input_type].shape)
best_model.save("{}-based_{}_BAYESIAN_OPT.h5".format(input_type, hyp), save_format="tf")

# Saves the best hyperparameters
with open(
    "{}-based_{}_BAYESIAN_OPT_params.pickle".format(input_type, hyp), "wb"
) as handle:
    pickle.dump(best_hyperparameters.values, handle, protocol=pickle.HIGHEST_PROTOCOL)

# @author lucasmiranda42

from datetime import datetime
from source.preprocess import *
from source.hypermodels import *
from kerastuner import BayesianOptimization
from tensorflow import keras
import argparse
import os, pickle

parser = argparse.ArgumentParser(
    description="hyperparameter tuning for DeepOF autoencoder models"
)

parser.add_argument("--train_path", "-tp", help="set training set path", type=str)
parser.add_argument("--val_path", "-vp", help="set validation set path", type=str)
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
    "S2SVAE",
    "S2SVAE-ELBO",
    "S2SVAE-MMD",
    "S2SVAEP",
    "S2SVAEP-ELBO",
    "S2SVAEP-MMD",
], "Invalid hypermodel. Type python hyperparameter_tuning.py -h for help."

log_dir = os.path.abspath(
    "logs/fit/{}_{}".format(hyp, datetime.now().strftime("%Y%m%d-%H%M%S"))
)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

with open(
    os.path.abspath(train_path + "/DLC_social_1_exp_conditions.pickle"), "rb"
) as handle:
    Treatment_dict = pickle.load(handle)

# Which angles to compute?
bp_dict = {
    "B_Nose": ["B_Left_ear", "B_Right_ear"],
    "B_Left_ear": ["B_Nose", "B_Right_ear", "B_Center", "B_Left_flank"],
    "B_Right_ear": ["B_Nose", "B_Left_ear", "B_Center", "B_Right_flank"],
    "B_Center": [
        "B_Left_ear",
        "B_Right_ear",
        "B_Left_flank",
        "B_Right_flank",
        "B_Tail_base",
    ],
    "B_Left_flank": ["B_Left_ear", "B_Center", "B_Tail_base"],
    "B_Right_flank": ["B_Right_ear", "B_Center", "B_Tail_base"],
    "B_Tail_base": ["B_Center", "B_Left_flank", "B_Right_flank"],
}

DLC_social_1 = project(
    path=train_path,  # Path where to find the required files
    smooth_alpha=0.85,  # Alpha value for exponentially weighted smoothing
    distances=[
        "B_Center",
        "B_Nose",
        "B_Left_ear",
        "B_Right_ear",
        "B_Left_flank",
        "B_Right_flank",
        "B_Tail_base",
    ],
    ego=False,
    angles=True,
    connectivity=bp_dict,
    arena="circular",  # Type of arena used in the experiments
    arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
    exp_conditions=Treatment_dict,
)

DLC_social_2 = project(
    path=val_path,  # Path where to find the required files
    smooth_alpha=0.85,  # Alpha value for exponentially weighted smoothing
    distances=[
        "B_Center",
        "B_Nose",
        "B_Left_ear",
        "B_Right_ear",
        "B_Left_flank",
        "B_Right_flank",
        "B_Tail_base",
    ],
    ego=False,
    angles=True,
    connectivity=bp_dict,
    arena="circular",  # Type of arena used in the experiments
    arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
)

DLC_social_1_coords = DLC_social_1.run(verbose=True)
DLC_social_2_coords = DLC_social_2.run(verbose=True)

# Coordinates for training data
coords1 = DLC_social_1_coords.get_coords()
distances1 = DLC_social_1_coords.get_distances()
angles1 = DLC_social_1_coords.get_angles()
coords_distances1 = merge_tables(coords1, distances1)
coords_angles1 = merge_tables(coords1, angles1)
coords_dist_angles1 = merge_tables(coords1, distances1, angles1)

# Coordinates for validation data
coords2 = DLC_social_2_coords.get_coords()
distances2 = DLC_social_2_coords.get_distances()
angles2 = DLC_social_2_coords.get_angles()
coords_distances2 = merge_tables(coords2, distances2)
coords_angles2 = merge_tables(coords2, angles2)
coords_dist_angles2 = merge_tables(coords2, distances2, angles2)


input_dict_train = {
    "coords": coords1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
    "dists": distances1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
    "angles": angles1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
    "coords+dist": coords_distances1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
    "coords+angle": coords_angles1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
    "coords+dist+angle": coords_dist_angles1.preprocess(
        window_size=11, window_step=10, scale=True, random_state=42, filter="gauss"
    ),
}

input_dict_val = {
    "coords": coords2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
    "dists": distances2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
    "angles": angles2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
    "coords+dist": coords_distances2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
    "coords+angle": coords_angles2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
    "coords+dist+angle": coords_dist_angles2.preprocess(
        window_size=11, window_step=1, scale=True, random_state=42, filter="gauss"
    ),
}

for input in input_dict_train.keys():
    print("{} train shape: {}".format(input, input_dict_train[input].shape))
    print("{} validation shape: {}".format(input, input_dict_val[input].shape))
    print()


def tune_search(train, test, project_name, hyp):
    """Define the search space using keras-tuner and bayesian optimization"""
    if hyp == "S2SAE":
        hypermodel = SEQ_2_SEQ_AE(input_shape=train.shape)
    elif hyp == "S2SVAE":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape,
            loss="ELBO+MMD",
            predictor=False,
            number_of_components=k,
        )
    elif hyp == "S2SVAE-MMD":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape, loss="MMD", predictor=False, number_of_components=k
        )
    elif hyp == "S2SVAE-ELBO":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape,
            loss="ELBO",
            predictor=False,
            number_of_components=k,
        )
    elif hyp == "S2SVAEP":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape,
            loss="ELBO+MMD",
            predictor=True,
            number_of_components=k,
        )
    elif hyp == "S2SVAEP-MMD":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape, loss="MMD", predictor=True, number_of_components=k
        )
    elif hyp == "S2SVAEP-ELBO":
        hypermodel = SEQ_2_SEQ_GMVAE(
            input_shape=train.shape, loss="ELBO", predictor=True, number_of_components=k
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

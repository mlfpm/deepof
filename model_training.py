# @author lucasmiranda42

from datetime import datetime
from source.preprocess import *
from source.models import *
from tensorflow import keras
import argparse
import os, pickle

parser = argparse.ArgumentParser(
    description="Autoencoder training for DeepOF animal pose recognition"
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
    "--predictor",
    "-p",
    help="Activates the prediction branch of the variational Seq 2 Seq model. Defaults to True",
    default=True,
    type=bool,
)
parser.add_argument(
    "--variational",
    "-v",
    help="Sets the model to train to a variational Bayesian autoencoder. Defaults to True",
    default=True,
    type=bool,
)
parser.add_argumant(
    "--loss",
    "-l",
    help="Sets the loss function for the variational model. "
    "It has to be one of ELBO+MMD, ELBO or MMD. Defaults to ELBO+MMD",
    default="ELBO+MMD",
    type=str,
)
parser.add_argument(
    "--kl_warmup",
    "-klw",
    help="Number of epochs during which the KL weight increases linearly from zero to 1. Defaults to 10",
    default=10,
    type=int,
)
parser.add_argument(
    "--mmd_warmup",
    "-mmdw",
    help="Number of epochs during which the MMD weight increases linearly from zero to 1. Defaults to 10",
    default=10,
    type=int,
)

args = parser.parse_args()
train_path = os.path.abspath(args.train_path)
val_path = os.path.abspath(args.val_path)
input_type = args.input_type
k = args.components
predictor = args.predictor
variational = args.variational
loss = args.loss
kl_wu = args.kl_warmup
mmd_wu = args.wwu_warmup

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

log_dir = os.path.abspath(
    "logs/fit/{}{}_{}_{}_{}_{}_{}".format(
        ["GMVAE" if variational else "AE"],
        ["P" if predictor else ""],
        "components={}".format(k),
        "loss={}".format(loss),
        "kl_warmup={}".format(kl_wu),
        "mmd_warmup={}".format(mmd_wu),
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
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

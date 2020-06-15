# @author lucasmiranda42

from datetime import datetime
from source.preprocess import *
from source.models import *
from tensorflow import keras
import argparse
import os, pickle


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="Autoencoder training for DeepOF animal pose recognition"
)

parser.add_argument("--train-path", "-tp", help="set training set path", type=str)
parser.add_argument("--val-path", "-vp", help="set validation set path", type=str)
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
    It must be one of coords, dists, angles, coords+dist, coords+angle or coords+dist+angle. \
    Defaults to coords.",
    type=str,
    default="coords",
)
parser.add_argument(
    "--predictor",
    "-p",
    help="Activates the prediction branch of the variational Seq 2 Seq model. Defaults to True",
    default=True,
    type=str2bool,
)
parser.add_argument(
    "--variational",
    "-v",
    help="Sets the model to train to a variational Bayesian autoencoder. Defaults to True",
    default=True,
    type=str2bool,
)
parser.add_argument(
    "--loss",
    "-l",
    help="Sets the loss function for the variational model. "
    "It has to be one of ELBO+MMD, ELBO or MMD. Defaults to ELBO+MMD",
    default="ELBO+MMD",
    type=str,
)
parser.add_argument(
    "--kl-warmup",
    "-klw",
    help="Number of epochs during which the KL weight increases linearly from zero to 1. Defaults to 10",
    default=10,
    type=int,
)
parser.add_argument(
    "--mmd-warmup",
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
variational = bool(args.variational)
loss = args.loss
kl_wu = args.kl_warmup
mmd_wu = args.mmd_warmup

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
        ("GMVAE" if variational else "AE"),
        ("P" if predictor else ""),
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

# Training loop
if not variational:
    encoder, decoder, ae = SEQ_2_SEQ_AE(input_dict_train[input_type].shape).build()
    ae.build(input_dict_train[input_type].shape)

    print(ae.summary())

    # Fit the specified model to the data
    history = ae.fit(
        x=input_dict_train[input_type],
        y=input_dict_train[input_type],
        epochs=250,
        batch_size=512,
        verbose=1,
        validation_data=(input_dict_val[input_type], input_dict_val[input_type]),
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping("val_mae", patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                "./logs/checkpoints/",
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                save_freq="epoch",
            ),
        ],
    )

else:
    (
        encoder,
        generator,
        grouper,
        gmvaep,
        kl_warmup_callback,
        mmd_warmup_callback,
    ) = SEQ_2_SEQ_GMVAE(
        input_dict_train[input_type].shape,
        loss=loss,
        number_of_components=k,
        kl_warmup_epochs=kl_wu,
        mmd_warmup_epochs=mmd_wu,
        predictor=predictor,
    ).build()
    gmvaep.build(input_dict_train[input_type].shape)

    print(gmvaep.summary())

    if not predictor:
        history = gmvaep.fit(
            x=input_dict_train[input_type],
            y=input_dict_train[input_type],
            epochs=250,
            batch_size=512,
            verbose=1,
            validation_data=(input_dict_val[input_type], input_dict_val[input_type]),
            callbacks=[
                tensorboard_callback,
                kl_warmup_callback,
                mmd_warmup_callback,
                tf.keras.callbacks.EarlyStopping("val_mae", patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    "./logs/checkpoints/",
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=True,
                    save_freq="epoch",
                ),
            ],
        )
    else:
        history = gmvaep.fit(
            x=input_dict_train[input_type][:-1],
            y=[input_dict_train[input_type][:-1], input_dict_train[input_type][1:]],
            epochs=250,
            batch_size=512,
            verbose=1,
            validation_data=(
                input_dict_val[input_type][:-1],
                [input_dict_val[input_type][:-1], input_dict_val[input_type][1:]],
            ),
            callbacks=[
                tensorboard_callback,
                kl_warmup_callback,
                mmd_warmup_callback,
                tf.keras.callbacks.EarlyStopping("val_mae", patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    "./logs/checkpoints/",
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=True,
                    save_freq="epoch",
                ),
            ],
        )

# TODO:
#    - Input dictionary with parameters for the models (optional)
#    - Check that all checkpoints are being saved
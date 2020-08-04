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
parser.add_argument(
    "--val-num",
    "-vn",
    help="set number of videos of the training" "set to use for validation",
    type=int,
    default=0,
)
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
    It must be one of coords, dists, angles, coords+dist, coords+angle, dists+angle or coords+dist+angle. \
    Defaults to coords.",
    type=str,
    default="dists",
)
parser.add_argument(
    "--predictor",
    "-pred",
    help="Activates the prediction branch of the variational Seq 2 Seq model. Defaults to True",
    default=0,
    type=float,
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
parser.add_argument(
    "--hyperparameters",
    "-hp",
    help="Path pointing to a pickled dictionary of network hyperparameters. "
    "Thought to be used with the output of hyperparameter_tuning.py",
)
parser.add_argument(
    "--encoding-size",
    "-e",
    help="Sets the dimensionality of the latent space. Defaults to 16.",
    default=16,
    type=int,
)
parser.add_argument(
    "--overlap-loss",
    "-ol",
    help="If True, adds the negative MMD between all components of the latent Gaussian mixture to the loss function",
    default=False,
    type=str2bool,
)
parser.add_argument(
    "--batch-size",
    "-bs",
    help="set training batch size. Defaults to 512",
    type=int,
    default=512,
)
parser.add_argument(
    "--stability-check",
    "-s",
    help="Sets the number of times that the model is trained and initialised. If greater than 1 (the default), "
    "saves the cluster assignments to a dataframe on disk",
    type=int,
    default=1,
)

args = parser.parse_args()
train_path = os.path.abspath(args.train_path)
val_num = args.val_num
val_path = os.path.abspath(args.val_path) if args.val_path else None
input_type = args.input_type
k = args.components
predictor = float(args.predictor)
variational = bool(args.variational)
loss = args.loss
kl_wu = args.kl_warmup
mmd_wu = args.mmd_warmup
hparams = args.hyperparameters
encoding = args.encoding_size
batch_size = args.batch_size
overlap_loss = args.overlap_loss
runs = args.stability_check

if not train_path:
    raise ValueError("Set a valid data path for the training to run")
if not val_path and not val_num:
    raise ValueError(
        "Set a valid data path / validation number for the validation to run"
    )
if val_path and val_num:
    raise ValueError(
        "Set only one of valid data path / validation number for the validation to run"
    )
assert input_type in [
    "coords",
    "dists",
    "angles",
    "coords+dist",
    "coords+angle",
    "dists+angle",
    "coords+dist+angle",
], "Invalid input type. Type python model_training.py -h for help."

# Loads hyperparameters, most likely obtained from hyperparameter_tuning.py
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

try:
    with open(
        os.path.join(
            train_path, [i for i in os.listdir(train_path) if i.endswith(".pickle")][0]
        ),
        "rb",
    ) as handle:
        Treatment_dict = pickle.load(handle)
except IndexError:
    Treatment_dict = None


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

DLC_social_1_coords = project(
    path=train_path,  # Path where to find the required files
    smooth_alpha=0.50,  # Alpha value for exponentially weighted smoothing
    distances=[
        "B_Center",
        "B_Nose",
        "B_Left_ear",
        "B_Right_ear",
        "B_Left_flank",
        "B_Right_flank",
        "B_Tail_base",
    ],
    ego="B_Center",
    subset_condition="B",
    angles=True,
    connectivity=bp_dict,
    arena="circular",  # Type of arena used in the experiments
    arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
    exp_conditions=Treatment_dict,
).run(verbose=True)

# Coordinates for training data
coords1 = DLC_social_1_coords.get_coords(center="B_Center", align="B_Nose")
distances1 = DLC_social_1_coords.get_distances()
angles1 = DLC_social_1_coords.get_angles()
coords_distances1 = merge_tables(coords1, distances1)
coords_angles1 = merge_tables(coords1, angles1)
dists_angles1 = merge_tables(distances1, angles1)
coords_dist_angles1 = merge_tables(coords1, distances1, angles1)


input_dict_train = {
    "coords": coords1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "dists": distances1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "angles": angles1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "coords+dist": coords_distances1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "coords+angle": coords_angles1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "dists+angle": dists_angles1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
    "coords+dist+angle": coords_dist_angles1.preprocess(
        window_size=13,
        window_step=5,
        scale="standard",
        # filter="gaussian",
        sigma=55,
        align="center",
        test_videos=val_num,
    ),
}
# If a validation path is specified, use it to build a validation set
if val_path:

    DLC_social_2_coords = project(
        path=val_path,  # Path where to find the required files
        smooth_alpha=0.50,  # Alpha value for exponentially weighted smoothing
        distances=[
            "B_Center",
            "B_Nose",
            "B_Left_ear",
            "B_Right_ear",
            "B_Left_flank",
            "B_Right_flank",
            "B_Tail_base",
        ],
        ego="B_Center",
        subset_condition="B",
        angles=True,
        connectivity=bp_dict,
        arena="circular",  # Type of arena used in the experiments
        arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
        video_format=".mp4",
        table_format=".h5",
    ).run(verbose=True)

    # Coordinates for validation data
    coords2 = DLC_social_2_coords.get_coords(center="B_Center", align="B_Nose")
    distances2 = DLC_social_2_coords.get_distances()
    angles2 = DLC_social_2_coords.get_angles()
    coords_distances2 = merge_tables(coords2, distances2)
    coords_angles2 = merge_tables(coords2, angles2)
    dists_angles2 = merge_tables(distances2, angles2)
    coords_dist_angles2 = merge_tables(coords2, distances2, angles2)

    input_dict_val = {
        "coords": coords2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "dists": distances2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "angles": angles2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "coords+dist": coords_distances2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "coords+angle": coords_angles2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "dists+angle": dists_angles2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
        "coords+dist+angle": coords_dist_angles2.preprocess(
            window_size=13,
            window_step=5,
            scale="standard",
            # filter="gaussian",
            sigma=55,
            shuffle=True,
            align="all",
        ),
    }

# Select training and validation set
if val_path:
    X_train = input_dict_train[input_type]
    X_val = input_dict_val[input_type]
elif val_num:
    X_train = input_dict_train[input_type][0]
    X_val = input_dict_train[input_type][1]

# Training loop
for run in range(runs):

    # To avoid stability issues
    tf.keras.backend.clear_session()

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

    cp_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            "./logs/checkpoints/" + run_ID + "/cp-{epoch:04d}.ckpt",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        ),
    )

    onecycle = OneCycleScheduler(X_train.shape[0] // batch_size * 250, max_rate=0.005,)

    if not variational:
        encoder, decoder, ae = SEQ_2_SEQ_AE(X_train.shape, **hparams).build()
        ae.build(X_train.shape)

        print(ae.summary())
        ae.save_weights("./logs/checkpoints/cp-{epoch:04d}.ckpt".format(epoch=0))
        # Fit the specified model to the data
        history = ae.fit(
            x=X_train,
            y=X_train,
            epochs=25,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_val, X_val),
            callbacks=[
                tensorboard_callback,
                cp_callback,
                onecycle,
                tf.keras.callbacks.EarlyStopping(
                    "val_loss", patience=10, restore_best_weights=True
                ),
            ],
        )

        ae.save_weights("{}_final_weights.h5".format(run_ID))

    else:
        (
            encoder,
            generator,
            grouper,
            gmvaep,
            kl_warmup_callback,
            mmd_warmup_callback,
        ) = SEQ_2_SEQ_GMVAE(
            X_train.shape,
            loss=loss,
            number_of_components=k,
            kl_warmup_epochs=kl_wu,
            mmd_warmup_epochs=mmd_wu,
            predictor=predictor,
            overlap_loss=overlap_loss,
            **hparams
        ).build()
        gmvaep.build(X_train.shape)

        print(gmvaep.summary())

        callbacks_ = [
            tensorboard_callback,
            cp_callback,
            onecycle,
            tf.keras.callbacks.EarlyStopping(
                "val_loss", patience=10, restore_best_weights=True
            ),
        ]

        if "ELBO" in loss and kl_wu > 0:
            callbacks_.append(kl_warmup_callback)
        if "MMD" in loss and mmd_wu > 0:
            callbacks_.append(mmd_warmup_callback)

        if predictor == 0:
            history = gmvaep.fit(
                x=X_train,
                y=X_train,
                epochs=250,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_val, X_val,),
                callbacks=callbacks_,
            )
        else:
            history = gmvaep.fit(
                x=X_train[:-1],
                y=[X_train[:-1], X_train[1:]],
                epochs=250,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_val[:-1], [X_val[:-1], X_val[1:]],),
                callbacks=callbacks_,
            )

        gmvaep.save_weights("{}_final_weights.h5".format(run_ID))

    # To avoid stability issues
    tf.keras.backend.clear_session()

# TODO:
#    - Investigate partial methods for preprocess (lots of calls with the same parameters!)
#    - Investigate how goussian filters affect reproducibility (in a systematic way)
#    - Investigate how smoothing affects reproducibility (in a systematic way)
#    - Check if MCDropout effectively enhances reproducibility or not

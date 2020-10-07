# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Model training command line tool for the deepof package.
usage: python -m examples.model_training -h

"""

from deepof.data import *
from deepof.models import *
from deepof.utils import *
from .train_utils import *
from tensorflow import keras

parser = argparse.ArgumentParser(
    description="Autoencoder training for DeepOF animal pose recognition"
)

parser.add_argument("--train-path", "-tp", help="set training set path", type=str)
parser.add_argument(
    "--val-num",
    "-vn",
    help="set number of videos of the training" "set to use for validation",
    type=int,
    default=1,
)
parser.add_argument(
    "--components",
    "-k",
    help="set the number of components for the GMVAE(P) model. Defaults to 1",
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
    "Thought to be used with the output of hyperparameter tuning",
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
parser.add_argument(
    "--gaussian-filter",
    "-gf",
    help="Convolves each training instance with a Gaussian filter before feeding it to the autoencoder model",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--window-size",
    "-ws",
    help="Sets the sliding window size to be used when building both training and validation sets. Defaults to 15",
    type=int,
    default=15,
)
parser.add_argument(
    "--window-step",
    "-wt",
    help="Sets the sliding window step to be used when building both training and validation sets. Defaults to 5",
    type=int,
    default=5,
)
parser.add_argument(
    "--smooth-alpha",
    "-sa",
    help="Sets the exponential smoothing factor to apply to the input data. "
    "Float between 0 and 1 (lower is more smooting)",
    type=float,
    default=0.99,
)
parser.add_argument(
    "--exclude-bodyparts",
    "-exc",
    help="Excludes the indicated bodyparts from all analyses. "
    "It should consist of several values separated by commas",
    type=str,
    default="",
)
parser.add_argument(
    "--arena-dims",
    "-adim",
    help="diameter in mm of the utilised arena. Used for scaling purposes",
    type=int,
    default=380,
)
parser.add_argument(
    "--hyperparameter-tuning",
    "-tune",
    help="If True, hyperparameter tuning is performed. See documentation for details",
    type=str2bool,
    default=False,
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

arena_dims = args.arena_dims
batch_size = args.batch_size
bayopt_trials = args.bayopt
encoding = args.encoding_size
exclude_bodyparts = tuple(args.exclude_bodyparts.split(","))
gaussian_filter = args.gaussian_filter
hparams = args.hyperparameters
hyp = args.hypermodel
input_type = args.input_type
k = args.components
kl_wu = args.kl_warmup
loss = args.loss
mmd_wu = args.mmd_warmup
overlap_loss = args.overlap_loss
predictor = float(args.predictor)
runs = args.stability_check
smooth_alpha = args.smooth_alpha
train_path = os.path.abspath(args.train_path)
tune = args.hyperparameter_tuning
val_num = args.val_num
variational = bool(args.variational)
window_size = args.window_size
window_step = args.window_step

if not train_path:
    raise ValueError("Set a valid data path for the training to run")
if not val_num:
    raise ValueError(
        "Set a valid data path / validation number for the validation to run"
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

# Loads model hyperparameters and treatment conditions, if available
hparams = load_hparams(hparams, encoding)
treatment_dict = load_treatments(train_path)

# noinspection PyTypeChecker
project_coords = project(
    arena="circular",  # Type of arena used in the experiments
    arena_dims=tuple(
        [arena_dims]
    ),  # Dimensions of the arena. Just one if it's circular
    exclude_bodyparts=exclude_bodyparts,
    exp_conditions=treatment_dict,
    path=train_path,  # Path where to find the required files
    smooth_alpha=smooth_alpha,  # Alpha value for exponentially weighted smoothing
    table_format=".h5",
    video_format=".mp4",
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

print("Creating training and validation sets...")
# Get training and validation sets
X_train = input_dict_train[input_type][0]
X_val = input_dict_train[input_type][1]
print("Done!")


# Proceed with training mode. Fit autoencoder with the same parameters,
# as many times as specified by runs
if not tune:

    # Training loop
    for run in range(runs):

        # To avoid stability issues
        tf.keras.backend.clear_session()

        run_ID, tensorboard_callback, cp_callback, onecycle = get_callbacks(
            X_train, batch_size, variational, predictor, k, loss, kl_wu, mmd_wu
        )

        if not variational:
            encoder, decoder, ae = SEQ_2_SEQ_AE(hparams).build(X_train.shape)
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
                loss=loss,
                number_of_components=k,
                kl_warmup_epochs=kl_wu,
                mmd_warmup_epochs=mmd_wu,
                predictor=predictor,
                overlap_loss=overlap_loss,
                architecture_hparams=hparams,
            ).build(
                X_train.shape
            )
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

else:
    # Runs hyperparameter tuning with the specified parameters and saves the results

    run_ID, tensorboard_callback, cp_callback, onecycle = get_callbacks(
        X_train, batch_size, variational, predictor, k, loss, kl_wu, mmd_wu
    )

    best_hyperparameters, best_model = tune_search(
        X_train,
        X_val,
        bayopt_trials=bayopt_trials,
        hypermodel=hyp,
        k=k,
        kl_wu=kl_wu,
        loss=loss,
        mmd_wu=mmd_wu,
        overlap_loss=overlap_loss,
        predictor=predictor,
        project_name="{}-based_{}_BAYESIAN_OPT".format(input_type, hyp),
        tensorboard_callback=tensorboard_callback,
    )

    # Saves a compiled, untrained version of the best model
    best_model.build(input_dict_train[input_type].shape)
    best_model.save(
        "{}-based_{}_BAYESIAN_OPT.h5".format(input_type, hyp), save_format="tf"
    )

    # Saves the best hyperparameters
    with open(
        "{}-based_{}_BAYESIAN_OPT_params.pickle".format(input_type, hyp), "wb"
    ) as handle:
        pickle.dump(
            best_hyperparameters.values, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

# TODO:
#    - Investigate how goussian filters affect reproducibility (in a systematic way)
#    - Investigate how smoothing affects reproducibility (in a systematic way)
#    - Check if MCDropout effectively enhances reproducibility or not

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
from deepof.train_utils import *
from tensorflow import keras

parser = argparse.ArgumentParser(
    description="Autoencoder training for DeepOF animal pose recognition"
)

parser.add_argument(
    "--animal-id",
    "-id",
    help="Id of the animal to use. Empty string by default",
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
    "--batch-size",
    "-bs",
    help="set training batch size. Defaults to 512",
    type=int,
    default=512,
)
parser.add_argument(
    "--bayopt",
    "-n",
    help="sets the number of Bayesian optimization iterations to run. Default is 25",
    type=int,
    default=25,
)
parser.add_argument(
    "--components",
    "-k",
    help="set the number of components for the GMVAE(P) model. Defaults to 1",
    type=int,
    default=1,
)
parser.add_argument(
    "--exclude-bodyparts",
    "-exc",
    help="Excludes the indicated bodyparts from all analyses. It should consist of several values separated by commas",
    type=str,
    default="",
)
parser.add_argument(
    "--gaussian-filter",
    "-gf",
    help="Convolves each training instance with a Gaussian filter before feeding it to the autoencoder model",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--hyperparameter-tuning",
    "-tune",
    help="If True, hyperparameter tuning is performed. See documentation for details",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--hyperparameters",
    "-hp",
    help="Path pointing to a pickled dictionary of network hyperparameters. "
    "Thought to be used with the output of hyperparameter tuning",
    type=str,
    default=None,
)
parser.add_argument(
    "--input-type",
    "-d",
    help="Select an input type for the autoencoder hypermodels. "
    "It must be one of coords, dists, angles, coords+dist, coords+angle, dists+angle or coords+dist+angle."
    "Defaults to coords.",
    type=str,
    default="dists",
)
parser.add_argument(
    "--kl-warmup",
    "-klw",
    help="Number of epochs during which the KL weight increases linearly from zero to 1. Defaults to 10",
    default=10,
    type=int,
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
    "--mmd-warmup",
    "-mmdw",
    help="Number of epochs during which the MMD weight increases linearly from zero to 1. Defaults to 10",
    default=10,
    type=int,
)
parser.add_argument(
    "--overlap-loss",
    "-ol",
    help="If True, adds the negative MMD between all components of the latent Gaussian mixture to the loss function",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--phenotype-classifier",
    "-pheno",
    help="Activates the phenotype classification branch with the specified weight. Defaults to 0.0 (inactive)",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--predictor",
    "-pred",
    help="Activates the prediction branch of the variational Seq 2 Seq model with the specified weight. "
    "Defaults to 0.0 (inactive)",
    default=0.0,
    type=float,
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
    "--stability-check",
    "-s",
    help="Sets the number of times that the model is trained and initialised. "
    "If greater than 1 (the default), saves the cluster assignments to a dataframe on disk",
    type=int,
    default=1,
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
    "--variational",
    "-v",
    help="Sets the model to train to a variational Bayesian autoencoder. Defaults to True",
    default=True,
    type=str2bool,
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

args = parser.parse_args()

animal_id = args.animal_id
arena_dims = args.arena_dims
batch_size = args.batch_size
bayopt_trials = args.bayopt
exclude_bodyparts = list(args.exclude_bodyparts.split(","))
gaussian_filter = args.gaussian_filter
hparams = args.hyperparameters
input_type = args.input_type
k = args.components
kl_wu = args.kl_warmup
loss = args.loss
mmd_wu = args.mmd_warmup
overlap_loss = args.overlap_loss
pheno_class = float(args.phenotype_classifier)
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
hparams = load_hparams(hparams)
treatment_dict = load_treatments(train_path)

# noinspection PyTypeChecker
project_coords = project(
    animal_ids=tuple([animal_id]),
    arena="circular",
    arena_dims=tuple([arena_dims]),
    exclude_bodyparts=exclude_bodyparts,
    exp_conditions=treatment_dict,
    path=train_path,
    smooth_alpha=smooth_alpha,
    table_format=".h5",
    video_format=".mp4",
)

if animal_id:
    project_coords.subset_condition = animal_id

project_coords = project_coords.run(verbose=True)
undercond = "" if animal_id == "" else "_"

# Coordinates for training data
coords = project_coords.get_coords(
    center=animal_id + undercond + "Center",
    align=animal_id + undercond + "Spine_1",
    align_inplace=True,
)
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
X_train, y_train, X_val, y_val = batch_preprocess(input_dict_train[input_type])
# Get training and validation sets

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
if pheno_class > 0:
    print("Training set label shape:", y_train.shape)
    print("Validation set label shape:", y_val.shape)

print("Done!")

# Proceed with training mode. Fit autoencoder with the same parameters,
# as many times as specified by runs
if not tune:

    # Training loop
    for run in range(runs):

        # To avoid stability issues
        tf.keras.backend.clear_session()

        run_ID, tensorboard_callback, onecycle, cp_callback = get_callbacks(
            X_train, batch_size, True, variational, predictor, loss,
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
                        "val_mae", patience=15, restore_best_weights=True
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
                architecture_hparams=hparams,
                batch_size=batch_size,
                compile_model=True,
                kl_warmup_epochs=kl_wu,
                loss=loss,
                mmd_warmup_epochs=mmd_wu,
                number_of_components=k,
                overlap_loss=overlap_loss,
                phenotype_prediction=pheno_class,
                predictor=predictor,
            ).build(
                X_train.shape
            )
            print(gmvaep.summary())

            callbacks_ = [
                tensorboard_callback,
                cp_callback,
                onecycle,
                tf.keras.callbacks.EarlyStopping(
                    "val_mae", patience=15, restore_best_weights=True
                ),
            ]

            if "ELBO" in loss and kl_wu > 0:
                callbacks_.append(kl_warmup_callback)
            if "MMD" in loss and mmd_wu > 0:
                callbacks_.append(mmd_warmup_callback)

            Xs, ys = [X_train], [X_train]
            Xvals, yvals = [X_val], [X_val]

            if predictor > 0.0:
                Xs, ys = X_train[:-1], [X_train[:-1], X_train[1:]]
                Xvals, yvals = X_val[:-1], [X_val[:-1], X_val[1:]]

            if pheno_class > 0.0:
                ys += [y_train]
                yvals += [y_val]

            history = gmvaep.fit(
                x=Xs,
                y=ys,
                epochs=250,
                batch_size=batch_size,
                verbose=1,
                validation_data=(Xvals, yvals,),
                callbacks=callbacks_,
            )

            gmvaep.save_weights("{}_final_weights.h5".format(run_ID))

        # To avoid stability issues
        tf.keras.backend.clear_session()

else:
    # Runs hyperparameter tuning with the specified parameters and saves the results

    hyp = "S2SGMVAE" if variational else "S2SAE"

    run_ID, tensorboard_callback, onecycle = get_callbacks(
        X_train, batch_size, False, variational, predictor, loss
    )

    best_hyperparameters, best_model = tune_search(
        data=[X_train, y_train, X_val, y_val],
        bayopt_trials=bayopt_trials,
        hypermodel=hyp,
        k=k,
        kl_warmup_epochs=kl_wu,
        loss=loss,
        mmd_warmup_epochs=mmd_wu,
        overlap_loss=overlap_loss,
        pheno_class=pheno_class,
        predictor=predictor,
        project_name="{}-based_{}_BAYESIAN_OPT".format(input_type, hyp),
        callbacks=[
            tensorboard_callback,
            onecycle,
            tf.keras.callbacks.EarlyStopping(
                "val_mae", patience=15, restore_best_weights=True
            ),
        ],
        n_replicas=3,
        n_epochs=30,
    )

    # Saves a compiled, untrained version of the best model
    best_model.build(X_train.shape)
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

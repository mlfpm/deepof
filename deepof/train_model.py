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
from tensorboard.plugins.hparams import api as hp
from sklearn.metrics import roc_auc_score

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
    "--components",
    "-k",
    help="set the number of components for the GMVAE(P) model. Defaults to 1",
    type=int,
    default=1,
)
parser.add_argument(
    "--encoding-size",
    "-es",
    help="set the number of dimensions of the latent space. 16 by default",
    type=int,
    default=16,
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
    "--hpt-trials",
    "-n",
    help="sets the number of hyperparameter tuning iterations to run. Default is 25",
    type=int,
    default=25,
)
parser.add_argument(
    "--hyperparameter-tuning",
    "-tune",
    help="Indicates whether hyperparameters should be tuned either using 'bayopt' of 'hyperband'. "
    "See documentation for details",
    type=str,
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
    "--montecarlo-kl",
    "-mckl",
    help="Number of samples to compute when adding KLDivergence to the loss function",
    default=10,
    type=int,
)
parser.add_argument(
    "--neuron-control",
    "-nc",
    help="If True, adds the proportion of dead neurons in the latent space as a metric",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--output-path",
    "-o",
    help="Sets the base directory where to output results. Default is the current directory",
    type=str,
    default=".",
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
hypertun_trials = args.hpt_trials
encoding_size = args.encoding_size
exclude_bodyparts = tuple(args.exclude_bodyparts.split(","))
gaussian_filter = args.gaussian_filter
hparams = args.hyperparameters
input_type = args.input_type
k = args.components
kl_wu = args.kl_warmup
loss = args.loss
mmd_wu = args.mmd_warmup
mc_kl = args.montecarlo_kl
neuron_control = args.neuron_control
output_path = os.path.join(args.output_path)
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

# Logs hyperparameters  if specified on the --logparam CLI argument
logparam = {
    "encoding": encoding_size,
    "k": k,
    "loss": loss,
}
if pheno_class:
    logparam["pheno_weight"] = pheno_class

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
    propagate_labels=(pheno_class > 0),
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
            X_train=X_train,
            batch_size=batch_size,
            cp=True,
            variational=variational,
            phenotype_class=pheno_class,
            predictor=predictor,
            loss=loss,
            logparam=logparam,
            outpath=output_path,
        )

        logparams = [
            hp.HParam(
                "encoding",
                hp.Discrete([2, 4, 6, 8, 12, 16]),
                display_name="encoding",
                description="encoding size dimensionality",
            ),
            hp.HParam(
                "k",
                hp.IntInterval(min_value=1, max_value=15),
                display_name="k",
                description="cluster_number",
            ),
            hp.HParam(
                "loss",
                hp.Discrete(["ELBO", "MMD", "ELBO+MMD"]),
                display_name="loss function",
                description="loss function",
            ),
            hp.HParam(
                "run",
                hp.Discrete([0, 1, 2]),
                display_name="trial run",
                description="trial run",
            ),
        ]

        rec = "reconstruction_" if pheno_class else ""
        metrics = [
            hp.Metric("val_{}mae".format(rec), display_name="val_{}mae".format(rec)),
            hp.Metric("val_{}mse".format(rec), display_name="val_{}mse".format(rec)),
        ]
        logparam["run"] = run
        if pheno_class:
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

        with tf.summary.create_file_writer(
            os.path.join(output_path, "hparams", run_ID)
        ).as_default():
            hp.hparams_config(
                hparams=logparams,
                metrics=metrics,
            )

        if not variational:
            encoder, decoder, ae = SEQ_2_SEQ_AE(hparams).build(X_train.shape)
            print(ae.summary())

            ae.save_weights(
                os.path.join(
                    output_path, "/checkpoints/cp-{epoch:04d}.ckpt".format(epoch=0)
                )
            )
            # Fit the specified model to the data
            history = ae.fit(
                x=X_train,
                y=X_train,
                epochs=35,
                batch_size=batch_size,
                verbose=1,
                validation_data=(X_val, X_val),
                callbacks=[
                    tensorboard_callback,
                    cp_callback,
                    onecycle,
                    CustomStopper(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                        start_epoch=max(kl_wu, mmd_wu),
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
                encoding=encoding_size,
                kl_warmup_epochs=kl_wu,
                loss=loss,
                mmd_warmup_epochs=mmd_wu,
                montecarlo_kl=mc_kl,
                neuron_control=neuron_control,
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
                # cp_callback,
                onecycle,
                CustomStopper(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                    start_epoch=max(kl_wu, mmd_wu),
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
                epochs=35,
                batch_size=batch_size,
                verbose=1,
                validation_data=(
                    Xvals,
                    yvals,
                ),
                callbacks=callbacks_,
            )

            gmvaep.save_weights(
                os.path.join(
                    output_path,
                    "trained_weights",
                    "GMVAE_loss={}_encoding={}_k={}_{}{}run_{}_final_weights.h5".format(
                        loss,
                        encoding_size,
                        k,
                        ("pheno={}_".format(pheno_class) if pheno_class else ""),
                        ("predictor={}_".format(predictor) if predictor else ""),
                        run,
                    ),
                )
            )

            # noinspection PyUnboundLocalVariable
            def tensorboard_metric_logging(run_dir: str, hpms: Any):
                output = gmvaep.predict(X_val)
                if pheno_class or predictor:
                    reconstruction = output[0]
                    prediction = output[1]
                    pheno = output[-1]
                else:
                    reconstruction = output

                with tf.summary.create_file_writer(run_dir).as_default():
                    hp.hparams(hpms)  # record the values used in this trial
                    val_mae = tf.reduce_mean(
                        tf.keras.metrics.mean_absolute_error(X_val, reconstruction)
                    )
                    val_mse = tf.reduce_mean(
                        tf.keras.metrics.mean_squared_error(X_val, reconstruction)
                    )
                    tf.summary.scalar("val_{}mae".format(rec), val_mae, step=1)
                    tf.summary.scalar("val_{}mse".format(rec), val_mse, step=1)

                    if predictor:
                        pred_mae = tf.reduce_mean(
                            tf.keras.metrics.mean_absolute_error(X_val, prediction)
                        )
                        pred_mse = tf.reduce_mean(
                            tf.keras.metrics.mean_squared_error(X_val, prediction)
                        )
                        tf.summary.scalar(
                            "val_prediction_mae".format(rec), pred_mae, step=1
                        )
                        tf.summary.scalar(
                            "val_prediction_mse".format(rec), pred_mse, step=1
                        )

                    if pheno_class:
                        pheno_acc = tf.keras.metrics.binary_accuracy(
                            y_val, tf.squeeze(pheno)
                        )
                        pheno_auc = roc_auc_score(y_val, pheno)

                        tf.summary.scalar(
                            "phenotype_prediction_accuracy", pheno_acc, step=1
                        )
                        tf.summary.scalar("phenotype_prediction_auc", pheno_auc, step=1)

            # Logparams to tensorboard
            tensorboard_metric_logging(
                os.path.join(output_path, "hparams", run_ID),
                logparam,
            )

        # To avoid stability issues
        tf.keras.backend.clear_session()

else:
    # Runs hyperparameter tuning with the specified parameters and saves the results

    hyp = "S2SGMVAE" if variational else "S2SAE"

    run_ID, tensorboard_callback, onecycle = get_callbacks(
        X_train=X_train,
        batch_size=batch_size,
        cp=False,
        variational=variational,
        phenotype_class=pheno_class,
        predictor=predictor,
        loss=loss,
        logparam=None,
    )

    best_hyperparameters, best_model = tune_search(
        data=[X_train, y_train, X_val, y_val],
        encoding_size=encoding_size,
        hypertun_trials=hypertun_trials,
        hpt_type=tune,
        hypermodel=hyp,
        k=k,
        kl_warmup_epochs=kl_wu,
        loss=loss,
        mmd_warmup_epochs=mmd_wu,
        overlap_loss=overlap_loss,
        pheno_class=pheno_class,
        predictor=predictor,
        project_name="{}-based_{}_{}".format(input_type, hyp, tune.capitalize()),
        callbacks=[
            tensorboard_callback,
            onecycle,
            CustomStopper(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                start_epoch=max(kl_wu, mmd_wu),
            ),
        ],
        n_replicas=3,
        n_epochs=30,
    )

    # # Saves a compiled, untrained version of the best model
    # best_model.build(X_train.shape)
    # # noinspection PyArgumentList
    # best_model.save(
    #     os.path.join(
    #         output_path, "{}-based_{}_{}.h5".format(input_type, hyp, tune.capitalize())
    #     ),
    #     save_format="tf",
    # )

    # Saves the best hyperparameters
    with open(
        os.path.join(
            output_path,
            "{}-based_{}_{}_params.pickle".format(input_type, hyp, tune.capitalize()),
        ),
        "wb",
    ) as handle:
        pickle.dump(
            best_hyperparameters.values, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

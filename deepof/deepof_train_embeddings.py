# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Model training command line tool for the deepof package.
usage: python -m examples.model_training -h

"""

import argparse
import os
import pickle

import numpy as np

import deepof.data
import deepof.unsupervised_utils
import deepof.utils

parser = argparse.ArgumentParser(
    description="Autoencoder training for DeepOF animal pose recognition"
)

parser.add_argument(
    "--animal-ids",
    "-ids",
    help="Id of the animals in the loaded dataset to use. Empty string by default",
    type=str,
    default="",
)
parser.add_argument(
    "--animal-to-preprocess",
    "-idprep",
    help="Id of the animal to preprocess if multiple animals are being tracked. None by default",
    type=str,
    default=None,
)
parser.add_argument(
    "--arena-dims",
    "-adim",
    help="diameter in mm of the utilised arena. Used for scaling purposes",
    type=int,
    default=380,
)
parser.add_argument(
    "--automatic-changepoints",
    "-ruptures",
    help="Algorithm to use to rupture the time series. L2-regularized BottomUp approach (l2) by default."
    "Must be one of 'rbf', 'linear' or False (a sliding window is used instead).",
    type=str,
    default="rbf",
)
parser.add_argument(
    "--batch-size",
    "-bs",
    help="set training batch size. Defaults to 256",
    type=int,
    default=512,
)
parser.add_argument(
    "--n-components",
    "-k",
    help="set the number of components for the unsupervised model. Defaults to 5",
    type=int,
    default=5,
)
parser.add_argument(
    "--encoding-size",
    "-es",
    help="set the number of dimensions of the latent space. 16 by default",
    type=int,
    default=16,
)
parser.add_argument(
    "--embedding-model",
    "-embedding",
    help="Algorithm to use to embed and cluster the time series. Must be one of: VQVAE (default), GMVAE, or Contrastive",
    type=str,
    default="VQVAE",
)
parser.add_argument(
    "--exclude-bodyparts",
    "-exc",
    help="Excludes the indicated bodyparts from all analyses. It should consist of several values separated by commas",
    type=str,
    default="",
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
    "It must be one of coords, dists, angles, coords+dist, coords+angle, dists+angle or coords+dist+angle. "
    "To any of these, '+speed' can be added at the end, which includes overall speed of each bodypart. "
    "Defaults to coords.",
    type=str,
    default="coords",
)
parser.add_argument(
    "--output-path",
    "-o",
    help="Sets the base directory where to output results. Default is the current directory",
    type=str,
    default=".",
)
parser.add_argument(
    "--kmeans-loss",
    "-kmeans",
    help="If > 0, adds a regularization term controlling for correlation between dimensions in the latent space",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--cat-kl-loss",
    "-catkl",
    help="If > 0, adds a regularization term that minimizes the KL divergence between cluster assignment frequencies"
    "and a uniform distribution",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--smooth-alpha",
    "-sa",
    help="Sets the exponential smoothing factor to apply to the input data. "
    "Float between 0 and 1 (lower is more smooting)",
    type=float,
    default=4,
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
    "--window-size",
    "-ws",
    help="Sets the sliding window size to be used when building both training and validation sets. Defaults to 15",
    type=int,
    default=5,
)
parser.add_argument(
    "--window-step",
    "-wt",
    help="Sets the sliding window step to be used when building both training and validation sets. Defaults to 5",
    type=int,
    default=1,
)
parser.add_argument(
    "--run",
    "-rid",
    help="Sets the run ID of the experiment (for naming output files only). If 0 (default), uses a timestamp instead",
    type=int,
    default=0,
)

args = parser.parse_args()

try:
    animal_ids = args.animal_ids
    animal_to_preprocess = args.animal_to_preprocess
    arena_dims = args.arena_dims
    automatic_changepoints = args.automatic_changepoints
    if automatic_changepoints == "False":
        automatic_changepoints = False
    batch_size = args.batch_size
    hypertun_trials = args.hpt_trials
    encoding_size = args.encoding_size
    embedding_model = args.embedding_model
    exclude_bodyparts = [i for i in args.exclude_bodyparts.split(",") if i]
    hparams = args.hyperparameters if args.hyperparameters is not None else {}
    input_type = args.input_type
    n_components = args.n_components
    output_path = os.path.join(args.output_path)
    kmeans_loss = float(args.kmeans_loss)
    cat_kl_loss = float(args.cat_kl_loss)
    smooth_alpha = args.smooth_alpha
    train_path = os.path.abspath(args.train_path)
    tune = args.hyperparameter_tuning
    val_num = args.val_num
    window_size = args.window_size
    window_step = args.window_step
    run = args.run

except TypeError:
    raise ValueError(
        "One or more mandatory arguments are missing. Use --help for details on how to run the CLI"
    )

if not train_path:
    raise ValueError("Set a valid data path for the training to run")

assert input_type.replace("+speed", "") in [
    "coords",
    "dists",
    "angles",
    "coords+dist",
    "coords+angle",
    "dists+angle",
    "coords+dist+angle",
], "Invalid input type. Type python model_training.py -h for help."

# Loads model hyperparameters and treatment conditions, if available
treatment_dict = deepof.unsupervised_utils.load_treatments(train_path)

# Logs hyperparameters  if specified on the --logparam CLI argument
logparam = {"encoding": encoding_size, "k": n_components}

# noinspection PyTypeChecker
project_coords = deepof.data.Project(
    animal_ids=animal_ids.split(","),
    arena="circular-autodetect",
    arena_dims=arena_dims,
    enable_iterative_imputation=True,
    exclude_bodyparts=exclude_bodyparts,
    exp_conditions=treatment_dict,
    path=train_path,
    smooth_alpha=smooth_alpha,
    table_format=".h5",
    video_format=".mp4",
)

project_coords = project_coords.run(verbose=True)

# Coordinates for training data
coords = project_coords.get_coords(
    center="Center",
    align="Spine_1",
    align_inplace=True,
    propagate_labels=False,
    propagate_annotations=False,
    selected_id=animal_to_preprocess,
)
speeds = project_coords.get_coords(speed=1, selected_id=animal_to_preprocess)
distances = project_coords.get_distances(selected_id=animal_to_preprocess)
angles = project_coords.get_angles(selected_id=animal_to_preprocess)
coords_distances = coords.merge(distances)
coords_angles = coords.merge(angles)
dists_angles = distances.merge(angles)
coords_dist_angles = coords.merge(distances, angles)


def batch_preprocess(tab_dict):
    """Returns a preprocessed instance of the input table_dict object"""

    return tab_dict.preprocess(
        window_size=window_size,
        window_step=window_step,
        automatic_changepoints=automatic_changepoints,
        scale="standard",
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

to_preprocess = input_dict_train[input_type.replace("+speed", "")]
if "speed" in input_type:
    to_preprocess = to_preprocess.merge(speeds)

print("Preprocessing data...")
X_train, y_train, X_val, y_val = batch_preprocess(to_preprocess)
# Get training and validation sets

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Done!")

# Proceed with training mode. Fit autoencoder with the same parameters,
# as many times as specified by runs
if not tune:

    trained_models = project_coords.deep_unsupervised_embedding(
        (X_train, y_train, X_val, y_val),
        batch_size=batch_size,
        latent_dim=encoding_size,
        embedding_model=embedding_model,
        n_components=n_components,
        output_path=output_path,
        save_checkpoints=False,
        save_weights=True,
        input_type=input_type,
        # Parameters that control the training process
        kmeans_loss=kmeans_loss,
        reg_cat_clusters=cat_kl_loss,
        run=run,
    )

    # Save data encoded with the current trained model
    deep_encodings_per_video = {}
    deep_assignments_per_video = {}
    deep_breaks_per_video = {}

    for key in to_preprocess.keys():

        # Get preprocessed data for current video
        curr_prep = to_preprocess.filter_videos([key]).preprocess(
            window_size=window_size,
            window_step=window_step,
            automatic_changepoints=automatic_changepoints,
            scale="standard",
            test_videos=0,
            shuffle=False,
        )[0]

        # Get breakpoints per video
        deep_breaks_per_video[key] = np.all(curr_prep != 0, axis=2).sum(axis=1)

        # Get current model weights
        curr_weights = trained_models[3].get_weights()

        # Load weights into a newly created model, buit with the current input shape
        if embedding_model == "VQVAE":
            ae_models = deepof.models.VQVAE(
                input_shape=curr_prep.shape,
                latent_dim=encoding_size,
                n_components=n_components,
            )
            curr_deep_encoder, curr_deep_grouper, curr_ae = (
                ae_models.encoder,
                ae_models.soft_quantizer,
                ae_models.vqvae,
            )

        elif embedding_model == "GMVAE":
            ae_models = deepof.models.GMVAE(
                input_shape=curr_prep.shape,
                batch_size=batch_size,
                latent_dim=encoding_size,
                n_components=n_components,
            )
            curr_deep_encoder, curr_deep_grouper, curr_ae = (
                ae_models.encoder,
                ae_models.grouper,
                ae_models.gmvae,
            )

        elif embedding_model == "Contrastive":
            raise NotImplementedError

        # noinspection PyUnboundLocalVariable
        curr_ae.set_weights(curr_weights)

        # Embed current video in the autoencoder and add to the dictionary
        # noinspection PyUnboundLocalVariable
        mean_encodings = curr_deep_encoder(curr_prep)
        deep_encodings_per_video[key] = mean_encodings

        # Obtain groupings for current video and add to the dictionary
        # noinspection PyUnboundLocalVariable
        deep_assignments_per_video[key] = curr_deep_grouper(curr_prep)

    with open(
        os.path.join(
            output_path,
            "deepof_unsupervised_VQVAE_encodings_input={}_k={}_latdim={}_kmeans_loss={}_run={}.pkl".format(
                input_type, n_components, encoding_size, kmeans_loss, run
            ),
        ),
        "wb",
    ) as x:
        pickle.dump(
            [
                deep_encodings_per_video,
                deep_assignments_per_video,
                deep_breaks_per_video,
            ],
            x,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

else:
    # Runs hyperparameter tuning with the specified parameters and saves the results
    (
        run_ID,
        tensorboard_callback,
        reduce_lr_callback,
    ) = deepof.unsupervised_utils.get_callbacks(
        input_type=input_type,
        cp=False,
        logparam=logparam,
        outpath=output_path,
        kmeans_loss=kmeans_loss,
        run=run,
    )

    best_hyperparameters, best_model = deepof.unsupervised_utils.tune_search(
        data=[X_train, y_train, X_val, y_val],
        batch_size=batch_size,
        encoding_size=encoding_size,
        hypertun_trials=hypertun_trials,
        hpt_type=tune,
        k=n_components,
        kmeans_loss=kmeans_loss,
        project_name="{}-based_VQVAE_{}".format(input_type, tune.capitalize()),
        callbacks=[
            tensorboard_callback,
            reduce_lr_callback,
            deepof.unsupervised_utils.CustomStopper(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                start_epoch=max(kl_wu, mmd_wu),
            ),
        ],
        n_replicas=1,
        n_epochs=30,
        outpath=output_path,
    )

    # Saves the best hyperparameters
    with open(
        os.path.join(
            output_path,
            "{}-based_VQVAE_{}_params.pickle".format(input_type, tune.capitalize()),
        ),
        "wb",
    ) as handle:
        pickle.dump(
            best_hyperparameters.values, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

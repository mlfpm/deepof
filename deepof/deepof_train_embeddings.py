# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Model training command line tool for the deepof package.
usage: python -m examples.model_training -h

"""

import argparse
import calendar
import time
import os
import pickle
import deepof.data
import deepof.model_utils
import deepof.utils

if __name__ == "__main__":

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
        help="Id of the animal to preprocess if multiple animals are being tracked. None by default, which results in "
        "all animals being processed.",
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
        choices=["False", "linear", "rbf"],
        nargs="?",
        default="False",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        help="set training batch size. Defaults to 256",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n-components",
        "-k",
        help="set the number of components for the unsupervised model. Defaults to 5",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--encoding-size",
        "-es",
        help="set the number of dimensions of the latent space. 16 by default",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--embedding-model",
        "-embedding",
        help="Algorithm to use to embed and cluster the time series. Must be one of: VQVAE (default), VaDE, "
        "or Contrastive",
        nargs="?",
        choices=["VQVAE", "VaDE", "Contrastive"],
        default="VQVAE",
    )
    parser.add_argument(
        "--encoder-type",
        "-encoder",
        help="Encoder architecture to use when embedding the time series. Must be one of: recurrent (default), "
        "TCN, or transformer",
        nargs="?",
        choices=["recurrent", "TCN", "transformer"],
        default="recurrent",
    )
    parser.add_argument(
        "--exclude-bodyparts",
        "-exc",
        help="Excludes the indicated bodyparts from all analyses. "
        "It should consist of several values separated by commas.",
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
        choices=[False, "bayopt", "hyperband"],
        nargs="?",
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
        "It must be one of coords (features treated as independent and passed as a tensor) "
        "and graph (default - animals represented as graphs, with coords and speeds per bodypart as node features, "
        "and distances as edge features).",
        type=str,
        default="graph",
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
        default=0.0,
    )
    parser.add_argument(
        "--cat-kl-loss",
        "-catkl",
        help="If > 0, adds a regularization term that minimizes the KL divergence between cluster assignment "
        "frequencies and a uniform distribution",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--smooth-alpha",
        "-sa",
        help="Sets the exponential smoothing factor to apply to the input data. "
        "Float between 0 and 1 (lower is more smooting)",
        type=float,
        default=2,
    )
    parser.add_argument("--train-path", "-tp", help="set training set path", type=str)
    parser.add_argument(
        "--val-num",
        "-vn",
        help="set number of videos of the training" "set to use for validation",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--window-size",
        "-ws",
        help="Sets the sliding window size to be used when building both training and validation sets. Defaults to 15",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--window-step",
        "-wt",
        help="Sets the sliding window step to be used when building both training and validation sets. "
        "Defaults to 5",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-epochs",
        "-epochs",
        help="Sets the maximum number of epochs to train. It's usually cut short by early stopping.",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--load-project",
        "-load",
        help="If provided, loads an existing project instead of creating a new one. Use to avoid recomputing.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run",
        "-rid",
        help="Sets the run ID of the experiment (for naming output files only). "
        "If 0 (default), uses a timestamp instead",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--exp-condition-path",
        "-ec",
        help="If provided, loads the given experimental condition file. None by default.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

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
        encoder_type = args.encoder_type
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
        max_epochs = args.max_epochs
        load_project = args.load_project
        exp_condition_path = args.exp_condition_path
        run = args.run

    except TypeError:
        raise ValueError(
            "One or more mandatory arguments are missing. Use --help for details on how to run the CLI"
        )

    if not train_path:
        raise ValueError("Set a valid data path for the training to run")

    assert input_type in [
        "coords",
        "graph",
    ], "Invalid input type. Type python model_training.py -h for help."

    # Logs hyperparameters  if specified on the --logparam CLI argument
    logparam = {"encoding": encoding_size, "k": n_components}

    if load_project is None:
        # noinspection PyTypeChecker
        project_coords = deepof.data.Project(
            animal_ids=animal_ids.split(","),
            arena="circular-autodetect",
            video_scale=arena_dims,
            enable_iterative_imputation=250,
            exclude_bodyparts=exclude_bodyparts,
            exp_conditions=treatment_dict,
            project_path=train_path,
            video_path=os.path.join(train_path, "Videos"),
            table_path=os.path.join(train_path, "Tables"),
            project_name="deepof_experiments_{}".format(time_stamp),
            smooth_alpha=smooth_alpha,
            table_format="autodetect",
            video_format=".mp4",
        )

        project_coords = project_coords.create(verbose=True)

    else:
        project_coords = deepof.data.load_project(load_project)

    # If provided, load experimental conditions
    if exp_condition_path is not None:
        project_coords.load_exp_conditions(exp_condition_path)

    print("Preprocessing data...")

    if input_type == "coords":

        # Coordinates for training data
        to_preprocess = project_coords.get_coords(
            center="Center",
            align="Spine_1",
            align_inplace=True,
            propagate_labels=False,
            propagate_annotations=False,
            selected_id=animal_to_preprocess,
        )

        preprocessed_object, global_scaler = to_preprocess.preprocess(
            window_size=window_size,
            window_step=window_step,
            automatic_changepoints=automatic_changepoints,
            scale="standard",
            test_videos=val_num,
            shuffle=True,
        )

        print("Training set shape:", preprocessed_object[0].shape)
        print("Validation set shape:", preprocessed_object[2].shape)

    elif input_type == "graph":

        # Get graph dataset
        (
            preprocessed_object,
            adjacency_matrix,
            to_preprocess,
            global_scaler,
        ) = project_coords.get_graph_dataset(
            animal_id=animal_to_preprocess,
            center="Center",
            align="Spine_1",
            preprocess=True,
            window_size=window_size,
            window_step=window_step,
            automatic_changepoints=automatic_changepoints,
            test_videos=val_num,
            scale="standard",
            shuffle=True,
        )

        print("Training node set shape:", preprocessed_object[0].shape)
        print("Training edge set shape:", preprocessed_object[1].shape)
        print("Validation node set shape:", preprocessed_object[3].shape)
        print("Validation edge set shape:", preprocessed_object[4].shape)

    print("Done!")

    # Proceed with training mode. Fit autoencoder with the same parameters,
    # as many times as specified by runs
    if not tune:

        # noinspection PyUnboundLocalVariable
        trained_models = project_coords.deep_unsupervised_embedding(
            preprocessed_object,
            adjacency_matrix=(None if input_type == "coords" else adjacency_matrix),
            batch_size=batch_size,
            latent_dim=encoding_size,
            embedding_model=embedding_model,
            encoder_type=encoder_type,
            n_components=n_components,
            output_path=output_path,
            save_checkpoints=False,
            save_weights=True,
            input_type=input_type,
            # Parameters that control the training process
            kmeans_loss=kmeans_loss,
            reg_cat_clusters=cat_kl_loss,
            epochs=max_epochs,
            run=run,
        )

        # Get embeddings, soft_counts, and breaks per video
        embeddings, soft_counts, breaks = deepof.model_utils.embedding_per_video(
            coordinates=my_deepof_project,
            to_preprocess=to_preprocess,
            model=trained_model,
            animal_id=animal_to_preprocess,
            global_scaler=global_scaler,
        )

        with open(
            os.path.join(
                output_path,
                "deepof_unsupervised_{}_encoder_{}_encodings_input={}_k={}_latdim={}_changepoints_{}_kmeans_loss={}_run={}.pkl".format(
                    embedding_model,
                    encoder_type,
                    input_type,
                    n_components,
                    encoding_size,
                    automatic_changepoints,
                    kmeans_loss,
                    run,
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
        ) = deepof.model_utils.get_callbacks(
            input_type=input_type,
            cp=False,
            logparam=logparam,
            outpath=output_path,
            kmeans_loss=kmeans_loss,
            run=run,
        )

        best_hyperparameters, best_model = deepof.model_utils.tune_search(
            preprocessed_object=[X_train, y_train, X_val, y_val],
            batch_size=batch_size,
            encoding_size=encoding_size,
            hypertun_trials=hypertun_trials,
            hpt_type=tune,
            k=n_components,
            project_name="{}-based_VQVAE_{}".format(input_type, tune.capitalize()),
            callbacks=[
                tensorboard_callback,
                reduce_lr_callback,
                deepof.model_utils.CustomStopper(
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

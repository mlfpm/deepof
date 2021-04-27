# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Model training command line tool for the deepof package.
usage: python -m examples.model_training -h

"""

from deepof.data import *
from deepof.train_utils import *
from deepof.utils import *

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
    "--kl-annealing-mode",
    "-klam",
    help="Weight annealing to use for ELBO loss. Can be one of 'linear' and 'sigmoid'",
    default="linear",
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
    "--entropy-knn",
    "-entminn",
    help="number of nearest neighbors to take into account when computing latent space entropy",
    default=100,
    type=int,
)
parser.add_argument(
    "--entropy-samples",
    "-ents",
    help="Samples to use to compute cluster purity",
    default=10000,
    type=int,
)
parser.add_argument(
    "--latent-reg",
    "-lreg",
    help="Sets the strategy to regularize the latent mixture of Gaussians. "
    "It has to be one of none, categorical (an elastic net penalty is applied to the categorical distribution),"
    "variance (l2 penalty to the variance of the clusters) or categorical+variance. Defaults to none.",
    default="none",
    type=str,
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
    "--mmd-annealing-mode",
    "-mmdam",
    help="Weight annealing to use for MMD loss. Can be one of 'linear' and 'sigmoid'",
    default="linear",
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
    "--next-sequence-prediction",
    "-nspred",
    help="Activates the next sequence prediction branch of the variational Seq 2 Seq model with the specified weight. "
    "Defaults to 0.0 (inactive)",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--phenotype-prediction",
    "-ppred",
    help="Activates the phenotype classification branch with the specified weight. Defaults to 0.0 (inactive)",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--rule-based-prediction",
    "-rbpred",
    help="Activates the rule-based trait prediction branch of the variational Seq 2 Seq model "
    "with the specified weight Defaults to 0.0 (inactive)",
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
    "--run",
    "-rid",
    help="Sets the run ID of the experiment (for naming output files only). If 0 (default), uses a timestamp instead",
    type=int,
    default=0,
)

args = parser.parse_args()

animal_id = args.animal_id
arena_dims = args.arena_dims
batch_size = args.batch_size
hypertun_trials = args.hpt_trials
encoding_size = args.encoding_size
exclude_bodyparts = [i for i in args.exclude_bodyparts.split(",") if i]
gaussian_filter = args.gaussian_filter
hparams = args.hyperparameters if args.hyperparameters is not None else {}
input_type = args.input_type
k = args.components
kl_annealing_mode = args.kl_annealing_mode
kl_wu = args.kl_warmup
entropy_knn = args.entropy_knn
entropy_samples = args.entropy_samples
latent_reg = args.latent_reg
loss = args.loss
mmd_annealing_mode = args.mmd_annealing_mode
mmd_wu = args.mmd_warmup
mc_kl = args.montecarlo_kl
output_path = os.path.join(args.output_path)
overlap_loss = args.overlap_loss
next_sequence_prediction = float(args.next_sequence_prediction)
phenotype_prediction = float(args.phenotype_prediction)
rule_based_prediction = float(args.rule_based_prediction)
smooth_alpha = args.smooth_alpha
train_path = os.path.abspath(args.train_path)
tune = args.hyperparameter_tuning
val_num = args.val_num
window_size = args.window_size
window_step = args.window_step
run = args.run

if not train_path:
    raise ValueError("Set a valid data path for the training to run")

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
treatment_dict = load_treatments(train_path)

# Logs hyperparameters  if specified on the --logparam CLI argument
logparam = {
    "encoding": encoding_size,
    "k": k,
    "loss": loss,
}
if next_sequence_prediction:
    logparam["next_sequence_prediction_weight"] = next_sequence_prediction
if phenotype_prediction:
    logparam["phenotype_prediction_weight"] = phenotype_prediction
if rule_based_prediction:
    logparam["rule_based_prediction_weight"] = rule_based_prediction

# noinspection PyTypeChecker
project_coords = project(
    animal_ids=tuple([animal_id]),
    arena="circular",
    arena_dims=tuple([arena_dims]),
    enable_iterative_imputation=True,
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
    propagate_labels=(phenotype_prediction > 0),
    propagate_annotations=(
        False if not rule_based_prediction else project_coords.rule_based_annotation()
    ),
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
if any([phenotype_prediction, rule_based_prediction]):
    print("Training set label shape:", y_train.shape)
    print("Validation set label shape:", y_val.shape)

print("Done!")

# Proceed with training mode. Fit autoencoder with the same parameters,
# as many times as specified by runs
if not tune:

    trained_models = project_coords.deep_unsupervised_embedding(
        (X_train, y_train, X_val, y_val),
        batch_size=batch_size,
        encoding_size=encoding_size,
        hparams={},
        kl_annealing_mode=kl_annealing_mode,
        kl_warmup=kl_wu,
        log_history=True,
        log_hparams=True,
        loss=loss,
        mmd_annealing_mode=mmd_annealing_mode,
        mmd_warmup=mmd_wu,
        montecarlo_kl=mc_kl,
        n_components=k,
        output_path=output_path,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_based_prediction,
        save_checkpoints=False,
        save_weights=True,
        reg_cat_clusters=("categorical" in latent_reg),
        reg_cluster_variance=("variance" in latent_reg),
        entropy_samples=entropy_samples,
        entropy_knn=entropy_knn,
        input_type=input_type,
        run=run,
    )

else:
    # Runs hyperparameter tuning with the specified parameters and saves the results
    run_ID, tensorboard_callback, entropy, onecycle = get_callbacks(
        X_train=X_train,
        batch_size=batch_size,
        phenotype_prediction=phenotype_prediction,
        next_sequence_prediction=next_sequence_prediction,
        rule_based_prediction=rule_base_prediction,
        loss=loss,
        loss_warmup=kl_wu,
        warmup_mode=kl_annealing_mode,
        X_val=(X_val if X_val.shape != (0,) else None),
        input_type=input_type,
        cp=False,
        entropy_samples=entropy_samples,
        entropy_knn=entropy_knn,
        logparam=logparam,
        outpath=output_path,
        run=run,
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
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_base_prediction,
        project_name="{}-based_GMVAE_{}".format(input_type, tune.capitalize()),
        callbacks=[
            tensorboard_callback,
            onecycle,
            entropy,
            CustomStopper(
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
            "{}-based_GMVAE_{}_params.pickle".format(input_type, tune.capitalize()),
        ),
        "wb",
    ) as handle:
        pickle.dump(
            best_hyperparameters.values, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

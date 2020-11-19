# @author lucasmiranda42

# Add to the system path for the script to find deepof
import sys

sys.path.insert(0, "../../")

from train_utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error
from deepof.data import *
from deepof.models import *
from deepof.utils import *
from tqdm import tqdm
import argparse
import numpy as np
import os, re
import pandas as pd
import umap


def flip_axes(projections):
    """Flips projected PCA axes for subsequent latent representations of the input
     to remain closer to one another in the visualization"""

    projs = deepcopy(projections)

    for i, proj in enumerate(projections[1:]):

        if np.linalg.norm(projs[i][:, 0] - projs[i + 1][:, 0]) > np.linalg.norm(
            projs[i][:, 0] + projs[i + 1][:, 0]
        ):

            projs[i + 1][:, 0] = -proj[:, 0]

        if np.linalg.norm(projs[i][:, 1] - projs[i + 1][:, 1]) > np.linalg.norm(
            projs[i][:, 1] + projs[i + 1][:, 1]
        ):

            projs[i + 1][:, 1] = -proj[:, 1]

    return projs


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
parser.add_argument(
    "--checkpoint-path",
    "-cp",
    help="Sets the path into which the checkpoint files of the desired model are located",
    type=str,
)
parser.add_argument(
    "--samples",
    "-sa",
    help="Sets the number of samples (without replacement) to take from the validation set.",
    default=5000,
    type=int,
)
parser.add_argument(
    "--reducer",
    "-r",
    help="Sets the dimensionality reduction method of preference to represent the latent space",
    default="umap",
    type=str,
)

args = parser.parse_args()

samples = args.samples
red = args.reducer
animal_id = args.animal_id
arena_dims = args.arena_dims
batch_size = args.batch_size
bayopt_trials = args.bayopt
checkpoints = args.Zcheckpoint_path
exclude_bodyparts = tuple(args.exclude_bodyparts.split(","))
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


# Load checkpoints and build dataframe with predictions
path = checkpoints
checkpoints = sorted(
    list(
        set(
            [
                path + re.findall('(.*\.ckpt).data', i)[0]
                for i in os.listdir(path)
                if "ckpt.data" in i
            ]
        )
    )
)

pttest_idx = np.random.choice(list(range(X_train.shape[0])), samples)
pttest = X_val[pttest_idx]

# Instanciate all models
clusters = []
predictions = []
reconstructions = []

(encoder, generator, grouper, gmvaep, _, _,) = SEQ_2_SEQ_GMVAE(
    loss=loss,
    number_of_components=k,
    predictor=predictor,
    kl_warmup_epochs=10,
    mmd_warmup_epochs=10,
    **hparams
).build(pttest.shape)
gmvaep.build(pttest.shape)

predictions.append(encoder.predict(pttest))
if predictor:
    reconstructions.append(gmvaep.predict(pttest)[0])
else:
    reconstructions.append(gmvaep.predict(pttest))

print("Building predictions from pretrained models...")

for checkpoint in tqdm(checkpoints):

    gmvaep.load_weights(checkpoint)
    clusters.append(grouper.predict(pttest))
    predictions.append(encoder.predict(pttest))
    if predictor:
        reconstructions.append(gmvaep.predict(pttest)[0])
    else:
        reconstructions.append(gmvaep.predict(pttest))

print("Done!")

print("Reducing latent space to 2 dimensions for dataviz...")
if red == "LDA":
    reducer = LinearDiscriminantAnalysis(n_components=2)

elif red == "UMAP":
    reducer = umap.UMAP(n_components=2)

elif red == "tSNE":
    reducer = TSNE(n_components=2)

encs = []
for i in range(len(checkpoints) + 1):

    if i == 0:
        clusts = (
            np.array([int(i) for i in np.random.uniform(0, k, samples)])
            if variational
            else np.zeros(samples)
        )
        if red == "LDA":
            encs.append(reducer.fit_transform(predictions[i], clusts))
        else:
            encs.append(reducer.fit_transform(predictions[i]))
    else:
        if red == "LDA":
            encs.append(
                reducer.fit_transform(
                    predictions[i], np.argmax(clusters[i - 1], axis=1)
                )
            )
        else:
            encs.append(reducer.fit_transform(predictions[i]))


# As projection direction is difficult to predict in LDA,
# axes are flipped to maintain subsequent representations
# of the input closer to one another
flip_encs = flip_axes(encs)
print("Done!")

# Latent space animated PCA over epochs
dfencs = pd.DataFrame(MinMaxScaler().fit_transform(np.concatenate(flip_encs)))
dfcats = pd.concat(
    [
        pd.DataFrame(
            (
                [str(int(i)) for i in np.random.uniform(0, k, samples)]
                if variational
                else np.zeros(samples)
            )
        ),
        pd.DataFrame(
            np.array(np.argmax((np.concatenate(clusters)), axis=1), dtype=str)
        ),
    ],
).reset_index(drop=True)

dfcats_max = pd.concat(
    [
        pd.DataFrame((np.zeros(samples))),
        pd.DataFrame(np.array(np.max((np.concatenate(clusters)), axis=1), dtype=str)),
    ],
).reset_index(drop=True)

dfencs = pd.concat([dfencs, dfcats, dfcats_max], axis=1)

dfencs["epoch"] = np.array(
    [j + 1 for j in range(len(flip_encs)) for i in range(len(flip_encs[0]))]
)
dfencs.columns = ["x", "y", "cluster", "confidence", "epoch"]
dfencs["trajectories"] = np.tile(pttest[:, 6, 1], len(checkpoints) + 1)
dfencs["reconstructions"] = np.concatenate(reconstructions)[:, 6, 1]

# Cluster membership animated over epochs
clust_occur = pd.DataFrame(
    dfencs.loc[:, ["cluster", "epoch"]].groupby(["epoch", "cluster"]).cluster.count()
)
clust_occur.rename(columns={"cluster": "count"}, inplace=True)
clust_occur = clust_occur.reset_index()

# MAE over epochs
maedf = pd.DataFrame()
maedf["epoch"] = list(range(1, len(checkpoints) + 2))
maedf["mae"] = [
    mean_absolute_error(
        dfencs.loc[dfencs["epoch"] == i, "trajectories"],
        dfencs.loc[dfencs["epoch"] == i, "reconstructions"],
    )
    for i in range(1, len(checkpoints) + 2)
]

# Time ID
time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Save dataframes to .h5
dfencs.to_hdf("dash_data_1_{}.h5".format(time), key="df", mode="w")
clust_occur.to_hdf("dash_data_2_{}.h5".format(time), key="df", mode="w")
maedf.to_hdf("dash_data_3_{}.h5".format(time), key="df", mode="w")

# @author lucasmiranda42

# Add to the system path for the script to find deepof
import sys

sys.path.insert(1, "../")

from copy import deepcopy
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from source.preprocess import *
from source.models import *
from tqdm import tqdm
import argparse
import numpy as np
import os, pickle, re
import pandas as pd
import umap


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

parser.add_argument("--data-path", "-vp", help="set validation set path", type=str)
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
    "-pred",
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
    "--checkpoint-path",
    "-cp",
    help="Sets the path into which the checkpoint files of the desired model are located",
    type=str,
)
parser.add_argument(
    "--samples",
    "-s",
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
data_path = os.path.abspath(args.data_path)
input_type = args.input_type
k = args.components
predictor = args.predictor
variational = bool(args.variational)
loss = args.loss
hparams = args.hyperparameters
encoding = args.encoding_size
checkpoints = args.checkpoint_path
samples = args.samples
red = args.reducer

if not data_path:
    raise ValueError("Set a valid data path for the data to be loaded")
assert input_type in [
    "coords",
    "dists",
    "angles",
    "coords+dist",
    "coords+angle",
    "coords+dist+angle",
], "Invalid input type. Type python train_viz_app.py -h for help."

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

# with open(
#     os.path.join(
#         data_path, [i for i in os.listdir(data_path) if i.endswith(".pickle")][0]
#     ),
#     "rb",
# ) as handle:
#     Treatment_dict = pickle.load(handle)

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

DLC_social = project(
    path=data_path,  # Path where to find the required files
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
    # exp_conditions=Treatment_dict,
)


DLC_social_coords = DLC_social.run(verbose=True)

# Coordinates for training data
coords1 = DLC_social_coords.get_coords()
distances1 = DLC_social_coords.get_distances()
angles1 = DLC_social_coords.get_angles()
coords_distances1 = merge_tables(coords1, distances1)
coords_angles1 = merge_tables(coords1, angles1)
coords_dist_angles1 = merge_tables(coords1, distances1, angles1)

input_dict = {
    "coords": coords1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
    "dists": distances1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
    "angles": angles1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
    "coords+dist": coords_distances1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
    "coords+angle": coords_angles1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
    "coords+dist+angle": coords_dist_angles1.preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=True,
    ),
}


# Load checkpoints and build dataframe with predictions
path = checkpoints
checkpoints = sorted(
    list(
        set(
            [
                path + re.findall("(.*\.ckpt).data", i)[0]
                for i in os.listdir(path)
                if "ckpt.data" in i
            ]
        )
    )
)

pttest_idx = np.random.choice(list(range(input_dict[input_type].shape[0])), samples)
pttest = input_dict[input_type][pttest_idx]

# Instanciate all models
clusters = []
predictions = []
reconstructions = []

if not variational:
    encoder, decoder, ae = SEQ_2_SEQ_AE(pttest.shape, **hparams).build()
    ae.build(pttest.shape)

    predictions.append(encoder.predict(pttest))
    reconstructions.append(ae.predict(pttest))

else:
    (encoder, generator, grouper, gmvaep, _, _,) = SEQ_2_SEQ_GMVAE(
        pttest.shape,
        loss=loss,
        number_of_components=k,
        predictor=predictor,
        kl_warmup_epochs=10,
        mmd_warmup_epochs=10,
        **hparams
    ).build()
    gmvaep.build(pttest.shape)

    predictions.append(encoder.predict(pttest))
    if predictor:
        reconstructions.append(gmvaep.predict(pttest)[0])
    else:
        reconstructions.append(gmvaep.predict(pttest))


print("Building predictions from pretrained models...")

for checkpoint in tqdm(checkpoints):

    if variational:
        gmvaep.load_weights(checkpoint)
        clusters.append(grouper.predict(pttest))
        predictions.append(encoder.predict(pttest))
        if predictor:
            reconstructions.append(gmvaep.predict(pttest)[0])
        else:
            reconstructions.append(gmvaep.predict(pttest))

    else:
        ae.load_weights(checkpoint)
        clusters.append(np.zeros(samples))
        predictions.append(encoder.predict(pttest))
        reconstructions.append(ae.predict(pttest))

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
                reducer.fit_transform(predictions[i], np.argmax(clusters[i - 1], axis=1))
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

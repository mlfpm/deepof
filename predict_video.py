# @author lucasmiranda42

# Add to the system path for the script to find deepof
import sys

sys.path.insert(1, "../")

from source.preprocess import *
from source.models import *
from tqdm import tqdm
import argparse
import cv2
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

parser.add_argument("--data-path", "-dp", help="set validation set path", type=str)
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
    "--model-path",
    "-mp",
    help="Sets the path into which the desired model weights can be found",
    type=str,
)
parser.add_argument(
    "--video-name", "-n", help="Sets the video to process", type=str,
)
parser.add_argument(
    "--frame-limit",
    "-f",
    help="Sets the maximum number of frames of video to process",
    type=int,
    default=-1,
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
model_path = args.model_path
video_name = args.video_name
frame_limit = args.frame_limit

if not data_path:
    raise ValueError("Set a valid data path for the data to be loaded")
assert input_type in [
    "coords",
    "dists",
    "angles",
    "coords+dist",
    "coords+angle",
    "dists+angle",
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
    path=os.path.join(data_path),  # Path where to find the required files
    smooth_alpha=0.90,  # Alpha value for exponentially weighted smoothing
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
)


DLC_social_coords = DLC_social.run(verbose=True)

# Coordinates for training data
coords1 = DLC_social_coords.get_coords(center="B_Center")
distances1 = DLC_social_coords.get_distances()
angles1 = DLC_social_coords.get_angles()
coords_distances1 = merge_tables(coords1, distances1)
coords_angles1 = merge_tables(coords1, angles1)
dists_angles1 = merge_tables(distances1, angles1)
coords_dist_angles1 = merge_tables(coords1, distances1, angles1)

input_dict = {
    "coords": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
    "dists": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
    "angles": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
    "coords+dist": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
    "coords+angle": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
    "dists+angle": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=10,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
    ),
    "coords+dist+angle": table_dict(
        ((k, coords1[k]) for k in [video_name]), typ="coords"
    ).preprocess(
        window_size=11,
        window_step=1,
        scale=True,
        random_state=42,
        filter="gaussian",
        sigma=55,
        shuffle=False,
    ),
}

# Instantiate model and load trained model weights
(encoder, generator, grouper, ae, _, _,) = SEQ_2_SEQ_GMVAE(
    input_dict[input_type].shape,
    loss=loss,
    number_of_components=k,
    predictor=predictor,
    kl_warmup_epochs=10,
    mmd_warmup_epochs=10,
    **hparams
).build()
ae.build(input_dict[input_type].shape)

ae.load_weights(model_path)

# Predict cluster labels per frame
frame_labels = np.argmax(grouper.predict(input_dict[input_type]), axis=1)

print(frame_labels)
print(frame_labels.shape)

# Open video and print cluster label
cap = cv2.VideoCapture(
    os.path.join(
        data_path,
        "Videos",
        [i for i in os.listdir(os.path.join(data_path, "Videos")) if video_name in i][
            0
        ],
    ),
)

# Loop over the first frames in the video to get resolution and center of the arena
if frame_limit == -1:
    frame_limit = np.inf
fnum = 0
while cap.isOpened() and fnum < frame_limit:
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break



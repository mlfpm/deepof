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

parser.add_argument(
    "--model-path", "-mp", help="set the path where to find the trained model", type=str
)
parser.add_argument("--data-path", "-vp", help="set data path", type=str)
parser.add_argument("--video-name", "-n", help="name of video of interest", type=str)
parser.add_argument(
    "-frames", "-f", help="set frames to analyse. Defaults to all", type=int, default=-1
)

args = parser.parse_args()
video = args.video_name
data = args.data_path
model = args.model_path
frames = args.frames

# TODO:
#       - Load video, data and model
#       - predict clusters for each frame of the video
#       - output video with cluster labels

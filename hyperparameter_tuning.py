# @author lucasmiranda42

from datetime import datetime
from source.preprocess import *
from source.hypermodels import *
from kerastuner import BayesianOptimization
from sys import argv
from tensorflow import keras

script, input_type, hyp, path = argv

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

with open(path + "DLC_social_1_exp_conditions.pickle", "rb") as handle:
    Treatment_dict = pickle.load(handle)

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

DLC_social_1 = project(
    path=path,  # Path where to find the required files
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
    exp_conditions=Treatment_dict,
)

DLC_social_1_coords = DLC_social_1.run(verbose=True)

coords = DLC_social_1_coords.get_coords()
distances = DLC_social_1_coords.get_distances()
angles = DLC_social_1_coords.get_angles()
coords_distances = merge_tables(coords, distances)
coords_angles = merge_tables(coords, angles)
coords_dist_angles = merge_tables(coords, distances, angles)

coords_train, coords_test = coords.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
dist_train, dist_test = distances.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
angles_train, angles_test = angles.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
coords_dist_train, coords_dist_test = coords_distances.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
coords_angles_train, coords_angles_test = coords_angles.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
coords_dist_angles_train, coords_dist_angles_test = coords_dist_angles.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)

print("Training set of shape: {}".format(coords_train.shape))
print("Validation set of shape: {}".format(coords_test.shape))
print()
print("Training set of shape: {}".format(dist_train.shape))
print("Validation set of shape: {}".format(dist_test.shape))


def tune_search(train, test, project_name, hyp):
    """Define the search space using keras-tuner and bayesian optimization"""
    COORDS_INPUT_SHAPE = train.shape
    if hyp == "S2SAE":
        hypermodel = SEQ_2_SEQ_AE(input_shape=COORDS_INPUT_SHAPE)
    elif hyp == "S2SVAE":
        hypermodel = SEQ_2_SEQ_VAE(input_shape=COORDS_INPUT_SHAPE, loss="ELBO+MMD")
    elif hyp == "S2SVAE-MMD":
        hypermodel = SEQ_2_SEQ_VAE(input_shape=COORDS_INPUT_SHAPE, loss="MMD")
    elif hyp == "S2SVAE-ELBO":
        hypermodel = SEQ_2_SEQ_VAE(input_shape=COORDS_INPUT_SHAPE, loss="ELBO")
    else:
        raise ValueError(
            "Hypermodel not recognised. Try one of S2SAE, S2SVAE, S2SVAE-ELBO or S2SVAE-MMD"
        )

    tuner = BayesianOptimization(
        hypermodel,
        max_trials=25,
        executions_per_trial=1,
        objective="val_mae",
        seed=42,
        directory="BayesianOptx",
        project_name=project_name,
    )

    print(tuner.search_space_summary())

    tuner.search(
        train,
        train,
        epochs=30,
        validation_data=(test, test),
        verbose=1,
        batch_size=256,
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping("val_mae", patience=3),
        ],
    )

    print(tuner.results_summary())
    return tuner.get_best_models()[0]


if input_type == "coords":
    best_model = tune_search(
        coords_train, coords_test, "Coord-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", hyp=hyp
    )
    best_model.save("Coords-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "dists":
    best_model = tune_search(
        dist_train, dist_test, "Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", hyp=hyp
    )
    best_model.save("Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "angles":
    best_model = tune_search(
        angles_train, angles_test, "Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", hyp=hyp
    )
    best_model.save("Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "coords+dist":
    best_model = tune_search(
        coords_dist_train,
        coords_dist_test,
        "Coords+Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5",
        hyp=hyp,
    )
    best_model.save("Coords+Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "coords+angle":
    best_model = tune_search(
        coords_angles_train,
        coords_angles_test,
        "Coords+Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5",
        hyp=hyp,
    )
    best_model.save("Coords+Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "coords+dist+angle":
    best_model = tune_search(
        coords_dist_angles_train,
        coords_dist_angles_test,
        "Coords+Dist+Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5",
        hyp=hyp,
    )
    best_model.save(
        "Coords+Dist+Angle-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf"
    )

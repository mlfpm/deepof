# @author lucasmiranda42

from source.classes import *
from source.hypermodels import *
from kerastuner import BayesianOptimization
from multiprocessing import cpu_count
from sys import argv

script, input_type, path = argv

with open(path + "DLC_social_1_exp_conditions.pickle", "rb") as handle:
    Treatment_dict = pickle.load(handle)

DLC_social_1 = get_coordinates(
    path=path,  # Path where to find the required files
    p=cpu_count(),  # Number of processes used for parallelization
    smooth_alpha=0.1,  # Alpha value for exponentially weighted smoothing
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
    arena="circular",  # Type of arena used in the experiments
    arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
    exp_conditions=Treatment_dict,
)

DLC_social_1_coords = DLC_social_1.run()

coords = DLC_social_1_coords.get_coords()
distances = DLC_social_1_coords.get_distances
coords_train, coords_test = coords.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)
dist_train, dist_test = distances.preprocess(
    window_size=50, window_step=10, test_proportion=0.05, scale=True, random_state=42
)

print("Training set of shape: {}".format(coords_train.shape))
print("Validation set of shape: {}".format(coords_test.shape))
print()
print("Training set of shape: {}".format(dist_train.shape))
print("Validation set of shape: {}".format(dist_test.shape))


def tune_search(train, test, project_name):
    """Define the search space using keras-tuner and bayesian optimization"""
    COORDS_INPUT_SHAPE = train.shape
    hypermodel = SEQ_2_SEQ_AE(input_shape=COORDS_INPUT_SHAPE)

    tuner = BayesianOptimization(
        hypermodel,
        max_trials=100,
        executions_per_trial=3,
        objective="val_mae",
        seed=42,
        directory="BayesianOptx",
        project_name=project_name,
        # distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    print(tuner.search_space_summary())

    tuner.search(
        train,
        train,
        epochs=30,
        validation_data=(test, test),
        verbose=1,
        batch_size=1024,
        callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=3)],
    )

    print(tuner.results_summary())
    return tuner.get_best_models()[0]


if input_type == "coords":
    best_model = tune_search(
        coords_train, coords_test, "Coord-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5"
    )
    best_model.save("Coords-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

elif input_type == "dist":
    best_model = tune_search(
        dist_train, dist_test, "Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5"
    )
    best_model.save("Dist-based_SEQ2SEQ_AE_BAYESIAN_OPT.h5", save_format="tf")

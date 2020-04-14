import acrona, pickle
from AutoEncoder_HyperModels import *
from kerastuner import BayesianOptimization

with open(
    "../../Desktop/DLC_social_1/DLC_social_1_exp_conditions.pickle", "rb"
) as handle:
    Treatment_dict = pickle.load(handle)

DLC_social_1 = acrona.get_coordinates(
    path="../../Desktop/DLC_social_1/",  # Path where to find the required files
    p=16,  # Number of processes used for parallelization
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
    ego="B_Center",
    arena="circular",  # Type of arena used in the experiments
    arena_dims=[380],  # Dimensions of the arena. Just one if it's circular
    video_format=".mp4",
    table_format=".h5",
    exp_conditions=Treatment_dict,
    verbose=True,
)

DLC_social_1_coords = DLC_social_1.run()

coords = DLC_social_1_coords.get_coords()
distances = DLC_social_1_coords.get_distances()
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

# Trajectory-based AE hyperparameter tuning
COORDS_INPUT_SHAPE = coords_train.shape

coords_hypermodel = SEQ_2_SEQ_AE(input_shape=COORDS_INPUT_SHAPE)

coords_tuner = BayesianOptimization(
    coords_hypermodel,
    max_trials=100,
    executions_per_trial=3,
    objective="val_mae",
    seed=42,
    directory="BayesianOptx",
    project_name="Trajectory_seq2seq_AE_tuning",
    distribution_strategy=tf.distribute.MirroredStrategy(),
)

# Distance-based AE hyperparameter tuning
DIST_INPUT_SHAPE = dist_train.shape

dist_hypermodel = SEQ_2_SEQ_AE(input_shape=DIST_INPUT_SHAPE)

dist_tuner = BayesianOptimization(
    dist_hypermodel,
    max_trials=100,
    executions_per_trial=3,
    objective="val_mae",
    seed=42,
    directory="BayesianOptx",
    project_name="Distance_seq2seq_AE_tuning",
    distribution_strategy=tf.distribute.MirroredStrategy(),
)


print(coords_tuner.search_space_summary())
print(dist_tuner.search_space_summary())

# @author lucasmiranda42

import cv2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import pims
import re
import scipy
import seaborn as sns
from itertools import cycle, combinations
from joblib import Parallel, delayed
from numba import jit
from numpy.core.umath_tests import inner1d
from scipy import spatial
from sklearn import mixture
from tqdm import tqdm_notebook as tqdm


##### QUALITY CONTROL AND PREPROCESSING #####

# Likelihood quality control
def Likelihood_qc(dframe, threshold=0.9):
    """Returns only rows where all lilelihoods are above a specified threshold"""
    Likes = np.array([dframe[i]["likelihood"] for i in list(dframe.columns.levels[0])])
    Likes = np.nan_to_num(Likes, nan=1.0)
    return np.all(Likes > threshold, axis=0)


def bp2polar(tab):
    tab_ = np.array(tab)
    complex_ = tab_[:, 0] + 1j * tab_[:, 1]
    polar = pd.DataFrame(np.array([abs(complex_), np.angle(complex_)]).T)
    polar.rename(columns={0: "rho", 1: "phi"}, inplace=True)
    return polar


def tab2polar(tabdict):
    result = []
    for df in list(tabdict.columns.levels[0]):
        result.append(bp2polar(tabdict[df]))
    result = pd.concat(result, axis=1)
    idx = pd.MultiIndex.from_product(
        [list(tabdict.columns.levels[0]), ["rho", "phi"]], names=["bodyparts", "coords"]
    )
    result.columns = idx
    return result


def compute_dist(pair_df, arena_abs, arena_rel):
    a, b = pair_df[:, :2], pair_df[:, 2:]
    ab = a - b
    dist = np.sqrt(inner1d(ab, ab))
    return pd.DataFrame(dist * arena_abs / arena_rel)


def bpart_distance(dataframe, arena_abs, arena_rel):
    indexes = combinations(dataframe.columns.levels[0], 2)

    dists = []
    for idx in indexes:
        dist = compute_dist(np.array(dataframe.loc[:, list(idx)]), arena_abs, arena_rel)
        dist.columns = [idx]
        dists.append(dist)

    return pd.concat(dists, axis=1)


def angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = inner1d(ba, bc) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    angle = np.arccos(cosine_angle)

    return angle


def angle_trio(array, degrees=False):
    a, b, c = array

    return np.array([angle(a, b, c), angle(a, c, b), angle(b, a, c),])


def smooth_boolean_array(a):
    """Returns a boolean array in which isolated appearances of a feature are smoothened"""
    for i in range(1, len(a) - 1):
        if a[i - 1] == a[i + 1]:
            a[i] = a[i - 1]
    return a == 1


def rolling_window(a, window_size, window_step, write=True):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=write
    )[::window_step]

    ##### IMAGE/VIDEO PROCESSING FUNCTIONS #####


def index_frames(video_list, sample=False, index=0, pkl=False):
    """Pickles a 4D numpy array per video in video list, for easy random access afterwards"""

    pbar = tqdm(total=len(video_list))

    for i, vid in enumerate(video_list):

        v = np.array(pims.PyAVReaderIndexed(vid))

        if sample:
            v = v[np.random.choice(v.shape[0], sample)]

        if type(index) != int:
            v = v[index]

        if pkl:
            with open(pkl, "wb") as f:
                pickle.dump(v, f, protocol=4)

        pbar.update(1)

    return True


@jit
def smooth_mult_trajectory(series, alpha=0.15):
    """smoothens a trajectory using exponentially weighted averages"""

    result = [series[0]]
    for n in range(len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

    return np.array(result)

    ##### BEHAVIOUR RECOGNITION FUNCTIONS #####


# Nose to Nose contact
def nose_to_nose(pos_dict, fnum, tol):
    """Takes DLC dataframe as input. Returns True when distances of both noses are closer to tolerance"""

    return np.linalg.norm(pos_dict["B_Nose"] - pos_dict["W_Nose"]) < tol


# Black nose to white tail contact
def nose_to_tail(pos_dict, fnum, tol, mouse1="B", mouse2="W"):
    """Takes DLC dataframe as input. Returns True when the distance of nose1 and tail2 are closer to tolerance"""

    return (
        np.linalg.norm(pos_dict[mouse1 + "_Nose"] - pos_dict[mouse2 + "_Tail_base"])
        < tol
    )


# Side by side (noses and tails close)
def side_by_side(pos_dict, fnum, tol, rev=False):
    """Takes DLC dataframe as input. Returns True when mice are side by side"""
    w_nose = pos_dict["W_Nose"]
    b_nose = pos_dict["B_Nose"]
    w_tail = pos_dict["W_Tail_base"]
    b_tail = pos_dict["B_Tail_base"]

    if rev:
        return (
            np.linalg.norm(w_nose - b_tail) < tol
            and np.linalg.norm(w_tail - b_nose) < tol
        )

    else:
        return (
            np.linalg.norm(w_nose - b_nose) < tol
            and np.linalg.norm(w_tail - b_tail) < tol
        )


def recognize_arena(
    Videos, vid_index, path=".", recoglimit=1, arena_type="circular",
):

    cap = cv2.VideoCapture(path + Videos[vid_index])

    # Loop over the first frames in the video to get resolution and center of the arena
    fnum, h, w = 0, None, None

    while cap.isOpened() and fnum < recoglimit:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if arena_type == "circular":

            # Detect arena and extract positions
            arena = circular_arena_recognition(frame)[0]
            if h == None and w == None:
                h, w = frame.shape[0], frame.shape[1]

        fnum += 1

    return arena


def circular_arena_recognition(frame):
    """Returns x,y position of the center and the radius of the recognised arena"""

    # Convert image to greyscale, threshold it, blur it and detect the biggest best fitting circle
    # using the Hough algorithm
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, 0)
    frame = cv2.medianBlur(thresh, 9)
    circle = cv2.HoughCircles(
        frame,
        cv2.HOUGH_GRADIENT,
        1,
        300,
        param1=50,
        param2=10,
        minRadius=0,
        maxRadius=0,
    )

    circles = []

    if circle is not None:
        circle = np.uint16(np.around(circle[0]))
        circles.append(circle)

    return circles[0]


def climb_wall(arena, pos_dict, fnum, tol, mouse):
    """Returns True if the specified mouse is climbing the wall"""

    nose = pos_dict[mouse + "_Nose"]
    center = np.array(arena[:2])

    return np.linalg.norm(nose - center) > arena[2] + tol


def rolling_speed(dframe, pause=10, rounds=5):
    """Returns the average speed over 10 frames in pixels per frame"""

    distances = np.linalg.norm(np.array(dframe) - np.array(dframe.shift()), axis=1)
    distances = pd.Series(distances, index=dframe.index)
    speeds = np.round(distances.rolling(pause).mean(), rounds)
    speeds[np.isnan(speeds)] = 0.0

    return speeds


def huddle(pos_dict, fnum, tol, tol2, mouse="B"):
    """Returns true when the specified mouse is huddling"""

    return (
        np.linalg.norm(pos_dict[mouse + "_Left_ear"] - pos_dict[mouse + "_Left_flank"])
        < tol
        and np.linalg.norm(
            pos_dict[mouse + "_Right_ear"] - pos_dict[mouse + "_Right_flank"]
        )
        < tol
        and np.linalg.norm(pos_dict[mouse + "_Center"] - pos_dict[mouse + "_Tail_base"])
        < tol2
    )


def following_path(distancedf, dframe, follower="B", followed="W", frames=20, tol=0):
    """Returns true if follower is closer than tol to the path that followed has walked over
    the last specified number of frames"""

    # Check that follower is close enough to the path that followed has passed though in the last frames
    shift_dict = {i: dframe[followed + "_Tail_base"].shift(i) for i in range(frames)}
    dist_df = pd.DataFrame(
        {
            i: np.linalg.norm(dframe[follower + "_Nose"] - shift_dict[i], axis=1)
            for i in range(frames)
        }
    )

    # Check that the animals are oriented follower's nose -> followed's tail
    right_orient1 = (
        distancedf[tuple(sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distancedf[tuple(sorted([follower + "_Tail_base", followed + "_Tail_base"]))]
    )

    right_orient2 = (
        distancedf[tuple(sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distancedf[tuple(sorted([follower + "_Nose", followed + "_Nose"]))]
    )

    return pd.Series(
        np.all(
            np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]),
            axis=0,
        ),
        index=dframe.index,
    )


def Single_behaviour_analysis(
    behaviour_name,
    treatment_dict,
    behavioural_dict,
    plot=False,
    stats=False,
    save=False,
    ylim=False,
):
    """Given the name of the behaviour, a dictionary with the names of the groups to compare, and a dictionary
       with the actual taggings, outputs a box plot and a series of significance tests amongst the groups"""

    beh_dict = {condition: [] for condition in treatment_dict.keys()}

    for condition in beh_dict.keys():
        for ind in treatment_dict[condition]:
            beh_dict[condition].append(
                np.sum(behavioural_dict[ind][behaviour_name])
                / len(behavioural_dict[ind][behaviour_name])
            )

    if plot:
        sns.boxplot(list(beh_dict.keys()), list(beh_dict.values()), orient="vertical")

        plt.title("{} across groups".format(behaviour_name))
        plt.ylabel("Proportion of frames")

        if ylim != False:
            plt.ylim(*ylim)

        plt.tight_layout()
        plt.savefig("Exploration_heatmaps.pdf")

        if save != False:
            plt.savefig(save)

        plt.show()

    if stats:
        for i in combinations(treatment_dict.keys(), 2):
            print(i)
            print(scipy.stats.mannwhitneyu(beh_dict[i[0]], beh_dict[i[1]]))

    return beh_dict

    ##### MAIN BEHAVIOUR TAGGING FUNCTION #####


def Tag_video(
    Tracks,
    Videos,
    Track_dict,
    Distance_dict,
    Like_QC_dict,
    vid_index,
    show=False,
    save=False,
    fps=25.0,
    speedpause=50,
    framelimit=np.inf,
    recoglimit=1,
    path="./",
    classifiers={},
):
    """Outputs a dataframe with the motives registered per frame. If mp4==True, outputs a video in mp4 format"""

    vid_name = re.findall("(.*?)_", Tracks[vid_index])[0]

    cap = cv2.VideoCapture(path + Videos[vid_index])
    dframe = Track_dict[vid_name]
    h, w = None, None
    bspeed, wspeed = None, None

    # Disctionary with motives per frame
    tagdict = {
        func: np.zeros(dframe.shape[0])
        for func in [
            "nose2nose",
            "bnose2tail",
            "wnose2tail",
            "sidebyside",
            "sidereside",
            "bclimbwall",
            "wclimbwall",
            "bspeed",
            "wspeed",
            "bhuddle",
            "whuddle",
            "bfollowing",
            "wfollowing",
        ]
    }

    # Keep track of the frame number, to align with the tracking data
    fnum = 0
    if save:
        writer = None

    # Loop over the first frames in the video to get resolution and center of the arena
    while cap.isOpened() and fnum < recoglimit:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Detect arena and extract positions
        arena = circular_arena_recognition(frame)[0]
        if h == None and w == None:
            h, w = frame.shape[0], frame.shape[1]

        fnum += 1

    # Define behaviours that can be computed on the fly from the distance matrix
    tagdict["nose2nose"] = smooth_boolean_array(
        Distance_dict[vid_name][("B_Nose", "W_Nose")] < 15
    )
    tagdict["bnose2tail"] = smooth_boolean_array(
        Distance_dict[vid_name][("B_Nose", "W_Tail_base")] < 15
    )
    tagdict["wnose2tail"] = smooth_boolean_array(
        Distance_dict[vid_name][("B_Tail_base", "W_Nose")] < 15
    )
    tagdict["sidebyside"] = smooth_boolean_array(
        (Distance_dict[vid_name][("B_Nose", "W_Nose")] < 40)
        & (Distance_dict[vid_name][("B_Tail_base", "W_Tail_base")] < 40)
    )
    tagdict["sidereside"] = smooth_boolean_array(
        (Distance_dict[vid_name][("B_Nose", "W_Tail_base")] < 40)
        & (Distance_dict[vid_name][("B_Tail_base", "W_Nose")] < 40)
    )

    B_mouse_X = np.array(
        Distance_dict[vid_name][
            [j for j in Distance_dict[vid_name].keys() if "B_" in j[0] and "B_" in j[1]]
        ]
    )
    W_mouse_X = np.array(
        Distance_dict[vid_name][
            [j for j in Distance_dict[vid_name].keys() if "W_" in j[0] and "W_" in j[1]]
        ]
    )

    tagdict["bhuddle"] = smooth_boolean_array(classifiers["huddle"].predict(B_mouse_X))
    tagdict["whuddle"] = smooth_boolean_array(classifiers["huddle"].predict(W_mouse_X))

    tagdict["bclimbwall"] = smooth_boolean_array(
        pd.Series(
            (
                spatial.distance.cdist(
                    np.array(dframe["B_Nose"]), np.array([arena[:2]])
                )
                > (w / 200 + arena[2])
            ).reshape(dframe.shape[0]),
            index=dframe.index,
        )
    )
    tagdict["wclimbwall"] = smooth_boolean_array(
        pd.Series(
            (
                spatial.distance.cdist(
                    np.array(dframe["W_Nose"]), np.array([arena[:2]])
                )
                > (w / 200 + arena[2])
            ).reshape(dframe.shape[0]),
            index=dframe.index,
        )
    )
    tagdict["bfollowing"] = smooth_boolean_array(
        following_path(
            Distance_dict[vid_name],
            dframe,
            follower="B",
            followed="W",
            frames=20,
            tol=20,
        )
    )
    tagdict["wfollowing"] = smooth_boolean_array(
        following_path(
            Distance_dict[vid_name],
            dframe,
            follower="W",
            followed="B",
            frames=20,
            tol=20,
        )
    )

    # Compute speed on a rolling window
    tagdict["bspeed"] = rolling_speed(dframe["B_Center"], pause=speedpause)
    tagdict["wspeed"] = rolling_speed(dframe["W_Center"], pause=speedpause)

    if any([show, save]):
        # Loop over the frames in the video
        pbar = tqdm(total=min(dframe.shape[0] - recoglimit, framelimit))
        while cap.isOpened() and fnum < framelimit:

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            if Like_QC_dict[vid_name][fnum]:

                # Extract positions
                pos_dict = {
                    i: np.array([dframe[i]["x"][fnum], dframe[i]["y"][fnum]])
                    for i in dframe.columns.levels[0]
                    if i != "Like_QC"
                }

                if h == None and w == None:
                    h, w = frame.shape[0], frame.shape[1]

                # Label positions
                downleft = (int(w * 0.3 / 10), int(h / 1.05))
                downright = (int(w * 6.5 / 10), int(h / 1.05))
                upleft = (int(w * 0.3 / 10), int(h / 20))
                upright = (int(w * 6.3 / 10), int(h / 20))

                # Display all annotations in the output video
                if tagdict["nose2nose"][fnum] and not tagdict["sidebyside"][fnum]:
                    cv2.putText(
                        frame,
                        "Nose-Nose",
                        (downleft if bspeed > wspeed else downright),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                if tagdict["bnose2tail"][fnum] and not tagdict["sidereside"][fnum]:
                    cv2.putText(
                        frame, "Nose-Tail", downleft, font, 1, (255, 255, 255), 2
                    )
                if tagdict["wnose2tail"][fnum] and not tagdict["sidereside"][fnum]:
                    cv2.putText(
                        frame, "Nose-Tail", downright, font, 1, (255, 255, 255), 2
                    )
                if tagdict["sidebyside"][fnum]:
                    cv2.putText(
                        frame,
                        "Side-side",
                        (downleft if bspeed > wspeed else downright),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                if tagdict["sidereside"][fnum]:
                    cv2.putText(
                        frame,
                        "Side-Rside",
                        (downleft if bspeed > wspeed else downright),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                if tagdict["bclimbwall"][fnum]:
                    cv2.putText(
                        frame, "Climbing", downleft, font, 1, (255, 255, 255), 2
                    )
                if tagdict["wclimbwall"][fnum]:
                    cv2.putText(
                        frame, "Climbing", downright, font, 1, (255, 255, 255), 2
                    )
                if tagdict["bhuddle"][fnum] and not tagdict["bclimbwall"][fnum]:
                    cv2.putText(frame, "huddle", downleft, font, 1, (255, 255, 255), 2)
                if tagdict["whuddle"][fnum] and not tagdict["wclimbwall"][fnum]:
                    cv2.putText(frame, "huddle", downright, font, 1, (255, 255, 255), 2)
                if tagdict["bfollowing"][fnum] and not tagdict["bclimbwall"][fnum]:
                    cv2.putText(
                        frame,
                        "*f",
                        (int(w * 0.3 / 10), int(h / 10)),
                        font,
                        1,
                        ((150, 150, 255) if wspeed > bspeed else (150, 255, 150)),
                        2,
                    )
                if tagdict["wfollowing"][fnum] and not tagdict["wclimbwall"][fnum]:
                    cv2.putText(
                        frame,
                        "*f",
                        (int(w * 6.3 / 10), int(h / 10)),
                        font,
                        1,
                        ((150, 150, 255) if wspeed < bspeed else (150, 255, 150)),
                        2,
                    )

                if (bspeed == None and wspeed == None) or fnum % speedpause == 0:
                    bspeed = tagdict["bspeed"][fnum]
                    wspeed = tagdict["wspeed"][fnum]

                cv2.putText(
                    frame,
                    "W: " + str(np.round(wspeed, 2)) + " mmpf",
                    (upright[0] - 20, upright[1]),
                    font,
                    1,
                    ((150, 150, 255) if wspeed < bspeed else (150, 255, 150)),
                    2,
                )
                cv2.putText(
                    frame,
                    "B: " + str(np.round(bspeed, 2)) + " mmpf",
                    upleft,
                    font,
                    1,
                    ((150, 150, 255) if bspeed < wspeed else (150, 255, 150)),
                    2,
                )

                if show:
                    cv2.imshow("frame", frame)

                if save:

                    if writer is None:
                        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                        # Define the FPS. Also frame size is passed.
                        writer = cv2.VideoWriter()
                        writer.open(
                            re.findall("(.*?)_", Tracks[vid_index])[0] + "_tagged.avi",
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            fps,
                            (frame.shape[1], frame.shape[0]),
                            True,
                        )
                    writer.write(frame)

            if cv2.waitKey(1) == ord("q"):
                break

            pbar.update(1)
            fnum += 1

    cap.release()
    cv2.destroyAllWindows()

    tagdf = pd.DataFrame(tagdict)

    return tagdf, arena


def max_behaviour(array, window_size=50):
    """Returns the most frequent behaviour in a window of window_size frames"""
    array = array.drop(["bspeed", "wspeed"], axis=1).astype("float")
    win_array = array.rolling(window_size, center=True).sum()[::50]
    max_array = win_array[1:].idxmax(axis=1)
    return list(max_array)

    ##### MACHINE LEARNING FUNCTIONS #####


def gmm_compute(x, n_components, cv_type):
    gmm = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type=cv_type,
        max_iter=100000,
        init_params="kmeans",
    )
    gmm.fit(x)
    return [gmm, gmm.bic(x)]


def GMM_Model_Selection(
    X,
    n_components_range,
    n_runs=100,
    part_size=10000,
    n_cores=False,
    cv_types=["spherical", "tied", "diag", "full"],
):
    """Runs GMM clustering model selection on the specified X dataframe, outputs the bic distribution per model,
       a vector with the median BICs and an object with the overall best model"""

    # Set the default of n_cores to the most efficient value
    if not n_cores:
        n_cores = min(multiprocessing.cpu_count(), n_runs)

    bic = []
    m_bic = []
    lowest_bic = np.inf

    pbar = tqdm(total=len(cv_types) * len(n_components_range))

    for cv_type in cv_types:

        for n_components in n_components_range:

            res = Parallel(n_jobs=n_cores, prefer="threads")(
                delayed(gmm_compute)(X.sample(part_size), n_components, cv_type)
                for i in range(n_runs)
            )
            bic.append([i[1] for i in res])

            pbar.update(1)
            m_bic.append(np.median([i[1] for i in res]))
            if m_bic[-1] < lowest_bic:
                lowest_bic = m_bic[-1]
                best_bic_gmm = res[0][0]

    return bic, m_bic, best_bic_gmm

    ##### PLOTTING FUNCTIONS #####


def plot_speed(Behaviour_dict, Treatments):
    """Plots a histogram with the speed of the specified mouse.
       Treatments is expected to be a list of lists with mice keys per treatment"""

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))

    for Treatment, Mice_list in Treatments.items():
        hist = pd.concat([Behaviour_dict[mouse] for mouse in Mice_list])
        sns.kdeplot(hist["bspeed"], shade=True, label=Treatment, ax=ax1)
        sns.kdeplot(hist["wspeed"], shade=True, label=Treatment, ax=ax2)

    ax1.set_xlim(0, 7)
    ax2.set_xlim(0, 7)
    ax1.set_title("Average speed density for black mouse")
    ax2.set_title("Average speed density for white mouse")
    plt.xlabel("Average speed")
    plt.ylabel("Density")
    plt.show()


def plot_heatmap(dframe, bodyparts, xlim, ylim, save=False):
    """Returns a heatmap of the movement of a specific bodypart in the arena.
       If more than one bodypart is passed, it returns one subplot for each"""

    fig, ax = plt.subplots(1, len(bodyparts), sharex=True, sharey=True)

    for i, bpart in enumerate(bodyparts):
        heatmap = dframe[bpart]
        if len(bodyparts) > 1:
            sns.kdeplot(heatmap.x, heatmap.y, cmap="jet", shade=True, alpha=1, ax=ax[i])
        else:
            sns.kdeplot(heatmap.x, heatmap.y, cmap="jet", shade=True, alpha=1, ax=ax)
            ax = np.array([ax])

    [x.set_xlim(xlim) for x in ax]
    [x.set_ylim(ylim) for x in ax]
    [x.set_title(bp) for x, bp in zip(ax, bodyparts)]

    if save != False:
        plt.savefig(save)

    plt.show()


def model_comparison_plot(
    bic,
    m_bic,
    best_bic_gmm,
    n_components_range,
    cov_plot,
    save,
    cv_types=["spherical", "tied", "diag", "full"],
):
    """Plots model comparison statistics over all tests"""

    m_bic = np.array(m_bic)
    color_iter = cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_bic_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(12, 8))
    spl = plt.subplot(2, 1, 1)
    covplot = np.repeat(cv_types, len(m_bic) / 4)

    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            spl.bar(
                xpos,
                m_bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                color=color,
                width=0.2,
            )
        )

    spl.set_xticks(n_components_range)
    plt.title("BIC score per model")
    xpos = (
        np.mod(m_bic.argmin(), len(n_components_range))
        + 0.5
        + 0.2 * np.floor(m_bic.argmin() / len(n_components_range))
    )
    spl.text(xpos, m_bic.min() * 0.97 + 0.1 * m_bic.max(), "*", fontsize=14)
    spl.legend([b[0] for b in bars], cv_types)
    spl.set_ylabel("BIC value")

    spl2 = plt.subplot(2, 1, 2, sharex=spl)
    spl2.boxplot(list(np.array(bic)[covplot == cov_plot]), positions=n_components_range)
    spl2.set_xlabel("Number of components")
    spl2.set_ylabel("BIC value")

    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()

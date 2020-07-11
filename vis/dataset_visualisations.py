"""Scripts for producing datasets for the keypoint & segmentation datasets collected on MTurk"""

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import json
from my_work.utils import Vector
import os
from my_work.annotated_joints_plotter import colours, marker_func

joinp = os.path.join


fig_dir = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\dataset_statistics" # target directory for figures
src = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\segments\keypoints.json"
all_data = json.load(open(src, "r"))
all_data_by_img = {i["img_path"]:i for i in all_data}

n_keypoints = 18
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"]

ordered_colours = [colours[i] for i in keypoint_list]

def r2d(rad):
  return rad * 180 / np.pi

skeleton = {
#ANKLE
8: (7, 9),
2: (1, 3),
11: (10, 12),
5: (4,6),
# #Paws
# 1: (0, 2),
# 4: (0, 5),
# 7: (0, 8),
# 10: (0, 11),
# #Knee
# 3: (2, 0),
# 6: (5, 0),
# 9: (8, 0),
# 12: (11, 0),
# #Tail
# 13: (0, 14),
# 14: (13, 0),
}

def plot_skeleton(ax, data, plot_zeros = False):
    for i, (c1, c2) in skeleton.items():
        for c in [c1, c2]:
            x1, y1 = data[i-1]
            x2, y2 = data[c-1]
            if plot_zeros or all(i != 0 for i in [x1, x2, y1, y2]): # Require no non-zero points if plot_zeros flag is False
                ax.plot([x1, x2], [y1, y2], c = ordered_colours[i-1])

def produce_normalised_skeleton(origin_joint = "Nose", target_joint = ("Chin")):
    """Produce a skeleton of average keypoints.
    Works by transforming all points such that
    origin_joint -> (0, 0), and target_joint -> (0, -1)."""

    data = [] # (n_imgs, n_joints, 2) array of normalised keypoint positions for each frame

    origin_idx, target_idx = keypoint_list.index(origin_joint), keypoint_list.index(target_joint) # get indices of nose, chin

    for i in range(8000):
        joints = np.array(all_data[i]["joints"])[:, :2]
        target, origin = joints[target_idx], joints[origin_idx] # extract join data

        # Center all data to nose.
        centered_data = joints - origin

        # Calculate angle between chin and nose
        angle = - np.arctan2(origin[0] - target[0], - (origin[1] - target[1])) # - delta y as y coordinates of image are inverted
        # Rotate all data by this angle
        R = np.array([ [np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_data = np.matmul(R, centered_data.T) # Rotated points

        # Calculate length between chin and nose
        length = abs(rotated_data[1, target_idx] - rotated_data[1, origin_idx])
        if length == 0: continue # ignore any entries with improperly entered nose, chin
        # Shrink all data by this value
        T = np.array([[1/length, 0], [0, 1/length]])
        transformed_data = np.matmul(T, rotated_data)  # Fully transformed (centered, rotated, enlarged) data

        # SET ANY NULL POINTS TO ZERO FOR TRANSFORMED POINTS
        for n in range(n_keypoints):
            if (joints[n] == 0.0).all():
                transformed_data[:, n] = 0.0

        data.append(transformed_data.T)

        # # PLOT ALL
        # if i == 55:
        # fig, (axes) = plt.subplots(ncols = 2)
        # # plot regular
        # x, y = zip(*joints)
        # axes[0].scatter(x, y, c = ordered_colours)
        # axes[0].imshow(plt.imread(joinp(r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset", all_data[i]["img_path"])))
        #
        # # plot normalised
        # plot_skeleton(axes[1], p.T)
        # axes[1].scatter(*p, c = ordered_colours)
        #
        # plt.gca().invert_yaxis()
        # plt.show()

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12,6))

    handles = []

    for n in range(n_keypoints):
        all_locs = data[:, n]
        if n != origin_idx:
            non_zero_locs = all_locs[np.all(all_locs, axis=1)] # only non zero points
        else: non_zero_locs = all_locs # chin is centered on (0,0

        mean_x, mean_y = np.mean(non_zero_locs, axis=0) # Calculate mean position
        std_x, std_y = np.std(non_zero_locs, axis = 0) # calculate standard deviations

        # ax.scatter([mean_x], [mean_y], c = [ordered_colours[n]], marker = marker_func(keypoint_list[n]))

        s = ax.errorbar([mean_x], [mean_y], yerr=std_y/3, xerr=std_x/3,
                    fmt = marker_func(keypoint_list[n]), c = ordered_colours[n], alpha = 0.5)

        handles.append(s)

    ax.invert_yaxis()
    ax.legend(handles, keypoint_list)

    plt.title(f"All keypoints, normalised to {origin_joint} and {target_joint}\nError bars depict (std dev)/3")
    plt.xlabel("Normalised x")
    plt.ylabel("Normalised y")
    plt.tight_layout()
    plt.savefig(joinp(fig_dir, f"Normalised skeleton {origin_joint.replace(':', '')} - {target_joint.replace(':', '')}"))

def calculate_all_rots():
    """For all joints in skeleton, work out the angles for each image, and return dict of joints: [angles]"""
    out = {}
    for i in all_data:
        kp_data = i["joints"]
        for joint, (jc1, jc2) in skeleton.items():
            p_target, p1, p2 = [Vector(*kp_data[i-1][:2]) for i in [joint, jc1, jc2]]

            if all(p.length() != 0 for p in [p_target, p1, p2]): # Only calculate for all 3 joints are non zero
                a = (p_target - p1).angle_between(p_target - p2)
                out[joint] = out.get(joint, []) + [a]

    return out

def plot_polar_hist(ax, data, n_bins=30,
    c = (0,0), r=1.0, colour="blue"):
    """Given an axis, plots the polar histogram of the density of the angles"""

    hist, bins = np.histogram(data, bins=n_bins, density = True)

    for n in range(n_bins):
        d = hist[n]
        t1 = r2d(bins[n])
        t2 = r2d(bins[n+1])

        w = Wedge(c, r, t1, t2, alpha=d, color=colour)
        ax.add_patch(w)

def plot_all_joint_rom():
    """Calculate the mean skeleton as a (Nx2) array,
    plot on it the joint rotations for each joint"""

    fig, ax = plt.subplots()
    rot_data = calculate_all_rots()

    lf = np.array([[0,0], [0.1,0.2], [0.15, 0.4]])
    dx = np.array([0.6, 0])
    skeleton_data = np.array([*lf,
                     *(lf - dx),
                     *(lf - 0.1 * dx),
                     *(lf - 1.1 * dx),
                     [-0.6, 0.5], [-0.7, 0.5],
                     [0.2, 0.6], [0.24, 0.6],
                     [0.25, 0.55], [0.24, 0.5]]) # example skeleton

    # Plot skeleton
    for n in range(n_keypoints):
        plt.scatter(*skeleton_data[n], c = [ordered_colours[n]], marker=marker_func(keypoint_list[n]))
    plot_skeleton(ax, skeleton_data, plot_zeros=True)

    r = (np.max(skeleton_data[:, 0]) - np.min(skeleton_data[:, 0])) / 30 # Maximum x length defines radius
    for n_joint, angles in rot_data.items():
        x, y = skeleton_data[n_joint - 1]
        plot_polar_hist(ax, angles, c = (x,y), r = r, colour = ordered_colours[n_joint - 1])

    ax.axis("equal")
    plt.title("Joint Rotations")
    plt.savefig(joinp(fig_dir, "joint_rotations"))

def plot_keypoint_distributions():
    """Plot distribution of the keypoint dataset:
        - with number of entries per n keypoints
        - with number of entries per keypoint type (eg front right paw)"""
    X = []
    Y = {} #
    for i in all_data:
        j = i["joints"]
        X.append(len([x for x in j if x != [0.0, 0.0, 0.0]]))
        for n in range(18):
            if j[n] != [0.0, 0.0, 0.0]:
                Y[keypoint_list[n]] = Y.get(keypoint_list[n], 0) + 1 # increment count of keypoint by type

    plt.hist(X, bins = [i-.5 for i in range(8, 20)], rwidth=0.8)
    plt.ylabel("Number of entries")
    plt.xlabel("Number of identified keypoints")
    plt.savefig(joinp(fig_dir, "n_keypoint_hist"))
    plt.close("all")

    plt.figure(figsize=(12,6))
    sorted_kp = sorted(Y.keys(), key = lambda kp: Y[kp], reverse=True) # keypoints in descending order of number identified
    plt.bar([i*3 for i in range(18)], [Y[i] for i in sorted_kp],
            tick_label=["\n".join(i.split(" ")) for i in sorted_kp], width = 0.8)
    plt.ylabel("Number of entries")
    plt.xlabel("Keypoint")
    plt.subplots_adjust(bottom=0.19, top = 0.98)
    plt.savefig(joinp(fig_dir, "type_keypoint_hist"))

# produce_normalised_skeleton(origin_joint="Left front leg: top", target_joint="Left front leg: middle joint")
# plot_keypoint_distributions()
# plot_all_joint_rom()
# plt.show()

### TO DO. FIND WAY TO NORMALISE KEYPOINTS ACROSS ENTRIES, TO SOME FORM OF NORMALISED SKELETON
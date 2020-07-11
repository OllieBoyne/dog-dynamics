"""Various scripts for the handling of MTurk outputs (namely joint annotations and segmentations)."""

import csv, os, sys
sys.path.append(os.path.dirname(sys.path[0]))
osp = os.path
joinp = osp.join

from my_work.data_loader import DataSources
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
import json
import tqdm as tqdm
import io, base64
import sys
from scipy import ndimage
from scipy import signal
from functools import reduce
from my_work.utils import image_to_polygon, PolyArea, fill_segment, try_mkdir
from time import perf_counter as pf
import cv2

# dir = os.path.join(DataSources.datasets, "Dogs", "Test dataset")
dog_test_dataset_1 = dict(
keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "Test dataset", "results.csv"),
segment_loc = os.path.join(DataSources.datasets, "Dogs", "Test dataset", "results_segmentation.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "Test dataset"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "Annotation"),
bbox_file_src = os.path.join(DataSources.datasets, "Dogs", "Annotation", "bboxs.csv"),
keypoint_list = ["Left front paw", "Left front ankle", "Left front knee", "Left rear paw", "Left rear ankle", "Left rear knee", "Right front paw", "Right front ankle", "Right front knee", "Right rear paw", "Right rear ankle", "Right rear knee", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin",
]
)

dog_test_dataset_2 = dict(
keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "Test dataset", "results.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "Test dataset"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "Annotation"),
bbox_file_src = os.path.join(DataSources.datasets, "Dogs", "Annotation", "bboxs.csv"),
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"]
)

dog_dataset = dict(
keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "Full dog dataset", "results.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "Full dog dataset"),
segment_loc = os.path.join(DataSources.datasets, "Dogs", "Full dog dataset", "results_segmentations.csv"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "Annotation"),
bbox_file_src = os.path.join(DataSources.datasets, "Dogs", "Annotation", "bboxs.csv"),
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin", "Right ear", "Left ear"]
)

# Horse data set
horse_data_set = dict(
keypoints_loc = os.path.join(DataSources.datasets, "Horses", "weizmann_horse_db", "horse-images", "results.csv"),
images_dir = os.path.join(DataSources.datasets, "Horses", "weizmann_horse_db", "horse-images"),
silhouette_dir = os.path.join(DataSources.datasets, "Horses", "weizmann_horse_db", "silhouettes"),
bbox_file_src = os.path.join(DataSources.datasets, "Horses", "weizmann_horse_db", "silhouettes", "bboxs.csv"),
keypoint_list = ["Left front leg: hoof", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: hoof", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: hoof",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: hoof", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"]
)

# Set collected at Vet Department of dogs in horse arena
arena_dog_dataset = dict(
keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "Arena dataset", "results.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "Arena dataset"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "Arena dataset"),
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"]
)

# Set collected from YouTube
youtube_dog_dataset = dict(

keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "youtube_clips_1", "results - v2.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "youtube_clips_1", "output_frames"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "youtube_clips_1"),
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"],
data_in_subfolders = False,
file_mapping = os.path.join(DataSources.datasets, "Dogs", "youtube_clips_1", "set_2", "mapping.json")
)

zebris_dataset = dict(

keypoints_loc = os.path.join(DataSources.datasets, "Dogs", "Zebris dataset", "results.csv"),
images_dir = os.path.join(DataSources.datasets, "Dogs", "Zebris dataset", "images"),
annotation_dir = os.path.join(DataSources.datasets, "Dogs", "Zebris dataset"),
keypoint_list = ["Left front leg: paw", "Left front leg: middle joint", "Left front leg: top",
"Left rear leg: paw", "Left rear leg: middle joint", "Left rear leg: top", "Right front leg: paw",
"Right front leg: middle joint", "Right front leg: top", "Right rear leg: paw", "Right rear leg: middle joint",
"Right rear leg: top", "Tail start", "Tail end", "Base of left ear", "Base of right ear", "Nose", "Chin"],
data_in_subfolders = False
)

colours = {
"Left front leg: hoof": "#d82400",
"Left front leg: middle joint": "#d82400",
"Left front leg: top": "#d82400",
"Left rear leg: hoof": "#fcfc00",
"Left rear leg: middle joint": "#fcfc00",
"Left rear leg: top": "#fcfc00",
"Right front leg: hoof": "#48b455",
"Right front leg: middle joint": "#48b455",
"Right front leg: top": "#48b455",
"Right rear leg: hoof": "#0090aa",
"Right rear leg: middle joint": "#0090aa",
"Right rear leg: top": "#0090aa",
"Tail start": "#d848ff",
"Tail end": "#d848ff",
"Base of left ear": "#fc90aa",
"Base of right ear": "#006caa",
"Nose": "#d89000",
"Chin": "#d89000",
"Right ear": "#006caa",
"Left ear" : "#fc90aa",

"Left front leg: paw": "#d82400",
"Left rear leg: paw": "#fcfc00",
"Right front leg: paw": "#48b455",
"Right rear leg: paw": "#0090aa",

"Left front paw": "#d82400",
"Left front ankle": "#d82400",
"Left front knee": "#d82400",
"Left rear paw": "#fcfc00",
"Left rear ankle": "#fcfc00",
"Left rear knee": "#fcfc00",
"Right front paw": "#48b455",
"Right front ankle": "#48b455",
"Right front knee": "#48b455",
"Right rear paw": "#0090aa",
"Right rear ankle": "#0090aa",
"Right rear knee": "#0090aa",

}

def marker_func(label):
    """Given a label, returns the decided marker"""
    if "paw" in label or "hoof" in label: return "^"
    elif "ankle" in label or "middle" in label: return "D"
    elif "knee" in label or "top" in label: return "s"

    return "o"

mean = lambda arr: sum(arr)/len(arr)

def rms(xs, ys):
    """For a series of xs and ys, calculate the mean point, and return the rms from all the points to the mean point"""
    assert len(xs) == len(ys), "xs and ys must be same length for rms"
    mean_x, mean_y = mean(xs), mean(ys)

    tot_error = 0
    for x, y in zip(xs, ys):
        tot_error += (x-mean_x)**2 + (y-mean_y)**2

    return (tot_error / len(xs))**(0.5)

def most_anomalous(xs, ys):
    """For a series of xs and ys, identify the index of the point that is furthest from the mean point"""
    assert len(xs) == len(ys), "xs and ys must be same length for rms"
    mean_x, mean_y = mean(xs), mean(ys)

    return np.argmax([(x - mean_x)**2 + (y - mean_y) ** 2 for x,y in zip(xs, ys)])

segment_colour_column_names ={
    i:f"Answer.annotatedResult.labelMappings.{i}.color" for i in ["Left ear tip", "Right ear tip", "Left ear", "Right ear", "Tail"]
}

column_names = {
    "worker":"WorkerId",
    "multiple_dogs": "Answer.AreMultipleDogs.MultipleDogs",
    "img_height": "Answer.annotatedResult.inputImageProperties.height",
    "img_width": "Answer.annotatedResult.inputImageProperties.width",
    "answer": "Answer.annotatedResult.keypoints",
    "approve": "Approve",
    "reject": "Reject",
    "img": "Input.image_url",
    "status": "AssignmentStatus",
    "segment": "Answer.annotatedResult.labeledImage.pngImageData"

} # Dict of shorthand:full column name

for k, v in segment_colour_column_names.items(): column_names[k] = v

def try_index(arr, index):
    """Tries array.index method, returns -1 if not found"""
    try:
        return arr.index(index)
    except ValueError:
        return -1

def get_columns(headers, data, *entries):
    """Given a header column, scans each shorthand entry in entries, finds the relevant column, and returns that index in data"""
    data += [""] * (len(headers) - len(data)) # Pad data with blank entries
    return [data[try_index(headers, column_names[entry])] for entry in entries]


def process_keypoint_data(keypoints_loc, images_dir, keypoint_list, bbox_file_src ="", plot_simple = True, plot_detailed=False, minimum_valid_keypoints=8,
                          data_in_subfolders = True, file_mapping = None, **kwargs):
    """Load labelled keypoint and bbox data, and output the data as visual graphs, and as a json dict for NN training.

    plot_simple & plot_detailed decide whether to save plots of final keypoints (simple) or all data submitted (detailed)
    Minimum valid keypoints discards any images with fewer than that identified"""

    dpi = 200

    data = {}  # data to be dict of img_src: [labelled_keypoints]
    img_sizes = {}  # data to be dict of img_src: (width, height)
    img_bounds = {} # dict of img_src: [top_left_x, top_left_y, width, height]
    is_multiple_dogs_dict = {}

    # Create sub directories if necessary
    try_mkdir(os.path.join(images_dir, "labelled_keypoints"))
    if data_in_subfolders:
        if len(os.listdir(os.path.join(images_dir, "labelled_keypoints"))) == 0:
            for folder in [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]:
                os.mkdir(os.path.join(images_dir, "labelled_keypoints", folder))

    with open(keypoints_loc, "r") as infile:
        reader = csv.reader(infile)

        headers = next(reader)

        if file_mapping is not None:  # Fix for youtube set, maps certain image names (00001) to other image names using .json
            mapping = json.load(open(file_mapping))
        else:
            mapping = {}

        for line in reader:
            *_, img, is_multiple_dogs, height, width, answers, status = get_columns(headers, line, "img",
                                                                                    "multiple_dogs", "img_height",
                                                                                    "img_width", "answer", "status")
            if img in mapping:
                img = mapping[img] # map image where required

            # answers originally an array of dicts, where each dict has keys label, x and y
            # convert to a single dict, with format label: (x,y)

            answers = eval(answers)

            if status != "Rejected":
                answer_dict = {i["label"]: (i["x"], i["y"]) for i in answers}
                is_multiple_dogs_dict[img] = is_multiple_dogs_dict.get(img, []) + [
                    True if is_multiple_dogs == "true" else False]

                data[img] = data.get(img, []) + [answer_dict]  # add to dictionary
                img_sizes[img] = (int(width), int(height))

    if bbox_file_src != "":
        with open(bbox_file_src, "r") as infile:
            reader = csv.reader(infile)
            next(reader)
            # TODO For data not within subfolders, need to edit this so other_data[0] isn't in full_img_src. Fine for now for proper dataset.
            for line in reader:
                img_src, x0, y0, width, height, *other_data  = line
                if other_data != []: full_img_src = f"{other_data[0]}/{img_src}"
                else: full_img_src = img_src
                img_bounds[full_img_src] = list(map(int, [x0, y0, width, height]))

    # Create relevant directories if they don't exist
    for desired_dir in ["detailed_keypoints", "labelled_keypoints"]:
        if desired_dir not in os.listdir(images_dir): os.mkdir(os.path.join(images_dir, desired_dir))

    # For now, load all data from any assignments, and just average any data to form the basis for the actual plot.
    # Scheme to dismiss points:
    # If labelled in only 1 dataset, dismiss
    # If labelled in only 2 datasets, if rms error > threshold, dismiss
    # If labelled in 3 datasets, if rms error > threshold, dismiss
    # If in any case, not dismissed, final position is the average of any datasets

    processed_labelled_keypoints = {}
    # Dictionary to fill with values { 'horse_001.jpg', [[x_0, y_0, vis_0], [x_1, y_1, vis_1], ...., [x_n. y_n, vis_n]], [top_left_x, top_left_y, width, height], ...}

    output_data = [] # Data from each image, in format {img_path: "", img_width: "", img_height: "", img_bbox: [], joints: []}

    progress_bar = tqdm.tqdm(total=len(data))

    # Declare figures outside of loop, to minimise memory wastage of creating thousands of figures

    for img, answers in list(data.items()):
        # plot_simple = img == "Gracie run for ball\\3_00009.png" # to plot a specific item

        n_valid_keypoints = 0 # count valid keypoints to validate sets

        img_data = plt.imread(os.path.join(images_dir, img))
        width, height = img_data.shape[0:2]

        pad = 1.1 # pad for bbox

        if img in img_bounds: x0, y0, bbox_width, bbox_height = img_bounds[img]
        else: x0, y0, bbox_height, bbox_width = 0, 0, *img_data.shape[:2] # Bbox is whole image if not provided

        xm, ym = (x0+width)/2, (y0+height)/2
        in_bbox = lambda x, y: (xm - (width*pad/2)) < x < (xm + (width*pad/2)) and \
                        (ym - (height*pad/2)) < y < (ym + (height*pad/2))
        not_in_bbox_count = 0 # count of those keypoints not in bbox

        if plot_simple:
        # Start 2 new figs, one for the detailed result of the output, one for the simple plot of labelled keypoints
            fig_simple = plt.figure()
            ax_simple = fig_simple.add_axes([0, 0, 1, 1], label=img)
            ax_simple.imshow(img_data)
            ax_simple.axis("off")
        if plot_detailed:
            fig_detailed = plt.figure()
            ax_detailed = fig_detailed.add_axes([0, 0, 1, 1], label=img)
            ax_detailed.imshow(img_data)

        # Set thresholds for deciding if data set is valid
        width, height = img_sizes[img]
        threshold = (width + height) / 80

        image_keypoint_data = dict(img_path = img, img_width = width, img_height = height, joints = []) # place data here for .json output
        joint_data = {} # dict of joint : (mean_x, mean_y, visibility) for all keypoints

        for keypoint_label in keypoint_list:
            colour = colours[keypoint_label] # get colour (ignoring hidden command)

            if any(keypoint_label in answer for answer in answers):
                visibility = 1 # not a hidden marker
            elif any(keypoint_label + " (hidden)" in answer for answer in answers):
                visibility = 0 # hidden marker
                keypoint_label = keypoint_label + " (hidden)" # change keypoint name to reflect this
            else:
                image_keypoint_data["joints"].append([0, 0, 0])
                continue # if not found, ignore this keypoint label. Add as [0,0,0]

            all_xs, all_ys = zip(*[answer.get(keypoint_label) for answer in answers if keypoint_label in answer]) # Get all non blank xs
            mean_x, mean_y = mean(all_xs), mean(all_ys)

            n_valid = len(all_xs)

            discard = False

            # See which points aren't in bbox
            for x, y in zip(all_xs, all_ys):
                not_in_bbox_count += not in_bbox(x, y)

            if n_valid == 1:
                # For now, if fewer than 3 submitted, accept 1x n_valid. Change this later when dataset is finished
                discard = len(answers) >= 3

            while n_valid > 2:
                if rms(all_xs, all_ys) > threshold:
                    # Discard worst point and try as 2 data points
                    n_dis = most_anomalous(all_xs, all_ys)
                    all_xs, all_ys = all_xs[:n_dis] + all_xs[n_dis+1:], all_ys[:n_dis] + all_ys[n_dis+1:]
                    mean_x, mean_y = mean(all_xs), mean(all_ys)
                    n_valid -= 1
                else: break

            if n_valid == 2 and rms(all_xs, all_ys) > threshold: discard = True

            if not discard:

                image_keypoint_data["joints"].append([mean_x, mean_y, visibility])  # for non entered keypoints, enter simply as 0,0,0
                n_valid_keypoints += 1

                if plot_detailed:
                    for x, y in zip(all_xs, all_ys):
                        ax_detailed.plot(x, y, "x", color=colour, alpha = 0.5, ms = 2)

                    ax_detailed.plot(mean_x, mean_y, "o", marker = marker_func(keypoint_label), color=colour, label=f"{keypoint_label} [{n_valid}]", ms = 2)

                if plot_simple:
                    joint_data[keypoint_label.replace(" (hidden)", "")] = (mean_x, mean_y, visibility)
                    # ax_simple.plot(mean_x, mean_y, "Xo"[visibility], color=colour, ms = 15)

            else:
                image_keypoint_data["joints"].append([0, 0, 0]) # for non entered keypoints, enter simply as 0,0,0
                if plot_detailed: ax_detailed.plot([], [], "o-", color="grey", label=f"X {keypoint_label} [{n_valid}]")

        ### Modifications to work with newest versions of deep network
        # Pad joints so there are 24 - even if not labelled
        while len(image_keypoint_data["joints"]) < 24:
            image_keypoint_data["joints"].append([0, 0, 0])

        image_keypoint_data["has_segmentation"] = False # Assume no seg
        image_keypoint_data["ear_tips_labelled"] = False  # Assume no seg
        image_keypoint_data["img_bbox"] = [x0, y0, bbox_width, bbox_height]
        ###

        # Criteria for if there are multiple dogs:
        # If more than 50 % of workers marked it as multiple dogs
        # If >10 of all labelled keypoints are outside of the bbox, with a padding

        if len(joint_data) == 0 : is_multiple_dogs = False
        else:
            is_multiple_dogs = sum(is_multiple_dogs_dict[img]) / len(
                is_multiple_dogs_dict[img]) > 0.5 or not_in_bbox_count > 10

        image_keypoint_data["is_multiple_dogs"] = is_multiple_dogs

        if plot_simple:
            for label, (x, y, vis) in joint_data.items():
                marker = "X" if vis == 0 else "^" if "paw" in label else "s" if "top" in label else "o"
                ax_simple.plot(x, y, "o", marker=marker, color = colours[label], ms = 2)

            # Join up legs
            for side in "Right", "Left":
                for end in "rear", "front":
                    leg_joints_names = [j for j in joint_data.keys() if f"{side} {end}" in j]
                    leg_joints_data = [joint_data[j] for j in leg_joints_names]
                    for (start, finish) in [["paw", "middle"], ["middle", "top"]]:
                        start_idx = [n for n, i in enumerate(leg_joints_names) if start in i]
                        finish_idx = [n for n, i in enumerate(leg_joints_names) if finish in i]
                        if len(start_idx) > 0 and len(finish_idx) > 0: # start and end data, draw connection
                            x1, y1, v1 = leg_joints_data[start_idx[0]]
                            x2, y2, v2 = leg_joints_data[finish_idx[0]]
                            ax_simple.plot([x1, x2], [y1, y2], color = colours[f"{side} {end} paw"])

            # Add title indicating number of entries:
            ax_simple.text(10, 10, f"x{len(answers)}", bbox = dict(facecolor="white"))

            if image_keypoint_data["is_multiple_dogs"]:
                ax_simple.text(10, 10, "MD", ha = "left", va="top", fontsize=14, color="white",
                               bbox=dict(facecolor='black'))

        if n_valid_keypoints >= minimum_valid_keypoints and not is_multiple_dogs:

            output_data.append(image_keypoint_data)

            if plot_detailed:
                ax_detailed.legend(bbox_to_anchor=(0.8, 0.5), loc='center left', ncol=1, fontsize=9)
                fig_detailed.subplots_adjust(right = 0.6)

            # Plot bbox
            image_keypoint_data["img_bbox"] = [x0, y0, bbox_width, bbox_height]
            *xy, width, height = [x0, y0, bbox_width, bbox_height]

            if plot_detailed:ax_detailed.add_patch(Rectangle(xy, width, height, fill = False, ec = "red", lw=1))
            if plot_simple: ax_simple.add_patch(Rectangle(xy, width, height, fill = False, ec = "red", lw=1))

            # if "/" in img: # if img contains a folder loc
            #     for folder in ["detailed_keypoints", "labelled_keypoints"]: # make the relevant subfolders
            #         if not os.path.isdir(os.path.join(images_dir, folder, img.split("/")[0])): # check if existing dir
            #             os.mkdir(os.path.join(images_dir, folder, img.split("/")[0]))

            if plot_detailed or plot_simple:
                subfolders = ["detailed_keypoints"] * plot_detailed + ["labelled_keypoints"] * plot_simple
                figs = []
                if plot_simple: figs += [fig_simple]
                if plot_detailed: figs += [fig_detailed]


                for (sf, fig) in zip(subfolders, figs):

                    if data_in_subfolders:
                        # Make subfolder for this breed if necessary [USED FOR ORIGINAL DOG DATASET]
                        splitter = "/" if "/" in img else "\\"
                        img_subfolder = img.split(splitter)[0]
                        if img_subfolder not in os.listdir(os.path.join(images_dir, sf)):
                            os.mkdir(os.path.join(images_dir, sf, img_subfolder))

                    fig.savefig(os.path.join(images_dir, sf, img), dpi = dpi)


                plt.close("all")

        progress_bar.update(1)

    with open(os.path.join(images_dir, "labelled_keypoints", "keypoints.json"), 'w') as outfile:
        json.dump(output_data, outfile)


def get_bboxs_from_silhouettes(silhouette_dir, images_dir, **kwargs):
    """For a directory of silhouette images, extract bboxs for each image, and save as csv of:
    img_src, top_left_x, top_left_y, width, height.

    In the case that the scale of the actual images and the silhouettes don't match, use actual_images_dir to find the bboxs of the actual images"""

    bboxs = [ ["file", "top_left_x", "top_left_y", "width", "height"]] # Array of desired data to write to csv

    for file in [f for f in os.listdir(silhouette_dir) if f[-4:].lower() in [".jpg", ".png"]]:
        img = plt.imread(os.path.join(silhouette_dir, file))

        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        ymin, ymax= np.where(rows)[0][[0, -1]]
        xmin, xmax= np.where(cols)[0][[0, -1]]

        # Scale to actual images
        actual_img = plt.imread(os.path.join(images_dir, file))
        silhouette_height, silhouette_width = img.shape
        actual_height, actual_width, _ = actual_img.shape

        width_ratio = actual_width / silhouette_width
        height_ratio = actual_height / silhouette_height

        ymin, ymax = int(round(ymin * height_ratio)), int(round(ymax * height_ratio))
        xmin, xmax = int(round(xmin * width_ratio)), int(round(xmax * width_ratio))

        bboxs.append([file, xmin, ymin, (xmax - xmin), (ymax - ymin)])

    with open(os.path.join(silhouette_dir, "bboxs.csv"), "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(bboxs)

def get_bboxs_from_annotations(annotation_dir, **kwargs):
    """For a directory of annotations for a variety of, extract bboxs for each image, and save as csv of:
    img_src, top_left_x, top_left_y, width, height."""

    bboxs = [ ["file", "top_left_x", "top_left_y", "width", "height", "folder"]] # Array of desired data to write to csv

    for folder in [fo for fo in os.listdir(annotation_dir) if os.path.isdir(os.path.join(annotation_dir, fo))]:
        folder_dir = os.path.join(annotation_dir, folder)

        for file in [f for f in os.listdir(folder_dir)]:
            with open(os.path.join(folder_dir, file), "r") as infile:
                file_text = "".join([l for l in infile.readlines()])


            # print((file_text.split(f"<ymax>")[1]).split(f"</ymax>")[0])
            bbox_corners = [0, 0, 0, 0]
            for n, param in enumerate(["xmin", "xmax", "ymin", "ymax"]):
                bbox_corners[n] =  int((file_text.split(f"<{param}>")[1]) .split(f"</{param}>")[0])

            xmin, xmax, ymin, ymax = bbox_corners
            bboxs.append([file + ".jpg", xmin, ymin, (xmax - xmin), (ymax - ymin), folder])

        with open(os.path.join(annotation_dir, "bboxs.csv"), "w", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(bboxs)

def view_entries_by_worker(worker_id, img_dir, min_kp=1000):
    """Given a worker id, save figures of all entries with fewer than min_kp."""

    if not os.path.isdir(os.path.join(img_dir, "worker_evals", worker_id)):
        os.mkdir(os.path.join(img_dir, "worker_evals", worker_id))

    results = os.path.join(img_dir, "results.csv")
    with open(results, "r") as infile:
        reader = csv.reader(infile)
        headers = next(reader)

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        for n, line in enumerate(reader):

            img, is_multiple_dogs, height, width, answers, this_worker_id, status = get_columns(headers, line, "img", "multiple_dogs", "img_height", "img_width", "answer", "worker", "status")
            is_rejected = status == "Rejected"

            if this_worker_id == worker_id and not is_rejected: # If correct worker and not already rejected
                print(img, n)
                # answers originally an array of dicts, where each dict has keys label, x and y
                # convert to a single dict, with format label: (x,y)

                answers = eval(answers)
                if answers == []: answer_dict = {} # Only store non blank entries

                else:   answer_dict = {i["label"]: (i["x"], i["y"]) for i in answers}

                if len(answer_dict) < min_kp:
                    ax.imshow(plt.imread(os.path.join(img_dir, img)))

                    for l, (x, y) in answer_dict.items():
                        ax.scatter(x, y, c=colours[l.replace(" (hidden)", "")], s = 5)

                    plt.savefig(os.path.join(img_dir, "worker_evals", worker_id, f"{n}.png"))
                    plt.cla()


def reject_works(lines, file, message = "Insufficient accuracy of keypoints - not enough accurate keypoints labelled."):
    """Given a list of line numbers (corresponding to line of data i.e 0 is the first row of data),
     reject each line in results.csv with the description '...', and save as results_rejected.csv"""

    message = message
    outrows = []
    with open(file, "r") as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        outrows += [headers]

        for n, line in enumerate(reader):
            data = line + [""] * (len(headers) - len(line))
            if n in lines:
                data[len(headers)-1] = message

            outrows += [data]

    with open(file.replace("results", "results_rejected"), "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(outrows)

def get_file_list_with_bbox_violations(images_dir, bbox_file_src,**kwargs):
    """Gets the file list for a folder of folders with images in, excluding any images for which the bbox is not properly contained within the images (to exclude images of dogs being out of frame)."""
    bbox_in_frame = 0
    bbox_out_of_frame = 0

    n_limit = 1e6
    n = 0

    file_list = [["image_url"]]

    from random import randint

    with open(bbox_file_src, "r") as infile:
        reader = csv.reader(infile)
        next(reader)
        for line in reader:
            img_name, x0, y0, bb_width, bb_height, *other_data  = line

            img_loc = os.path.join(images_dir, other_data[0], img_name)

            img_data = plt.imread(img_loc)

            img_height, img_width = img_data.shape[:2]

            if int(x0) + int(bb_width) < img_width - 1 and int(y0) + int(bb_height) < img_height - 1:
                bbox_in_frame += 1
                file_list.append([f"{other_data[0]}/{img_name}"])
            else:
                bbox_out_of_frame += 1

            n+=1
            if n > n_limit: break

    tot_ims = bbox_in_frame + bbox_out_of_frame
    print(f"IN FRAME: {bbox_in_frame} ({round(100 * bbox_in_frame/tot_ims,1)})")
    print(f"OUT OF FRAME: {bbox_out_of_frame} ({round(100*bbox_out_of_frame / tot_ims, 1)})")

    with open(os.path.join(images_dir, "file_list.csv"), "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(file_list)

def read_uri(uri):
    """Returns uri as numpy img array"""
    try:
        segment_rgb = plt.imread(io.BytesIO(base64.b64decode(uri)))
        return segment_rgb
    except Exception as e:
        return None # for now, ignore read errors

from numpngw import write_png
def bitmask_to_uri(bitmask):
    """Encodes a bitmas as a uri"""
    f = io.BytesIO()
    write_png(f, bitmask)

    return base64.b64encode(f.getvalue()).decode("utf-8")


def extract_segmentations(images_dir, segment_loc, plot_entries=False, plot_rejects=False, plot_seg = False, plot_vis = False,
                          reject_by_entry=True, reject_by_correlation=True, **kwargs):
    """Given a results csv of images and their marked silhouettes, collect and average all of them"""

    data = {}  # data to be dict of img_src: [labelled_keypoints]
    data_by_row = {} # dict of n_row : (img, segment_uri)
    data_out = [] # Output data, containing dicts of: img_path, img_width, img_height, segment array

    # Create sub directories if necessary

    out_dir = os.path.join(images_dir, "segments")
    try_mkdir(out_dir)

    # MAKE SUB DIRS: ENTRIES, REJECTS, SEGS, SEGS_VIS
    for folder in ["entries", "rejects", "segs", "segs_vis"]:
        try_mkdir(os.path.join(out_dir, folder))

        # Make subfolders too
        if folder != "rejects":
            for subfolder in [f for f in os.listdir(images_dir) if f[:3] == "n02"]:
                try_mkdir(os.path.join(out_dir, folder, subfolder))

    with open(segment_loc, "r") as infile:
        reader = csv.reader(infile)

        headers = next(reader)

        for n_row, line in enumerate(reader):
            *_, img, height, width, segment_uri, status = get_columns(headers, line, "img",
                                                                      "img_height",
                                                                      "img_width", "segment", "status")

            if status != "Rejected":
                data[img] = data.get(img, []) + [(n_row, segment_uri)]  # add to dictionary, store row of operation
                data_by_row[n_row]  = (img, segment_uri)

    progress = tqdm.tqdm(total = len(data_by_row))
    rejects = [] # list of rejected rows
    errors = []
    corrs = []

    # REJECTION/DISCARDING CRITERIA
    correlation_rejection = 0.75 # 75% or worse
    correlation_discard = 0.8 # 80% or worse

    for i, (img, segments) in enumerate(data.items()):
        img_data = plt.imread(os.path.join(images_dir, img))
        if plot_entries:
            fig, axes = plt.subplots(ncols=len(segments))
            if len(segments) == 1: axes = [axes]
            [ax.imshow(img_data) for ax in axes]

        H, W, _ = img_data.shape

        ## CONVERT EACH URI TO A NUMPY ARRAY AND PLOT
        colours = ["red", "green", "blue"]
        binary_data = []

        for n, (n_row, segment_uri) in enumerate(segments):

            segment_rgb = read_uri(segment_uri)

            if segment_rgb is None:
                errors.append(n_row)
                continue # Ignore errors for now

            if plot_entries:
                ax = axes[n]
                ax.imshow(segment_rgb, alpha = 0.5)

            grayscale_segment = np.dot(segment_rgb[...,:3], [0.2989, 0.5870, 0.1140])
            binary_segment = grayscale_segment > 0.1
            binary_data.append(binary_segment.astype(int)) # store binary segment as float, not bool

            if reject_by_entry:

                if (grayscale_segment < 0.01).all():
                    rejects.append(n_row) # If blank, append to rejects and continue
                    continue

                # segment = fill_segment(binary_segment)
                points = image_to_polygon(binary_segment) # of shape (n_polygons, n_points, 2)

                # only consider polygons with area greater than 1% of total image area
                filtered_points = [p for p in points if PolyArea(p)/(W*H)>0.01]

                if len(filtered_points) == 0: # No sufficiently large polygons identified
                    rejects.append(n_row)

            progress.update()

        accepted_segments = []
        if reject_by_correlation: # Rejection stage considering all 3 entries together
            if len(binary_data) >= 3 or any(s[0] in rejects for s in segments): # Dataset not finished or rejection not updated, only consider data with =3 entries
                n_positive = [i.sum() for i in binary_data]

                text = ""
                correlations_by_idx = {} # correlation values by index of image (0, 1, 2)

                combs = []
                N = len(binary_data)
                for i in range(N):
                    for j in range(i+1, N):
                        combs.append((i,j)) # add all combination pairs

                for i, j in combs:
                    A, B = binary_data[i], binary_data[j]

                    # COMPUTE RATIO OF TOTAL PIXELS POSITIVE IN BOTH, TO MAXIMUM NUMBER OF POSITIVE PIXELS IN EITHER IMAGE

                    max_positive = max((n_positive[i], n_positive[j]))

                    if max_positive < 1: correlation = 0
                    else: correlation = (A * B).sum() / max_positive

                    for idx in [i, j]:
                        correlations_by_idx[idx] = correlations_by_idx.get(idx, []) + [correlation]

                    text += f"\nC{i}{j} = {round(correlation, 3)}"

                for idx in range(N):
                    corr = max(correlations_by_idx[idx]) # Maximum correlation with other images
                    corrs.append(corr)

                    if corr < correlation_rejection: # REJECT ENTRY
                        n_row = segments[idx][0]
                        rejects.append(n_row)
                    elif corr < correlation_discard: # DISCARD ENTRY
                        pass # do nothing for now
                    else: # if accepted, store in accepted entries
                        accepted_segments.append(binary_data[idx])

        if plot_entries:
            plt.savefig(os.path.join(out_dir, "entries", img))
            plt.close(fig)

        # IF ANY VALID ENTRIES, STORE OUTPUT
        if not reject_by_correlation: accepted_segments = binary_data # If none rejected, accept all binary data
        if len(accepted_segments) > 0:
            # INCLUDE ANY PIXEL IF IN MAJORITY OF ENTRIES
            n_accept = len(accepted_segments)
            seg_out = (sum(binary_data) / n_accept) > 0.5
            seg_out = 255 * np.stack([seg_out]*3, axis=-1).astype(np.uint8)

            # plt.imshow(seg_out)
            # plt.show()

            if plot_seg:
                cv2.imwrite(os.path.join(out_dir, "segs", img),
                       seg_out, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # write image with no compression
                # plt.imsave(os.path.join(out_dir, "segs", img), seg_out)

            if plot_vis:
                plt.imshow(img_data)
                plt.imshow(seg_out, alpha = 0.5)
                plt.savefig(os.path.join(out_dir, "segs_vis", img))
                plt.close("all")

        progress.set_description(f"OUT: {len(data_out)}| REJECT: {len(rejects)}| ERROR: {len(errors)}")

    # PLOT ALL REJECTS
    if plot_rejects:
        for n_row in rejects:
            img, uri = data_by_row[n_row]

            plt.imshow(plt.imread(os.path.join(images_dir, img)))
            plt.imshow(read_uri(uri), alpha=0.5)

            plt.savefig(joinp(out_dir, "rejects", f"{n_row}.png"))
            plt.close()

    # REJECT WORK
    if reject_by_correlation or reject_by_entry:
        pass
        # reject_works(rejects, segment_loc, message = "Incomplete/inaccurate segmentation submitted.")

    np.save(joinp(out_dir, "correlations.npy"), corrs)

def filter_keypoints_by_segmentations(**kwargs):
    """Produces a filtered version of keypoints.json, that includes only images for which all keypoints fit inside the segmentation"""

    segments_dir = osp.join(DataSources.datasets, "Dogs", "Full dog dataset", "segments")
    unfiltered_keypoints_loc = osp.join(segments_dir, "keypoints_10k.json")

    # load
    unfiltered = json.load(open(unfiltered_keypoints_loc, "r"))
    filtered = []

    no_seg_rejects = 0
    other_rejects = 0

    N = len(unfiltered)
    progress = tqdm.tqdm(total = N)

    pad = 5 # allow 5 pixels for joints very close to segmentation
    for n, entry in enumerate(unfiltered):
        img_path = entry["img_path"]
        joints = [(x,y) for (x,y,v) in entry["joints"] if (x,y)!=(0,0) and v == 1] # Only count non zero, visible keypoints

        seg_path = joinp(segments_dir, "segs", img_path)

        if osp.isfile(seg_path): # Only consider for acceptance if there is a seg file
            rgb_img = plt.imread(seg_path)
            segment = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140]) > (rgb_img.max() / 2) # load image as binary mask
            # Values must exceed half of the maximum value to deal with compression resulting in dimly lit pixels

            n_joints = len(joints)
            n_joints_succesful = 0 # number of joints within the segmentation

            for (x, y) in joints:
                # try every combination of x rounded up and down. If any of these are in segmentation, accept
                if segment[int(y-pad):int(y+pad), int(x-pad):int(x+pad)].sum() >= 1:
                    n_joints_succesful += 1

            if n_joints - n_joints_succesful <= 1:
                filtered.append(entry)

            else:
                other_rejects += 1
                # print(n_joints, n_joints_succesful)
                # plt.scatter([j[0] for j in joints], [j[1] for j in joints])
                # plt.imshow(rgb_img, alpha = 0.5)
                # plt.show()
                #

        else: # if no seg file
            no_seg_rejects += 1

        progress.update()
        progress.set_description(f"Accepted: {len(filtered)}, No Seg: {no_seg_rejects}, Other: {other_rejects}")

    with open(os.path.join(segments_dir, "keypoints.json"), 'w') as outfile:
        json.dump(filtered, outfile)

def produce_file_mapping(src_dir):
    """Youtube dataset was having issues with file names being irregular. This function:
    Takes a src dir, maps every image to a number, saves the mapping in the folder, and renames each file"""

    if os.path.isfile(joinp(src_dir, "mapping.json")):
        if input("Mapping.json already exists. Check mapping is necessary. 'y' to continue: ") != "y":
            return None

    mapping = {}
    isimg = lambda f: any(ext in f for ext in [".png", ".jpg"])
    for n, file in enumerate([f for f in os.listdir(src_dir) if isimg(f)]):
        new_name = f"{n:05d}.png"
        mapping[new_name] = file
        os.rename(joinp(src_dir, file), joinp(src_dir, new_name))

    with open(joinp(src_dir, "mapping.json"), "w") as outfile:
        json.dump(mapping, outfile)

def apply_file_mapping_to_keypoints(keypoints, mapping):
    """Given a json mapping file-names to new names, edit the entries of a keypoints.json file"""

    kp_data = json.load(open(keypoints))
    raw_mapping = json.load(open(mapping))
    mapping = {v:k for k, v in raw_mapping.items()}

    print(mapping)

    for entry in kp_data:
        entry['img_path'] = mapping[entry['img_path']]

    json.dump(kp_data, open(keypoints.replace(".json", "_mapped.json"), "w"))

def modify_bboxs_to_keypoints(json_src, pad_frac = 0.1, plot = False):
    """Goes through any unidentified bboxs, and modifies them to be the bounding box of the joints, + padding.

    pad_frac is a frac of avg(img height, img width)"""
    img_dir = r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\output_frames"

    with open(json_src) as infile:
        data = json.load(infile)

    for i in data:
        x0, y0, width, height = i['img_bbox']
        if x0==0 and y0==0 and width==i['img_width'] and height==i['img_height']:
            pad = pad_frac * (width + height) / 2
            J = np.array(i['joints'])
            J = J[J[:, 0] > 0]
            x0 = max(J[:, 0].min() - pad, 0)
            width = min(J[:, 0].max() - J[:, 0].min() + (2*pad), i['img_width'])
            y0 = max(J[:, 1].min() - pad, 0)
            height = min(J[:, 1].max() - J[:, 1].min() + (2*pad), i['img_height'])

            i['img_bbox'] = [x0, y0, width, height]

            if plot:
                fig, ax = plt.subplots()
                img_data = plt.imread(os.path.join(img_dir, i['img_path']))
                x, y, z = zip(*i['joints'])
                ax.scatter(x, y)
                ax.imshow(img_data)
                ax.plot([x0, x0+width, x0+width, x0, x0], [y0, y0, y0+height, y0+height, y0])
                plt.show()

    with open(json_src.replace("mapped", "mapped_bbox_fix"), "w") as outfile:
        json.dump(data, outfile)

modify_bboxs_to_keypoints(r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\keypoints_mapped.json",
                          plot = False)

# yt_set_2_targ = r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\set_2"
yt_all = r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\output_frames"
# produce_file_mapping(yt_all)

# apply_file_mapping_to_keypoints(r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\keypoints.json",
#                                 joinp(yt_all, "mapping.json"))

# SEGMENTATION WORK
# extract_segmentations(**dog_dataset, plot_entries=True, plot_seg=True, plot_vis=True,
#                       reject_by_entry=True, reject_by_correlation=True, plot_rejects=True)

# plot_correlations()

# filter_keypoints_by_segmentations(**dog_dataset)

#
# rejects = [int(i.split(".")[0]) for i in os.listdir(r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\segments\rejects")]
# reject_works(rejects, dog_dataset["segment_loc"],
#              message = "Insufficient accuracy of segmentation. Either dog(s) not segmented at all, or not accurately enough.")

data_set = youtube_dog_dataset

# plot_correlations()

# get_bboxs_from_silhouettes(silhouette_dir, actual_images_dir = images_dir)
# get_bboxs_from_annotations(**data_set)
# get_file_list_with_bbox_violations(**data_set)

# process_keypoint_data(**data_set, minimum_valid_keypoints=8, plot_simple=True)
#
# FOR ARENA SET - WORKER ANALYSIS:
# workers = ['A1H44G8YJPEX2A', 'A3ODX7RC28VYZX', 'A1KVKZYFGD0FAR', 'A2RV6QX6SN4YIC', 'A1JAWZLPP7B23M', 'AB5BVAK003LIX', 'A8MGW12LQD1C2']
# workers = "A1OGNQQVNUT3K8, A1PEMQRATTPWER, ARVU9MVZ3A5MO, A1W6KRCE6M04TF, A3CL56BPWX71PA, A3ODX7RC28VYZX, A1JAWZLPP7B23M, A15SUHWVUBK9G6, A1XORD1EZH6UI0".split(", ")
# workers = "A2T82A0E42NI51, AMK5G3916QF5N".split(", ")
# for w in workers:
#     if not os.path.isdir(os.path.join(arena_dog_dataset['images_dir'], "worker_evals", w)):
#         os.mkdir(os.path.join(arena_dog_dataset['images_dir'], "worker_evals", w))
#
#     view_entries_by_worker(w, data_set['images_dir'])

# view_entries_by_worker("A1JAWZLPP7B23M", data_set['images_dir'], min_kp=10)

# rejects = [int(i.split(".")[0]) for i in os.listdir(os.path.join(arena_dog_dataset["images_dir"], "worker_evals", "A2T82A0E42NI51"))]
# rejects += [3311, 4135, 4157, 4418, 4583, 7023, 7121, 574, 7, 7058, 7225, 5127, 5015, 4924, 4742, 1365, 1303, 1228, 1264, 6223, 7037, 6924, 6971, 5759, 5877, 4898, 4904, 4994, 4670, 4763]
# reject_works(rejects, file = os.path.join(arena_dog_dataset["images_dir"], "results.csv"))
from my_work.annotated_joints_plotter import *
from scipy.ndimage import center_of_mass as COM
from matplotlib import colors

def extract_colours(rgb_array):
    """Given an array of (r,g,b) values, returns list of unique colours"""
    return np.unique(rgb_array.reshape(-1, rgb_array.shape[2]), axis=0)

def hex_to_rgb(hex):
    return np.array([int(hex[i:i+2], 16) for i in (0, 2, 4)])

def hex_to_unit_rgb(hex):
    out = np.array([int(hex[i:i+2], 16) for i in (0, 2, 4)])
    if out.max() > 1:
        out = out /255    # convert to r, g, b values between 0 and 1
    return out

def rgb_to_bin(rgb, t=0.1):
    """Make rgb into binary mask array"""
    gs = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return  gs > (gs.max() * t)

def extract_segmentations(images_dir, segment_loc, plot_entries=False, plot_rejects=False, plot_seg = False, plot_vis = False,
                          reject_by_entry=True, reject_by_correlation=True, **kwargs):
    """Given a results csv of images and their marked silhouettes, collect and average all of them.

    Produce output JSON in correct order, with numpy arrays of """

    data = {}  # data to be dict of img_src: [labelled_keypoints]
    data_by_row = {} # dict of n_row : (img, segment_uri)

    out_dir = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments"
    src_keypoints = joinp(out_dir, "keypoints_base.json")
    input_data = json.load(open(src_keypoints, "r")) # load existing data
    tt = np.load(joinp(out_dir, "splits_base.npz"))
    train, val = list(tt["train_list"]), list(tt["val_list"])

    # Create sub directories if necessary


    try_mkdir(out_dir)

    results_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\results_ear_tail.csv"
    with open(results_loc, "r") as infile:
        reader = csv.reader(infile)

        headers = next(reader)

        for n_row, line in enumerate(reader):
            *_, img, height, width, segment_uri, status = get_columns(headers, line, "img",
                                                                      "img_height",
                                                                      "img_width", "segment", "status")

            if n_row == 0:
                # define dicitonary of name : colour
                seg_names = ["Left ear tip", "Right ear tip", "Left ear", "Right ear", "Tail"]
                colour_assignments = get_columns(headers, line, *seg_names)

            if status != "Rejected":
                data[img] = data.get(img, []) + [(n_row, segment_uri)]  # add to dictionary, store row of operation
                data_by_row[n_row]  = (img, segment_uri)


    p = tqdm.tqdm(total = len(input_data))
    n_succ, n_errors = 0, 0
    n_ear_tips_labelled = 0

    for index, entry in enumerate(input_data):
        no_error = True

        img_src = entry["img_path"]
        W, H = entry["img_width"], entry["img_height"]
        keypoints = entry["joints"]

        img_data = plt.imread(os.path.join(images_dir, img_src), format="jpg")

        H, W, _ = img_data.shape

        if img_src not in data:
            no_error = False

        elif len([i for i in keypoints if i != [0,0,0]]) < 5:
            no_error = False

        else:
            segments = data[img_src]

            segment_data = {}
            for n, (n_row, segment_uri) in enumerate(segments):
                segment_rgb = (read_uri(segment_uri)[:,:,:3] * 255).astype(np.int16) # get as rgb

                ## separate by colour
                # colours = extract_colours(segment_rgb)
                for i, hex_col in enumerate(colour_assignments):
                    name = seg_names[i]
                    colour = hex_to_rgb(hex_col[1:])
                    indices_list = np.any(segment_rgb == colour, axis=-1) # extract only this colour

                    this_channel = np.zeros((H, W))
                    this_channel[indices_list] =  True

                    segment_data[name] = segment_data.get(name, []) + [this_channel]

            # Average across segments to produce binary mask
            for seg_name in segment_data:
                segment_data[seg_name] = np.mean(segment_data[seg_name], axis = 0) > 0.5

            # add ears to keypoint
            r_ear = COM(segment_data["Right ear tip"])[::-1]
            l_ear = COM(segment_data["Left ear tip"])[::-1]
            ear_tips = False
            for ear in (r_ear, l_ear):
                if any(np.isnan(i) for i in ear):
                    entry["joints"].append([0,0,0])
                else:
                    entry["joints"].append([*ear, 1])
                    ear_tips = True

            entry["ear_tips_labelled"] = ear_tips
            n_ear_tips_labelled += ear_tips

            main_segment = plt.imread(joinp(images_dir, "segments", "segs", img_src)) # segmentation of main body

            # OUTPUT ARRAY with 0 for nothing, 1 for main dog, 2 for left ear, 3 for right ear, 4 for tail
            segment_out = np.zeros((H, W))

            segment_out[rgb_to_bin(main_segment, t = 0.5)] = 1
            segment_out[segment_data["Left ear"]] = 2
            segment_out[segment_data["Right ear"]] = 3
            segment_out[segment_data["Tail"]] = 4

            folder = img_src.split("/")[0]
            try_mkdir(joinp(out_dir, "segs", folder))
            np.save(joinp(out_dir, "segs", img_src.replace("jpg", "npy")), segment_out.astype(np.int16))

        if no_error:
            n_succ += 1

        else:
            if index in train: train.remove(index)
            if index in val: val.remove(index)
            n_errors += 1

        if len(entry["joints"]) < 20:
            entry["joints"] += [[0,0,0]] * (len(entry["joints"]) - 20)

        p.update()
        p.set_description(f"Success: {n_succ}, Errors: {n_errors}, n_ET = {n_ear_tips_labelled}")

    with open(joinp(out_dir, "keypoints_filtered.json"), "w") as outfile:
        json.dump(input_data, outfile)

    np.save(joinp(out_dir, "train.npy"), train)
    np.save(joinp(out_dir, "val.npy"), val)


def get_variation_by_keypoint(images_dir, keypoints_loc, keypoint_list, **kwargs):
    """Produce a dictionary of size (N_keypoints, X, 2) which gives the deviations of each keypoint in each frame
    from the mean."""

    output_data = {}  #

    filtered_json_src = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\keypoints_filtered.json"
    filtered_json = {i["img_path"]: i for i in json.load(open(filtered_json_src))}

    data = {}  # data to be dict of img_src: [labelled_keypoints]
    segment_worker_data = {} # data to be dict of img_src: segment data
    img_sizes = {}  # data to be dict of img_src: (width, height)

    # load normal keypoints
    with open(keypoints_loc, "r") as infile:
        reader = csv.reader(infile)

        headers = next(reader)

        for line in reader:
            *_, img, is_multiple_dogs, height, width, answers, status = get_columns(headers, line, "img",
                                                                                    "multiple_dogs", "img_height",
                                                                                    "img_width", "answer", "status")

            # answers originally an array of dicts, where each dict has keys label, x and y
            # convert to a single dict, with format label: (x,y)

            answers = eval(answers)

            if status != "Rejected":
                answer_dict = {i["label"]: (i["x"], i["y"]) for i in answers}

                data[img] = data.get(img, []) + [answer_dict]  # add to dictionary
                img_sizes[img] = (int(width), int(height))

    # load ear tail keypoints
    results_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\results_ear_tail.csv"
    with open(results_loc, "r") as infile:
        reader = csv.reader(infile)

        headers = next(reader)

        for n_row, line in enumerate(reader):
            *_, img, height, width, segment_uri, status = get_columns(headers, line, "img",
                                                                      "img_height",
                                                                      "img_width", "segment", "status")

            if n_row == 0:
                # define dicitonary of name : colour
                seg_names = ["Left ear tip", "Right ear tip", "Left ear", "Right ear", "Tail"]
                colour_assignments = get_columns(headers, line, *seg_names)

            if status != "Rejected":
                segment_worker_data[img] = segment_worker_data.get(img, []) + [(n_row, segment_uri)]  # add to dictionary, store row of operation

    # For now, load all data from any assignments, and just average any data to form the basis for the actual plot.
    # Scheme to dismiss points:
    # If labelled in only 1 dataset, dismiss
    # If labelled in only 2 datasets, if rms error > threshold, dismiss
    # If labelled in 3 datasets, if rms error > threshold, dismiss
    # If in any case, not dismissed, final position is the average of any datasets

    progress_bar = tqdm.tqdm(total=len(data))

    # Declare figures outside of loop, to minimise memory wastage of creating thousands of figures

    for img, answers in list(data.items()):
        if img not in filtered_json:
            progress_bar.update()
            continue # skip any not in final json

        entry = filtered_json[img]

        # Set thresholds for deciding if data set is valid
        width, height = entry["img_width"], entry["img_height"]
        threshold = (width + height) / 80

        image_keypoint_data = dict(img_path=img, img_width=width, img_height=height,
                                   joints=[])  # place data here for .json output
        joint_data = {}  # dict of joint : (mean_x, mean_y, visibility) for all keypoints

        for i, keypoint_label in enumerate(keypoint_list):
            colour = colours[keypoint_label]  # get colour (ignoring hidden command)

            if any(keypoint_label + " (hidden)" in answer for answer in answers):
                keypoint_label = keypoint_label + " (hidden)"  # change keypoint name to reflect this

            if [answer.get(keypoint_label) for answer in answers if keypoint_label in answer] == []:
                continue

            all_xs, all_ys = zip(
                *[answer.get(keypoint_label) for answer in answers if keypoint_label in answer])  # Get all non blank xs
            mean_x, mean_y = mean(all_xs), mean(all_ys)

            n_valid = len(all_xs)

            discard = False

            if n_valid == 1:
                # For now, if fewer than 3 submitted, accept 1x n_valid. Change this later when dataset is finished
                discard = len(answers) >= 3

            while n_valid > 2:
                if rms(all_xs, all_ys) > threshold:
                    # Discard worst point and try as 2 data points
                    n_dis = most_anomalous(all_xs, all_ys)
                    all_xs, all_ys = all_xs[:n_dis] + all_xs[n_dis + 1:], all_ys[:n_dis] + all_ys[n_dis + 1:]
                    mean_x, mean_y = mean(all_xs), mean(all_ys)
                    n_valid -= 1
                else:
                    break

            for n, (x, y) in enumerate(zip(all_xs, all_ys)):
                if x != 0 and y != 0:
                    output_data[i] = output_data.get(i, []) + [(x - mean_x, y - mean_y)]

        # Get ear tips data
        if img not in segment_worker_data:
            progress_bar.update()
            continue

        segments = segment_worker_data[img]

        segment_data = {}

        for n, (n_row, segment_uri) in enumerate(segments):
            segment_rgb = (read_uri(segment_uri)[:,:,:3] * 255).astype(np.int16) # get as rgb

            ## separate by colour
            # colours = extract_colours(segment_rgb)
            for i, hex_col in enumerate(colour_assignments):
                name = seg_names[i]
                colour = hex_to_rgb(hex_col[1:])
                indices_list = np.any(segment_rgb == colour, axis=-1) # extract only this colour

                this_channel = np.zeros((height, width))
                this_channel[indices_list] =  True

                segment_data[name] = segment_data.get(name, []) + [this_channel]

        r_ears = [COM(i) for i in segment_data["Right ear tip"]]
        l_ears = [COM(i) for i in segment_data["Left ear tip"]]

        # Average across segments to produce binary mask
        segment_data_mean = {}
        for seg_name in segment_data:
            segment_data_mean[seg_name] = np.mean(segment_data[seg_name], axis = 0) > 0.5

        # add ears to keypoint
        r_ear_mean = COM(segment_data_mean["Right ear tip"])
        l_ear_mean = COM(segment_data_mean["Left ear tip"])

        if entry["ear_tips_labelled"]:
            for ear_idx in [0,1]:
                mean_y, mean_x = [r_ear_mean, l_ear_mean][ear_idx]
                ear_tip = [r_ears, l_ears][ear_idx]
                for (y, x) in ear_tip:
                    if not np.isnan(x) and not np.isnan(y):
                        idx = 18 + ear_idx
                        output_data[idx] = output_data.get(idx, []) + [(x - mean_x, y - mean_y)]

        progress_bar.update()


    with open(r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\variation_by_keypoint.json", "w") as outfile:
        json.dump(output_data, outfile)

def plot_keypoint_heatmaps():
    """Produce plot of variation for each keypoint on to an annotated dog image"""

    data_src = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\variation_by_keypoint.json"
    data = json.load(open(data_src, "r"))

    img_dir = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset" # select another
    img_src = r"n02091244-Ibizan_hound/n02091244_3373.jpg"
    img = plt.imread(joinp(img_dir, img_src))

    # Load keypoints
    keypoint_src = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\keypoints_filtered.json"

    # # USE THIS CODE BLOCK TO FIND IMAGES WITH A GOOD NUMBER OF KEYPOITNS LABELLED
    # for i in json.load(open(keypoint_src)):
    #     keypoints = i["joints"]
    #     n_k = len([x for x in keypoints if x != [0.0, 0.0, 0.0]])
    #     if n_k == 20:
    #         plt.imshow(plt.imread(joinp(img_dir, i["img_path"])))
    #         plt.title(i["img_path"])
    #         print(i["img_path"])
    #         plt.show()

    img_entry = [i for i in json.load(open(keypoint_src)) if i["img_path"] == img_src][0]
    keypoint_data = img_entry["joints"]
    X, Y, V = [list(i) for i in zip(*keypoint_data)]

    H, W = img_entry["img_height"],img_entry["img_width"]

    dpi = 300
    fig, ax = plt.subplots(figsize = (W / dpi, H / dpi))
    ax.imshow(img)

    pix_width = 20.5 # search range for bins

    for n in range(20):
        d = data[str(n)]
        d = [(x,y) for (x,y) in d if not np.isnan(x) and not np.isnan(y)]

        if n >= 18: # switch x and y for n >= 18 due to error in ear tips
            X[n], Y[n] = Y[n], X[n]

        if X[n] != 0:
            x_hist, y_hist = zip(*d)
            # normalise to center of actual keypoint

            x_norm = [x + X[n] for x in x_hist]
            y_norm = [y + Y[n] for y in y_hist]

            # for bins, only consider spread within range (-10, 10) from center
            bin_x_range = np.arange(X[n] - pix_width, X[n] + pix_width, 1.0)
            bin_y_range = np.arange(Y[n] - pix_width, Y[n] + pix_width, 1.0)

            colour = hex_to_unit_rgb(colours[dog_dataset["keypoint_list"][n]][1:]) # get colour in (r,g,b) format
            colour_ramp = [[*colour, i] for i in np.arange(0, 1, 0.01)] # from full alpha to full colour
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap', colour_ramp)

            h, *_ = np.histogram2d(x_norm, y_norm)

            vmax = sorted(h.flatten())[-5] # will be distorted by maximum 4 verts, so look below that

            ax.hist2d(x_norm, y_norm, bins = [bin_x_range, bin_y_range], cmap=cmap, vmax = vmax)
            # ax.scatter(X[n], Y[n], c = [colour])

            # fig, ax = plt.subplots()
            # print(X[n], Y[n])
            # h, xedg, yedg, image = ax.hist2d(x_norm, y_norm, bins = [np.arange(X[n]- 9.5, X[n] + 9.5, 1.0), np.arange(Y[n]- 9.5, Y[n] + 9.5, 1.0)], cmap = cmap)
            # print(h)
            # plt.show()

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    plt.subplots_adjust(left=0, bottom=0, right = 1, top = 1)

    plt.savefig(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\dataset_statistics\heatmap.png", dpi = dpi)


# extract_segmentations(**dog_dataset)
# get_variation_by_keypoint(**dog_dataset)
# plot_keypoint_heatmaps()
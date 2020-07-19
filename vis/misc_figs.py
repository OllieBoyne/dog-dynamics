"""Miscallaneous figures produced for project"""

from dataset_production.mturk_processor import *
from vis.utils import *
import json
from scipy import ndimage
from matplotlib import colors

def brighten_with_mask(img, mask, factor=1.0):
    """Given an image of size (H x W x 4), and a mask of size (H x W), brightens all pixels in mask by factor <factor>.
    Always returns rgb in range 0-1"""

    assert img.shape[:2] == mask.shape[:2], "Image and mask must share first 2 dimensions"
    assert img.shape[-1] in [3,4], "Image must be RGB or RGBA"

    img = img.astype(np.float32)
    if img.max() > 1:
        img *= 1/255

    # create modified rgb
    mod_rgb = img.copy()
    hsv = colors.rgb_to_hsv(mod_rgb)
    hsv[:,:,2] *= factor
    hsv = np.clip(hsv, a_min=0, a_max=1)
    mod_rgb = colors.hsv_to_rgb(hsv)

    # ignore anything within the mask in the original image, anything outside of the mask in the brightened image
    img[np.invert(mask)] = [0,0,0]
    mod_rgb[mask] = [0,0,0]

    return (img + mod_rgb)

def video_of_labelled_ally():
    """Produce video of Ally jumping, labelled, from cam x"""

    kp_src = joinp(arena_dog_dataset["annotation_dir"], "labelled_keypoints", "keypoints.json")
    kp_data = json.load(open(kp_src))
    kp_data_by_img = {i["img_path"]: i["joints"] for i in kp_data}

    img_dir = joinp(arena_dog_dataset["images_dir"], "Ally Jump")

    images = sorted([i for i in os.listdir(img_dir) if "2_" in i])[10:]

    dpi = 300
    fig, ax = plt.subplots(figsize=(1280/dpi, 720/dpi))
    ax.axis("off")

    p = tqdm.tqdm(total=len(images))

    def anim(i):
        ax.clear()
        img = images[i]

        img_data = plt.imread(joinp(img_dir, img))
        ax.imshow(img_data)

        full_img_path = f"Ally Jump\\{img}"
        if full_img_path in kp_data_by_img:

            kp = kp_data_by_img[full_img_path]

            for n, (x,y,v) in enumerate(kp):
                if [x,y,v] != [0.0,0.0,0.0]:
                    kp_name = arena_dog_dataset["keypoint_list"][n]
                    col = colours[kp_name]
                    ax.scatter(x, y, c=[col], s = 0.5)

        p.update()

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    save_animation(anim, fig, frames = len(images), dpi = dpi, fps = 10, dir = r"C:\Users\Ollie\Videos\iib_project\misc_videos", title="Ally jump labelled")

def dataset_collage_selector():
    """Displays images of dogs to select individual ones for a collage. Selections added to: ../Full dog dataset/good_examples.txt."""
    dataset = dog_dataset
    images_dir = dataset["images_dir"]

    seg_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\segs"
    keypoints_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\keypoints_filtered.json"

    # criteria:
    # 20 keypoints
    # seg

    json_arr = json.load(open(keypoints_loc))
    np.random.shuffle(json_arr) # randomise order

    for entry in json_arr:
        img_path = entry["img_path"]
        W, H = entry["img_width"], entry["img_height"]
        kp = entry["joints"]

        seg_path = osp.join(seg_loc, img_path.replace(".png", ".npy").replace(".jpg", ".npy"))

        X, Y, V = np.swapaxes(kp, 0, 1)

        if (X > 0).sum() == 20 and os.path.isfile(seg_path):
            seg = np.load(seg_path) > 0  # load seg, convert to bin mask

            img_data = plt.imread(joinp(images_dir, img_path))

            plt.imshow(img_data)

            for n, (x,y) in enumerate(zip(X,Y)):
                kp_name = dataset["keypoint_list"][n]
                col = colours[kp_name]
                plt.scatter(x, y, c=[col])

            plt.imshow(seg, alpha = 0.5)
            print(img_path)
            plt.show()

connections = [[0,1],[1,2], [3,4],[4,5], [6,7],[7,8], [9,10],[10,11], [15, 18], [14, 19]]
def dataset_collage_plotter():
    """Collage of dataset of labelled dogs, with segmentations/keypoints shown"""
    dataset = dog_dataset
    images_dir = dataset["images_dir"]
    seg_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\segs"
    keypoints_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\keypoints_filtered.json"

    seg_col = [14/255, 164/255, 121/255, 0.6]

    selected_imgs_src = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\good_examples.txt"
    selected_imgs = [l.replace("\n","") for l in open(selected_imgs_src, "r").readlines()]

    json_arr = json.load(open(keypoints_loc))
    json_dict = {i["img_path"] : i for i in json_arr} # dict of img_path : entry

    target_width, target_height = 250, 250 # want each image to be this shape

    M, N = 3, 8

    dpi = 300
    fig, axes = plt.subplots(nrows=M, ncols=N, figsize=(target_width * N / dpi, target_height * M / dpi))

    p = tqdm.tqdm(total=M*N)
    for i in range(M*N):
        img_src = selected_imgs[i]
        ax = axes[i//N, i%N]
        ax.axis("off")

        entry = json_dict[img_src]
        W, H = entry["img_width"], entry["img_height"]
        kp = entry["joints"]
        X, Y, V = np.swapaxes(kp, 0, 1)
        x0, y0, bbox_width, bbox_height = entry["img_bbox"]

        # default pad
        pad = min(x0, y0, W-bbox_width-x0, H-bbox_height-y0) # largest available pad that won't cause issues with image not fitting
        x0  = max(x0 - pad, 0)
        y0 = max(y0 - pad, 0)
        bbox_width = min(bbox_width + 2* pad, W)
        bbox_height = min(bbox_height + 2 * pad, H)

        # To avoid any scatter points being within <scat_pad> pixels of bbox bounds, increase bbox in these cases
        scat_pad = 5
        for (x, y) in zip(X, Y):
            if (x - x0) < scat_pad:
                x0 -= scat_pad
                bbox_width += scat_pad
            if (x0 + bbox_width - x) < scat_pad:
                bbox_width += scat_pad
            if (y - y0) < scat_pad:
                y0 -= scat_pad
                bbox_height += scat_pad
            if (y0 + bbox_height - y) < scat_pad:
                bbox_height += scat_pad

        # cap all boundaries
        x0, y0, bbox_height, bbox_width = max(x0, 0), max(y0, 0), min(bbox_height, H), min(bbox_width, W)

        # work out individual pads to accomodate bboxs close to the left/top edges
        left_xpad = 0 #min(x0, tot_xpad//2)
        right_xpad = 0 #tot_xpad - left_xpad
        top_ypad = 0 #min(y0, tot_ypad//2)
        bottom_ypad = 0 #tot_ypad - top_ypad

        seg_path = osp.join(seg_loc, img_src.replace(".png", ".npy").replace(".jpg", ".npy"))

        img_data = plt.imread(joinp(images_dir, img_src))

        # first, rescale such that bbox is target width x target height (add padding later)
        resize_x, resize_y = target_width/bbox_width, target_height/bbox_height

        # if any significant aspect ratios, add padding in desired direction
        aspect_pad = 70
        if resize_x/resize_y > 1.4:
            left_xpad = min(x0, aspect_pad//2)
            right_xpad = aspect_pad - left_xpad
            x0 = max(x0-left_xpad, 0)
            bbox_width = min(bbox_width + right_xpad + left_xpad, W)
            resize_x = target_width/bbox_width
        if resize_y/resize_x > 1.4:
            top_ypad = min(y0, aspect_pad)
            bottom_ypad = aspect_pad - top_ypad
            y0 = max(y0 - top_ypad, 0)
            bbox_height = min(bbox_height + bottom_ypad + top_ypad, H)
            resize_y = target_height / bbox_height

        img_data = ndimage.zoom(img_data, [resize_y, resize_x, 1])

        # resize bbox
        [x0, bbox_width] = map(lambda v: int(round(resize_x * v)), [x0, bbox_width])
        [y0, bbox_height] = map(lambda v: int(round(resize_y * v)), [y0, bbox_height])

        img_data_cropped = img_data[y0: y0 + bbox_height, x0: x0 + bbox_width]

        seg = np.load(seg_path)  # load seg
        seg = ndimage.zoom(seg, [resize_y, resize_x]) > 0 # convert to resized, bin mask
        seg_cropped = seg[y0: y0 + bbox_height, x0 : x0 + bbox_width]
        seg_img = np.array([ [ [[0,0,0,0], seg_col][int(seg_cropped[j,i])] for i in range(seg_cropped.shape[1])] for j in range(seg_cropped.shape[0])]) # convert to colours

        img_data_cropped = brighten_with_mask(img_data_cropped, seg_cropped, factor = 1)
        ax.imshow(img_data_cropped)
        # ax.imshow(seg_img, alpha = 1) # this shows segmentation in plot too

        # plot segmentation contour
        polygon = image_to_polygon(seg_cropped)[0]
        x_pol, y_pol = zip(*polygon)
        # plot back to index 0 to close loop
        ax.plot([*x_pol, x_pol[0]], [*y_pol, y_pol[0]], lw = 1.25, c=[0,0,1], alpha=0.7) #lawngreen, darkturquoise

        X_scaled, Y_scaled = X * resize_x - x0, Y * resize_y - y0 # scale joint coords to fit in resized image

        # plot joints & skeleton
        for n, (x,y) in enumerate(zip(X_scaled,Y_scaled)):
            kp_name = dataset["keypoint_list"][n]
            col = colours[kp_name]
            ax.scatter(x, y, c=[col], s = 2, zorder=10)

        # for start, end in connections:
        #     col = colours[dataset["keypoint_list"][start]]
        #     ax.plot(X_scaled[[start, end]], Y_scaled[[start, end]], color=col, lw=.5)

        p.update()

    plt.subplots_adjust(wspace=0, hspace = -.1, left = 0, bottom = 0, top = 1, right = 1)
    plt.savefig(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\collage_report.png", dpi = dpi)

def plot_single_dog(img_path = r"n02100236-German_short-haired_pointer/n02100236_3871.jpg"):
    """Given a single dog, plot the joints results"""

    dataset = dog_dataset
    images_dir = dataset["images_dir"]
    keypoints_loc = r"E:\IIB Project Data\Training data sets\Dogs\Full dog dataset\ear_tail_segments\keypoints_filtered.json"

    json_arr = json.load(open(keypoints_loc))
    json_dict = {i["img_path"]: i for i in json_arr}  # dict of img_path : entry

    entry = json_dict[img_path]
    kp = entry["joints"]
    X, Y, V = np.swapaxes(kp, 0, 1)

    w, h = entry["img_width"], entry["img_height"]
    dpi = 200
    fig = plt.figure(figsize = (w/dpi, h/dpi))
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")

    img_data = plt.imread(joinp(images_dir, img_path))
    ax.imshow(img_data)

    for n, (x, y) in enumerate(zip(X, Y)):
        kp_name = dataset["keypoint_list"][n]
        col = colours[kp_name]
        ax.scatter(x, y, c=[col], s=10, zorder=10)

    plt.savefig(r"E:\IIB Project Data\smal fit results\n02100236_3871 overview\5 gt skel.png", dpi = dpi)

def plot_correlations():
    """Take the current correlations file for segmentations, and plot"""
    src_dir = os.path.join(dog_dataset["images_dir"], "segments")
    plt.rc("text", usetex=True)
    dx = 0.0008
    print(joinp(src_dir, "correlations.npy"))
    data = np.load(joinp(src_dir, "correlations.npy"))
    fig, ax = plt.subplots(figsize=(6.5,3))
    ax_cum = ax.twinx()

    n, bins, hist, = ax.hist(data, bins = [dx * i for i in range(int(1/dx))], density=True, label="Distribution")
    data_cum = dx * np.cumsum(n)
    x = np.arange(0, 1-dx, dx)
    data_cum *= 100 # convert ax_cum to %
    cum = ax_cum.plot(x, data_cum, "r--", label="Cumulative")

    ax.set_xlabel("Acc($A^w$)")
    ax.set_ylabel("Density of entries (\%)")
    ax_cum.set_ylabel("Cumulative count (\%)")

    ax.set_xlim(0.7, 1)
    ax_cum.set_ylim(0,100)
    plt.legend([hist[0]] + cum, ["Distribution", "Cumulative"])

    ax.spines["right"].set_visible(False)
    ax_cum.spines["right"].set_linestyle("dashed")
    ax_cum.spines["right"].set_color("red")

    mu, std = data.mean(), np.std(data)
    ax.text(.8, 10.0, f"$\mu$ = {100*mu:.1f}\% \\ $\sigma$ = {100*std:.1f}\%", ha = "center")

    plt.tight_layout()
    plt.subplots_adjust(top=.98, right=.91, bottom=.15, left =.09)
    plt.savefig(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\segmentations"
                r"\correlations.png", dpi = 300)

if __name__ == "__main__":
    # plot_correlations()
    dataset_collage_plotter()
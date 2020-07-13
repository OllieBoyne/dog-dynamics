"""Scripts for management of the animal-pose dataset, used as a test dataset for our deep network."""

import os, json
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import numpy as np
from shutil import copyfile
from tqdm import tqdm
from matplotlib.patches import Rectangle
from shutil import copy
from vis.utils import try_mkdir

joinp = os.path.join

base = r"E:\IIB Project Data\Training data sets\Dogs\animal-pose"
jbase = lambda x: joinp(base, x)

anno_loc = jbase("annos")
img_dir = jbase("pascal\JPEGImages")  # source of all images
seg_dirs = [jbase("pascal\SegmentationClass"), jbase("pascal\SegmentationObject")]
seg_annos = jbase(r"pascal\Annotations")

img_output_dir = jbase("images")  # source of target images
seg_output_dir = jbase("segs")  # source of target images
segimg_output_dir = jbase("segs_as_img")

keypoints_loc = jbase("keypoints.json")
out_val_loc = jbase("val_animal_pose.npy")

col_map = {
	"l_f": "red", "r_f": "green", "l_b": "yellow", "r_b": "blue", "tail": "pink", "chin": "orange",
	"nose": "orange", "l_ear": "teal", "r_ear": "pink"
}

full_keypoints = 24
kp_to_idx = {"L_F_Paw": 0,
			 "L_F_Knee": 1,
			 "L_F_Elbow": 2,
			 "L_B_Paw": 3,
			 "L_B_Knee": 4,
			 "L_B_Elbow": 5,
			 "R_F_Paw": 6,
			 "R_F_Knee": 7,
			 "R_F_Elbow": 8,
			 "R_B_Paw": 9,
			 "R_B_Knee": 10,
			 "R_B_Elbow": 11,
			 "TailBase": 12,
			 "L_EarBase": 14,
			 "R_EarBase": 15,
			 "Nose": 16,
			 "L_Eye": 20,
			 "R_Eye": 21,
			 "Withers": 22,
			 "Throat": 23,
			 }  # Map of keypoint : index in global keypoint system

idx_to_kp = {i: v for v, i in kp_to_idx.items()}  # inverse lookup

vis_keypoints_blacklist = ["Withers", "Throat", "L_eye", "R_eye"]


def count_vis_keypoints(keypoints):
	"""Given a dict of name : [x, y, z], count the number of valid (not in blacklist),
	visible keypoints"""

	res = 0
	for n, (x, y, v) in keypoints.items():
		if n not in vis_keypoints_blacklist and v == 1:
			res += 1

	return res


minimum_keypoints = 8


def extract_keypoints(plot=False, animal="dog", reject_multiple=True):
	"""Extract keypoints, save to keypoints.json.
	Apply filter based on minimum keypoints"""
	valid_xmls = []  # only filter valid xmls
	seg_count = 0
	output_data = []

	src_dir = os.path.join(anno_loc, animal)
	xmls = os.listdir(src_dir)
	progress = tqdm(total=len(xmls))

	for n_x, xml in enumerate(xmls):
		file_loc = joinp(src_dir, xml)
		tree = ET.parse(file_loc)

		# LOAD XML DATA
		img_src, bbox = tree.find("image").text, tree.find("visible_bounds")
		bbox_map = {k: v for k, v in bbox.items()}
		x0, y0, bbox_width, bbox_height = [float(bbox_map[k]) for k in ["xmin", "ymin", "width", "height"]]
		keypoints_el = tree.find("keypoints")

		keypoints = {}
		for keypoint_el in keypoints_el:
			name, vis, x, y, z = [i[1] for i in keypoint_el.items()]
			keypoints[name] = [float(x), float(y), int(vis)]
		# print(keypoint.items())
		# print(keypoint.name, keypoint.x)

		is_seg = False
		for seg_dir in seg_dirs:
			if img_src + ".png" in os.listdir(seg_dir):
				is_seg = True
		if is_seg: seg_count += 1

		### FILTER
		## Filter by minimum keypoint
		valid = True
		if count_vis_keypoints(keypoints) < minimum_keypoints:
			valid = False

		is_multiple = len([x for x in xmls if x[:-6] == xml[:-6]]) > 1  # if mult xml files for same img
		if reject_multiple and is_multiple:
			valid = False

		if valid:
			valid_xmls += [xml]

			H, W, *_ = plt.imread(joinp(img_dir, img_src + ".jpg")).shape
			joints = []
			for n in range(full_keypoints):
				if n in idx_to_kp:
					joints.append(keypoints[idx_to_kp[n]])
				else:
					joints.append([0, 0, 0])

			data = {"img_path": img_src + ".jpg",
					"img_width": W, "img_height": H,
					"joints": joints, "is_multiple_dogs": is_multiple,
					"img_bbox": [x0, y0, bbox_width, bbox_height],
					"ear_tips_labelled": False,
					"has_segmentation": False
					}

			output_data.append(data)

		if plot:
			# # LOAD IMAGE
			img_data = plt.imread(joinp(img_dir, img_src + ".jpg"))
			plt.title(xml)
			plt.imshow(img_data)

			for name, (x, y, v) in keypoints.items():
				if v == 1:
					c = "grey"
					for tag in col_map:
						if tag in name.lower(): c = col_map[tag]

					marker = "o"
					if "Knee" in name: marker = "s"
					if "Elbow" in name: marker = "^"

					plt.scatter(x, y, label=name, c=c, marker=marker)

				plt.plot([x0, x0, x0 + bbox_width, x0 + bbox_width, x0],
						 [y0, y0 + bbox_height, y0 + bbox_height, y0, y0], lw=2, c="red")

			# PLOT IMAGE
			plt.legend()
			plt.show()

		progress.update()
		progress.set_description(f"Valid: {int(100 * len(output_data) / (n_x + 1))}%")

	# print(len([i for i in output_data if i["is_multiple_dogs"]]))
	out_src = keypoints_loc.replace(".json", f"_{animal}.json")
	with open(out_src, "w") as outfile:
		json.dump(output_data, outfile)


def produce_img_dir(animal="dog"):
	"""For each image file described in keypoints.json, extract and save to /images/"""
	keypoints_src = keypoints_loc.replace(".json", f"_{animal}.json")
	json_data = json.load(open(keypoints_src))
	progress = tqdm(total=len(json_data))

	out_dir = os.path.join(img_output_dir, animal)
	try_mkdir(out_dir)

	for i in json_data:
		img_path = i["img_path"]

		if img_path not in os.listdir(out_dir):
			copyfile(joinp(img_dir, img_path), joinp(out_dir, img_path))

		progress.update()


def dominant_colour(a, is_not=None):
	"""Given an (MxNx3) RGB array, returns the most dominant colour.
	Allows for filter 'is_not', (X x 3) of rejected colours"""
	colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
	include = [np.all([not np.array_equal(c, other) for other in is_not]) for c in colors]
	colors, count = colors[include], count[include]

	return colors[count.argmax()]


def produce_seg_dir(plot=False):
	"""For each image file described in keypoints.json,
	if there is only one dog in the image, extract segmentation as .npy binary file
	and save to /segs/"""

	json_data = json.load(open(keypoints_loc))
	progress = tqdm(total=len(json_data))

	# colour = np.array([0.5019608, 0., 0.]) # Colour corresponding to silh of dog 1

	for i in json_data:
		img_path = i["img_path"].replace(".jpg", ".png")

		if "2011_002279" in img_path:
			plot = True
		else:
			plot = False

		seg_data = None

		if not i["is_multiple_dogs"]:

			# get bbox from xml
			xml_targ = joinp(seg_annos, img_path.replace(".png", ".xml"))
			tree = ET.parse(xml_targ)

			# LOAD XML DATA
			x = tree.findall("object")

			targ = [obj for obj in x if obj.find("name").text == "dog"][0]
			bndbox = targ.find("bndbox")
			x0, y0, x1, y1 = [int(bndbox.find(t).text) for t in ["xmin", "ymin", "xmax", "ymax"]]

			for seg_dir in seg_dirs:
				if img_path in os.listdir(seg_dir):
					seg_data = plt.imread(joinp(seg_dir, img_path))

		if seg_data is not None:
			# Get most common colour from keypoints
			kp = [(int(x[1]), int(x[0])) for x in i["joints"] if x[2] == 1]
			all_colours = []
			for (y, x) in kp:
				for d in [0, 1]: all_colours.append(seg_data[y, x + d])
				for d in [0, 1]: all_colours.append(seg_data[y + d, x])

			colour = dominant_colour(np.array(all_colours),
									 # find colour of desired item as most prevelant in all keypoint pixels
									 is_not=np.array([[0, 0, 0], [0.8784314, 0.8784314, 0.7529412]]).astype(
										 np.float32))  # COLOUR IS NOT ONE OF BLACK (BACKGROUND), OR LIGHT GREY (SEPERATOR)

			W, H = (i["img_width"], i["img_height"])
			indices_list = np.all(seg_data == colour, axis=-1)  # extract only this colour

			seg = np.zeros((H, W))
			seg[indices_list] = True

			if plot:
				fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
				img_data = plt.imread(joinp(img_output_dir, i["img_path"]))
				ax1.imshow(img_data)
				rect = Rectangle((x0, y0), (x1 - x0), (y1 - y0), fill=None, lw=5, ec="red")
				ax1.add_patch(rect)
				ax2.imshow(seg_data)
				ax3.imshow(seg)
				plt.show()

			else:
				plt.imsave(joinp(segimg_output_dir, img_path), seg)

			np.save(joinp(seg_output_dir, img_path.replace("png", "npy")), seg)

		progress.update()


def add_has_seg():
	"""CHANGE KEYPOINTS.JSON TO ADD [HAS_SEGMENTATION] section"""
	data = json.load(open(keypoints_loc))
	for i in data:
		i["has_segmentation"] = i["img_path"].replace(".jpg", ".npy") in os.listdir(seg_output_dir)

	with open(keypoints_loc, "w") as outfile:
		json.dump(data, outfile)


def produce_val():
	"""Produce val split"""
	n_val = len(json.load(open(keypoints_loc)))

	out = np.arange(n_val)
	np.save(out_val_loc, out)


def copy_images_with_segs(results="results 18-4-20"):
	"""Copy all images from results dir /<i> dir to /<i>_with_segs dir that have an accompanying seg"""
	source = jbase(results)
	targ = jbase(results + "_w_segs")
	try_mkdir(targ)

	for seg_img in os.listdir(seg_output_dir):
		img = "images_" + seg_img.replace("npy", "jpg")
		copy(joinp(source, img), joinp(targ, img))


def integrate_bboxs(keypoints_json, bboxs_json):
	"""Given a json of all keypoint entries, integrate the bbox for each entry from the bboxs_json"""

	data = json.load(open(keypoints_json))
	bboxs = json.load(open(bboxs_json))

	for i in data:
		path = i['img_path'].replace("\\", "/")  # deals with subfolders being named in different ways
		i['img_bbox'] = bboxs[path]

	with open(keypoints_json, "w") as outfile:
		json.dump(data, outfile)


if __name__ == "__main__":
	animal = "cat"
	extract_keypoints(animal=animal, reject_multiple=True)
	produce_img_dir(animal=animal)

	# extract_keypoints(plot = False)
	# produce_img_dir()
	# produce_seg_dir(plot = False)
	# add_has_seg()
	# produce_val()
	# copy_images_with_segs()
	# integrate_bboxs(r"E:\IIB Project Data\Training data sets\Dogs\Arena dataset\keypoints.json", r"E:\IIB Project Data\Training data sets\Dogs\Arena dataset\bboxs.json")
	# integrate_bboxs(r"E:\IIB Project Data\Training data sets\Dogs\animal-pose\keypoints.json", r"E:\IIB Project Data\Training data sets\Dogs\animal-pose\bboxs.json")
	# integrate_bboxs(r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\keypoints.json", r"E:\IIB Project Data\Training data sets\Dogs\youtube_clips_1\bboxs.json")

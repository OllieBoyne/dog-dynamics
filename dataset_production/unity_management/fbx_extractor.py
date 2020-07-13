"""Attempt at extracting info from fbx files.

Has several functions for extracting data from directories of obj files, and producing a GUI to select and extract select vertex data.

Selected vertex data exported as a .csv"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import art3d

from vis.mesh_viewer import BBox

default_cm = plt.get_cmap("hot")
get_from_default_cm = lambda N, pad=10: default_cm(np.linspace(0, 1, N + 2 * pad))[
										pad:-pad]

# title="Wolf"
# src_dir = r"E:\IIB Project Data\Blender files\dog runs\extracted_plys"+f"\{title}"
# Add if clause for - if not existing dir, run script to make .obj files

# definitions
# paw = top of paw bit, chin is just under mouth, etc.
keypoint_list = [
	'chin', 'nose', 'base left ear', 'base right ear', 'end of tail',
	'right front paw', 'right front ankle', 'right front knee', 'right front shoulder',
	'left front paw', 'left front ankle', 'left front knee', 'left front shoulder',
	'right rear paw', 'right rear ankle', 'right rear knee', 'right rear shoulder',
	'left rear paw', 'left rear ankle', 'left rear knee', 'left rear shoulder',
]

keypoint_colors = get_from_default_cm(len(keypoint_list))


def try_float(v):
	try:
		return float(v)
	except:
		return v


def extract_first_mesh_from_obj(file, out_dir=r"E:\IIB Project Data\Dog 3D models\all_models\clean_static_meshes",
								title=None):
	"""Given a .obj file, extracts only the first listed mesh, and saves the output in the desired folder, with name =
	obj filename """

	recording = False  # flag if started to read the file for mesh data
	out = []
	if title is None: title = file.split("\\")[-2]
	with open(file) as infile:
		for line in infile.readlines():
			split = line.split(" ")

			if split[0] == "o":  # if new object
				if recording:
					break
				else:
					recording = True

			out += [line]

	with open(os.path.join(out_dir, title + ".obj"), "w") as outfile:
		outfile.writelines(out)


def extract_HP_mesh_from_obj(file, out_dir=r"E:\IIB Project Data\Dog 3D models\all_models\clean_static_meshes",
							 title=None):
	"""Given a .obj file, extracts only the first listed mesh, and saves the output in the desired folder, with name =
	obj filename """

	HP_codes = ["HP", "5k"]  # codes in a mesh name denoting that it is HP

	recording = False  # flag if started to read the file for mesh data
	v0, vn0 = 0, 0  # first index of v and vn for this mesh
	out = []
	if title is None: title = file.split("\\")[-2]
	with open(file) as infile:
		for line in infile.readlines():
			split = line.split(" ")

			if split[0] == "o":  # if new object
				if recording: break  # quit if already loaded object
				if any(c in split[1] for c in HP_codes): recording = True

			if not recording and split[0] == "v": v0 += 1
			if not recording and split[0] == "vn": vn0 += 1

			if recording:
				if split[0] == "f":  # f stored in format: f v1/vt1/vn1 v2/vt2/vn2 ...
					out_line = "f"
					for entry in split[1:]:
						v, vt, vn = list(map(int, entry.split("/")))
						v -= v0
						vn -= vn0
						out_line += f" {v}//{vn}"  # Ignore vt in out
					out_line += "\n"

				else:
					out_line = line

				out += out_line

	if not out: return extract_first_mesh_from_obj(file, out_dir, title=title)  # extract first if nothing else found

	with open(os.path.join(out_dir, title + ".obj"), "w") as outfile:
		outfile.writelines(out)


def read_obj(file):
	"""Loads all vertices from an .obj file, and returns x y z data.

	For files with multiple meshes in, loads the mesh which is high poly (HP)."""
	all_x, all_y, all_z = [], [], []
	recording = False  # flag if started to read the file for mesh data
	with open(file) as infile:
		for line in infile.readlines():
			split = line.split(" ")

			if split[0] == "v":
				v, y, z, x = map(try_float, split)
				# For some reason, data comes in rotated 90 about y axis. Loading data this way fixes this, and isn't disastrous
				# As the correct orientation of the dog doesn't need to be maintained for the 3D prior (as long as it looks normal)
				# If this code is going to be used anywhere else, find a fix for this.
				all_x.append(x), all_y.append(y), all_z.append(z)

	return all_x, all_y, all_z


def tidy_up_dir(src):
	"""For each folder in dir:
	- removes .mtl files
	- Renames all .obj files"""

	all_output_folder = src

	# Clean up directory
	# Delete all mtl files, rename all other ones
	for subf in [f for f in os.listdir(all_output_folder) if os.path.isdir(os.path.join(all_output_folder, f))]:
		directory = os.path.join(all_output_folder, subf)
		for file in os.listdir(directory):
			if ".mtl" in file:
				os.remove(os.path.join(directory, file))
			elif ".obj" in file:
				pass
			else:
				# Original: _00001
				# New: 1.obj
				# Remove _ and trailing 0s, add .obj
				new_file = str(int(file.replace("_", ""))) + ".obj"
				if not os.path.isfile(os.path.join(directory, new_file)):  # For now, overwriting not allowed
					os.rename(os.path.join(directory, file), os.path.join(directory, new_file))

		# # Remove non obj files
		for file in os.listdir(directory):
			if ".obj" not in file:
				os.remove(os.path.join(directory, file))


def convert_to_npy(src_dir, target_dir=r"."):
	"""Given a directory for frames of OBJ files, saves a npy file of the same name in the chosen directory.
	This npy file is a data np array of (n_frames, n_vertices, 3)."""

	outfile_name = src_dir.split("\\")[-1]  # lowest folder name

	n_frames = max(map(lambda x: int(x.split(".")[0]), os.listdir(src_dir))) - 1  # Get frame number

	data = []
	for i in range(n_frames):
		data.append(list(zip(*read_obj(
			src_dir + f"/{i + 1}.obj"))))  # Load file as number frame, with leading digits so it is 6 digits long

	out = np.array(data, dtype="float32")  # Save as 32 bit to save space
	np.save(os.path.join(target_dir, outfile_name + ".npy"), out)  # Save output as numpy array


def run_npy_generation():
	obj_folder = r"E:\IIB Project Data\Dog 3D models\all_models\obj"
	tidy_up_dir(obj_folder)

	# EXTRACT ANIMATION FOR EACH MESH
	anim_dir = r"E:\IIB Project Data\Dog 3D models\all_models\animation_stills"
	for f in os.listdir(obj_folder):
		if not os.path.isdir(os.path.join(anim_dir, f)): os.mkdir(os.path.join(anim_dir, f))

		for frame in [1] + list(range(50, 251, 50)):  # first 50 frames are fairly static
			extract_HP_mesh_from_obj(os.path.join(obj_folder, f, f"{frame}.obj"), out_dir=os.path.join(anim_dir, f),
									 title=f"{f}_{frame}")


inactive_col = (0.0, 0.0, 1.0, 1.0)
active_col = (0.0, 1.0, 0.0, 1.0)


def closest_node(node, nodes):
	"""Memory efficient way of identifying nearest point in <nodes> to <node>.
	Source: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points"""

	nodes = np.asarray(nodes)
	deltas = nodes - node
	dist_2 = np.einsum('ij,ij->i', deltas, deltas)
	return np.argmin(dist_2)


class VertexMesh:
	"""Mesh of vertices defining the dog"""

	def __init__(self, ax, data):
		"""Data is a numpy array of shape (n_frames, n_vertices, 3), and stores all the relevant data"""

		self.data = data[:, ::4, :]  # only load every 4th vertex
		self.ax = ax

		self.n_frames, self.n_vertices, _ = self.data.shape
		self._nv = self.n_vertices
		self.plot = ax.scatter([0] * self._nv, [0] * self._nv, [0] * self._nv, c=np.array([inactive_col] * self._nv),
							   alpha=1)

		self.default_size = self.plot._sizes[0]
		self.sizes = [self.default_size] * self._nv

		# self.default_size = self.plot._sizes[0]
		# self._sizes = [self.default_size] * self._nv

		self.active_vertices = []
		self._frame = 0
		self.plot_frame()

		self.connect_id = 0  # for enabling and disabling clicker mode

	def plot_frame(self, i=0):
		f_data = self.data[i]
		self.plot._offsets3d = art3d.juggle_axes(*zip(*f_data), "z")
		bbox = BBox(*zip(*f_data))
		bbox.equal_3d_axes(self.ax, zoom=2)
		self._frame = i

	def on_click(self, event):
		"""Identify nearest vertex. Toggle this to active"""
		if hasattr(event.inaxes, "_axis3don"):  # Dirty check to check it is the 3D axis
			x, y = event.xdata, event.ydata
			idx = closest_node([x, y], self.plot._offsets)  # IDX of vertex closest to mouse (IN 2d Projection form)
			colour = active_vert_label.colour

			# Add clause for index not already being one of the selected list
			if event.button == 1:
				# If already existing idx for this keypoint, replace it
				if active_vert_label.val is not None:
					self.plot._facecolor3d[active_vert_label.val, :] = inactive_col
					self.sizes[active_vert_label.val] = self.default_size

				self.sizes[idx] = 100
				self.plot._facecolor3d[idx] = colour

				active_vert_label.on_selected(idx)

			update_mode("Rotate")
			self.plot.set_sizes(self.sizes)

			fig.canvas.draw()


class VertexLabel():
	def __init__(self, ax, x, y, label=""):
		self.ax = ax
		self.label = label
		# Use url as tag to verify it is a Vertex Label
		self.colour = keypoint_colors[keypoint_list.index(label)]
		self.text = ax.text(x, y, s="-", ha="center", bbox=dict(facecolor=self.colour), picker=True, url=label)

		self.val = None

	def on_click(self):
		global active_vert_label
		self.text.set_bbox(dict(edgecolor=(.39, 1, .39, 1), lw=5, facecolor=self.colour))  # highlight box
		update_mode("Select")
		active_vert_label = self
		fig.canvas.draw()

	def on_selected(self, val):
		"""After new vertex has been selected"""
		self.val = val
		self.text.set_text(str(val))
		self.text.set_bbox(dict(edgecolor=(.39, 1, .39, 0), facecolor=self.colour))  # Unhighlight box
		fig.canvas.draw()


def on_pick(event):
	if hasattr(event.artist, "_url") and event.artist._url is not None:
		tag = event.artist._url
		vert_labels_dict[tag].on_click()


fig = plt.figure(figsize=(12, 7))
gs = GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[5, 1], width_ratios=[4, 1])
ax = fig.add_subplot(gs[0, 0], projection="3d")
ax.axis("off")
ax_slide = fig.add_subplot(gs[1, 0])
ax_export = fig.add_subplot(gs[1, 1])
ax_table = fig.add_subplot(gs[0, 1])
ax_table.axis("off")

# data = []
# for i in range(n_frames):
#     data.append(list(zip(*read_obj(src_dir+f"/{i+1}.obj")))) # Load file as number frame, with leading digits so it is 6 digits long
# data = np.array(data)

src = r"E:\IIB Project Data\Dog 3D models\all_models\npy\Amstaff01_SV_RM_MX.npy"  # source of numpy
title = src.split("\\")[-1]
data = np.load(src)
n_frames, *_ = data.shape

dog = VertexMesh(ax, data)

slider = Slider(ax_slide, "Frame", 0, n_frames - 1, valinit=0, valstep=1, valfmt='%0.0f')
slider.on_changed(lambda v: dog.plot_frame(int(v)))

# Table of selected vertices
ax_table.text(x=0.5, y=0.95, s="Selected Vertices", ha="center", weight="bold")
vert_id_labels = []  # List of text objects
vert_labels_dict = {}  # dict of keypoint:VertexLabel object

active_vert_label = None

fig.canvas.mpl_connect('pick_event', on_pick)


def setup_table():
	"""Produce table of keypoint : selected index.
	- used when keypoint hasn't yet been selected.
	When clicked on, a '-' can then be changed by clicking on a corresponding vertex in the mesh."""

	for n, (label, col) in enumerate(zip(keypoint_list, keypoint_colors)):
		ax_table.text(x=0.3, y=0.9 - n / len(keypoint_list), s=label, ha="center",
					  bbox=dict(facecolor=keypoint_colors[keypoint_list.index(label)]))

		vl = VertexLabel(ax_table, 0.7, 0.9 - n / len(keypoint_list), label)
		vert_labels_dict[label] = vl


setup_table()


def update_mode(mode):
	"""Based on selected mode, edit behaviour of plot"""
	if mode == "Rotate":
		ax.mouse_init()
		fig.canvas.mpl_disconnect(dog.connect_id)
	elif mode == "Select":
		ax.disable_mouse_rotation()
		dog.connect_id = fig.canvas.mpl_connect('button_press_event', dog.on_click)


def export_selection(event):
	"""Based on current selected nodes, exports the data as numpy file."""

	out = []
	n_kp = 0
	for kp in keypoint_list:
		vert_idx = vert_labels_dict[kp].val
		if vert_idx is None:
			p = np.zeros((n_frames, 5))
		else:
			p = dog.data[:, vert_idx]
			n_kp += 1

		out.append(p)

	out = np.swapaxes(out, 0, 1)  # swap so of shape (frames, joints, 5)
	np.save(os.path.join(r"E:\IIB Project Data\Dog 3D models\SMAL 3D prior data", title), out)
	print("DATA EXPORTED. KEYPOINTS SELECTED = ", n_kp)


export = Button(ax_export, "Export data")
export.on_clicked(export_selection)

plt.tight_layout()
plt.show()
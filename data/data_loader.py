"""Classes for loading various data types within the project, including

- Force plate XML data
- C3D data (to be moved here from another script)
- PLY data (to be moved here from another script)"""

import c3d

from vis.utils import *
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import tqdm
from scipy.ndimage.measurements import center_of_mass, label
from scipy import signal
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.widgets import Slider
import plyfile
import os, sys, csv, torch

path_join = os.path.join  # simplify function name

from smal_fitter.smbld_model.smbld_mesh import SMBLDMesh


class DataSources:
	"""Gives the locations of sources for various forms of data within the project.

	Absolute references are used for large data files only accessed on my HP computer
	Relative references are for smaller data types that need to be accessed on multiple computers"""

	ply_collections = r"E:\IIB Project Data\produced data\ply collections"  # Absolute reference
	forceplate_data = r"E:\IIB Project Data\produced data\forceplate_data"  # Absolute reference
	dynamics_data = r"E:\IIB Project Data\produced data\dynamics data"  # Absolute reference
	smal_outputs = r"E:\IIB Project Data\produced data\smal_outputs"  # Absolute reference

	c3d_data = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Pipeline\c3d_data"  # Absolute reference

	datasets = r"E:\IIB Project Data\Training data sets"  # Absolute reference


def cluster_recognition(data, merge=20, cap=False):
	"""Takes an mxn array, returns B bounding boxes, in form ((x0, y0), (x1,y1)).
	Merges any bboxs that have centres within <merge> of each other"""
	labelled, num_features = label(data)

	bboxs = []
	for f in range(1, num_features + 1):
		a = np.where(labelled == f)
		y0, y1, x0, x1 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
		bboxs.append(((x0, y0), (x1, y1)))

	## IF ANY BBOXES OVERLAP, COMBINE THEM
	for bbox in bboxs:
		(x0, y0), (x1, y1) = bbox
		gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
		for other in [b for b in bboxs if b is not bbox]:
			(ox0, oy0), (ox1, oy1) = other
			ogx, ogy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
			# print(bbox, other, (abs(ogx - gx) + abs(ogy - gy)))
			if (abs(ogx - gx) + abs(ogy - gy)) <= merge:
				bboxs.remove(bbox)
				bboxs.remove(other)
				bboxs.append(((min(x0, ox0), min(y0, oy0)), (max(x1, oy1), max(y1, oy1))))

	# if cap:
	#     fig, (ax1, ax2) = plt.subplots(ncols=2)
	#     ax1.imshow(data)
	#     ax2.imshow(labelled)
	#     print(bboxs)
	#     plt.show()

	return bboxs


paw_colours = {
	"front right": "#99ffcc",  # green
	"front left": "#ffff99",  # red
	"rear left": "#66ffff",  # blue
	"rear right": "#ff00ff",  # fuschia
}


def centre_of_pressure(data):
	"""Given an (MxN) array, return the (x, y) coord of the CoP"""

	y, x = center_of_mass(data)

	return x, y


class Paw(Rectangle):
	"""The paw for a particular frame. Has info on - name of paw, net reaction force, and bbox at that instant"""

	def __init__(self, bboxs, forces, frame_start, frame_stop):
		"""Bbox in format ((x0, y0), (x1, y1)) first seen
		force = force (first seen)
		f0 = frame first seen"""
		(x0, y0), (x1, y1) = bboxs[0]
		self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

		self.bboxs = bboxs

		self.mean_width = self.x0 - self.x1
		self.mean_height = self.y0 - self.y1

		self.x_mean, self.y_mean = x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0)

		self.forces = forces
		self.frame_start, self.frame_stop = frame_start, frame_stop

	def add_frame(self, force, bbox):
		self.forces.append(force)
		self.bboxs.append(bbox)

	def calc_means(self):
		all_means = np.array([[0.5 * (x0 + x1), 0.5 * (y0 + y1)] for (x0, y0), (x1, y1) in self.bboxs])
		self.x_mean, self.y_mean = all_means.mean(axis=0)

	def bbox_is_paw(self, bbox, disp=0):
		"""Returns True if bbox is part of current paw. Criteria is:
		CoM, x position is within 5 squares of mean
		CoM, y position, displaced by <disp>, is within 5 squares of initial
		"""

		(x0, y0), (x1, y1) = bbox

		within_n = lambda a, b, n=20: abs(a - b) <= n

		x_mean, y_mean = x0 + .5 * (x1 - x0), y0 + .5 * (y1 - y0)

		return within_n(x_mean, self.x0) and within_n(y_mean - disp, self.y_mean)

	def identify(self, side, end):
		self.end = end
		self.side = side
		self.name = f"{end} {side}"
		self.colour = paw_colours[self.name]

		super().__init__((self.x0, self.y0), (1 + self.x1 - self.x0), (1 + self.y1 - self.y0), fill=None,
						 ec=self.colour, lw=2)

	def add_text(self, ax):
		self.text = ax.text((self.x0 + self.x1) / 2, self.y1, va="top", ha="center", s=self.name, color="white")

	def clear(self):
		self.remove()
		if hasattr(self, "text"):   self.text.remove()  # remove text as well if it has been created

	def set_frame(self, f, ax):
		n_frames = self.frame_stop - self.frame_start

		if f < self.frame_start:
			pass
		elif f == self.frame_start:
			ax.add_patch(self)
		elif f < self.frame_start + n_frames:
			(x0, y0), (x1, y1) = self.bboxs[f - self.frame_start]
			self.set_xy((x0, y0))
			self.set_width(1 + x1 - x0), self.set_height(1 + y1 - y0)
		elif f == self.frame_start + n_frames + 1:
			self.remove()


class ForcePlateData():
	def __init__(self, src="", freq=100, n_frames=None, play_every=1, playback_speed=1.0,
				 vis=False):

		self.fps = playback_speed * freq / play_every

		if src == "":
			src = askopenfilename()

		else:
			src += ".xml"
		self.src = src

		file_loc = os.path.join(DataSources.forceplate_data, src)
		tree = ET.parse(file_loc)

		for child in tree.getroot():
			if "movements" in child.tag:
				movements = child

		for child in movements:
			id = 0

		movement = list(movements)[0]
		*_, clips = list(movement)

		forceplate_data, velocity_data = list(clips)

		velocities = np.array([float(i.text) for i in list(velocity_data)[-1]])
		vdt = velocities * 1 / freq  # v * dt

		# Extract forceplate data
		_, id, begin, freq, count, units, cell_count, cell_size, data = list(forceplate_data)

		n_x, n_y = int(list(cell_count)[0].text), int(list(cell_count)[1].text)
		self.x_size, self.y_size = float(list(cell_size)[0].text), float(list(cell_size)[0].text)

		self.pressure_data = []

		for quant in data:
			cell_begin, cell_count, cells = list(quant)
			cell_data = cells.text

			split_data = (cell_data.replace("\t", "").replace("\n", " ").split(" "))[
						 1:-1]  # ignore first and last point as these are blanks
			split_data = list(map(float, split_data))

			x_start, y_start = int(list(cell_begin)[0].text), int(list(cell_begin)[1].text)
			x_count, y_count = int(list(cell_count)[0].text), int(list(cell_count)[1].text)

			data = np.reshape(split_data, (y_count, x_count))

			## pad data correctly (note: y _start measured from bottom)
			data = np.pad(data, [(n_y - y_count - y_start, y_start), (x_start, n_x - x_count - x_start)])

			self.pressure_data.append(data)

		# Only select data from desired range of number of frames
		if n_frames is None: n_frames = len(self.pressure_data)
		self.n_frames = n_frames

		self.pressure_data = self.pressure_data[:n_frames:play_every]
		self.pressure_data = np.array(self.pressure_data)

		# For now, assume x and y cell size is in *mm*, so divide this result by 100 to convert from mm to cm
		self.force_data = self.pressure_data * self.x_size * self.y_size / 100

		# Cluster the force data to get the reaction in each foot as a function of time
		self.bounding_boxes = bounding_boxes = []

		# Identify the forces for each paw in each frame.
		# Current method:
		# Identify each paw using a cluster recognition function that identifies bounding boxes of connected data
		# Identify which paw by:
		# Front/back based on whether ahead/behind of the overall average y
		# Left/right based on whether ahead/behind of the local average x (either front x or back x).
		# This will need to be converted into some kind of running average to work out which foot
		# This process may need refining for longer clips where the dog moves relative to the treadmill more - TODO review

		# First, identify center point as the mean (x, y) coordinate of all non zero data
		def centre_of_pressure(data):
			"""Given a series of frames, each with an array of numerical data corresponding to pressure plate data, return the overall average x and y positions"""
			all_y_centres, all_x_centres = zip(*[center_of_mass(frame) for frame in data])

			return np.nanmean(all_x_centres), np.nanmean(all_y_centres)

		_, self.global_y_centre = centre_of_pressure(self.pressure_data)  # Center of pressure for all data

		y_centre_index = int(round(self.global_y_centre))  # convert to int for indexing

		# Apply clusters to all frames
		self.paws = paws = []

		clusters, n_features = label(self.force_data > 0)  # Identify individual paws

		minmax = lambda a: (np.min(a), np.max(a))
		for n_feature in range(1, n_features + 1):
			a = np.where(clusters == n_feature)

			f_start, f_stop = minmax(a[0])

			bboxs = []
			forces = []

			for f in range(f_start, f_stop + 1):
				b = np.where(clusters[f] == n_feature)
				y0, y1 = minmax(b[0])
				x0, x1 = minmax(b[1])

				bboxs.append(((x0, y0), (x1, y1)))
				forces.append(self.force_data[f, y0: y1 + 1, x0: x1 + 1].sum())

			paws.append(Paw(bboxs, forces, f_start, f_stop))

		overlap = lambda a1, a2, b1, b2, buff=10: (a1 - buff < b1 < a2 + buff) or (a1 - buff < b2 < a2 + buff)
		for paw in paws:
			# get list of all paws that are present in same timeframe
			concurrent_paws = [other for other in paws if
							   overlap(other.frame_start, other.frame_start + len(other.bboxs),
									   paw.frame_start, paw.frame_start + len(paw.bboxs))]

			mean_y = np.mean([p.y_mean for p in concurrent_paws])
			end = ["front", "rear"][paw.y_mean >= mean_y]

			# only get mean_x from this end
			if end == "front":
				mean_x = np.mean([p.x_mean for p in concurrent_paws if p.y_mean <= mean_y])
			else:
				mean_x = np.mean([p.x_mean for p in concurrent_paws if p.y_mean >= mean_y])
			side = ["left", "right"][paw.x_mean >= mean_x]

			paw.identify(side, end)

		## compute grfs
		assignments = {"front left": 0, "front right": 1, "rear left": 2, "rear right": 3}
		self.grfs = grfs = np.zeros((self.n_frames, 4))

		for paw in paws:
			idx = assignments[paw.name]
			f0, n_f = paw.frame_start, len(paw.forces)
			grfs[f0:f0 + n_f, idx] += paw.forces

		self.save_data(grfs, title=self.src)

		if vis: self.plot_pressure(n_frames=n_frames)

	def plot_pressure(self, n_frames=None, title="output"):

		fig, (ax_p, ax_grf) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 3]})

		if n_frames is None:
			data = self.pressure_data
			n_frames = self.n_frames
		else:
			data = self.pressure_data[:n_frames]

		progress = tqdm.tqdm(total=len(data) - 1)

		plot = ax_p.imshow(data[0], cmap='hot', interpolation='nearest')

		# calc all grfs
		assignments = {"front left": 0, "front right": 1, "rear left": 2, "rear right": 3}
		for f in range(4):
			name = {v: k for k, v in assignments.items()}[f]
			ax_grf.plot(self.grfs[:n_frames, f], label=name,
						color=paw_colours[name])
		ax_grf.legend()
		vline = ax_grf.axvline(0, ls="--")

		for paw in self.paws:
			paw.set_frame(0, ax_p)

		def anim(i):
			ax_p.set_title(f"Frame = {i}")
			vline.set_data([i, i], [0, 1])
			# Currently, i=0 runs twice for some weird reason. For now, just skip frame 1.
			if i == 0: return None

			plot.set_data(data[i])  # Set pressure data

			# draw bounding boxes
			# [paw.clear() for paw in self.paws[i-1]] # remove existing paws from fig
			for paw in self.paws:
				paw.set_frame(i, ax_p)
			# paw.add_text(ax)

			progress.update(1)

		# ax_grf.set_xlim(0, 1000) # so everything is visible

		# Add legend with colour of each paw
		for name, colour in paw_colours.items():
			ax_p.plot([], [], color=colour, label=name)
		ax_p.legend(loc="lower center")

		save_animation(anim, fig, title=title, frames=len(data), fps=self.fps,
					   dir=r"C:\Users\Ollie\Videos\iib_project\forceplate_vis")

	def save_data(self, data, title=""):
		"""Takes the reaction data for each paw at each frame, and saves the whole set of data as a numpy array
		of size (n_frames, 4) where each frame has an array of data:
		 [front left force, front right force, rear left force, rear right force]."""

		title = title.replace(r"\\", " -- ").replace(r"/", " -- ").replace("\\", " -- ")[
				:-4]  # Title tells both the set and the name of the file. Also crop out the extension .xml with the [:-4] command.
		np.save(os.path.join(DataSources.forceplate_data, "processed", title),
				data)  # save as title without .xml, and no spaces on the end


def get_delay_between(title, start, end):
	"""Gets the delay between any 3 clip types:
	- SMAL (video) - Mocap
	- Zebris - mocap
	- SMAL (video) - Zebris.

	T = 0 is SMAL (Video) start"""

	def clean(x):
		for s in " /-":
			x = x.replace(s, "")
		return x

	t_delay = None
	with open(os.path.join(DataSources.forceplate_data, "processed", "timings.csv"), "r") as infile:
		reader = csv.reader(infile)
		next(reader)
		for data_title, zebris_start_time, mocap_start_time in reader:
			# some discrepancy with symbols/spaces, so remove spaces, /, -:
			if clean(data_title) == clean(title):
				mocap_start_time, zebris_start_time = float(mocap_start_time), float(zebris_start_time)
				break

		else:  # no break
			print(f"Delay for '{title}' not found.")
			return 0

	if start == "smal":
		t_start = 0
	elif start == "zebris":
		t_start = zebris_start_time
	else:
		raise ValueError("Start must be one of 'smal', 'zebris'")

	if end == "mocap":
		t_end = mocap_start_time
	elif end == "zebris":
		t_end = zebris_start_time
	else:
		raise ValueError("End must be one of 'mocap', 'zebris'")

	return t_end - t_start


def load_force_plate_data(title, is_mocap=True):
	"""Loads processed forceplate data, as well as the timings required for the start time of zebris and mocap relative to the video clip."""

	if is_mocap:
		t_delay = get_delay_between(title, "zebris", "mocap")

	else:
		t_delay = - get_delay_between(title, "smal", "zebris")

	return np.load(os.path.join(DataSources.forceplate_data, "processed", title + ".npy")), t_delay


# f = ForcePlateData(src="example_data_finn", n_frames = None, play_every=1, playback_speed=0.5, vis=True)
# ForcePlateData(src=r"set_1 - 14-10-19\Ally 3kph run 4 data")


# MESH PLY FILE USAGE
def extract_ply(src):
	"""Return vertices, faces"""
	with open(src, "rb") as infile:
		plydata = plyfile.PlyData.read(infile)

	vertices = plydata.elements[0].data

	clean_verts = []
	# Problem here, numpy not understanding the above datatype, and creating each (x,y,z) coordinate as a numpy.void object. The below fixes it for now but is quite slow. Review?
	for vert in vertices:
		x, y, z = vert
		clean_verts.append([x, y, z])

	faces = plydata["face"].data

	return np.array(clean_verts), faces


class MeshData:
	"""Given a folder source, loads all the relevant data for retrieval"""

	def __init__(self, dir, n_frames=None, freq=60):

		full_dir = os.path.join(DataSources.ply_collections, dir)  # The full directory for the file

		files = sorted([f for f in os.listdir(full_dir) if ".ply" in f],
					   key=lambda x: int(x.split("_")[1].split(".")[0]))  # Sort .ply files by number

		if len(
				files) == 0:  # If files have not been extracted, go through each sub folder to find optimisation for that stage
			files = [f"frame_{n + 1}/st10_ep0.ply" for n in range(len(os.listdir(full_dir)))]

		if n_frames is not None: files = files[:n_frames]  # only load the first n frames if chosen

		self.n_frames = len(files) if n_frames != 1 else 1
		self.freq = freq

		self.face_data, self.vertex_data = [], []

		progress = tqdm.tqdm(total=len(files))
		progress.write("Loading Mesh Files")

		for file in files:
			src = os.path.join(full_dir, file)
			frame_vert_data, faces = extract_ply(src)
			frame_clean_face_data = [list(face[0]) for face in faces]  # clean up the formatting of the face data

			self.face_data.append(frame_clean_face_data)
			self.vertex_data.append(frame_vert_data)

			progress.update()

		self.vertex_data = np.array(self.vertex_data)


class JointData():
	"""Kinematic data for a series of joints."""

	def __init__(self, ax, data, freq=60, marker_labels=[], norm=False):

		self.ax = ax
		self.freq = freq
		self.n_frames, self.n_markers = len(data), len(data[0])
		self.clip_length = self.n_frames / self.freq

		self.marker_labels = marker_labels

		self.norm = norm
		if norm:    data = np.array(data) / np.amax(data)  # crude normalisation for now

		# Convert data to Marker objects
		self.markers = []

		for m in range(self.n_markers):
			label = "" if marker_labels == [] else self.marker_labels[m]
			self.markers.append(Marker(ax, data=None, label=label, index=m))

		for i, frame in enumerate(data):
			for n, point in enumerate(frame):
				x, y, z, *_ = point  # Points gives x, y, z, error, n_cameras
				self.markers[n].add_point(x, y, z)

		for marker in self.markers:
			marker.bake_data(marker.data)

		self.set_all_data()

		self.get_bounds()

	def set_all_data(self):
		self.all_data = np.swapaxes(np.array([marker.data for marker in self.markers]), 0,
									1)  # All data as an array of size (n_frames, n_markers, 3)

	def get_bounds(self):
		self.x_bounds = minmax(self.all_data[:, :, 0])
		self.y_bounds = minmax(self.all_data[:, :, 1])
		self.z_bounds = minmax(self.all_data[:, :, 2])

	def __getitem__(self, item):
		"""Loads a specific marker by name"""
		for marker in self.markers:
			if marker.label == item:
				return marker

		print(f"Error in Joint Data: Name {item} not found in set {[m.label for m in self.markers]}")

	def get_marker_indices(self, *names):
		return [self[name].index for name in names]

	def get_marker_data_by_names(self, *names):
		output = []
		for name in names:
			marker = self[name]
			output.append(marker.data)  # add data, but zip as original data in form [(x_data), (y_data), (z_data)]

		return np.swapaxes(output, 0, 1)  # swap axes 0 and 1 so that shape is (n_frames, n_markers, 3)

	def __next__(self):
		"""gives all x,y,z data for next frame"""
		[next(marker) for marker in self.markers]

	def update_frame(self, n_frame=0):

		[marker.update_frame(n_frame) for marker in self.markers]

	def __len__(self):
		return self.n_frames

	def create_averaged_marker(self, new_marker_name, *existing_marker_names):
		"""Takes marker names, and combines the data from each into a new marker that averages their data."""

		new_marker_data = np.average(self.get_marker_data_by_names(*existing_marker_names), 1)
		new_marker = Marker(self.ax, data=new_marker_data, label=new_marker_name, index=len(self.markers))
		self.markers.append(new_marker)
		self.marker_labels.append(new_marker_name)
		self.set_all_data()  # redefine the full data

	def resample_at(self, target_freq=100):
		"""Resamples kinematic data to different frequency. Updates entire mocap object accordingly"""

		target_frames = int(self.n_frames * target_freq / self.freq)

		new_data = signal.resample(self.all_data, target_frames)

		JointData.__init__(self, self.ax, new_data, target_freq, self.marker_labels, self.norm)

	def generate_skeleton_mapping(self):
		"""Generates:
		dictionary of bone_name: [start_joint_index, end_joint_index]
		list of joints in body
		list of joints in paws (no torque joints)"""

		# First dictionaries map bones to joints by joint name
		all_bones = {}
		for end in ["rear", "front"]:
			for side in ["left", "right"]:
				s = f"{side} {end} "
				all_bones = {**all_bones,
							 **{s + "1": [s + "paw", s + "ankle"],
								s + "2": [s + "ankle", s + "knee"],
								s + "3": [s + "knee", s + "shoulder"],
								s + "4": [s + "shoulder", s + "upper"]
								}}

		body_joints = [self.get_marker_indices(f"left {end} upper", f"right {end} upper") for end in
					   ["rear", "front"]]

		leg_spring_joints = [self.get_marker_indices(f"{side} {end} shoulder")[0] for end in
							 ["front", "rear"] for side in ["left", "right"]]

		# Now map these to indices within the motion capture data
		flatten = lambda arr: [item for sublist in arr for item in sublist]

		all_bones_indexed = {}  # dictionary to return, with indices instead of joint names
		for bone, (jointA, jointB) in all_bones.items():
			jointA_idx, jointB_idx = self.get_marker_indices(jointA, jointB)
			all_bones_indexed[bone] = [jointA_idx, jointB_idx]

		return all_bones_indexed, body_joints, self.get_marker_indices(
			"left front paw", "right front paw", "left rear paw", "right rear paw"), leg_spring_joints


class C3DData(JointData):
	"""Interprets C3D format. See https://medium.com/@yvanscher/explanation-of-the-c3d-file-format-c8e065300510 for explanation of format"""

	def __init__(self, src=None, ax=None, alt_views=None, interpolate=True, norm=True, crop=None,
				 fix_rotations=False):
		"""crop: only take first x seconds of clip"""

		if src is None:
			Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
			src = askopenfilename(initialdir=DataSources.c3d_data)  # Load correct file
			self.name = src.split("\\")[-1][:-4]

		else:
			self.name = src.split("\\")[-1][:-4]

		print("LOADING", src)
		if not os.path.isfile(src):
			src = os.path.join(DataSources.c3d_data, src)  # If relative reference, append to directory

		with open(src, 'rb') as handle:
			reader = c3d.Reader(handle)
			self._reader = reader

			# Extract meta data from file
			n_markers = reader.header.point_count  # Work out the total number of markers
			n_frames = reader.header.last_frame - reader.header.first_frame + 1

			freq = reader.header.frame_rate

			labels = str(reader.groups["POINT"].params["LABELS"].bytes)[
					 2:]  # Extract text of all labels, separated by many spaces
			# Convert separating spaces into commas
			for spaces in [25 - i for i in range(15)]:
				labels = labels.replace(" " * spaces, ", ")
			marker_labels = labels.split(", ")[:n_markers]

			data = []
			for i, points, analog in reader.read_frames():
				frame_data = []
				# Analog gives force plate data (I think)

				for n, point in enumerate(points):
					x, y, z, *_ = point  # Points gives x, y, z, error, n_cameras
					frame_data.append([x, y, z])

				data.append(frame_data)

			def interpolate_zero_vals(data):
				"""Gievn a time series of (x,y,z) points, identifies any ranges of (0,0,0) values, and replaces these with interpolated values either side of the zero range"""
				is_zero = lambda arr: np.linalg.norm(arr) == 0
				for n, pos in enumerate(data):
					if n == 0: continue
					if n + 2 >= len(data): continue

					if is_zero(pos):
						# if 0 found, search ahead to find the range for which this goes on, and then begin interpolation

						n_start = n
						n_finish = n + 1

						while n_finish + 2 <= len(data) and is_zero(data[n_finish]):
							n_finish += 1

						# Note: n_start = index of first zero value
						# n_finish = index of first NON ZERO value

						# now the start and finish of the zero range is found
						start = Vector(*data[n_start - 1])
						finish = Vector(*data[n_finish])

						# Perform interpolation
						for i in range(n_finish - n_start):
							data[n_start + i] = (start + (finish - start) * (i + 1) / (n_finish - n_start + 1))

						data = interpolate_zero_vals(data)  # Run recursively to restart search for zeros
						break  # end this search within this run of the function

				return data

			# Split up into individual marker time series, interpolate each individually, and then combine together for processing
			if interpolate: data = list(zip(*[interpolate_zero_vals(list(marker_data)) for marker_data in zip(*data)]))

			# REMOVE ROTATIONAL MODES OF PAWS. IDENTIFY SIGNIFICANT TROUGHS, SET TO (0, 0, 0)
			data = np.array(data)

			if fix_rotations:
				for idx in [n for n, i in enumerate(marker_labels) if "paw" in i]:
					vert_data = data[:, idx, 2]
					troughs, properties = signal.find_peaks(-vert_data, prominence=5, rel_height=.1, width=(None, 5))
					# print(signal.peak_widths(-vert_data, troughs, rel_height=0.1)[0])

					window = 10  # n frames to look before for start of trough
					for t in troughs:
						# go backwards to find where paw starts rotating (change in sign of grad)
						prev_cusps = np.where(np.diff(vert_data[t - window:t]) > 0)[0]  # cusps before trough
						if len(prev_cusps) == 0: continue  # skip ones where previous cannot be found
						rot_start = t - window + 1 + prev_cusps[-1]  # idx of start of trough

						next_higher = np.where(vert_data[rot_start + 1:] >= vert_data[rot_start])[
							0]  # next time vert is at a higher value
						if len(next_higher) == 0: continue
						rot_end = rot_start + 1 + next_higher[0]

						# replace trough with linear interpolation
						x = np.arange(rot_end - rot_start)
						m = (vert_data[rot_end] - vert_data[rot_start]) / (rot_end - rot_start)

						vert_data[rot_start: rot_end] = vert_data[rot_start] + m * x

			# if idx == 23:
			# plt.plot(vert_data)
			# plt.scatter(troughs, vert_data[troughs])
			# plt.show()

			if crop:
				data = data[:int(freq * crop)]

			super().__init__(ax, data, freq=freq, marker_labels=marker_labels, norm=norm)

			# self.n_frames = i

			self.get_bounds()

			### FIXES FOR MOCAP DATA THAT IS MISSING MARKERS:
			if "set_1" in src:
				self.create_averaged_marker("front top", "left front upper",
											"right front upper")  # form new joint at the front top of the system
				self.create_averaged_marker("right front ankle", "right front knee",
											"right front paw")  # right front ankle missing from data, for now just add as avg

			if "set_2" in src:
				self.create_averaged_marker("front top", "left front upper",
											"right front upper")  # form new joint at the front top of the system
				self.create_averaged_marker("rear top", "right rear upper",
											"left rear upper")  # form new joint at the front top of the system
				self.create_averaged_marker("left rear ankle", "left rear knee",
											"left rear paw")  # right front ankle missing from data, for now just add as avg

	def plot_data(self):
		"""Creates interactive figure capable of cycling through the frames of the C3D animation"""

		fig = plt.figure()
		ax = fig.add_subplot(211, projection="3d")
		ax2 = fig.add_subplot(212)

		plot = ax.scatter(*zip(*self.all_data[0]))

		texts = []
		for i in range(self.n_markers):
			x, y, z = self.all_data[0, i]
			texts.append(ax.text(x=x, y=y, z=z, s=i))

		def plot_frame(i):
			plot._offsets3d = art3d.juggle_axes(*zip(*self.all_data[i]), "z")

		slider = Slider(ax2, "Frame", 0, self.n_frames - 1, valinit=0, valstep=1, valfmt='%0.0f')
		slider.on_changed(lambda v: plot_frame(int(v)))

		plt.show()


def extract_npz(smal_npz):
	"""Extract .npz file, return as dict"""
	data = np.load(smal_npz)
	keys = list(data.keys())

	out = {}
	for k in keys:
		out[k] = data[k]

	return out


SMAL_LABELLED_JOINTS = [
	10, 9, 8,  # left front (0, 1, 2)
	20, 19, 18,  # left rear (3, 4, 5)
	14, 13, 12,  # right front (6, 7, 8)
	24, 23, 22,  # right rear (9, 10, 11)
	25, 31,  # tail start -> end (12, 13)
	34, 33,  # right ear, left ear (14, 15)
	7, 11,  # front left shoulder, front right shoulder
	17, 21,  # rear left shoulder, rear right shoulder
	6, 6,  # Front L/R Upper
	0, 0,  # rear L/R Upper
]


# FLIP TEST
# SMAL_LABELLED_JOINTS = [
#         14, 13, 12,  # left front (0, 1, 2)
#         24, 23, 22,  # left rear (3, 4, 5)
#         10, 9, 8,  # right front (6, 7, 8)
#         20, 19, 18, # right rear (9, 10, 11)
#         25, 31,  # tail start -> end (12, 13)
#         34, 33,  # right ear, left ear (14, 15)
#         7, 11, # front left shoulder, front right shoulder
#         17, 21, # rear left shoulder, rear right shoulder
#         6, 6, # Front L/R Upper
#         0, 0, # rear L/R Upper
#     ]


def get_smal_data(smal_npz, smooth=False):
	data = extract_npz(smal_npz)
	# joint_data = data['joints3d']
	# n_frames, n_joints, _ = joint_data.shape

	pose_data = data['pose']

	n_frames = len(pose_data)

	## LOAD POSE PRIOR, CORRECT BY IT
	import pickle as pkl
	prior_src = r"E:\IIB Project Data\Dog 3D models\Pose Prior\unity_pose_prior_with_cov_35parts.pkl"
	pose_correction = 3e-5
	with open(prior_src, "rb") as f:
		res = pkl.load(f, encoding='latin1')

		cov = res['cov']

		precs = res['pic'].r
		mean = res['mean_pose']

		# print(precs.shape, mean.shape)

		x = pose_data
		res = np.tensordot((x - mean), precs, axes=([1], [0]))
		pose_data -= pose_correction * res
	# raise ValueError

	diags = np.diag(precs)

	means = mean.reshape(35, 3)
	vars = np.diag(cov)  # .reshape(35, 3)

	pose = torch.from_numpy(pose_data.reshape(n_frames, 35, 3))
	smbld = SMBLDMesh(opts=None, n_batch=n_frames)

	if smooth:
		pose = torch.from_numpy(signal.savgol_filter(pose, window_length=31, polyorder=5, axis=0))  # smooth pose
	else:
		pose = pose  # smooth pose

	mean_betas = torch.from_numpy(data['betas'].mean(axis=0))
	for n, b in enumerate(mean_betas):
		smbld.multi_betas[n] = b

	# smalx.global_rot[:] = pose[:, 0]
	smbld.joint_rot[:] = pose[:, 1:]

	smbld.joint_rot[:, 0] *= 0  # get rid of first motion - LEFT - RIGHT.
	v, f, J = smbld.get_verts(return_joints=True)

	v = v.detach().numpy()
	J = J.detach().numpy()[:, SMAL_LABELLED_JOINTS]

	return {"joints": J, "verts": v, "faces": f}


class SMALData(JointData):
	labels = ["left front paw", "left front ankle", "left front knee",
			  "left rear paw", "left rear ankle", "left rear knee",
			  "right front paw", "right front ankle", "right front knee",
			  "right rear paw", "right rear ankle", "right rear knee",
			  "tail start", "tail end",
			  "right ear", "left ear",
			  "left front shoulder", "right front shoulder",
			  "left rear shoulder", "right rear shoulder",
			  "left front upper", "right front upper",
			  "left rear upper", "right rear upper"]  # labels corresponding to SMAL_LABELLED_JOINTS

	paw_smal_vertices = {
		# "left front paw" : 1258,#1258,
		"left rear paw": 848,
		"right rear paw": 2812,
		# "right front paw": 2588,
	}  # for paws, get data from vertices for now instead.

	def __init__(self, src, freq=30, norm=False, crop=None, smooth=False):
		"""Crop = time in seconds to crop to"""

		full_src = path_join(DataSources.smal_outputs, src + ".npz")
		loaded_smal_data = get_smal_data(full_src, smooth=smooth)
		loaded_joint_data = loaded_smal_data["joints"]

		## replace joint data for paws with mesh vertex data - temp change
		for paw, verts_idx in self.paw_smal_vertices.items():
			joint_idx = self.labels.index(paw)
			loaded_joint_data[:, joint_idx] = loaded_smal_data["verts"][:, verts_idx]

		if crop is not None:
			loaded_joint_data = loaded_joint_data[:crop * freq]

		super().__init__(None, loaded_joint_data, freq=freq, marker_labels=self.labels, norm=norm)


class Marker():
	plot_kw = {}  # dict(mfc="red", mec="red")

	def __init__(self, ax, data, label="", dims="xyz", alt_views=None, index=0):

		if ax is not None:
			self.plot, = ax.plot([], [], [], "o", label=index, **self.plot_kw)

		self.alt_views = {}
		self.mesh_vertex = None  # used for optimisation

		if alt_views is not None:
			for vars, axis in alt_views.items():
				p, = axis.plot([], [], "o", **self.plot_kw)

				self.alt_views[vars] = p

		self.label = label
		self.index = index

		if data is not None:  # If data given, bake instantly
			self.bake_data(data)
		else:
			self.all_data = []

	def bake_data(self, data):
		self.all_data = np.array(data)

		# Set bounds
		self.xmin, self.xmax = minmax(self.all_data[:, 0])
		self.ymin, self.ymax = minmax(self.all_data[:, 1])
		self.zmin, self.zmax = minmax(self.all_data[:, 2])

	def add_point(self, x, y, z):
		self.all_data.append([x, y, z])

	@property
	def position(self):
		return self.x, self.y, self.z

	@property
	def data(self):
		return self.all_data

	def set_data(self, data):
		"""Takes data in the usual shape (n_frames, 3), and applies it to marker format"""

		self.x_data, self.y_data, self.z_data = zip(*data)

	def hide(self):
		"""For now, just shrink marker size to zero"""
		for line in [self.plot] + list(self.alt_views.values()):
			line.set_markersize(0)

	def update_frame(self, n_frame=0):
		# update x,y,z
		self.x, self.y, self.z = self.all_data[n_frame]

		# Trick to not plot any markers with zero errors for that frame
		if any(v == 0 for v in (self.x, self.y, self.z)):
			self.x, self.y, self.z = [], [], []

		self.plot.set_xdata(self.x)
		self.plot.set_ydata(self.y)
		self.plot.set_3d_properties(self.z)

		for (var1, var2), axis in self.alt_views.items():
			axis.set_xdata(getattr(self, var1))
			axis.set_ydata(getattr(self, var2))

# mocap = C3DData(r"E:\IIB Project Data\Dog 3D models\SMAL 3D prior data\test.c3d", #Amstaff01_SV_RM_MX.c3d")
#                 interpolate=False)
# mocap.plot_data()

# for d in ["border_collie", "gus", "lab"]:
#     f = ForcePlateData(f"set_3\\{d}")
#     f.plot_pressure(title=d, n_frames = 500)

# f = ForcePlateData(r"set_3\gus")
# f.plot_pressure(title="gus", n_frames = 500)

# f = ForcePlateData(r"set_2\6kph run1")
# f.plot_pressure(title="set2-6r1", n_frames = 500)

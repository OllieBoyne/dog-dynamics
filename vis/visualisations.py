"""Scripts for producing various animations and graphics for presenting results of pipeline.

*NOTE: OUTDATED* - This was made relatively early on in the project, so some of the functions may need to be modified to work with the
other scripts."""

from dynamics.dynamics import *
from data.data_loader import *

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from tqdm import tqdm

# load in some other functions to replace modifications
m = Model()
bone_length_definitions = m.bone_length_definitions  # load default bone length definitions
load_paw_reactions = load_force_plate_data
paw_names = foot_joint_labels

# Currently an issue with the arrow drawing:
# ValueError: Given lines do not intersect. Please verify that the angles are not equal or differ by 180 degrees.
# Not sure what it means

def plot_mocap_skeleton(data, bones, body):
	"""Given mocap data (frames, joints, 3), and bone objects, and body objects, plot and save animation"""
	loc = r"C:\Users\Ollie\Videos\iib_project\mo_cap_vis"

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	bone_plots = []
	for b in bones:
		s, e = b.start, b.end
		bone_plots.append(ax.plot(*np.swapaxes(data[100, [s, e]], 0, 1), label=b.name)[0])

	ax.set_xlabel("Length")
	ax.set_ylabel("Width")
	ax.set_zlabel("height")

	body_kin = np.mean(np.stack([data[:, body.start_joints], data[:, body.end_joints]], axis=1), axis=2)

	# body_kin = (self.joint_pos[:, body_joints[0]] + self.joint_pos[:, body_joints[1]]) / 2
	ax.plot(*np.swapaxes(body_kin[100], 0, 1))
	frames = len(data)
	p = tqdm(total=frames)

	def anim(i):
		ax.clear()

		for b in bones:
			s, e = b.start, b.end
			bone_plots.append(ax.plot(*np.swapaxes(data[i, [s, e]], 0, 1), label=b.name)[0])

		ax.plot(*np.swapaxes(body_kin[i], 0, 1))

		p.update()

	save_animation(anim, fig, frames, title="set 2 3r3 bones", dir=loc)


class ReactionPlotter:
	"""Object responsible for plotting force arrows corresponding to the ground reaction forces.
	Requires positional data of the reaction points, and the vertical reaction points at each frame.

	May be updated to allow for non vertical reactions"""

	def __init__(self, ax, reaction_data, position_data, time_delay=0, freq=100,
				 baseline_weight=250, arrow_scale=100, **kwargs):

		self.freq = freq
		frame_delay = int(freq * time_delay)  # Number of frames to start clip at

		# As both clips are the same frequency, clip reaction data and the end of position data
		self.reaction_data, self.position_data = reaction_data[
												 frame_delay: frame_delay + len(position_data)], position_data

		self.arrow_scale = arrow_scale  # scale to multiply arrows by to show in plot
		self.baseline_weight = baseline_weight  # basis for standard arrow size

		self.arrows = [Arrow3D([0, 0], [0, 0], [0, 0], **kwargs, zorder=11) for n in range(4)]
		[ax.add_artist(arrow) for arrow in self.arrows]  # add arrows to axis

		self.n_frames = min(reaction_data.shape[0], position_data.shape[0])

	# For now, have text underneath that displays rounded force
	# self.labels = [ax.text("",0,0,0) for arrow in self.arrows]

	def update_frame(self, n_frame=0):
		"""Update arrow positions for the given frame"""

		if n_frame + 1 > self.n_frames: return None

		forces, positions = self.reaction_data[n_frame], self.position_data[n_frame]

		for arrow, force, pos in zip(self.arrows, forces, positions):
			# if np.linalg.norm(pos) != 0: # For zero positions, assume error in marker and do not plot
			pass

	# force = max(force, 0.01) # prevent force from going to values that would cause issues
	# arrow.set_data(*pos, 0, 0, force * self.arrow_scale / self.baseline_weight) # move arrow

	# arrow.set_mutation_scale(force / self.baseline_weight) # set mutation scale

	# label.set_position(*pos)
	# label.set_text(round(force,0))


class MeshPlotter:
	def __init__(self, ax, vertex_data, face_data, freq=60, **trisurf_kwargs):
		self.ax = ax
		self.vertex_data = vertex_data
		self.face_data = face_data

		self.freq = freq

		self.trisurf_kwargs = trisurf_kwargs
		self.plot = ax.plot_trisurf([0, 1, 2], [0, 1, 1], [0, 0, 0], **trisurf_kwargs)

		vert_data = np.array(vertex_data)
		X, Y, Z = vert_data[:, :, 0], vert_data[:, :, 1], vert_data[:, :, 2]
		ax.set_xlim(np.amin(X), np.amax(X))
		ax.set_ylim(np.amin(Y), np.amax(Y))
		ax.set_zlim(np.amin(Z), np.amax(Z))

		self.n_frames = len(vertex_data)

	def update_frame(self, n_frame=0):
		if n_frame + 1 > self.n_frames: return None

		self.plot.remove()
		X, Y, Z = zip(*self.vertex_data[n_frame])

		tri = Triangulation(X, Y, triangles=self.face_data[n_frame])
		self.plot = self.ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color="gray", **self.trisurf_kwargs)

	def update_time(self, t=0):
		self.update_frame(n_frame=int(t * self.freq))

	def get_vertex_data_at_time(self, t=0):
		n_frame = int(t * self.freq)
		return self.vertex_data[n_frame]


class ModelPlotter(JointData):
	"""An object for plotting the dynamic model of the skeleton - a series of cylinders constrained by joints at either end"""

	def __init__(self, ax, data, body_joints, freq=60, norm=False, bone_dict={}):
		super().__init__(ax, data, freq=freq, norm=True)

		self.bone_dict = bone_dict
		self.bones = [b for b in bone_dict]  # ordered list

		self.cylinders = [Cylinder(ax, radius=0.05, start=Vector(0, 0, 0), end=Vector(0.1, 0.1, 0.1), name=bone) for
						  bone in self.bones]
		self.body = Cylinder(ax, radius=0.05, start=Vector(0, 0, 0), end=Vector(0.1, 0.1, 0.1))

		self.body_joints = body_joints

	def update_frame(self, n_frame=0):
		"""Update frame data for each cylinder"""
		for cylinder in self.cylinders:
			joint1_idx, joint2_idx = self.bone_dict[cylinder.name]

			start, end = Vector(*self.all_data[n_frame, joint1_idx]), Vector(*self.all_data[n_frame, joint2_idx])

			cylinder.update_data(
				start=start, end=end, radius=bone_length_definitions["normal"](start > end)["outer_radius"],
			)

			cylinder.radius = bone_length_definitions["normal"](cylinder.length)

		# Update body separately
		start, end = Vector(*np.mean(self.all_data[n_frame, self.body_joints[0]], axis=0)), Vector(
			*np.mean(self.all_data[n_frame, self.body_joints[1]], axis=0))

		self.body.update_data(
			start=start, end=end, radius=bone_length_definitions["body"](start > end)["outer_radius"],
		)


def kinematics_with_reaction_forces(raw_mocap_src="", raw_reactions_src="", reactions_time_delay=0,
									mesh_data=None,
									predicted_skeleton=[], predicted_reactions=[],
									):
	"""Takes various input data to show both the moving skeleton and the ground reaction forces. Can take:
	- Raw data: MoCap marker data, and ground reaction forces as functions of times
		- Raw reaction forces must be from the forceplate_data/processed folder. To make these, use the raw .xml as a src into the ForcePlateData class in data_loader.py, and it will save the reactions into processed/
		- reactions_time_delay must be manually inputted for each clip to synchronize the data with the motiona capture. Use the timestamp in the Zebras video to synchronize. For now, these are just written in onenote.
	- Mesh data: SMAL mesh kinematic data (in format of a MeshData object (see data_loader.py))
	- Processed data: Predicted skeleton and reaction data

	Will plot these on sequential axes horizontally for each one provided.

	Differing frame rates:
	- Forceplate @ 100 Hz
	- MoCap @ 60 Hz
	- SMAL @ 30 Hz


	This function needs reintegrating with the latest joint skeleton mapping
	"""

	# First identify the number of plots required, and which subplot corresponds to which data
	is_raw_mocap, is_raw_reactions = raw_mocap_src != "", raw_reactions_src != ""
	is_raw_data, is_mesh_data, is_predicted_data = is_raw_mocap, mesh_data != None, predicted_skeleton != [] or predicted_reactions != []

	n_subplots = int(is_raw_data) + int(is_mesh_data) + int(is_predicted_data)  # number of subplots required

	fig = plt.figure(figsize=(8, 4))
	axes = GridSpec(1, n_subplots)

	existing_axes = []
	if is_raw_data:
		ax_raw = fig.add_subplot(axes[0, 0], projection="3d")
		existing_axes.append(ax_raw)
	if is_mesh_data:
		ax_mesh = fig.add_subplot(axes[0, int(is_raw_data)],
								  projection="3d")  # Mesh is axis 0 if no raw data, 1 otherwise
		existing_axes.append(ax_mesh)
		mesh_plot = MeshPlotter(ax_mesh, mesh_data.vertex_data, mesh_data.face_data)

	if is_predicted_data:
		ax_predicted = fig.add_subplot(axes[0, int(is_raw_data) + int(is_mesh_data)],
									   projection="3d")  # Similar to above
		existing_axes.append(ax_predicted)

	# Set up 3D view
	view_kw = dict(elev=0, azim=-90)
	for ax in existing_axes:
		ax.view_init(**view_kw)
		ax.axis("off")
		ax.patch.set_edgecolor("black")
		ax.patch.set_linewidth('1')

	buffer = lambda low, high, val=0: (low - val, high + val)

	# Set up raw data
	if is_raw_data:
		mocap_data = C3DData(ax=ax_raw, src=raw_mocap_src, interpolate=True)  # Load MoCap data

		# Averaging required for set 2:
		mocap_data.create_averaged_marker("front top", "left front upper",
										  "right front upper")  # form new joint at the front top of the system
		mocap_data.create_averaged_marker("rear top", "right rear upper",
										  "left rear upper")  # form new joint at the front top of the system
		mocap_data.create_averaged_marker("left rear ankle", "left rear knee",
										  "left rear paw")  # right front ankle missing from data, for now just add as avg

		x_bounds, y_bounds, z_bounds = [buffer(*getattr(mocap_data, f"{dim}_bounds"), val=0.01) for dim in
										"xyz"]  # Get all dimensions with a buffer of 10
		# Set bounds and add axes arrows
		ax_raw.set_xlim(*x_bounds)
		ax_raw.set_ylim(*y_bounds)
		ax_raw.set_zlim(*z_bounds)

		# Will only plot raw reactions if mocap data also provided.
		if is_raw_reactions:
			raw_reaction_data, t_delay = load_force_plate_data(raw_reactions_src)
			reaction_positions = mocap_data.get_marker_data_by_names(*paw_names)

			raw_reaction_plotter = ReactionPlotter(ax_raw, reaction_data=raw_reaction_data,
												   position_data=reaction_positions, time_delay=t_delay,
												   mutation_scale=20, lw=3, arrowstyle="-|>", color="r",
												   baseline_weight=250, arrow_scale=(z_bounds[1] - z_bounds[0]) / 5)

	if is_predicted_data:
		predicted_skeleton_plotter = ModelPlotter(ax_predicted, mocap_data.all_data, freq=60,
												  bone_dict=BoneMappings().get_mocap_mapping(mocap_data)[0],
												  body_joints=BoneMappings().get_mocap_mapping(mocap_data)[
													  1])  # Review frequency here

		# Set bounds
		x_bounds, y_bounds, z_bounds = [buffer(*getattr(predicted_skeleton_plotter, f"{dim}_bounds"), val=0.01) for dim
										in
										"xyz"]  # Get all dimensions with a buffer of 10
		ax_predicted.set_xlim(*x_bounds)
		ax_predicted.set_ylim(*y_bounds)
		ax_predicted.set_zlim(*z_bounds)

		predicted_reactions_plotter = ReactionPlotter(ax_predicted, *predicted_reactions,
													  freq=predicted_skeleton_plotter.freq)

	n_frames = len(mocap_data)
	fps = 60
	playback_speed = 1
	progress = tqdm.tqdm(total=n_frames + 1)

	def time_to_frame(t, freq):
		"""Given a time and a frequency, returns the nearest integer frame to that time instance."""
		frame = round(t * freq)
		return int(frame)

	def anim(i):
		t = i / fps

		mocap_data.update_frame(time_to_frame(t, freq=mocap_data.freq))
		raw_reaction_plotter.update_frame(time_to_frame(t, freq=raw_reaction_plotter.freq))

		if is_mesh_data and i % 2: mesh_plot.update_frame(i)

		if is_predicted_data:
			predicted_skeleton_plotter.update_frame(time_to_frame(t, freq=predicted_skeleton_plotter.freq))
			predicted_reactions_plotter.update_frame(time_to_frame(t, freq=predicted_reactions_plotter.freq))

		progress.update(1)

	plt.tight_layout()
	save_animation(anim, fig, frames=100, dir=r"C:\Users\Ollie\Videos\iib_project\dynamics_overview",
				   fps=fps * playback_speed)


# mocap_data = C3DData(ax=None, src=r"set_1/3kph capture 4.c3d")

ally_set_1_3kph_run4 = dict(
	raw_mocap_src=r"set_1/3kph capture 4.c3d",
	raw_reactions_src="Ally 3kph run 4", reactions_time_delay=13.64,
	# mesh_data = MeshData("ply collections/Ally set 1 3kph run4 SMAL 3")
	# predicted_skeleton = mocap_data.all_data, # Load MoCap data,
	# predicted_reactions = load_paw_reactions("mocap_forces", mocap=True, mocap_src=r"set_1 - 14-10-19/3kph capture 4.c3d"),
)  # Kinematic and dynamic data from ally set 1 3kph run 4


# mocap_data = C3DData(ax=None, src=r"set_2/3 kph run 3.c3d")
# ally_set_2_3kph_run3 = dict(
#     raw_mocap_src = r"set_2/3 kph run 3.c3d",
#     raw_reactions_src = "set_2 -- 3kph run3", reactions_time_delay = 13.64,
#     # mesh_data = MeshData("ply collections/Ally set 1 3kph run4 SMAL 3")
#     predicted_skeleton = None, # Either load SMAL Mesh data, or just use raw mocap,
#     predicted_reactions = load_paw_reactions("set_2_3r3", mocap=True, mocap_src=r"set_2/3 kph run 3.c3d"),
# ) # Kinematic and dynamic data from ally set 1 3kph run 4


# For some reason, positions of foot joints in ReactionPlotter is NOT in line with actual foot joints.
# kinematics_with_reaction_forces(**ally_set_2_3kph_run3)

## Synchronizing zebris and mocap

def mocap_and_forceplate_time_series(mocap_src=r"set_1 - 14-10-19/3kph capture 4.c3d",
									 reactions_src="Ally 3kph run 4"):
	mocap_data = C3DData(ax=None, src=mocap_src)  # Load MoCap data
	reaction_positions = mocap_data.get_marker_data_by_names(*paw_names)

	freq_mocap = mocap_data.freq
	freq_forceplate = 100
	t_delay = 13.3  # 13.64 # time at which zebris data starts
	frame_delay = int(freq_forceplate * t_delay)
	n_frames_forceplate = int(len(
		reaction_positions) * freq_forceplate / freq_mocap)  # number of frames for forceplate to be same time length as mocap

	total_time = len(reaction_positions) / freq_mocap
	t_range_mocap = np.arange(0, total_time, 1 / freq_mocap)
	t_range_forceplate = np.arange(0, total_time, 1 / freq_forceplate)

	raw_reaction_data = load_force_plate_data(reactions_src)[frame_delay: frame_delay + n_frames_forceplate]

	fig = plt.figure()
	gs = GridSpec(ncols=1, nrows=4, figure=fig)

	for n_paw in range(4):
		ax = fig.add_subplot(gs[n_paw, 0])

		ax.plot(t_range_forceplate, raw_reaction_data[:, n_paw], color="red")
		ax.plot(t_range_mocap, reaction_positions[:, n_paw, 2], color="blue")

	plt.show()


def compare_predicted_forcing_time_series(mocap_src=r"set_1 - 14-10-19/3kph capture 4.c3d",
										  actual_reactions_src="Ally 3kph run 4",
										  predicted_reactions_src=""):
	mocap_data = C3DData(ax=None, src=mocap_src)  # Load MoCap data

	freq_mocap = mocap_data.freq
	freq_smal = 30
	freq_forceplate = 100

	# Load all data
	predicted_mocap_reactions, _ = load_paw_reactions(predicted_reactions_src, mocap=True, mocap_src=mocap_src)
	# predicted_smal_reactions, _ = load_paw_reactions(predicted_reactions_src, mocap=False)

	# predicted_mocap_reactions, _ = load_vert_reactions(trunk_names, predicted_reactions_src, mocap=True, mocap_src=mocap_src)

	try:
		load_force_plate_data(actual_reactions_src)
	except FileNotFoundError:
		ForcePlateData(src=actual_reactions_src.replace(" -- ",
														r"/"))  # Load and save force data if not found (making sure to change the filename to a valid directory type)

	actual_reactions, t_delay = load_force_plate_data(actual_reactions_src)

	frame_delay = int(freq_forceplate * t_delay)
	n_frames_forceplate = int(len(
		predicted_mocap_reactions) * freq_forceplate / freq_mocap)  # number of frames for forceplate to be same time length as mocap

	actual_reactions = actual_reactions[
					   frame_delay: frame_delay + n_frames_forceplate]  # crop forceplate data to match mocap/SMAL data

	total_time = len(predicted_mocap_reactions) / freq_mocap
	t_range_mocap = np.arange(0, total_time, 1 / freq_mocap)
	# t_range_smal = np.arange(0, total_time, 1/freq_smal)[:len(predicted_smal_reactions)] # crop SMAL time at end of SMAl data
	t_range_forceplate = np.arange(0, total_time, 1 / freq_forceplate)

	fig = plt.figure()
	gs = GridSpec(ncols=1, nrows=5, figure=fig)

	norm = lambda arr: arr / np.amax(arr)
	norm = lambda arr: (arr)

	for n_paw in range(4):
		ax = fig.add_subplot(gs[n_paw, 0])

		ax.plot(t_range_forceplate[:-1], norm(actual_reactions[:, n_paw]), color="red", label="Zebris")
		ax.plot(t_range_mocap[50:], norm(predicted_mocap_reactions[:, n_paw][50:]), color="blue",
				label="Predicted from MoCap")
		ax.set_title(paw_names[n_paw].title())
	# ax.plot(t_range_smal[50:], norm(predicted_smal_reactions[:, n_paw][50:]), color="green", label="Predicted from SMAL")

	ax = fig.add_subplot(gs[4, 0])
	ax.plot(t_range_forceplate[:-1], np.sum(actual_reactions, axis=1), color="red", label="Zebris")
	ax.plot(t_range_mocap[50:], np.sum(predicted_mocap_reactions[50:], axis=1), color="blue",
			label="Predicted from MoCap")
	ax.set_title("TOTAL")

	plt.subplots_adjust(hspace=0.35)
	plt.ylabel("Reaction Force F/N")
	plt.xlabel("Elapsed time, t/s")

	# Put a legend to the right of the current axis
	ax.legend(bbox_to_anchor=(0, -0.6, 1, 0.2), loc="lower center",
			  borderaxespad=0, ncol=2)

	plt.show()


def compare_predicted_forcing_time_series_left_front(mocap_src=r"set_1 - 14-10-19/3kph capture 4.c3d",
													 actual_reactions_src="Ally 3kph run 4",
													 predicted_reactions_src=""):
	"""Plots the time series just for the left paw, with slightly better presentation for powerpoints etc"""

	plt.rc("text", usetex=True)

	plt.rc('font', weight='bold')
	plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

	mocap_data = C3DData(ax=None, src=mocap_src)  # Load MoCap data

	freq_mocap = mocap_data.freq
	freq_forceplate = 100

	# Load all data
	predicted_mocap_reactions, _ = load_paw_reactions(predicted_reactions_src, mocap=True, mocap_src=mocap_src)

	try:
		load_force_plate_data(actual_reactions_src)
	except FileNotFoundError:
		ForcePlateData(src=actual_reactions_src.replace(" -- ",
														r"/"))  # Load and save force data if not found (making sure to change the filename to a valid directory type)

	actual_reactions, t_delay = load_force_plate_data(actual_reactions_src)

	frame_delay = int(freq_forceplate * t_delay)
	n_frames_forceplate = int(len(
		predicted_mocap_reactions) * freq_forceplate / freq_mocap)  # number of frames for forceplate to be same time length as mocap

	actual_reactions = actual_reactions[
					   frame_delay: frame_delay + n_frames_forceplate]  # crop forceplate data to match mocap/SMAL data

	total_time = len(predicted_mocap_reactions) / freq_mocap
	t_range_mocap = np.arange(0, total_time, 1 / freq_mocap)
	t_range_forceplate = np.arange(0, total_time, 1 / freq_forceplate)

	fig, ax = plt.subplots(figsize=(6, 9 / 4))

	colours = [(0.13, 0.12, 0.122), (200 / 255, 24 / 255, 24 / 255)]

	m = 25.7
	g = 9.81

	n_paw = 0  # left paw
	ax.plot(t_range_forceplate[:-1], actual_reactions[:, n_paw] / (m * g), color=colours[0],
			label="Forceplate measurements", lw=3)
	ax.plot(t_range_mocap[50:], predicted_mocap_reactions[:, n_paw][50:] / (m * g), color=colours[1],
			label="Inverse Dynamics predictions", lw=2)
	# ax.set_title(paw_names[n_paw].title())

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.ylabel(r"\textbf{Force, \% of bodyweight}")
	plt.xlabel(r"\textbf{Elapsed time, t/s}")

	# Put a legend to the right of the current axis
	ax.legend()

	ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x * 100)))

	plt.tight_layout()
	plt.subplots_adjust(left=0.09, bottom=0.20, right=1, top=1)
	ax.set_ylim(top=1.10)
	plt.savefig(
		r"C:\Users\Ollie\Videos\iib_project\dynamics_overview\id_graph_left_foot.png",
		dpi=300, )


# plt.savefig(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Presentations\Michaelmas presentation\id_graph.png", dpi = 300, transparent=True)


def compare_predicted_forcing_heatmap(mocap_src=r"set_1 - 14-10-19/3kph capture 4.c3d",
									  actual_reactions_src="Ally 3kph run 4",
									  predicted_reactions_src=""):
	# Transparent color map from https://stackoverflow.com/questions/51601272/python-matplotlib-heatmap-colorbar-from-transparent
	ncolors = 256
	color_array = plt.get_cmap('hot')(range(ncolors))  # get colormap
	color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)  # change alpha values
	map_object = LinearSegmentedColormap.from_list(name='hot_alpha', colors=color_array)  # create a colormap object
	plt.register_cmap(cmap=map_object)  # register this new colormap with matplotlib

	# Modified version of https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/gradient_bar.html
	# Makes bar charts with color based on data
	def heatmap_image(ax, extent, data, direction=0.3, cmap_range=(0, 1), bin_width=5, **kwargs):
		"""
		Draw a gradient image based on a colormap.

		Parameters
		----------
		ax : Axes
			The axes to draw on.
		extent
			The extent of the image as (xmin, xmax, ymin, ymax).
			By default, this is in Axes coordinates but may be
			changed using the *transform* kwarg.
		direction : float
			The direction of the gradient. This is a number in
			range 0 (=vertical) to 1 (=horizontal).
		cmap_range : float, float
			The fraction (cmin, cmax) of the colormap that should be
			used for the gradient, where the complete colormap is (0, 1).
		**kwargs
			Other parameters are passed on to `.Axes.imshow()`.
			In particular useful is *cmap*.
		"""

		# Need to get bin size as an integer value to work with imshow. So find nearest multiple of bin_width above max of data
		max_height = np.amax(data) + (bin_width - np.amax(data) % bin_width)

		hist, bin_edges = np.histogram(data, bins=int(max_height // bin_width), range=(0, max_height))

		X = []

		for density in hist: [X.append([density] * 10) for i in
							  range(bin_width)]  # add a row of density for each row of pixels in the bin

		X = np.array(X)

		a, b = cmap_range
		X = a + (b - a) / X.max() * X

		im = ax.imshow(X, extent=extent, interpolation='bicubic',
					   vmin=0, vmax=1, **kwargs)
		return im

	def gradient_bar(ax, x, y_data, label="", width=0.2, bottom=0):
		left, top = x, np.amax(y_data)
		right = left + width

		heatmap_image(ax, extent=(left, right, bottom, top), data=np.array(y_data),
					  cmap="gist_yarg", cmap_range=(0, 0.8))

	mocap_data = C3DData(ax=None, src=mocap_src)  # Load MoCap data

	freq_mocap = mocap_data.freq
	freq_forceplate = 100

	# Load all data
	predicted_mocap_reactions, _ = load_paw_reactions(predicted_reactions_src, mocap=True, mocap_src=mocap_src)

	try:
		load_force_plate_data(actual_reactions_src)
	except FileNotFoundError:
		ForcePlateData(src=actual_reactions_src.replace(" -- ",
														r"/"))  # Load and save force data if not found (making sure to change the filename to a valid directory type)

	actual_reactions, t_delay = load_force_plate_data(actual_reactions_src)

	frame_delay = int(freq_forceplate * t_delay)
	n_frames_forceplate = int(len(
		predicted_mocap_reactions) * freq_forceplate / freq_mocap)  # number of frames for forceplate to be same time length as mocap

	actual_reactions = actual_reactions[
					   frame_delay: frame_delay + n_frames_forceplate]  # crop forceplate data to match mocap/SMAL data

	total_time = len(predicted_mocap_reactions) / freq_mocap
	t_range_mocap = np.arange(0, total_time, 1 / freq_mocap)
	# t_range_smal = np.arange(0, total_time, 1/freq_smal)[:len(predicted_smal_reactions)] # crop SMAL time at end of SMAl data
	t_range_forceplate = np.arange(0, total_time, 1 / freq_forceplate)

	fig, ax = plt.subplots()

	width = 0.1
	sep = 4 * width
	for n_paw in range(4):
		gradient_bar(ax, x=n_paw * sep, y_data=actual_reactions[:, n_paw], label=paw_names[n_paw], width=width)
		gradient_bar(ax, x=n_paw * sep + width, y_data=predicted_mocap_reactions[:, n_paw][50:], label=paw_names[n_paw],
					 width=width)

	ax.set_xlim(0, n_paw * sep + 2 * width)
	ax.set_ylim(0, 250)
	ax.set_aspect('auto')
	# Put a legend to the right of the current axis
	# ax.legend(bbox_to_anchor=(0, -0.6, 1, 0.2), loc="lower center",
	#            borderaxespad=0, ncol=2)

	plt.show()


if __name__ == "__main__":
	mocap_and_forceplate_time_series()
# compare_predicted_forcing(mocap_src = r"set_1/3 kph run 4.c3d",
#                               actual_reactions_src = r"set_1 -- 3kph run 4",
#                                         predicted_reactions_src = "set_1_3r4") # SET 1 example

# compare_predicted_forcing_time_series(mocap_src =r"set_2/3 kph run 3.c3d",
#                                       actual_reactions_src = r"set_2 -- 3kph run3",
#                                       predicted_reactions_src = "set_2_3r3") # SET 2 example
#
# compare_predicted_forcing_time_series_left_front(mocap_src =r"set_2/3 kph run 3.c3d",
#                                       actual_reactions_src = r"set_2 -- 3kph run3",
#                                       predicted_reactions_src = "set_2_3r3") # SET 2 example - For report

# compare_predicted_forcing_heatmap(mocap_src =r"set_2/3 kph run 3.c3d",
#                                       actual_reactions_src = r"set_2 -- 3kph run3",
#                                       predicted_reactions_src = "set_2_3r3") # SET 2 example

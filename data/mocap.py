from qualisys import load_qtm_data

import matplotlib.gridspec as gridspec

from tqdm import tqdm

from data.data_loader import C3DData, Marker
from vis.utils import *


class QTMData():
	def __init__(self, src, ax, label="", dims="xyz", bones=None):
		"""
		:param src:
		:param label:
		:param dims: xyz if 3D, xy if 2D
		bones - array of (markerA, markerB), bones that connect individual markers
		"""

		if bones is None: bones = []

		self.data, self.metadata = load_qtm_data(src, multi_index=True)
		self.label = label
		self.dims = dims

		self.marker_names = self.metadata["marker_names"]
		self.frame = -1  # for moving through frames

		self.markers = []
		for marker_name in self.marker_names:
			marker = Marker(ax, data=[self.data[marker_name][dim].values for dim in self.dims], name=marker_name,
							dims=self.dims)
			self.markers.append(marker)

		self.get_bounds()  # identify

	def get_bounds(self):
		"""For each of x,y,z, get min and max of all data to define bounds"""
		self.x_bounds = [-500, 2500]
		self.y_bounds = [-500, 2500]
		self.z_bounds = [-500, 2500]

	# for n, dim in enumerate(self.dims):
	#     flattened_data = [item for sublist in self.data_by_marker[dim] for item in sublist]
	#     self.__setattr__(dim+"_bounds", (min(flattened_data), max(flattened_data)))

	def __next__(self):
		"""gives all x,y,z data for next frame"""
		[next(marker) for marker in self.markers]


class Bone():

	def __init__(self, ax, markerA, markerB, name=""):
		self.plot, = ax.plot([], [])
		self.markerA, self.markerB = markerA, markerB
		self.name = name

	def __next__(self):
		Ax, Ay, Az = self.markerA.position
		Bx, By, Bz = self.markerB.position

		self.plot.set_xdata([Ax, Bx])
		self.plot.set_ydata([Ay, By])
		self.plot.set_3d_properties([Az, Bz])


def animateQTM(src, playback_speed=1.0):
	dims = "xyz"
	# gui
	fig = plt.figure()
	if "z" in dims:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)
	p, = ax.plot([], [], [], "o")

	mocap_data = QTMData(ax=ax, src="example_data.txt", dims=dims)
	frames = int(mocap_data.metadata["no_of_frames"])

	freq = mocap_data.metadata["frequency"]

	fps = int(freq * playback_speed)

	ax.set_xlim(*mocap_data.x_bounds)
	ax.set_ylim(*mocap_data.y_bounds)
	if "z" in mocap_data.dims: ax.set_zlim(*mocap_data.z_bounds)

	def animate(i):
		try:
			next(mocap_data)
		except StopIteration:
			pass

	save_animation(animate, fig, frames, fps=fps)


def animateC3D(src=None):
	"""Plots .c3D data in 3D, front and side views."""

	# Set up plots
	fig = plt.figure()

	gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])
	ax_3d = fig.add_subplot(gs[1, :], projection='3d')
	ax_front, ax_side = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

	ax_front.set_title("Front")
	ax_side.set_title("Side")
	ax_3d.set_title("3D", pad=-10)

	ax_3d.axis("off")
	ax_3d.patch.set_edgecolor("black")
	ax_3d.patch.set_linewidth('1')

	for ax in [ax_side, ax_front]:
		ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
		ax.tick_params(axis="y", which="both", left=False, labelleft=False)

	mocap_data = C3DData(ax=ax_3d, src=src, alt_views={"xz": ax_front, "yz": ax_side})  # Load MoCap data

	fig.suptitle(mocap_data.name)

	frames = mocap_data.n_frames
	fps = mocap_data.frame_rate

	buffer = lambda low, high, val=0: (low - val, high + val)

	x_bounds, y_bounds, z_bounds = [buffer(*getattr(mocap_data, f"{dim}_bounds"), val=50) for dim in
									"xyz"]  # Get all dimensions with a buffer of 10
	# Set bounds and add axes arrows
	ax_3d.set_xlim(*x_bounds)
	ax_3d.set_ylim(*y_bounds)
	ax_3d.set_zlim(*z_bounds)

	ax_front.axis("equal")
	ax_side.axis("equal")

	ax_front.set_xlim(*x_bounds)
	ax_front.set_ylim(*z_bounds)

	ax_side.set_xlim(*y_bounds)
	ax_side.set_ylim(*z_bounds)

	scale = ((mocap_data.x_bounds[1] - mocap_data.x_bounds[0]) + (mocap_data.y_bounds[1] - mocap_data.y_bounds[0]) +
			 ((mocap_data.z_bounds[1] - mocap_data.z_bounds[0]))) / 3

	arrow_args = dict(arrowstyle="-|>", color="red", mutation_scale=scale / 150, lw=scale / 400)
	arrow_length = min([mocap_data.z_bounds[1], mocap_data.x_bounds[1], mocap_data.y_bounds[1]]) * 0.3

	a = Arrow3D((0, 0), (0, 0), (0, arrow_length), **arrow_args)
	b = Arrow3D((0, arrow_length), (0, 0), (0, 0), **arrow_args)
	c = Arrow3D((0, 0), (0, arrow_length), (0, 0), **arrow_args)
	[ax_3d.add_artist(i) for i in [a, b, c]]

	ax_3d.legend()

	progress = tqdm(total=frames)

	def animate(i):
		try:
			next(mocap_data)
			progress.update(1)

		except StopIteration:
			pass

	save_animation(animate, fig, frames, fps=fps, title=mocap_data.name)


import subprocess as sp


def file_selector():
	root = Tk()
	root.withdraw()
	return askopenfilename()


def two_vids_side_by_side(title="output", fps=24, dir=r"."):

	file1 = "0001-0222.mp4"
	file2 = "Ally walk 3 identified.mp4"

	cwd = os.getcwd()  # Get current directory
	os.chdir(dir)

	FFMPEG_BIN = r"ffmpeg"

	command = [FFMPEG_BIN, "-y",  # -y always overwrites
			   "i", os.path.join(dir, file1),
			   "-i", os.path.join(dir, file2),
			   "-t", "00:00:04",
			   '-filter_complex', 'hstack=inputs=2',
			   F'{title}.mp4']  # Output video, making sure that the correct number of videos are stacked

	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)


if __name__ == "__main__":
	two_vids_side_by_side(dir=r"C:\Users\Ollie\Videos\iib_project\mo_cap_vis\Ally walk 3 HD first render")
	# animateC3D("Ally walk 3 identified.c3d")
	# animateC3D("Ally walk0061 identified.c3d")
	# for file in os.listdir("c3d_data"):
	#     animateC3D(file)

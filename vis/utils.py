"""Various utils for visualisation."""

from matplotlib.animation import FuncAnimation, writers
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os, csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from time import perf_counter

import numpy as np


def minmax(iterable):
	return np.amin(iterable), np.amax(iterable)


def consecutive(data, stepsize=1):
	"""Split data by sections with stepsize between values"""
	return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def consecutive_or_less(data, stepsize=1):
	"""Split data by sections with stepsize or lower between values"""
	return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def file_selector():
	root = Tk()
	root.withdraw()
	return askopenfilename()


def folder_selector():
	root = Tk()
	root.withdraw()
	return askdirectory()


def try_mkdir(loc):
	"""If not a dir already, makes it"""
	if not os.path.isdir(loc):
		os.mkdir(loc)


def save_animation(func, fig, frames, title="output", fps=24,
				   dir=None, dpi=200):
	"""Saves animations"""

	if dir == None:
		dir = folder_selector()

	cwd = os.getcwd()  # Get current directory
	os.chdir(dir)

	plt.rcParams["text.usetex"] = True
	FFMPEG_BIN = "ffmpeg"  # Location of FFMPEG .exe for rendering
	plt.rcParams['animation.ffmpeg_path'] = FFMPEG_BIN

	# Set up formatting for the movie files
	ani = FuncAnimation(fig, func, frames=frames,
						interval=1000 // fps)

	Writer = writers['ffmpeg']
	writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

	ani.save(F"{title}.mp4", writer=writer, dpi=dpi)

	os.chdir(cwd)  # reset working directory


def animation_from_dir(dir, title="output_animation"):
	"""Given a directory of images, sorts them based on their number and plays these in order in an animation, saved to animation_outputs"""

	files = os.listdir(dir)
	files = sorted(files, key=lambda x: int(x.split(".")[0]))

	# print(files)

	fig, ax = plt.subplots()
	ax.axis("off")

	imshow = ax.imshow(plt.imread(os.path.join(dir, files[0])))

	def animate(i):
		imshow.set_data(plt.imread(os.path.join(dir, files[i])))

	plt.tight_layout()

	save_animation(animate, fig, frames=20, fps=2, dir=r"C:\Users\Ollie\Videos\iib_project\misc_videos", title=title)


def convert_1D_data_to_2D(x, y, z):
	"""Converts 1D x, y, and z arrays to the appropriate shapes used for 3D pyplot plotting"""

	X, Y = np.meshgrid(x, y)
	Z = np.tile(z, (len(z), 1))

	return X, Y, Z


class Timer:
	"""Object that, once initialised, begins a timer. Every time 'lap' is called with a label,
	records the time since previously lapped.
	Printing this object gives all labels: time"""

	def __init__(self):
		self.last_lap = perf_counter()

		self.times, self.labels = [], []

	def lap(self, label):
		time = perf_counter()
		elapsed = time - self.last_lap
		self.last_lap = time

		self.labels += [label]
		self.times += [elapsed]

	def __repr__(self):
		return "\n".join(["TIMER SUMMARY: "] + [f"{l}: {t}s" for l, t in zip(self.labels, self.times)])


class Arrow3D(FancyArrowPatch):
	def __init__(self, xs, ys, zs, *args, **kwargs):
		FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
		self._verts3d = xs, ys, zs

	def draw(self, renderer):
		xs3d, ys3d, zs3d = self._verts3d
		xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
		self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
		FancyArrowPatch.draw(self, renderer)

	def set_data(self, x, y, z, x_mag, y_mag, z_mag):
		self._verts3d = (x, x + x_mag), (y, y + y_mag), (z, z + z_mag)


from scipy.spatial.transform import Rotation


class Cylinder:
	"""Object that plots a 3D cylinder in space, with options for animating it.
	Takes radius, and start and end Vector object coordinates of its centerline as defining inputs"""

	def __init__(self, ax, radius, start: 'Vector', end: 'Vector', name=""):
		self.n_sides = 10  # approximates a cylinder as a prismatic regular polygon of n-sides
		self.trisurfs = []

		self.ax = ax
		self.radius = radius
		self.start = start
		self.end = end

		self.length = 0

		self.name = name

		self.draw()

	def update_data(self, start, end, radius=None):
		self.start = start
		self.end = end

		self.radius = radius if radius is not None else self.radius

		self.clear()
		self.draw()

	def clear(self):
		[t.remove() for t in self.trisurfs]

	def draw(self):
		# The points defining the cylinder are the n_side number of points all a radius away from the centerline at either end
		# To find these, find a normal to the centerline, rotate it n_sides time and select the vertex at every point
		centerline = (self.end - self.start).unit()
		normal = centerline.find_normal()

		self.start_points = []
		self.end_points = []

		for n in range(self.n_sides + 1):
			rotated_normal = normal.rotate_about(centerline, angle=n * (2 * np.pi) / self.n_sides).unit() * self.radius

			self.start_points.append(self.start + rotated_normal)
			self.end_points.append(self.end + rotated_normal)

		x, y, z = zip(*self.start_points, *self.end_points)
		X, Y, Z = convert_1D_data_to_2D(x, y, z)

		def i(arr):
			c = np.zeros(len(arr))
			c[0::2] = arr[:len(arr) // 2]
			c[1::2] = arr[len(arr) // 2:]
			return c

		# = lambda arr: arr[0::2] + arr[1::2]

		n_tris = self.n_sides * 2
		X, Y, Z = [i(v) for v in [x, y, z]]
		trisurf_kwaargs = dict(shade=False, edgecolor="none", linewidth=0, antialiased=False)

		t1 = self.ax.plot_trisurf(X, Y, Z,
								  triangles=[[n % n_tris, n + 1 % n_tris, n + 2 % n_tris] for n in range(n_tris)],
								  color="green", **trisurf_kwaargs)

		# Fill each end face using triangulation
		t2 = self.ax.plot_trisurf(*zip(*self.start_points, self.start), color="red", **trisurf_kwaargs)
		t3 = self.ax.plot_trisurf(*zip(*self.end_points, self.end), color="red", **trisurf_kwaargs)

		self.trisurfs = [t1, t2, t3]


class Vector(np.ndarray):

	def __new__(cls, x, y, z=0.0, t=None, id=None):
		obj = np.array([x, y, z]).view(cls)
		obj.t = t
		obj.id = id
		return obj

	def __gt__(self, other):
		"""Computes distance from one vector to another"""
		if isinstance(other, Vector):
			return (self - other).length()

	def __add__(self, other):
		if isinstance(other, Vector):
			return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

	def cross(self, other):
		if isinstance(other, Vector) and other.shape == self.shape:
			return Vector(*np.cross(self, other))
		elif isinstance(other, np.ndarray) and other.shape == self.shape:
			return Vector(*np.cross(self, Vector(*other)))

	def find_normal(self):
		"""Finds a normal to the vector, first by trying to cross with i, then j"""
		i, j = Vector(1, 0, 0), Vector(0, 1, 0)
		if self.cross(i).length() != 0:
			return self.cross(i)
		else:
			return self.cross(j)

	def rotate_about(self, axis, angle):
		"""Applies scipy.spatial.transform.Rotation about a given axis, and a given angle"""
		rot = Rotation.from_rotvec(angle * axis.unit())
		return Vector(*rot.apply(self))

	def express_vec_as_euler(self, seq="XYZ"):
		"""Express the (unit) vector as 3 elementary rotations, by default X, Y, Z"""
		rot = Rotation.from_rotvec(self.unit())
		return rot.as_euler(seq=seq)

	def unit(self):
		if self.length() < 1e-10:  # avoid divide by zero errors
			return Vector(np.nan, np.nan, np.nan)
		return self / self.length()

	def length(self):
		return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

	def diff(self, other, t=None, dt=None):
		"""Gives the differential d(self -> other)/dt."""
		if None not in [self.t, other.t]:
			vec = (other - self) / (other.t - self.t)
			vec.t = t  # Set correct time
			return vec

		elif dt is not None:
			return (other - self) / dt

		raise ValueError("No time data for either self or other, nor a valid dt.")

	def is_parallel(self, other):
		"""Returns True if self and other are parallel"""
		if isinstance(other, np.ndarray): other = Vector(*other)
		if isinstance(other, Vector):
			return abs(np.dot(self, other)) == self.length() * other.length()

	def angle_between(self, other):
		"""Get angle between two vectors"""
		self_unit = self.unit()
		other_unit = Vector(*other).unit()
		return np.arccos(np.clip(np.dot(self_unit, other_unit), -1.0, 1.0))

	@property
	def x(self):
		return self[0]

	@property
	def y(self):
		return self[1]

	@property
	def z(self):
		return self[2]

	def above_threshold(self, value):
		"""Returns true if the absolute value of x, y, or z are above 'value'"""
		return any(abs(i) > value for i in [self.x, self.y, self.z])

	def __iter__(self):
		return iter([self.x, self.y, self.z])

	def __str__(self):
		if getattr(self, "t", None) is None:
			return F"({self.x}, {self.y}, {self.z})"
		return F"({self.x}, {self.y}, {self.z}, t = {self.t})"

	def __repr__(self):
		return str(self)


class VectorSet(list):
	"""Array that contains a list of vectors. Shortcuts for getting all, say, y values of vectors"""

	def get_all(self, dim="x"):
		"""Returns all of a certain attribute, x by default"""
		return [getattr(v, dim) for v in self]

	def __getitem__(self, item):
		return VectorSet(list.__getitem__(self, item))

	def change_to(self, vectors):
		"""Used to get around using global vars"""
		self = VectorSet(vectors)


def batch_rodrigues(thetas):
	"""
	Takes theta, an (Nx3) numpy array dictating N rotations, described separately about X, Y, Z.

	Returns R, an (N x 3 x 3) array of rotation matrices.
	"""

	N = len(thetas)
	out = np.zeros((N, 3, 3))

	for n, theta in enumerate(thetas):
		rotvec = Rotation.from_euler("xyz", theta)
		out[n] = rotvec.as_matrix()

	return out


def list_all_files(folder, title="image_url"):
	"""Given a folder, lists all files in the folder and all its subfolders, as their relative paths,
	and outputs these as folder/file_list.csv.
	Optional parameter for a title for the csv, that is present as the first row (for use for MTurk)"""

	rows = []
	if title is not None: rows += [title]

	os.chdir(folder)
	for item in os.listdir("."):

		if item == "file_list.csv": continue  # ignore file list as file

		if os.path.isdir(item):
			rows += [os.path.join(item, subitem) for subitem in os.listdir(item)]
		else:
			rows += [item]

	with open("file_list.csv", "w", newline="") as outfile:
		writer = csv.writer(outfile)
		writer.writerows([[i] for i in rows])  # Have to convert to 2D array so each string isn't saved inidividually


import imantics


def fill_segment(data):
	"""For a binary array of data, of size (height, width), fills in all small blank spots"""

	## LOGIC FOR NOW - IF SURROUNDED BY A FILLED PIXEL IN EVERY CARDINAL DIRECTION ( within distance 5), FILL IN

	if (data < 0.1).all(): return data

	all_true = np.where(data > 0.1)
	top, bottom = min(all_true[0]), max(all_true[0])
	left, right = min(all_true[1]), max(all_true[1])

	pad = 5  # size of search region

	for n in range(top, bottom):
		for m in range(left, right):
			if data[n, m] == 0:
				if all(1 in arr for arr in
					   [data[n, m - pad:m], data[n, m:m + pad], data[n - pad:n, m], data[n:n + pad, m]]):
					data[n, m] = 1  # fill in array

	return data


def image_to_polygon(image):
	"""Given a binary image as a numpy array:
	- converts to grayscale
	- identifies polygons
	- returns properties"""

	H, W = image.shape

	polygons = imantics.Mask(image).polygons()
	polygon = polygons  # Main polygon for now

	if len(polygon.points) == 0: return np.zeros(0)

	polygons_by_area = sorted(polygon.points, key=lambda arr: PolyArea(arr), reverse=True)
	return polygons_by_area


def PolyArea(arr):
	"""Area of a polygon, for an array of shape (n_verts, 2) """
	if len(arr) == 0: return 0
	x, y = arr[:, 0], arr[:, 1]
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def add_point_to_line(l, x, y):
	"""Add (x,y) point to pyplot line2d"""

	l.set_xdata(np.concatenate([l._x, [x]]))
	l.set_ydata(np.concatenate([l._y, [y]]))

"""Program to take a .ply file and show in matplotlib. Will add more mesh functionality with time"""


import plyfile
from matplotlib import pyplot as plt
import numpy as np
import c3d, csv, tqdm
import os
from vis.visualisations import MeshPlotter
from data.data_loader import MeshData, C3DData, DataSources
from vis.utils import save_animation
src = "blank_mesh.ply"

class BBox:
	def __init__(self, x, y, z, pad = 0.0):
		self.data = data = {"x":np.array(x), "y":np.array(y), "z":np.array(z)}
		for dim in "xyz":
			for f, mult in zip([min, max], [-1, +1]):
				self.__setattr__(dim + f.__name__, f(data[dim]) + mult * pad)

	def apply_to_axes(self, ax, pad=0.1):
		"""Applies bbox limits to axes"""
		ax.set_xlim(self.xmin, self.xmax)
		ax.set_ylim(self.ymin, self.ymax)
		ax.set_zlim(self.zmin, self.zmax)

	def equal_3d_axes(self, ax, zoom=1):
		"""Sets all axes to same lengthscale through trick found here:
		https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is
		-not-equal-to """

		xmax, xmin, ymax, ymin, zmax, zmin = self.xmax, self.xmin, self.ymax, self.ymin, self.zmax, self.zmin

		max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

		mid_x = (xmax + xmin) * 0.5
		mid_y = (ymax + ymin) * 0.5
		mid_z = (zmax + zmin) * 0.5
		ax.set_xlim(mid_x - max_range, mid_x + max_range)
		ax.set_ylim(mid_y - max_range, mid_y + max_range)
		ax.set_zlim(mid_z - max_range, mid_z + max_range)

	def scale_data_to(self, x, y, z):
		"""Scales existing x, y, z data to the bounds of the BBox objects"""
		new_x, new_y, new_z = [], [], []

		for n, (source, target) in enumerate([[x, new_x], [y, new_y], [z, new_z]]):
			low, high, range = min(source), max(source), max(source) - min(source)

			dim = "xyz"[n]
			bbox_data = self.data[dim]
			bbox_low, bbox_range = min(bbox_data), max(bbox_data) - min(bbox_data)

			for val in source: # Linearly interpolate each value to fit in the new range
				frac = (val-low)/range

				target.append(bbox_low + frac * bbox_range)

		return new_x, new_y, new_z

	def __repr__(self):
		return f"[{self.xmin}, {self.xmax}], [{self.ymin}, {self.ymax}], [{self.zmin}, {self.zmax}]"

def plot_vertices(src):

	with open(src, "rb") as infile:
		plydata = plyfile.PlyData.read(infile)

	print(plydata["vertex"][0])

	X_mesh, Y_mesh, Z_mesh = list(zip(*plydata.elements[0].data))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	ax.scatter(X_mesh, Y_mesh, Z_mesh) # Plot mesh points

	# Add invisible box to make all axes same aspect ratio
	X, Y, Z = np.array(X_mesh), np.array(Y_mesh), np.array(Z_mesh)

	plt.show()

def plot_faces(mesh_dir, ax, mocap_src = None, plot_frames = None, highlight_src = "", faces_off = False, label_verts=False):
	"""Plots mesh, along with the chosen vertices on the mesh for fitting, and the corresponding MoCap markers"""
	if not faces_off:
		if plot_frames == None: mesh_data = MeshData(mesh_dir, freq=60)
		else: mesh_data = MeshData(mesh_dir, n_frames=plot_frames+1, freq=60)
		mesh_plotter = MeshPlotter(ax, mesh_data.vertex_data, mesh_data.face_data, alpha=0.5,
								   freq=mesh_data.freq)
		mesh_plotter.update_frame(n_frame=0)

	if mocap_src is not None: # Plot mocap data
		mocap_data = C3DData(ax=None, src=mocap_src, interpolate=False, norm=True)

		bbox = BBox(*zip(*mocap_data.all_data[0]), pad = -0.31) # align to zero frame atm


	else:
		bbox = BBox([-1,1], [-1,1], [-1,1])

	bbox.equal_3d_axes(ax)

	# Get marker assigments:
	with open(os.path.join(DataSources.c3d_data, "marker_assignments", highlight_src), "r") as infile:
		reader = csv.reader(infile)
		next(reader)
		marker_assignments = [(descr, vertex_idx) for descr, vertex_idx in reader]

	# For now, make every frame correspond to each frame of the mesh data, and then match the mocap data with that
	freq = mesh_data.freq
	n_frames = min(mocap_data.n_frames, mesh_data.n_frames) if plot_frames is None else plot_frames
	progress = tqdm.tqdm(total = n_frames + 1)
	tqdm.tqdm.write("Rendering mesh animation")

	ms, lw = 8, 3

	markers_plot, = ax.plot([], [], [], "o", color="blue", label="MoCap data", ms=ms)
	verts_plot, = ax.plot([], [], [], "o", color="green", label="SMAL Mesh Vertices", ms=ms)
	connectors = [ax.plot([], [], [], color="red", lw = lw)[0] for i in range(mocap_data.n_markers)]

	plt.plot([], [], [], color="red", label="Error", lw = lw) # for the legend
	plt.legend(ncol=3, loc = "upper center")

	def set_plot_data(plot, x,y,z):
		plot.set_xdata(x)
		plot.set_ydata(y)
		plot.set_3d_properties(z)

	def anim(n_frame, list_errors=False):

		t = n_frame / freq

		tot_error = 0

		frame_marker_points = []
		frame_verts_points = []

		[set_plot_data(connector, [], [], []) for connector in connectors] #Clear all connectors

		for n, (descr, vertex_idx) in enumerate(marker_assignments):
			if vertex_idx != "":
				x_mesh, y_mesh, z_mesh = mesh_plotter.get_vertex_data_at_time(t)[int(vertex_idx)]

				ax.plot([x_mesh], [y_mesh], [z_mesh], )#, label=descr)

				if label_verts: ax.text(x_mesh, y_mesh, z_mesh, s=descr, fontsize=5)

				if mocap_src is not None:
					mocap_frame_data = mocap_data.all_data[n_frame, mocap_data.get_marker_indices(descr)[0]]

					if np.linalg.norm(mocap_frame_data)!=0: # Only plot non-zero mocap data
						x_mocap, y_mocap, z_mocap = mocap_frame_data

						frame_marker_points.append([x_mesh, y_mesh, z_mesh])
						frame_verts_points.append([x_mocap, y_mocap, z_mocap])

						set_plot_data(connectors[n], [x_mesh, x_mocap], [y_mesh, y_mocap], [z_mesh, z_mocap]) # set data for connectingline between marker and vert

						# ax.plot([x_mocap], [y_mocap], [z_mocap], "o", color="green")
						# ax.plot([x_mesh, x_mocap], [y_mesh, y_mocap], [z_mesh, z_mocap], color="red")

					error = np.linalg.norm([[x_mesh-x_mocap], [y_mesh-y_mocap], [z_mesh-z_mocap]])
					tot_error+=error
					if list_errors:
						print(f"{descr} - {error}")

		all_marker_data = list(zip(*frame_marker_points))
		all_verts_data = list(zip(*frame_verts_points))
		set_plot_data(markers_plot, *all_marker_data)
		set_plot_data(verts_plot, *all_verts_data)

		if list_errors: print(f"Tot error: {round(tot_error,2)}")

		mesh_plotter.update_time(t)

		progress.update(1)

	if plot_frames == 0:
		anim(plot_frames)#, list_errors=True)

		# ax.set_xlim(-0.3, 0.3)
		ax.axis("off")
		plt.tight_layout()
		plt.savefig(r"C:\Users\Ollie\Videos\iib_project\SMAL optimisation\optimisation_image.png", dpi = 300, transparent=True)

	else:
		save_animation(anim, fig, frames = n_frames, title="mesh_fitting_optimisation", fps=freq//2,
					   dir = r"C:\Users\Ollie\Videos\iib_project\SMAL optimisation")



fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# plot_faces(mesh_dir=r"ply collections\Ally walk 3, full clip, sampling 4", ax=ax, mocap_src = r"old set (preproject)\Ally walk 3 identified.c3d", plot_frames=20, highlight_src="Ally preproject.csv")#, faces_off=True)
# plot_faces(mesh_dir=r"ply collections\Ally set_2 SMAL 4", ax=ax, mocap_src = r"set_2 - 17-10-19\3 kph run 3.c3d", highlight_src="Ally set_2.csv")#, faces_off=True)

# One frame testing on mesh fitting
# plot_faces(mesh_dir=r"Ally set_2 3r3 v2.1", ax=ax, mocap_src = r"set_2 - 17-10-19\3 kph run 3.c3d", highlight_src="Ally set_2.csv")
# plot_faces(mesh_dir=r"set_2_3r3 v2.1", ax=ax, highlight_src="Ally set_2.csv", mocap_src="set_2/3 kph run 3.c3d", plot_frames=0)

# plt.show()

# current_src = r"D:\IIB Project Outputs\Mocap Render outputs\checkpoints\20190919-154647\frame_1"
# srcs = [f"{current_src}/{file}" for file in os.listdir(current_src)
#                            if ".ply" in file]
#
# plot_mesh_fit(fig, srcs = srcs, mocap_data="Ally walk0061 identified.c3d", mocap_frame=1)
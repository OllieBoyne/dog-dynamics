"""SMAL Renderer object responsible for visualising a SMAL sequence"""

from data.data_loader import extract_npz, DataSources
from vis.mesh_viewer import BBox
import torch
from smal_fitter.smbld_model.smbld_mesh import SMBLDMesh
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from vis.utils import save_animation
from tqdm import tqdm
import os

path_join = os.path.join


class SMALRenderer:
	out_dir = r"C:\Users\Ollie\Videos\iib_project\KDN meshes"

	def __init__(self, src, freq=30, crop=None, smooth=False):

		self.src = src
		self.freq = freq

		data = extract_npz(path_join(DataSources.smal_outputs, src + ".npz"))
		self.data = data
		self.img_names = data['imgname']
		self.has_bbox = data['has_bbox']

		n_frames, *_ = data['pose'].shape
		self.n_frames = n_frames

		pose = torch.from_numpy(data['pose'].reshape(n_frames, 35, 3))
		smbld = SMBLDMesh(n_batch=n_frames)

		if smooth:
			pose = torch.from_numpy(signal.savgol_filter(pose, window_length=9, polyorder=5, axis=0))  # smooth pose
		else:
			pose = pose  # smooth pose

		self.n_betas = 26
		mean_betas = data['betas'].mean(axis=0).astype(np.float64)
		for n, b in enumerate(mean_betas):
			smbld.multi_betas[n] = b

		# temp fix to add global rotation if desired
		# j = 2
		# smalx.global_rot[:, j] = np.pi - pose[:, 0, j] # only keep z rotation globally

		smbld.joint_rot[:] = pose[:, 1:]
		smbld.joint_rot[:, 0] *= 0  # rear-to-front rotation to 0 for now

		v, f, J = smbld.get_verts(return_joints=True)

		if crop is not None:
			crop_frames = int(crop * freq)
			v = v[:crop_frames]
			J = J[crop_frames]
			self.n_frames = crop_frames

		self.v, self.f, self.J = v.detach().numpy(), f, J.detach().numpy()

	def plot(self, azim=90, elev=0, playback=1.0, ext=""):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")

		X, Y, Z = np.swapaxes(self.v[0], 0, 1)

		b = BBox(X, Y, Z)
		b.equal_3d_axes(ax, zoom=1.75)
		ax.axis("off")

		ax.view_init(azim=azim, elev=elev)

		# PLOT FACES
		tri = Triangulation(X, Y, triangles=self.f)

		p = tqdm(total=self.n_frames)

		plot = [ax.plot([], [])[0]]

		def anim(i):
			plot[0].remove()
			X, Y, Z = np.swapaxes(self.v[i], 0, 1)
			plot[0] = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color="#064ac7", shade=True)

			p.update()

		anim(1)
		plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
		# plt.show()

		title = self.src if ext == "" else f"{self.src}_{ext}"
		save_animation(anim, fig, self.n_frames, title=title, dir=self.out_dir,
					   fps=self.freq * playback)


class MultiviewRenderer(SMALRenderer):
	imgdir = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\clips for kdn"
	"""Object for rendering SMAL of multiview clips"""

	def __init__(self, src, freq=30, crop=None, smooth=False):
		super().__init__(src, freq, crop, smooth)

		self.ncams = 4
		self.idxs_by_cam = []  # array of idxs for each view

		for n in range(1, self.ncams + 1):
			files = [i for i, f in enumerate(self.img_names) if f"{n}_" in f]
			self.idxs_by_cam.append(files)

		self.n_frames = len(self.idxs_by_cam[0])
		assert all(len(idxs) == self.n_frames for idxs in self.idxs_by_cam), "Each view is not same length"

		self.has_bbox_mat = np.array(self.has_bbox).reshape(self.ncams, self.n_frames)

	def plot_anim(self, azim=90, elev=0, playback=1.0, ext=""):
		fig = plt.figure()
		# fixed at 4 cams for now
		axes = [fig.add_subplot(f"22{n}", projection="3d") for n in range(self.ncams)]

		X, Y, Z = np.swapaxes(self.v[0], 0, 1)

		plots = []
		for ax in axes:
			b = BBox(X, Y, Z)
			b.equal_3d_axes(ax, zoom=1.75)
			ax.axis("off")

			ax.view_init(azim=azim, elev=elev)

			plots.append(ax.plot([], [])[0])  # blank plots to modify

		# PLOT FACES
		tri = Triangulation(X, Y, triangles=self.f)

		p = tqdm(total=self.n_frames)

		def anim(i):
			for n in range(self.ncams):
				plots[n].remove()
				idx = self.idxs_by_cam[n][i]
				X, Y, Z = np.swapaxes(self.v[idx], 0, 1)

				colour = "#064ac7" if self.has_bbox[idx] else "red"
				plots[n] = axes[n].plot_trisurf(X, Y, Z, triangles=tri.triangles, color=colour, shade=True)

			p.update()

		plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

		title = self.src if ext == "" else f"{self.src}_{ext}"
		save_animation(anim, fig, self.n_frames, title=title, dir=self.out_dir,
					   fps=self.freq * playback)

	def plot_image(self, azim=90, elev=0, frame=2):
		"""provides plot of stills (top row) and mesh (bottom row)"""
		fig = plt.figure(figsize=(8, 2))
		# fixed at 4 cams for now
		still_axes = [fig.add_subplot(f"24{n + 1}") for n in range(self.ncams)]
		smal_axes = [fig.add_subplot(f"24{n + 5}", projection="3d") for n in range(self.ncams)]

		for ncam, ax in enumerate(still_axes):
			idx = self.idxs_by_cam[ncam][frame]
			imgsrc = path_join(self.imgdir, "crops", self.img_names[idx])
			ax.imshow(plt.imread(imgsrc))

			ax.axis("off")

		X, Y, Z = np.swapaxes(self.v[0], 0, 1)
		tri = Triangulation(X, Y, triangles=self.f)
		for ncam, ax in enumerate(smal_axes):
			b = BBox(X, Y, Z)
			b.equal_3d_axes(ax, zoom=1.75)
			ax.axis("off")

			ax.view_init(azim=azim, elev=elev)

			idx = self.idxs_by_cam[ncam][frame]
			X, Y, Z = np.swapaxes(self.v[idx], 0, 1)

			colour = "#064ac7" if self.has_bbox[idx] else "red"
			ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color=colour, shade=True)

		# anim(1)
		plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0)
		# plt.show()
		plt.savefig(
			r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\multiview fit\douglas_walk.png",
			dpi=300)

	def compare(self):

		betas = self.data['betas'].reshape(self.ncams, self.n_frames, self.n_betas)
		pose = self.data['pose'][:, 3:].reshape(self.ncams, self.n_frames, 102)  # joint rot (no global rot)

		# betas = self.data['betas'].reshape(self.ncams, self.n_frames)

		all_pose_dev = []  # list of arrays of size (102,) for std dev of pose in each frame
		all_shape_dev = []

		for f in range(self.n_frames):
			valid = self.has_bbox_mat[:, f]

			frame_betas = betas[valid, f]
			frame_pose = pose[valid, f]

			if valid.sum() > 1:
				# print(f"Frame {f}. {valid.sum()} found.")
				pose_dev = np.std(frame_pose, axis=0)
				shape_dev = np.std(frame_betas, axis=0)

				all_pose_dev.append(pose_dev)
				all_shape_dev.append(shape_dev)

		mean_pose_dev = np.mean(all_pose_dev)

		all_shape_dev = np.array(all_shape_dev)
		mean_beta_dev = np.mean(all_shape_dev[:, :20])
		mean_scalefactors_dev = np.mean(all_shape_dev[:, 20:])

		print(self.src, f"CLIPS EVALUATED: {len(all_pose_dev)}", f"Pose: {mean_pose_dev * 180 / np.pi:.2f}",
			  f"Betas: {mean_beta_dev:.3f}",
			  f"Scale Factors: {mean_scalefactors_dev:.3f}")


if __name__ == "__main__":

	# S = SMALRenderer("zebris_lab", crop = None, smooth=True)
	# S.plot(azim=70, elev = 20, playback = 0.5, ext="smoothed")

	mv_clips = {
		"ally": "ally_jump_mv_dynamic",
		"douglas": "douglas_walk_mv_dynamic",
		"gracie": "gracie_playing_mv_dynamic",
		"patrick": "patrick_walk_mv_p9",
	}

	MV = MultiviewRenderer(mv_clips["gracie"])
	MV.compare()
	# MV.plot_image(frame=0)
	# MV.plot_anim(playback=0.1)

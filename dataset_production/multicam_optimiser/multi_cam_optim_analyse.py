import json, os, tqdm
import numpy as np
from torch.nn import functional
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from dataset_production.multicam_optimiser.multi_cam_optimiser import *
from vis.utils import *
from scipy.spatial.transform import Rotation

ordered_colours = ["#d82400"] * 3 + ["#fcfc00"] * 3 + ["#48b455"] * 3 + ["#0090aa"] * 3 + ["#d848ff", "#d848ff",
																						   "#fc90aa", "#006caa",
																						   "#d89000", "#d89000"]

keypoints_src = "E:\IIB Project Data\Training data sets\Dogs\Arena dataset\labelled_keypoints\keypoints.json"
clips_src = "E:\IIB Project Data\Training data sets\Dogs\Arena dataset"


def load_clip(clip="Gracie Playing", n_cams=4, n_kp=18):
	"""Given a clip name, returns an (n_frames, n_cams, n_keypoints, 2) array of (x,y) of each keypoint for each camera at each frame.
	Also returns a dictionary of metadata"""

	with open(keypoints_src, "r") as infile:
		json_data = json.load(infile)
		this_clip_data = [i for i in json_data if clip in i["img_path"]]

		# assume all are same resolution, then extract width and height as so:
		width, height = this_clip_data[0]["img_width"], this_clip_data[0]["img_height"]
		print(f"W: {width}, H: {height}")

		n_frames = len(this_clip_data) // 4  # min number of frames (accounting for rounding)

		data = np.zeros((n_frames, n_cams, n_kp, 2))  # array of size (n_frames, n_cams, n_keypoints, 2)

		for cam in range(n_cams):
			frame_list = [i for i in this_clip_data if f"{cam + 1}_" in i["img_path"]]
			frame_list = sorted(frame_list, key=lambda i: i["img_path"])  # sort frames in order (by image name)
			for frame in range(n_frames):
				data[frame, cam] = np.array(frame_list[frame]["joints"])[:, :2]  # ignore visibility flag for now

		return data, {"img_width": width, "img_height": height}


def optim(clip="Gracie Playing", live_plot=False, lr=0.1, n_it=2000):
	"""Given a clip, optimises camera params to clip"""

	n_cams = 4
	n_kp = 18

	data, metadata = load_clip(clip, n_cams, n_kp)
	width, height = metadata["img_width"], metadata["img_height"]

	centres = np.zeros((n_cams, 2))
	centres[:, 0] = width
	centres[:, 1] = height

	multicam = MultiCam(data, centres, discard_cams=[2])
	optimiser = torch.optim.Adam(multicam.params, lr=lr)
	n_it = n_it

	progress = tqdm.tqdm(total=n_it)
	losses = []

	# SET UP PLOT
	fig, (ax_loss, ax_cam) = plt.subplots(ncols=2, figsize=(8, 6))
	if live_plot:   plt.show(block=False)

	trans = multicam.trans
	scats = []

	for cam in range(n_cams):
		t = trans[cam].detach().numpy()
		scats.append(ax_cam.scatter(t[0], t[1], label=str(cam + 1)))
	scat_pred_global = ax_cam.scatter([0], [0], label="pred_global")

	ax_cam.set_xlim(-5, 30)
	ax_cam.set_ylim(-5, 30)

	loss_plots = {}
	for l in ["all", "keypoint", "time"]:
		loss_plots[l], = ax_loss.semilogx([], [], label=l)

	ax_loss.set_xlim(1, n_it)

	ax_cam.legend()
	ax_loss.legend()

	loss_breakdowns = []  # array of loss_breakdown for each step

	for i in range(n_it):
		# step optimizer
		optimiser.zero_grad()  # Initialise optimiser
		loss, loss_breakdown = multicam()  # run forward loop
		loss.backward()

		optimiser.step()

		losses.append(loss)

		# update tqdm
		progress.update()
		progress.set_description(f"LOSS = {loss}")

		if live_plot:
			# UPDATE PLOTS
			# MAIN LOSSES
			add_point_to_line(loss_plots["all"], i, loss)
			if i == 0: ax_loss.set_ylim(0,
										1.5 * loss.detach().numpy())  # set y lim on first move (assume loss only gets lower)

			# LOSS BREAKDOWN
			for label, l in loss_breakdown.items():
				add_point_to_line(loss_plots[label], i, sum(l))

			# plot position of cameras, and CoM
			for cam in range(n_cams):
				x, y, z = multicam.trans[cam].detach().numpy()
				scats[cam].set_offsets([x, y])

			pred_global = multicam.pred_global.detach().numpy()
			com = np.mean(pred_global, axis=(0, 1))
			scat_pred_global.set_offsets(com[:2])  # plot X and Y of CoM

			plt.draw()
			plt.pause(1e-8)  # necessary to update pyplot

		else:
			loss_breakdowns.append(loss_breakdown)

	# Update graph at end if not live plot
	if not live_plot:
		loss_plots["all"].set_xdata(list(range(n_it)))
		loss_plots["all"].set_ydata(losses)
		ax_loss.set_ylim(0, max(map(lambda i: i.detach().numpy(), losses)))

		for l in loss_breakdowns[0].keys():
			values = [sum(i[l]) for i in loss_breakdowns]
			loss_plots[l].set_xdata(list(range(n_it)))
			loss_plots[l].set_ydata(values)

		# plot position of cameras, and CoM at each frame
		for cam in range(n_cams):
			x, y, z = multicam.trans[cam].detach().numpy()
			scats[cam].set_offsets([x, y])

		pred_global = multicam.pred_global.detach().numpy()
		com = np.mean(pred_global, axis=(1))
		ax_cam.plot(*zip(*com[:, :2]), "-o", label="pred global traj",
					color="brown", alpha=.6, markersize=1)
		ax_cam.legend()
	# scat_pred_global.set_offsets(com[:2])  # plot X and Y of CoM

	out_src = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\camera_fit"
	try_mkdir(os.path.join(out_src, clip))

	plt.savefig(os.path.join(out_src, clip, "fit_process.png"), dpi=300)

	for param in ["trans", "rots", "pred_global", "scale_factor", "focal", "distortion", "pred_local", "mask"]:
		np.save(os.path.join(out_src, clip, f"{param}.npy"), getattr(multicam, param).detach().numpy())

	return loss  # return final loss as output


def plot_pred_global(clip="Gracie Playing"):
	"""Given a clip name, plots the predicted global movement, saves in desired dir"""
	src_dir = f"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\camera_fit/{clip}"

	n_cams = 4
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	pred_global = np.load(os.path.join(src_dir, "pred_global.npy"))

	frames = len(pred_global)
	p = tqdm.tqdm(total=frames)

	all_x, all_y, all_z = np.rollaxis(pred_global, axis=2, start=0)
	x0, y0, z0 = all_x[0], all_y[0], all_z[0]

	scat = ax.scatter(x0, y0, z0, c=ordered_colours)

	## PLOT CAMERAS - as small planes
	rots, trans = [np.load(os.path.join(src_dir, f"{i}.npy")) for i in ["rots", "trans"]]

	I = np.array([1, 0, 0])
	cam_cs = []
	for c in range(n_cams):
		rot = Rotation.from_euler("XYZ", rots[c])
		norm = np.matmul(rot.as_matrix(), I)  # apply rotations to i vector
		centre = trans[c]
		cam_cs.append(centre)
		v1 = np.cross(norm, I)  # get one perp vector
		v2 = np.cross(norm, v1)  # get perp vector
		if np.count_nonzero(v1) > 0 and np.count_nonzero(v2) > 0:
			v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
			verts = []  # 4 corners of camera plane
			for (d1, d2) in [(-v1, v2), (v1, v2), (v1, -v2), (-v1, -v2), (-v1, v2)]:
				verts.append(centre + d1 + d2)

			x, y, z = np.swapaxes(verts, 0, 1)
			plt.plot(x, y, z, lw=5)

	# All limits
	cam_cs = np.array(cam_cs)

	# Load generated mask
	mask = np.load(os.path.join(src_dir, "mask.npy"))  # load mask of data to include in plot
	mask = np.swapaxes(mask, 0, 1)
	mask_3d = np.mean(mask, axis=(0, -1)) > 0.5  # For now, accept any kp seen by more than 50% of camera

	data_masked = pred_global * np.stack([mask_3d] * 3, axis=2)
	pred_global_com = np.mean(data_masked, axis=1)
	x_com, y_com, z_com = np.swapaxes(pred_global_com, 0, 1)  # get masked x, y, z

	ax.set_xlim(min(cam_cs[:, 0].min(), x_com.min()), max(cam_cs[:, 0].max(), x_com.max()))
	ax.set_ylim(min(cam_cs[:, 1].min(), y_com.min()), max(cam_cs[:, 1].max(), y_com.max()))
	ax.set_zlim(0, cam_cs[:, 2].max())

	def anim(i):
		X, Y, Z = np.swapaxes(pred_global[i], 0, 1)
		X[np.invert(mask_3d[i])] = None
		Y[np.invert(mask_3d[i])] = None
		Z[np.invert(mask_3d[i])] = None
		# print(X[:2], Y[:2], Z[:2])
		scat._offsets3d = (X, Y, Z)
		p.update()

	save_animation(anim, fig, frames=frames, fps=2, dir=src_dir, title="pred_global")


def plot_pred_local(clip="Gracie Playing"):
	"""Given a clip name, plots the predicted local pixels from each cam individually, saves in desired dir"""
	src_dir = f"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\camera_fit/{clip}"

	n_cams = 4
	# Load original clips, images
	with open(keypoints_src, "r") as infile:
		json_data = json.load(infile)
		this_clip_data = [i for i in json_data if clip in i["img_path"]]

		# assume all are same resolution, then extract width and height as so:
		width, height = this_clip_data[0]["img_width"], this_clip_data[0]["img_height"]
		print(f"W: {width}, H: {height}")

		n_frames = len(this_clip_data) // 4  # min number of frames (accounting for rounding)

		all_img_data = np.zeros((n_frames, n_cams, height, width, 3))  # array of all_img_data

		# load image for each frame
		for cam in range(n_cams):
			frame_list = [i for i in this_clip_data if f"{cam + 1}_" in i["img_path"]]
			frame_list = sorted(frame_list, key=lambda i: i["img_path"])  # sort frames in order (by image name)
			for frame in range(n_frames):
				img_path = frame_list[frame]["img_path"]
				img_data = plt.imread(os.path.join(clips_src, img_path))
				all_img_data[frame, cam] = img_data

	# Load generated mask
	mask = np.load(os.path.join(src_dir, "mask.npy"))  # load mask of data to include in plot

	fig, axes = plt.subplots(nrows=2, ncols=2)
	pred_local = np.load(os.path.join(src_dir, "pred_local.npy"))  # get local predictions
	pred_local = pred_local * mask  # For now, set all data outside of mask to 0

	scats = []
	imshows = []
	for c in range(4):
		ax = axes[c // 2, c % 2]
		# load image for each
		imshows.append(ax.imshow(all_img_data[0, c]))

		scats.append(ax.scatter([0] * 18, [0] * 18, s=1, c=ordered_colours))

		# set so adapts xlim, ylim from data
		ax.set_xlim(0, width)
		ax.set_ylim(height, 0)

	frames = pred_local.shape[0]
	p = tqdm.tqdm(total=frames)

	def anim(i):
		for cam in range(4):
			data = pred_local[:, cam]

			ax = axes[cam // 2, cam % 2]

			scat = scats[cam]
			scat.set_offsets(data[i])
			imshows[cam].set_data(all_img_data[i, cam])

		p.update()

	plt.tight_layout()
	plt.subplots_adjust(left=.07, bottom=0, right=.98, top=1.0, wspace=.16, hspace=0)
	save_animation(anim, fig, frames=frames, fps=2, dir=src_dir, title="pred_local")


if __name__ == "__main__":
	clip = "Gracie Playing"  # "Ally Jump"
	optim(clip=clip, live_plot=False, n_it=1000)
	plot_pred_global(clip)

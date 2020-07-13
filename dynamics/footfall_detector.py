"""Quick test to see if there is any viability in a NN to detect footfalls"""

import torch
import numpy as np

from tqdm import tqdm
from data.data_loader import load_force_plate_data, C3DData, SMALData
from vis.utils import consecutive, add_point_to_line, consecutive_or_less
from matplotlib import pyplot as plt
from scipy import signal
import os

nn = torch.nn
functional = nn.functional

path_join = os.path.join

save_loc = r"E:\IIB Project Data\produced data\ff trainer"

sample_len = 200


def sigmoid(s, k=1):
	return 1 / (1 + torch.exp(-k * s))


def sigmoidPrime(s):
	# derivative of sigmoid
	return s * (1 - s)


def time_deriv(X, dt=.01):
	X = X.detach().numpy()
	X_smoothed = signal.savgol_filter(X, window_length=51, polyorder=5)

	return torch.from_numpy((X_smoothed[1:] - X_smoothed[:-1]) / dt)


freq_forceplate = 100


def norm_kin_data(kin_data):
	"""Normalise kinematic data"""

	# scale so minimum is at (0,0,0)
	for dim in [0, 1, 2]:
		kin_data[:, :, dim] -= kin_data[:, :, dim].min()

	kin_data = 0.635 * kin_data / np.max(kin_data[:, :, 2])

	return kin_data


def get_dyn_data(dynamic_src, clip_length, mass, is_mocap=True, target_freq=100):
	"""Loads and returns kinematic data"""

	force_plate_data, force_plate_tdelay = load_force_plate_data(dynamic_src, is_mocap)
	raw_dyn_data = force_plate_data
	raw_dyn_data *= 1 / (mass * 9.81)

	# resample if requested
	if target_freq != freq_forceplate:
		target_frames = int(len(raw_dyn_data) * target_freq / freq_forceplate)
		dyn_data = signal.resample(raw_dyn_data, target_frames)

		# this resampling causes a jumpiness for the periods of zero value. Fix that here:
		tol = 1e-4
		for paw in range(dyn_data.shape[1]):
			# get indices where should be 0
			antifootfalls = consecutive(np.where(raw_dyn_data[:, paw] < tol)[0])
			min_width = 10  # in frames

			for aff in antifootfalls:
				if len(aff) < min_width: continue
				start, end = aff[0] * target_freq / freq_forceplate, aff[-1] * target_freq / freq_forceplate
				# ^ start and end indices, in remapped frame
				dyn_data[int(start):int(end), paw] = 0  # set to 0

		freq = target_freq

	else:
		freq = freq_forceplate
		dyn_data = raw_dyn_data

	frame_delay = int(freq * force_plate_tdelay)
	n_frames_forceplate = int(clip_length * freq)  # number of frames for forceplate to be same time length as mocap

	if frame_delay == 0:
		return dyn_data[:n_frames_forceplate]

	if frame_delay > 0:  # crop forceplate data
		return dyn_data[frame_delay: frame_delay + n_frames_forceplate]  # crop forceplate data to match mocap/SMAL data

	else:  # fdelay <0, pad forceplate data
		return np.pad(dyn_data, ((int(-frame_delay), 0), (0, 0)))[:n_frames_forceplate]


def load_kin_data(kin_src, clip_length, mocap=True, resample_freq=100):
	if mocap:
		joint_data = C3DData(ax=None, src=kin_src, interpolate=True, crop=clip_length,
							 fix_rotations="3 kph" in kin_src)

	else:
		joint_data = SMALData(kin_src, freq=30, norm=True, crop=clip_length, smooth=True)

	joint_data.resample_at(resample_freq)  ### TRY RESAMPLING DATA TO 100 Hz
	target_bones, body_joints, no_torque_joints, ls_joints = joint_data.generate_skeleton_mapping()

	# Normalise data based on z data, so that the dog is roughly 0.5m high. Also smooth data
	kin_data = np.array(joint_data.all_data)
	kin_data = norm_kin_data(kin_data)

	paw_data = kin_data[:, no_torque_joints, 2]

	if not mocap:
		paw_data = signal.savgol_filter(paw_data, window_length=51, polyorder=2, axis=0)

	return paw_data


class FootfallDetector(nn.Module):

	def __init__(self, train=False, load=False, name=""):
		"""Takes up to 100 frames and detects footfalls."""

		super().__init__()

		self.name = name

		## declare network for now as simple, 4 layer
		self.input_size = sample_len
		self.h1_size = 50
		self.h2_size = 50
		self.output_size = sample_len

		if train:
			# weights
			self.W1 = nn.Parameter(torch.randn(self.input_size, self.h1_size))
			self.W2 = nn.Parameter(torch.randn(self.h1_size, self.h2_size))
			self.W3 = nn.Parameter(torch.randn(self.h2_size, self.output_size))

		if load:  # preload
			data = np.load(path_join(save_loc, f"weights_{name}.npz"))
			self.W1 = torch.from_numpy(data['W1'])
			self.W2 = torch.from_numpy(data['W2'])
			self.W3 = torch.from_numpy(data['W3'])

	def forward(self, X):

		# normalise X from 0 to 1
		X_norm = X - X.min()
		X_norm *= 1 / X.max()

		# V = time_deriv(X)
		# A = time_deriv(V)

		# input_data = torch.cat([X_norm, V, A])

		v = X_norm

		for weight in [self.W1, self.W2, self.W3]:
			v = torch.matmul(v, weight)
			v = sigmoid(v)

		return sigmoid(v - 0.5, k=10)

	def loss(self, input_samples, gt_samples):

		loss = 0

		nsamples = len(input_samples)
		for i, gt in zip(input_samples, gt_samples):
			loss += functional.l1_loss(self(i), gt)

		return loss / nsamples

	def save(self):

		W1 = self.W1.detach().numpy()
		W2 = self.W2.detach().numpy()
		W3 = self.W3.detach().numpy()

		title = "weights.npz"
		if self.name != "": title = f"weights_{self.name}.npz"

		np.savez(path_join(save_loc, title),
				 W1=W1, W2=W2, W3=W3
				 )
		print("SAVED")

	def process_clip(self, X):
		"""Process clip of any length"""

		X = torch.from_numpy(X).float()
		net_out = np.zeros_like(X)
		for s in np.arange(0, X.shape[0], self.input_size):  # sample each section and analyse
			net_out[s: s + self.input_size] = self(X[s: s + self.input_size]).detach().numpy()

		# process output into binary array
		output = np.zeros_like(net_out)
		min_width, tol = 10, .1
		merge_width = 4  # merge adjacent footfalls
		for ff in consecutive_or_less(np.where(net_out > tol)[0], merge_width):
			ff_start, ff_end = ff.min(), ff.max()
			if ff_end - ff_start > min_width:
				output[ff_start:ff_end + 1] = 1

		return output


def load_data(kin_srcs, grf_srcs, mocap=True):
	if type(kin_srcs) == str: kin_srcs = [kin_srcs]
	if type(grf_srcs) == str: grf_srcs = [grf_srcs]

	# frame_delay = 23 # for mocap
	freq = sample_len
	clip_length = 11 if mocap else 5

	train_in = []
	train_gt = []

	val_in = []
	val_gt = []

	ntrain = 2000
	nval = 200

	nsets = len(kin_srcs)

	train_per_set = ntrain // nsets
	val_per_set = nval // nsets

	for s in range(nsets):

		kin_src = kin_srcs[s]
		grf_src = grf_srcs[s]

		kin_data = load_kin_data(kin_src, clip_length=clip_length, mocap=mocap, resample_freq=freq)
		dyn_data = get_dyn_data(grf_src, clip_length=clip_length, mass=25.7, is_mocap=mocap, target_freq=freq)

		# Smoothing
		# if not mocap:
		#     kin_data = signal.savgol_filter(kin_data, window_length=99, polyorder=3, axis=0)

		# Select frame delay through quick optimisation of minimum area
		test_fds = []
		test_fd_range = np.arange(-50, 50, 1)
		for i in test_fd_range:
			l = ((np.roll(dyn_data, -i, axis=0) > 0)[:, 0] * kin_data[:, 0]).sum()
			test_fds.append(l)

		# if s == 1:
		#     plt.plot(test_fd_range,test_fds)
		#     plt.show()

		frame_delay = test_fd_range[np.argmin(test_fds)]
		print(f"Selected Frame Delay: {frame_delay}")
		#
		# if s == 0:
		#     plt.plot(kin_data[:, 0], "-o", ms=1)
		#     plt.twinx().plot(np.roll(dyn_data[:, 0], -frame_delay) >0, color="orange")
		#     plt.show()

		for i in range(train_per_set):
			p = np.random.choice([0, 1, 2, 3])  # random paw choice

			# sample from start to sample_len frames before the end
			if frame_delay > 0:
				start_frame = np.random.randint(0, clip_length * freq - sample_len - frame_delay)
			else:
				start_frame = np.random.randint(-frame_delay, clip_length * freq - sample_len + frame_delay)

			kin_sample = kin_data[start_frame:start_frame + sample_len, p]

			dyn_sample = dyn_data[start_frame + frame_delay:start_frame + sample_len + frame_delay, p] > 0

			train_in.append(kin_sample)
			train_gt.append(dyn_sample)

		for i in range(val_per_set):
			p = 3  # for now, have 4th paw as val

			# sample from start to sample_len frames before the end (inc. frame delay in calculation too).
			if frame_delay > 0:
				start_frame = np.random.randint(0, clip_length * freq - sample_len - frame_delay)
			else:
				start_frame = np.random.randint(-frame_delay, clip_length * freq - sample_len + frame_delay)

			kin_sample = kin_data[start_frame:start_frame + sample_len, p]
			dyn_sample = dyn_data[start_frame + frame_delay:start_frame + sample_len + frame_delay, p] > 0

			val_in.append(kin_sample)
			val_gt.append(dyn_sample)

	train_in = torch.FloatTensor(train_in)
	train_gt = torch.FloatTensor(train_gt)

	val_in = torch.FloatTensor(val_in)
	val_gt = torch.FloatTensor(val_gt)

	return (train_in, train_gt), (val_in, val_gt)


def train(name="mocap"):
	if name == "mocap":
		kin_srcs = [r"set_2/3 kph run 4.c3d", r"set_2/6 kph run 1.c3d"]
		grf_srcs = [r"set_2 -- 3kph run4", r"set_2 -- 6kph run1"]
		# name, kin_src = "mocap", r"set_2/3 kph run 4.c3d" # Mocap

		(train_in, train_gt), (val_in, val_gt) = load_data(kin_srcs=kin_srcs,
														   grf_srcs=grf_srcs,
														   mocap=True)


	else:
		kin_srcs = [r"zebris_lab_test"]  # [r"set 2 3r4 - dynamic test", r"set 2 3r4 - left"] # r"set 2 3r4" # SMAL
		grf_srcs = [r"set_3 -- lab"]  # [r"set_2 -- 3kph run4", r"set_2 -- 3kph run4"] # r"set_2 -- 3kph run4"
		(train_in, train_gt), (val_in, val_gt) = load_data(kin_srcs=kin_srcs, grf_srcs=grf_srcs,
														   mocap=False)

	nit = 3000
	gamma = 0.8
	milestone = 200
	save_every = 500

	detector = FootfallDetector(train=True, name=name)
	# optimiser = torch.optim.SGD(detector.parameters(), lr=0.1, momentum=0.9)

	optimiser = torch.optim.Adam(detector.parameters(), lr=1e-2)
	progress = tqdm(total=nit)

	losses = []

	fig1, ax_loss = plt.subplots()
	ax_loss.set_title(name)
	train_plot, = ax_loss.semilogx([])
	val_plot, = ax_loss.semilogx([])
	ax_loss.set_xlim(1, nit)
	ax_loss.set_ylim(0, 0.5)
	plt.show(block=False)

	train_size, val_size = len(train_in), len(val_in)
	train_sample = 100  # samples per batch
	val_sample = 35  # samples per batch

	for epoch in range(nit):
		if epoch > 0 and epoch % milestone == 0:  # every milestone epochs
			for p in optimiser.param_groups:
				p['lr'] *= gamma

			ax_loss.axvline(epoch, ls="--", color="orange")

		if epoch > 0 and epoch % save_every == 0:
			detector.save()

		optimiser.zero_grad()

		# Sample for train and val
		train_idxs = np.random.choice(train_size, train_sample, replace=False)
		val_idxs = np.random.choice(val_size, val_sample, replace=False)

		loss = detector.loss(train_in[train_idxs], train_gt[train_idxs])
		val = detector.loss(val_in[val_idxs], val_gt[val_idxs])

		loss.backward()

		optimiser.step()

		losses.append(loss)
		add_point_to_line(train_plot, epoch, loss)
		add_point_to_line(val_plot, epoch, val)
		plt.draw()
		plt.pause(1e-8)  # necessary to update pyplot

		progress.update()
		progress.set_description(f"Loss = {loss:.4f}. LR = {optimiser.param_groups[0]['lr']:.2f}")

	detector.save()

	fig2, ax_sample = plt.subplots()
	sample = np.random.randint(0, 1000)
	ax_input = ax_sample.twinx()
	ax_input.plot(train_in[sample], label="input")
	ax_sample.plot(train_gt[sample], label="gt", color="green")

	pred = detector(train_in[sample]).detach().numpy()
	ax_sample.plot(pred, label="predicted", color="red")
	fig2.legend()

	plt.show()


def run():
	"""Run pretrained network on test set"""

	freq = sample_len  # n frames
	frame_delay = 16
	clip_length = 10

	kin_src = r"set_2/6 kph run 1.c3d"
	# kin_src = r"set 2 3r4"
	grf_src = r"set_2 -- 6kph run1"

	kin_data = load_kin_data(kin_src, clip_length=clip_length, mocap=True, resample_freq=freq)
	dyn_data = get_dyn_data(grf_src, clip_length=clip_length, mass=25.7, is_mocap=True, target_freq=freq)

	targ_paw = 0

	paw_kin = kin_data[:, targ_paw]
	gt_tensor = torch.from_numpy(dyn_data[:, targ_paw] > 0).float()

	detector = FootfallDetector(train=False, load=True, name="mocap")

	pred = detector.process_clip(paw_kin)

	fig, ax = plt.subplots()
	ax_input = ax.twinx()
	ax_input.plot(paw_kin, label="input")
	# ax.plot(np.roll(gt_tensor, -frame_delay), label="gt", color="green")
	ax.plot(pred, label="predicted", color="red")
	fig.legend()

	plt.show()


if __name__ == "__main__":
	train(name="smal")
# run()

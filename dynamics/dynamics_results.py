from dynamics.dynamics import *
from data.data_loader import C3DData, load_force_plate_data, ForcePlateData, SMALData, get_delay_between

from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import signal

from matplotlib.animation import FuncAnimation
from vis.mesh_viewer import BBox

plt.rc("text", usetex = True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc("savefig", dpi = 300)

# freq_forceplate = 100
sample_freq = 200
clip_length = 10# seconds


def estimate_mass(dynamic_src, **kwargs):
	"""Given a dynamic source, estimate the mass by taking the rms vertical force on the dog."""
	force_plate_data, force_plate_tdelay = load_force_plate_data(dynamic_src,)

	tot_force = force_plate_data.sum(axis=1)

	dt = .01
	T = len(tot_force) * dt#

	integrate = lambda x: np.cumsum(x * dt)
	dintF = integrate(integrate(tot_force))

	# From force balance, delx = int(int(F)) - mgT^2 / 2 = 0 (assume little vertical change)
	m = dintF[-1] * 2 / (g * T ** 2)

	print(f"Mass {m:.2f}kg")

	# for error, assume a maximum change in vertical height between the beginning and end of .2 m
	delx = 0.2
	print(f"Error {2 * m * delx / (g * T ** 2) :.5f} kg")


def anim_smal_and_mocap(mocap_src, smal_src):
	"""View animation of smal alongside mocap"""

	freq = 60

	mocap_data = C3DData(ax=None, src = mocap_src, crop = clip_length)
	smal_data = SMALData(smal_src, freq = 30) # full length clip

	for data in [mocap_data, smal_data]:
		data.resample_at(freq)

	delay = get_delay_between(mocap_data.name, "smal", "mocap")
	paw_labels = ["left front", "right front", "left rear", "right rear"]


	mocap_paws = mocap_data.get_marker_indices(*[p + " paw" for p in paw_labels])
	smal_paws = smal_data.get_marker_indices(*[p + " paw" for p in paw_labels])

	# quickly plot 3d of each
	# mocap_all_data = norm_kin_data()

	mocap_all = norm_kin_data(mocap_data.all_data)[:, mocap_paws, 2]
	smal_all = norm_kin_data(smal_data.all_data)[:, smal_paws, 2]

	# get frame matching delay through estimation
	timed_fdelay = int(delay * freq) # delay before frame matching shift
	test_fds = []
	frame_match_range = np.arange(-50, 50, 1)
	for i in frame_match_range:
		# roll smal data, and crop to desired time range, before element wise product with mocap data
		l = (smal_all[timed_fdelay+i:timed_fdelay+i+len(mocap_all), 0] * mocap_all[:, 0]).sum()
		test_fds.append(l)

	frame_delay = frame_match_range[np.argmax(test_fds)]
	print(f"Selected Frame Delay: {frame_delay}")

	# plt.plot(frame_match_range, test_fds)
	# plt.show()

	# TODO REVIEW SCALING HERE - I THINK THERE'S A BIG DISCREPANCY BETWEEN HOW MOCAP AND SMAL ARE SCALED.
	# NEEDS TO BE BASED ON SAME JOINTS (MAYBE CHANGE FROM SHOULDERS TO UPPER?)

	f = 20
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = "3d")
	b_map_mocap, body_joints, *_ = mocap_data.generate_skeleton_mapping()
	mocap_3d = norm_kin_data(mocap_data.all_data, targ_markers=np.ravel(body_joints))
	mocap_scat = ax.scatter(*zip(*mocap_3d[f]), color="green")
	mocap_plots = []
	for b, (j1, j2) in b_map_mocap.items():
		mocap_plots.append(plt.plot(*zip(*mocap_3d[f, [j1, j2]]), color="green")[0])

	b_map_smal, body_joints, *_ = smal_data.generate_skeleton_mapping()
	smal_3d = norm_kin_data(smal_data.all_data, targ_markers=np.ravel(body_joints))[timed_fdelay+frame_delay:timed_fdelay+frame_delay+len(mocap_all)]
	smal_scat = ax.scatter(*zip(*smal_3d[f]), color="red")
	smal_plots = []
	for b, (j1, j2) in b_map_smal.items():
		smal_plots.append(plt.plot(*zip(*smal_3d[f, [j1, j2]]), color="red")[0])

	def anim(i):

		for data, scat, plots, b_map in [[mocap_3d, mocap_scat, mocap_plots, b_map_mocap], [smal_3d, smal_scat, smal_plots, b_map_smal]]:
			scat._offsets3d = np.swapaxes(data[i], 0, 1)
			for n,  (j1, j2) in enumerate(b_map.values()):
				plot = plots[n]
				for dim, method in enumerate([plot.set_xdata, plot.set_ydata, plot.set_3d_properties]):
					method(data[i, [j1,j2], dim])


	X, Y, Z = np.swapaxes(mocap_3d, 0, 2)
	bbox = BBox(np.ravel(X), np.ravel(Y), np.ravel(Z))
	bbox.equal_3d_axes(ax, zoom = 1.5)

	func = FuncAnimation(fig, anim, frames=100)
	plt.show()

def compare_smal_to_mocap(mocap_src, smal_src, print_vals = False):
	"""Produces graph of z position for all paw joints, for mocap and smal src"""
	freq = 60

	mocap_data = C3DData(ax=None, src = mocap_src, crop = clip_length, fix_rotations = True)
	smal_data = SMALData(smal_src, freq = 30, smooth=True) # full length clip

	# for data in [mocap_data, smal_data]:
	#     data.resample_at(freq)

	delay = get_delay_between(mocap_data.name, "smal", "mocap")
	paw_labels = ["left front", "right front", "left rear", "right rear"]
	colours = ["green", "red"]

	fig, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all",
							 figsize = (8, 4))
	ymax = 0 # running y max

	mocap_paws = mocap_data.get_marker_indices(*[p + " paw" for p in paw_labels])
	smal_paws = smal_data.get_marker_indices(*[p + " paw" for p in paw_labels])

	# smal_paws[1] = mocap_data.get_marker_indices("right front ankle")[0]

	# quickly plot 3d of each
	# mocap_all_data = norm_kin_data()

	mocap_all = norm_kin_data(mocap_data.all_data)[:, mocap_paws, 2]
	smal_all = norm_kin_data(smal_data.all_data)[:, smal_paws, 2]

	# get frame matching delay through estimation
	timed_fdelay = int(delay * smal_data.freq) # delay before frame matching shift
	test_fds = []
	frame_match_range = np.arange(-50, 50, 1)
	for i in frame_match_range:
		# roll smal data, and crop to desired time range, before element wise product with mocap data
		l = (smal_all[timed_fdelay-i:timed_fdelay-i+len(mocap_all), 1] * mocap_all[:,1]).sum()
		test_fds.append(l)

	frame_delay = frame_match_range[np.argmax(test_fds)]
	print(f"Selected Frame Delay: {frame_delay}")

	print_data = np.zeros((int(mocap_data.clip_length * mocap_data.freq), 10)) # t_mocap, t_smal, 4*mocap, 4*smal

	plots = []
	for n, data in enumerate([mocap_data, smal_data]):
		b_mapping, body_joints, *_ = data.generate_skeleton_mapping()
		all_data = norm_kin_data(data.all_data, targ_markers=np.ravel(body_joints))

		## crop to clip

		for j in range(4):
			ax = axes[j // 2, j % 2]
			paw_idx = [mocap_paws, smal_paws][n][j]
			joint_data = all_data[:, paw_idx, 2]
			t = np.arange(0, data.clip_length, 1/data.freq)
			if n ==0:
				t_mocap = t.copy()

			if n ==0: mocap_joint_data = joint_data # store for correcting
			if n == 1:
				t -= (delay - frame_delay / freq) # temporal shift to match data
				# crop to same time region as mocap
				f_start, f_end = int(delay*data.freq), int((delay+clip_length)*data.freq)
				joint_data = joint_data[f_start: f_end]
				t = t[f_start: f_end]
				t_smal = t.copy()

				# Correcting factor, subtract off height difference
				# joint_data -= (np.percentile(joint_data, 10) - np.percentile(mocap_joint_data,10))
				# ax = ax.twinx()

			plots.append(
				ax.plot(t, joint_data, color = colours[n], label = ["Mocap", "KDN"][n])
				[0])

			print_data[:len(joint_data), 2 + (4*n) + j] = joint_data

			# only consider mocap in ymax for now
			if n == 0: ymax = max(ymax, joint_data.max())

		# ax.set_xlim(t.min(), t.max()) # set xrange based off of mocap data ( smaller range )

	print_data[:len(t_mocap), 0] = t_mocap
	print_data[:len(t_smal), 1] = t_smal

	if print_vals:
		# convert to mm
		print_data[:, 2:] *= 1000
		print("Tmocap mocap1 mocap2 mocap3 mocap4")
		idxs = [0, 2, 3, 4, 5]
		for i in range(len(print_data)):
			print(" ".join([f"{v:.2f}" for v in print_data[i, idxs]]))

		print("--------")
		print("Tsmal smal1 smal2 smal3 smal4")
		idxs = [1, 6, 7, 8, 9]
		for i in range(len(print_data)):
			print(" ".join([f"{v:.2f}" for v in print_data[i, idxs]]))

	xtitle="Time (s)"
	ytitle = "Height (m)"

	[axes[j//2, j%2].set_title(paw_labels[j].title()) for j in range(4)]
	[ax.set_xlabel(xtitle) for ax in axes[-1, :]]
	[ax.set_ylabel(ytitle) for ax in axes[:, 0]]

	ax.set_ylim(0, ymax)

	fig.legend(["Motion capture", "KDN"], handles=[plots[0], plots[5]], ncol=2, loc = 8)
	plt.subplots_adjust(left=.1, right=.92, top=.93, bottom=.2, wspace = .19, hspace = .25)

	# plt.savefig(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\smal_results_displacement\set_2_3r4.png",
	#             dpi = 300)
	plt.show()

sine_curve = lambda x, *coeff: sum([c * np.sin(np.pi * x * (f+1)) for f, c in enumerate(coeff)])

def cosine_pulse(x, width, delta, s=0):
	"""Half cosine pulse. Pulses of width <width>, separated by <delta>. Starts at <s>.
	x is assumed to be uniform.
	Pulses can overlap!"""

	out = np.zeros_like(x).astype(np.float64)
	dx = x[1] - x[0] # assumes x is uniform!
	centres = np.arange(s, x[-1], delta)
	for p in centres:
		start = int(max(p-width/2, x[0]) / dx) # get start of pulse as idx
		end = int(min(p+width/2, x[-1]) / dx) # get end of pulse as idx

		crop_start = int(min(p-width/2, x[0]) / dx)
		crop_end = int(min(p+width/2, x[-1]) / dx)

		window_range = np.arange(end-start).astype(np.float64)
		if crop_start < x[0]: window_range += start-crop_start
		if crop_end > x[-1]: window_range += crop_end-end

		out[ start : end ] += np.sin(window_range * np.pi / (width/dx))

	return out

def square_pulse(x, width, delta, s = 0.0):
	"""Square wave pulse.
	Pulses are seperated by <delta>, have full width <width>.
	The centre of the first peak is at s."""

	out = np.zeros_like(x)
	centres = np.arange(s, x[-1], delta)

	for n in range(x.size):
		out[n] = np.any(np.abs((centres - x[n])) < (width/2))

	return out


class IDResults:
	"""Object for handling result of ID solution, comparing with forceplate data, and providing output"""

	def __init__(self, kin_src, dynamic_src, clip_length, mocap = True, mass = None, load = False):
		self.solver = solver = load_solver(kin_src, clip_length, mocap=mocap, resample_freq=sample_freq)
		self.dyn_data = get_dyn_data(dynamic_src, clip_length, mass, is_mocap=mocap,
									 target_freq=sample_freq)

		efd = 10

		#TODO add load functionality

		forces, torques = self.solver.solve_forces(save=False, end_frames_disregarded=efd, report_equations=False)

		self.forces = forces
		self.torques = torques

		### GET GRFS
		paw_joints = solver.foot_joints
		self.pred_grfs = forces[:, paw_joints, 2] / (solver.total_mass * g)

		self.data ={
			'preds': self.pred_grfs,
			'actual': self.dyn_data
		}

		targ = 2 # left front knee



		self.n_modes = 4

		self.calc_stance_data()
		# self.calc_modal_error()

	def view_knee_torque(self, j=0, print_vals=False):

		idx = {0: 2,
			   1: 7,
			   2: 22,
			   3: 18,} # left front knee

		fig, ax = plt.subplots(figsize=(8,4))

		ax.set_xlabel("Time (s)")
		ax.set_ylabel("Torque / Nm")
		ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: x / self.solver.freq))

		ax.plot(self.torques[:800, idx[j]]) # plot up to frame 800 for now

		if print_vals:
			trange = np.arange(0, clip_length, 1/self.solver.freq)
			for i in range(800):
				print(f"{trange[i]:.3f} {self.torques[i, idx[j]]:.3f}")

		plt.show()

	def plot_time_series(self, print_vals=False):
		# Estimate frame delay by maximising product of two series
		test_fd_range = np.arange(-50, 50, 1)
		test_fds = []
		for i in test_fd_range:
			l = ((np.roll(self.dyn_data, -i, axis=0) > 0) * self.pred_grfs).sum()
			test_fds.append(l)

		frame_delay = test_fd_range[np.argmax(test_fds)]
		print(f"Selected Frame Delay: {frame_delay}")

		fig, axes = plt.subplots(nrows=4, sharex="all", sharey="row")
		axes[-1].set_xlabel("Time (s)")
		[ax.set_ylabel("Norm Force\n$F/mg$ ") for ax in axes]
		[ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: x / self.solver.freq)) for ax in np.ravel(axes)]

		self.rms_errors = [] # compute rms error between two series for each paw

		n_frames = len(self.dyn_data)
		all_data = np.zeros((n_frames, 9)) # time, 4 pred, 4 actual
		all_data[:, 0] = np.arange(0, clip_length, 1/self.solver.freq)

		for i in range(4):
			ax = axes[i]
			ax.set_title(foot_joint_labels[i].title())
			pred = self.pred_grfs[:, i]
			act = np.roll(self.dyn_data[:, i], -frame_delay)
			ax.plot(pred, label="Predicted")
			ax.plot(act, label="Measured")
			ax.set_ylim(0, 1.0)
			ax.set_xlim(0, clip_length*self.solver.freq)

			error = ((pred - act)**2).mean()**.5
			self.rms_errors.append(error)

			all_data[:, 1 + i] = pred
			all_data[:, 5 + i] = act

		if print_vals:
			print("Time pred1 pred2 pred3 pred4 act1 act2 act3 act4")
			for i in range(n_frames):
				print(" ".join([f"{v:.2f}" for v in all_data[i]]))

		fig.legend(labels=["Predicted", "Actual"], ncol=2, loc=8)
		# ax = axes[2, 0]
		# ax.set_title("Overall")
		# ax.plot(self.pred_grfs[:].sum(axis=1))
		# ax.plot(self.dyn_data[:].sum(axis=1))
		# ax.set_ylim(0, 1.5)
		# axes[2, 1].axis("off")

		fig.subplots_adjust(wspace=.08, hspace=.29, left=.11, bottom=.17, right=.98, top=.95)

	def calc_stance_data(self):
		"""For each paw, produce data of:
		% stance against GRF/bw, for pred and actual"""

		self.stances ={
			'preds': {},
			'actual': {}
		}

		self.modes ={
			'preds': np.zeros((4, self.n_modes)),
			'actual': np.zeros((4, self.n_modes))
		} # coefficients of modal fit

		self.first_order_errors = [] # store first order errors
		self.first_order_error_pct = [] # store first order pct errors

		for data_stream in ['preds', 'actual']:

			data = self.data[data_stream]
			out_dict = self.stances[data_stream]
			modes = self.modes[data_stream]

			for paw in range(4):
				paw_data = data[:, paw]

				# GET DATA FOR EACH FOOTFALL
				# to include 'zeros' at start and end of footfall, check 0 condition for point before *or* after
				tol = 1e-3
				min_width = 30 # min frames width for footfall
				is_footfall = np.where((np.roll(paw_data, 1) > tol) + (np.roll(paw_data, -1) > tol))[0]
				footfalls = consecutive(is_footfall)  # list of indices for individual footfalls
				trange = np.arange(0, clip_length, 1 / self.solver.freq)

				all_x, all_y = [], []

				for ff in footfalls:
					if len(ff) < min_width: continue
					t = trange[ff]
					x = (t - t.min()) / (t.max() - t.min())  # format to % of stance
					all_x += list(x)
					all_y += list(paw_data[ff])
					# plt.plot(x, paw_data[ff])
					# plt.show()

				all_x = np.array(all_x)
				all_y = np.array(all_y)

				idx = all_x.argsort()
				all_x, all_y = all_x[idx], all_y[idx]

				all_y = np.array([y for _, y in sorted(zip(all_x, all_y))])  # sort y by x
				all_x = np.sort(all_x)

				out_dict[paw] = [all_x, all_y]

				# FIT TO SINE
				# NEW METHOD: FIT SINES TO CURVE. First mode is sin(pi x), as is 0 at x=0, x=1
				# print(all_x, all_y)
				coeff, cov = optimize.curve_fit(sine_curve, all_x, all_y, p0=[0.5] + [0] * (self.n_modes - 1))

				modes[paw] = coeff

		# Calculate errors
		for paw in range(4):
			coeff_preds = self.modes['preds']
			coeff_actual = self.modes['actual']
			foe = coeff_preds[paw, 0] - coeff_actual[paw, 0]
			self.first_order_errors.append(foe)
			self.first_order_error_pct.append(foe / coeff_actual[paw, 0])

	def calc_modal_error(self):
		"""Outputs (4 x n_freq) error, giving the % error between dyn and pred modal values"""
		out_loc = r"E:\IIB Project Data\produced data\dynamics data\modal data"

		out = ["paw mode error\n"]

		for paw in range(4):
			for m in range(self.n_modes):
				dyn, pred = self.modes['actual'][paw], self.modes['preds'][paw]
				error = abs(dyn[m] - pred[m] / dyn[m] )

				out.append(f"{paw} {m} {error:.3f}\n" )

		outfile = open(os.path.join(out_loc, self.solver.name + ".dat"), "w", newline="\n")
		outfile.writelines(out)

	def plot_footfall_means(self, print_symmetry=False):

		dyn_data = self.dyn_data
		pred = self.pred_grfs

		# PLOT GRF AS A FRAC OF STANCE, AND SPECTRAL DENSITY
		fig_stance, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all")
		fig_spectral, axes_sd = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="all")

		[ax.set_xlabel("$p$, \% Stance") for ax in axes[1]]
		[ax.set_ylabel("$F$, GRF / $mg$") for ax in axes[:, 0]]
		[ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: int(100 * x))) for ax in np.ravel(axes)]

		xrange = np.arange(0, 1, .01) # x stance range
		for paw in range(4):
			ax = axes[paw // 2, paw % 2]
			ax.set_title(foot_joint_labels[paw].title())

			# from pred grf, split by non-zero. Then, for each, define x stance %
			pred_grf = pred[:, paw]

			for n, data_stream in enumerate(['preds', 'actual']):

				stance_x, stance_y = self.stances[data_stream][paw]
				modes = self.modes[data_stream][paw]

				col = ["blue", "orange"][n]

				y_fit = sine_curve(xrange, *modes)

				ax.plot(xrange, y_fit, lw=2.5, alpha=1.0, color=col, label=["Predicted", "Actual"][n])
				# ax.scatter(stance_x, stance_y, alpha = .2)
				ax_sd = axes_sd[paw // 2, paw % 2]
				ax_sd.bar(list(range(1, self.n_modes + 1)), modes, alpha=.6)

				err = np.zeros_like(xrange)  # error as a function of stance, based on time average window
				window = .02
				for j, h in enumerate(xrange):
					err[j] = np.std(stance_y[(h - window / 2 < stance_x) & (stance_x <= h + window / 2)] - y_fit[j])

				# smooth error
				err = signal.savgol_filter(err, window_length=21, polyorder=2)

				ax.fill_between(xrange, y_fit + err, y_fit - err, color=col, alpha=.4)

			ax.set_xlim(0, 1)
			ax.set_ylim(bottom=0, top = 0.8)

		if print_symmetry:
			for stream in ["preds", "actual"]:
				Fz = self.modes[stream][:, 0]
				front = abs((Fz[0] - Fz[1]) / (Fz[0]+Fz[1]))
				rear = abs((Fz[2] - Fz[3]) / (Fz[2]+Fz[3]))
				print(stream, front, rear)

		fig_stance.subplots_adjust(bottom=.2, top = .95, left = .1)
		fig_stance.legend(labels= ["Predicted", "Actual", "Predicted Deviation", "Actual Deviation"], ncol=2, loc=8)

	def show_errors(self):
		"""Print all errors for presenting in report"""
		print("Name | foe | foe (%) | rms")
		for p, (rms, foe, foepct) in enumerate(zip(self.rms_errors, self.first_order_errors, self.first_order_error_pct)):
			name = foot_joint_labels[p]
			print(f"{name} | {foe:.2f}  | {100*foepct:.1f} | {rms:.2f}")

set_2_3r4_mocap = dict(
	kin_src = r"set_2/3 kph run 4.c3d",
	dynamic_src =r"set_2 -- 3kph run4",
	mocap = True,    mass = 25.7
)

set_2_6r1_mocap = dict(
	kin_src = r"set_2/6 kph run 1.c3d",
	dynamic_src =r"set_2 -- 6kph run1",
	mocap = True,    mass = 25.7
)


set_2_3r4_smal = dict(
	kin_src = r"set 2 3r4 - left",
	dynamic_src =r"set_2 -- 3kph run4",
	mocap = False,    mass = 25.7
)

lab = dict(
	kin_src = "zebris_lab_test",
	dynamic_src =r"set_3 -- lab",
	mocap = False,
	mass = 37.3 # approximate
)

gus = dict(
	kin_src = "zebris_gus_dynamic",
	dynamic_src=r"set_3 -- gus",
	mocap=False,
	mass = 34.3
)

collie = dict(
	kin_src = "zebris_collie",
	dynamic_src=r"set_3 -- border_collie",
	mocap=False,
	mass = 18.6
)

if __name__ == "__main__":
	selected_set = set_2_6r1_mocap
	r = IDResults(**selected_set, clip_length=clip_length)

	r.plot_time_series(print_vals=False)
	r.plot_footfall_means(print_symmetry=True)
	r.show_errors()
	plt.show()

	# compare_smal_to_mocap("set_2/3 kph run 4.c3d", "set 2 3r4 - pad test", print_vals=True)
	# anim_smal_and_mocap("set_2/3 kph run 4.c3d", "set 2 3r4")
	# estimate_mass(**collie)
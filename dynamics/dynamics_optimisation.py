"""Process for optimising certain parameters of the Dynamic model to better match data sources"""
from dynamics.dynamics import *
from data.data_loader import C3DData, load_force_plate_data, ForcePlateData

from tqdm import tqdm
from scipy import interpolate, optimize
import matplotlib as mpl

import torch
nn = torch.nn

plt.rc("text", usetex=True)

class ModelOptimiser(Model):
	"""Object used for optimising the ID model to given output force data.
	Minimises the RMS error between force data and predicted force data (both data normalised by weight of dog/dog model).
	Optimisation uses GD scheme.
	Optimisation params chosen in __init__.

	NOTE: requires kinematic and dynamic data to have the same frequency. Current implementation is to resample
	mocap at forceplate frequency"""

	def __init__(self, solver_kwargs, kin_data, dyn_data,
				 param_selection = "paws", dyn_format = "grfs", dyn_freq=100):
		nn.Module.__init__(self)
		Model.__init__(self)

		self.solver_kwargs = solver_kwargs

		self.kin_data = kin_data
		self.target = dyn_data

		self.dyn_format = dyn_format
		self.dyn_freq = 100

		## dict of param_name : value
		param_dict = {}
		if param_selection == "paws":
			param_dict = self.paw_params_normalised
			param_dict["frame_delay"] = 0 # number of frames dynamic data is *ahead* of kinematic data

		self.target_params = ["k_front", "k_rear", "L0_front", "L0_rear"]
		self.param_dict = param_dict

		self.losses = []

	def forward(self):
		"""Loss is the rms value of the product of the time series (element wise a * b)"""

		solver = InverseDynamicsSolver(joint_data = self.kin_data, **self.solver_kwargs, model=self)
		forces, torques = solver.solve_forces(save = False)

		if self.dyn_format == "grfs":
			paw_joints = self.solver_kwargs["foot_joints"]
			pred = forces[:, paw_joints, 2] / (solver.total_mass * g)

		f_delay = int(self.param_dict["frame_delay"])

		if f_delay > 0: targ, pred = self.target[f_delay:], pred[:-f_delay]
		elif f_delay < 0: targ, pred = self.target[:f_delay], pred[-f_delay:]
		else: targ, pred = self.target, pred


		F, N = pred.shape

		# conv_losses = [(np.convolve(targ[:,i], pred[:,i], mode = "same") **2).mean()**.5 for i in range(N)]
		# loss = sum(conv_losses)/len(conv_losses)


		loss = ((pred-targ) ** 2) .mean() ** .5

		return loss

	def get_gradient(self, params, delta = .01):
		"""Gets the gradient of the input param list, by evaluating each parameter increasing by .1%
		(where the parameter is very small, change by .01 instead)"""

		grads = {}
		loss = self.forward() # loss with no gradients applied

		for param in params:
			x = self.param_dict[param]
			if abs(x) < 1e-5:
				dx = .1 # May need refining
			else:
				dx = delta * x

			self.edit_paw_param(param, x+dx)

			loss_dx = self.forward()
			grads[param] = (loss_dx - loss) / dx
			# print(param, loss, loss_dx, dx, (loss_dx - loss) / dx)

			self.edit_paw_param(param, x+dx) # reset param

		return grads, loss

	def optimise(self, lr = 0.1, n_it = 20, sample=3):
		"""Run gradient descent"""

		loss = "?"
		losses = []
		progress = tqdm(total=n_it)
		param_keys = list(self.param_dict.keys())
		for i in range(n_it):
			# param_choice = np.random.choice(param_keys, sample, replace=False)
			param_choice = self.target_params
			progress.set_description(f"LOSS = {loss}. P = {param_choice}")
			grads, loss = self.get_gradient(param_choice)
			print(grads)
			for param, grad in grads.items():
				self.param_dict[param] -= lr * grad # update parameter
			self.calc_paw_params()

			progress.update()#, progress.set_description(f"LOSS = {loss}")
			losses.append(loss)

		print(*[(p, v) for p,v in self.param_dict.items()], sep="\n")

		self.losses = losses

	def single_variable(self, param):
		"""Analyses loss for various values of param. Saves for later review"""

		multipliers = [i/20 for i in range(40)] + [.95, .97, .98, .99, .995, .99, 1.01, 1.005, 1.01, 1.02, 1.03, 1.04, 1.05]+[1]
		# multipliers = [1,2]
		x = float(self.param_dict[param]) # store initial value
		vals, losses = [], []

		progress = tqdm(total=len(multipliers))
		for m in multipliers:
			self.param_dict[param] = m * x
			self.calc_paw_params()

			progress.update()

			loss = self.forward()

			losses.append(loss), vals.append(m*x)

		out = np.array([vals, losses])
		np.save(path_join(DataSources.dynamics_data, "param optimisation", param), out)
		plt.plot(vals, losses)
		plt.show()

	def view_single_var_optim(self):

		src = path_join(DataSources.dynamics_data, "param optimisation")
		for file in os.listdir(src):
			data = np.load(os.path.join(src, file))
			x, y = data
			plt.plot(x,y, label = file[:-4])

		plt.legend()
		plt.show()

def optimise_paws_to_c3d(mocap_src, dynamic_src = r"set_2 -- 3kph run3"):
	"""Given a .c3d file, and the resultant GRFs, optimise the paws"""

	freq_forceplate = 100
	clip_length = 10 # seconds

	# LOAD MOCAP DATA
	mocap_data = C3DData(ax=None, src=mocap_src, interpolate = True, crop = clip_length)
	mocap_data.resample_at(100) ### TRY RESAMPLING DATA TO 100 Hz
	target_bones, body_joints, no_torque_joints = mocap_data.generate_skeleton_mapping()

	# Normalise data based on z data, so that the dog is roughly 0.5m high. Also smooth data
	kin_data = np.array(mocap_data.all_data)
	kin_data = 0.635 * kin_data / np.amax(kin_data[:, :, 2])

	solver_kwargs = dict(target_bones = target_bones,
						 body_joints=body_joints, no_torque_joints=no_torque_joints,
						 foot_joints = no_torque_joints, freq=mocap_data.freq)

	# LOAD FORCEPLATE DATA
	force_plate_data, force_plate_tdelay = load_force_plate_data(dynamic_src)
	dyn_data = force_plate_data
	mass = 25.7 # normalise dynamic data by mass of dog
	dyn_data *= 1/(mass * g)
	frame_delay = 10 # number of frames dynamic data is *ahead* of kinematic daya

	# NEED TO CROP FOOTPLATE DATA TO CORRECT SIZE CLIP
	freq_mocap = mocap_data.freq
	frame_delay = int(freq_forceplate * force_plate_tdelay)
	n_frames_forceplate = int(mocap_data.n_frames * freq_forceplate/freq_mocap) # number of frames for forceplate to be same time length as mocap
	dyn_data = dyn_data[frame_delay: frame_delay + n_frames_forceplate] # crop forceplate data to match mocap/SMAL data

	optimiser = ModelOptimiser(solver_kwargs, kin_data, dyn_data, param_selection="paws", dyn_format = "grfs")
	# optimiser.single_variable("L0_front")
	# optimiser.single_variable("L0_rear")
	# optimiser.view_single_var_optim()
	# return None

	# optimiser.optimise(n_it = 20)


	#########
	### PLOTTING FOR TESTING###
	solver = InverseDynamicsSolver(joint_data=kin_data, **solver_kwargs, model=optimiser)
	# solver.view_ground_displacements(deriv = 0)
	# solver.view_com_displacements(deriv=0)

	### save dyn data and kin
	# src = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\inverse_dynamics\l0_characterisation"
	# np.save(path_join(src, "disps.npy"), np.swapaxes(solver.joint_pos[:, no_torque_joints, 2], 0, 1))
	# np.save(path_join(src, "unsmoothed_disps.npy"), np.swapaxes(solver.unsmoothed_data[:, no_torque_joints, 2], 0, 1))
	# np.save(path_join(src, "grfs.npy"), np.swapaxes(dyn_data, 0, 1))
	# print("saved")

	efd = 10
	forces, torques = solver.solve_forces(save=False, end_frames_disregarded=efd , report_equations=False)
	# print(optimiser.forward())

	# plt.plot(torques[:, :5])
	# plt.show()


	paw_joints = solver_kwargs["foot_joints"]
	pred = forces[:, paw_joints, 2] / (solver.total_mass * g)

	### PLOTTING TODO MOVE OVER TO FIGURES.PY

	fig, axes = plt.subplots(nrows=3, ncols=2, sharex="all", sharey="all")
	[ax.set_xlabel("Time (s)") for ax in axes[2]]
	[ax.set_ylabel("Norm Force \\ $F/mg$ ") for ax in axes[:, 0]]
	[ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: x/solver.freq)) for ax in np.ravel(axes)]

	for i in range(4):
		ax = axes[i//2, i%2]
		ax.set_title(foot_joint_labels[i].title())
		ax.plot(pred[:, i], label="Predicted")
		ax.plot(np.roll(dyn_data[:, i], 0*frame_delay), label="Measured")
	# ax.set_ylim(0, 1..0)


	ax.legend()

	ax = axes[2, 0]
	ax.set_title("Overall")
	ax.plot(pred[:].sum(axis=1))
	ax.plot(dyn_data[:].sum(axis=1))

	axes[2,1].axis("off")
	fig.subplots_adjust(wspace=.08, hspace=.29, left = .09, bottom=.09, right=.98, top=.95)

	# PLOT GRF AS A FRAC OF STANCE, AND SPECTRAL DENSITY
	fig, axes = plt.subplots(nrows=2, ncols = 2, sharex="all", sharey="all")
	fig, axes_sd = plt.subplots(nrows=2, ncols = 2, sharex="all", sharey="all")

	[ax.set_xlabel("\% Stance") for ax in axes[1]]
	[ax.set_ylabel("Normalised Force $F/mg$ ") for ax in axes[:, 0]]
	[ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: int(100*x))) for ax in np.ravel(axes)]

	for i in range(4):
		ax = axes[i // 2, i % 2]
		ax.set_title(foot_joint_labels[i].title())

		# from pred grf, split by non-zero. Then, for each, define x stance %
		pred_grf = pred[:, i]

		for n, data in enumerate([pred_grf, dyn_data[:, i]]):

			col = ["blue", "orange"][n]

			# Plot pred stance graphs
			footfalls = consecutive(np.where(data > 1e-5)[0]) # list of indices for individual footfalls
			trange = np.arange(0, clip_length, 1/solver.freq)
			all_x, all_y = [], []
			for footfall in footfalls:
				t = trange[footfall]
				if t.min() == t.max(): continue # skip blank footfalls
				x = (t - t.min()) / (t.max() - t.min()) # format to % of stance
				all_x += list(x)
				all_y += list(data[footfall])

			all_x = np.array(all_x)
			all_y = np.array(all_y)

			idx = all_x.argsort()
			all_x, all_y = all_x[idx], all_y[idx]

			all_y = np.array([y for _,y in sorted(zip(all_x, all_y))]) # sort y by x
			all_x = np.sort(all_x)

			# dx = .01
			# window = .1
			# X = np.arange(0, 1, dx)
			# Y = np.zeros_like(X)
			# for j, h in enumerate(X):
			#     y_inrange = all_y[(h-window/2 < all_x) & (all_x <= h+window/2)]
			#     Y[j] = y_inrange.mean()

			# NEW METHOD: FIT SINES TO CURVE. First mode is sin(pi x), as is 0 at x=0, x=1
			n_freq = 5
			sine_curve = lambda x, *coeff: sum([c * np.sin(np.pi * x * f) for f, c in enumerate(coeff)])
			coeff, cov = optimize.curve_fit(sine_curve, all_x, all_y, p0 = [.6] + [0] * (n_freq-1))

			xrange = np.arange(0, 1, .01)
			y_fit = sine_curve(xrange, *coeff)
			ax.plot(xrange, y_fit, lw=3, alpha=1.0, color=col, label=["Predicted", "Actual"][n])
			# ax.scatter(all_x, all_y, alpha = .2)
			ax_sd = axes_sd[i // 2, i % 2]
			ax_sd.bar(list(range(1, n_freq+1)), coeff, alpha = .6)

			err = np.zeros_like(xrange) # error as a function of stance, based on time average window
			window = .2
			for j, h in enumerate(xrange):
				err[j] = np.std(all_y[(h-window/2 < all_x) & (all_x <= h+window/2)] - y_fit[j])

			ax.fill_between(xrange, y_fit + err, y_fit - err,color=col, alpha = .6)

		# ax.plot(X, Y, lw=7, alpha=.6, color=col, label=["Predicted", "Actual"][n])
		# ax_sd = axes_sd[i // 2, i % 2]
		# ax_sd.plot(spec_density(Y))

		# LOOK INTO FFT - DOESN'T SEEM GREAT

		# old method of time averaging
		# def window(size): return np.ones(size)/float(size)
		# window_size = int(len(all_x) / 10)
		#
		# av_grf = np.convolve(all_y, window(window_size), "same") # convolve all_y to average them
		#
		# # TODO ATTEMPT TO EXTRACT SENSIBLE FFT FROM DATA
		# sampling_rate = 100 # sample every 10th value
		# all_x, av_grf = all_x[::sampling_rate], av_grf[::sampling_rate]
		#
		# ax.plot(all_x, av_grf, lw = 7, alpha=.6, color=col, label=["Predicted", "Actual"][n])
		#
		#

	fig.subplots_adjust(wspace=.08, hspace=.29, left = .09, bottom=.09, right=.98, top=.95)
	plt.legend()
	plt.show()
###########



# f = ForcePlateData(src=r"set_2/6kph run1")
# f.fps = 20
# f.plot_pressure(title="set2-6r1")

optimise_paws_to_c3d(r"set_2/3 kph run 4.c3d", dynamic_src = r"set_2 -- 3kph run4")

### CURRENT TASKS:
# IMPROVE MASS ESTIMATION
# TWO TASKS FOR TOMORROW MORNING:
# SEARCH FOR MORE DATA ON STIFFNESS & L0 FOR PAWS
# COME UP WITH METHOD FOR VALIDATING INTERNAL FORCES - GRFS ALMOST EXCLUSIVELY MEASURE ACCURACY OF PAW SPRING MODEL
# LOOK INTO FY, FX. SEE IF CONSISTENT WITH FRICTIONAL FORCE ON DOG?

# STILL WORK TO DO WITH ID.
# I THINK SOLN WILL COME FROM INTENSE SMOOTHING AND SIMPLIFICATION.

# TODO MAKE SURE ALL OF .C3D IS PERFECT
# TODO FIX ISSUE WHERE ~FRAME 800 OF FORCEPLATE DATA, BOTH FRONT PAWS ARE CONUNTED AS FRONT LEFT
# TODO FIX DYN DATA PREDICTIONS ON STANCE GRAPH
# TODO FIX PAW RECOGNITION - DOESN'T WORK FOR 6 KPH RUN!
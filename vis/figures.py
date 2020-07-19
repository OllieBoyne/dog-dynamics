"""Misc figures for report"""
import matplotlib as mpl
from dynamics.dynamics import *
from scipy import optimize
import os
joinp = os.path.join

src = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures"

plt.rc("text", usetex=True)

def l0_characterisation():
	"""Explore methods of identifying L0"""

	loc = r"inverse_dynamics\l0_characterisation"

	freq = 100
	frame_crop = 300
	fig, axes = plt.subplots(nrows=3, ncols=4, sharey="row", sharex=True)

	grfs = np.load(joinp(src, loc, "grfs.npy"))[:, :frame_crop]
	disps = np.load(joinp(src, loc, "disps.npy"))[:, :frame_crop]

	vline_kwargs = dict(ls="-", alpha=0.5)
	vline_cols = {"start": "green", "end": "red"}
	delay = 13  # frame delay

	for foot, (grf, disp) in enumerate(zip(grfs, disps)):

		if foot == 2:   grf[120:150] *= 0  # fixes small issue in data

		disp_smoothed = signal.savgol_filter(disp, window_length=49, polyorder=5)

		ax_disp, ax_accel, ax_grf = axes[:, foot]
		ax_disp.plot(disp, alpha=.6)
		ax_disp.plot(disp_smoothed)
		ax_grf.plot(grf)
		ax_disp.set_title(["Front left", "Front right", "Rear left", "Rear right"][foot])
		ax_grf.set_xlabel("Frame")

		starts = np.where((np.diff(grf) > 0) * (grf[:-1] == 0))[0]
		ends = np.where((np.diff(grf) < 0) * (grf[1:] == 0))[0]

		for s in starts:
			ax_grf.axvline(s, **vline_kwargs, color=vline_cols["start"])
			ax_accel.axvline(s + delay, **vline_kwargs, color=vline_cols["start"])
			ax_disp.axvline(s + delay, **vline_kwargs, color=vline_cols["start"])

		for e in ends:
			ax_grf.axvline(e, **vline_kwargs, color=vline_cols["end"])
			ax_accel.axvline(e + delay, **vline_kwargs, color=vline_cols["end"])
			ax_disp.axvline(e + delay, **vline_kwargs, color=vline_cols["end"])

		# CALC FROM MASS
		L0 = 6.9 * 25.7 * 1e-4
		ax_disp.axhline(L0)

		#####
		# METHOD TO IDENTIFY L0 VIA DERIVATIVES
		accel = nth_time_deriv(disp, 1 / freq, n=2)

		# Find all peaks above the RMS value
		peaks, _ = signal.find_peaks(disp, height=((disp ** 2).mean() ** .5, None))
		window_length = int(np.diff(peaks).mean()) // 2  # set window length as avg distance between peaks
		if window_length % 2 == 0: window_length -= 1  # make odd

		accel_smoothed = signal.savgol_filter(accel, window_length=window_length, polyorder=5)

		zero_crossings = np.where(np.diff(np.sign(accel_smoothed)))[0]
		## TODO REVIEW THIS:
		# TAKE THE CLOSEST 2 ZERO CROSSINGS TO EACH PEAK. THIS PREVENTS TURNING POINTS MID MOTION BEING CONSIDERED
		filt_zero_crossings = [zero_crossings[np.argpartition(np.abs(zero_crossings - peak), kth=2)[:2]] for peak in
							   peaks]
		filt_zero_crossings = np.ravel(filt_zero_crossings) + 1

		zero_crossings_disp = disp[filt_zero_crossings]
		L0_deriv = zero_crossings_disp.mean()
		ax_disp.axhline(L0_deriv, ls=":")
		#####

		accel = nth_time_deriv(disp_smoothed, 1 / 100, n=2)
		smoothed_accel = signal.savgol_filter(accel, window_length=49, polyorder=2)
		accel[:10] = smoothed_accel[:10] = np.nan  # discard first 10 frames
		ax_accel.plot(accel, color="orange", alpha=.6)
		ax_accel.plot(smoothed_accel, color="blue")

		ax_accel.axhline(0, ls="--")  # 0 line

		ax_disp.get_yaxis().set_major_formatter(
			mpl.ticker.FuncFormatter(lambda x, p: format(int(x * 1000))))

	plt.axvline(0, 0, 0, **vline_kwargs, color=vline_cols["start"], label="Starts")
	plt.axvline(0, 0, 0, **vline_kwargs, color=vline_cols["end"], label="Ends")

	plt.legend()
	axes[0, 0].set_ylabel("Vertical paw position (mm)")
	axes[1, 0].set_ylabel("Ground Reaction Force (N)")

	plt.subplots_adjust(left=.04, bottom=.05, right=.98, top=.97, wspace=.05, hspace=.07)
	plt.show()
	plt.savefig(joinp(src, loc, "characteristic_graphs.png"), dpi=300)


def paw_on_ground_identification():
	"""Identify paw off ground condition.
	Plot"""

	loc = r"inverse_dynamics\l0_characterisation"

	freq = 100
	frame_crop = 300
	fig, axes = plt.subplots(nrows=2, ncols=4, sharey="row", sharex=True)

	grfs = np.load(joinp(src, loc, "grfs.npy"))[:, :frame_crop]
	disps = np.load(joinp(src, loc, "unsmoothed_disps.npy"))[:, :frame_crop]

	vline_kwargs = dict(ls="-", alpha=0.5)
	vline_cols = {"start": "green", "end": "red"}
	delay = 10  # frame delay

	for foot, (grf, disp) in enumerate(zip(grfs, disps)):

		if foot == 2:   grf[120:150] *= 0  # fixes small issue in data

		disp_smoothed = signal.savgol_filter(disp, window_length=49, polyorder=5)

		ax_disp, ax_grf = axes[:, foot]
		ax_disp.plot(disp, "-o", ms=.5, alpha=.6)
		# ax_disp.plot(disp_smoothed)
		ax_grf.plot(grf)
		ax_disp.set_title(["Front left", "Front right", "Rear left", "Rear right"][foot])
		ax_grf.set_xlabel("Frame")

		starts = np.where((np.diff(grf) > 0) * (grf[:-1] == 0))[0]
		ends = np.where((np.diff(grf) < 0) * (grf[1:] == 0))[0]

		for s in starts:
			ax_grf.axvline(s, **vline_kwargs, color=vline_cols["start"])
			ax_disp.axvline(s + delay, **vline_kwargs, color=vline_cols["start"])

		for e in ends:
			ax_grf.axvline(e, **vline_kwargs, color=vline_cols["end"])
			ax_disp.axvline(e + delay, **vline_kwargs, color=vline_cols["end"])

		eps = np.zeros((len(disp)))  # will give eps - the displacement of the paw from equilibrium
		# for when the paw is in contact with the ground

		Z = disp

		# THIS SELECTS SUFFICIENTLY PROMINENT PEAKS, Above nth percentile BELOW Z RMS MEAN.
		contact_ends, _ = signal.find_peaks(Z, prominence=2e-3, height=(None, (Z ** 2).mean() ** .5))

		# for now, to avoid erroneous peaks, filter out 'peaks' signficantly below mean of peaks.
		# TODO REVIEW AND PRODUCE MORE ROBUST METHOD
		contact_ends = contact_ends[Z[contact_ends] > Z[contact_ends].mean() - .002]

		proms, *_ = signal.peak_prominences(Z, contact_ends)

		# IDENTIFY NORMAL PEAKS
		peaks, _ = signal.find_peaks(Z, prominence=10e-3, height=((Z ** 2).mean() ** .5, None))

		# FOR EACH PEAK, IDENTIFY L0, AND GET REGION BETWEEN START OF CONTACT AND END OF CONTACT
		for peak in peaks:

			i = 0
			future_ce = contact_ends[contact_ends > peak]
			if len(future_ce) == 0: break  # no future peaks
			trigger_height = Z[future_ce[0]]  # else, get next contact end height after this peak

			in_contact = False
			while peak + i < len(disp):
				if not in_contact and Z[peak + i + 1] <= trigger_height:
					in_contact = True
				if in_contact:
					## calculate eps. clip at 0 to prevent negative values from discretisation
					eps[peak + i] = max(trigger_height - disp[peak + i],
										1e-5)  # 1e-5 ensures marked as 'offground' for now
				if in_contact and Z[peak + i + 1] >= trigger_height:
					break  # end contact

				i += 1

		disp_contact = disp.copy()
		disp_contact[eps == 0] = np.nan
		ax_disp.plot(disp_contact, c="green")

		ax_disp.scatter(contact_ends, disp[contact_ends])

		ax_disp.get_yaxis().set_major_formatter(
			mpl.ticker.FuncFormatter(lambda x, p: format(int(x * 1000))))

		## plot eps on grf axis
		ax2 = ax_grf.twinx()
		ax2.plot(np.roll(eps, -delay), color="green")

	plt.axvline(0, 0, 0, **vline_kwargs, color=vline_cols["start"], label="Starts")
	plt.axvline(0, 0, 0, **vline_kwargs, color=vline_cols["end"], label="Ends")

	plt.legend()
	axes[0, 0].set_ylabel("Vertical paw position (mm)")
	axes[1, 0].set_ylabel("Ground Reaction Force (N)")

	plt.subplots_adjust(left=.06, bottom=.05, right=.98, top=.95, wspace=.05, hspace=.07)
	plt.show()
	plt.savefig(joinp(src, loc, "characteristic_graphs.png"), dpi=300)


def x_wise_forces():
	"""Graph used for comparing predicted net external X forces acting on a dog,  with the D'Alembert force
	Due to CoM movement."""

	mocap_src = r"set_2/3 kph run 3.c3d"
	dynamic_src = r"set_2 -- 3kph run3"

	clip_length = 5

	# LOAD MOCAP DATA
	mocap_data = C3DData(ax=None, src=mocap_src, interpolate=True, crop=clip_length)
	# mocap_data.resample_at(20) ### TRY RESAMPLING DATA TO 100 Hz
	target_bones, body_joints, no_torque_joints = mocap_data.generate_skeleton_mapping()

	# Normalise data based on z data, so that the dog is roughly 0.5m high. Also smooth data
	kin_data = np.array(mocap_data.all_data)
	kin_data = 0.635 * kin_data / np.amax(kin_data[:, :, 2])

	solver_kwargs = dict(target_bones=target_bones,
						 body_joints=body_joints, no_torque_joints=no_torque_joints,
						 foot_joints=no_torque_joints, freq=mocap_data.freq)

	#########
	### PLOTTING FOR TESTING###
	solver = InverseDynamicsSolver(joint_data=kin_data, **solver_kwargs)
	efd = 1
	forces, torques = solver.solve_forces(save=False, end_frames_disregarded=efd, report_equations=False)

	##### plot x grfs
	fig, ax = plt.subplots(figsize=(8, 4))

	paw_joints = solver_kwargs["foot_joints"]
	Fx = forces[:, paw_joints, 0].sum(axis=1)

	com = solver.get_com_position()[:, 0]
	com_smoothed = signal.savgol_filter(com, window_length=29, polyorder=5)

	def diff(y, dx):
		return np.diff(y) / dx

	dt = 1 / solver.freq

	com_force = diff(diff(com_smoothed, dt), dt) * solver.total_mass

	start = 10  # frame
	end = 450  # frame

	Fx[:start] = Fx[end:] = np.nan
	com_force[:start] = com_force[end:] = np.nan

	t1 = np.arange(Fx.size, dtype=np.float64) * 1 / solver.freq
	t2 = np.arange(com_force.size, dtype=np.float64) * 1 / solver.freq

	mg = solver.total_mass * g
	# ax.set_title("Front left paw")

	## OUTPUT TO CSV
	out_src = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\fig_data\x-wise-forces.csv"
	with open(out_src, "w", newline="") as outfile:
		out_data = [["Time", "Pred Fx", "Double Int"]]
		for t, f, c in zip(t2, Fx / mg, com_force / mg):
			out_data.append([t, f, c])
		w = csv.writer(outfile)
		w.writerows(out_data)

	ax.plot(t1, Fx / mg, color="red", label="Solver output $F_x$")
	ax.plot(t2, com_force / mg, color="green", label=r"Kinematic data $m\ddot x_{\small \textrm{COG}}$")
	ax.legend()

	ax.set_xlabel("Time, s")
	ax.set_ylabel("Normalised Force $F/mg$ ")

	plt.tight_layout()
	plt.savefig(joinp(src, "inverse_dynamics", "x_wise_forces 2-3r4"), dpi=300)


###########

def non_lin_spring():
	"""Fits data from paper: MECHANICAL PROPERTIES AND FUNCTION OF THE PAW PADS OF SOME MAMMALS (ALEXANDAR ET AL, 1986).
	TO NON LINEAR SPRING CHAR. Data is from Fig 7, from metacarpal of a foxhound"""

	# Pg 6 states that 'force/weight' for carpal limbs of foxhound describes force / (0.3 * full body weight).
	norm_force = np.array([0, 0.6, 1.6, 6, 3, 1])
	norm_force *= 0.3  # get force / body weight

	disp = np.array([0, 1, 2, 3, 2.5, 1.5])
	L0 = 16
	norm_disp = disp / L0

	func = lambda x, a, b: a * x ** 3 + b * x
	[a, b], _ = optimize.curve_fit(func, norm_disp, norm_force)

	print(a, b)
	print(func(.31, a, b))

	xrange = np.arange(0, .2, .01)

	plt.scatter(norm_disp, norm_force)
	plt.plot(xrange, func(xrange, a, b))
	plt.show()


def force_disp_char():
	"""Part of L0 characterisation. Plot graph of GRF against vert displacement from L0."""

	loc = r"inverse_dynamics\l0_characterisation"

	freq = 100
	frame_crop = 300
	fig, axes = plt.subplots(nrows=3, ncols=4, sharey="row", gridspec_kw={"width_ratios": [5, 1, 1, 1]})

	grfs = np.load(joinp(src, loc, "grfs.npy"))[:, :frame_crop]
	disps = np.load(joinp(src, loc, "disps.npy"))[:, :frame_crop]

	delay = 5  # frame delay

	for foot, (grf, disp) in enumerate(zip(grfs, disps)):
		ax_d, ax_f, ax_k = axes[:, foot]

		disp_smoothed = signal.savgol_filter(disp, window_length=49, polyorder=5)

		ax = axes[foot]

		L0 = 6.9 * 25.7 * 1e-4

		ax_f.plot(grf)
		ax_d.plot(disp_smoothed)
		# ax_k.plot(grf/disp_smoothed)
		idxs = (disp_smoothed < L0) * (grf > 0)
		ax_k.scatter((L0 - disp_smoothed[idxs]), grf[idxs], color="blue")
		break

	plt.show()


if __name__ == "__main__":
	# l0_characterisation()
	# paw_on_ground_identification()
	x_wise_forces()
	# non_lin_spring()
	# force_disp_char()

from dynamics.dynamics import *

fig_dir = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\inverse_dynamics\unit_tests"
fig_dpi = 300

plt.rc("text", usetex=True)

# ALL DATA HERE WILL BE IN FORMAT [RFRONT PAW, RFRONT ANKLE, RFRONT KNEE, RFRONT SHOULDER, LFRONT, ..., RREAR, ..., LREAR]
n_joints = 16 # 4 per leg

target_bones = {}
for m, side in enumerate(["right", "left"]):
	for n, end in enumerate(["front", "rear"]):
		for o, bone in enumerate(["bottom", "middle", "top"]):
			target_bones[f"{side} {end} {bone}"] = [4*m + 8*n + o, 4*m + 8*n + o + 1] # Access correct joint indices

body_joints = [[3, 7], [11, 15]]
foot_joints = [0, 4, 8, 12]

### TRY STATIC TEST FOR SINGLE LEG BONE
n_joints = 8
target_bones = {"right front": [0,1], "left front": [2,3], "right rear": [4,5], "left rear": [6,7]}
body_joints = [[1, 3], [5, 7]]
foot_joints = [0, 2, 4, 6]

def blank_dog():
	"""Set up (16, 3) array of dog with initial joint positions"""
	length = 0.5
	width = 0.2
	ankle_length = 0.1
	ankle_to_knee = 0.2
	knee_to_shoulder = 0.05

	O = Vector(0,0,0) # origin

	out = []
	for lengthwise in [-1, +1]:
		for widthwise in [+1, -1]:
			foot = O + length * Vector(lengthwise/2,0,0) + width * Vector(0, widthwise/2, 0)
			ankle = foot + ankle_length * Vector(-0.3, 0, 1).unit()
			knee = ankle + ankle_to_knee * Vector(-0.1, 0, 1).unit()
			shoulder = knee + knee_to_shoulder * Vector(0.05,0,1).unit()

			if n_joints == 16: out += [foot, ankle, knee, shoulder]
			elif n_joints == 8: out += [foot, shoulder]

	return np.array(out)

def plot_data_frame(frame_data):
	"""Plot skeleton for target bones"""

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	for b, (i, j) in target_bones.items():
		ax.plot(*zip(*frame_data[[i,j]]), label=b)

		x, y, z = frame_data[1]
		plt.plot([x,x], [y,y], [z, z - 0.2])

	ax.legend()
	plt.show()

def static_test():
	"""Uniform stand"""

	n_frames = 100
	freq = 100

	data = np.zeros((n_frames, n_joints, 3))
	init = blank_dog()  # blank dog

	for n in range(n_frames): data[n] += init

	#Fix L0, k
	k_per_kg = 3.4e3
	L_per_kg = 6.84e-4

	d = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq=freq,
							  no_torque_joints=foot_joints, foot_joints=foot_joints,
							  name="static_test",
							  k_paws_calculations=lambda mass: {"front": k_per_kg * mass, "rear": k_per_kg * mass},
							  L0_calculations=lambda mass: {"front": L_per_kg * mass, "rear": L_per_kg * mass})

	# READJUST SO RESTING ON SPRINGS
	m = d.total_mass
	h_static = L_per_kg * m - m * g / (4 * k_per_kg * m) # height at which springs would support weight

	data[:, :, 2] += h_static
	d.joint_pos = data

	print(f"TOTAL MASS {round(d.total_mass, 1)} kg")
	d.solve_forces(report_equations=False)

def uniform_fall():
	"""Uniform fall onto floor"""

	n_frames = 100
	freq = 100

	data = np.zeros((n_frames, n_joints, 3))
	init = blank_dog() # blank dog

	# Add height h, subtract h for each t
	h = 1

	x, y =[], []
	for n, frame in enumerate(data):
		t = n / freq
		z = 0 # z = max((h - 0.5 * g * t ** 2),0)
		x.append(t), y.append(z)
		data[n] = init + np.ones_like(init) * z # fall, stop at ground

	d = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq = freq,
							  no_torque_joints = foot_joints, foot_joints = foot_joints,
							  name = "uniform_fall_test")

	# GET L0S
	L0s = {"front": 6.84e-4 * d.total_mass, "rear": 6.91e-4 * d.total_mass}

	print(f"TOTAL MASS {round(d.total_mass,1)} kg")
	d.solve_forces(report_equations=True
				   )

def pure_rotation():

	n_frames = 100
	freq = 100

	data = np.zeros((n_frames, n_joints, 3))
	init = blank_dog()  # blank dog

	for n in range(n_frames): data[n] += init

	#Fix L0, k
	k_per_kg = 3.4e3
	L_per_kg = 6.84e-4

	d = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq=freq,
							  no_torque_joints=foot_joints, foot_joints=foot_joints,
							  name="pure_rotation_test",)

	alpha = 1 # angular acceleration
	for i in range(n_frames):
		t = i / freq
		theta = - 0.5 * alpha * t ** 2
		# Rotate right front paw about right front ankle, rotation about the y axis
		O = data[i, 1] # rotate about

		R = np.array([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-np.sin(theta), 0, np.cos(theta)]])
		data[i, 0] = (R @ (data[i, 0] - O)) + O # translate to rotate about origin, rotate, and then translate back

	# READJUST SO RESTING ON SPRINGS (only resting on 3 out of the 4 springs)
	m = d.total_mass
	h_static = L_per_kg * m - m * g / (3 * k_per_kg * m) # height at which springs would support weight
	data[:, :, 2] += h_static
	data[:, :2, 2] += L_per_kg * m # lift right foot up above spring so free to rotate
	d.joint_pos = data

	bone = [b for b in d.target_bones if "right front bottom" == b.name or "right front" == b.name][0]
	print(bone.name)
	print(f"Iyy = {round(bone.I[1,1], 8)} m-4") # Get bone, print I
	print(f"l = {bone.length}, m = {bone.mass}")
	d.solve_forces(report_equations=False, end_frames_disregarded=2)

def paw_oscillation():
	"""Identifies the bheaviour of the dog oscillating about the equilibrium point"""
	n_frames = 200
	freq = 200

	data = np.zeros((n_frames, n_joints, 3))
	init = blank_dog()

	for n in range(n_frames): data[n] += init

	#Fix L0, k
	k_per_kg = 3.4e3
	L_per_kg = 6.84e-4
	c_per_kg = 5

	d = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq=freq,
							  no_torque_joints=foot_joints, foot_joints=foot_joints,
							  name="oscillation_test")

	# READJUST SO RESTING ON SPRINGS
	m = d.total_mass
	h_static = L_per_kg * m - m * g / (4 * k_per_kg * m) # height at which springs would support weight

	data[:, :, 2] += h_static # fix to h static
	## OSCILLATE ACCORDING TO IMPULSE RESPONSE
	k = k_per_kg * m
	L = L_per_kg * m
	c = c_per_kg * m
	alpha = 0.5 * m * g # say

	zeta = c * (k*m)**(-.5)
	psi = np.arcsin(zeta)
	wn = 2 * (k/m)**0.5
	wd = wn * (1-zeta**2)**.5
	X = alpha / (4 * k)
	print(L, h_static)

	zs =[]
	for i, t in enumerate(np.arange(0, n_frames/freq, 1/freq)):
		z =  X * (1 - np.exp(- zeta * wn * t) * np.cos(wd * t - psi)/np.cos(psi))
		zs.append(h_static + z)
		data[i, :, 2] += z

	plt.plot(zs)
	plt.show()

	d.joint_pos = data
	print(f"TOTAL MASS {round(d.total_mass, 1)} kg")

	d.solve_forces()

def plot_pure_rotation():
	"""Plot the forces and torques predicted for pure rotation by the ID solver, as well as theoretical values"""
	plot_dict = view_one_joint("pure_rotation_test", joint=1)

	ax_force, ax_torque = plot_dict["ax_force"], plot_dict["ax_torque"]

	t = np.arange(0, 1, 0.01)
	alpha = 1
	theta_0 = 1.7038464928472865
	theta = - (theta_0 + 0.5 * alpha * t ** 2)
	omega = - alpha * t

	# physical params
	Iyy = 0.02338183
	l = 0.3478015 / 2
	m = 2.2421197

	# ANALYTICAL SOLUTION - dynamics
	T = m * g * l * np.cos(theta) - (Iyy + m * l**2) * alpha
	V = m * (-g + l * alpha * np.cos(theta) + omega ** 2 * l * np.sin(theta))
	H = m * (omega ** 2 * l * np.cos(theta) - alpha * l * np.sin(theta))

	torque_plot = ax_torque.plot(T, "--")
	h = ax_force.plot(- H, "--")
	v = ax_force.plot(V, "--") # negative as on second joint

	ax_force.legend(plot_dict["lines_force"] + h + v, ["$F_{1,x}$", "$F_{1,y}$", "$F_{1,z}$", "$H$", "$V$"])
	ax_torque.legend(plot_dict["lines_torque"] + torque_plot, ["$T_1$", "$T$"])

	ax_force.set_xlabel("Frame"), ax_force.set_ylabel("Force (N)")
	ax_torque.set_xlabel("Frame"), ax_torque.set_ylabel("Torque (Nm)")

	plt.tight_layout()
	plt.show()
	# plt.savefig(os.path.join(fig_dir, "pure_rotation.png"), dpi = fig_dpi)

def view_grfs(test="uniform_fall_test", forces=True, end_frames_disregarded=5):

	src = os.path.join(DataSources.dynamics_data, test)
	forces, torques = [np.load(os.path.join(src, f"{x}.npy")) for x in ["forces", "torques"]]
	fig, axes = plt.subplots(nrows = 3, ncols=2, sharey="row", sharex = "col")

	mass = 43.9

	for m, end in enumerate(["front", "rear"]):
		for n, side in enumerate(["right", "left"]):
			ax = axes[m, n]
			l = ax.plot(forces[end_frames_disregarded:-end_frames_disregarded-1,n_joints//2*m + n_joints//4*n] / (mass * g))
			# l = ax.plot(torques[:, 4 * m + 2 * n] / (mass * g))
			ax.set_title(f"{end} {side}".title())
			ax.legend(l, ["$F_x$", "$F_y$", "$F_z$"])

	# titles
	axes[2,0].set_title("Total")
	for ax in axes[:,0]: ax.set_ylabel("Force / Mass")
	axes[2,0].set_xlabel("Frame")

	total_grf = forces[end_frames_disregarded:-end_frames_disregarded, [n_joints//2 * m + n_joints//4 * n for n in range(2) for m in range(2)]].sum(axis=1)
	# total_grf = torques[:, [4 * m + 2 * n for n in range(2) for m in range(2)]].sum(axis=1)
	l = axes[2,0].plot(total_grf/ (mass * g))
	axes[2,0].legend(l, ["$F_x$", "$F_y$", "$F_z$"])

	axes[2,1].axis("off")

	plt.tight_layout()
	plt.savefig(os.path.join(fig_dir, "static.png"), dpi = fig_dpi)

def view_one_joint(fig, test="pure_rotation_test", joint = 1, end_frames_disregarded=2):
	"""Returns dict of pyplot objects for plotting. Including
	force_ax, torque_ax, force_lines, torque_lines"""

	src = os.path.join(DataSources.dynamics_data, test)
	forces, torques = [np.load(os.path.join(src, f"{x}.npy")) for x in ["forces", "torques"]]
	fig, (ax_forces, ax_torques) = plt.subplots(ncols=2)

	# H = (forces[end_frames_disregarded:-end_frames_disregarded, joint, 0])
	# V = (forces[end_frames_disregarded:-end_frames_disregarded, joint, 2])
	# T = torques[end_frames_disregarded:-end_frames_disregarded, joint]
	# t = np.arange(H.size) / 100 # 1 to 100 time
	# for datrow in zip(t, H, V, T):
	#     print(" ".join(map(lambda i: f"{i:.3f}", datrow)))

	l_forces = ax_forces.plot(forces[end_frames_disregarded:-end_frames_disregarded, joint])
	l_torques = ax_torques.plot(torques[end_frames_disregarded:-end_frames_disregarded, joint])# / 0.00076152)

	return dict(fig=fig, ax_force = ax_forces, ax_torque = ax_torques, lines_force = l_forces, lines_torque = l_torques)

# uniform_fall()
# static_test()
# pure_rotation()

# plot_pure_rotation()

from tqdm import tqdm
def get_cov(Np = 50, freq = 60, sigma_x = 1e-3):
	"""As detailed in inverse_dynamics report, perturbates the position data to analyse the matrix of covariance
	of the various equations in the inverse dynamics process.

	sigma_x = deviation of position vectors (m)"""

	# Nf = number of frames, Np = number of perturbations per frame, m = number of equations

	efd = 5 # end frames disregarded

	clip_length = 2
	mocap_data = C3DData(ax=None, src=r"set_2/3 kph run 4.c3d", interpolate = True, crop = clip_length)

	Nf = mocap_data.n_frames - 2 * efd
	target_bones, body_joints, no_torque_joints = mocap_data.generate_skeleton_mapping()
	data = mocap_data.all_data
	bbar = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq=freq,
								 no_torque_joints=foot_joints, foot_joints=foot_joints,
								 name="COV").return_equations() # bbar has shape (Nf x m)

	b = [] # will have shape (Nf x Np x m)

	progress = tqdm(total=Np)
	progress.set_description("Rendering perturbations")
	for l in range(Np):
		perts = np.random.normal(scale=sigma_x, size = data.shape) # perturbations
		data = data + perts

		p_b = InverseDynamicsSolver(joint_data=data, target_bones=target_bones, body_joints=body_joints, freq=freq,
									no_torque_joints=foot_joints, foot_joints=foot_joints,
									name="COV").return_equations()

		b.append(p_b)
		progress.update()

	b = np.swapaxes(b, 0, 1) # swap axes of b so correct shape

	n_equations = b.shape[-1]

	COV = np.zeros((n_equations, n_equations))

	for i in range(n_equations):
		for j in range(n_equations):
			COV[j, i] = sum((b[k, l, i] - bbar[k, i])*(b[k, l, j] - bbar[k, j]) for l in range(Np) for k in range(Nf)) / (Np * Nf)

	S = np.linalg.cholesky(COV)
	W = np.linalg.inv(S)

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	print(*[", ".join(map(str, W[i])) for i in range(n_equations)], sep="\n")

	# print("COV = ")
	np.save(r"E:\IIB Project Data\produced data\dynamics data\cov\out", W)


if __name__ == "__main__":

	# view_grfs("static_test")
	# pure_rotation()
	plot_pure_rotation()
	# paw_oscillation()
	# view_grfs("oscillation_test", forces=True)
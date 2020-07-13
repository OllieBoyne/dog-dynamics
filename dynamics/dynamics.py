"""DEFINES THE INVERSEDYNAMICS SOLVER, A Solver for solving the joint based model of a dog."""

from scipy import optimize, signal

from data.data_loader import C3DData, load_force_plate_data, ForcePlateData, SMALData, get_delay_between, DataSources, \
	path_join
from vis.utils import *
from vis import visualisations
from dynamics.footfall_detector import FootfallDetector
from tqdm import tqdm

# pure constants (no optimisation needed)
g = 9.81
freq_forceplate = 100  # Hz
foot_joint_labels = ["front left", "front right", "rear left", "rear right"]
foot_joint_indices = [0, 9, 23, 20]  # for set 2 3r3


class Model:
	"""ID Model, with all parameters derived/optimised"""

	def __init__(self):
		# CONSTANTS
		self.paws = {}
		self.bone_density = 1950  # Estimate - needs refining! From paper: Development of a neuromusculoskeletal computer model in a chondrodystrophic dog.
		self.muscle_density = 1060  # From above

		# params to optimise
		self.bone_length_definitions = {
			"normal": lambda l: dict(inner_radius=0.01, outer_radius=0.05, displacement=0),
			"body": lambda l: dict(inner_radius=l / 20, outer_radius=l / 7, displacement=l / 4 - l / 20), }

		# Paw parameters. All scaled to be in standard form - exponent in separate dict.
		self.paw_params_normalised = {
			"L0_front": 6.9,  # 6.9 # in .1mm
			"L0_rear": 6.9,  # in .1mm
			"k_front": 3.42 * .18,  # in kN/m
			"k_rear": 2.0 * .21,  # in kN/m
			"c_front": 20,
			"c_rear": 20,
			"k_rear_prop": 0.85,  # k = k_rear * m **.85
			"frame_delay": 0  # Used for analysis of paw treadmill forces. Not used for normal ID solver
		}
		self.paw_exponents = {
			"L0_front": -4,
			"L0_rear": -4,
			"k_front": 3,
			"k_rear": 3,
			"c_front": 0,
			"c_rear": 0,
			"k_rear_prop": 0,
			"frame_delay": 0
		}

		self.calc_paw_params()

		self.freq_par_data = 200

		# weightings used in dynamics calculations
		self.equation_weighting = {
			"Inertial": 2,
			"Rotational": 2,
			"Leg spring": 0.5,
			"Paw spring": 1,
		}

	def calc_paw_params(self):
		"""Calculates paw parameters (separate function for optimisation purposes)"""
		for param, val in self.paw_params_normalised.items():
			self.paws[param] = val * 10 ** (self.paw_exponents[param])

	def edit_paw_param(self, param, val):
		"""Edit paw parameter (separate for optimisation purposes)"""
		self.paw_params_normalised[param] = val
		self.calc_paw_params()


model = Model()


def time_deriv(X, dt):
	"""Finds the time derivative of a given series of data.
	Always treats the first dimension as time - works for any number of dimensions (n_frames, M, N, O, ...).
	For all except first and last val, calcs difference over 2 timesteps"""

	diff = np.zeros_like(X)

	diff[0] = X[1] - X[0]
	diff[1:-1] = (X[2:] - X[:-2]) / 2
	diff[-1] = X[-1] - X[-2]

	return diff * 1 / dt


def nth_time_deriv(X, dt, n=2):
	"""Recursively get the nth time derivative"""

	if n == 1:
		return time_deriv(X, dt)
	else:
		return time_deriv(nth_time_deriv(X, dt, n=n - 1), dt)


def get_principal_axes(vector=Vector(1, 0, 0), cardinal=np.identity(3)):
	"""Given a vector, devise a basis of principle axis with any two perpendicular vectors (for application of an
	axisymmetric object - cylinder) """

	i, j, k = cardinal
	K = vector.unit()
	# Now find any two perp vectors to K
	if not K.is_parallel(i):
		I = K.cross(i).unit()
		J = K.cross(I).unit()
	else:
		I = K.cross(j).unit()
		J = K.cross(I).unit()

	return np.array([I, J, K])


def I_cylinder(density, length, radius):
	mass = density * np.pi * (radius ** 2) * length
	Ixx, Izz = (length ** 2) / 12 + (radius ** 2) / 4, radius ** 2 / 2
	return mass * np.diag([Ixx, Ixx, Izz])


class DoubleCylinder:
	"""An object comprised of a cylinder of given length between two end points, of radius inner_radius and density bone_density,
	and an outer cylinder that does NOT share the same central axis, of radius outer_radius, displaced by a distance <displacement> normally from the centerline.

	Cylinder is defined with the centerline vertical (z direction), and the displacement always in the normal closest to the z direction downwards.

	For InverseDynamics calculations, this object will have a start and end index, which correspond to the joint indices in which the end point data is held.
	"""

	def __init__(self, start, end, length, inner_radius, outer_radius, displacement, freq=50.0, name=""):

		self.name = name
		self.freq = freq  # Frequency, in Hz

		self.start = start
		self.end = end

		self.length = length
		self.displacement = displacement

		if outer_radius is None: outer_radius = inner_radius

		self.inner_mass = model.bone_density * np.pi * inner_radius ** 2 * self.length
		self.outer_mass = model.muscle_density * np.pi * self.length * (outer_radius ** 2 - inner_radius ** 2)
		self.mass = self.inner_mass + self.outer_mass

		I_bone = I_cylinder(model.bone_density, length, inner_radius)
		I_muscle = I_cylinder(model.muscle_density, length, outer_radius) - I_cylinder(model.muscle_density, length,
																					   inner_radius)

		# By parallel axis theorem, add component of I due to outer radius being displaced from the centerline axis
		I_axis_displacement = np.zeros((3, 3))
		I_axis_displacement[0, 0] = self.outer_mass * displacement ** 2

		self.I = I_bone + I_muscle + I_axis_displacement  # Inertia tensor in a reference frame in which the bone is lengthwise facing upwards

	def get_kinematics(self, data):
		"""Given a numpy array of time, data, of shape (n_frames, 2, 3),
		giving the position data of both ends of the cylinder over time, compute the kinematics of the cylinder"""

		X = self.X = np.array(data)  # positions
		V = self.V = time_deriv(X, 1 / self.freq)  # velocities
		A = self.A = time_deriv(V, 1 / self.freq)  # accelerations

		self.XG = np.mean(X, axis=1)  # average over X
		self.VG = np.mean(V, axis=1)  # average over V
		self.AG = np.mean(A, axis=1)  # average over A

		# Rotational
		R = self.R = [Vector(*x[1]) - Vector(*x[0]) for x in X]  # Vector from bone start to end in each frame
		local_axes = [get_principal_axes(r) for r in R]  # Get principal axes for each frame

		# theta_g = (n_frame, 3) of angular rotation about i, j, k for each frame
		# angular rotation about each axis is defined as 0 for the next vector in the cycle
		# i.e. angular rotation about i = 0 for a vector parallel to j
		zero_angles = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # definition of 'zero angle' vector for i, j, k
		theta_g = []

		# Compute theta_g in local axes first, where K is the unit vector
		for n_frame in range(len(X) - 1):
			local_ax = local_axes[n_frame]
			# representation as a a single rotation theta about an axis e (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
			a = R[n_frame]  # rotation from one frame...
			b = R[n_frame + 1]  # ...to the next
			if np.array_equal(a, b):
				theta_g += [[0, 0, 0]]  # If no rotation, return 0
			else:
				axis = np.cross(a, b) / (np.linalg.norm(np.cross(a, b)))  # unit vector of omega
				with np.errstate(invalid='raise'):
					try:
						alignment = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
						alignment = np.clip(alignment, a_min=-1,
											a_max=1)  # clip between -1 and 1 to deal with rounding errors
						angle = np.arccos(alignment)  # magnitude of theta
					except:
						print((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
						raise ValueError("INVALID ANGLE", a, b)

				theta_g += [axis * angle]

		theta_g = np.array(theta_g)
		self.theta_g = signal.savgol_filter(theta_g, window_length=19, polyorder=2, axis=0)
		self.omega_g = time_deriv(self.theta_g, dt=1 / self.freq)
		self.alpha_g = time_deriv(self.omega_g, dt=1 / self.freq)  # angular acceleration

		self.I_fixed = [la.T @ self.I @ la for la in local_axes]  # compute I in fixed reference frame at each frame

	def get_dynamics(self):
		"""Compute dynamics (F_net, Torque_net) at each frame"""
		self.F_net = [self.mass * a_g for a_g in self.AG]
		self.tau_net = [I_f @ alpha for (I_f, alpha) in zip(self.I_fixed, self.alpha_g)]


class Body(DoubleCylinder):
	"""A unique case of double cylinder, where the (multiple) joints connect at the cylindrical surface at either end.
	These joint attachments are defined by an angle from the i direction normal to the centerline at initialisation.

	Dynamics for the body then must be calculated using a separate set of equations. Define the body such that all
	joints bones go into it rather than out of it (i.e. all input forces are positive on the body) """

	def __init__(self, start_joints, end_joints, all_joint_positions, **cylinder_kwaargs):
		"""From the indices given by start_joints and end_joints, identify a cylinder shape that best fits these
		points on either side, and create that as the cylinder. """

		self.start_joints = start_joints
		self.end_joints = end_joints

		start_pos = Vector(*np.mean(all_joint_positions[40, start_joints], axis=0))
		end_pos = Vector(*np.mean(all_joint_positions[40, end_joints], axis=0))

		length = start_pos > end_pos

		super().__init__(start=None, end=None, length=length, **model.bone_length_definitions["body"](length),
						 **cylinder_kwaargs)

	def get_centre_of_gravity(self, start: 'Vector', end: 'Vector'):
		"""Calculates the centre of gravity based on the displacement from the centerline."""
		centreline_g = 0.5 * (start + end)

		# to find a normal that is closest to z, find N possible equispaced normals, and see which one has the greatest .k product
		normal = (start - end).find_normal()
		N = 20  # number of normals to consider
		all_normals = [normal.rotate_about((start - end).unit(), angle=(n * 2 * np.pi / N)) for n in range(N)]

		idx = np.argmax([v.dot(Vector(0, 0, -1)) for v in all_normals])
		chosen_normal = all_normals[idx]  # choose most downwards normal

		return centreline_g + self.displacement * chosen_normal

	def get_kinematics(self, data):
		"""For body, data is of shape (n_frames, 2, 2, 3), where it is split by rear and front.
		So average across rear and front to get behaviour of centerline, and then run normal get_kinematics"""
		super().get_kinematics(np.mean(data, axis=2))


def weighted_bound_least_squares(A, b, weights=None, bounds=None, **kwargs):
	"""Completes a least squares solve of the equation A x = b, to solve N unknowns from M equations
	where A is an M x N matrix, x is an N x 1 vector, and b is an M x 1 vector.
	Applies weightings to each row to favour certain datapoints. weights is an M x 1 vector.

	Applies bounds where bounds is an M x 2 array. each tuple in the array gives the LB and UB for the given equation"""

	if weights is None: weights = np.ones_like(b)  # If no weight given, equal weight to all
	w = np.array(weights)

	weighted_A, weighted_b = np.array(A) * w[:, np.newaxis], np.array(b) * w  # Apply weights to A, b

	try:
		solve = optimize.lsq_linear(weighted_A, weighted_b, bounds=list(zip(*bounds)), tol=1e-2)
		return solve["x"]

	except np.linalg.LinAlgError as e:
		out = f"SVD did not converge in Lin Least Sq. Printing params: {A}, {b}"
		raise ArithmeticError(out)


class InverseDynamicsSolver:
	"""Through scipy optimisation, Skeleton finds a set of force data that corresponds to the correct kinematic data.
	Takes a skeleton, and the relevant bones and joints, and solves the set of forces that correspond to correct kinematics."""

	def __init__(self, joint_data, target_bones, body_joints, no_torque_joints=None, no_reaction_joints=None,
		foot_joints=None, leg_spring_joints=None, model=Model(),
		freq=50.0, name="output", is_mocap=True):

		for var in [foot_joints, leg_spring_joints, no_reaction_joints, no_torque_joints]:
			if var is None:
				var = []

		self.name = name
		self.freq = freq
		self.n_frames, self.n_joints, _ = joint_data.shape

		self.model = model
		self.is_mocap = is_mocap

		# Preprocess joint data - basic smoothing
		if is_mocap:
			window_length = self.freq // 2
		else:
			window_length = 0.75 * self.freq

		if window_length % 2 == 0: window_length -= 1

		self.T = self.n_frames / self.freq

		self.smooth = lambda X, p=5: signal.savgol_filter(X, window_length=int(window_length), polyorder=p, axis=0)

		p = 5 if self.is_mocap else 2

		self.unsmoothed_data = joint_data  # save unsmoothed data for other uses
		self.joint_pos = self.smooth(joint_data, p=p)
		self.joint_vel = time_deriv(self.joint_pos, 1 / freq)
		self.joint_accel = time_deriv(self.smooth(self.joint_vel), 1 / freq)

		self.foot_joints = foot_joints
		self.body_joints = body_joints
		self.get_foot_joint_from_index = {}  # Identify which foot from the index
		for fj in self.foot_joints:
			for bone, (j1, j2) in target_bones.items():
				if fj in [j1, j2]:
					self.get_foot_joint_from_index[fj] = bone

		self.no_torque_joints = no_torque_joints
		self.no_reaction_joints = no_reaction_joints

		self.target_bones_dict = target_bones  # for use in plotting
		self.target_bones = []

		self.total_mass = 0

		for bone, (joint1, joint2) in target_bones.items():
			# Calculate length using the initial positions of jointA and B.
			# Smoothing functions can cause the issues for the first few frames, so take avg of later frames

			frames = [50, 51, 52, 53, 54, 55, 56]
			n_averaging = len(frames)
			length = 0

			for frame in frames:
				posA = Vector(*self.joint_pos[frame, joint1])
				posB = Vector(*self.joint_pos[frame, joint2])

				if posA.length() == 0 or posB.length() == 0:
					n_averaging -= 1
				else:
					length += posA > posB

			length = length / n_averaging  # avg of all the frames data taken from

			if length == 0:
				print(f"Warning: Error in calculating length of '{bone}'")
				length = 0.01

			b = DoubleCylinder(start=joint1, end=joint2, length=length, name=bone, freq=freq,
							   **self.model.bone_length_definitions["normal"](length))

			self.target_bones.append(b)  # add bone to list
			self.total_mass += b.mass

		self.body = Body(*body_joints, self.joint_pos, freq=freq, name="body")

		self.body.get_kinematics(
			np.stack([self.joint_pos[:, body_joints[0]], self.joint_pos[:, body_joints[1]]], axis=1))
		self.body.get_dynamics()

		self.total_mass += self.body.mass

		# Paw parameters
		m = self.total_mass
		paw_d = self.model.paws
		self.L0_paws = {"front": paw_d["L0_front"] * m, "rear": paw_d["L0_rear"] * m}
		self.k_paws = {"front": paw_d["k_front"] * m, "rear": paw_d["k_rear"] * m ** paw_d["k_rear_prop"]}
		self.c_paws = {"front": paw_d["c_front"] * m, "rear": paw_d["c_rear"] * m}

		# if self.model.equation_weighting['Paw spring'] > 0:
		self.set_paw_equilibrium()

		self.get_dynamics()
		self.leg_spring_joints = leg_spring_joints

		self.calc_leg_lengths()

		self.equation_weighting = model.equation_weighting

	def get_dynamics(self):
		"""Gets dynamics of centre of mass of each bone & body"""
		for bone in self.target_bones:
			bone.get_kinematics(self.joint_pos[:, [bone.start, bone.end]])
			bone.get_dynamics()

		body = self.body
		body.get_kinematics(
			np.stack([self.joint_pos[:, body.start_joints], self.joint_pos[:, body.end_joints]], axis=1))
		body.get_dynamics()

	def calculate_forces(self, n_frame, report_equations=True):
		"""
		Sets up a system of linear equations governing the motion of the skeleton at a given frame.

		These equations are:

		- FREE JOINTS:
		The torques at free joints are zero. Free joints are joints only connected to one bone, on the end of the body eg the feet

		- INERTIA:
		On each bone, the sum of the two joint forces is equal to the mass * acceleration of the bone

		- ROTATION:
		On each bone, the net torque about the bone is equal to the I * alpha_g of the bone

		- BODY:
		The body is set up as a slightly different type of bone, in which it has several joints connected at either end, and its position is dictated by all of those joints.
		See the code for it below, it has its own set of inertial and rotational equations.

		This is set up as a least squares problem Ax = b, where A is a matrix of coefficients to multiply the unknowns by,
		x is the unknowns (in the form [F_1_x, F_1_y, F_1_z, F_2_x, ... T_1, T, ...]
		b is the result of the equations.

		A weighting is also applied to each row to weight the least squares problem (eg to priorities free joint equations)

		The problem also has bounds applied to it. For now, these bounds are simply that foot joint vertical reaction forces are non negative.

		Improvements:
		- Replace the current spinal system with a large non axisymmetric cylinder to represent the body
		- Add a sphere to represent the head

		"""

		# Consult report for explanation of system

		A = []
		b = []
		weights = []  # Collect weightings for each equation as they are added to the system
		equation_weighting = self.equation_weighting

		# Reasonable bounds for each force, and for each torque. Current limits set at 10 * weight for mass, 10 * mass at one metre for torque
		max_force = 3 * self.total_mass * g
		max_torque = 3 * self.total_mass * g
		# bounds can be adjusted further for specific joints (eg no downards reaction at the feet)
		bounds = [(-max_force, max_force)] * (3 * self.n_joints) + [(-max_torque, max_torque)] * (self.n_joints)

		def A_row(vals={}):
			"""Returns a row of 0s length 4 * self.n_joints, with other vectors in any indices in vals.
			vals is a dict of index:vector"""
			row = [0.0] * 4 * self.n_joints
			for index, val in vals.items():
				row[index] = val
			return row

		def add_blank_row():
			A.append(A_row({}))
			b.append(0)
			weights.append(0)

		def add_n_blank_rows(n=1):
			for i in range(n): add_blank_row()

		null, unit, g_vec = Vector(0, 0, 0), Vector(1, 1, 1), Vector(0, 0, -g)

		n_joints = self.n_joints

		def get_index(joint, dimension=0, is_force=True):
			"""Get correct index of D"""
			return (3 * n_joints * int(not is_force)) + ([1, 3][is_force] * joint) + dimension

		# dimension = 0 for x, 1 for y, 2 for z

		# First, add the equations to show that the torques in each of the foot joints are zero
		for no_torque_joint in self.no_torque_joints:
			# Set up the equation 1 * tau_{foot_joint} = 0
			# BOUNDARY CONDITIONS ARE FIXED, RATHER THAN AN ADDITIONAL EQUATION. SO INCORPORATE THEM INTO BOUNDS
			bounds[get_index(no_torque_joint, is_force=False)] = (0, 1e-10)

		for no_reaction_joint in self.no_reaction_joints:  # BC : no reactions
			for dim in [0, 1, 2]:
				bounds[get_index(no_reaction_joint, dimension=dim, is_force=True)] = (0, 1e-10)

		for foot_joint in self.foot_joints:
			## If the feet are a certain amount off the ground for that foot, also assign the reaction forces to be zero
			bone_name = self.get_foot_joint_from_index[foot_joint]

			end = bone_name.split(" ")[1]  # get 'front' or 'rear'
			L0 = self.L0_paws[end]  # get stiffness from 'front' or 'rear' in bone name
			# L0 = self.paw_equilibrium_values[foot_joint]

			k_paw = self.k_paws[end]
			c_paw = self.c_paws[end]

			paw_disp = self.paw_disps[foot_joint][n_frame]

			paw_off_ground = self.joint_pos[n_frame, foot_joint, 2] >= L0  # BC: no reaction in foot off ground
			paw_off_ground = paw_disp == 0

			if paw_off_ground:  # BC: no reaction in foot off ground
				for dim in [0, 1, 2]:
					bounds[get_index(foot_joint, dimension=dim, is_force=True)] = (0, 1e-10)

				add_n_blank_rows(4)  # for consistency of number of eqns

			else:  # If paw near ground, add force due to spring

				height = self.unsmoothed_data[n_frame, foot_joint, 2]

				eps = L0 - height  # min((L0 - height), L0/2)
				eps_dot = self.joint_vel[n_frame, foot_joint, 2]

				F_damp = 0  # c_paw * eps_dot

				if self.model.equation_weighting['Paw spring'] > 0:
					## PAW SPRING MODEL
					eps = paw_disp
					F_spring = k_paw * eps + c_paw * eps_dot

					if foot_joint != 20:
						A.append(A_row({get_index(foot_joint, dimension=2, is_force=True): 1}))
						b.append(F_spring + F_damp)
						weights.append(equation_weighting["Paw spring"])

				if self.model.equation_weighting['Leg spring'] > 0:
					## LEG SPRING MODEL
					K = 3000 if end == "front" else 2000
					for dim in [0, 1, 2]:
						# component = self.leg_vecs[foot_joint][n_frame][dim]
						F_spring = self.leg_disps[foot_joint][n_frame] * K  # * component
						A.append(A_row({get_index(foot_joint, dimension=dim, is_force=True): 1}))
						b.append(F_spring + F_damp)
						weights.append(equation_weighting["Leg spring"])

				# Set bounds for foot joints to only have positive vertical reactions
				bounds[get_index(foot_joint, dimension=2, is_force=True)] = (0, max_force)
				bounds[get_index(foot_joint, dimension=1, is_force=True)] = (0, 1e-10)  # set Fy=0

		for bone in self.target_bones:
			j_1, j_2 = bone.start, bone.end
			x_1, x_2 = bone.X[n_frame]

			# F_1 + F_2 + F_grav = F_net
			F_net = bone.F_net[n_frame]
			for dim in [0, 1, 2]:
				A.append(A_row({get_index(j_1, dim): 1, get_index(j_2, dim): - 1}))
				b.append((F_net - bone.mass * g_vec)[dim])
				weights.append(equation_weighting["Inertial"])

			tau_net = bone.tau_net[n_frame]
			x_g = bone.XG[n_frame]
			r_1, r_2 = (x_1 - x_g), (x_2 - x_g)

			# direction of each T is perpendicular to the bones that the joint is on
			adjacent_1_bone = [b for b in self.target_bones if b.end == j_1 and b != bone]
			if len(adjacent_1_bone) == 1:  # if there is an adjacent bone
				adj_bone = adjacent_1_bone[0]
				T_1_dir = Vector(*r_1).cross((adj_bone.X[n_frame, 1] - adj_bone.XG[n_frame])).unit()
			if len(adjacent_1_bone) == 0 or np.isnan(T_1_dir).any():  # if no adjacent, or if above calc causes error
				T_1_dir = (0, 1, 0)  # Improve later, for now say all torques about y axis

			adjacent_2_bone = [b for b in self.target_bones if b.start == j_2 and b != bone]
			if len(adjacent_2_bone) == 1:  # if there is an adjacent bone
				adj_bone = adjacent_2_bone[0]
				T_2_dir = Vector(*r_2).cross((adj_bone.X[n_frame, 0] - adj_bone.XG[n_frame])).unit()
			if len(adjacent_2_bone) == 0 or np.isnan(T_2_dir).any():  # if no adjacent, or if above calc causes error
				T_2_dir = (0, 1, 0)  # Improve later, for now say all torques about y axis

			for dim in [0, 1, 2]:
				# This loop essentially writes out the following equations into A and b for each dimension (x,y,z):
				# r1 x F1 + r2 x F2 + T1 + T2 = T_net

				# The cross product of r = (x,y,z) and F = (Fx, Fy, Fz) yields (Fz*y - Fy*z, ...)
				# Take the x component, x -> Fz*y - Fy*z
				# Notice that Fy is negative, and Fz is positive. This is always true, that, for the forces, one lower dimension than the current is positive, and one higher is negative (cyclical relations)
				# use this below

				# Get dim above and below, wrapping round for below x and above z
				dim_below = (dim - 1) % 3
				dim_above = (dim + 1) % 3

				coeff_dict = {
					get_index(j_1, dim): 0,
					# eg no effect of F_x in the x directional torque (not relevant statement, only here for readability)
					get_index(j_1, dim_above): - r_1[dim_below],  # eg multiply - z by Fy in the x direction
					get_index(j_1, dim_below): r_1[dim_above],  # eg multiply y by Fz in the x direction

					# Reversed polarity for joint 2 as the desired force is - F2
					get_index(j_2, dim_above): r_2[dim_below],
					get_index(j_2, dim_below): - r_2[dim_above],

					# Add the torques on each joint
					get_index(j_1, is_force=False): T_1_dir[dim],
					get_index(j_2, is_force=False): -T_2_dir[dim]

				}

				A.append(A_row(coeff_dict))
				b.append(tau_net[dim])
				weights.append(equation_weighting["Rotational"])

		### SOLVE FORCES ON BODY. Note body defined so all joint forces/torques on it are positive
		body = self.body
		F_net = body.F_net[n_frame]

		#  BODY INERTIAL FORCES
		for dim in [0, 1, 2]:
			A.append(A_row({get_index(j, dim): 1 for j in self.body.start_joints + self.body.end_joints}))
			b.append((F_net - body.mass * g_vec)[dim])
			weights.append(equation_weighting["Inertial"])

		# BODY ROTATIONAL FORCES - same as for bones
		x_g = body.XG[n_frame]
		tau_net = body.tau_net[n_frame]

		# Improve above later, for now say all torques about y axis
		T_dir = (0, 1, 0)

		for dim in [0, 1, 2]:
			coeff_dict = {}
			for joint in body.start_joints + body.end_joints:
				x_j = self.joint_pos[n_frame, joint]
				r_j = (x_j - x_g)  # position vector to centre

				# Get dim above and below, wrapping round for below x and above z
				dim_below, dim_above = (dim - 1) % 3, (dim + 1) % 3

				coeff_dict[get_index(joint, dim_above)] = -r_j[dim_below]  # eg multiply - z by Fy in the x direction
				coeff_dict[get_index(joint, dim_below)] = r_j[dim_above]  # eg multiply y by Fz in the x direction

				coeff_dict[get_index(joint, is_force=False)] = T_dir[dim]  # Add pure torque of pin

			A.append(A_row(coeff_dict))
			b.append(tau_net[dim])
			weights.append(equation_weighting["Rotational"])

		# print each line of the equations defined by A, b, with the final result
		# Only print variables with both non-zero values, and non-zero coefficients
		if report_equations:
			print(f"----Frame {n_frame}----")
			params = []

			for joint in range(self.n_joints):
				for dim in "xyz":
					params.append(F"F_{joint}_{dim}")  # Add forces by joint

			for joint in range(self.n_joints):
				params.append(F"T_{joint}")  # Add torques by joint

			for n, (coeffs, result) in enumerate(zip(A, b)):
				s = []
				for j, (coeff, param) in enumerate(zip(coeffs, params)):
					if coeff != 0:
						s.append(f"{round(coeff, 3)} * {param}")

				# b_actual = np.dot(A[n], D)
				# pct_error = abs(100 * (b_actual - result) / b_actual)
				if n <= 7:
					print(f"{' + '.join(s)} = {round(result, 3)}")  # ({round(b_actual, 3)}) [{round(pct_error, 2)}%]")

		return A, b, weights, bounds

	def solve_forces(self, report_equations=False, end_frames_disregarded=5, prefix="",
					 save=True):
		"""Solves the forces at each frame for the system, collects them and saves them to .npy files.

		Note: Currently, due to smoothing, the first 5 and last 5 frames are disregarded"""

		self.get_dynamics()
		n_joints = self.n_joints

		if report_equations:
			print("Solving system...")
			print(f"Total mass {round(self.total_mass, 2)} kg.")

		# If dir doesn't exist, make it
		dir = path_join(DataSources.dynamics_data, self.name)
		if self.name not in os.listdir(DataSources.dynamics_data):
			os.mkdir(dir)

		forces, torques = [], []

		f_shape, t_shape = (self.n_joints, 3), (self.n_joints,)
		# Add zeros either end due to not being able to calculate for the first or last 2 frames
		for i in range(end_frames_disregarded):
			forces.append(np.zeros(f_shape))
			torques.append(np.zeros(t_shape))

		calc_forces = []
		calc_torques = []

		progress = tqdm(total=self.n_frames - 2 * end_frames_disregarded)
		for n_frame in range(end_frames_disregarded, self.n_frames - end_frames_disregarded):
			A, b, weights, bounds = self.calculate_forces(n_frame, report_equations=report_equations)

			D = weighted_bound_least_squares(A, b, weights, bounds, rcond=None)

			f, tau = D[:(3 * n_joints)], D[(3 * n_joints):]

			f, tau = f.reshape((n_joints, 3)), tau.reshape((n_joints))

			calc_forces.append(f)
			calc_torques.append(tau)

			progress.update()

		forces[end_frames_disregarded: - end_frames_disregarded] = calc_forces
		torques += calc_torques

		for i in range(end_frames_disregarded):
			forces.append(np.zeros(f_shape))
			torques.append(np.zeros(t_shape))

		if save:
			np.save(path_join(dir, prefix + "forces.npy"), forces)
			np.save(path_join(dir, prefix + "torques.npy"), torques)

		return np.array(forces), np.array(torques)

	def get_com_position(self):
		"""Calculates the position of the centre of mass of the whole system at each timestep"""
		return sum(b.XG * b.mass for b in self.target_bones + [self.body]) / self.total_mass

	def return_equations(self, end_frames_disregarded=5):
		"""For each frame, return the equation vector b"""
		self.get_dynamics()
		bs = []

		for n_frame in range(end_frames_disregarded, self.n_frames - end_frames_disregarded):
			A, b, weights, bounds = self.calculate_forces(n_frame, report_equations=False)
			bs.append(b)

		return np.array(bs)

	def set_paw_equilibrium(self):
		"""Get paw equilibrium from mocap data by finding the drop of the paw.
		This method will work for the current dataset, but is likely not robust, so can be replaced with
		a better method of finding the paw equilibrium at a later date"""

		if self.is_mocap:
			paw_z_heights = self.unsmoothed_data[:, self.foot_joints, 2]

		else:
			paw_z_heights = self.unsmoothed_data[:, self.foot_joints, 2]


		self.paw_disps = {}  # paw joint: displacement over time, for paw spring model

		min_contacts_detected = 3  # minimum requirement to use peak detection mode

		plot = True
		if plot:
			fig, axes = plt.subplots(nrows=2, ncols=2)

		footfall_detector = FootfallDetector(train=False, load=True, name=["smal", "mocap"][self.is_mocap])
		for n, paw in enumerate(self.foot_joints):
			contact_ends_failed = False
			disp = np.zeros((self.n_frames))  # will give eps - the displacement of the paw from equilibrium
			# for when the paw is in contact with the ground

			Z = paw_z_heights[:, n]

			on_ground = footfall_detector.process_clip(Z)
			on_ground_idxs = np.where(on_ground > 0)[0]

			if plot:
				axes[n // 2, n % 2].plot(Z.mean() * (on_ground), color="red", alpha=0.3)

			min_footfall_width = 3  # 3 frames long minimum to count as a footfall
			footfalls = consecutive(on_ground_idxs)
			trigger_height = np.percentile(np.array([Z[ff].max() for ff in footfalls]), 25)  # mean trigger height
			for footfall in footfalls:
				if len(footfall) > min_footfall_width:
					# disp[footfall] = Z[footfall].max() - Z[footfall] # old
					disp[footfall] = np.clip(trigger_height - Z[footfall], a_min=0, a_max=None)

			self.paw_disps[paw] = disp

			if plot:
				ax = axes[n // 2, n % 2]
				ax.plot(Z)

				Z_on_ground = Z.copy()
				Z_on_ground[disp == 0] = np.nan
				ax.plot(Z_on_ground, color="green")
				ax.plot(disp)
				Z_smoothed = self.joint_pos[:, paw, 2]

				ax.set_title(n)

		if plot:
			plt.show(block=False)
			plt.draw()
			plt.pause(1e-8)

	def view_ground_displacements(self, deriv=0):
		"""Plot and show a graph of vertical displacement against frames for each paw - identifying L0 for each paw"""

		fig, axes = plt.subplots(nrows=4)
		for n, j in enumerate(self.foot_joints):
			label = foot_joint_labels[n]
			ax = axes[n]
			if deriv == 0:
				X = self.joint_pos[:, j, 2]
				X_unsmoothed = self.unsmoothed_data[:, j, 2]
				ax.plot(X)
				ax.plot(X_unsmoothed, alpha=.6)
				# ax.axhline(self.paw_equilibrium_values[j], ls = "--")
				ax.axhline(self.L0_paws[label.split(" ")[0]])
			elif deriv == 1:
				ax.plot(self.joint_vel[:, j, 2])

			ax.set_title(label)

		plt.show()

	def view_com_displacements(self, deriv=0):
		"""Plot and show graph of X, Y, and Z motion of CoM of dog.
		If deriv > 0, plot that derivative of the displacement"""

		fig, ax = plt.subplots()
		com_data = self.get_com_position()
		if deriv > 0:
			com_data = nth_time_deriv(com_data, 1 / self.freq, n=deriv)

		for i in [0, 1, 2]:
			ax.plot(com_data[:, i], label="xyz"[i])

		ax.legend()
		plt.show()

	def calc_leg_lengths(self):
		"""Uses the compliant-legged walking model estimation to work out the average length of legs.
		Assume legs are undeformed while off ground. Work out avg distance from leg to COM"""

		self.leg_disps = {}  # length of leg over time for each paw
		self.leg_vecs = {}  # normalised vector of leg spring direction for each paw

		plot = True
		if plot: fig, axes = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="row")

		for n, paw in enumerate(self.foot_joints):
			is_front = n < 2  # Assumes order of f left, f right, r left, r right

			tol = 1e-3
			on_ground = self.paw_disps[paw] > tol
			off_ground = self.paw_disps[paw] <= tol

			# centre_of_rot = self.body.XG[:]#self.body.X[:, int(is_front)]
			# centre_of_rot = self.unsmoothed_data[:, self.body_joints[is_front][n%2]]
			if self.is_mocap:
				centre_of_rot = self.unsmoothed_data[:, self.leg_spring_joints[n]]
				paw_pos = self.unsmoothed_data[:, paw]

			else:
				centre_of_rot = self.unsmoothed_data[:, self.leg_spring_joints[n]]
				paw_pos = self.unsmoothed_data[:, paw]

			X, Z = np.swapaxes(centre_of_rot[:, [0, 2]], 0, 1)  # get X, Z position of CoM
			X_PAW, Z_PAW = np.swapaxes(paw_pos[:, [0, 2]], 0, 1)  # get X, Z position of CoM

			THETA = np.arctan((X_PAW - X) / (Z - Z_PAW))  # angle between spring and vertical

			L = ((X - X_PAW) ** 2 + (Z - Z_PAW) ** 2) ** .5
			L0 = (L).max()

			z_disp = (L - L0) * np.cos(THETA)
			x_disp = (L - L0) * np.sin(THETA)

			# get z displacement by footfall
			disp = np.zeros(self.n_frames)

			# if self.is_mocap:
			for ff in consecutive(np.where(on_ground)[0]):
				if len(ff) < 3: continue  # min width of footfall required
				disp[ff] = z_disp[ff].max() - z_disp[ff]

			# else:
			#     disp = -z_disp

			self.leg_disps[paw] = disp

			if plot:
				ax = axes[n // 2, n % 2]

				# ax.plot(L)
				ax.plot(L - L0)
				ax.plot(disp, color="green")

		if plot:
			plt.tight_layout()
			# plt.show()
			plt.show(block=False)
			plt.draw()
			plt.pause(1e-8)


def norm_kin_data(kin_data, targ_markers=None):
	"""Normalise kinematic data.
	If targ_markers given, normalise so these markers are at desired height"""

	norm_height = 0.4  # 0.635 # fixed to Ally height for now

	# scale so minimum is at (0,0,0)
	for dim in [0, 1, 2]:
		kin_data[:, :, dim] -= kin_data[:, :, dim].min()

	if targ_markers is None:
		kin_data = norm_height * kin_data / np.max(kin_data[:, :, 2])

	elif targ_markers is not None:
		height_target = kin_data[:, targ_markers, 2].mean()
		kin_data = norm_height * kin_data / height_target

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


kin_src_to_solver_name = lambda s: s.replace("/", " ").replace(" ", "_").replace(".c3d", "")


def load_solver(kin_src, clip_length, mocap=True, resample_freq=100):
	if mocap:
		joint_data = C3DData(ax=None, src=kin_src, interpolate=True, crop=clip_length,
							 fix_rotations="3 kph" in kin_src)  # only fix rotations for 3 kph for now

	else:
		joint_data = SMALData(kin_src, freq=30, norm=True, crop=clip_length, smooth=True)

	joint_data.resample_at(resample_freq)  ### TRY RESAMPLING DATA TO 100 Hz
	target_bones, body_joints, no_torque_joints, leg_spring_joints = joint_data.generate_skeleton_mapping()

	# Normalise data based on z data, so that the dog is roughly 0.5m high. Also smooth data
	kin_data = np.array(joint_data.all_data)
	kin_data = norm_kin_data(kin_data, targ_markers=leg_spring_joints)

	solver_kwargs = dict(target_bones=target_bones,
						 body_joints=body_joints, no_torque_joints=no_torque_joints,
						 foot_joints=no_torque_joints, leg_spring_joints=leg_spring_joints,
						 freq=joint_data.freq,
						 name=kin_src_to_solver_name(kin_src))

	solver = InverseDynamicsSolver(joint_data=kin_data, **solver_kwargs, is_mocap=mocap)
	print(f"Solver loaded. Mass = {solver.total_mass:.1f} kg.")
	return solver

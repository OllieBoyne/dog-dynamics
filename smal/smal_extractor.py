"""Script for taking .npz outputs from the Deep Network WLDO optimiser, and producing kinematic data.

Also includes scripts for:
- plotting mesh as segmentation
- Extracting priors"""

import numpy as np
from my_work.data_loader import DataSources, path_join, C3DData, get_smal_data
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
from matplotlib.colors import LightSource
from my_work.utils import save_animation, batch_rodrigues
from my_work.mesh_viewer import BBox
from tqdm import tqdm
from scipy import signal

# Get smal deformer mod from smalst package
import sys, os, copy
sys.path.append("E:\IIB Project Data\smalst")
import smal_deformer_mod

from my_work.priors import *

import torch

joint_connections = [(i, i+1) for i in [0, 1, 3, 4, 6, 7, 9, 10]] + [(12, 13)]

shade_colours = {
    "grey": "grey",
    "blue": "#1261A0"
}



def plot_joint_disp(smal_npz, j=0, mocap_src = None, **kwargs):

    J = get_smal_data(smal_npz)

    freq = 30

    fig, ax = plt.subplots()
    j=3

    t = np.arange(J.shape[0]) / freq
    dims = "z"
    for d in dims:
        n = dims.index(d)
        x = J[:, j, n]
        x_smoothed = signal.savgol_filter(x, window_length=15, polyorder=3)
        ax.plot(t, x, alpha = .1, label=d)
        ax.plot(t, x, label=d)

    if mocap_src is not None:
        ## Add plot of correct mocap joint - for now always left front bottom
        mocap_data = C3DData(ax=None, src=mocap_src, interpolate=True)
        joint = "left rear paw"
        mocap_j_data = mocap_data.get_marker_data_by_names(joint)

        mocap_z_data = mocap_j_data[:, 0, 2]
        t0 = 6.55
        mocap_t = np.arange(4, 4 + mocap_data.n_frames/mocap_data.freq, 1/mocap_data.freq)

        ax2 = ax.twinx()
        ax2.plot(mocap_t, mocap_z_data)

    plt.legend()
    plt.show()

def joints_as_anim(smal_npz, clip_name = None, img_dir = None, fps=10, **kwargs):
    """Animate joint sequence. Save to correct dir.
    Use frame dir to view actual clip"""
    smal_npz = path_join(DataSources.smal_outputs, smal_npz)
    J = get_smal_data(smal_npz)["joints"]

    if clip_name is None:
        clip_name = smal_npz.split("/")[-1].split("\\")[-1].split(".")[0] # get name of .npz files

    fig = plt.figure()
    img_ax = fig.add_subplot(211)
    ax = fig.add_subplot(212, projection="3d")

    ## Load images
    img_data = []
    for f in sorted(os.listdir(img_dir)):
        img_data.append(plt.imread(path_join(img_dir, f)))

    img_ax.axis("off")
    img_plot = img_ax.imshow(img_data[0])

    frames = len(J)

    X, Y, Z = np.rollaxis(J.reshape(-1, 3), -1)
    bbox = BBox(X, Y, Z)
    bbox.equal_3d_axes(ax, zoom = 1.5)

    scat = ax.scatter(*np.swapaxes(J[0], 0, 1), s = 2, c = "blue")
    plots = []
    for (j1, j2) in joint_connections:
        plots.append(ax.plot(*np.swapaxes(J[0, [j1,j2]], 0, 1), color = "red")[0])

    progress = tqdm(total = frames)
    def anim(i):
        img_plot.set_data(img_data[i])

        scat._offsets3d = np.swapaxes(J[i], 0, 1)
        for n, (j1, j2) in enumerate(joint_connections):
            plot = plots[n]
            for dim, method in enumerate([plot.set_xdata, plot.set_ydata, plot.set_3d_properties]):
                method(J[i, [j1,j2], dim])
        plt.draw()
        progress.update()

    # # FOR VIEWING JOINT LABEL IDS
    # anim(50)
    # for n, (x,y,z) in enumerate(J[21]):
    #     ax.text(x, y, z, str(n))
    # plt.show()

    save_animation(anim, fig, frames, fps = fps, title=clip_name, dir = r"E:\iib_project_other\smal_joints_from_clip\\")

def mesh_as_anim(smal_npz, clip_name = None, img_dir = None, fps=10, **kwargs):
    """Animate joint sequence. Save to correct dir.
    Use frame dir to view actual clip"""
    smal_npz = path_join(DataSources.smal_outputs, smal_npz)
    data = get_smal_data(smal_npz, smooth=True)
    v, faces = data["verts"], data["faces"]

    if clip_name is None:
        clip_name = smal_npz.split("/")[-1].split("\\")[-1].split(".")[0] # get name of .npz files

    fig = plt.figure()
    # img_ax = fig.add_subplot(221)
    # ax_norm = fig.add_subplot(222, projection="3d")
    ax_front = fig.add_subplot(111, projection="3d")
    # ax_top = fig.add_subplot(224, projection="3d")

    ax_front.view_init(azim=90, elev = 0)
    # ax_top.view_init(azim=0, elev=90)

    axes_3d = [ax_front] # [ax_norm, ax_front, ax_top]

    ## Load images
    # img_data = []
    # for f in sorted(os.listdir(img_dir)):
    #     img_data.append(plt.imread(path_join(img_dir, f)))
    # img_ax.axis("off")
    # img_plot = img_ax.imshow(img_data[0])

    frames = len(data['joints'])

    X, Y, Z = np.rollaxis(v, -1)
    bbox = BBox(np.ravel(X), np.ravel(Y), np.ravel(Z))
    [bbox.equal_3d_axes(ax, zoom = 1.5) for ax in axes_3d]

    x = np.asarray(X[0], dtype=np.float64)
    y = np.asarray(Y[0], dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("ME: x and y must be equal-length 1-D arrays")

    tri = Triangulation(X[0], Y[0], triangles=faces)
    light = LightSource(altdeg=20, azdeg=70)
    trisurfs = []
    for ax in axes_3d:
        ax.axis("off")
        trisurfs.append(ax.plot_trisurf(X[0], Y[0], Z[0], triangles=tri.triangles))

    # plt.show()

    progress = tqdm(total=frames)
    def anim(i):
        # img_plot.set_data(img_data[i])
        for n, ax in enumerate(axes_3d):
            trisurfs[n].remove()
            trisurfs[n] = ax.plot_trisurf(X[i], Y[i], Z[i], triangles=tri.triangles, color=shade_colours["blue"], shade=True,
                                         lightsource=light)
        progress.update()

    plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0, wspace = 0, hspace=0)

    save_animation(anim, fig, frames, fps = fps, title=clip_name, dir = r"E:\iib_project_other\smal_mesh_from_clip\\")


rs_dog = dict(
    smal_npz = path_join(DataSources.smal_outputs, "rs_dog.npz"),
    img_dir = r"E:\IIB Project Data\Training data sets\BADJA\extra_videos\rs_dog\rgb",
)

set2_3r4 = dict(
    smal_npz = path_join(DataSources.smal_outputs, "set 2 3r4 - pad test.npz"),
    img_dir = r"E:\IIB Project Data\Data set_2 17-10-19\Ollie export 17-10-19\3r4 right",
    mocap_src = r"set_2/3 kph run 4.c3d",
    title="set2_3r4"
)

set2_3r4_left = dict(
    smal_npz = path_join(DataSources.smal_outputs, "set 2 3r4 - left.npz"),
    img_dir = r"E:\IIB Project Data\Data set_2 17-10-19\Ollie export 17-10-19\3r4 left",
    title="set2_3r4_left"
)

zebris_lab = dict(
    smal_npz = path_join(DataSources.smal_outputs, "zebris_lab - p11.npz",),
    img_dir = r"E:\IIB Project Data\Zebris files - 29-04\lab_frames"
)

zebris_gus = dict(
    smal_npz = path_join(DataSources.smal_outputs, "zebris_gus_dynamic.npz",),
    img_dir = r"E:\IIB Project Data\Zebris files - 29-04\gus_frames"
)

# plot_joints(path_join(DataSources.smal_outputs, "set 2 3r4.npz"))
mesh_as_anim(**zebris_lab, fps=15)
# plot_joint_disp(**set2_3r4)

def plot_smal_template():
    """Plots SMAL template
    """

    smalx = smal_deformer_mod.SMALX(opts=None, n_batch=1)
    smalx.global_rot[:,0] = 0#1.209
    smalx.global_rot[:, 1] = 0#1.209
    smalx.global_rot[:, 2] = -np.pi/2

    v = smalx.v_template
    f = smalx.faces

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = np.swapaxes(v, 0, 1)

    from my_work.mesh_viewer import BBox
    b = BBox(X, Y, Z)
    b.equal_3d_axes(ax, zoom = 1.95)
    ax.axis("off")

    ax.view_init(azim = 65, elev = 20)

    # PLOT FACES
    tri = Triangulation(X, Y, triangles=f)
    light = LightSource(altdeg=40, azdeg=65)
    plot = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color=shade_colours["blue"], shade = True,
                           lightsource=light)

    plt.subplots_adjust(left=0, right = 1, top = 1, bottom=0)

    # show = True
    # if show:
    #     plt.show()
    #     return None

    out_loc = os.path.join(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\smal_template.png")
    plt.savefig(out_loc, dpi = 300)

# plot_smal_template()

def plot_shape_var():
    """Produces figure of four forms of shape variation for SMAL Mesh.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")


    opts = {
        26: 0, # do nothing
        20: 0.223, # lengthen legs
        22: -.693, # shorten tail
        25: .693, # puff tail
    }

    disp = .9 # displacement between meshes

    smalx = smal_deformer_mod.SMALX(opts=None, n_batch=4,
                                    betas_fixed = False, use_smal_template=True)
    smalx.global_rot[:, 0] = 0  # 1.209
    smalx.global_rot[:, 1] = 0  # 1.209
    smalx.global_rot[:, 2] = np.pi / 2 - .15

    for n, (b, opt) in enumerate(opts.items()):
        smalx.multi_betas[n, b] = opt  #apply shape transformations to each of 4 meshes

    v, f = smalx.get_verts()
    v = v.detach().numpy()

    ## adjust each mesh so feet are on z =0
    for i in range(4):
        v[i, :, 2] -= v[i, :, 2].min()

    from my_work.mesh_viewer import BBox
    all_x = v[..., 0]
    xrange = [-2*disp + all_x.min(), 2*disp + all_x.max()] # maximum x range
    b = BBox(xrange, *[np.ravel(v[..., i]) for i in [1,2]]) # pass all verts through to equal axis maker
    b.equal_3d_axes(ax, zoom = 2)

    ax.view_init(azim=60, elev=0)

    for n in range(4):
        X, Y, Z = np.swapaxes(v[n], 0, 1)
        X -= (n-2) * disp

        # PLOT FACES
        tri = Triangulation(X, Y, triangles=f)
        #azdeg=315, altdeg=45
        light = LightSource(altdeg=20, azdeg=50)
        plot = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color=shade_colours["blue"], shade=True,
                               lightsource=light)

    plt.subplots_adjust(left=0, right = 1, top = 1, bottom=0)

    # show = True
    # if show:
    #     plt.show()
    #     return None

    out_loc = os.path.join(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\image_processing\shape variation\all.png")
    plt.savefig(out_loc, dpi = 400)


def smal_to_seg(rots = {}, betas = None, elev = 0, azim = 0, title = "blank", shade = False, fix_tail=False,
                show = False, col = "gray", light_elev=20, light_az=90):
    """Given smal params, and a view init, get the 3d view, and extract as a binary segmentation image.
    Used for producing figures for report.

    :param rots: optional dict of joint: [x, y, z] rotation
    :param betas: either None (default), or set of betas
    :param elev:
    :param azim:
    :param title:
    :param shade: whether to shade the trisurf
    :param: fix_tail: modify initial beta to shorten tail of SMAL model
    :param: show: whether to show plot
    """

    smalx = smal_deformer_mod.SMALX(opts=None, n_batch=1)
    smalx.global_rot[:,0] = 0#1.209
    smalx.global_rot[:, 1] = 0#1.209
    smalx.global_rot[:, 2] = -np.pi/2

    if betas is not None:
        smalx.multi_betas[:] = torch.from_numpy(betas)

    if fix_tail:
        smalx.multi_betas[22] = min(smalx.multi_betas[22], 0.75) # cap tail lengthener at .75
        smalx.multi_betas[23] = min(smalx.multi_betas[23], 0.5) # cap tail widener at .5

    # smalx.multi_betas[21] = -.1 # 0.11546459 # narrow legs

    if rots != {}:
        for j, rot in rots.items():
            for i in range(3):
                smalx.joint_rot[:, j, i] = float(rot[i])

    v, f, j = smalx.get_verts(return_joints=True)
    v = v.detach().numpy()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = np.swapaxes(v[0], 0, 1)

    from my_work.mesh_viewer import BBox
    b = BBox(X, Y, Z)
    b.equal_3d_axes(ax, zoom = 1.75)
    ax.axis("off")

    ax.view_init(azim = azim, elev = elev)

    # PLOT FACES
    tri = Triangulation(X, Y, triangles=f)
    light = LightSource(altdeg=light_elev, azdeg=light_az)
    plot = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color=col, shade = shade,
                           lightsource=light)

    # ax.scatter(X, Y, Z)

    plt.subplots_adjust(left=0, right = 1, top = 1, bottom=0)

    if show:
        plt.show()
        return None

    out_loc = os.path.join(r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\mesh_segs", f"{title}.png")
    plt.savefig(out_loc, dpi = 300)

def plot_mixtures():
    """Produce segs for mixtures fig"""
    unity = UnimodalPrior("unity_with_sf")
    smal_to_seg(title="unimodal", betas = unity.mean)

    shapeA = unity.sample_from()
    shapeB = unity.sample_from()
    shapeA[20] -= .4 # Short legs
    shapeA[22] -= .4 # Short tail

    shapeB[22] = -.3 # reasonable tail
    shapeB[25] = .8 # Tall ears
    shapeB[20] = .3 # Short legs

    smal_to_seg(title="shapeA", betas = shapeA)
    smal_to_seg(title="shapeB", betas = shapeB)

    shapeC = .3 * shapeA + .7 * shapeB
    smal_to_seg(title="shapeC", betas = shapeC)

def plot_walking_seg():
    """Plots the dog walking as a segmentation"""

    unity = UnimodalPrior("unity_with_sf")
    # lengthen legs for visualisation
    betas = unity.mean
    betas[20] += .3

    LABELLED_JOINTS = [
        14, 13, 12,  # left front (0, 1, 2)
        24, 23, 22,  # left rear (3, 4, 5)
        10, 9, 8,  # right front (6, 7, 8)
        20, 19, 18,  # right rear (9, 10, 11)
        25, 31,  # tail start -> end (12, 13)
        34, 33,  # right ear, left ear (14, 15)
        35, 36,  # nose, chin (16, 17)
        38, 37,  # right tip, left tip (18, 19)
        39, 40]  # left eye, right eye

    # 6 = left front
    # 10 = right front
    # 20 = right rear
    # 16 = left rear

    #lf, rr on ground
    pose = {10: [0, -.6, 0], 12: [0, .7, 0], 13: [0, .7, 0], # left front
            16: [0.05, -0.6, 0], 18: [0, 0.3, 0], 19 : [0, 0.7, 0]}

    smal_to_seg(title="Walk1", betas = betas, rots=pose, elev = 40, azim = 30, shade = False)

    #rf, lr on ground
    pose = {6: [0, -.6, 0], 8: [0, .7, 0], 9: [0, .7, 0], # left front
            20: [0.05, -0.6, 0], 22: [0, 0.3, 0], 23 : [0, 0.7, 0]}

    smal_to_seg(title="Walk2", betas = betas, rots=pose, elev = 40, azim = 30, shade = False)

def plot_unity_fit(src = r"E:\IIB Project Data\Dog 3D models\animation_fit_to_mesh_outputs_with_mbetas_v3 - fixed shape family = 1-10-3-2020\boxer\data",
                   frame = 4):
    """Given file src, and frame num, plot unity fit"""

    betas = np.load(path_join(src, "multi_betas.npy"))
    pose = np.load(path_join(src, "joint_rot.npy"))[frame]

    rots = {i: pose[i] for i in range(pose.shape[0])}
    col = shade_colours['blue']
    # smal_to_seg(title="Unity fit before", shade = True, azim=210, elev=30, fix_tail=True, col=col)
    smal_to_seg(betas = betas, rots = rots, title="Unity fit after", shade = True, azim=210, elev=30, col=col)

# plot_unity_fit()

def produce_samples(shape_prior, pose_prior, n_samples = 1):
    """For each of shape and pose prior provided, produce n_samples random samples of each, and save."""

    settings = dict(shade = True, azim = 0, elev = 0, fix_tail=True, col=shade_colours['blue'])

    for n in range(n_samples):
        smal_to_seg(betas=shape_prior.sample_from(),
                        **settings, title = f"shape_sample_{n}")

        smal_to_seg(betas=shape_prior.mean, rots=pose_prior.sample_from(),
                        **settings, title = f"pose_sample_{n}")

    # Load all images, make into 2 rows, save as final image:
    out = []
    loc = r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Figures\mesh_segs"
    for i in ["shape", "pose"]:
        row = []
        for n in range(n_samples):
            file = os.path.join(loc, f"{i}_sample_{n}.png")
            row.append(plt.imread(file))
            os.remove(file)

        out.append(np.concatenate(row, axis = 1))

    out = np.concatenate(out, axis = 0)
    plt.imsave(os.path.join(loc, f"gallery_{n_samples}.png"), out)



def view_joint(joint=0, dim=1):
    """View the rotational axis for a given joint about a given direction#
    dim = 012 for xyz
    """

    smalx = smal_deformer_mod.SMALX(opts=None, n_batch=1)
    # smalx.global_rot[:,0] = 0#1.209
    # smalx.global_rot[:, 1] = 0#1.209
    # smalx.global_rot[:, 2] = -np.pi/2


    v, f, J = smalx.get_verts(return_joints=True)
    v = v.detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = np.swapaxes(v[0], 0, 1)

    from my_work.mesh_viewer import BBox
    b = BBox(X, Y, Z)
    b.equal_3d_axes(ax, zoom = 1.75)
    ax.axis("off")

    # PLOT FACES
    tri = Triangulation(X, Y, triangles=f)
    plot = ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, color="gray", shade = True)

    # ax.scatter(X, Y, Z)

    # plot joint axis
    j = J[0, joint].detach().numpy()
    unit_vec = np.zeros(3)
    unit_vec[dim] = 1
    ax.plot(*zip(*[j - unit_vec, j + unit_vec]))

    plt.subplots_adjust(left=0, right = 1, top = 1, bottom=0)

    plt.show()

# plot_walking_seg()
# plot_unity_fit()

# smal_to_seg(title = "v1", azim = -30, shade = True, show = True)

# p = PosePrior("unity_pose_prior_with_cov_35parts")
# s = UnimodalPrior()
# produce_samples(s, p, 5)

# rots = p.sample_from()
# smal_to_seg(rots= rots, show = True, shade = True)

# smal_to_seg(rots={7: [2, 0, 0]}, azim=-30, shade=True, title="rot_test")

# plot_smal_template()
# plot_shape_var()

# view_joint(10, 1)

# produce random deviations from SMAL template
# FOR FINAL FIGURE PRODUCED FOR PPT
# smal_rand_betas = np.random.randn(27)/2
# smal_rand_betas[20:] = 0 # no shape params
#
#
# # smal_to_seg(betas = smal_rand_betas,
# #             title="betas from template", azim=-30, elev=20, light_az=65, light_elev=40, shade=True,
# #             col=shade_colours['blue'], fix_tail=True)
#
# pose = {10: [0, -.6, 0], 12: [0, .7, 0], 13: [0, .7, 0],  # left front
#         16: [0.05, -0.6, 0], 18: [0, 0.3, 0], 19: [0, 0.7, 0]} # stolen from walking seg
#
# smal_to_seg(rots = pose,
#             title="thetas from template", azim=-30, elev=20, light_az=65, light_elev=40, shade=True,
#             col=shade_colours['blue'], fix_tail=True)

# smal_to_seg(title="shape_fam_1_template", azim=-30, elev=20, light_az=65, light_elev=40, shade=True,
#             col=shade_colours['blue'], fix_tail=True)
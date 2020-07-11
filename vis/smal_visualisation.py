"""SCRIPTS FOR VISUALISING THE OUTPUT SMAL MODELS.
Some of this taken from mesh_viewer, but this is much improved"""

from my_work.data_loader import extract_ply, MeshData
from my_work.visualisations import MeshPlotter
from my_work.utils import save_animation
from my_work.mesh_viewer import BBox
import os
import numpy as np
from matplotlib import pyplot as plt

osp = os.path

default_dir = r"E:\IIB Project Data\produced data\ply collections\Amstaff01"

def load_data_from_dir(mesh_dir, title = "smal_fitting_output"):
    """Given a dir of frame_x, in which each frame_x has a file st10_ep0.ply which is the final mesh for that stage,
    export the ply data and return as a numpy array"""

    dpi = 200
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi))
    ax = fig.add_subplot(111, projection="3d")

    md = MeshData(mesh_dir, freq=60)
    mesh_plotter = MeshPlotter(ax, md.vertex_data, md.face_data, alpha=0.7, freq=md.freq)

    bbox = BBox(md.vertex_data[:,:,0].flatten(), md.vertex_data[:,:,1].flatten(), md.vertex_data[:,:,2].flatten())
    bbox.equal_3d_axes(ax, zoom = 1.4)

    mesh_plotter.update_frame(n_frame=0)

    def anim(i):
        mesh_plotter.update_frame(n_frame=i)

    plt.tight_layout()
    ax.axis("off")
    ax.view_init(azim=90, elev = 0)
    save_animation(anim, fig, frames=md.n_frames, dir = r"E:\IIB Project Data\Dog 3D models\Smal fitting peformance", fps = 8, title=title, dpi = dpi)

load_data_from_dir(default_dir)
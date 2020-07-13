"""Script that:
- goes through a directory of FBX Binary files
- Converts each one individually to a directory of .obj files for each frame
- Cleans up the directories. (not yet working)

This can then be fed into another script which extracts vertex data from the .obj files

NOTE: THE EXTRACTED OBJ FILES ARE NOT PERFECT MESHES, BUT THE VERTICES ARE ACCURATE (I THINK BECAUSE NORMALS AREN'T EXTRACTED, THE FACES LOOK WEIRD. BUT THE VERTICES ARE ACCURATE).
"""

import bpy, os
import numpy as np

src_dir = r"E:\IIB Project Data\Dog 3D models\all_models\fbx_bin"
all_output_folder = r"E:\IIB Project Data\Dog 3D models\all_models\obj"
# bpy.ops.wm.open_mainfile(filepath=src_file)

osp = os.path

for fbx_file in ["Amstaff (all).fbx"]:#os.listdir(src_dir):
	title = fbx_file[:-4] # remove .fbx
	this_output_folder = osp.join(all_output_folder, title+"\\")
	dog_name = title.split("_")


	if title not in os.listdir(all_output_folder):
		os.mkdir(this_output_folder)

	bpy.ops.import_scene.fbx(filepath=osp.join(src_dir, fbx_file))

	sce = bpy.context.scene
	sce.frame_end = 1115 # Number of frames in scene
	objects = sce.objects
	objs = [obj for obj in objects if obj.name[-2:] == "LP"] # Load Low Poly (LP) mode of object (options are low poly, super low poly, high poly).
	if len(objs) > 0:
		obj = objs[0]
	else:
		objs = [obj for obj in objects if "LP" in obj.name] # Some mesh names are slightly different
		if len(objs) > 0:
			obj = objs[0]
		else:
			raise ValueError("No LP Found: " + title)

	bpy.context.view_layer.objects.active = obj # select this object as active layer to export
	bpy.ops.export_scene.obj(filepath=this_output_folder, use_selection=True, use_animation=True) # Save in folder. Saves Obj files (with no extension), and .mtl files
	#
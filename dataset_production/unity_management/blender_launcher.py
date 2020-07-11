"""Calls the console to launch blender and the blender_analyser script

Instructions:
-> In the .blend file, make sure the object to be called is named 'mesh'.
-> Run the blender_launcher script"""

import os

blender_loc = r"E:\Program Files\Blender\blender.exe"

os.system(f" \"{blender_loc}\" --background --python blender_extractor.py")#" --background --python myscript.py")"""
# os.system("blender_analyser.py")
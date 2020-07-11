"""Script for managing Dataset 2 - video of dogs from multiple angles. Several functions:

- Given a title, time start, and time finish, creates a directory of video form each camera between those times
- Calibrate cameras"""


import os, csv
import numpy as np

from matplotlib import pyplot as plt
from vis.utils import save_animation


src = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\raw"
clip_target = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\clips"
clip_frames_target = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\clips_frames"
ffmpeg = "ffmpeg" # location of ffmpeg.exe, or just "ffmpeg" if in PATH

labelled_src = r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\labelled clips"

osp = os.path

def collect_time_range(title, time_start, time_finish, overwrite=False):
    """From each camera, takes the footage in the range time_start:time_finish, and saves as a clip in the directoryu
    clip_target/<title>.

    times must be in format SS.xxx or HH:MM:SS.xxx"""

    # Convert to SS.xxx if necessary
    if ":" in time_start:
        time_start = sum(float(i)*j for i, j in zip(time_start.split(":"), [3600, 60, 1]))
    if ":" in time_finish:
        time_finish = sum(float(i)*j for i, j in zip(time_finish.split(":"), [3600, 60, 1]))

    time_start = float(time_start)
    time_finish = float(time_finish)

    if osp.isdir(osp.join(clip_target, title)):
        if not overwrite and input(f"{title} Already exists. Overwrite? (y/n)") != "y":
            raise InterruptedError()
    else:
        os.mkdir(osp.join(clip_target, title))
        os.mkdir(osp.join(clip_frames_target, title))

    # for cam in range(1,5):
    #     if not osp.isdir(osp.join(clip_frames_target, title, str(cam))):
    #         os.mkdir(osp.join(clip_frames_target, title, str(cam)))

    fps = 15

    offsets = {
        1: 0, 2: 14, 3: 10, 4: 11
    } # offset of each camera in frames. a positive number depicts how many frames fast it is

    for cam in range(1,5):
        src_vid = osp.join(src, f"cam {cam}.asf")
        src_out_clip = F'\"{clip_target}\\{title}\\{cam}.mp4\"'

        offset = offsets[cam] / fps
        commands = [ffmpeg, "-y", # -y always overwrites
                   "-i", F'\"{src_vid}\"',
                   "-ss", str(time_start - offset), # Get each video
                   "-t", str(time_finish - time_start), # Cut video to correct length
                   src_out_clip]

        os.system(" ".join(commands))

        # Convert clip to frames
        os.system(f"ffmpeg -i {src_out_clip} -r 1.5 \"{clip_frames_target}\\{title}\\{cam}_%05d.png\"") # Save frames at 1.5 fps

    # Convert clips to 2x2 grid
    command = ["ffmpeg", "-y"]
    for c in [1,2,3,4,1]: command += ["-i", F'\"{clip_target}\\{title}\\{c}.mp4\"']
    # command += ["-lavfi \"[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack\" -shortest", f"\"{osp.join(clip_target, title, 'all.mp4')}\""]
    command += ["-filter_complex \"[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]\" -map \"[v]\"",  f"\"{osp.join(clip_target, title, 'all.mp4')}\""]
    os.system(" ".join(command))

def collect_time_ranges_csv(src=r"E:\IIB Project Data\DATA SET 3 - 3D VIDEO\clips\clip_log.csv"):
    """Based on a .csv file of time ranges, convert each to a list of frames."""

    with open(src, "r") as infile:
        reader = csv.reader(infile)
        next(reader)
        for line in reader:
            type, dog, name, start, end = line
            # if f"{dog} {name}" == "Ally Jump":
            collect_time_range(f"{dog} {name}", start, end, overwrite=True)

def combined_labelled_clip(name = "gracie run for ball 1", playback_speed=1.0):
    """Given a directory of labelled images from each camera, produces a side by side animation of all sequences"""

    fig, axes = plt.subplots(nrows=2, ncols=2)

    source_dir = osp.join(labelled_src, name)
    cam_loc = lambda c: osp.join(source_dir, str(c)) # return full location for given camera folder

    # Issue with clips running at different frame rates, so run clip at highest frame rate = 15 and draw out other clips
    fps_max = 15
    len_dirs = [len(os.listdir(cam_loc(i))) for i in range(1, 5)]
    fps_dirs = [15 * l/max(len_dirs) for l in len_dirs]

    all_img_data = [[plt.imread(osp.join(cam_loc(c), f)) for f in sorted(os.listdir(cam_loc(c)),
                key=lambda i: int(i.split(".")[0]))] for c in range(1,5)] # Load all image data

    plots = []
    for c in range(1, 5):
        ax = np.take(np.take(axes, c>2, axis = 0), c%2==0, axis=0) # [[cam 1, cam 2], [cam 3, cam 4]]
        p = ax.imshow(all_img_data[c-1][0])
        plots.append(p)

        ax.axis("off")
        ax.set_title(f"Cam {c}")

    def animate(i):

        for c in range(1, 5):
            fps = fps_dirs[c-1]
            plots[c-1].set_data(all_img_data[c-1][int(i * fps / fps_max)]) # Load frame, rounded down for 'slower' fps

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    save_animation(animate, fig, frames=max(len_dirs), fps=fps_max * playback_speed, dir=source_dir, title= "output")




if __name__ == "__main__":

    collect_time_ranges_csv()

    ## Static clips
    # collect_time_range("Static clip Patrick", "1:05:50", "1:05:52")
    # collect_time_range("Static clip Douglas", "2:01:06", "2:01:08")
    # collect_time_range("Static clip Gracie", "1:44:23", "1:44:27")

    # combined_labelled_clip("oscar play", playback_speed=0.25)

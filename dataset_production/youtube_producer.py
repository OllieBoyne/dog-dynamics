"""Scripts for producing Dataset 3 - a supplementary set of dog images extracted from YouTube videos"""

import os, csv, pytube, urllib

osp = os.path
ffmpeg = "ffmpeg"  # location of ffmpeg.exe, or just "ffmpeg" if in PATH


def download_youtube_clips(url, timestamps, download_only=False, fast_mode=True):
	"""Given a youtube link, and an array of timestamps to select between, save each clip as a separate mp4 in the directory ../youtube_clips/clip_title
	time_start, time_finish in format HH:mm:ss.xxx

	download_only = Only download desired full videos of off youtube - don't cut to clips
	fast_mode = Don't cut any clips that have been cut before - assumes no clips added to any existing entries."""

	out_dir = r"E:\iib_project_other\dog_youtube_clips"
	if "youtu.be" in url:
		title = url.split("/")[-1]
	else:
		title = url.split("/")[-1].split("=")[1]  # Take youtube code as title

	out_dir += f"\\{title}"  # new directory with title in

	# Download full video clip to correct directory
	def download_stream(yt, out_dir, stream_num=0, filename="raw"):
		# Download given stream from youtube video to output dir
		# Returns True if successfully downloaded

		try:
			video = yt.streams.filter(adaptive=True).all()[stream_num]  # Get first video-only (adaptive) clip
			video.download(output_path=out_dir, filename="raw")
			return True
		except urllib.error.HTTPError:  # Unsuccessfully downloaded stream
			return False

	# If video directory doesn't exist, make it
	if not osp.isdir(out_dir):  os.mkdir(out_dir)

	# If video file doesn't exist, download it
	if len([f for f in os.listdir(out_dir) if "raw" in f]) == 0:  # If no file with 'raw' in title
		yt = pytube.YouTube(url)  # Load youtube streams
		print("Loading Video:", f"({url})")

		for i in range(10):  # Try to download from first 10 available streams
			if download_stream(yt, out_dir, i):
				break

	raw_src = [f for f in os.listdir(out_dir) if "raw" in f][
		0]  # Get name of raw file, normally raw.mp4, sometimes raw.other_ext

	# If fast mode is on, don't extract clips if any existing clips in folder
	if not download_only and (not fast_mode or len(os.listdir(out_dir)) < 2):
		# Extract each clip
		for n, (time_start, time_finish) in enumerate(timestamps):

			# Convert to SS.xxx
			if ":" in time_start:   time_start = sum(float(i) * j for i, j in zip(time_start.split(":"), [3600, 60, 1]))
			if ":" in time_finish:  time_finish = sum(
				float(i) * j for i, j in zip(time_finish.split(":"), [3600, 60, 1]))

			time_start = float(time_start)
			time_finish = float(time_finish)

			src_out_clip = osp.join(out_dir, f"{n:02d}.mp4")

			# Extract clip
			commands = [ffmpeg, "-y",  # -y always overwrites
						"-i", f"\"{osp.join(out_dir, raw_src)}\"",
						"-ss", str(time_start),  # Get each video
						"-t", str(time_finish - time_start),  # Cut video to correct length
						src_out_clip]

			os.system(" ".join(commands))


def download_youtube_clips_from_csv(csv_loc, download_only=False, fast_mode=False):
	"""Given a csv file, for which the format is
	 URL 1
	 TIMESTAP 1 START, TIMESTAMP 1 END,
	  ...
	  URL 2
	  ...
	extracts the clips from each url into separate directories"""

	with open(csv_loc, "r") as infile:
		reader = csv.reader(infile)
		for n, line in enumerate(reader):
			if line[1] == "":  # if a url row

				# IF not on first line, submit current url + timestamps for clip extractions
				if n > 0:
					download_youtube_clips(url, timestamps, download_only=download_only, fast_mode=fast_mode)

				timestamps = []  # reset timestamp array
				url, _ = line  # heading is url
			else:
				timestamps.append(line)

		if url != "":
			download_youtube_clips(url, timestamps, download_only=download_only,
								   fast_mode=fast_mode)  # Download last video


def extract_youtube_frames(src_dir, fps=1.5, fast_mode=False):
	"""For every subfolder in dir, extracts frames from all the (non-raw) videos in the directory,
	and saves to output directory

	fast_mode = only extract frames from videos not already present in some form in the output directory. Used for rerendering dataset multiple times.
	Do NOT use when rendering final dataset"""

	# Clear current frame folder
	# for f in os.listdir(osp.join(src_dir, "output_frames")):
	#     os.remove(osp.join(src_dir, "output_frames", f))
	for folder in [f for f in os.listdir(src_dir) if "." not in f and f != "output_frames"]:
		if not fast_mode or not any(folder in file for file in os.listdir(osp.join(src_dir,
																				   "output_frames"))):  # If fast mode, skip if existing files with same url in output_frames
			for video in [v for v in os.listdir(osp.join(src_dir, folder)) if "raw" not in v]:
				video_title = video.split(".")[0]
				os.system(
					f"ffmpeg -i {osp.join(src_dir, folder, video)} -r {fps} \"{src_dir}\\output_frames\\{folder}_{video_title}_%05d.png\"")


if __name__ == "__main__":
	# PIPELINE FOR YOUTUBE VIDEOS:
	# download_youtube_clips_from_csv(r"E:\iib_project_other\dog_youtube_clips\dog_comps_timestamps.csv", download_only=True) # Download all yt videos
	# download_youtube_clips_from_csv(r"E:\iib_project_other\dog_youtube_clips\dog_comps_timestamps.csv", fast_mode=True) # download/trim to clips
	extract_youtube_frames(r"E:\iib_project_other\dog_youtube_clips", fast_mode=True)  # convert clips to frames

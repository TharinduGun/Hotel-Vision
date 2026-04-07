import subprocess
import os

# ===== SETTINGS =====
input_video = "D:/4/test2.mp4"
output_folder = "D:/4/video_clips2"
segment_time = 20   # seconds
# ====================

# create output folder
os.makedirs(output_folder, exist_ok=True)

output_pattern = os.path.join(output_folder, "clip_%03d.mp4")

command = [
    "ffmpeg",
    "-i", input_video,
    "-c", "copy",          # no re-encoding (very fast)
    "-map", "0",
    "-segment_time", str(segment_time),
    "-f", "segment",
    "-reset_timestamps", "1",
    output_pattern
]

subprocess.run(command)

print("Video successfully split into 20-second clips.")
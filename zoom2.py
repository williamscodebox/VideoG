import subprocess

subprocess.run([
    "ffmpeg",
    "-framerate", "60",
    "-i", "frames/frame_%04d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "zoom_text.mp4"
])

from mochi_model import MochiModel

mochi = MochiModel()

mochi.generate_video(
    prompt="raining, sea, cinematic, 4k, moody lighting",
    num_frames=49,
    guidance_scale=7.5,
    fps=8,
    out_path="outputs/mochi_raining_sea.mp4"
)

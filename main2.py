import torch
from diffusers import DiffusionPipeline
import numpy as np
import os
import imageio

def save_video(frames, path, fps=8, quality=8):
    safe_frames = []

    for i, f in enumerate(frames):
        f = np.asarray(f)
        f = np.squeeze(f)

        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)

        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)

        if f.ndim != 3 or f.shape[2] not in (1, 2, 3, 4):
            raise ValueError(f"Frame {i} has invalid shape: {f.shape}")

        safe_frames.append(f)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, safe_frames, fps=fps, quality=quality)
    print(f"Video saved → {path}")


# Load Mochi
pipe = DiffusionPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()
pipe.enable_sequential_cpu_offload()

prompt = "raining, sea, cinematic, 4k, moody lighting"

with torch.autocast("cuda"):
    out = pipe(
        prompt=prompt,
        num_frames=64,
        guidance_scale=7.5,
    )

# Save video safely
save_video(out.frames, "outputs/mochi_output.mp4", fps=8)

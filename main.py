import os
import torch
from diffusers import DiffusionPipeline
import imageio
import numpy as np
import cv2


# ============================================================
#   LOAD MOCHI PIPELINE (3060 SAFE)
# ============================================================

pipe = DiffusionPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.float16,
)

pipe.reset_device_map()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()


# ============================================================
#   SAFE RESOLUTION FOR RTX 3060
# ============================================================

# HEIGHT = 1280   # vertical
# WIDTH = 720     # horizontal

# HEIGHT = 768
# WIDTH = 432

HEIGHT = 512
WIDTH = 288

# HEIGHT = 384
# WIDTH = 216


# ============================================================
#   RUN INFERENCE
# ============================================================

prompt = "A duck wearing sunglasses floating in outer space, neon lighting, surreal, comedic, cartoon‑realistic style., 4k"

with torch.autocast("cuda"):
    out = pipe(
        prompt=prompt,
        num_frames=24,
        guidance_scale=7.5,
        height=HEIGHT,
        width=WIDTH,
    )

raw_frames = out.frames


# ============================================================
#   FRAME NORMALIZATION
# ============================================================

frames = []

for f in raw_frames:
    arr = np.asarray(f)

    if arr.ndim == 4:
        for i in range(arr.shape[0]):
            frames.append(arr[i])
        continue

    if arr.ndim == 3:
        frames.append(arr)
        continue

    if arr.ndim == 2:
        frames.append(np.stack([arr]*3, axis=-1))
        continue

    raise ValueError(f"Unexpected frame shape: {arr.shape}")


final_frames = []
for f in frames:
    f = np.squeeze(f)

    if f.dtype != np.uint8:
        f = (f * 255).clip(0, 255).astype(np.uint8)

    if f.ndim == 2:
        f = np.stack([f]*3, axis=-1)

    final_frames.append(f)


# ============================================================
#   UPSCALE TO 1080×1920 FOR YOUTUBE SHORTS
# ============================================================

TARGET_W = 1088
TARGET_H = 1920

final_frames_resized = [
    cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    for f in final_frames
]


# ============================================================
#   SAVE VIDEO
# ============================================================

os.makedirs("outputs", exist_ok=True)
imageio.mimwrite("outputs/mochi_output.mp4", final_frames_resized, fps=8, quality=8)

print("Saved to outputs/mochi_output.mp4")

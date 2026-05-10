import os
import torch
from diffusers import DiffusionPipeline
import imageio
import numpy as np
import cv2


# ============================================================
#   LOAD MOCHI PIPELINE (RTX 3060 SAFE)
# ============================================================

pipe = DiffusionPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.float16,
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()
pipe.unet.enable_gradient_checkpointing()   # stability + VRAM savings


# ============================================================
#   INTERNAL GENERATION RESOLUTION
# ============================================================

HEIGHT = 512
WIDTH = 288


# ============================================================
#   PROMPTS FOR CONTINUATION
# ============================================================

base_prompt = "A cat typing on a laptop, surreal, 4k"

prompts = [
    base_prompt,
    base_prompt + ", continue the same scene",
    base_prompt + ", same scene, maintain motion",
    base_prompt + ", same scene, consistent lighting and camera",
]


# ============================================================
#   GENERATION FUNCTION (CHUNKED)
# ============================================================

def generate_chunk(prompt, frames, seed):
    with torch.autocast("cuda", dtype=torch.float16):
        out = pipe(
            prompt=prompt,
            num_frames=frames,
            guidance_scale=7.5,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    return out.frames


# ============================================================
#   GENERATE 96 FRAMES (4 × 24)
# ============================================================

all_frames = []
seed = 1234

for i in range(4):
    print(f"Generating chunk {i+1}/4...")
    chunk = generate_chunk(prompts[i], 24, seed + i)
    all_frames.extend(chunk)


# ============================================================
#   FRAME NORMALIZATION
# ============================================================

frames = []

for f in all_frames:
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

TARGET_W = 1080
TARGET_H = 1920

final_frames_resized = [
    cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    for f in final_frames
]


# ============================================================
#   SAVE VIDEO
# ============================================================

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/mochi_96frames.mp4"

imageio.mimwrite(output_path, final_frames_resized, fps=8, quality=8)

print(f"Saved to {output_path}")

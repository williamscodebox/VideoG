import torch
import numpy as np
import imageio
import os
from diffusers import DiffusionPipeline


# ---------------------------------------------------------
# 1. Load Mochi pipeline (VRAM-safe for RTX 3060)
# ---------------------------------------------------------
def load_mochi():
    pipe = DiffusionPipeline.from_pretrained(
        "genmo/mochi-1-preview",
        torch_dtype=torch.float16,
    )

    # Force CPU load + safe offload
    pipe.reset_device_map()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    return pipe


# ---------------------------------------------------------
# 2. Flatten any nested batches from Mochi
# ---------------------------------------------------------
def flatten_frames(raw_frames):
    flat = []

    for f in raw_frames:
        arr = np.asarray(f)

        # Case: (B, H, W, C) — nested batch
        if arr.ndim == 4:
            for i in range(arr.shape[0]):
                flat.append(arr[i])
            continue

        # Case: (H, W, C)
        if arr.ndim == 3:
            flat.append(arr)
            continue

        # Case: (H, W) grayscale
        if arr.ndim == 2:
            flat.append(np.stack([arr] * 3, axis=-1))
            continue

        raise ValueError(f"Unexpected frame shape: {arr.shape}")

    return flat


# ---------------------------------------------------------
# 3. Normalize frames to uint8 RGB
# ---------------------------------------------------------
def normalize_frames(frames):
    out = []

    for f in frames:
        f = np.squeeze(f)

        # Convert float → uint8
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)

        # Ensure RGB
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)

        out.append(f)

    return out


# ---------------------------------------------------------
# 4. Save video safely
# ---------------------------------------------------------
def save_video(frames, path, fps=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8)
    print(f"Saved video → {path}")


# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def main():
    pipe = load_mochi()

    prompt = "raining, sea, cinematic, 4k, moody lighting"

    with torch.autocast("cuda"):
        out = pipe(
            prompt=prompt,
            num_frames=64,
            guidance_scale=7.5,
        )

    # Step 1: flatten weird Mochi batches
    frames = flatten_frames(out.frames)

    # Step 2: normalize dtype + RGB
    frames = normalize_frames(frames)

    # Step 3: save
    save_video(frames, "outputs/mochi_output.mp4", fps=8)


if __name__ == "__main__":
    main()

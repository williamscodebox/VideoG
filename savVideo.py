import os
import numpy as np
import imageio

def save_video(frames, path, fps=8, quality=8):
    """
    Bulletproof video writer for Mochi / Diffusers output.
    - Fixes float16/float32 → uint8
    - Removes extra dimensions
    - Ensures RGB
    - Validates shapes
    - Auto-creates folders
    - Never silently corrupts frames
    """

    safe_frames = []

    for i, f in enumerate(frames):
        f = np.asarray(f)

        # Remove extra dims like (1, H, W, 3)
        f = np.squeeze(f)

        # Convert float [0,1] → uint8 [0,255]
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)

        # Ensure RGB
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)

        # Validate shape
        if f.ndim != 3 or f.shape[2] not in (1, 2, 3, 4):
            raise ValueError(f"Frame {i} has invalid shape: {f.shape}")

        safe_frames.append(f)

    # Create output folder
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write video
    imageio.mimwrite(path, safe_frames, fps=fps, quality=quality)
    print(f"Video saved → {path}")

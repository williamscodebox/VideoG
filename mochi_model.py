import torch
from diffusers import DiffusionPipeline
import imageio
import os
from datetime import datetime


class MochiModel:
    def __init__(self):
        print("[Mochi] Loading model… this will take ~40–60 seconds on first run")

        self.pipe = DiffusionPipeline.from_pretrained(
            "genmo/mochi-1-preview",
            torch_dtype=torch.float16
        )

        # VRAM‑safe settings for RTX 3060
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()

        print("[Mochi] Model loaded successfully")

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 49,
        guidance_scale: float = 7.5,
        fps: int = 8,
        out_path: str = None
    ):
        """
        Generate a video using Mochi-1 Preview.
        """

        if out_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"mochi_{timestamp}.mp4"

        print(f"[Mochi] Generating {num_frames} frames…")
        print(f"[Mochi] Prompt: {prompt}")

        with torch.autocast("cuda"):
            result = self.pipe(
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
            )

        video = result.videos[0]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        imageio.mimwrite(out_path, video, fps=fps, quality=8)

        print(f"[Mochi] Video saved → {out_path}")
        return out_path

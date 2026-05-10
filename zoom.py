import os
import math
from PIL import Image

def generate_zoom_frames(
    input_png="outputs/text_frame.png",
    out_dir="frames",
    duration=3,
    fps=60,
    zoom_start=1.0,
    zoom_end=1.3,
):
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(input_png).convert("RGB")
    w, h = img.size

    total_frames = duration * fps

    for i in range(total_frames):
        t = i / total_frames

        # smooth ease-in-out
        eased = (1 - math.cos(t * math.pi)) / 2
        zoom = zoom_start + (zoom_end - zoom_start) * eased

        # scale
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        frame = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # crop back to original size (center crop)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        frame = frame.crop((left, top, left + w, top + h))

        frame.save(f"{out_dir}/frame_{i:04d}.png")

    print("Frames generated:", total_frames)

generate_zoom_frames()

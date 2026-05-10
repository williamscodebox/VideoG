import os
import math
import random
from PIL import Image, ImageFilter

def generate_zoom_frames(
    input_png="outputs/text_frame2.png",
    out_dir="frames",
    duration=3,
    fps=60,
    zoom_start=1.0,
    zoom_end=1.3,
    glow_radius=25,
    glow_intensity=1.8,
    wiggle_amount=6,   # pixels
):
    os.makedirs(out_dir, exist_ok=True)

    base = Image.open(input_png).convert("RGBA")
    w, h = base.size

    total_frames = duration * fps

    # --- Precompute glow layer ---
    glow = base.copy()
    glow = glow.filter(ImageFilter.GaussianBlur(glow_radius))

    # Boost glow brightness
    glow = Image.blend(glow, base, alpha=1/glow_intensity)

    for i in range(total_frames):
        t = i / total_frames

        # smooth ease-in-out
        eased = (1 - math.cos(t * math.pi)) / 2
        zoom = zoom_start + (zoom_end - zoom_start) * eased

        # scale
        new_w = int(w * zoom)
        new_h = int(h * zoom)

        frame_base = base.resize((new_w, new_h), Image.Resampling.BICUBIC)
        frame_glow = glow.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # wiggle offset
        wig_x = random.randint(-wiggle_amount, wiggle_amount)
        wig_y = random.randint(-wiggle_amount, wiggle_amount)

        # crop back to original size
        left = (new_w - w) // 2 + wig_x
        top = (new_h - h) // 2 + wig_y

        frame_base = frame_base.crop((left, top, left + w, top + h))
        frame_glow = frame_glow.crop((left, top, left + w, top + h))

        # composite glow + text
        final = Image.alpha_composite(frame_glow, frame_base)

        final.convert("RGB").save(f"{out_dir}/frame_{i:04d}.png")

    print("Frames generated:", total_frames)

generate_zoom_frames()
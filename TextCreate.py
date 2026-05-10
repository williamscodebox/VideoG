from PIL import Image, ImageDraw, ImageFont

def generate_text_frame(
    output_path="outputs/text_frame.png",
    width=4096,
    height=4096,
    bg_color=(0, 255, 255),  # neon cyan
    text_color=(0, 0, 0),
    font_path="arial.ttf",  # replace with your bold font
    font_size=620,
):
    lines = ["WHAT", "DID", "I", "JUST", "WATCH"]

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(font_path, font_size)

    # Calculate vertical layout
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_text_height = sum(line_heights) + (len(lines) - 1) * 40  # spacing

    y = (height - total_text_height) // 2

    for line, lh in zip(lines, line_heights):
        w = draw.textbbox((0, 0), line, font=font)[2]
        x = (width - w) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += lh + 40

    img.save(output_path)
    print(f"Saved PNG → {output_path}")

generate_text_frame()

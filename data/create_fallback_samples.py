"""
Generates fallback underwater sample images if local samples are needed immediately.
Creates realistic underwater gradient, caustic patterns, and geometric scene elements.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os


def generate_underwater_scene(filename: str, scene_type: str):
    width, height = 800, 600
    
    # 1. Base underwater depth gradient (lighter turquoise at surface, deeper navy-cyan at bottom)
    img_arr = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        depth_factor = y / height
        # Surface: [10, 110, 150], Deep: [2, 35, 70]
        r = 10 * (1.0 - depth_factor) + 2 * depth_factor
        g = 120 * (1.0 - depth_factor) + 40 * depth_factor
        b = 160 * (1.0 - depth_factor) + 75 * depth_factor
        img_arr[y, :, 0] = r
        img_arr[y, :, 1] = g
        img_arr[y, :, 2] = b

    # 2. Add simulated optical backscatter and caustic light ripples
    np.random.seed(42 if scene_type == "net" else 101)
    noise = np.random.normal(0, 12, (height, width, 3))
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # 3. Add seabed sandy benthos at bottom
    seabed_color = (25, 75, 95)
    draw.polygon([(0, 480), (width, 450), (width, height), (0, height)], fill=seabed_color)

    if scene_type == "net":
        # Draw coral mound on left
        draw.ellipse([40, 360, 320, 520], fill=(40, 110, 120), outline=(50, 130, 140))
        # Draw tangled discarded net on lower right
        for _ in range(35):
            x1 = np.random.randint(460, 720)
            y1 = np.random.randint(340, 530)
            x2 = x1 + np.random.randint(-40, 40)
            y2 = y1 + np.random.randint(-30, 30)
            draw.line([(x1, y1), (x2, y2)], fill=(160, 185, 195), width=2)
        # Small fish
        draw.ellipse([240, 210, 265, 222], fill=(180, 160, 40))
        draw.polygon([(265, 216), (275, 210), (275, 222)], fill=(180, 160, 40))

    elif scene_type == "drum":
        # Draw rusted submerged chemical drum on seabed
        draw.rectangle([480, 380, 660, 500], fill=(65, 80, 70), outline=(130, 75, 50), width=3)
        draw.ellipse([480, 365, 660, 395], fill=(75, 90, 80), outline=(140, 85, 55), width=3)
        # Coral branches
        draw.arc([100, 350, 250, 500], start=0, end=180, fill=(35, 125, 110), width=6)

    elif scene_type == "reef":
        # Colorful but bleached coral reef
        draw.ellipse([80, 300, 350, 510], fill=(170, 190, 195), outline=(190, 210, 215)) # Bleached cluster
        draw.ellipse([400, 320, 700, 520], fill=(30, 130, 120), outline=(40, 150, 140)) # Healthy brain coral
        # Diverse fish school
        for fx in [310, 350, 380, 420]:
            fy = 180 + (fx % 50)
            draw.ellipse([fx, fy, fx + 25, fy + 12], fill=(210, 150, 30))

    # Apply soft underwater haze blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    out_dir = r"C:\project\ALLBACKUP\AquaSentinel\data\samples"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    img.save(save_path, quality=92)
    print(f"Generated sample: {save_path}")


if __name__ == "__main__":
    generate_underwater_scene("ghost_net_reef.jpg", "net")
    generate_underwater_scene("sunken_drum_debris.jpg", "drum")
    generate_underwater_scene("coral_bleaching_reef.jpg", "reef")

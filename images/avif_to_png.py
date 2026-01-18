from PIL import Image
import os

def avif_to_png(avif_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(avif_path)[0] + ".png"

    with Image.open(avif_path) as img:
        # Ensure full precision is preserved
        img = img.convert("RGBA") if img.mode in ("P", "LA") else img
        img.save(output_path, format="PNG", optimize=False)

    print(f"Converted: {output_path}")

# Example usage
avif_to_png("../other/lock_screen3.avif")

from PIL import Image

def scale_down_to_720p(input_path, output_path):
    with Image.open(input_path) as img:
        # Preserve color depth & alpha
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")

        # SCALE DOWN using highest-quality filter
        scaled = img.resize(
            (1280, 720),
            resample=Image.LANCZOS
        )

        # Save losslessly
        scaled.save(output_path, format="PNG", optimize=False)

        print(f"Scaled down → {scaled.size}")

# Example
scale_down_to_720p("img1.png", "img1_output_720p.png")

import os
import re
from PIL import Image, ImageOps

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]

def images_to_pdf(
    image_folder="images",
    output_pdf="hand_notes.pdf",
    max_width=1600,
    jpeg_quality=70
):
    files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort(key=natural_sort_key)

    if not files:
        raise ValueError("No images found")

    images = []

    for file in files:
        path = os.path.join(image_folder, file)
        img = Image.open(path)

        # Fix EXIF orientation (phone images)
        img = ImageOps.exif_transpose(img)

        # 🔄 FORCE ROTATION:
        img = img.rotate(90, expand=True)

        # Convert to RGB
        img = img.convert("RGB")

        # Resize (keep aspect ratio)
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        images.append(img)

    images[0].save(
        output_pdf,
        save_all=True,
        append_images=images[1:],
        quality=jpeg_quality,
        optimize=True
    )

    print(f"✅ Compressed & rotated PDF created: {output_pdf}")

# =========================
# RUN
# =========================
images_to_pdf()

from PIL import Image
import numpy as np

img = Image.open("gl_log.png").convert("RGBA")
data = np.array(img)

r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

# Only remove pixels that are very close to pure black
# threshold of 15 — catches near-black background but not dark blocks
is_black = (r < 15) & (g < 15) & (b < 15)

data[:,:,3] = np.where(is_black, 0, a)

result = Image.fromarray(data)
result.save("gitlink_logo_transparent.png")
print("Saved gitlink_logo_transparent.png")
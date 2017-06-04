import numpy as np
from PIL import Image

IMAGLOC = ("images/pixelNN.png")

im = Image.open(IMAGLOC)
arr = np.array(im)
print(arr)

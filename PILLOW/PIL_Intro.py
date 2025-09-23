import numpy as np
from PIL import Image
import pandas as pd

pd.set_option("display.max_rows", 1000, "display.min_rows", 200, "display.max_columns", None, "display.width", None)

image = Image.open('Image.PILLOW/Beach_Buildings_VolleyballNet.jpg')

pixels = np.array(image)

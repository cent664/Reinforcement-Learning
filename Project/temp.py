import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = Image.open('84by84.png').convert('L')
im_data = np.asarray(im)
print(im_data.shape)

img = Image.fromarray(im_data, 'L')
img.show()
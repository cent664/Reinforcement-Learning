import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image
import cv2

im = Image.open("84by84.png")
im_data = np.asarray(im)
print(im_data.shape)

im = Image.open("84by84.png").convert('L')
im_data = np.asarray(im)
print(im_data.shape)

im = np.expand_dims(im, axis=0)
im = np.expand_dims(im, axis=0)

img = Image.fromarray(im, 'LA')
img.show()

im_data = np.asarray(im)
print(im_data.shape)
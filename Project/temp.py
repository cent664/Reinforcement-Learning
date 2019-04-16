import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image

im = Image.open("84by84.png").convert('L')
print(type(im))
im.show()

im_data = np.asarray(im)
print(im_data.shape)
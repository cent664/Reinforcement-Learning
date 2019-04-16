import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image

index = 0   
holdings = 0

observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

high = np.array([index, holdings])

action_space = spaces.Discrete(3)
observation_space = spaces.Box(-high, high, dtype=np.float32)

observation_space = observation_space.shape[0]
action_space = action_space.n

print(action_space, observation_space)
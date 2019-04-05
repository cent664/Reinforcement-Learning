import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# DQN Stocks

df = pd.read_csv("NFLX.csv")  # Reading the data

current_index = 4

for i in range(current_index - 4, current_index + 1):
    data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values

data = [1.2, 1.4, 2.9, 3.1]
plt.bar(1, data[1]-data[0], width=0.8, bottom=data[0], align='center', data=None)
plt.bar(1, data[2]-data[1], width=0.8, bottom=data[1], align='center', data=None)
plt.bar(1, data[3]-data[2], width=0.8, bottom=data[2], align='center', data=None)
data = [1.3, 1.6, 2.7, 3.5]
plt.bar(2, data[1]-data[0], width=0.8, bottom=data[0], align='center', data=None)
plt.bar(2, data[2]-data[1], width=0.8, bottom=data[1], align='center', data=None)
plt.bar(2, data[3]-data[2], width=0.8, bottom=data[2], align='center', data=None)
data = [1.1, 1.3, 2.5, 3.6]
plt.bar(3, data[1]-data[0], width=0.8, bottom=data[0], align='center', data=None)
plt.bar(3, data[2]-data[1], width=0.8, bottom=data[1], align='center', data=None)
plt.bar(3, data[3]-data[2], width=0.8, bottom=data[2], align='center', data=None)
data = [0.9, 1.8, 2.6, 3.7]
plt.bar(4, data[1]-data[0], width=0.8, bottom=data[0], align='center', data=None)
plt.bar(4, data[2]-data[1], width=0.8, bottom=data[1], align='center', data=None)
plt.bar(4, data[3]-data[2], width=0.8, bottom=data[2], align='center', data=None)
data = [1.4, 1.6, 2.1, 3.8]
plt.bar(5, data[1]-data[0], width=0.8, bottom=data[0], align='center', data=None)
plt.bar(5, data[2]-data[1], width=0.8, bottom=data[1], align='center', data=None)
plt.bar(5, data[3]-data[2], width=0.8, bottom=data[2], align='center', data=None)
plt.show()

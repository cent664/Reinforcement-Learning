import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# DQN Stocks

df = pd.read_csv("NFLX.csv")  # Reading the data

current_index = 3944

for i in range(current_index - 4, current_index + 1):
    data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values
    plt.bar(i + 1, data[1] - data[0], width=0.8, bottom=data[0], align='center', data=None, color='White', edgecolor='Black', linewidth=1.0)
    if data[2] > data[1]:
        plt.bar(i + 1, data[2] - data[1], width=0.8, bottom=data[1], align='center', data=None, color='Green', edgecolor='Black', linewidth=1.0)
    else:
        plt.bar(i + 1, data[1] - data[2], width=0.8, bottom=data[2], align='center', data=None, color='Red', edgecolor='Black', linewidth=1.0)
    plt.bar(i + 1, data[3] - data[2], width=0.8, bottom=data[2], align='center', data=None, color='White', edgecolor='Black', linewidth=1.0)

plt.show()

"""
for i in range(0, 4000):

    data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values

    if data[2] > data[1]:
        gap1 = data[1] - data[0]
        gap2 = data[2] - data[1]
        gap3 = data[3] - data[2]
    else:
        gap1 = data[2] - data[0]
        gap2 = data[1] - data[2]
        gap3 = data[3] - data[1]


    if ((gap1 > 2 and gap1 < 5) and (gap2 > 2 and gap2 < 5) and (gap3 > 2 and gap3 < 5)):
        print(data[0], data[1], data[2], data[3])
        print(gap1, gap2, gap3)
        print(i)
"""

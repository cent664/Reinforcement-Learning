import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image


class StockTradingEnv():
    """
    Description:
    State:
        Num	Observation                 Min         Max
        0   Image                       ?           ?
        1   Holdings                    ?           ?
        2   Volume                      0           ?

    Actions:
        Num	    Action
        0	    Sell
        1	    Hold
        2       Buy

    Reward:
        Difference in portfolio values = [p_current*h_current + b_current] - [p_old*h1_old + b1_old]

    Starting State:
        Price of the stock at the start of the time-series
    """

    def __init__(self):
        df = pd.read_csv("NFLX.csv")  # Reading the data

        self.data_close = df['Adj Close'].values
        self.data_volume = df['Volume'].values

        self.index = 5  # Initial state index
        self.holdings = 0  # Initial number of stocks I own
        self.window_size = 5  # Window size to consider when computing the state

        # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

        self.action_space = 3
        self.observation_space = (1, 84, 84)

    def compute_im(self, current_price_index, window_size):
        im = Image.open("84by84.png").convert('L')  # Converting to grayscale

        # Expanding from 2D to 4D since the CNN expends 4 dimensions
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)

        # Convering the image to np array
        im_data = np.asarray(im)
        return im_data

    def reset(self):
        # Compute the np array representation of the image at that index of size 'window_size'
        im = self.compute_im(5, self.window_size)

        volume = float(self.data_volume[5])  # Volume at starting index
        self.state = [im, self.holdings, volume]
        return self.state

    def step(self, action):
        im, holdings, volume = self.state

        if (action == 0):  # Sell
            new_holdings = holdings - 1

        if (action == 1):  # Hold
            new_holdings = holdings

        if (action == 2):  # Buy
            new_holdings = holdings + 1

        # Reward is price (difference x holdings) for the Adjusted Closing Price
        current_portfolio_value = (holdings*float(self.data_close[self.index]))
        new_portfolio_value = (new_holdings*float(self.data_close[self.index + 1]))

        reward = new_portfolio_value - current_portfolio_value

        self.index = self.index + 1  # Incrementing the window
        new_im = self.compute_im(self.index, self.window_size)
        new_volume = float(self.data_volume[self.index])

        done = False

        self.state = (new_im, new_holdings, new_volume)
        return np.array(self.state), reward, done, {}
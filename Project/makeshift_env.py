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
        0   Np array                    ?           ?
        1   Holdings                    ?           ?
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

        self.data = df['Adj Close'].values

        self.index = 0  # Initial state index
        self.holdings = 0
        self.window_size = 5

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def compute_metric(self, current_price_index, window_size):
        metric = float(self.data[current_price_index])  # For now, metric = current adjusted closing price
        return metric

    def reset(self):
        metric = self.compute_metric(self.index, self.window_size)
        self.state = [metric, self.holdings]
        return self.state

    def step(self, action):
        metric, holdings = self.state

        if (action == 0):  # Sell
            new_holdings = holdings - 1

        if (action == 1):  # Hold
            new_holdings = holdings

        if (action == 2):  # Buy
            new_holdings = holdings + 1

        current_portfolio_value = (holdings*float(self.data[self.index]))
        new_portfolio_value = (new_holdings*float(self.data[self.index + 1]))

        reward = new_portfolio_value - current_portfolio_value

        self.index = self.index + 1  # Incrementing the window
        new_metric = self.compute_metric(self.index, self.window_size)

        done = False

        self.state = (new_metric, new_holdings)
        return np.array(self.state), reward, done, {}
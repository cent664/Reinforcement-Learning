import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image
from np_array_data import compute_array, reduce_dim, coloring

class StockTradingEnv():
    """
    Description:
    State:
        Num	Observation                 Type/Context
        0   Image                       2D np array of static size
        1   Holdings                    (1 x 1) np vector containing holdings of current day
        2   Volume                      (1 x window_size) np vector containing volume of stocks for last window_size days

    Actions:
        Num	    Action
        0	    Sell
        1	    Hold
        2       Buy

    Reward:
        Difference in portfolio values = (p_current * h_current) - (p_old * h1_old)
        Difference in portfolio values = (close_current * h_current) - (open_current * h_current)

    Starting Price:
        Adjusted Closing Price of the stock at the start of the time-series
    """

    def __init__(self, mode):
        if mode == 'test':
            self.df = pd.read_csv("NFLX_test.csv")  # Reading the data
        elif mode == 'train':
            self.df = pd.read_csv("NFLX.csv")  # Reading the data

        self.data_date = self.df['Date'].values
        self.data_open = self.df['Open'].values
        self.data_close = self.df['Close'].values
        self.data_volume = self.df['Volume'].values

        self.window_size = 5  # Number of data points in the state
        self.index = self.window_size - 1  # Initial state index
        # TODO: Define long, short and hold. Is it the Holdings or the action (If you buy is it long and vice versa?)
        self.holdings = 0  # Initial number of stocks I own
        self.precision = 1  # Number of significant digits after the decimal
        self.static_image_size = (512, 40)  # Shape on input image into the CNN. Hard coded for now.

        self.action_space = 3
        self.observation_space = (1, 512, 40)

    def compute_im(self, current_price_index, window_size, mode):

        test_array = compute_array(mode, current_price_index, window_size, self.precision)
        test_array = reduce_dim(test_array)
        im_data = coloring(test_array, self.static_image_size)

        im_data = np.expand_dims(im_data, axis=0)
        im_data = np.expand_dims(im_data, axis=0)

        return im_data

    def reset(self, mode):
        # Compute the np array representation of the image at that index of size 'window_size'
        self.index = self.window_size - 1
        im = self.compute_im(self.index, self.window_size, mode)

        # TODO: Try removing volume/holdings
        # Volume at from starting index - 5 -> starting index
        volume = []
        for i in range(0, self.window_size - 1):
            volume.append(float(self.data_volume[i]))
        volume = np.reshape(volume, (1, -1))  # Reshaping along the right axis
        self.state = [im, np.asarray([[self.holdings]]), volume]
        return self.state

    def step(self, action, mode):

        im, holdings, volume = self.state

        if (action == 0):  # Sell
            new_holdings = holdings[0][0] - 1

        if (action == 1):  # Hold
            new_holdings = holdings[0][0]

        if (action == 2):  # Buy
            new_holdings = holdings[0][0] + 1

        # Reward is (price difference x holdings) for the Adjusted Closing Price
        current_portfolio_value = (holdings[0][0]*float(self.data_close[self.index]))
        new_portfolio_value = (new_holdings*float(self.data_close[self.index + 1]))

        # Reward is (price difference x holding_current) between open and close price
        # current_portfolio_value = (holdings[0][0] * float(self.data_open[self.index]))
        # new_portfolio_value = (holdings[0][0]*float(self.data_close[self.index]))

        reward = new_portfolio_value - current_portfolio_value

        # Saving the image
        if mode == 'test':
            actual_image = im[0][0]
            date = self.data_date[self.index]
            date = date.replace("/", "-")
            actual_image = Image.fromarray(np.uint8(actual_image), 'L')
            actual_image.save("Images/{}-{}-{}.bmp".format(action, str(round(reward,2)), date))

        self.index = self.index + 1  # Incrementing the window

        # Computing new image, volume array with respect to the new index
        new_im = self.compute_im(self.index, self.window_size, mode)

        new_volume = []
        for i in range(self.index - self.window_size + 1, self.index):
            new_volume.append(float(self.data_volume[i]))
        new_volume = np.reshape(new_volume, (1, -1))


        done = False

        self.state = (new_im, np.asarray([[new_holdings]]), new_volume)
        return self.state, reward, done, {}
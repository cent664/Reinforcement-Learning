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

    Actions:
        Num	    Action
        0	    Sell
        1	    Hold
        2       Buy

    Reward:
        Difference in portfolio values = position * (close_current - open_current)

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

        self.window_size = 16  # Number of data points in the state
        self.index = self.window_size - 1  # Initial state index
        self.precision = 1  # Number of significant digits after the decimal
        self.static_image_size = (64, self.window_size)  # Shape on input image into the CNN. Hard coded for now.

        self.action_space = 3
        self.observation_space = (1, 64, self.window_size)

    def compute_im(self, current_price_index, window_size, mode):

        if mode == 'train':
            df = pd.read_csv("NFLX.csv")  # Reading the data
        elif mode == 'test':
            df = pd.read_csv("NFLX_test.csv")  # Reading the data

        test_array = compute_array(df, mode, current_price_index, window_size, self.precision)
        test_array = reduce_dim(test_array)
        im_data = coloring(test_array, self.static_image_size)

        im_data = np.expand_dims(im_data, axis=0)
        im_data = np.expand_dims(im_data, axis=0)

        return im_data

    def reset(self, mode):

        # Compute the np array representation of the image at that index of size 'window_size'
        self.index = self.window_size - 1
        im = self.compute_im(self.index, self.window_size, mode)
        self.state = im

        return self.state

    def step(self, action, mode):

        im = self.state

        if (action == 0):  # Sell/Short
            position = -1.0

        if (action == 1):  # Hold/Nothing
            position = 0.0

        if (action == 2):  # Buy/Long
            position = +1.0

        # Reward is (current_position x price difference between close and open for the next day)

        reward = position * (float(self.data_close[self.index + 1]) - float(self.data_open[self.index + 1]))

        # Saving the image
        if mode == 'test':
            actual_image = im[0][0]
            date = self.data_date[self.index]
            date = date.replace("/", "-")
            actual_image = Image.fromarray(np.uint8(actual_image), 'L')
            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'
            actual_image.save("Images/{}ing-{}-{}.bmp".format(action_actual, date, str(round(reward,2))))

        self.index = self.index + 1  # Incrementing the window

        # Computing new image, volume array with respect to the new index
        new_im = self.compute_im(self.index, self.window_size, mode)

        self.state = new_im
        return self.state, reward, {}
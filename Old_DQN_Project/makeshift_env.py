import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image
from np_array_data import compute_array, reduce_dim, coloring


class StockTradingEnv:
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

    def __init__(self, mode, steps):
        if mode == 'Test':
            self.df = pd.read_csv("S&P500_test.csv")  # Reading the data
        elif mode == 'Train':
            self.df = pd.read_csv("S&P500_train.csv")  # Reading the data

        # Converting String to datetime
        self.data_date = self.df['Date']
        self.data_date = pd.to_datetime(self.data_date)
        self.data_date = self.data_date.dt.date

        self.data_open = self.df['Open'].values
        self.data_close = self.df['Close'].values
        self.data_low = self.df['Low'].values
        self.data_high = self.df['High'].values

        self.window_size = 16  # Number of data points in the state
        self.index = self.window_size - 1  # Initial state index
        self.static_image_size = (64, self.window_size)  # Shape on input image into the CNN. Hard coded for now.

        self.action_space = 3
        self.observation_space = (1, 64, self.window_size)

        # Calculating Scaling factor
        maxRange = -1000000
        for i in range(0, steps):
            maxRange = max(maxRange, max(self.data_high[i:i + self.window_size]) - min(self.data_low[i:i + self.window_size]))

        self.dollars_per_pixel = maxRange/64
        self.scaling_factor = 1 / self.dollars_per_pixel
        self.scaling_factor = self.scaling_factor / 2  # To account for the shift from centering close

    def compute_im(self, current_price_index, window_size):

        test_array = compute_array(self.df, current_price_index, window_size)
        test_array = reduce_dim(test_array, self.scaling_factor)
        im_data = coloring(test_array, self.static_image_size)

        im_data = np.expand_dims(im_data, axis=0)
        im_data = np.expand_dims(im_data, axis=0)

        return im_data

    def reset(self, mode):

        # Compute the np array representation of the image at that index of size 'window_size'
        self.index = self.window_size - 1
        im = self.compute_im(self.index, self.window_size)
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
        if mode == 'Test':
            actual_image = im[0][0]
            date = self.data_date[self.index]
            actual_image = Image.fromarray(np.uint8(actual_image), 'L')
            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            actual_image.save("Images/{}-{}ing-{}.bmp".format(date, action_actual, str(round(reward,2))))

        self.index = self.index + 1  # Incrementing the window

        # Computing new image, volume array with respect to the new index
        new_im = self.compute_im(self.index, self.window_size)

        self.state = new_im
        return self.state, reward, {}
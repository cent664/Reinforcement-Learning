import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from PIL import Image
from np_array_data import compute_array, reduce_dim, coloring
from Trends import get_trends
import os
import errno


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

    def __init__(self, mode, steps, stock, trend, window_size):
        self.stockname = stock
        self.trendname = trend
        self.filename = self.stockname + "_" + self.trendname

        # Reading the data
        if mode == 'Test':
            self.df_stocks = pd.read_csv(self.stockname + "_test.csv")
        elif mode == 'Train':
            self.df_stocks = pd.read_csv(self.stockname + "_train.csv")
        self.df_trends = pd.read_csv(self.trendname + "_candlesticks.csv")

        # Converting String to datetime
        self.data_stocks_date = self.df_stocks['Date']
        self.data_stocks_date = pd.to_datetime(self.data_stocks_date)
        self.data_stocks_date = self.data_stocks_date.dt.date
        self.data_trends_date = self.df_trends['Date']
        self.data_trends_date = pd.to_datetime(self.data_trends_date)
        self.data_trends_date = self.data_trends_date.dt.date

        self.data_stocks_open = self.df_stocks['Open'].values
        self.data_stocks_close = self.df_stocks['Close'].values
        self.data_stocks_low = self.df_stocks['Low'].values
        self.data_stocks_high = self.df_stocks['High'].values
        self.data_trends_open = self.df_trends['Open'].values
        self.data_trends_close = self.df_trends['Close'].values
        self.data_trends_low = self.df_trends['Low'].values
        self.data_trends_high = self.df_trends['High'].values

        self.window_size = window_size  # Number of data points in the state
        self.index = self.window_size - 1  # Initial state index (Add ints to shift the starting index)
        self.static_image_size = (self.window_size, self.window_size)  # Shape of the input image into the CNN

        self.action_space = 3
        self.observation_space = (2, self.window_size, self.window_size)

        # Calculating Scaling factor for stocks
        stocks_maxRange = -1000000
        for i in range(0, steps):
            stocks_maxRange = max(stocks_maxRange, max(self.data_stocks_high[i:i + self.window_size]) - min(self.data_stocks_low[i:i + self.window_size]))

        self.stocks_dollars_per_pixel = stocks_maxRange/self.window_size
        self.stocks_scaling_factor = 1 / self.stocks_dollars_per_pixel
        self.stocks_scaling_factor = self.stocks_scaling_factor / 2  # To account for the shift from centering close

        # Calculating Scaling factor for trends
        trends_maxRange = -1000000
        for i in range(0, steps):
            trends_maxRange = max(trends_maxRange, max(self.data_trends_high[i:i + self.window_size]) - min(self.data_trends_low[i:i + self.window_size]))

        self.trends_dollars_per_pixel = trends_maxRange / self.window_size
        self.trends_scaling_factor = 1 / self.trends_dollars_per_pixel
        self.trends_scaling_factor = self.trends_scaling_factor / 2  # To account for the shift from centering close

        # Calculating the date range of the data in question (used when saving graphs/files)
        self.start_stocks_date = self.df_stocks['Date'][self.window_size]
        self.end_stocks_date = self.df_stocks['Date'][self.window_size + steps]
        self.date_stocks_range = self.start_stocks_date + " to " + self.end_stocks_date
        self.date_stocks_range = self.date_stocks_range.replace('/', '-')

    def compute_im(self, current_price_index, window_size):

        # Stock image
        test_array = compute_array(self.df_stocks, current_price_index, window_size)
        test_array = reduce_dim(test_array, self.stocks_scaling_factor)
        stocks_im_data = coloring(test_array, self.static_image_size)

        # Trend image
        test_array = compute_array(self.df_trends, current_price_index, window_size)
        test_array = reduce_dim(test_array, self.trends_scaling_factor)
        trends_im_data = coloring(test_array, self.static_image_size)

        trends_im_data = np.expand_dims(trends_im_data, axis=0)
        stocks_im_data = np.expand_dims(stocks_im_data, axis=0)

        im_data = np.concatenate([stocks_im_data, trends_im_data], axis=0)
        im_data = np.expand_dims(im_data, axis=0)

        return im_data

    def reset(self):
        self.index = self.window_size - 1  # Initial state index (Add ints to shift the starting index)

        # Compute the np array representation of the image at that index of size 'window_size'
        im = self.compute_im(self.index, self.window_size)
        self.state = im

        return self.state, self.date_stocks_range, self.stockname

    def step(self, action, mode):

        im = self.state

        if action == 0:  # Sell/Short
            position = -1.0

        if action == 1:  # Hold/Nothing
            position = 0.0

        if action == 2:  # Buy/Long
            position = +1.0

        # Reward is (current_position x price difference between close and open for the next day)
        reward = position * (float(self.data_stocks_close[self.index + 1]) - float(self.data_stocks_open[self.index + 1]))

        # Saving the image
        if mode == 'Test':
            # [0][0] for one the first image, [0][1] for the second
            stock_image = im[0][0]
            trend_image = im[0][1]
            stock_image = Image.fromarray(np.uint8(stock_image), 'L')
            trend_image = Image.fromarray(np.uint8(trend_image), 'L')

            date = self.data_stocks_date[self.index]

            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            # Creating directory if it doesn't exist
            graphpath = 'Results/{} ({}). Window Size - {}/Test Images/'.format(self.stockname, self.date_stocks_range,
                                                                                self.window_size)
            if not os.path.exists(os.path.dirname(graphpath)):
                try:
                    os.makedirs(os.path.dirname(graphpath))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            stock_image.save(graphpath + "stock_{}-{}ing-{}.bmp".format(date, action_actual, str(round(reward, 2))))
            trend_image.save(graphpath + "trend_{}-{}ing-{}.bmp".format(date, action_actual, str(round(reward, 2))))
        self.index = self.index + 1  # Incrementing the window

        # Computing new image, volume array with respect to the new index
        new_im = self.compute_im(self.index, self.window_size)

        self.state = new_im
        return self.state, reward, {}
import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np
from Candlesticks import candle

# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, action, episode, window_size, mode):

    if mode == 'train':
        # Plotting cumulative reward
        plt.plot(steps, rewardsum, label='Episode {}'.format(episode))
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Reward ($)')
        plt.title('Training: Days vs Cumulative Reward - Window size {}'.format(window_size))

    if mode == 'test':

        # CUMULATIVE REWARD
        plt.subplot(4, 1, 1)
        plt.plot(steps, rewardsum)
        # plt.xlabel('Days')
        plt.ylabel('Cumulative Reward ($)')
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.title('Days vs Cumulative Reward - Window size {}'.format(window_size))
        plt.grid(True)

        rewards = rewards[0: len(steps) - 1]
        rewards.insert(0, 0)

        # IMMEDIATE REWARD
        plt.subplot(4, 1, 2)
        plt.plot(steps, rewards, label='Immediate Reward')  # Step
        plt.legend(loc='upper right')
        # plt.xlabel('Days')
        plt.ylabel('Immediate Reward ($)')
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.title('Days vs Immediate Reward - Window size {}'.format(window_size))
        plt.grid(True)

        # Getting close and open prices
        df = pd.read_csv("S&P500_test.csv")
        data_close = df['Close'].values
        data_close = np.asarray(data_close)
        price_c = data_close[window_size - 1: window_size + len(steps) - 1]

        data_open = df['Open'].values
        data_open = np.asarray(data_open)
        price_o = data_open[window_size - 1: window_size + len(steps) - 1]

        color = ['yellow']  # Setting the initial position to hold

        position = action
        for i in range(0, len(steps) - 1):  # Discarding last entry
            if position[i] == 0:  # Sell
                color.append('red')
            if position[i] == 1:  # Hold
                color.append('yellow')
            if position[i] == 2:  # Buy
                color.append('blue')

        # STOCK PRICES
        plt.subplot(4, 1, 3)
        plt.plot(steps, price_c, label='Close price')
        plt.plot(steps, price_o, label='Open price')
        plt.scatter(steps, price_o, c=color, s=50)
        plt.grid(True)

        plt.legend(loc='upper right')
        # plt.xlabel('Days')
        plt.ylabel('Close and Open Prices ($)')
        plt.title('Prices and Position at each day')
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.subplot(4, 1, 4)
        # candle()

        # Writing down the actions taken
        if episode == 1:
            f = open("action.txt", "w")
        else:
            f = open("action.txt", "a")

        f.write("Episode = {}.\n\n".format(episode))

        for i in range(0, len(steps)):
            f.write("Step = {}. Action = {}\n".format(steps[i], action[i]))
        f.write("\n")
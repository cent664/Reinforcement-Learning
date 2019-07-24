import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, action, episode, window_size, mode):

    if mode == 'train':
        # Plotting cumulative reward
        plt.plot(steps, rewardsum, label='Episode {}'.format(episode))
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Reward')
        plt.title('Training: Days vs Cumulative Reward - Window size {}'.format(window_size))

    if mode == 'test':
        # Plotting cumulative reward
        plt.plot(steps, rewardsum)
        plt.xlabel('Days')
        plt.ylabel('Cumulative Reward')
        plt.title('Testing: Days vs Cumulative Reward - Window size {}'.format(window_size))
        plt.show()

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

        plt.subplot(2, 1, 1)
        plt.plot(steps, price_c, label='Close price')
        plt.plot(steps, price_o, label='Open price')
        plt.scatter(steps, price_o, c=color, s=50)
        plt.grid(True)

        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Close and Open Prices ($)')
        plt.title('Testing: Position at each day')

        rewards = rewards[0: len(steps) - 1]
        rewards.insert(0, 0)
        # Plotting the Immediate reward
        plt.subplot(2, 1, 2)
        plt.plot(steps, rewards, label='Immediate Reward') # Step
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Immediate Reward ($)')
        plt.title('Testing: Days vs Immediate Reward - Window size {}'.format(window_size))
        plt.grid(True)

        # Writing down the actions taken
        if episode == 1:
            f = open("action.txt", "w")
        else:
            f = open("action.txt", "a")

        f.write("Episode = {}.\n\n".format(episode))

        for i in range(0, len(steps)):
            f.write("Step = {}. Action = {}\n".format(steps[i], action[i]))
        f.write("\n")
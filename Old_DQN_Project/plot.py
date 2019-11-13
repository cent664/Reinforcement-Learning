import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np
from Candlesticks import candle
import os
import os
import errno


# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, actions, episode, window_size, date_range, filename, mode):

    graphpath = 'Results/{} ({}). Window Size - {}/'.format(filename, date_range, window_size)

    if mode == 'Train':

        df = pd.read_csv(filename + '_train.csv')

        # ------------------------------------------ CUMULATIVE REWARD ------------------------------------------
        # fig = plt.figure(figsize=(16.5, 10.0))
        plt.plot(steps, rewardsum, label='Episode {}'.format(episode))
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Reward ($)')
        plt.title('Training: Days vs Cumulative Reward - Window size {}'.format(window_size))

        # Creating directory if it doesn't exist
        if not os.path.exists(os.path.dirname(graphpath)):
            try:
                os.makedirs(os.path.dirname(graphpath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # Saving the graph
        plt.savefig(graphpath + '{}ing.png'.format(mode))

        # ------------------------------------------ SAVING THE ACTIONS ------------------------------------------
        saving_actions(episode, graphpath, steps, actions, rewards, mode)

    if mode == 'Test':

        df = pd.read_csv(filename + '_test.csv')

        # ------------------------------------------ 1. CUMULATIVE REWARD ------------------------------------------
        fig = plt.figure(figsize=(16.5, 10.0))

        ax1 = plt.subplot2grid((4, 1), (0, 0))
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
        plt.scatter(steps, rewardsum, s=50)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(len(steps) + 3))
        plt.grid(True)

        # ------------------------------------------ 2. IMMEDIATE REWARD ------------------------------------------

        ax2 = plt.subplot2grid((4, 1), (1, 0))
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
        plt.scatter(steps, rewards, s=50)
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(len(steps) + 3))
        plt.grid(True)

        # ------------------------------------------ 3. STOCK PRICES ------------------------------------------

        # Getting close and open prices
        data_close = df['Close'].values
        data_close = np.asarray(data_close)
        price_c = data_close[window_size: window_size + len(steps)]

        data_open = df['Open'].values
        data_open = np.asarray(data_open)
        price_o = data_open[window_size: window_size + len(steps)]

        color = []

        position = actions
        for i in range(0, len(steps)):
            if position[i] == 0:  # Sell
                color.append('red')
            if position[i] == 1:  # Hold
                color.append('yellow')
            if position[i] == 2:  # Buy
                color.append('blue')

        ax3 = plt.subplot2grid((4, 1), (2, 0))
        plt.plot(steps, price_c, label='Close price')
        plt.plot(steps, price_o, label='Open price')
        plt.scatter(steps, price_o, c=color, s=50)
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(len(steps) + 3))
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

        # ------------------------------------------ 4. CANDLESTICKS ------------------------------------------

        start = window_size

        # Converting date to pandas datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        og_dates = []

        # Converting datetime objects to the correct date format
        for i in range(0, len(df['Date'])):
            og_dates.append(df['Date'][i].strftime('%d-%m-%Y'))

        df["Date"] = df["Date"].apply(mdates.date2num)

        # Creating required data in new DataFrame OHLC
        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']]

        # In case you want to check for shorter timespan
        ohlc = ohlc[start: start + len(steps)]
        ohlc = ohlc.values
        ohlc = ohlc.tolist()

        # Replacing x axis values with consecutive dates to eliminate gaps
        for i in range(0, len(steps)):
            ohlc[i][0] = ohlc[0][0] + i
            ohlc[i] = tuple(ohlc[i])

        ax4 = plt.subplot2grid((4, 1), (3, 0))

        # Plot the candlesticks
        candlestick_ohlc(ax4, ohlc, width=.6)

        for label in ax4.xaxis.get_ticklabels():
            label.set_rotation(45)

        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(len(steps) + 3))
        ax4.grid(True)

        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('S&P500')

        # Replacing x labels with the correct dates
        locs, labels = plt.xticks()
        locs = locs[1:]
        plt.xticks(locs, list(og_dates[start - 1: start + len(steps) + 1]))
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

        # Creating directory if it doesn't exist
        if not os.path.exists(os.path.dirname(graphpath)):
            try:
                os.makedirs(os.path.dirname(graphpath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # Saving the graph
        plt.savefig(graphpath + '{}ing.png'.format(mode))

        # ------------------------------------------ SAVING THE ACTIONS ------------------------------------------
        saving_actions(episode, graphpath, steps, actions, rewards, mode)


def saving_actions(episode, graphpath, steps, actions, rewards, mode):
    filename = 'Actions_taken'
    if mode == 'Train':
        filename += '_train.txt'
    else:
        filename += '_test.txt'

    if episode == 1:
        f = open(graphpath + filename, "w")
    else:
        f = open(graphpath + filename, "a")

    f.write("Episode = {}.\n\n".format(episode))

    for i in range(0, len(steps)):
        if actions[i] == 0:
            f.write("Step = {}. Selling. Immediate Reward = {}\n".format(steps[i], rewards[i]))
        if actions[i] == 1:
            f.write("Step = {}. Holding. Immediate Reward = {}\n".format(steps[i], rewards[i]))
        if actions[i] == 2:
            f.write("Step = {}. Buying. Immediate Reward = {}\n".format(steps[i], rewards[i]))
    f.write("\n")
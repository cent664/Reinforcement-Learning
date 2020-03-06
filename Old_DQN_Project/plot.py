import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import numpy as np
from Candlesticks import candle
import os
import errno
from np_array_data import compute_array, reduce_dim, coloring, make_graph


# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, actions, episode, window_size, date_range, filename, mode):

    # Getting the current date of prediction for folder name
    folder_name = str(datetime.datetime.date(datetime.datetime.now()) - datetime.timedelta(days=1))
    graphpath = 'Results/{}/{} ({}). Window Size - {}/'.format(folder_name, filename, date_range, window_size)
    stockname, _ = filename.split("_")

    if mode == 'Train':
        df = pd.read_csv('{}_Stock.csv'.format(stockname))
        df = df[len(df) - (window_size + len(steps) + 1): len(df) - 1]

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
        df = pd.read_csv('{}_Stock.csv'.format(stockname))
        df = df[len(df) - (window_size + len(steps)): len(df)]  # Last 'window size' number of days, including today

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
        ax4 = plt.subplot2grid((4, 1), (3, 0))

        test_array = []

        # Creating the array
        for i in range(window_size, window_size + len(steps)):
            if i < 0:
                print("Indexing error")
                break

            data = df[['Low', 'Close', 'Open', 'High']].iloc[i].values

            low = data[0]
            close = data[1]
            open = data[2]
            high = data[3]

            test_array.append([high, close, open, low])
        test_array = np.transpose(test_array)

        # Converting date to pandas datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        og_dates = []

        # Converting datetime objects to the correct date format
        for i in range(window_size, window_size + len(steps)):
            og_dates.append((df['Date'].iloc[i]).strftime('%d-%m-%Y'))

        # Graph drawing parameters
        w = 0.6
        lw = 0.5

        for j in range(0, len(test_array[0])):  # 0 -> Total number of columns
            for i in range(0, len(test_array)):  # 0 -> Total number of rows

                low = test_array[len(test_array) - 1][j]
                open = test_array[len(test_array) - 2][j]
                close = test_array[len(test_array) - 3][j]
                high = test_array[len(test_array) - 4][j]

            # Coloring the graph based on open and close differences
            if close > open:
                ax4.bar(og_dates[j], close - open, width=w, bottom=open, color='Black',
                        linewidth=lw)
            else:
                ax4.bar(og_dates[j], open - close, width=w, bottom=close, color='Red',
                        linewidth=lw)

        for label in ax4.xaxis.get_ticklabels():
            label.set_rotation(45)

        ax4.xaxis.set_major_locator(mticker.MaxNLocator(len(steps) + 3))
        ax4.grid(True)

        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('S&P500')

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
    filename += "_{}.txt".format(mode)

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
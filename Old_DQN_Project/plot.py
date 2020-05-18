import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import errno
from matplotlib.ticker import MaxNLocator


# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, actions, episode, window_size, date_range, filename, folder_name, mode, compile):
    stockname, trendname = filename.split("_")
    # Getting the current date of prediction for folder name
    if not compile:
        graphpath = 'Results_{}_{}/{}/{} ({}). Window Size - {}/'.format(stockname, trendname, folder_name, filename, date_range, window_size)
    else:  # if compiling results, save under main folder
        graphpath = 'Results_{}_{}/{}/'.format(stockname, trendname, folder_name)
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
        saving_actions(episode, rewardsum, graphpath, steps, actions, rewards, mode)

    if mode == 'Test':
        df = pd.read_csv('{}_Stock.csv'.format(stockname))
        df = df[len(df) - (window_size + len(steps)): len(df)]  # Last 'window size' number of days, including today

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16.5, 10.0))

        # ------------------------------------------ 1. CUMULATIVE REWARD ------------------------------------------
        ax1.plot(steps, rewardsum)
        # plt.xlabel('Days')
        ax1.set_ylabel('Cumulative Reward ($)')
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax1.set_title('Days vs Cumulative Reward - Window size {}'.format(window_size))
        ax1.scatter(steps, rewardsum, s=50)
        ax1.grid(True)

        # ------------------------------------------ 2. IMMEDIATE REWARD ------------------------------------------
        ax2.plot(steps, rewards, label='Immediate Reward')  # Step
        ax2.legend(loc='upper right')
        # plt.xlabel('Days')
        ax2.set_ylabel('Immediate Reward ($)')
        ax2.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax2.set_title('Days vs Immediate Reward - Window size {}'.format(window_size))
        ax2.scatter(steps, rewards, s=50)
        ax2.grid(True)

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

        ax3.plot(steps, price_c, label='Close price')
        ax3.plot(steps, price_o, label='Open price')
        ax3.scatter(steps, price_o, c=color, s=50)
        ax3.grid(True)

        ax3.legend(loc='upper right')
        # plt.xlabel('Days')
        ax3.set_ylabel('Close and Open Prices ($)')
        ax3.set_title('Prices and Position at each day')
        ax3.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        # ------------------------------------------ 4. CANDLESTICKS ------------------------------------------

        start = window_size
        closes = []
        opens = []
        col = []
        bottoms = []
        colors = []
        w = 0.6

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

            closes.append(close)
            opens.append(open)

        # Converting date to pandas datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        og_dates = [(df['Date'].iloc[window_size - 1]).strftime('%d-%m-%Y')]

        # Converting datetime objects to the correct date format
        for i in range(window_size, window_size + len(steps)):
            og_dates.append((df['Date'].iloc[i]).strftime('%d-%m-%Y'))

        # Graph drawing parameters
        lw = 0.5

        for i in range(0, len(closes)):  # 0 -> Total number of columns
            col.append(abs(closes[i] - opens[i]))
            if closes[i] > opens[i]:
                bottoms.append(opens[i])
                colors.append('Black')
            else:
                bottoms.append(closes[i])
                colors.append('Red')

        ax4.bar(steps, col, bottom=bottoms, color=colors, width=w)
        ax4.set_xticklabels(og_dates)
        for label in ax4.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.set_xticks(np.arange(len(og_dates)))  # Forcing every label

        ax4.grid(True)

        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price ($)')
        ax4.set_title('S&P500')

        # Creating directory if it doesn't exist
        if not os.path.exists(os.path.dirname(graphpath)):
            try:
                os.makedirs(os.path.dirname(graphpath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Saving the graph
        plt.savefig(graphpath + '{}ing.png'.format(mode))
        # plt.show()

        # ------------------------------------------ SAVING THE ACTIONS ------------------------------------------
        saving_actions(episode, rewardsum, graphpath, steps, actions, rewards, mode)


def saving_actions(episode, rewardsum, graphpath, steps, actions, rewards, mode):
    filename = 'Actions_taken'
    filename += "_{}.txt".format(mode)

    if episode == 1:
        f = open(graphpath + filename, "w")
    else:
        f = open(graphpath + filename, "a")

    f.write("Episode = {}.\n\n".format(episode))

    # Counting profitability
    poscount = 0
    negcount = 0
    zerocount = 0

    for i in range(0, len(steps)):
        if rewards[i] > 0:
            poscount += 1
        elif rewards[i] < 0:
            negcount += 1
        else:
            zerocount += 1

        if actions[i] == 0:
            f.write("Step = {}. Selling. Immediate Reward = {}. Cumulative Reward = {}\n".format(steps[i], rewards[i], rewardsum[i]))
        if actions[i] == 1:
            f.write("Step = {}. Holding. Immediate Reward = {}. Cumulative Reward = {}\n".format(steps[i], rewards[i], rewardsum[i]))
        if actions[i] == 2:
            f.write("Step = {}. Buying. Immediate Reward = {}. Cumulative Reward = {}\n".format(steps[i], rewards[i], rewardsum[i]))
    f.write("\n")
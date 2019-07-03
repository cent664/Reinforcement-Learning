import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        df = pd.read_csv("NFLX_test.csv")
        data_close = df['Close'].values
        data_close = np.asarray(data_close)
        price_c = data_close[window_size - 1: window_size + len(steps) - 1]

        data_open = df['Open'].values
        data_open = np.asarray(data_open)
        price_o = data_open[window_size - 1: window_size + len(steps) - 1]

        color = []

        position = action  # TODO: action or holdings?
        for i in range(0, len(steps)):
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
        plt.ylabel('Close and Open Prices')
        plt.title('Testing: Position at each day')
        # plt.show()

        # Plotting the Immediate reward
        plt.subplot(2, 1, 2)
        plt.step(steps, rewards, label='Immediate Reward')
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Immediate Reward')
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
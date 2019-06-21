import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from np_array_data import compute_array, make_graph

# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, rewards, action, holdings, episode, window_size, mode):

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

        # Getting close prices
        df = pd.read_csv("NFLX_test.csv")
        data_close = df['Close'].values
        data_close = np.asarray(data_close)
        price = data_close[0:len(steps)]

        # Uncomment for candlesticks
        # current_index = len(steps) - 1
        # precision = 3
        # test_array = compute_array(mode, current_index, len(steps), precision)
        # # test_array = reduce_dim(test_array)
        # plt.subplot(3, 1, 1)
        # make_graph(test_array, current_index, len(steps))

        color = []

        # position = action  # TODO: action or holdings?
        # for i in range(0, len(steps)):
        #     if position[i] == 0:  # Sell
        #         color.append('red')
        #     if position[i] == 1:  # Hold
        #         color.append('yellow')
        #     if position[i] == 2:  # Buy
        #         color.append('green')

        position = holdings  # TODO: action or holdings?
        for i in range(0, len(steps)):
            if position[i] < 0:  # Short
                color.append('red')
            if position[i] == 0:  # None
                color.append('yellow')
            if position[i] > 0:  # Long
                color.append('green')

        plt.subplot(2, 1, 1)
        plt.plot(steps, price, label='Close price')
        plt.scatter(steps, price, c=color, s=50)

        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Close Price')
        plt.title('Testing: Position at each day')
        # plt.show()

        # Plotting the Immediate reward
        plt.subplot(2, 1, 2)
        plt.plot(steps, rewards, label='Immediate Reward')
        plt.legend(loc='upper right')
        plt.xlabel('Days')
        plt.ylabel('Immediate Reward')
        plt.title('Testing: Days vs Immediate Reward - Window size {}'.format(window_size))

        # Writing down the actions taken
        if episode == 1:
            f = open("action.txt", "w")
        else:
            f = open("action.txt", "a")

        f.write("Episode = {}.\n\n".format(episode))

        for i in range(0, len(steps)):
            f.write("Step = {}. Action = {}\n".format(steps[i], action[i]))
        f.write("\n")
import numpy as np
import random
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import load_img
from makeshift_env import StockTradingEnv
import matplotlib.pyplot as plt
from plot import twodplot
from utils import timeit
from PIL import Image
import os
import errno
# from Stocks import scraping
from Trends import get_trends
from Rename import rename_file
from Cleaning import convert_and_clean
import pandas as pd
import datetime

# Parameters

gamma = 0.95  # Discount factor
learning_rate = 0.00025  # Learning rate

# TODO: Play with memory and batch size
memory_size = 1000  # Memory pool for experience replay, forgets older values as the size is exceeded
batch_size = 32  # Batch size for random sampling in the memory pool

# TODO: Play with exploration rate and decay
exploration_max = 1.0  # Initial exploration rate
exploration_min = 0.1  # Min value of exploration rate post decay
# exploration_steps = 0.95 * memory_size
# exploration_decay = (exploration_max - exploration_min)/exploration_steps  # Exploration rate decay rate
exploration_decay = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space, mode):
        print('Instantiating the network...')

        self.exploration_rate = exploration_max  # Sets initial exploration rate to max
        self.old_episode = 0  # To check for ep changes (needed to decay exploration rate)

        self.action_space = action_space  # Action space = 3 (Sell, Hold, Buy)
        self.memory = deque(maxlen=memory_size)  # Will forget old values as new ones are appended

        # Defining the network structure
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 4, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        if mode == 'Test':
            # Loading weights here
            print('Loading weights...')
            self.exploration_rate = 0
            self.model.load_weights('CNN_DQN_weights.h5', by_name=True)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))  # Remembering instances in memory for future use

    def act(self, state):
        # Random action -> 0, 1, 2
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        # Q values based on model prediction on current state (Initially based on random weights)
        q_values = self.model.predict(state)

        return np.argmax(q_values[0])  # Argmax of tuple of 3 Q values, one for each action

    def experience_replay(self, episode):

        if len(self.memory) < batch_size:  # If has enough memory obtained, perform random batch sampling among those
            return

        batch = random.sample(self.memory, batch_size)  # Get a random batch
        for state, action, reward, state_next in batch:

            # Obtain Q value based on immediate reward and predicted q* value of next state
            q_update = reward + gamma * np.amax(self.model.predict(state_next))

            q_values = self.model.predict(state)  # Obtain q value tuple for that state
            q_values[0][action] = q_update  # Update the q value for that state, action (one that we took)
            # Update the weights of the network based on the updated q value (based on immediate reward)
            self.model.fit(state, q_values, epochs=1, verbose=0)

        # Decaying Exploration rate at every step
        self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum

        # Saving the weights
        self.model.save_weights('CNN_DQN_weights.h5')

        # # To decay the exploration rate if the episode changes
        # if episode != self.old_episode:
        #     self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        #     self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum
        # self.old_episode = episode

    def instantiate_load_and_return_model(self):
        self.model.load_weights('CNN_DQN_weights.h5', by_name=True)
        return self.model


@timeit
def DQN_Agent(mode, stock, trend, date, window_size):

    # ------------------------------------------------ TRAINING --------------------------------------------------------
    if mode == 'Train':
        episodes = 100
        steps = 200

        print('\nTraining...\n')

        # DQN Stocks
        env = StockTradingEnv(mode, steps, stock, trend, date, window_size)  # Object of the environment

        # Get action and observation space
        observation_space = env.observation_space
        action_space = env.action_space
        window_size = env.window_size

        # Object for the solver
        dqn_solver = DQNSolver(observation_space, action_space, mode)

        episode = 1
        score = [0]

        # Running for a number of episodes
        while episode <= episodes:

            #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
            state, date_range, filename = env.reset()  # Get initial state
            step = 1
            cumulative_reward = 0

            # To append step, cumulative reward, corresponding action to plot for each episode
            step_x = []
            cumulative_reward_y1 = []
            rewards_y2 = []
            actions_taken = []

            #  Going through time series data
            while step <= steps:
                action = dqn_solver.act(state)  # Get action based on argmax of the Q value approximation from the NN
                state_next, reward, info = env.step(action, mode)

                cumulative_reward += reward

                if action == 0:
                    action_actual = 'Sell'
                if action == 1:
                    action_actual = 'Hold'
                if action == 2:
                    action_actual = 'Buy'

                step_x.append(step)
                cumulative_reward_y1.append(cumulative_reward)
                rewards_y2.append(reward)
                actions_taken.append(action)

                # print("{} {}ing: Reward = {} Cumulative reward = {}".format(step, action_actual, reward, cumulative_reward))

                dqn_solver.remember(state, action, reward, state_next)  # Remember this instance
                state = state_next  # Update the state

                dqn_solver.experience_replay(episode)  # Perform experience replay to update the network weights

                if step == steps:

                    score.append(cumulative_reward)
                    twodplot(step_x, cumulative_reward_y1, rewards_y2, actions_taken, episode, window_size, date_range, filename, date, mode, False)
                    break
                else:
                    step += 1

            print("Episode: {}. Net Reward : {}".format(episode, score[episode]))
            episode += 1

        # plt.show()

    # ------------------------------------------------ TESTING ---------------------------------------------------------
    if mode == 'Test':
        test_episodes = 1
        test_steps = 14

        print('\nTesting...\n')

        # DQN Stocks
        env = StockTradingEnv(mode, test_steps, stock, trend, date, window_size)  # Resetting the environment

        # Get action and observation space
        observation_space = env.observation_space
        action_space = env.action_space
        window_size = env.window_size

        # Object for the solver
        dqn_solver = DQNSolver(observation_space, action_space, mode)

        episode = 1
        score = [0]

        # Running for a number of episodes
        while episode <= test_episodes:
            #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
            state, date_range, filename = env.reset()  # Get initial state
            step = 1
            cumulative_reward = 0

            # To append step, cumulative reward, corresponding action to plot for each episode
            step_x = []
            cumulative_reward_y1 = []
            rewards_y2 = []
            actions_taken = []

            #  Going through time series data
            while step <= test_steps:
                action = dqn_solver.act(state)  # Get action based on argmax of the Q value approximation from the NN
                state_next, reward, info = env.step(action, mode)

                cumulative_reward += reward

                if action == 0:
                    action_actual = 'Sell'
                if action == 1:
                    action_actual = 'Hold'
                if action == 2:
                    action_actual = 'Buy'

                step_x.append(step)
                cumulative_reward_y1.append(cumulative_reward)
                rewards_y2.append(reward)
                actions_taken.append(action)

                print("{} {}ing: Reward = {} Cumulative reward = {}".format(step, action_actual, reward, cumulative_reward))

                state = state_next  # Update the state
                if step == test_steps:
                    score.append(cumulative_reward)

                    # Saving plot into to be compiled every 7 days
                    f = open(r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\TestArea\temp_{}_{}.txt".format(stock, trend), "a")
                    f.write("{}~{}~{}~{}~{}~{}~{}~{}~{}\n".format(step_x[0], cumulative_reward_y1[0], rewards_y2[0], actions_taken[0], episode, window_size, date_range, filename, mode))
                    f.close()
                    twodplot(step_x, cumulative_reward_y1, rewards_y2, actions_taken, episode, window_size, date_range, filename, date, mode, False)
                    break
                else:
                    step += 1

            # print("Episode: {}. Score : {}".format(episode, score[episode]))
            episode += 1

        # plt.show()

        # Counting the number of folders in the current results folder
        # folders = 0
        # path = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\Results_{}_{}".format(stock, trend)
        # for _, dirnames, _ in os.walk(path):
        #     folders += len(dirnames)
        #
        # if folders % (7*4) == 0:  # if they're 7 days worth of results (or multiples of it - 4 folders for each day)
        #     results(stock, trend, date)  # compile and save a weeks worth of results


# To compile the last 7 days results
def results(stock, trend, date):
    f = open(r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\TestArea\temp_{}_{}.txt".format(stock, trend), "r")
    step_x = []
    rewards_y2 = []
    cumulative_reward_y1 = []
    cumulative_reward = 0
    actions_taken = []
    for i in range(7):
        _, _, reward, action, episode, window_size, date_range, filename, mode = (f.readline()).split("~")

        step_x.append(i+1)
        rewards_y2.append(float(reward))
        cumulative_reward += float(reward)
        cumulative_reward_y1.append(cumulative_reward)
        actions_taken.append(int(action))
    mode = mode.strip()  # To get rid of new line
    twodplot(step_x, cumulative_reward_y1, rewards_y2, actions_taken, int(episode), int(window_size), date_range, filename, date, mode, True)
    f.close()
    os.remove(r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\TestArea\temp_{}_{}.txt".format(stock, trend))
    print("Old file removed! Ready for a new week of predictions.")


def visualization(window_size):  # To visualize intermediate layers

    # Setting and getting the model
    dqn_solver = DQNSolver((1, window_size, window_size), 3, 'Test')
    model = dqn_solver.instantiate_load_and_return_model()
    visualization_layer_number = 3  # Output of the layer that you want to visualize

    # model.summary()

    # Summarizing feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # Checking for convolution layers
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)

    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[visualization_layer_number].output)
    img = load_img('TestArea/test_image.bmp', target_size=(16, 16))
    img = np.asarray(img)
    img = img[:, :, 0]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    feature_maps = model.predict(img)
    fshape = feature_maps.shape
    print(fshape)

    # Creating directory if it doesn't exist
    path = "TestArea/Intermediate_layer/Layer {}/".format(visualization_layer_number)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    for i in range(0, fshape[1]):
        image_temp = feature_maps[0, i, :, :]
        image_temp = Image.fromarray(np.uint8(image_temp), 'L')
        image_temp.save(path + "{}.bmp".format(i))


def experiments(stock, trend, window_size, date_of_prediction):
    # Downloading stock data
    scraping(stock)

    # Defining input directory (post download) and target directory (post renaming)
    input_dir = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\Data"
    target = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\{}_Stock.csv".format(stock)

    # Renaming and saving
    rename_file(input_dir, target)

    # Downloading trend data
    days_to_be_scraped = 365
    get_trends(trend, days_to_be_scraped, date_of_prediction)

    # Converting trends data to candlesticks, truncating and cleaning both data files
    training_set = 200
    testing_set = 1
    final_length = training_set + testing_set + window_size
    stock_df = pd.read_csv('{}_Stock.csv'.format(stock))
    trend_df = pd.read_csv('{}_Trend.csv'.format(trend))

    go = convert_and_clean(stock_df, trend_df, trend, stock, final_length)
    # go = True
    if not go:
        print("Stock Market is closed today!")
    else:
        print("Stock Market is open. Let's go:")
        """ --------------------------------------- ACTUAL ALGORITHM ----------------------------------------------- """
        mode = 'Train'
        DQN_Agent(mode, stock, trend, window_size, date_of_prediction)
        mode = 'Test'
        DQN_Agent(mode, stock, trend, window_size, date_of_prediction)


if __name__ == "__main__":
    stockname = 'S&P500'
    trendname = 'S&P500 stock'
    windowsize = 16

    # Testing out stuff
    date_of_prediction = str(datetime.date(2020, 2, 20))  # Y-M-D ((2019, 12, 31) and (2020/02/20))
    mode = 'Train'
    DQN_Agent(mode, stockname, trendname, date_of_prediction, windowsize)
    # mode = 'Test'
    # DQN_Agent(mode, stockname, trendname, date_of_prediction, windowsize)

    # Experiments Automation
    # date_of_prediction = datetime.datetime.date(datetime.datetime.now()) - datetime.timedelta(days=1)
    # experiments(stockname, trendname, windowsize, date_of_prediction)

    # Intermediate Layers Visualization
    # visualization(window_size)

#TODO:
# 14 days of testing -> 200 + 16 + 14 = 230 final length
# Experiments date ranges - 1 year back from -> (12/31/2019 and 02/20/2020)
# Experiment keywords -> Any Stock + Stockmarket Crash (2019), Any Stock + Coronavirus (2020)
# HOW TO TEST (manually): Set date of prediction, Download stocks, Run trends, Run Cleaning, Run DQN
# HOW TO TEST (automatically): Comment out lines 375 to 380. Uncomment 385 nad 386. Run the Batchfile post midnight.
# # Train on less data?
# Decay exploration faster?
# Train more epochs?
# Batches
# Filter size? Different image sizes?
# Using important dates (4th of July) for feature augmentation
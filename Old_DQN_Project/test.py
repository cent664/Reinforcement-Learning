import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from makeshift_env import StockTradingEnv
import matplotlib.pyplot as plt
from plot import twodplot

# Parameters

gamma = 0  # Discount factor
learning_rate = 0.00025  # Learning rate

# TODO: Play with memory and batch size
memory_size = 100  # Memory pool for experience replay, forgets older values as the size is exceeded
batch_size = 32  # Batch size for random sampling in the memory pool

# TODO: Play with exploration rate and decay
exploration_max = 1.0  # Initial exploration rate
exploration_min = 0.01  # Min value of exploration rate post decay
exploration_decay = 0.9  # Exploration rate decay rate

episodes = 10
steps = 1260


class DQNSolver:

    def __init__(self, observation_space, action_space, mode):
        self.exploration_rate = exploration_max  # Sets initial exploration rate to max
        self.old_episode = 0  # To check for ep changes (needed to decay exploration rate)

        self.action_space = action_space  # Action space = 3 (Sell, Hold, Buy)
        self.memory = deque(maxlen=memory_size)  # Will forget old values as new ones are appended

        # Defining the network structure
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 4, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 2, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))
        self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        if mode == 'Test':
            # Loading weights here
            print('Loading weights')
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
        # To decay the exploration rate if the episode changes
        if episode != self.old_episode:
            self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
            self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum
        self.old_episode = episode

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

        # Saving the weights
        self.model.save_weights('CNN_DQN_weights.h5')


def DQN_Agent(mode):

    # ------------------------------------------------ TRAINING --------------------------------------------------------
    if mode == 'Train':
        print('\nTraining:\n')

        # DQN Stocks
        env = StockTradingEnv(mode, steps)  # Object of the environment

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
                    twodplot(step_x, cumulative_reward_y1, rewards_y2, actions_taken, episode, window_size, date_range, filename, mode)
                    break
                else:
                    step += 1

            print("Episode: {}. Net Reward : {}".format(episode, score[episode]))
            episode += 1

        plt.show()

    # ------------------------------------------------ TESTING ---------------------------------------------------------
    if mode == 'Test':
        test_episodes = 1
        test_steps = 20

        print('\nTesting:\n')

        # DQN Stocks
        env = StockTradingEnv(mode, test_steps)  # Resetting the environment

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
                    twodplot(step_x, cumulative_reward_y1, rewards_y2, actions_taken, episode, window_size, date_range, filename, mode)
                    break
                else:
                    step += 1

            print("Episode: {}. Score : {}".format(episode, score[episode]))
            episode += 1

        plt.show()


if __name__ == "__main__":
    mode = 'Train'
    DQN_Agent(mode)


#TODO:
# Batches
# Include week, month, more months image sets
# Upload env to gym
# MAIN: Test reward structure, different image sizes/precision, volume and holdings or not
# Clean redundant variables and calls
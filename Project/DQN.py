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

gamma = 0.95  # Discount factor
learning_rate = 0.00025  # Learning rate

# TODO: Play with memory and batch size
memory_size = 100  # Memory pool for experience replay, forgets older values as the size is exceeded
batch_size = 32  # Batch size for random sampling in the memory pool

# TODO: Play with exploration rate and decay
exploration_max = 1.0  # Initial exploration rate
exploration_min = 0.01  # Min value of exploration rate post decay
exploration_decay = 0.995  # Exploration rate decay rate

episodes = 10
steps = 252

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = exploration_max

        self.action_space = action_space  # Action space = 3 (Sell, Hold, Buy)
        self.memory = deque(maxlen=memory_size)  # Will forget old values as new ones are appended

        # Defining the network structure
        # TODO: Check filter numbers and size. Check the negative subtraction error
        self.model_CNN = Sequential()
        self.model_CNN.add(Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model_CNN.add(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model_CNN.add(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model_CNN.add(Flatten())

        self.model_FC = Sequential()
        self.model_FC.add(Dense(512, activation="relu"))
        self.model_FC.add(Dense(action_space))

        self.model_FC.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        # print(self.model_CNN.summary())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Remembering instances in memory for future use

    def act(self, state):
        # Random action -> 0, 1, 2
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)

        weights = self.model_CNN.predict(state[0])  # Getting weights from the CNN (input = image)
        features = np.concatenate((weights, state[1]), axis=1)  # Adding holdings and volume to the weights before FC

        # Q values based on model prediction on current state (Initially based on random weights)
        q_values = self.model_FC.predict(features)

        return np.argmax(q_values[0])  # Argmax of tuple of 3 Q values, one for each action

    def experience_replay(self):
        if len(self.memory) < batch_size:  # If has enough memory obtained, perform random batch sampling among those
            return

        batch = random.sample(self.memory, batch_size)  # Get a random batch
        for state, action, reward, state_next, done in batch:
            q_update = reward  # Reward obtained for a particular state, action pair
            if not done:

                weights = self.model_CNN.predict(state_next[0])  # Getting weights from the CNN (input = image)
                features = np.concatenate((weights, state[1]), axis=1)  # Adding holdings and volume to the weights before FC

                # Obtain Q value based on immediate reward and predicted q* value of next state
                q_update = reward + gamma * np.amax(self.model_FC.predict(features))

            weights = self.model_CNN.predict(state[0])  # Getting weights from the CNN (input = image)
            features = np.concatenate((weights, state[1]), axis=1)  # Adding holdings and volume to the weights before FC

            q_values = self.model_FC.predict(features)  # Obtain q value tuple for that state
            q_values[0][action] = q_update  # Update the q value for that state, action (one that we took)
            # Update the weights of the network based on the updated q value (based on immediate reward)
            self.model_FC.fit(features, q_values, epochs=1, verbose=0)

        self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum

def DQN_Agent():

    # ------------------------------------------------ TRAINING --------------------------------------------------------

    print('\nTraining:\n')
    mode = 'train'

    # DQN Stocks
    env = StockTradingEnv(mode)  # Object of the environment

    # Get action and observation space
    observation_space = env.observation_space
    action_space = env.action_space

    # Object for the solver
    dqn_solver = DQNSolver(observation_space, action_space)

    episode = 1
    score = [0]

    # Running for a number of episodes
    while episode <= episodes:

        #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
        state = env.reset(mode)  # Get initial state
        step = 1
        cumulative_reward = 0

        # To append step, cumulative reward, corresponding action to plot for each episode
        emptyx = []
        emptyy = []
        emptyaction = []

        #  Going through time series data
        while step <= steps:
            action = dqn_solver.act(state)  # Get action based on argmax of the Q value approximation from the NN
            state_next, reward, done, info = env.step(action, mode)

            if not done:
                reward = reward
            else:
                reward = -reward

            cumulative_reward += reward

            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            emptyx.append(step)
            emptyy.append(cumulative_reward)
            emptyaction.append(action_actual)

            # print("{} {}ing: Holdings = {} Cumulative reward = {}".format(step, action_actual, state_next[1], cumulative_reward))

            dqn_solver.remember(state, action, reward, state_next, done)  # Remember this instance
            state = state_next  # Update the state

            dqn_solver.experience_replay()  # Perform experience replay to update the network weights

            if done or step == steps:
                score.append(cumulative_reward)
                twodplot(emptyx, emptyy, emptyaction, episode)
                break
            else:
                step += 1

        print("Episode: {}. Score : {}".format(episode, score[episode]))
        episode += 1

    plt.show()

    # ------------------------------------------------ TESTING ---------------------------------------------------------

    test_episodes = 1
    test_steps = 20

    print('\nTesting:\n')
    mode = 'test'

    # DQN Stocks
    env = StockTradingEnv(mode)  # Resetting the environment

    # Resetting everything, except the CNN

    episode = 1
    score = [0]

    # Running for a number of episodes
    while episode <= test_episodes:

        #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
        state = env.reset(mode)  # Get initial state
        step = 1
        cumulative_reward = 0

        # To append step, cumulative reward, corresponding action to plot for each episode
        emptyx = []
        emptyy = []
        emptyaction = []

        #  Going through time series data
        while step <= test_steps:
            action = dqn_solver.act(state)  # Get action based on argmax of the Q value approximation from the NN
            state_next, reward, done, info = env.step(action, mode)

            if not done:
                reward = reward
            else:
                reward = -reward

            cumulative_reward += reward

            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            emptyx.append(step)
            emptyy.append(cumulative_reward)
            emptyaction.append(action_actual)

            print("{} {}ing: Holdings = {} Cumulative reward = {}".format(step, action_actual, state_next[1], cumulative_reward))

            dqn_solver.remember(state, action, reward, state_next, done)  # Remember this instance
            state = state_next  # Update the state

            dqn_solver.experience_replay()  # Perform experience replay to update the network weights

            if done or step == test_steps:
                score.append(cumulative_reward)
                twodplot(emptyx, emptyy, emptyaction, episode)
                break
            else:
                step += 1

        print("Episode: {}. Score : {}".format(episode, score[episode]))
        episode += 1

    plt.show()

if __name__ == "__main__":
    DQN_Agent()


#TODO:
# Deconv instead of concat
# Find a way to use GPU
# Reward structure
# Batches
# Include week, month, more months image sets
# Upload env to gym
# MAIN: Test reward structure, different image sizes/precision, volume and holdings or not
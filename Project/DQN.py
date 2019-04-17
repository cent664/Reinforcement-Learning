import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
from makeshift_env import StockTradingEnv
import matplotlib.pyplot as plt
from plot import twodplot

# DQN Stocks

env = StockTradingEnv()  # Object of the environment

gamma = 0.95  # Discount factor
learning_rate = 0.001  # Learning rate

memory_size = 1000  # Memory pool for experience replay, forgets older values as the size is exceeded
batch_size = 32  # Batch size for random sampling in the memory pool

exploration_max = 1.0  # Initial exploration rate
exploration_min = 0.01  # Min value of exploration rate post decay
exploration_decay = 0.995  # Exploration rate decay rate

episodes = 5
steps = 50

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = exploration_max

        self.action_space = action_space  # Action space = 3 (Sell, Hold, Buy)
        self.memory = deque(maxlen=memory_size)  # Will forget old values as new ones are appended

        # Defining the network structure
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=observation_space, data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))

        self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Remembering instances in memory for future use

    def act(self, state):
        # Random action -> 0, 1, 2
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # Q values based on model prediction on current state (Initially based on random weights)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Argmax of tuple of 3 Q values, one for each action

    def experience_replay(self):
        if len(self.memory) < batch_size:  # If has enough memory obtained, perform random batch sampling among those
            return
        batch = random.sample(self.memory, batch_size)  # Get a random batch
        for state, action, reward, state_next, done in batch:
            q_update = reward  # Reward obtained for a particular state, action pair
            if not done:
                # Obtain Q value based on immediate reward and predicted q* value of next state
                q_update = reward + gamma * np.amax(self.model.predict(state_next))

            q_values = self.model.predict(state)  # Obtain q value tuple for that state
            q_values[0][action] = q_update  # Update the q value for that state, action (one that we took)
            # Update the weights of the network based on the updated q value (based on immediate reward)
            self.model.fit(state, q_values, epochs=1, verbose=0)

        self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum

def DQN_Agent():
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
        state = env.reset()  # Get initial state
        # state = np.reshape(state, [1, observation_space])
        step = 1
        cumulative_reward = 0

        # To append step, cumulative reward, corresponding action to plot for each episode
        emptyx = []
        emptyy = []
        emptyaction = []

        #  Going through time series data
        while step <= steps:
            action = dqn_solver.act(state)  # Get action based on argmax of the Q value approximation from the NN
            state_next, reward, done, info = env.step(action)

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

            #print("{} {}ing: Holdings = {} Cumulative reward = {}".format(step, action_actual, state_next[1], cumulative_reward))

            # state_next = np.reshape(state_next, [1, observation_space])
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

if __name__ == "__main__":
    DQN_Agent()

"""    
TODO:
Check input dimensions, observation space AND image dimensions (axis in np.expand_dims)
Input my np array. Decide on dimensions.
Fix rewards, state representation (add holdings/position, volume)
Remove adjusted close, use close for reward?
Include week, month, more months image sets
Upload env to gym

"""
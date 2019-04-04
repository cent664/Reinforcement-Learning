import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding

# Q learning mountain car

obj = gym.make('MountainCar-v0')

q = {}
state_count = {}  # To count the number of times each state is being visited
epsilon = 1  # Exploration threshold
alpha = 1  # Learning rate

gamma = 1  # No discount
win = 0  # To count the number x wins

for i in range(0, 19):
    for j in range(0, 15):
        # Initializing q table with indexes referring to position and velocity
        q[round(((i / 10) - 1.2), 1), round(((j / 100) - 0.07), 2), 0] = 0
        q[round(((i / 10) - 1.2), 1), round(((j / 100) - 0.07), 2), 1] = 0
        q[round(((i / 10) - 1.2), 1), round(((j / 100) - 0.07), 2), 2] = 0

def argmax(position, velocity):  # For greedy action
    best = -999999999999
    for action in range(0, 3):  # Looping through all the actions
        if (best < q[position, velocity, action]):  # Picking the one with best immediate reward
            best = q[position, velocity, action]
            optimal_action = action
    return optimal_action

if __name__ == '__main__':

    for episode in range(0, 5000):
        count = 0  # To count the number of steps each episode
        state = obj.reset()  # Initializing the initial position and velocity

        while (count < 200):
            count = count + 1  # Counting the number of iterations
            #if (episode > 0):
                #obj.render()  # Rendering environment

            position = round(state[0], 1)
            velocity = round(state[1], 2)

            if(np.random.uniform(0,1) > epsilon):  # Greedy action
                action = argmax(position, velocity)
            else:
                action = np.random.randint(0, 3)  # Random action

            new_state, reward, done, _ = obj.step(action)  # Playing the game

            new_position = round(new_state[0], 1)
            new_velocity = round(new_state[1], 2)

            qmax_of_new_state = q[new_position, new_velocity, argmax(new_position, new_velocity)]  # Max Q value of the new state

            q[position, velocity, action] = q[position, velocity, action] + alpha*(reward + (gamma * qmax_of_new_state) - q[position, velocity, action])

            state = new_state
            if (state[0] >= 0.5):
                win = win + 1
            if (done):  # If we reach the top, end episode
                break

        if (((episode + 1) % 500) == 0):
            print(episode + 1, win)
            win = 0

        epsilon = epsilon * 0.9
        alpha = alpha * 0.9
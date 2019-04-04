import matplotlib as plt

# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, action):

    plt.plot(steps, rewardsum)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Steps vs Cumulative reward')
    plt.show()

    f = open("action.txt", "w")
    for i in range(0, 4001):
        f.write(action[i])
        f.write("\n")
    f.close()
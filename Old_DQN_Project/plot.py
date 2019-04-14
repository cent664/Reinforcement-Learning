import matplotlib.pyplot as plt

# To plot the steps vs cumulative reward
def twodplot(steps, rewardsum, action, episode):

    plt.plot(steps, rewardsum, label='Episode {}'.format(episode))
    plt.legend(loc='upper right')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Steps vs Cumulative reward')

    if episode == 1:
        f = open("action.txt", "w")
    else:
        f = open("action.txt", "a")

    f.write("Episode = {}.\n\n".format(episode))

    for i in range(0, len(steps)):
        f.write("Step = {}. Action = {}\n".format(steps[i], action[i]))
    f.write("\n")
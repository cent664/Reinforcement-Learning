# policy iteration
import matplotlib.pyplot as plt
import numpy as np

states_count = 101
pi = np.array(states_count)
pi = np.zeros(states_count)
v = np.zeros(states_count)
v[states_count-1] = 1.0
pHeads = 0.4

def policyEvaluation():
    while True:
        delta = 0
        for s in range(1, states_count - 1):
            action = int(pi[s])
            vOld = v[s]
            v[s] = pHeads * v[s + action] + (1 - pHeads) * v[s - action]
            delta = max(delta, abs(vOld - v[s]))

        if delta < 1e-10:
            break

def policyIteration():
    for s in range(1, states_count):
        old_action = pi[s]

        maxV = -1
        bestAction = -1
        # for a in range(min(s, (states_count - s - 1)), 0, -1):
        for a in range(1,min(s, (states_count - s - 1)) + 1):
            v_state = pHeads * v[s + a] + (1 - pHeads) * v[s - a]
            if v_state >= maxV:
                maxV = v_state
                bestAction = a

        if old_action!=bestAction:
            pi[s] = bestAction
            policyEvaluation()
        else:
            break

policyIteration()

# plt.plot(v, drawstyle="steps")
plt.plot(pi, drawstyle="steps")
plt.show()

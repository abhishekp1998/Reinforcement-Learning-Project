import gym
import numpy as np
from code import MultiArmedBandit
from code import QLearning
import matplotlib.pyplot as plt

env = gym.make('SlotMachines-v0')
agent1 = MultiArmedBandit()
rewardsarray1 = []
rewardsarray2 = []
agent2 = QLearning()
for i in range(10):
    action_values1, rewards1 = agent1.fit(env, steps = 100000)
    rewardsarray1.append(rewards1)
    action_values2, rewards2 = agent2.fit(env, steps= 100000)
    rewardsarray2.append(rewards2)

rewardsarray1half = []
rewardsarray2half = []
for i in range(5):
    rewardsarray1half.append(rewardsarray1[i])
    rewardsarray2half.append(rewardsarray2[i])

rewardsarray1 = np.asarray(rewardsarray1)
rewardsarray2 = np.asarray(rewardsarray2)
rewardsarray1half = np.asarray(rewardsarray1half)
rewardsarray2half = np.asarray(rewardsarray2half)

halfmean1 = np.mean(rewardsarray1half, axis = 0)
halfmean2 = np.mean(rewardsarray2half, axis = 0)
fullmean1 = np.mean(rewardsarray1, axis = 0)
fullmean2 = np.mean(rewardsarray2, axis = 0)


x_vals = np.arange(100)

plt.plot(x_vals, rewardsarray1[0], label = "Rewards Average 1")
plt.plot(x_vals, halfmean1, label = "Rewards Average 5")
plt.plot(x_vals, fullmean1, label = "Rewards Average 10")
plt.ylabel("rewards for MAB")
plt.xlabel("index")
plt.legend()
plt.show

plt.plot(x_vals, fullmean1, label = "Rewards Average for MAB")
plt.plot(x_vals, fullmean2, label = "Rewards Average for Qlearner")
plt.ylabel("rewards for QLearner and MAB")
plt.xlabel("index")
plt.legend()
plt.show

env = gym.make('FrozenLake-v0')
agent1F = MultiArmedBandit()
rewardsarray1F = []
rewardsarray2F = []
agent2F = QLearning()
for i in range(10):
    action_values1, rewards1 = agent1F.fit(env, steps = 100000)
    rewardsarray1F.append(rewards1)
    action_values2, rewards2 = agent2F.fit(env, steps= 100000)
    rewardsarray2F.append(rewards2)

rewardsarray1F = np.asarray(rewardsarray1F)
rewardsarray2F = np.asarray(rewardsarray2F)
fullmean1F = np.mean(rewardsarray1F, axis = 0)
fullmean2F = np.mean(rewardsarray2F, axis = 0)

plt.plot(x_vals, fullmean1F, label = "Rewards Average for MAB Lake")
plt.plot(x_vals, fullmean2F, label = "Rewards Average for Qlearner Lake")
plt.ylabel("rewards for QLearner and MAB Lake")
plt.xlabel("index")
plt.legend()
plt.show

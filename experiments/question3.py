import gym
import numpy as np
from code import QLearning
import matplotlib.pyplot as plt

np.random.seed(0)
env = gym.make('FrozenLake-v0')
env.seed(0)

agent1 = QLearning(epsilon=0.2, discount=0.95)
agent2 = QLearning(epsilon=0.5, discount =0.95)
rewardsarray1 = []
rewardsarray2 = []
for i in range(10):
    action_values1, rewards1 = agent1.fit(env, steps = 100000)
    rewardsarray1.append(rewards1)
    action_values2, rewards2 = agent2.fit(env, steps= 100000)
    rewardsarray2.append(rewards2)

rewardsarray1 = np.asarray(rewardsarray1)
rewardsarray2 = np.asarray(rewardsarray2)
fullmean1 = np.mean(rewardsarray1, axis = 0)
fullmean2 = np.mean(rewardsarray2, axis = 0)
x_vals = np.arange(100)

plt.plot(x_vals, fullmean1, label = "Rewards Average for epsilon .2")
plt.plot(x_vals, fullmean2, label = "Rewards Average for epsilon .5")
plt.ylabel("rewards for QLearner on Frozen Lake")
plt.xlabel("index")
plt.legend()
plt.show

agent3 = QLearning(epsilon=0.5, discount=0.95, adaptive= True)

rewardsarray3 = []

for i in range(10):
    action_values3, rewards3 = agent1.fit(env, steps = 100000)
    rewardsarray3.append(rewards3)

rewardsarray3 = np.asarray(rewardsarray3)
fullmean3 = np.mean(rewardsarray3, axis = 0)

plt.plot(x_vals, fullmean1, label = "Rewards Average for epsilon .2")
plt.plot(x_vals, fullmean2, label = "Rewards Average for epsilon .5")
plt.plot(x_vals, fullmean3, label = "Rewards Average for adaptive epsilon .5")
plt.ylabel("rewards for QLearner on Frozen Lake")
plt.xlabel("index")
plt.legend()
plt.show

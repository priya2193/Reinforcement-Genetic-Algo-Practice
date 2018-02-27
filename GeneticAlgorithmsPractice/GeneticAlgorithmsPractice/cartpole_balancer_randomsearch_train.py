import gym
import numpy as np
from msvcrt import getch
from shm_nn import *

x = []
y = []
env = gym.make('CartPole-v0')

models = []
total_rewards = []
#nn = FullyConnectedNeuralNet(load_path='best_cartpole_model_200.0reward.nn')

for i_episode in range(1000):
    observation = env.reset()
    nn = FullyConnectedNeuralNet([4, 10, 2], activation='relu')
    total_reward = 0
    t = 0
    while True:
        env.render()

        action_probs = nn.feed_forward(np.array([observation]))
        action = action_probs[0].argmax()

        observation, reward, done, info = env.step(action)
        total_reward += reward
        t += 1
        if np.abs(np.rad2deg(observation[2])) > 45:
            models.append(nn)
            total_rewards.append(total_reward)
            print(total_reward, np.max(total_rewards))
            print(observation)
            break
        #if t >= 100:
        #    if done == False:
        #        continue
        #    else:
        #        break
    

best_idx = np.argmax(total_rewards)
models[best_idx].save('best_cartpole_model' + '_' + str(rewards[best_idx]) + 'reward.nn')
#k = 0
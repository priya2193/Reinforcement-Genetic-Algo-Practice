import gym
import numpy as np
from msvcrt import getch
from shm_nn import *

x = []
y = []
env = gym.make('CartPole-v0')
#nn = FullyConnectedNeuralNet([2, 64, 32, 16, 8, 2], learn_rate=.1)
nn = FullyConnectedNeuralNet(load_path='cartpole_model_supervised.nn')

observations = []
for i_episode in range(20):
    observation = env.reset()
    prev_action = 1
    observations.append(observation)
    for t in range(1000):
        env.render()
        print(observation)
        
        action_probs = nn.feed_forward(np.array([observation]))
        action = action_probs[0].argmax()
        print(action_probs, action)

        #input = ord(getch())
        #if input == 75:
        #    action = 0
        #    prev_action = action
        #elif input == 77:
        #    action = 1
        #    prev_action = action
        #else:
        #    action = prev_action

        if action == 0:
            print('LEFT')
        elif action == 1:
            print('RIGHT')
        observation, reward, done, info = env.step(action)

        #if len(x) < 100:
        #    x.append([observation[1], observation[2]])
        #    y.append(action)
        #else:
        #    x_train = np.array(x)
        #    y = np.array(y)
        #    y_onehot = to_one_hot(y, 2)
        #    loss = nn.train_step(x_train, y_onehot)
        #    print('-------------------->', loss)
        #    x = []
        #    y = []
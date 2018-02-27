import gym
import numpy as np
from msvcrt import getch
from shm_nn import *

x = []
y = []
env = gym.make('CartPole-v0')

models = []
total_rewards = []

def noisen_matrix(m, frac=.3):
    idx = np.arange(m.flatten().shape[0])
    np.random.shuffle(idx)
    noise_len = int(frac * idx.shape[0])
    noise = np.random.randn(noise_len)
    noise_idx = idx[:noise_len]
    m_f = m.flatten()
    m_f[noise_idx] = noise
    m = m_f.reshape(m.shape)
    return m


def add_noise(model, noise_frac=.3):
    for i in range(1, model.num_layers):
        model.layers[i].params = noisen_matrix(model.layers[i].params)
    return model

def duplicate_model(model):
    ret_model = FullyConnectedNeuralNet(model.neuron_counts)
    ret_model.layers[0].x = model.layers[0].x.copy()
    for i in range(1, model.num_layers):
        ret_model.layers[i].params = model.layers[i].params.copy()
    return ret_model

#mode = 'train'
mode = 'test'

#if mode == 'test':
#    nn = FullyConnectedNeuralNet(load_path='cartpole_hillclimbing_model.nn')
#elif mode == 'train':
#    nn = FullyConnectedNeuralNet([4, 32, 16, 2], activation='relu')

nn = FullyConnectedNeuralNet(load_path='cartpole_hillclimbing_model.nn')
#nn = FullyConnectedNeuralNet([4, 10, 10, 2], activation='relu')

best_model = duplicate_model(nn)
best_reward = 0

noise_frac = .1
decay_rate = .45
total_episodes = 10000

noise_frac_curr = noise_frac
actions = ['LEFT', 'RIGHT']
for i_episode in range(total_episodes):
    observation = env.reset()

    if mode == 'train':
        noise_frac_curr = noise_frac * decay_rate**((i_episode + 1) / total_episodes)
        prev_nn = duplicate_model(nn)
        add_noise(nn, noise_frac_curr)

    total_reward = 0
    t = 0

    while True:
        env.render()

        action_probs = nn.feed_forward(np.array([observation]))
        action = action_probs[0].argmax()
        #print(actions[action])

        observation, reward, done, info = env.step(action)
        total_reward += reward
        t += 1
        if np.abs(np.rad2deg(observation[2])) > 170:
            if mode == 'train':
                if total_reward > best_reward:
                    best_model = duplicate_model(nn)
                    best_model.save('cartpole_hillclimbing_model.nn')
                    best_reward = total_reward
                else:
                    nn = duplicate_model(prev_nn)
            print(total_reward, best_reward, noise_frac_curr)
            break
    

best_idx = np.argmax(total_rewards)
models[best_idx].save('best_cartpole_model' + '_' + str(rewards[best_idx]) + 'reward.nn')
#k = 0
'''
In battle, in forest, at the precipice in the mountains;
On the dark great sea, in the midst of javelins and arrows;
In sleep, in confusion, in the depths of shame,
The good deeds the man has done before defend him.
'''


import gym
import numpy as np
from gym import wrappers
import scipy.signal
#from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import matplotlib.pyplot as plt
import numpy as np
import sys
from mlagents.envs import UnityEnvironment
#alias python='python3'


def init_env(env_name):
    #env_name = "../UnitySDK/BananaCollector"
    train_mode = True  # Whether to run the environment in training or inference mode
    env = UnityEnvironment(file_name=env_name)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space
    state = env_info.vector_observations[0]
    print('Observations', state)
    state_size = len(state)
    print('Observations have length:', state_size)
    obs_dim = env_info.vector_observations.shape[0]
    print('shape of observations', obs_dim)

    return env, brain, brain_name

    #return env, env_info, action_size, state_size


def random_actions(env, brain, default_brain):
    for episode in range(10):
        env_info = env.reset(train_mode=True)[default_brain]
        done = False
        episode_rewards = 0
        while not done:
            action_size = brain.vector_action_space_size
            if brain.vector_action_space_type == 'continuous':
                env_info = env.step(np.random.randn(len(env_info.agents),
                                                    action_size[0]))[default_brain]
            else:
                action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
                env_info = env.step(action)[default_brain]
            #episode_rewards += env_info.rewards[0]
            print(np.mean(env_info.rewards))
            episode_rewards += np.mean(env_info.rewards)
            done = env_info.local_done[0]
        print("Total reward this episode: {}".format(episode_rewards))



def train(num_episodes, param_1, param_2, param_3, param_4):


    return None


if __name__ == '__main__':
    #env_name = "../ml-agents/UnitySDK/BananaCollector"
    env_name = "banana_collector"
    env_specs, brain, brain_name = init_env(env_name)
    random_actions(env_specs, brain, brain_name)

    '''
    Use this link as the basis for training your agent
    https://github.com/kotogasy/unity-ml-banana/blob/master/double_dqn/Navigation-ddqn.ipynb
    
    for MADRL use this link: good implementation of MADDPG
    https://github.com/fdasilva59/Udacity-DRL-Collaboration-and-Competition/blob/master/Tennis_Project.ipynb
    
    
    
    '''





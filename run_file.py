from mlagents.envs import UnityEnvironment
import numpy as np

import random
import torch
import os
from collections import deque
import time
import matplotlib.pyplot as plt

train_mode = True  # Whether to run the environment in training or inference mode
env_name = "Tennis_2_agents"
#env_name = "Banana_2_agents"
env = UnityEnvironment(file_name=env_name)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

model_dir = os.getcwd() + "/model_dir"
# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
# size of each action
ENV_ACTION_SIZE = brain.vector_action_space_size[0]
# size of the state space
states = env_info.vector_observations  # Array of states for all agents in teh enviroonments
ENV_STATE_SIZE = states.shape[1]
print('There are {} agents. Each observes a state with length: {} and act within an action space of length: {}'.format(states.shape[0],
                                                                                                                       ENV_STATE_SIZE,
                                                                                                                       ENV_ACTION_SIZE))


from MADDPG import MADDPG
from hyperparams import *

def train():



    # Seeding
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Instantiate the MADDPG agents
    maddpg = MADDPG(ENV_STATE_SIZE, ENV_ACTION_SIZE, num_agents, SEED)

    # Monitor the score
    scores_deque = deque(maxlen=100)
    all_scores = []
    all_avg_score = []


    # Intialize amplitude OU noise (will decay during training)
    noise = NOISE

    all_steps =0   # Monitor total number of steps performed

    # Training Loop
    for i_episode in range(NB_EPISODES+1):

        env_info = env.reset(train_mode=True)[brain_name]          # reset the environment
        maddpg.reset()                                             # reset the agents

        states = env_info.vector_observations                      # get the current state for each agent
        scores = np.zeros(num_agents)                              # initialize the score (for each agent)

        for steps in range(NB_STEPS):

            all_steps+=1

            actions = maddpg.act(states, noise)                    # retrieve actions to performe for each agents
            noise *= NOISE_REDUCTION                               # Decrease action noise
            env_info = env.step(actions)[brain_name]               # send all actions to tne environment
            next_states = env_info.vector_observations             # get next state for each agent
            rewards = env_info.rewards                             # get reward (for each agent)
            dones = env_info.local_done                            # see if episode finished

            # Save experience in replay memory, and use random sample from buffer to learn
            maddpg.step(states, actions, rewards, next_states, dones, i_episode)

            scores += env_info.rewards                             # update the score (for each agent)
            print("in episode score is: " + str(scores))
            states = next_states                                   # roll over states to next time step
            if np.any(dones):                                      # exit loop if episode finished
                #print("   ** Debug: episode= {} steps={} rewards={} dones={}".format(i_episode, steps,rewards,dones))
                break

        # Save scores and compute average score over last 100 episodes
        episode_score  = np.max(scores)  # Consider the maximum score amongs all Agents
        all_scores.append(episode_score)
        scores_deque.append(episode_score)
        avg_score = np.mean(scores_deque)

        # Display statistics
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpisode score (max over agents): {:.2f}'.format(i_episode, avg_score, episode_score), end="")
        if i_episode>0 and i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} (nb of total steps={}   noise={:.4f})'.format(i_episode, avg_score, all_steps, noise))
            maddpg.checkpoints()
            all_avg_score.append(avg_score)

        # Early stop
        if (i_episode > 99) and (avg_score >=0.5):
            print('\rEnvironment solved in {} episodes with an Average Score of {:.2f}'.format(i_episode, avg_score))
            maddpg.checkpoints()
            return all_scores

    return all_scores

scores = train()
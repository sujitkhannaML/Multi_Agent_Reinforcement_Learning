import numpy as np
import random
import copy

from Model import Actor, Critic
from memory import ReplayBuffer
from hyperparams import *
import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4         # actor learning rate
LR_CRITIC = 1e-3        # critic learning rate
WEIGHT_DECAY = 0.       # L2 regularization weight decay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, random_seed, num_agents=1):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        #Raw and Targer Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=LR_ACTOR)

        ##copying the weights of the raw network to the target network
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

        #Raw and Target CRITIC Network
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        ##copying the weights of the raw network to the target network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        ##Creating Noise Process
        self.noise = OrnUhlNoise(action_size,random_seed)

        ###Replay Memory; in MADDPG, the replay buffer is common to all agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


    def step(self, state, action, reward, next_state, done):
        """Shared Memory; to save experiences in replay memory, and use random sample from buffer to learn"""
        #check if this is accurately implemented in training the MADDPG agent

        self.memory.add(state, action,reward,next_state,done)

        #start learning if the buffer size is full
        if len(self.memory)>BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self,state, noise=0.0):
        """uses current ploicy to output the next action"""
        ''' Please understand the below code snippet in detail '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if ADD_OU_NOISE:
            action+=self.noise.sample()*noise
        return np.clip(action, -1,  1)

    def reset(self):
        self.noise.reset()



    def learn(self, experiences, gamma ):
        ''' only used in traininf DDPG agent, not for MADDPG'''
        #saves policy and value params in Experience tuples

        states, actions, rewards, next_states, dones = experiences

        #update critic
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma*next_Q_targets*(1 - dones)) #Q targets for current states
        Q_Expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_Expected, Q_targets)
        #minimizing the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #pdate actor
        ##computing actions_loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states,actions_pred).mean()
        ##minimizing the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


class OrnUhlNoise:
    def __init__(self, size, seed, mu=MU, theta=THETA, sigma=SIGMA):
        """Initialize parameters and noise process."""
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)


    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state



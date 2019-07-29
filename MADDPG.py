import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

from Agent import Agent
from memory import ReplayBuffer
from hyperparams import *
from utils import encode, decode
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:

    def __init__(self, state_size, action_size, num_agents, random_seed):
        super(MADDPG, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)


        ##Create agents in the enviromnent

        self.agents = [ Agent(state_size, action_size, random_seed, num_agents) for i in range(num_agents)]

        ###create shared Memory Replay Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def reset(self):
        '''reset all the agents'''
        for agent in self.agents:
            agent.reset()

    def act(self, states,noise):
        return [ agent.act(state, noise) for agent, state in zip(self.agents,states)]

    def step(self, states, actions, rewards, next_states, dones,num_current_episode):

        ''' save experiences in replay memory and use them to learn'''
        self.memory.add(encode(states), encode(actions), rewards, encode(next_states), dones)

        if (len(self.memory)>BATCH_SIZE) and (num_current_episode % UPDATE_EVERY_NB_EPISODE==0):

            for i in range(MULTIPLE_LEARN_PER_UPDATE):
                experiences = self.memory.sample()  #SAMPLE A BATCH OF EXP FROM MEMORY
                ###as of now maddpg_learn only works with 2 agents;
                ###modify it accept n number of agents
                '''Update agent 0 '''
                self.maddpg_learn(experiences, own_idx=0, other_idx=1)
                experiences = self.memory.sample()
                '''update agent 1'''
                self.maddpg_learn(experiences, own_idx=1, other_idx=0)

    def maddpg_learn(self, experiences, own_idx, other_idx, gamma=GAMMA):
        """
        Update the policy of the MADDPG "own" agent. The actors have only access to agent own
        information, whereas the critics have access to all agents information.
        """

        states, actions, rewards, next_states, dones = experiences

        # Filter out the agent OWN states, actions and next_states batch
        own_states =  decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states)

        # Filter out the OTHER agent states, actions and next_states batch
        other_states =  decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)

        # Concatenate both agent information (own agent first, other agent in second position)
        all_states=torch.cat((own_states, other_states), dim=1).to(device)
        all_actions=torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states=torch.cat((own_next_states, other_next_states), dim=1).to(device)

        agent = self.agents[own_idx]

        #update critic#
        # Get predicted next-state actions and Q values from target models
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)),
                                     dim =1).to(device)
        #print("all states, all actions" + str(all_next_states.shape) + " " + str(all_next_actions.shape) )
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor#
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim = 1).to(device)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        #update target networks#
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)


    def maddpg_learn_old(self, experiences, own_idx, other_idx, gamma=GAMMA):
        '''Only works for 2 agents systems; modify it for any number of agents'''

        states, actions, rewards, next_states, dones = experiences

        ##filtering out own states
        own_states = decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size,self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states)

        ##filter out other agent states
        other_states = decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size,self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)

        ##conacat both agent info
        all_states = torch.cat((own_states, other_states), dim=1).to(device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(device)

        agent = self.agents[own_idx]

        ######update the critic#######
        '''Get predicted next state action and Q values from target models'''
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)), dim=1).to(device)
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))  #Q target for current state

        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        ##minimize the loss
        agent.critc_optimizer.zero_grad()
        critic_loss.backward()
        if(CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critc_optimizer.step()

        #####Update Actor########
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()), dim=1).to(device)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        ###minimize the loss####
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        #####UPDATE TARGET NETWORKS########
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)


    def checkpoints(self):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'model_dir/checkpoint_actor_local_' + str(idx) + '.pth'
            critic_local_filename = 'model_dir/checkpoint_critic_local_' + str(idx) + '.pth'
            actor_target_filename = 'model_dir/checkpoint_actor_target_' + str(idx) + '.pth'
            critic_target_filename = 'model_dir/checkpoint_critic_target_' + str(idx) + '.pth'
            torch.save(agent.actor_local.state_dict(), actor_local_filename)
            torch.save(agent.critic_local.state_dict(), critic_local_filename)
            torch.save(agent.actor_target.state_dict(), actor_target_filename)
            torch.save(agent.critic_target.state_dict(), critic_target_filename)



import torch.nn.functional as F

# Default hyperparameters

SEED = 35                          # Random seed

NB_EPISODES = 8000                # # of episodes
NB_STEPS = 1000                    #  # of steps per episodes
UPDATE_EVERY_NB_EPISODE = 4        # # of episodes between learning process
MULTIPLE_LEARN_PER_UPDATE = 3      # # of multiple learning process performed in a row

BUFFER_SIZE = int(1e5)             # replay buffer size
BATCH_SIZE = 200                   # minibatch size

ACTOR_FC1_UNITS = 200          # Layer1 actor model
ACTOR_FC2_UNITS = 150          # Layer2 actor model
CRITIC_FCS1_UNITS = 200        # Layer1 critic model
CRITIC_FC2_UNITS = 150         # Layer2 critic model
NON_LIN = F.relu   #F.leaky_relu   # Non linearity operator used in the model
LR_ACTOR = 1e-4    #1e-4           # learning rate of the actor 
LR_CRITIC = 5e-3   #2e-3           # learning rate of the critic
WEIGHT_DECAY = 0   #0.0001         # L2 weight decay

GAMMA = 0.99 #0.99                # Discount factor
TAU = 1e-3
CLIP_CRITIC_GRADIENT = True       #Clip Critic gradient

ADD_OU_NOISE = True                # Add Ornstein-Uhlenbeck noise
MU = 0.                            # Mean
THETA = 0.15                       # Theta
SIGMA = 0.2                        # variance
NOISE = 1.0                        # Noise Level
NOISE_REDUCTION = 1.0              # Planar decay
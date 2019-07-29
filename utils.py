import numpy as np
import torch

# Helper functions to concatenate/extract multipe agents states/actions for use with the Replay Buffer memory.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode(sa):
    return np.array(sa).reshape(1,-1).squeeze()



def decode(size, num_agents, id_agent, sa, debug=False):
    list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(device)
    out = sa.index_select(1, list_indices)

    if (debug):
        print("\nDebug decode:\n size=",size, " num_agents=", num_agents, " id_agent=", id_agent, "\n")
        print("input:\n", sa,"\n output:\n",out,"\n\n\n")
    return  out
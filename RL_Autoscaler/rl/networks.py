import numpy as np 
import torch 
from torch import nn 
from torch.distributions import Categorical 

def layer(inputs, outputs, gain):
    """ 
        We create a simple linear layer with Orthogonal Weights and Zero Bias
    """
    linear = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear 

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden = 64):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden 
        
        #The Actor Network, A Sequential Network with 1 hidden layer, tanh() activation functions
        self.actor = nn.Sequential(
            layer(obs_dim, hidden, np.sqrt(2)),
            nn.Tanh(),
            layer(hidden, hidden, np.sqrt(2)),
            nn.Tanh(),
            layer(hidden, n_actions, 0.01),
        )
        
        #The Critic-Network, A Sequential Network with 1 hidden layer, tanh() activation functions and binary as output of the network
        self.critic = nn.Sequential(
            layer(obs_dim, hidden, np.sqrt(2)),
            nn.Tanh(),
            layer(hidden, hidden, np.sqrt(2)),
            nn.Tanh(),
            layer(hidden, 1, 1.0)
        )
        
    
    def distribution(self, obs):
        return Categorical(logits = self.actor(obs))
    
    def value(self, obs):
        return self.critic(obs).squeeze(-1)
    
    @torch.no_grad()
    def act(self, obs, greedy = False):
        dist = self.distribution(obs)
        action = dist.probs.argmax(dim = -1) if greedy else dist.sample()
        return action, dist.log_prob(action), self.value(obs)


    def evaluate(self, obs, actions):
        """
            Used in the update: Log-Prob and Entropy of the stored actions under the CURRENT weights
        """    
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.value(obs)
    
    def init_args(self):
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
        }
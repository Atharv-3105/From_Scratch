import numpy as np 
import torch 
import torch.nn as nn 
from torch.distributions import Bernoulli

from graph.lif_simulator import FlyBrainSimulator

class Encoder(nn.Module):
    def __init__(self, obs_dim, n_vp_neurons, current_scale = 2.0):
        super().__init__()
        self.linear = nn.Linear(obs_dim, n_vp_neurons)
        self.current_scale = current_scale
        
    def forward(self, obs):
        #We are using tanH activation function to get bounded [-1, 1] output and then scale to a sensible 
        #Current rane 
        raw = torch.tanh(self.linear(obs))
        return raw * self.current_scale
    

class Decoder(nn.Module):
    """ 
        A simple hidden layer with sigmoid activation function for probability output
    """
    def __init__(self, n_dn_neurons):
        super().__init__()
        self.linear = nn.Linear(n_dn_neurons, 1)
        
    def forward(self, dn_spike_counts):
        logit = self.linear(dn_spike_counts)
        return torch.sigmoid(logit)  #Probability of the Flapping
    
class FlyBrainAgent:
    def __init__(self, obs_dim = 5, internal_steps_per_frame = 15, current_scale = 2.0):
        self.sim = FlyBrainSimulator()
        self.internal_steps = internal_steps_per_frame
        
        self.encoder = Encoder(obs_dim, self.sim.n_vp, current_scale= current_scale)
        self.decoder = Decoder(self.sim.n_dn)
        
        print(f"Agent initialized: {self.sim.n_neurons} total neurons, "
              f"{self.sim.n_vp} input(visual_projection) neurons, "
              f"{self.sim.n_dn} output(descending) neurons")
        
    
    def reset(self):
        self.sim.reset()
        
    def act(self, obs, deterministic = False):
        """ 
            Obs: numpy array, shape (obs_dim) 
            Returns: action(0 or 1), and the flap probability
        """
        obs_t = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)  #Shape (1, obs_dim)
        
        external_current = self.encoder(obs_t).squeeze(0)  #shape (n_vp,)
        
        dn_spike_accumulator = torch.zeros(self.sim.n_dn)
        
        for  _ in range(self.internal_steps):
            self.sim.step(external_current)
            dn_spike_accumulator += self.sim.get_dn_spikes()
            
        flag_prob = self.decoder(dn_spike_accumulator.unsqueeze(0)).squeeze(0)
        
        dist = Bernoulli(probs = flag_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Truncate the backprop chain here: the simulator's membrane potential
        # and spike VALUES carry forward normally (biological recurrence is preserved),
        # but the GRADIENT HISTORY does not chain across game frames.
        # Each frame's log_prob only backprops through its own 15 internal steps.
        self.sim.V = self.sim.V.detach()
        self.sim.spikes = self.sim.spikes.detach()
        
        return int(action.item()), log_prob
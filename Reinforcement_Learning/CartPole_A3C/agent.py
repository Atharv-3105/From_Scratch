import torch
import torch.nn as nn 


''' 
shared layers extract features from observations.
policy outputs unnormalized logits for the action probabilities.
value outputs the estimated value of the current state.
Apply softmax on logits outside the model, when needed.
'''
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        #Define Shared Hidden Layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        
        #Actor:- Outputs Action Logits
        self.policy = nn.Linear(128, action_dim)
        
        #Critic:- Outputs Value of the current state
        self.value = nn.Linear(128, 1)
    
    def forward(self, x:torch.Tensor):
        x = self.shared(x)
        
        #Get the Action Probabilities 
        logits = self.policy(x)
        
        #Get the Value of the State
        value = self.value(x)
        
        return logits, value
        
        
        
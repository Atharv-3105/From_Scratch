import torch 
import torch.nn as nn
import torch.nn.functional as F 

class DQN(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim = 256, enable_dueling_dqn = True):
        super(DQN,self).__init__()
        
        self.enable_dueling_dqn = enable_dueling_dqn
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        if self.enable_dueling_dqn:
            #Value Stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1) #Through this we will get the value
            
            #Advantages Stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        if self.enable_dueling_dqn:
            #Calculate the Value
            v = F.relu(self.fc_value(x))
            V = self.value(v) 
            
            #Calculate the Advantage
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)
            
            #Calculate the Q-value {Value + Advantage - Avg Advantage}
            Q = V + A - torch.mean(A, dim=1, keepdim=True) 
        
        else:
            Q = self.output(x) 
        
        return Q
    

#Example to check functioning of our Network
if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    network = DQN(state_dim, action_dim)
    state = torch.randn((1, state_dim))
    output = network(state)
    
    assert output.shape == (1,action_dim)
    print(output)
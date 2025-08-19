from collections import deque
import random 

#===============Experience Replay class which will store the state,action,reward etc variable in a batch==================
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        
        if seed is not None:
            random.seed(seed)
    
    #transition is the tuple which will store all the observations
    def append(self, transition):
        self.memory.append(transition)
        
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
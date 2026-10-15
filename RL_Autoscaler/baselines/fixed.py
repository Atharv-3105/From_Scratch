import numpy as np 
from sim.batch import previous_day

def window_max(values, k):
    """ 
        result[:, i] = the biggest value among values[:, i], ...., values[:, i + k - 1]
        
        Done by doubling the window length each round (1, 2, 4, 8, ...) so it takes about log2(k) passes
        instead of k passes. 
        At the end 2 overlapping windows of the largest power of two <= k cover exactly k times
    """
    
    n_columns = values.shape[1]
    span = 1 
    
    current = values   #current[:, i] = max of 'span' items starting at i
    while span * 2 <= k:
        current = np.maximum(current[:, :-span], current[:, span:])
        span *= 2 
        
    count = n_columns - k + 1  #How many windows of length k fit 
    return np.maximum(current[:, :count], current[:, k - span : k - span + count])

class KeepAlive:
    """ 
        Fixed Keep-Alive strategy in this we keep as many instances as the busiest minute of the last k minutes.
        AWS uses about 10 minutes and Azure about 20, K = None means never unload:
        
        The first minute of a day look back into the previous day, so the policy needs the table and demand matrix
    """
    
    def __init__(self, k, table, demand):
        self.k = k
        self.table = table 
        self.demand = demand 
        
    def start(self, env, rows):
        width = env.demand.shape[1]  #Get the minutes-per-episode(1440 for a real day)
        if self.k is not None and not 1 <= self.k <= width:
            raise ValueError(f"k must be between 1 and {width}")
        
        #history = [yesterday | today] i.e the pool for minutes t only looks at columns BEFORE width + t
        #so today's later minutes never matter(a test checks also)
        history = np.concatenate([previous_day(self.table, self.demand, rows), env.demand], axis = 1)
        
        if self.k is None:
            busiest_so_far = np.maximum.accumulate(history, axis = 1)
            self.pools = busiest_so_far[:, width - 1 : 2 * width - 1]  #To get the busiest of everything before minute t
        else:
            self.pools = window_max(history, self.k)[:, width - self.k : 2 * width - self.k]  #For this window = minutes t - k, .... t - 1
        
        
    def __call__(self, env):
        return self.pools[:, env.minute]
    
    
def oracle(env):
    """ 
        This strategy follows this minute's demand,
    """
    return env.demand[:, env.minute]

def never_scale(env):
    return np.zeros(env.n_episodes, dtype = np.int64)


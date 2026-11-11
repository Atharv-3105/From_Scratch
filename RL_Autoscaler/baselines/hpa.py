import numpy as np 
from sim.batch import previous_day

class HPA:
    """ 
        The Kubernetes Horizontal Pod Autoscaler rule, run once per minute
        
            desired = ceil(current_replicas * utilisation / target)  
                utilisation = demand / replicas
            
            desired = ceil(demand / target) , the relicas cancel out
            
        - tolerance: if utilisation / target is within +-tolerance of 1, the pool is left alone(default 10%)
        - scale-up is immediate
        - scale-down uses a stabilization window: the pool only fails to the HIGHEST recommendation mdae in the last 'down_window' minutes
        - min_replicas = 1 is the real default of Kuberenetes; min_replicas = 0 means scale to zero is allowed
        All percentages are whole numbers, so the maths is exact; The metric is last minute's demand over the pool
    """
    
    def __init__(self, table, demand, target_pct = 70, tolerance_pct = 10, down_window = 5, min_replicas = 1):
        
        if not 1 <= target_pct <= 200:
            raise ValueError("target_pct must be between 1 and 200")
        
        self.table = table 
        self.demand = demand 
        self.target = target_pct 
        self.tolerance = tolerance_pct 
        self.window = down_window 
        self.min_replicas = min_replicas
        
    def start(self, env, rows):
        #last_minute[:, t] = the demand of minute t - 1(yesterday's last minute for t = 0)
        
        yesterday = previous_day(self.table, self.demand, rows)
        self.last_minute = np.concatenate([yesterday[:, -1:], env.demand[:, :-1]], axis = 1)
        
        #the recommendations of the last few minutes; -1 means "None yet"
        self.recommendations = np.full((env.n_episodes, self.window), -1, dtype = np.int64)
        
    def __call__(self, env):
        pool = env.pool #These are the current replicas
        seen = self.last_minute[:, env.minute]  #demand of the previous minute 
        
        #utlisation / target = (seen / pool) / (target / 100) 
        desired = -(-seen * 100 // self.target)  #this is ceil(seen / target)
        gap = np.abs(seen * 100 - pool * self.target)
        within_band = (pool > 0) & (gap * 100 <= self.tolerance * pool * self.target)
        desired = np.where(within_band, pool, desired)
        desired = np.maximum(desired, self.min_replicas)
        
        #Keep the last `window` recommendations (this minute's included) in a ring
        self.recommendations[:, env.minute % self.window] = desired 
        highest_recent = self.recommendations.max(axis = 1)
        
        new_pool = np.where(pool < desired, desired, pool)  #Scale-Up immediately
        new_pool = np.where(new_pool > highest_recent,  highest_recent, new_pool) #Scale Down: only the highest_recent recommendation
        
        return new_pool.astype(np.int64)
    
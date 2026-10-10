from collections import deque 
from fractions import Fraction 
import numpy as np 

def parse_actions(raw_actions):
    """ 
        [["absolute", 0], ["relative", "3/2"]] --> [("absolute", Fraction(0)), ("relative", Fraction(3,2))].
        Fraction keep the multipliers exact, so rounding up never goes wrong.
    """
    
    parsed = []
    for kind, value in raw_actions:
        if kind not in ("absolute", "relative"):
            raise ValueError(f"unknown action kind: {kind}")
        
        parsed.append((kind, Fraction(str(value))))
        
    return parsed 

def pool_for_action(action, ref_demand, max_pool, actions):
    """ 
        Turn the agent's choice into a number of warm instances.
    """   
    
    kind, value = actions[action]
    if kind == "absolute":
        pool = int(value)
    else:
        #Ceil(value * ref_demand) using only whole numbers(floor division rounds down)
        pool = -(-(ref_demand * value.numerator) // value.denominator)
        
    return min(pool, max_pool)

def pool_cap(train_demand):
    """ 
        Largest pool allowed for a function: twice its biggest training demand, at least 1
    """
    
    if len(train_demand) == 0:
        return 1 
    
    return max(1, 2 * int(np.max(train_demand)))

def reward_scale(train_demand):
    """ 
        Divide rewards by this so a big function and a small one are on the same scale:
            The average demand over the minutes where it was positive, but never below 1
    """
    
    demand = np.asarray(train_demand)
    positive = demand[demand > 0]
    if len(positive) == 0:
        return 1.0
    return max(1.0, float(positive.mean()))

class WarmPoolEnv:
    """ 
        One function, one day, one decision per minute.

    Before minute t the agent picks an action. The pool size comes from the action and from
    R = the biggest demand in the last `reference_window` minutes, which only uses minutes
    BEFORE t. Then the real demand of minute t is revealed and we count:
        cold  = demand the pool could not serve (new instances had to start)
        idle  = pool instances nobody used
        start = instances added compared with the previous minute
    reward = -(rho * cold + idle + kappa * start) / scale
    """
    
    def __init__(self, config):
        self.window = config["reference_window"]
        self.rho = config["rho"]
        self.kappa = config["kappa"]
        self.actions = parse_actions(config["actions"])
        self.n_actions = len(self.actions)
        
    def reset(self, demand, scale = 1.0, max_pool = 1_00_000, mem_mb = 0.0, previous_demand = (), initial_pool = 0):
        """ 
            demand: demand of each minute of this episode
            previous_demand: minutes just before the episode(ex: the end of the previous day)
            
        """
        self.demand = [int(d) for d in demand]  
        self.scale = scale 
        self.max_pool = max_pool 
        self.mem_mb = mem_mb
        
        #The last window demands, the deque drops the oldest values when a new one is added
        self.recent = deque([0] * self.window, maxlen = self.window)
        self.recent.extend(int(d) for d in previous_demand)
        
        self.minute = 0
        self.pool = initial_pool
        self.totals = {"cold": 0, "idle": 0, "start": 0, "idle_gbmin": 0.0, "reward": 0.0}
        return self._state()
    
    def _state(self):
        return {
            "minute": self.minute,
            "ref_demand": max(self.recent),
            "pool": self.pool,
        }
        
    def step(self, action):
        
        if self.minute >= len(self.demand):
            raise RuntimeError("the episode is over, call reset()")
        
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be between 0 and {self.n_actions - 1}")
        
        ref_demand = max(self.recent)
        new_pool = pool_for_action(action, ref_demand, self.max_pool, self.actions)
        
        demand = self.demand[self.minute]  #Revealed only AFTER the pool is chosen
        cold = max(0, demand - new_pool)
        idle = max(0, new_pool - demand)
        start = max(0, new_pool - self.pool)  #shrinking the pool is free
        reward = -(self.rho * cold + idle + self.kappa * start) / self.scale 
        idle_gbmin = idle * self.mem_mb / 1024
        
        info = {
            "minute": self.minute,
            "ref_demand": ref_demand,
            "pool": new_pool,
            "demand": demand,
            "cold": cold, 
            "idle": idle,
            "start": start, 
            "idle_gbmin": idle_gbmin,
            "reward": reward,
        }
        
        for name in self.totals:
            self.totals[name] += info[name]
            
        self.recent.append(demand)
        self.pool = new_pool
        self.minute += 1 
        done = self.minute == len(self.demand)
        return self._state(), reward, done, info 
    
    def summary(self):
        return dict(self.totals)
    
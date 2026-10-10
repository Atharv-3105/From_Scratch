import numpy as np 
from sim.env import parse_actions

class VecWarmPoolEnv:
    """ 
        Many function-days at once, Same rules and same numbers as WarmPoolEnv but every step
        works on whole arrays, one entry per episode. 
        All episodes must have the same length(a full day = 1440 minutes), so they all finish on the same step
    """
    
    def __init__(self, config):
        self.window = config["reference_window"]
        self.rho = config["rho"]
        self.kappa = config["kappa"]
        actions = parse_actions(config["actions"])
        self.n_actions = len(actions)
        
        #One entry per action, so a whole batch of choices can be turned into pools with array indexing
        self.is_relative = np.array([kind == "relative" for kind, _ in actions])
        self.numerator = np.array([value.numerator for _, value in actions], dtype = np.int64)
        self.denominator = np.array([value.denominator for _, value in actions], dtype = np.int64)
        
    def reset(self, demand, scale = 1.0, max_pool = 1_00_000, mem_mb = 0.0, previous_demand = None, initial_pool = 0):
        """ 
            demand:(episode, minutes) The other arguments are one value per episode(or one value for all)
        """
        
        demand = np.asarray(demand, dtype = np.int64)
        self.n_episodes, self.length = demand.shape 
        n = self.n_episodes 
        
        previous = np.zeros((n, self.window), dtype = np.int64)
        if previous_demand is not None:
            given = np.asarray(previous_demand, dtype = np.int64)[:, -self.window:]
            #Shorter history is padded with zeros on the left
            previous[:, self.window - given.shape[1]: ] = given 
            
        #R for EVERY minute at once, It only depends on the demand, never depends on the agent actions
        #so we can compute it up front, history = [minutes before the episode | the epsiode current]
        #R at minute t is the max of history[t ... t + window - 1] which are the `window` minutes before t
        history = np.concatenate([previous, demand], axis = 1)
        ref = history[:, :self.length + 1].copy()  #One extra column: R after the last minute
        for shift in range(1, self.window):
            ref = np.maximum(ref, history[:, shift:shift+self.length+1])
            
        self.demand = demand 
        self.ref = ref 
        self.scale = np.broadcast_to(np.asarray(scale, dtype = np.float64), (n,)).copy()
        self.max_pool = np.broadcast_to(np.asarray(max_pool, dtype = np.int64), (n, )).copy()
        self.mem_mb = np.broadcast_to(np.asarray(mem_mb, dtype = np.float64), (n, )).copy()
        self.pool = np.broadcast_to(np.asarray(initial_pool, dtype = np.int64), (n, )).copy()
        
        self.minute = 0
        self.totals = {
            "cold": np.zeros(n, dtype = np.int64), 
            "idle": np.zeros(n, dtype = np.int64),
            "start": np.zeros(n, dtype = np.int64), 
            "idle_gbmin": np.zeros(n),
            "reward": np.zeros(n),
        }
        return self._state()
    
    def _state(self):
        return {
            "minute": self.minute, 
            "ref_demand": self.ref[:, self.minute],
            "pool": self.pool.copy(),
        }
        
    def step(self, actions):
        if self.minute >= self.length:
            raise RuntimeError("the episode is over, call reset()")
        actions = np.asarray(actions)
        if actions.shape != (self.n_episodes):
            raise ValueError(f"Expected one action per episode, shape ({self.n_episodes})")
        
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(f"actions must be between 0 and {self.n_actions - 1}")
        
        ref_demand = self.ref[:, self.minute]  #built-only from minutes before this one 
        
        numerator, denominator = self.numerator[actions], self.denominator[actions]
        relative_pool = -(-(ref_demand * numerator) // denominator)  #Round-up using whole numbers
        absolute_pool = numerator // denominator 
        
        new_pool = np.minimum(np.where(self.is_relative[actions], relative_pool, absolute_pool), self.max_pool)
        return self._apply_pool(new_pool)
    
    def step_pool(self, pools):
        """ 
            Function to set the pool sizes directly instead of choosing an action, Baselines and the oracle use this
        """
        if self.minute >= self.length:
            raise RuntimeError("the episode is over, call reset()")
        
        pools = np.asarray(pools)
        if pools.shape != (self.n_episodes, ):
            raise ValueError(f"Expected one pool size per episode, shape ({self.n_episodes},)")

        if not np.issubdtype(pools.dtype, np.integer):
            raise TypeError("pool sizes must be whole numbers(round them yourself, e.g. with a ceil)")
        
        if pools.min() < 0:
            raise ValueError("pool sizes cannot be negative")
        
        return self._apply_pool(np.minimum(pools.astype(np.int64), self.max_pool))
    
    def _apply_pool(self, new_pool):
        """ 
            Function reveal's this minute's demand and count what the chosen pool cost
        """
        
        ref_demand = self.ref[:, self.minute]
        demand = self.demand[:, self.minute]  #Revealed only AFTER the pools are chosen
        
        cold = np.maximum(demand - new_pool, 0)
        idle = np.maximum(new_pool - demand, 0)
        start = np.maximum(new_pool - self.pool, 0) #Shrinking the pool is free
        reward = -(self.rho * cold + idle + self.kappa * start) / self.scale 
        idle_gbmin = idle * self.mem_mb / 1024
        
        info = {
            "minute": self.minute, "ref_demand": ref_demand, "pool": new_pool, "demand": demand,
            "cold": cold, "idle": idle, "start": start, "idle_gbmin": idle_gbmin,
            "reward": reward,
        }
        
        for name in self.totals:
            self.totals[name] += info[name]
            
            
        self.pool = new_pool
        self.minute += 1
        return self._state(), reward, self.minute == self.length, info 
        
        
    def summary(self):
        return {name: values.copy() for name,values in self.totals.items()}
        
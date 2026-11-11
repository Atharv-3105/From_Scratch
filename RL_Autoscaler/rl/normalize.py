import numpy as np 
class RunningNorm:
    """ 
        Keeps a running mean and variance of the observations and turns each feature into(x - mean) / std
        Updated in batched so that we never store old observations
    """
    def __init__(self, size, clip = 10.0, eps = 1e-8):
        self.size = size 
        self.clip = clip 
        self.eps = eps 
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4
        
    def update(self, batch):
        batch = np.asarray(batch, dtype = np.float64).reshape(-1, self.size)
        n = len(batch)
        if n == 0:
            return 
        
        batch_mean = batch.mean(axis = 0)
        batch_var = batch.var(axis = 0)
        
        #Merge two groups together without looking at the old data again
        delta = batch_mean - self.mean 
        total = self.count + n
        new_mean = self.mean + delta * n / total 
        
        #m2 = Sum of squared distances from each group's mean, plus a term for the gap between the two means
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total 
        
        self.mean = new_mean 
        self.var = m2 / total 
        self.count = total 
        
    def __call__(self, x):
        x = np.asarray(x, dtype = np.float64)
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
    
    
    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "clip": self.clip,
            "eps": self.eps,
        }
        
    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype = np.float64).copy()
        self.var = np.asarray(state["var"], dtype = np.float64).copy()
        self.count = float(state["count"])
        self.clip = state["clip"]
        self.eps = state["eps"]
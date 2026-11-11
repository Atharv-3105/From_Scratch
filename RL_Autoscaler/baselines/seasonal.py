import numpy as np 
from baselines.fixed import window_max
from sim.batch import previous_day 

class SeasonalPredictive:
    """
        Predict's from the same-time yesterday, but never trust it blindly
        forecast[t] = the biggest demand in yesterday's minutes t - w ..... t + w
        pool[t] = ceil(margin * max(forecast[t], demand[t - 1]))
        
        The forcast pre-warms a function whose traffic repeats every day, The last-minute term (max with demand[t - 1])
        keeps it reactive when today is busier than yesterday, first day there is no yesterday, so the forecast is 0 and the policy is purely
        reactive.
        margin_pct is a whole number (120 - 1.2) so the rounding is exact
    """
    
    def __init__(self, table, demand, half_window = 15, margin_pct = 120):
        self.table = table 
        self.demand = demand 
        self.half_window = half_window 
        self.margin_pct = margin_pct 
        
    def start(self, env, rows):
        width = env.demand.shape[1]
        w = self.half_window 
        if not 0 <= w < width:
            raise ValueError(f"half_window must be between 0 and {width - 1}")
        
        #history is yesterday or today: yesterday's minute t is history[t], today's minute t is history[width + t]
        history = np.concatenate([previous_day(self.table, self.demand, rows), env.demand], axis = 1)
        
        #forecast[:, t] = max(history[t - w ... t + w]). Pad w zeros on the left so the window at the very start of the day
        #is simply cut-off. The largest index used is t + w, which is before "now" (width + t) as w < width
        padded = np.concatenate([np.zeros((history.shape[0], w), dtype = history.dtype), history], axis = 1)
        forecast = window_max(padded, 2 * w + 1)[:, :width]
        
        last_minute = history[:, width - 1 : 2 * width - 1]
        base = np.maximum(forecast, last_minute)
        self.pools = -(-(base * self.margin_pct) // 100)
        
    def __call__(self, env):
        return self.pools[:, env.minute]
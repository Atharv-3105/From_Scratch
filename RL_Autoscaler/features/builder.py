import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view

N_MIN = 1440 #Capped number of minutes for History(that is 1 day only)
TRIGGERS = ["http", "timer", "queue", "orchestration", "event", "storage", "others"]
YESTERDAY_WINDOW = 15   #Same window yesterday means  the 15 minutes starting at this minute yesterday

#The agent will see these features in this order, Demand-like ones are divided by the function's scale, so a tiny function and a
#huge one look alike; log_scale can tell them apart

HISTORY_FEATURE_NAMES = [
    "last_minute", "mean_5", "ref_demand", "mean_15", "max_15", "mean_60",
    "max_60", "trend", "log_minutes_since_demand", "time_sin", "time_cos", 
    "weekday_sin", "weekday_cos", "yesterday_mean", "yesterday_max", "has_yesterday",
    "log_mem", "log_avg_duration", "log_scale"
]  + [f"trigger_{name}" for name in TRIGGERS]

#These 2 features depend on what the agent did before, so they can't be computed ahead of time
POOL_FEATURE_NAMES = ["pool", "pool_over_ref"]

FEATURE_NAMES = HISTORY_FEATURE_NAMES + POOL_FEATURE_NAMES

def trigger_index(name):
    return TRIGGERS.index(name) if name in TRIGGERS else TRIGGERS.index("others")

def history_features(today_so_far, yesterday, has_yesterday, day, statics):
    """ 
        The reference version: features of ONE decision, using only what happened before it,
        today_so_far: demand of today's minutes 0...t-1 (so t = its length). Yesterday: all 1440 minute 
    """
    today = np.asarray(today_so_far, dtype = np.float64)
    history = np.concatenate([np.asarray(yesterday, dtype = np.float64), today])  
    t = len(today)
    scale = statics["scale"]
    
    last5, last15, last60 = history[-5:], history[-15:], history[-60:]
    mean5, mean60 = last5.mean(), last60.mean()
    
    #Calculate the minutes since the last non-zero minute, 1 means "the minute just before current". This is capped at 1 day(meaning at max 1440)
    nonzero = np.nonzero(history)[0]
    since = N_MIN if len(nonzero) == 0 else min(len(history) - nonzero[-1], N_MIN)
    
    #The same minutes yesterday; history[t] is "now" 24 hours agon, every index is used before "now"
    yesterday_window = history[t: t + YESTERDAY_WINDOW]
    
    minute_angle = 2 * np.pi * t / N_MIN 
    weekday_angle = 2 * np.pi * ((day - 1) % 7) / 7  #Day 1 is treated as Monday
    
    one_hot = np.zeros(len(TRIGGERS))
    one_hot[trigger_index(statics["trigger"])] = 1.0 
    
    values = [history[-1] / scale, mean5 / scale, last5.max() / scale, last15.mean() / scale, last15.max() / scale, last60.mean() / scale, 
              last60.max() / scale, (mean5 - mean60) / scale, np.log1p(since), np.sin(minute_angle), np.cos(minute_angle), np.sin(weekday_angle),
              np.cos(weekday_angle), yesterday_window.mean() / scale, yesterday_window.max() / scale, float(has_yesterday), np.log1p(statics["mem_mb"]), 
              np.log1p(statics["avg_dur_ms"]), np.log(scale)]
    
    return np.concatenate([values, one_hot])

def precompute_history_features(demand, yesterday, has_yesterday, day, statics, dtype = np.float64):
    """ 
        The fast version of our history_features function: Calculate the same features for EVERY minute of many episode at once
        demand, yesterday: (episodes, 1440). 
        has_yesterday, 
        day: (episodes, )
        statics: dict of (episodes,) arrays
        Since trigger already turned into an index, Returns(episodes, minutes, 26)
        It must give the same numbers as history_features(a test checks this)
    """
    
    n, minutes = demand.shape 
    history = np.concatenate([yesterday, demand], axis = 1).astype(np.float64)
    now = N_MIN + np.arange(minutes)  #position of each decision inside history
    scale = np.asarray(statics["scale"], dtype = np.float64)[:, None]
    
    #Get the Means over the last K minutes from running sums: sum(a..b) = cumulative[b] - cumulative[a]
    cumulative = np.concatenate([np.zeros((n, 1)), np.cumsum(history, axis = 1)], axis = 1)
    
    def mean_last(k):
        return (cumulative[:, now] - cumulative[:, now - k]) / k 
    
    def max_last(k):
        #Windows[:, j] holds history[:, j...j+k-1]; the window ending just before `now` starts at now  - k
        return sliding_window_view(history, k, axis = 1).max(axis = 2)[:, now - k]
    
    mean5, mean60 = mean_last(5), mean_last(60)
    
    #Position of the last non-zero minute at or before each position, then look at the one before "now"
    position = np.arange(history.shape[1])
    last_nonzero = np.maximum.accumulate(np.where(history > 0, position, -1), axis = 1)
    
    before_now = last_nonzero[:, now - 1]
    since = np.where(before_now < 0, N_MIN, np.minimum(now - before_now, N_MIN))
    
    #Same 15 minutes yesterday: The window starting at position t(t = 0...minutes - 1)
    
    windows = sliding_window_view(history, YESTERDAY_WINDOW, axis = 1)
    yesterday_mean = windows.mean(axis = 2)[:, :minutes]
    yesterday_max = windows.max(axis = 2)[:, :minutes]
    
    minute_angle = 2 * np.pi * np.arange(minutes) / N_MIN 
    weekday_angle = 2 * np.pi * ((np.asarray(day) - 1) % 7) / 7
    
    def across(column): #one value per episode -> (episodes, minutes)
        return np.broadcast_to(np.asarray(column, dtype = np.float64)[:, None], (n, minutes)) 
    
    def over_time(row):  #one value per minute -> (episodes, minutes)
        return np.broadcast_to(row[None, :], (n, minutes))
    
    one_hot = np.eye(len(TRIGGERS))[np.asarray(statics["trigger"])] #(episodes, 7)
    
    columns = [history[:, now - 1] / scale, mean5 / scale, max_last(5) / scale, mean_last(15) / scale, 
               max_last(15) / scale, mean60 / scale, max_last(60) / scale, (mean5 - mean60) / scale, 
               np.log1p(since), over_time(np.sin(minute_angle)), over_time(np.cos(minute_angle)), across(np.sin(weekday_angle)), 
               across(np.cos(weekday_angle)), yesterday_mean / scale, yesterday_max / scale, across(has_yesterday), 
               across(np.log1p(statics["mem_mb"])), across(np.log1p(statics["avg_dur_ms"])), np.log(scale) * np.ones((1, minutes))] + [across(one_hot[:, k]) for k in range(len(TRIGGERS))]
    
    return np.stack(columns, axis = 2).astype(dtype)


def pool_features(pool, ref_demand, scale):
    """ 
        The two features that depend on the agent's previous choice:
        the pool size relative to the function's scale, and relative to the reference 
        demand (at least 1)
    """   
    pool = np.asarray(pool , dtype = np.float64)
    return np.stack([pool / scale, pool / np.maximum(ref_demand, 1)], axis = -1)

def full_observation(history_row, pool, ref_demand, scale):
    """ 
        History features of one minute(episodes, 26) + the pool features ==> (episodes, 26)
    """
    
    return np.concatenate([history_row, pool_features(pool, ref_demand, scale)], axis = -1)

class OnlineFeatureBuilder:
    """ 
        This will be for a live controller; feed it the  demand of each minute as it happens, ask for feature before each decision
        Uses the same function as the simulator, so our training and serving can't drift apart
    """
    
    def start_day(self, yesterday, has_yesterday, day, statics):
        self.yesterday = np.asarray(yesterday, dtype = np.float64)
        self.has_yesterday = has_yesterday
        self.day = day 
        self.statics = statics 
        self.today = []
        
    def observe(self, demand):
        self.today.append(float(demand))
        
    def features(self):
        return history_features(self.today, self.yesterday, self.has_yesterday, self.day, self.statics)
    
    

def function_avg_duration(table, train_days):
    """ 
        Each function's average duration(ms) over TRAIN days, weighted by invocations.
        The same-day value would be small lookahead(it averages the whole day), so the agent only ever
        sees this fixed number
    """
    
    train = table[table["day"].isin(train_days)]
    weighted = (train["dur_ms"] * train["total_invocations"]).groupby(train["func_id"]).sum()
    invocations = train["total_invocations"].groupby(train["func_id"]).sum()
    average = weighted / invocations.where(invocations > 0) #NaN for functions with no invocations
    fallback = average.median()  #Such functions get the typical value 
    return average.fillna(fallback)

def episode_inputs(table, demand, row_ids, constants, avg_duration):
    """ 
        This function gives everything the feature-builder needs for the chosen row
    """
    row_ids = np.asarray(row_ids)
    func_ids = table["func_id"].to_numpy()
    days = table["day"].to_numpy()
    row_of = {(f, d): i for i, (f, d) in enumerate(zip(func_ids, days))}
    
    yesterday = np.zeros((len(row_ids), demand.shape[1]), dtype = np.int64)
    has_yesterday = np.zeros(len(row_ids), dtype = bool)
    for k, row in enumerate(row_ids):
        before = row_of.get((func_ids[row], days[row] - 1))
        if before is not None:
            yesterday[k] = demand[before]
            has_yesterday[k] = True 
            
    return {
        "demand": demand[row_ids],
        "yesterday": yesterday,
        "has_yesterday": has_yesterday, 
        "day": days[row_ids],
        "statics": {
            "mem_mb": table["mem_mb"].to_numpy()[row_ids],
            "avg_dur_ms": avg_duration.loc[func_ids[row_ids]].to_numpy(),
            "scale": constants.loc[func_ids[row_ids]]["scale"].to_numpy(),
            "trigger": np.array([trigger_index(name) for name in table["Trigger"].to_numpy()[row_ids]]),
        },
    }
import numpy as np 


DEFAULTS = {
    "range_minutes": 240,      #histrogram covers idle times up to 4 hours; longer ones are "out of bounds"
    "head_pct": 5,             #the 5th percentile of idle times sets the pre-warm window
    "tail_pct": 99,            #the 99th percentile sets the end of the keep-alive window
    "margin_pct": 10,          #pre-warm window 10% shorter, keep-alive window 10% longer
    "cv_threshold": 2.0,       #histogram is "representative" only if the CV of it's bin counts is at least this
    "min_intervals": 10,       #fewer idle times than this
    "oob_fraction_pct": 50,    #more than this share of idle times out of bounds
    "forecast_margin_pct": 15, #the paper mentioned 15% margin aroung a forecast idle time
    "peak_window": 5,          #while warm, hold the busiest of the last 5 minutes ending at the last call
}

def interval_histogram(busy_minutes, range_minutes):
    """ 
        Idle times between consecutive busy minutes, 
        RETURNS: (counts, out_of_bounds)
        counts[i] = how many idle times of exactly i minutes: for i up to range_minutes
        out_of_bounds = the idle times that were longer than the range
    """
    
    gaps = np.diff(np.sort(np.asarray(busy_minutes, dtype = np.int64)))
    in_range = gaps[gaps <= range_minutes]
    counts = np.bincount(in_range, minlength = range_minutes + 1)
    return counts, gaps[gaps > range_minutes]

def choose_windows(counts, out_of_bounds, settings):
    
    #Number of historical intervals represented inside and outside the histogram
    n_in , n_out = int(counts.sum()), len(out_of_bounds)
    total = n_in + n_out 
    
    #fallback: use the full configured range as the keep-alive window policy when the data is insufficient to determine anything
    fallback = ("fallback", 0, settings["range_minutes"])
    
    #Without enough historical intervals, percentile estimates may be unreliable
    if total < settings["min_intervals"]:
        return fallback 
    
    #If too many intervals exceed the histogram's range, the in-range between may not
    #adequately represent the function's behavior
    #We then use the median of the most recent five out-of-bounds gaps as a simple forecast instead
    #of the ARIMA-Based forecasting which was mentioned in the Paper
    if 100 * n_out > settings["oob_fraction_pct"] * total:
        predicted = int(np.median(out_of_bounds[-5:]))
        margin = settings["forecast_margin_pct"]
        
        #Start the warm window 15% before the predicted gap, using floor division
        prewarm = predicted * (100 - margin) // 100
        
        #End the warm-window 15% after the predicted gap
        end = -(-predicted * (100 + margin) // 100)  #This is ceiling-division we are using
        
        return "forecast", prewarm, end 
    
    
    #Ignore count 0 when evaluating the distribution of positive gap lengths
    #The CV(coefficient-of-variation) measures the STD relative to the mean of the histogram's bin counts
    bins = counts[1:].astype(np.float64) 
    cv = bins.std() / bins.mean()
    
    #A low CV means the counts are relatively similar across bins, so this implementation treats the 
    #histogram as insufficiently informative
    if cv < settings["cv_threshold"]:
        return fallback 
    
    #convert histogram counts into cumulative counts
    #this tells us how many observed intervals have a gap <= i.
    cumulative = np.cumsum(counts)
    
    #Find the first gap length whose cumulative count reaches the desired presentile threshold
    head = int(np.searchsorted(cumulative, settings["head_pct"] * n_in / 100, side = "left"))
    
    tail = int(np.searchsorted(cumulative, settings["tail_pct"] * n_in / 100, side = "left"))
    
    margin = settings["margin_pct"]
    
    #Shorten the initial zero-pool period by the configured margin
    prewarm = head * (100 - margin) // 100
    
    #Calculate the interval between the pre-warm boundary and the tail, 
    #then extend that interval by the margin
    keep_alive = -(-(tail - prewarm) * (100 + margin) // 100)
    
    return "histogram", prewarm, prewarm + keep_alive 


class HybridHistogram:
    """ 
        The hybrid histogram policy mentioned in the Paper adapted to a minute-level pool

        Every function has it's own idle-time histogram built from ITS EARLIER DAYS ONLY
        After a busy minute, the pool is 0 for 'prewarm' minutes, then equals the recent peak until'end' minutes after the busy minute
        
        Differences from the paper:
            - minute-binned data instead of invocation timestamps
            - the histogram is rebuilt once per day, not after every invocation;
            - the median of reent out-of-bounds idle times replaces the ARIMA forecast
            - history odler than yesterday is unavailable to the pool rule, so very long windows never tirgger
    """
    
    def __init__(self, table, demand, settings = None):
        self.table = table 
        self.demand = demand 
        self.settings = {**DEFAULTS, **(settings or {})}
        
        #Store the function ID and day for each row so we can find and chronologically order
        #the history belonging to a particular function
        self.func_ids = table["func_id"].to_numpy()
        self.days = table["day"].to_numpy()
        
        #Cache the selected timing window for each function and day
        self.cache = {}
        
    def windows_for(self, func_id, day):
        if func_id not in self.cache:
            width = self.demand.shape[1]
            
            #Find all the rows for this function, then process them from earliest to latest so 
            #each day's decision uses only earlier day
            rows = np.nonzero(self.func_ids == func_id)[0]
            rows = rows[np.argsort(self.days[rows])]
            
            #Store the absolute busy-minute positions from days already processed
            busy_so_far = []
            result = {}
            
            for row in rows:
                if busy_so_far:
                    #Build the historical gap distribution using only previously processed days,
                    counts, out_of_bounds = interval_histogram(np.concatenate(busy_so_far), self.settings["range_minutes"])
                    
                else:
                    #There is no history for the first available day
                    counts = np.zeros(self.settings["range_minutes"] + 1, dtype = np.int64)
                    out_of_bounds = np.array([], dtype = np.int64)
                    
                
                #Select and store the policy mode and timing boundaries for this function on
                #this particular day
                result[int(self.days[row])] = choose_windows(counts, out_of_bounds, self.settings)
                
                #Identify the minute indices where demand is positive
                busy_minutes = np.nonzero(self.demand[row] > 0)[0]
                
                #Convert the day's local minute indices into positions on a shared timeline.
                #This let's interval_histogram calculate gaps between busy minutes on different days as well
                
                busy_so_far.append((int(self.days[row]) - 1) * width + busy_minutes)
                
            self.cache[func_id] = result 
            
        return self.cache[func_id][int(day)]
    
    def start(self, env, rows):
        from sim.batch import previous_day 
        
        width = env.demand.shape[1]
        rows = np.asarray(rows)
        
        #Select the historical policy window for each episode in the batch
        chosen = [self.windows_for(self.func_ids[row], self.days[row]) for row in rows]
        
        #Get the prewarm and end values
        # [:, None] changes each array from shape (episodes,) to
        # (episodes, 1), allowing NumPy to broadcast values over minutes.
        prewarm = np.array([c[1] for c in chosen])[:, None]
        end = np.array([c[2] for c in chosen])[:, None] 
        
        #Keep the selected mode for each epsiode for debugging purposes
        self.modes = [c[0] for c in chosen]
        
        # Put yesterday before today in one timeline.
        # This gives the policy context before the first minute of today's episode.
        history = np.concatenate([previous_day(self.table, self.demand, rows), env.demand],axis=1)
        
        #Create the time index corresponding to every column in the combined histogram
        position = np.arange(history.shape[1])
        
        #At busy position, store the position index; at idle positions store - 1
        #The cumulative maximum then propagates the most recent busy index
        last_busy = np.maximum.accumulate(np.where(history > 0, position, -1), axis = 1)
        
        #'now' contains the combined-history positions for each minute of today
        now = width + np.arange(width)
        
        #Look at the position immediately before each decision minute,
        latest = last_busy[:, now - 1]
        
        #Elapsed time from the latest busy position to the current decision position
        minutes_since = now[None, :] - latest 
        
        #Compute the maximum demand over a trailing window of `peak_window`
        #minutes at every position in the combined history
        peak = history.copy()
        
        for shift in range(1, self.settings["peak_window"]):
            #Compare the current peak with demand `shift` positions earlier
            peak[:, shift : ] = np.maximum(peak[:, shift:], history[:, :-shift])
            
        
        #A pool is warm only if:
            #1. An earlier busy minute exists
            #2. The initial zero-pool interval has elapsed
            #3. The keep-alive window has not expired
        warm = ((latest >= 0) & (minutes_since > prewarm) & (minutes_since <= end))
        
        self.pools = np.where(warm, np.take_along_axis(peak, np.maximum(latest, 0), axis = 1), 0)
        
    def __call__(self, env):
        return self.pools[:, env.minute]
        
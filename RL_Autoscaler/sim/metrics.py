from pathlib import Path 
import numpy as np 
import pandas as pd 

#The metrics that can be summed across multiple days of one function
SUM_COLUMNS = ["invocations", "cold_inv", "idle_gbmin", "served", "pool_minutes", "churn", "reversals", "minutes", "reward"]

#The order here determines the order of the rows in the final report
METRIC_NAMES = ["cold_pct_pooled", "cold_pct_mean", "cold_pct_p75", "waste_gbmin_per_day", "utilisation", "churn_per_day",
                "flaps_per_hour", "reward_per_day"]

def weighted_percentile(values, weights, q):
    """ 
        Function to get the weighted percentile.
        
        q = 0.75 means the 75th percentile of weight
        
    A function with weight 5 represents 5x population as a function with weight 1
    """
    
    values, weights = np.asarray(values, dtype = np.float64), np.asarray(weights, dtype = np.float64)
    #Sort the values, keeping weights aligned
    order = np.argsort(values, kind = "stable")
    values = values[order]
    
    #Get the cumulative fraction of represented population
    share = np.cumsum(weights[order] / weights.sum())
    
    #Find the first value the reaches the requested percentile
    position = np.searchsorted(share, q - 1e-9) #Add small tolerance
    position = min(position, len(values) - 1)
    return float(values[position])

def churn_and_reversals(pool, initial_pool = 0):
    """ 
        pool: (episodes, minutes). 
        churn = total size of all changes, 
        reversals = how many times the pool switches between growing and shrinking(a measure of flapping)
    """
    
    pool = np.asarray(pool, dtype = np.int64)
    start = np.broadcast_to(np.asarray(initial_pool, dtype = np.int64), (pool.shape[0], ))[:, None]
    
    change = np.diff(pool, axis = 1, prepend = start)
    
    #changes [2, 3, 0, -2] have churn 7
    churn = np.abs(change).sum(axis = 1)
    
    # +1 = scale up, -1 = scale down, 0 = no change
    direction = np.sign(change)
    
    #For every minute, the position of the latest earlier minute where the pool actually changed
    last_move = np.maximum.accumulate(np.where(direction != 0, np.arange(pool.shape[1]), -1), axis = 1)
    reversals = np.zeros(pool.shape[0], dtype = np.int64)
    
    #process one minute at a time, but process all episodes simulatenously
    for minute in range(pool.shape[1]):
        current = direction[:, minute]
        
        #A reversal occurs when the new direction differs from the previous non-zero direction
        changed_direction = ((current != 0) & (last_move != 0) & (current != last_move))
        reversals += changed_direction.astype(np.int64)
        
        #Only an actual change updates the last-direction, A zero change leaves the previous direction untouched
        last_direction = np.where(current != 0, current, last_direction)
        
        
    # earlier_move = np.concatenate([np.full((pool.shape[0], 1), -1), last_move[:, :-1]], axis = 1)
    # earlier_direction = np.where(earlier_move >= 0, np.take_along_axis(direction, np.maximum(earlier_move, 0), axis = 1), 0)
    
    # reversals = ((direction != 0) & (earlier_direction != 0) & (direction != earlier_direction)).sum(axis = 1)
    
    return churn, reversals 

def to_per_function(episodes):
    """ 
        Combine all days belonging to the same function
        
        Additive metrics are summed, Weight and Workload group are retained once per function
    """
    
    #Add up the daily measurements
    sums = episodes.groupby("func_id")[SUM_COLUMNS].sum()
    
    #Weight and group are assumed constant across the function's days
    metadata = episodes.groupby("func_id")[["weight", "group"]].first()
    
    #Count how many function-days contributed to each row
    day_counts = episodes.groupby("func_id").size().rename("n_days")
    
    return (metadata.join(sums).join(day_counts).reset_index())
    
def _columns(per_function):
    """ 
        Convert a per-function dataframe into Numpy array so the calculations are efficient
    """
    
    names = ["weight", "n_days"] + SUM_COLUMNS
    return {
        name: per_function[name].to_numpy(dtype = np.float64)
        for name in names 
    }
    
def _summarize(columns):
    """ 
        Function to calculate the eight aggregate metrics
    """
    
    weight = columns["weight"]
    
    #Weighted total of any metric
    def total(name):
        return np.sum(weight * columns[name])
    
    invocations = columns["invocations"]
    cold_inv = columns["cold_inv"]
    
    #cold percentages are undefined for functions with no traffic
    has_traffic = invocations > 0
    
    cold_pct = (100.0 * cold_inv[has_traffic] / invocations[has_traffic])
    
    #Keep the weights aligned with the functions that have traffic
    traffic_weights = weight[has_traffic]
    
    if has_traffic.any() and traffic_weights.sum() > 0:
        cold_pct_mean = np.average(
            cold_pct, weights = traffic_weights
        )
        
        cold_pct_p75 = weighted_percentile(
            cold_pct,
            traffic_weights,
            q=0.75,
        )
    else:
        cold_pct_mean = np.nan
        cold_pct_p75 = np.nan

    return {
        # Overall cold-start rate, dominated by high-traffic functions.
        "cold_pct_pooled": (
            100.0 * total("cold_inv")
            / max(total("invocations"), 1e-12)
        ),

        # Weighted average of individual functions' cold percentages.
        "cold_pct_mean": cold_pct_mean,

        # Weighted 75th percentile of function-level cold percentages.
        "cold_pct_p75": cold_pct_p75,

        # Wasted memory capacity per represented function-day.
        "waste_gbmin_per_day": (
            total("idle_gbmin") / max(total("n_days"), 1e-12)
        ),

        # Fraction of available pool instance-minutes that were served.
        "utilisation": (
            total("served") / max(total("pool_minutes"), 1e-12)
        ),

        # Absolute replica changes per represented function-day.
        "churn_per_day": (
            total("churn") / max(total("n_days"), 1e-12)
        ),

        # Direction reversals per represented hour.
        "flaps_per_hour": (
            total("reversals") / max(total("minutes") / 60, 1e-12)
        ),

        # Accumulated reward per represented function-day.
        "reward_per_day": (
            total("reward") / max(total("n_days"), 1e-12)
        ),
    }


def summarize(per_function):
    """Return one dictionary of aggregate performance metrics."""
    return _summarize(_columns(per_function))


def bootstrap_ci(per_function, n_boot=1000, level=0.95, seed=0):
    """
    Estimate metrics and confidence intervals by resampling functions.

    Functions are sampled with replacement within each workload group.
    Every bootstrap sample retains the original group sizes.
    """
    rng = np.random.default_rng(seed)

    columns = _columns(per_function)
    groups = per_function["group"].to_numpy()

    # Store the row indices belonging to each workload group.
    group_indices = [
        np.flatnonzero(groups == group)
        for group in np.unique(groups)
    ]

    # Metric estimates from the original data.
    estimate = _summarize(columns)

    bootstrap_results = []

    for _ in range(n_boot):
        sampled_indices = []

        # Resample functions within each group, not individual days.
        for indices in group_indices:
            sampled = rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )
            sampled_indices.append(sampled)

        sampled_indices = np.concatenate(sampled_indices)

        # Recalculate every metric on this bootstrap sample.
        sampled_columns = {
            name: values[sampled_indices]
            for name, values in columns.items()
        }

        bootstrap_results.append(_summarize(sampled_columns))

    bootstrap_results = pd.DataFrame(bootstrap_results)

    # For a 95% interval, use the 2.5th and 97.5th percentiles.
    tail = (1 - level) / 2

    return pd.DataFrame({
        "estimate": pd.Series(estimate),
        "low": bootstrap_results.quantile(tail),
        "high": bootstrap_results.quantile(1 - tail),
    }).loc[METRIC_NAMES]


def summarize_by_group(per_function, n_boot=1000, level=0.95, seed=0):
    """
    Generate confidence intervals for the overall population
    and separately for each workload group.
    """
    results = {
        "all": bootstrap_ci(per_function, n_boot, level, seed)
    }

    for group_name, group_data in per_function.groupby("group"):
        results[group_name] = bootstrap_ci(
            group_data,
            n_boot,
            level,
            seed,
        )

    return pd.concat(results)


def pareto_front(x, y):
    """
    Return indices of non-dominated points.

    Both metrics are assumed to be minimized.
    For our project:
        x = wasted memory
        y = cold-start percentage
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain finite values.")

    # Sort by x first, then y.
    # For equal x, the lowest y must come first.
    order = np.lexsort((y, x))

    keep = []
    best_y = np.inf

    for index in order:
        # Keep a point only if it improves on every previously
        # considered point's y value.
        if y[index] < best_y:
            keep.append(index)
            best_y = y[index]

    return np.asarray(keep, dtype=int)


def save_table(table, path):
    """Create the output directory and save a DataFrame as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    
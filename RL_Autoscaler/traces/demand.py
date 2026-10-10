from pathlib import Path 
import numpy as np 
import pandas as pd 

MS_PER_MINUTE = 60_000

def build_demand(counts, dur_ms, concurrency: int = 1, chunk_size = 20_000):
    """ 
        Function which tells us How Many Instances Must Be Busy In Each Minute?
        Inputs:
            counts:  (rows, 1440) invocations per minute
            dur_ms:  (rows,) average duration of that function-day, in ms
            concurrency: how many requests one instance handles at the same time (can be 1 or 4)
            demand = 0  , if there were no invocations
                   = max(1, ceil( work / capacity))  
                        work = invocations * durations(instance-milliseconds of work in the minute)
                        capacity = 60_000 * concurrency (what one instance can do in a minute)
    """
    
    if np.isnan(dur_ms).any() or (dur_ms < 0).any():
        raise ValueError("dur_ms must not contain NaN or negative values")
    
    #Convert the dur_ms to int so that the ceil() function remains whole-numbered not float-ceiled
    dur_ms = np.rint(dur_ms).astype(np.int64)
    capacity = MS_PER_MINUTE * concurrency
    
    # int32 (signed): the largest possible demand is about 9.4 million, which fits, and
    # signed numbers cannot wrap around when later code computes "pool - demand".
    demand = np.empty(counts.shape, dtype=np.int32)
    
    #Work in chunks of rows so the big int64 intermediate array stay small
    for start in range(0, len(counts), chunk_size):
        end = start + chunk_size 
        
        invocations = np.asarray(counts[start:end]).astype(np.int64)
        work = invocations * dur_ms[start:end, None]
        
        instances = -(-work // capacity)
        demand[start : end] = np.where(invocations > 0, np.maximum(instances, 1), 0)
        
    return demand 


def subset_demand(processed_dir, concurrency = 1):
    """ 
        Load the sampled functions and build their demand, 
    """
    processed_dir = Path(processed_dir)
    day_table = pd.read_parquet(processed_dir / "subset_function_days.parquet")
    counts = np.load(processed_dir / "subset_minutes.npy", mmap_mode = "r")
    return day_table, build_demand(counts, day_table["dur_ms"].to_numpy(), concurrency)


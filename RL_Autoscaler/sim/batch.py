import numpy as np 
import pandas as pd 

LAST_DAY = 14 

def function_constants(table, demand, train_days):
    """ 
        Per Functions, from TRAIN days only(same rules as pool_cap and reward_scale)
        max_pool = twice the biggest demand ever seen, at least 1 
        scale = average demand over the minutes where it was positive, at least 1 table and demand have the same row order
    """
    
    on_train_days = table["day"].isin(train_days).to_numpy()
    train_demand = demand[on_train_days]
    rows = pd.DataFrame({
        "func_id": table["func_id"].to_numpy()[on_train_days],
        "peak": train_demand.max(axis = 1),
        "total": train_demand.sum(axis = 1, dtype = np.int64),  #Demand is never negative, so this is the sum of the positive minutes
        "active": (train_demand > 0).sum(axis = 1),
    })
    per_func = rows.groupby("func_id").agg(peak = ("peak", "max"), total = ("total", "sum"), active = ("active", "sum"))
    per_func["max_pool"] = np.maximum(1, 2 * per_func["peak"])
    
    mean_positive = per_func["total"] / per_func["active"].where(per_func["active"] > 0)  #NaN if never active
    per_func["scale"] = np.maximum(1.0, mean_positive).fillna(1.0)
    return per_func[["max_pool", "scale"]]

def fill_silent_days(table, demand, mode = "inside"):
    """ 
        Function that only has a row for days it fired, Add all-zero rows for the missing days so a policy
        that keeps instances warm also pays for the quiet days.
            "inside" : only days between the function's first and last appearance
            "to_end" : from the first appearance until the day 14(a deleted function would be wrongly charged)
            "none" : add nothing
        
        Returns the new table and demand, Original rows keep their order; new rows go at the end 
    """
    
    table = table.reset_index(drop = True).copy()
    table["silent"] = False 
    if mode == "none":
        return table, demand 
    
    if mode not in ("inside", "to_end"):
        raise ValueError(f"unknown mode: {mode}")
    
    new_rows = []
    for func_id, days in table.groupby("func_id")["day"]:
        first, present = days.min(), set(days)
        last = days.max() if mode == "inside" else LAST_DAY
        template = table.loc[days.index[0]]  #Copy the function's settings 
        #Iterate over the days
        for day in range(first, last + 1):
            if day not in present:
                row = template.copy()
                row["day"], row["total_invocations"], row["silent"] = day, 0, True 
                
                if "dur_source" in row.index:
                    row["dur_source"] = "silent"
                
                if "source_row" in row.index:
                    row["source_row"] = -1 
                    
                new_rows.append(row)
                
    if not new_rows:
        return table, demand 
    
    added = pd.DataFrame(new_rows).reset_index(drop = True)
    new_demand = np.vstack([demand, np.zeros((len(added), demand.shape[1]), dtype = demand.dtype)])
    
    return pd.concat([table, added], ignore_index = True), new_demand 

def build_batch(table, demand, row_ids, constants, window = 5):
    """ 
        Arrays ready for VecWarmPoolEnv.reset() for the chosen rows
        previous_demand = the last 'window' minutes of the same function's previous day. 0 if that day has NO ROW
    """
    
    row_ids = np.asarray(row_ids)
    func_ids = table["func_id"].to_numpy()
    days = table["day"].to_numpy()
    
    row_of = {(f, d): i for i, (f, d) in enumerate(zip(func_ids, days))}
    
    previous = np.zeros((len(row_ids), window), dtype = np.int64)
    for k, row in enumerate(row_ids):
        before = row_of.get((func_ids[row], days[row] - 1))
        
        if before is not None:
            previous[k] = demand[before, -window : ]
            
    chosen = constants.loc[func_ids[row_ids]]
    
    return {
        "demand": demand[row_ids],
        "scale": chosen["scale"].to_numpy(),
        "max_pool": chosen["max_pool"].to_numpy(),
        "mem_mb": table["mem_mb"].to_numpy()[row_ids],
        "previous_demand": previous, 
    }
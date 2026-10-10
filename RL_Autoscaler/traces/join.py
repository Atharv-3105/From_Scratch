import numpy as np 
import pandas as pd 
FUNC_KEYS = ["HashOwner", "HashApp", "HashFunction"]
DAY_KEYS = FUNC_KEYS + ["day"]
APP_KEYS = ["HashOwner", "HashApp"]

def clean_durations(dur):
    d = dur.loc[(dur["Average"] >= 0) & (dur["Count"] > 0), DAY_KEYS + ["Average", "Count"]].copy()
    d["weighted"] = d["Average"] * d["Count"]
    g = d.groupby(DAY_KEYS, as_index = False)[["weighted", "Count"]].sum()
    g["dur_ms"] = g["weighted"] / g["Count"]
    return g[DAY_KEYS + ["dur_ms", "Count"]]

def function_avg(day_tbl, train_days):
    t = day_tbl[day_tbl["day"].isin(train_days)].copy()
    t["weighted"] = t["dur_ms"] * t["Count"]
    g = t.groupby(FUNC_KEYS, as_index = False)[["weighted", "Count"]].sum()
    g["fn_avg"] = g["weighted"] / g["Count"]
    return g[FUNC_KEYS + ["fn_avg"]]

def fill(out, value_col, source_col, candidate, label):
    missing = out[value_col].isna() & candidate.notna()
    out.loc[missing, value_col] = candidate[missing]
    out.loc[missing, source_col] = label 
    
def build_function_day_table(keys, duration, memory, train_days = range(1, 11)):
    train_days = list(train_days)
    output = keys.reset_index(drop = True).copy()
    n = len(output)
    
    #Merge Duration table on (HashOwner, HashApp, HashFunction, day)
    day_tbl = clean_durations(duration)
    output = output.merge(day_tbl[DAY_KEYS + ["dur_ms"]], on = DAY_KEYS, how = "left", validate = "many_to_one")
    output["dur_source"] = np.where(output["dur_ms"].notna(), "same_day", "")
    
    fn_avg = function_avg(day_tbl, train_days)
    output = output.merge(fn_avg, on = FUNC_KEYS, how = "left", validate = "many_to_one")
    fill(output, "dur_ms", "dur_source", output["fn_avg"], "function_avg")
    
    global_dur = day_tbl.loc[day_tbl["day"].isin(train_days), "dur_ms"].median()
    fill(output, "dur_ms", "dur_source", pd.Series(global_dur, index = output.index), "global")
    output = output.drop(columns = "fn_avg")
    
    m = memory[memory["day"].isin(train_days)]
    app_mem = (m.groupby(APP_KEYS, as_index = False)["AverageAllocatedMb"].median().rename(columns = {"AverageAllocatedMb": "app_mb"}))
    output = output.merge(app_mem, on = APP_KEYS, how = "left", validate = "many_to_one")
    output["mem_mb"] = output["app_mb"]
    output["mem_source"] = np.where(output["mem_mb"].notna(), "app_median", "")
    global_mem = memory["AverageAllocatedMb"].median()
    fill(output, "mem_mb", "mem_source", pd.Series(global_mem, index = output.index), "global")
    output = output.drop(columns = "app_mb")
    
    assert len(output) == n, "join changed the number of rows"
    return output 
    
    
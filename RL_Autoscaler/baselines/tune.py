import itertools 
import json 
import numpy as np 
import pandas as pd 
from baselines.fixed import KeepAlive
from baselines.hpa import HPA 
from baselines.hybrid import HybridHistogram
from baselines.seasonal import SeasonalPredictive 
from sim.evaluate import evaluate_cell
from sim.metrics import reward_at, summarize, to_per_function

BUDGET = 20   #every method tries exactly 20 settings
RHOS = (3, 10, 30)   #prices of one cold start, in idle-instance-minutes 

#Keep-Alive lengths in minutes, The first 2 are the AWZ and AZURE defaults
FIXED_KS = [10, 20, 1, 2, 3, 4, 5, 7, 12, 15, 25, 30, 40, 60, 90, 120, 180, 240, 480, 1440]

DEFAULTS = {
    "hpa": {"target_pct": 70, "tolerance_pct": 10, "down_window": 5, "min_replicas": 1},
    "hybrid": {"head_pct": 5, "tail_pct": 99, "cv_threshold": 2.0, "margin_pct": 10, "min_intervals": 10},
    "seasonal": {"half_window": 15, "margin_pct": 120},
}

#The full space each method may search, the other 19 settings are drawn from here at random
GRIDS = {
    "hpa": {"target_pct": [40, 50, 60, 70, 80, 90, 100, 120, 150, 200], "tolerance_pct": [10],
            "down_window": [5, 15], "min_replicas": [0, 1]},
    "hybrid": {"head_pct": [1, 5, 10, 20], "tail_pct": [90, 95, 99], "cv_threshold": [0.5, 1.0, 2.0, 4.0],
               "margin_pct": [0, 10, 25], "min_intervals": [5, 10, 20]},
    "seasonal": {"half_window": [0, 2, 5, 10, 15, 30, 60], "margin_pct": [100, 105, 110, 120, 150]},
}

def make_configs(seed = 0, budget = BUDGET):
    """ 
        The list of settings to try: the same number for every method always including its default
    """
    
    rng = np.random.default_rng(seed)
    configs = [{"method": "fixed", "params": {"k": k}} for k in FIXED_KS[:budget]]
    for method, grid in GRIDS.items():
        everything = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
        others = [params for params in everything if params != DEFAULTS[method]]
        picked = sorted(rng.choice(len(others), size = budget - 1, replace = False))
        configs.append({"method": method, "params": DEFAULTS[method]})
        configs.extend({"method": method, "params": others[i]} for i in picked)
        
    return configs 

def build_policy(method, params, table, demand):
    if method == "fixed":
        return KeepAlive(params["k"], table, demand)

    if method == "hpa":
        return HPA(table, demand, **params)
    
    if method == "hybrid":
        return HybridHistogram(table, demand, settings = params)
    
    if method == "seasonal":
        return SeasonalPredictive(table, demand, **params)
    
    raise ValueError(f"Unknown method: {method}")


def run_sweep(configs, table, demand, counts, splits, constants, sim_config, cells = ("train", "val"), batch_size = 2048, progress = None):
    """ 
        Function to evaluate every setting on the chosen cells
        One Row Per (Setting, Cell) with the reward for every rho   
    """
    
    rows = []
    for number, config in enumerate(configs, start = 1):
        policy = build_policy(config["method"], config["params"], table, demand)
        for cell in cells:
            episodes = evaluate_cell(policy, cell, table, demand, counts, splits, constants, sim_config, batch_size)
            per_function = to_per_function(episodes)
            row = {"method": config["method"], "params": json.dumps(config["params"], sort_keys = True), "cell": cell, **summarize(per_function)}
            for rho in RHOS:
                row[f"reward_rho{rho}"] = reward_at(per_function, rho, sim_config["kappa"])
            
            rows.append(row)
        
        if progress:
            progress(f"{number} / {len(configs)} {config['method']} {config["params"]}")
        
    return pd.DataFrame(rows)

def choose_best(sweep, rho, select_cell = "val"):
    """ 
        Function to choose the BEST policy 
    """
    scores = sweep[sweep["cell"] == select_cell]  
    return scores.sort_values(f"reward_rho{rho}", ascending = False).groupby("method").head(1)

def normalise_waste(sweep):
    """ 
        Function which adds waste as a multiple of the 10-minute keep-alive scheme in the same cell
    """
    
    base = sweep[(sweep["method"] == "fixed") & (sweep["params"] == json.dumps({"k":10}, sort_keys = True))]
    base = base.set_index("cell")["waste_gbmin_per_day"]
    sweep = sweep.copy()
    sweep["waste_vs_k10"] = sweep["waste_gbmin_per_day"] / sweep["cell"].map(base) 
    return sweep 


def gate_s1(sweep, cell = "val", waste_tolerance = 1.1):
    """ 
        Gate S1: Does the hybrid histogram have fewer cold starts than the 10-minute keep-alive at the comparable memory
    """
    rows = sweep[sweep["cell"] == cell]
    fixed10 = rows[(rows["method"] == "fixed") & (rows["params"] == json.dumps({"k": 10}, sort_keys = True))].iloc[0]
    hybrids = rows[rows["method"] == "hybrid"]
    allowed = hybrids[hybrids["waste_gbmin_per_day"] <= waste_tolerance * fixed10["waste_gbmin_per_day"]]
    best = allowed.sort_values("cold_pct_mean").iloc[0] if len(allowed) else None 
    passed = best is not None and best["cold_pct_mean"] < fixed10["cold_pct_mean"] 
    return {
        "passed": bool(passed),
        "fixed10_cold_pct_mean": float(fixed10["cold_pct_mean"]),
        "fixed10_waste": float(fixed10["waste_gbmin_per_day"]),
        "best_hybrid_cold_pct_mean": None if best is None else float(best["cold_pct_mean"]),
        "best_hybrid_waste": None if best is None else float(best["waste_gbmin_per_day"]),
        "best_hybrid_params": None if best is None else best["params"],
    }
    
    

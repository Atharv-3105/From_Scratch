import numpy as np 
import pandas as pd 
from sim.batch import build_batch
from sim.metrics import churn_and_reversals, to_per_function
from sim.vec_env import VecWarmPoolEnv
from traces.splits import cell_mask 


def evaluate_policy(policy, table, demand, counts, rows, constants, sim_config, batch_size = 512):
    """ 
        Run a policy on the chosen function-days and return one row of results per function-day
        
        A policy is any callable: policy(env) -> pool size for every episode
    
    """
    
    rows = np.asarray(rows)
    env = VecWarmPoolEnv(sim_config)
    results = []
    
    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin: begin + batch_size]
        batch = build_batch(table, demand, chunk, constants, window = sim_config["reference_window"])
        env.reset(**batch)
        
        #If the policy has a start-function use it 
        if hasattr(policy, "start"):
            policy.start(env, chunk)
            
        minutes = batch["demand"].shape[1]
        pool = np.zeros((len(chunk), minutes), dtype = np.int64)
        cold = np.zeros((len(chunk), minutes), dtype = np.int64)
        for t in range(minutes):
            _, _, _, info = env.step_pool(policy(env))
            
            pool[:, t], cold[:, t] = info["pool"], info["cold"]
            
        summary = env.summary()
        invocations = np.asarray(counts[chunk], dtype = np.int64)
        churn, reversals = churn_and_reversals(pool, initial_pool = 0)
        part = table.iloc[chunk][["func_id", "day", "group", "split", "weight"]].reset_index(drop = True)
        part["invocations"] = invocations.sum(axis = 1)
        
        #A cold start delays at least one invocation, but never more than that minute's invocations
        part["cold_inv"] = np.minimum(cold, invocations).sum(axis = 1)
        part["idle_gbmin"] = summary["idle_gbmin"]
        part["served"] = (batch["demand"] - cold).sum(axis = 1)  #Demand which the warm pool really served
        part["pool_minutes"] = pool.sum(axis = 1)
        part["churn"] = churn 
        part["reversals"] = reversals 
        part["minutes"] = minutes 
        part["reward"] = summary["reward"]
        results.append(part)
        
    return pd.concate(results, ignore_index = True)

def evaluate_cell(policy, cell_name, table, demand, counts, splits_config, constants, sim_config, batch_size = 512):
    """ 
        Function to evaluate on one-named cell(train, val), test_days, ..... Returns one row per
        function-DAY;
        pass it through to_per_function() before summarize() and bootstrap_ci()
    """
    rows = np.nonzero(cell_mask(table, cell_name, splits_config).to_numpy())[0]
    return evaluate_cell(policy, table, demand, counts, rows, constants, sim_config, batch_size)

import time 
from pathlib import Path 
import numpy as np 

from sim.batch import build_batch, function_constants
from sim.env import WarmPoolEnv
from sim.vec_env import VecWarmPoolEnv
from traces.demand import subset_demand
from traces.splits import cell_mask
from utils.config import load_config

PROCESSED_DIR = Path("data/processed")
LOG_FILE = Path("reports/logs/env_throughput.txt")

def main(pro_dir = PROCESSED_DIR, log_file = LOG_FILE, n_episodes = 256, n_scalar = 20, seed = 0):
    sim_config = load_config("configs/sim.yaml")
    splits = load_config("configs/splits.yaml")
    table, demand = subset_demand(pro_dir)
    constants = function_constants(table, demand, splits["days"]["train"])
    
    rng = np.random.default_rng(seed)
    train_rows = np.nonzero(cell_mask(table, "train", splits).to_numpy())[0]
    rows = rng.choice(train_rows, n_episodes, replace = False)
    batch = build_batch(table, demand, rows, constants)
    minutes = batch["demand"].shape[1]
    actions = rng.integers(0, len(sim_config["actions"]), size = (minutes, n_episodes))
    
    #Get the Vectorized Environment throughput metrics
    vec = VecWarmPoolEnv(sim_config)
    start = time.perf_counter()
    vec.reset(**batch)
    for t in range(minutes):
        vec.step(actions[t])
    vec_seconds = time.perf_counter() - start 
    
    #Get the Simple Environment throughput metrics
    scalar = WarmPoolEnv(sim_config)
    start = time.perf_counter()
    for i in range(n_scalar):
        scalar.reset(batch["demand"][i], scale = batch["scale"][i], max_pool = int(batch["max_pool"][i]), mem_mb = batch["mem_mb"][i], previous_demand = batch["previous_demand"][i])
        for t in range(minutes):
            scalar.step(int(actions[t, i]))
    
    scalar_seconds = time.perf_counter() - start 
    
    vec_rate = n_episodes * minutes / vec_seconds
    scalar_rate = n_scalar * minutes / scalar_seconds 
    
    lines = [
        f"vectorized: {n_episodes} episodes x {minutes} steps in {vec_seconds:.2f}s = {vec_rate:,.0f} env-steps/s",
        f"scalar:     {n_scalar} episodes x {minutes} steps in {scalar_seconds:.2f}s = {scalar_rate:,.0f} env-steps/s",
        f"speed-up: {vec_rate / scalar_rate:.0f}x",
        f"one PPO iteration of {n_episodes} function-days: {vec_seconds:.2f}s of pure environment time",
    ]
    Path(log_file).parent.mkdir(parents = True, exist_ok = True)
    Path(log_file).write_text("\n".join(lines) + "\n", encoding = "utf-8")
    print("\n".join(lines))
    
if __name__ == "__main__":
    main()
    

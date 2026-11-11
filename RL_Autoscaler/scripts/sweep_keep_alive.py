from pathlib import Path 
import numpy as np 
import pandas as pd 
from baselines.fixed import KeepAlive
from sim.batch import function_constants
from sim.evaluate import evaluate_cell
from sim.metrics import save_table, summarize, to_per_function
from traces.demand import subset_demand
from utils.config import load_config

PROCESSED_DIR = Path("data/processed")
OUT_FILE = Path("reports/tables/keep_alive_sweep.csv")
KEEP_ALIVE_MINUTES = [1, 5, 10, 20, 60, None]  #None = never unload

def main(processed_dir = PROCESSED_DIR, out_file = OUT_FILE, cells = ("train", "val"), ks = KEEP_ALIVE_MINUTES):
    sim_config = load_config("configs/sim.yaml")
    splits = load_config("configs/splits.yaml")
    table, demand = subset_demand(processed_dir)
    counts = np.load(Path(processed_dir) / "subset_minutes.npy", mmap_mode = "r")
    constants = function_constants(table, demand, splits["days"]["train"])
    
    rows = []
    for cell in cells:
        for k in ks:
            episodes = evaluate_cell(KeepAlive(k, table, demand), cell, table, demand, counts, splits, constants, sim_config)
            result = summarize(to_per_function(episodes))
            rows.append({"cell": cell, "keep_alive_min": "never_unload" if k is None else k, **result})
    
    sweep = pd.DataFrame(rows)
    
    
    # Waste as a multiple of the 10-minute keep-alive in the same cell (the paper's way of plotting it)
    base = sweep[sweep["keep_alive_min"] == 10].set_index("cell")["waste_gbmin_per_day"]
    sweep["waste_vs_k10"] = sweep["waste_gbmin_per_day"] / sweep["cell"].map(base)

    save_table(sweep.set_index(["cell", "keep_alive_min"]), out_file)
    print(sweep.round(3).to_string(index=False))
    return sweep


if __name__ == "__main__":
    main()
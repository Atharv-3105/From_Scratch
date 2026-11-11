import json 
import numpy as np 
import pandas as pd 
from baselines.fixed import oracle
from baselines.tune import RHOS, build_policy, choose_best, gate_s1, make_configs, normalise_waste, run_sweep
from sim.batch import function_constants
from sim.evaluate import evaluate_cell
from sim.metrics import save_table, summarize_by_group, to_per_function
from traces.demand import subset_demand
from utils.config import load_config
from vis.baselines import plot_pareto
from pathlib import Path 

PROCESSED_DIR = Path("data/processed")
TABLE_DIR = Path("reports/tables")
FIG_DIR = Path("reports/figures")

def main(processed_dir = PROCESSED_DIR, table_dir = TABLE_DIR, fig_dir = FIG_DIR, seed = 0, n_boot = 200):
    table_dir, fig_dir = Path(table_dir), Path(fig_dir)
    fig_dir.mkdir(parents = True, exist_ok = True)
    sim_config = load_config("configs/sim.yaml")
    splits = load_config("configs/splits.yaml")
    table, demand = subset_demand(processed_dir)
    counts = np.load(Path(processed_dir) / "subset_minutes.npy", mmap_mode = "r")
    constants = function_constants(table, demand, splits["days"]["train"])
    
    configs = make_configs(seed = seed)
    sweep = run_sweep(configs, table, demand, counts, splits, constants, sim_config, progress = print)
    sweep = normalise_waste(sweep)
    save_table(sweep.set_index(["method", "params", "cell"]), table_dir / "baseline_sweep.csv")
    
    chosen = pd.concat([choose_best(sweep, rho).assign(rho = rho) for rho in RHOS])
    save_table(chosen.set_index(["rho", "method"]), table_dir / "baseline_chosen.csv")
    (table_dir / "baseline_chosen.json").write_text(json.dumps({f"rho={r}": {row["method"]: json.loads(row["params"]) for _, row in chosen[chosen["rho"] == r].iterrows()}for r in RHOS}, indent = 2), encoding = "utf-8")
    print("\nChosen on Validation (reward per function-day at each RHO):")
    
    for rho in RHOS:
        part = chosen[chosen["rho"] == rho]
        print(f"rho = {rho}\n" + part[["method", "params", f"reward_rho{rho}", "cold_pct_mean", "cold_pct_p75", "waste_gbmin_per_day"]].round(2).to_string(index = False))
        
    
    gate = gate_s1(sweep)
    print(f"\n Gate S1(hybrid fewer cold starts than fixed 10 at comparable memory):", json.dumps(gate, indent = 2))
    
    parts = {}
    for _, row in chosen[chosen["rho"] == 10].iterrows():
        policy = build_policy(row["method"], json.loads(row["params"]), table, demand)
        parts[row["method"]] = summarize_by_group(to_per_function(evaluate_cell(policy, "val", table, demand, counts, splits, constants, sim_config, 2048)), n_boot = n_boot)
    parts["oracle (floor)"] = summarize_by_group(to_per_function(evaluate_cell(oracle, "val", table, demand, counts, splits, constants, sim_config, 2048)), n_boot = n_boot)
    per_group = pd.concat(parts, names = ["method", "group", "metric"])
    save_table(per_group, table_dir / "baseline_per_group_val.csv")
    
    
    plot_pareto(sweep, "val", fig_dir  / "pareto_baselines_val.png", chosen = chosen[chosen["rho"] == 10])
    print("\nSaved the sweep, chosen settings, per-group table and figure")
    return sweep, chosen, gate 



if __name__ == "__main__":
    main()
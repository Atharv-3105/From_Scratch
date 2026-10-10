from pathlib import Path
import numpy as np
import pandas as pd
from traces.splits import assign_splits, check_or_freeze, group_shares, validate_day_split, validate_unit_split
from traces.stats import N_MIN, add_function_id, compute_function_stats, label_functions, sample_by_group
from utils.config import load_config
from vis.eda import plot_eda, paper_facts

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")
CONFIG_FILE = Path("configs/classes.yaml")
SPLITS_FILE = Path("configs/splits.yaml")
FINGERPRINT_FILE = Path("configs/splits_fingerprint.txt")


def main(processed_dir=PROCESSED_DIR, fig_dir=FIG_DIR, table_dir=TABLE_DIR,
         config_file=CONFIG_FILE, splits_file=SPLITS_FILE, fingerprint_file=FINGERPRINT_FILE):
    
    processed_dir, fig_dir, table_dir = Path(processed_dir), Path(fig_dir), Path(table_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_file)
    splits = load_config(splits_file)
    day_table = pd.read_parquet(processed_dir / "function_day.parquet")
    # memmap = the matrix stays on disk and rows are read only when we ask for them
    minutes = np.memmap(processed_dir / "inv_minutes.bin", dtype=np.uint32, mode="r").reshape(-1, N_MIN)
    assert len(day_table) == minutes.shape[0], "table and minute matrix have different row counts"

    # Step 1: one row of statistics per function (train days only), then flags and groups
    print("Computing per-function statistics (train days only)...")
    train_days = config["train_days"]
    stats, minute_of_day_totals = compute_function_stats(day_table, minutes, train_days["first"], train_days["last"])
    stats = label_functions(stats, config)

    # Step 2: split by owner (or app), check the rules, and freeze the result
    validate_day_split(splits)
    stats = assign_splits(stats, splits)
    validate_unit_split(stats, splits)
    print(f"Split: {check_or_freeze(stats, fingerprint_file)} (fingerprint file: {fingerprint_file})")
    stats.to_parquet(processed_dir / "function_stats.parquet")
    print("\nFunctions per split:\n", stats["split"].value_counts().to_string())
    print("\nShare of each group inside each split (labelled functions):\n", group_shares(stats).round(3).to_string())

    # Step 3: draw the sample SEPARATELY inside each split, so every sampled function has a known split
    sample_parts = []
    for name, share in splits["fractions"].items():
        members = stats[stats["split"] == name]
        sample_parts.append(sample_by_group(
            members,
            size=round(config["subset"]["size"] * share),
            min_per_group=round(config["subset"]["min_per_group"] * share),
            seed=config["subset"]["seed"],
        ))
    sample = pd.concat(sample_parts).sort_index()

    # Step 4: collect ALL 14 days of the sampled functions (the cells pick the days later)
    day_table = add_function_id(day_table)
    day_table["source_row"] = np.arange(len(day_table))  # position in the big matrix, to copy the right rows
    sample_days = day_table.merge(sample[["func_id", "group", "weight", "split"]], on="func_id", how="inner")
    np.save(processed_dir / "subset_minutes.npy", np.asarray(minutes[sample_days["source_row"].to_numpy()]))
    sample_days.to_parquet(processed_dir / "subset_function_days.parquet")
    print(f"\nSubset: {len(sample):,} functions, {len(sample_days):,} function-days")
    print(sample.groupby(["split", "group"]).agg(functions=("weight", "size"), weight=("weight", "first")).round(1).to_string())

    # Step 5: the EDA figure and the comparison with the paper
    plot_eda(stats, minute_of_day_totals, config, fig_dir / "v9_eda.png")
    facts = paper_facts(stats, config)
    (table_dir / "paper_facts.txt").write_text(facts, encoding="utf-8")
    print("\n" + facts)


if __name__ == "__main__":
    main()
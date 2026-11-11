import json

import numpy as np
import pandas as pd
import pytest

from baselines.tune import (
    BUDGET, RHOS, make_configs, build_policy, run_sweep, choose_best, gate_s1, normalise_waste,
)
from sim.evaluate import evaluate_policy
from sim.metrics import reward_at, summarize, to_per_function

SIM_CONFIG = {
    "reference_window": 5, "rho": 10, "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}
MINUTES = 1440


def small_world(rows=6, seed=0):
    rng = np.random.default_rng(seed)
    demand = ((rng.random((rows, MINUTES)) < 0.03) * rng.integers(1, 4, size=(rows, MINUTES))).astype(np.int64)
    table = pd.DataFrame({"func_id": np.repeat(np.arange(rows // 2), 2), "day": np.tile([1, 2], rows // 2),
                          "group": "rare", "split": ["train", "train", "val", "val", "val", "val"][:rows],
                          "weight": 1.0, "mem_mb": 128.0})
    constants = pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=range(rows // 2))
    return table, demand, demand.astype(np.uint32), constants


# ---- reward at any rho from one evaluation ----
def test_reward_at_matches_the_reward_the_environment_computed():
    from baselines.fixed import KeepAlive
    table, demand, counts, constants = small_world()
    episodes = evaluate_policy(KeepAlive(10, table, demand), table, demand, counts, list(range(6)), constants, SIM_CONFIG)
    per_function = to_per_function(episodes)
    assert reward_at(per_function, rho=10, kappa=0.1) == pytest.approx(summarize(per_function)["reward_per_day"])


def test_a_dearer_cold_start_lowers_the_reward():
    from baselines.fixed import KeepAlive
    table, demand, counts, constants = small_world()
    per_function = to_per_function(evaluate_policy(KeepAlive(1, table, demand), table, demand, counts, list(range(6)), constants, SIM_CONFIG))
    assert reward_at(per_function, 3, 0.1) > reward_at(per_function, 10, 0.1) > reward_at(per_function, 30, 0.1)


# ---- the equal budget ----
def test_every_method_gets_the_same_number_of_configurations():
    configs = make_configs(seed=0)
    counts = pd.Series([c["method"] for c in configs]).value_counts()
    assert set(counts) == {BUDGET} and set(counts.index) == {"fixed", "hpa", "hybrid", "seasonal"}


def test_the_default_setting_of_each_method_is_one_of_its_configurations():
    configs = make_configs(seed=0)
    by_method = {m: [c["params"] for c in configs if c["method"] == m] for m in ("fixed", "hpa", "hybrid", "seasonal")}
    assert {"k": 10} in by_method["fixed"] and {"k": 20} in by_method["fixed"]
    assert {"target_pct": 70, "tolerance_pct": 10, "down_window": 5, "min_replicas": 1} in by_method["hpa"]
    assert {"half_window": 15, "margin_pct": 120} in by_method["seasonal"]
    assert any(p.get("head_pct") == 5 and p.get("tail_pct") == 99 and p.get("cv_threshold") == 2.0 for p in by_method["hybrid"])


def test_configurations_are_distinct_and_reproducible():
    a, b = make_configs(seed=0), make_configs(seed=0)
    assert a == b
    assert a != make_configs(seed=1)
    keys = [(c["method"], json.dumps(c["params"], sort_keys=True)) for c in a]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("method", ["fixed", "hpa", "hybrid", "seasonal"])
def test_every_configuration_builds_a_working_policy(method):
    table, demand, counts, constants = small_world()
    config = [c for c in make_configs(seed=0) if c["method"] == method][3]
    policy = build_policy(config["method"], config["params"], table, demand)
    result = evaluate_policy(policy, table, demand, counts, list(range(6)), constants, SIM_CONFIG)
    assert len(result) == 6 and np.isfinite(result["reward"]).all()


# ---- sweep, selection, gate ----
def test_run_sweep_has_a_row_per_config_and_cell_with_rewards_for_every_rho():
    table, demand, counts, constants = small_world()
    splits = {"days": {"train": [1], "val": [2], "test": [3]},
              "cells": {"train": {"functions": "train", "days": "train"}, "val": {"functions": "val", "days": "val"}}}
    table["split"] = ["train", "train", "val", "val", "val", "val"]
    table["day"] = [1, 2, 1, 2, 1, 2]
    configs = [c for c in make_configs(seed=0) if c["method"] == "fixed"][:3]
    sweep = run_sweep(configs, table, demand, counts, splits, constants, SIM_CONFIG, cells=("val",), batch_size=4)
    assert len(sweep) == 3 and set(sweep["cell"]) == {"val"}
    assert all(f"reward_rho{r}" in sweep for r in RHOS)


def sweep_frame(rows):
    return pd.DataFrame(rows, columns=["method", "params", "cell", "cold_pct_mean", "waste_gbmin_per_day", "reward_rho10"])


def test_selection_looks_only_at_the_validation_cell():
    sweep = sweep_frame([
        ("fixed", '{"k": 5}', "train", 50, 10, -100.0),     # best on train ...
        ("fixed", '{"k": 20}', "train", 40, 30, -300.0),
        ("fixed", '{"k": 5}', "val", 50, 10, -500.0),       # ... but worse on validation
        ("fixed", '{"k": 20}', "val", 40, 30, -200.0),
    ])
    best = choose_best(sweep, rho=10)
    assert best.iloc[0]["params"] == '{"k": 20}' and best.iloc[0]["cell"] == "val"


def test_gate_s1_passes_when_hybrid_has_fewer_cold_starts_at_similar_waste():
    sweep = sweep_frame([
        ("fixed", '{"k": 10}', "val", 40.0, 100.0, -1.0),
        ("hybrid", '{"head_pct": 5}', "val", 20.0, 105.0, -1.0),
    ])
    assert gate_s1(sweep)["passed"] is True


def test_gate_s1_fails_if_the_only_better_hybrid_wastes_much_more():
    sweep = sweep_frame([
        ("fixed", '{"k": 10}', "val", 40.0, 100.0, -1.0),
        ("hybrid", '{"head_pct": 5}', "val", 20.0, 300.0, -1.0),
        ("hybrid", '{"head_pct": 10}', "val", 45.0, 90.0, -1.0),
    ])
    assert gate_s1(sweep)["passed"] is False


def test_waste_is_normalised_to_the_ten_minute_keep_alive():
    sweep = sweep_frame([
        ("fixed", '{"k": 10}', "val", 40.0, 100.0, -1.0),
        ("fixed", '{"k": 60}', "val", 30.0, 250.0, -1.0),
        ("fixed", '{"k": 10}', "train", 45.0, 80.0, -1.0),
        ("fixed", '{"k": 60}', "train", 35.0, 200.0, -1.0),
    ])
    out = normalise_waste(sweep)
    assert out.loc[out["params"] == '{"k": 60}', "waste_vs_k10"].tolist() == [2.5, 2.5]
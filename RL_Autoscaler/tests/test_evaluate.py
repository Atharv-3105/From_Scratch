import numpy as np
import pandas as pd
import pytest

from sim.evaluate import evaluate_policy, evaluate_cell
from sim.metrics import to_per_function, summarize

SIM_CONFIG = {
    "reference_window": 5,
    "rho": 10,
    "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}
MINUTES = 1440


# ---- policies written the plain way: they get the environment and return a pool size per episode ----
def oracle(env):
    return env.demand[:, env.minute]                       # cheats: reads the minute it is deciding for


def never_scale(env):
    return np.zeros(env.n_episodes, dtype=np.int64)


class NeverUnload:
    """Keep as many instances as the busiest minute seen so far."""
    def start(self, env, rows):
        self.peak = np.zeros(env.n_episodes, dtype=np.int64)

    def __call__(self, env):
        if env.minute > 0:
            self.peak = np.maximum(self.peak, env.demand[:, env.minute - 1])
        return self.peak.copy()


def make_world():
    """One function-day: 2 invocations at minutes 10 and 11, 4 at minute 20. 60 s per call, so demand = invocations."""
    counts = np.zeros((1, MINUTES), dtype=np.uint32)
    counts[0, 10], counts[0, 11], counts[0, 20] = 2, 2, 4
    table = pd.DataFrame({"func_id": [0], "day": [1], "group": ["rare"], "split": ["train"],
                          "weight": [7.0], "mem_mb": [1024.0]})
    constants = pd.DataFrame({"max_pool": [100], "scale": [1.0]}, index=[0])
    return table, counts.astype(np.int64), counts, constants


def run(policy):
    table, demand, counts, constants = make_world()
    return evaluate_policy(policy, table, demand, counts, [0], constants, SIM_CONFIG).iloc[0]


# ---- the extremes ----
def test_oracle_has_no_cold_starts_and_no_waste():
    row = run(oracle)
    assert row["cold_inv"] == 0 and row["idle_gbmin"] == 0
    assert row["served"] == 8 and row["pool_minutes"] == 8
    assert row["reward"] == pytest.approx(-0.6)               # only the start-up cost: 2 + 4 instances started x 0.1


def test_never_scaling_makes_every_invocation_cold():
    row = run(never_scale)
    assert row["invocations"] == 8 and row["cold_inv"] == 8
    assert row["idle_gbmin"] == 0 and row["pool_minutes"] == 0
    assert row["reward"] == pytest.approx(-80.0)              # 10 * 8 cold


def test_never_unloading_by_hand():
    row = run(NeverUnload())
    # pool is 0 until minute 10 (cold 2), then 2 (cold 2 more at minute 20), then 4 for the rest of the day
    assert row["cold_inv"] == 4
    assert row["idle_gbmin"] == pytest.approx(16 + 4 * 1419)   # 8 minutes x 2 idle, then 1419 minutes x 4 idle
    assert row["served"] == 4 and row["pool_minutes"] == 5696
    assert row["churn"] == 4 and row["reversals"] == 0
    assert row["reward"] == pytest.approx(-(10 * 4 + 5692 + 0.1 * 4))


def test_oracle_beats_never_unload_beats_never_scale_on_cold_pct_and_waste_is_the_other_way():
    cold = {name: run(p)["cold_inv"] for name, p in [("oracle", oracle), ("keep", NeverUnload()), ("none", never_scale)]}
    assert cold["oracle"] < cold["keep"] < cold["none"]
    waste = {name: run(p)["idle_gbmin"] for name, p in [("oracle", oracle), ("keep", NeverUnload()), ("none", never_scale)]}
    assert waste["none"] < waste["oracle"] + 1 and waste["oracle"] < waste["keep"]


# ---- the harness itself ----
def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(0)
    counts = (rng.random((7, MINUTES)) < 0.05).astype(np.uint32) * rng.integers(1, 5, size=(7, MINUTES)).astype(np.uint32)
    table = pd.DataFrame({"func_id": range(7), "day": 2, "group": "rare", "split": "train", "weight": 1.0, "mem_mb": 100.0})
    constants = pd.DataFrame({"max_pool": 50, "scale": 1.0}, index=range(7))
    demand = counts.astype(np.int64)
    small = evaluate_policy(NeverUnload(), table, demand, counts, list(range(7)), constants, SIM_CONFIG, batch_size=2)
    large = evaluate_policy(NeverUnload(), table, demand, counts, list(range(7)), constants, SIM_CONFIG, batch_size=100)
    pd.testing.assert_frame_equal(small, large)


def test_a_policy_must_return_whole_numbers():
    with pytest.raises(TypeError):
        run(lambda env: np.zeros(env.n_episodes) + 0.5)


def test_evaluate_cell_picks_the_rows_of_that_cell():
    splits = {"days": {"train": [1], "val": [2], "test": [3]},
              "cells": {"train": {"functions": "train", "days": "train"}, "val": {"functions": "val", "days": "val"}}}
    table = pd.DataFrame({"func_id": [0, 1, 2], "day": [1, 1, 2], "split": ["train", "val", "val"],
                          "group": "rare", "weight": 1.0, "mem_mb": 100.0})
    counts = np.zeros((3, MINUTES), dtype=np.uint32)
    counts[:, 0] = 1
    constants = pd.DataFrame({"max_pool": 10, "scale": 1.0}, index=[0, 1, 2])
    result = evaluate_cell(never_scale, "val", table, counts.astype(np.int64), counts, splits, constants, SIM_CONFIG)
    assert result["func_id"].tolist() == [2]                  # only the val function on the val day
    assert result.loc[0, "invocations"] == 1


def test_summary_of_the_oracle_run():
    table, demand, counts, constants = make_world()
    episodes = evaluate_policy(oracle, table, demand, counts, [0], constants, SIM_CONFIG)
    result = summarize(to_per_function(episodes))
    assert result["cold_pct_pooled"] == 0 and result["waste_gbmin_per_day"] == 0 and result["utilisation"] == 1.0
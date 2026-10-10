import numpy as np
import pandas as pd
import pytest

from sim.batch import function_constants, fill_silent_days, build_batch
from sim.env import pool_cap, reward_scale

MINUTES = 1440


def make_table(rows):
    """rows: (func_id, day). Other columns are filled with simple values."""
    table = pd.DataFrame(rows, columns=["func_id", "day"])
    table["group"] = "rare"
    table["split"] = "train"
    table["weight"] = 5.0
    table["mem_mb"] = 128.0
    table["dur_ms"] = 100.0
    table["total_invocations"] = 1
    return table


def day_with(*values_at):
    # (minute, value) pairs -> one day of demand
    day = np.zeros(MINUTES, dtype=np.int32)
    for minute, value in values_at:
        day[minute] = value
    return day


# ---- per-function constants come from TRAIN days only ----
def test_constants_match_the_single_function_rules_and_ignore_other_days():
    table = make_table([(0, 1), (0, 2), (0, 11), (1, 1)])
    demand = np.vstack([
        day_with((10, 4), (20, 2)),          # f0 day 1
        day_with((10, 6)),                   # f0 day 2
        day_with((10, 9999)),                # f0 day 11 -> NOT a train day, must be ignored
        np.zeros(MINUTES, dtype=np.int32),   # f1 day 1: nothing at all
    ])
    constants = function_constants(table, demand, train_days=range(1, 11))
    train_values = np.concatenate([demand[0], demand[1]])
    assert constants.loc[0, "max_pool"] == pool_cap(train_values) == 12
    assert constants.loc[0, "scale"] == pytest.approx(reward_scale(train_values))   # (4+2+6)/3 = 4
    assert constants.loc[1, "max_pool"] == 1 and constants.loc[1, "scale"] == 1.0


# ---- silent days ----
def silent_fixture():
    table = make_table([(0, 1), (0, 3), (0, 6), (1, 14)])
    demand = np.vstack([day_with((0, 1))] * 4)
    return table, demand


def test_inside_mode_fills_gaps_between_first_and_last_day():
    table, demand = silent_fixture()
    new_table, new_demand = fill_silent_days(table, demand, mode="inside")
    added = new_table[new_table["silent"]]
    assert sorted(added[added.func_id == 0]["day"]) == [2, 4, 5]
    assert (added.func_id == 1).sum() == 0                       # seen on one day only: nothing to fill
    assert len(new_table) == len(new_demand) == 4 + 3


def test_to_end_mode_also_fills_the_days_after_the_last_appearance():
    table, demand = silent_fixture()
    new_table, _ = fill_silent_days(table, demand, mode="to_end")
    added = new_table[new_table["silent"]]
    assert sorted(added[added.func_id == 0]["day"]) == [2, 4, 5] + list(range(7, 15))


def test_none_mode_changes_nothing():
    table, demand = silent_fixture()
    new_table, new_demand = fill_silent_days(table, demand, mode="none")
    assert len(new_table) == 4 and not new_table["silent"].any()
    np.testing.assert_array_equal(new_demand, demand)


def test_silent_rows_have_zero_demand_and_keep_the_function_settings():
    table, demand = silent_fixture()
    table.loc[table.func_id == 0, "group"] = "hot"
    new_table, new_demand = fill_silent_days(table, demand, mode="inside")
    silent = new_table["silent"].to_numpy()
    assert (new_demand[silent] == 0).all()
    assert (new_table[silent & (new_table.func_id == 0).to_numpy()]["group"] == "hot").all()
    assert (new_table[silent]["total_invocations"] == 0).all()
    np.testing.assert_array_equal(new_demand[:4], demand)         # the original rows are untouched, in order


# ---- turning table rows into environment input ----
def test_build_batch_uses_the_previous_days_last_minutes():
    table = make_table([(0, 1), (0, 2), (1, 5)])
    demand = np.vstack([
        day_with((1435, 7), (1439, 3)),      # f0 day 1: ends with ... 7 . . . 3
        day_with((0, 1)),                    # f0 day 2
        day_with((0, 2)),                    # f1 day 5: no day 4 exists
    ])
    constants = function_constants(table, demand, train_days=range(1, 11))
    batch = build_batch(table, demand, [1, 2], constants)
    assert batch["previous_demand"][0].tolist() == [7, 0, 0, 0, 3]      # last 5 minutes of day 1
    assert batch["previous_demand"][1].tolist() == [0, 0, 0, 0, 0]      # no previous day -> zeros
    assert batch["demand"].shape == (2, MINUTES)
    assert batch["mem_mb"].tolist() == [128.0, 128.0]
    assert batch["max_pool"].tolist() == [constants.loc[0, "max_pool"], constants.loc[1, "max_pool"]]
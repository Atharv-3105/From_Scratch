import numpy as np
import pandas as pd
import pytest

from sim.metrics import (
    weighted_percentile, churn_and_reversals, summarize, bootstrap_ci,
    summarize_by_group, pareto_front, to_per_function,
)


def per_function(rows):
    """rows: (weight, group, invocations, cold_inv). Everything else is filled with simple values."""
    table = pd.DataFrame(rows, columns=["weight", "group", "invocations", "cold_inv"])
    table["n_days"] = 1
    table["idle_gbmin"] = 10.0
    table["served"] = 50.0
    table["pool_minutes"] = 100.0
    table["churn"] = 4.0
    table["reversals"] = 2
    table["minutes"] = 1440
    table["reward"] = -5.0
    return table


# ---- weighted percentile ----
def test_weighted_percentile():
    values, weights = [100.0, 0.0], [1.0, 3.0]              # 75% of the weight sits on the value 0
    assert weighted_percentile(values, weights, 0.75) == 0.0
    assert weighted_percentile(values, weights, 0.80) == 100.0
    assert weighted_percentile([5.0], [2.0], 0.75) == 5.0


# ---- pooled ratios are not the same as the mean of per-function ratios ----
def test_pooled_and_per_function_cold_pct_differ():
    table = per_function([(1, "a", 100, 100),     # function A: 100% cold, weight 1
                          (3, "b", 300, 0)])      # function B: 0% cold, weight 3 and three times the traffic
    result = summarize(table)
    assert result["cold_pct_pooled"] == pytest.approx(100 * (1 * 100) / (1 * 100 + 3 * 300))   # 10: busy functions dominate
    assert result["cold_pct_mean"] == pytest.approx((1 * 100 + 3 * 0) / 4)                     # 25: every function counts by weight
    assert result["cold_pct_p75"] == 0.0


def test_functions_without_traffic_are_left_out_of_per_function_cold_pct():
    table = per_function([(1, "a", 100, 50), (1, "b", 0, 0)])
    result = summarize(table)
    assert result["cold_pct_mean"] == pytest.approx(50.0)
    assert result["cold_pct_pooled"] == pytest.approx(50.0)


def test_other_metrics_are_weighted_ratios_of_sums():
    table = per_function([(1, "a", 100, 0), (3, "b", 100, 0)])
    table.loc[0, "idle_gbmin"] = 40.0                  # function a wastes 40 GB-min, b wastes 10
    result = summarize(table)
    assert result["waste_gbmin_per_day"] == pytest.approx((1 * 40 + 3 * 10) / 4)
    assert result["utilisation"] == pytest.approx(0.5)             # served 50 of 100 pool-minutes
    assert result["churn_per_day"] == pytest.approx(4.0)
    assert result["flaps_per_hour"] == pytest.approx(2 / 24)       # 2 reversals in 24 hours
    assert result["reward_per_day"] == pytest.approx(-5.0)


# ---- churn and flapping ----
def test_churn_and_reversals():
    pool = np.array([[0, 2, 2, 1, 3, 3, 0]])
    churn, reversals = churn_and_reversals(pool, initial_pool=0)
    assert churn.tolist() == [8]                   # 2 + 1 + 2 + 3
    assert reversals.tolist() == [3]               # up, down, up, down -> 3 changes of direction


def test_churn_counts_the_first_move_from_the_starting_pool():
    churn, reversals = churn_and_reversals(np.array([[3, 3, 3]]), initial_pool=0)
    assert churn.tolist() == [3] and reversals.tolist() == [0]
    churn, _ = churn_and_reversals(np.array([[3, 3, 3]]), initial_pool=3)
    assert churn.tolist() == [0]


# ---- bootstrap ----
def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    rows = [(1, "a" if i % 2 else "b", 100, int(rng.integers(0, 100))) for i in range(60)]
    table = per_function(rows)
    first = bootstrap_ci(table, n_boot=200, seed=1)
    second = bootstrap_ci(table, n_boot=200, seed=1)
    pd.testing.assert_frame_equal(first, second)
    row = first.loc["cold_pct_pooled"]
    assert row["low"] <= row["estimate"] <= row["high"] and row["low"] < row["high"]


def test_identical_functions_give_a_zero_width_interval():
    table = per_function([(1, "a", 100, 20)] * 10)
    row = bootstrap_ci(table, n_boot=100, seed=0).loc["cold_pct_pooled"]
    assert row["low"] == row["high"] == row["estimate"] == pytest.approx(20.0)


def test_more_functions_give_a_narrower_interval():
    def width(n):
        rng = np.random.default_rng(3)
        table = per_function([(1, "a", 100, int(rng.integers(0, 100))) for _ in range(n)])
        row = bootstrap_ci(table, n_boot=300, seed=0).loc["cold_pct_pooled"]
        return row["high"] - row["low"]
    assert width(400) < width(25)


def test_group_table_has_a_row_per_group_and_an_overall_row():
    table = per_function([(1, "a", 100, 0), (1, "b", 100, 100), (1, "b", 100, 100)])
    out = summarize_by_group(table, n_boot=50, seed=0)
    assert set(out.index.get_level_values(0)) == {"a", "b", "all"}
    assert out.loc[("a", "cold_pct_pooled"), "estimate"] == 0.0
    assert out.loc[("b", "cold_pct_pooled"), "estimate"] == 100.0


# ---- from function-days to functions ----
def test_to_per_function_adds_up_function_days():
    episodes = pd.DataFrame({
        "func_id": [1, 1, 2], "weight": [5.0, 5.0, 2.0], "group": ["a", "a", "b"],
        "invocations": [10, 20, 7], "cold_inv": [1, 2, 3], "idle_gbmin": [1.0, 2.0, 4.0], "served": [1, 1, 1],
        "pool_minutes": [5, 5, 5], "churn": [1, 1, 1], "reversals": [0, 1, 0], "minutes": [1440] * 3, "reward": [-1.0, -2.0, -3.0],
    })
    out = to_per_function(episodes).set_index("func_id")
    assert out.loc[1, "invocations"] == 30 and out.loc[1, "n_days"] == 2 and out.loc[1, "weight"] == 5.0
    assert out.loc[1, "minutes"] == 2880 and out.loc[2, "n_days"] == 1


# ---- the cost / quality trade-off ----
def test_pareto_front_keeps_only_points_nobody_beats():
    x = [1, 2, 3, 4, 2]
    y = [5, 3, 4, 1, 6]
    assert pareto_front(x, y).tolist() == [0, 1, 3]        # (3, 4) is beaten by (2, 3); (2, 6) by (2, 3)
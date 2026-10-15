import numpy as np
import pandas as pd
import pytest

from baselines.fixed import KeepAlive
from baselines.hybrid import HybridHistogram, interval_histogram, choose_windows, DEFAULTS
from sim.batch import build_batch
from sim.evaluate import evaluate_policy
from sim.vec_env import VecWarmPoolEnv

SIM_CONFIG = {
    "reference_window": 5, "rho": 10, "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}
DAY = 1440


def world(day_patterns):
    """day_patterns: list of lists of busy minutes (each value = demand 1), one list per day of one function."""
    demand = np.zeros((len(day_patterns), DAY), dtype=np.int64)
    for row, minutes in enumerate(day_patterns):
        demand[row, minutes] = 1
    table = pd.DataFrame({"func_id": 0, "day": np.arange(1, len(day_patterns) + 1), "group": "rare",
                          "split": "train", "weight": 1.0, "mem_mb": 1024.0})
    return table, demand


def constants_for(table):
    return pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=sorted(set(table["func_id"])))


def pools_of(policy, table, demand, rows):
    env = VecWarmPoolEnv(SIM_CONFIG)
    env.reset(**build_batch(table, demand, rows, constants_for(table), window=5))
    policy.start(env, rows)
    chosen = []
    for _ in range(demand.shape[1]):
        pool = policy(env)
        env.step_pool(pool)
        chosen.append(pool.copy())
    return np.array(chosen).T


# ---- the histogram ----
def test_interval_histogram_counts_gaps_and_out_of_bounds():
    counts, out_of_bounds = interval_histogram(np.array([0, 3, 6, 9, 400]), range_minutes=240)
    assert counts[3] == 3                                  # three gaps of 3 minutes
    assert counts.sum() == 3 and out_of_bounds.tolist() == [391]


# ---- choosing the windows ----
def test_a_regular_function_gets_a_narrow_window_around_its_period():
    counts, oob = interval_histogram(np.arange(0, 600, 15), 240)       # busy every 15 minutes
    mode, prewarm, end = choose_windows(counts, oob, DEFAULTS)
    assert mode == "histogram"
    assert prewarm == 13                                   # floor(15 * 0.9): unload for 13 minutes
    assert end == 16                                       # keep-alive = ceil((15 - 13) * 1.1) = 3 more minutes


def test_spread_out_idle_times_are_not_representative():
    gaps = np.arange(1, 51)                                # 50 different gaps, each once: a flat histogram
    counts, oob = interval_histogram(np.concatenate([[0], np.cumsum(gaps)]), 240)
    mode, prewarm, end = choose_windows(counts, oob, DEFAULTS)
    assert (mode, prewarm, end) == ("fallback", 0, 240)    # plain 4-hour keep-alive


def test_too_few_idle_times_fall_back():
    counts, oob = interval_histogram(np.array([0, 15, 30]), 240)
    assert choose_windows(counts, oob, DEFAULTS)[0] == "fallback"


def test_mostly_out_of_bounds_idle_times_use_a_forecast():
    counts, oob = interval_histogram(np.arange(0, 3600, 300), 240)     # one call every 5 hours (12 calls, 11 gaps)
    mode, prewarm, end = choose_windows(counts, oob, DEFAULTS)
    assert mode == "forecast"
    assert prewarm == 255 and end == 345                   # floor(300 * 0.85) and ceil(300 * 1.15)


def test_continuous_traffic_never_unloads():
    counts, oob = interval_histogram(np.arange(0, 100), 240)           # busy every minute: all gaps are 1
    mode, prewarm, end = choose_windows(counts, oob, DEFAULTS)
    assert mode == "histogram" and prewarm == 0


# ---- the pool over a day ----
def test_periodic_function_is_always_warm_exactly_when_needed():
    busy = list(range(0, DAY, 15))
    table, demand = world([busy, busy, busy])               # three identical days, every 15 minutes
    pools = pools_of(HybridHistogram(table, demand), table, demand, [2])[0]
    for minute in busy:
        assert pools[minute] >= 1, f"cold at minute {minute}"
    # unloaded for 13 minutes after each call, warm at minute 14 (idle) and 15 (the call), then the clock restarts
    assert pools[1:14].sum() == 0 and pools[14] == 1 and pools[15] == 1
    assert pools[16:29].sum() == 0 and pools[29] == 1


def test_hybrid_wastes_less_than_keep_alive_for_the_same_cold_starts_on_a_periodic_function():
    busy = list(range(0, DAY, 15))
    table, demand = world([busy] * 4)
    counts = demand.astype(np.uint32)
    constants = constants_for(table)
    hybrid = evaluate_policy(HybridHistogram(table, demand), table, demand, counts, [3], constants, SIM_CONFIG).iloc[0]
    keep15 = evaluate_policy(KeepAlive(15, table, demand), table, demand, counts, [3], constants, SIM_CONFIG).iloc[0]
    assert hybrid["cold_inv"] == keep15["cold_inv"] == 0
    assert hybrid["idle_gbmin"] < keep15["idle_gbmin"] / 3


def test_first_day_has_no_history_so_it_is_a_plain_four_hour_keep_alive():
    table, demand = world([[100]])
    pools = pools_of(HybridHistogram(table, demand), table, demand, [0])[0]
    assert pools[:101].sum() == 0 and (pools[101:341] == 1).all() and pools[341:].sum() == 0


def test_forecast_function_is_warm_when_the_call_arrives():
    busy = list(range(0, 1440, 300))                         # minutes 0, 300, ..., 1200
    table, demand = world([busy] * 5)                        # five identical days: enough idle times to trust
    pools = pools_of(HybridHistogram(table, demand), table, demand, [4])[0]
    assert (pools[[300, 600, 900, 1200]] >= 1).all()
    assert pools[301:556].sum() == 0                         # unloaded until the pre-warm window ends


# ---- no lookahead ----
def test_changing_the_days_later_minutes_does_not_change_earlier_pools():
    rng = np.random.default_rng(0)
    base = [list(np.nonzero(rng.random(DAY) < 0.05)[0]) for _ in range(3)]
    table, demand = world(base)
    t0 = 700
    changed = demand.copy()
    changed[2, t0:] += 3
    a = pools_of(HybridHistogram(table, demand), table, demand, [2])
    b = pools_of(HybridHistogram(table, changed), table, changed, [2])
    np.testing.assert_array_equal(a[:, :t0 + 1], b[:, :t0 + 1])
    assert not np.array_equal(a[:, t0 + 1:], b[:, t0 + 1:])


def test_the_histogram_uses_only_earlier_days():
    # changing day 3 must not change day 2's pools (day 3 is the future of day 2)
    rng = np.random.default_rng(1)
    base = [list(np.nonzero(rng.random(DAY) < 0.05)[0]) for _ in range(3)]
    table, demand = world(base)
    changed = demand.copy()
    changed[2] = 0
    a = pools_of(HybridHistogram(table, demand), table, demand, [1])
    b = pools_of(HybridHistogram(table, changed), table, changed, [1])
    np.testing.assert_array_equal(a, b)
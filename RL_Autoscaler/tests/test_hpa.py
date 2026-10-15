import numpy as np
import pandas as pd
import pytest

from baselines.hpa import HPA
from sim.batch import build_batch
from sim.evaluate import evaluate_policy
from sim.vec_env import VecWarmPoolEnv

SIM_CONFIG = {
    "reference_window": 5, "rho": 10, "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}


def world(demand_rows):
    demand = np.array(demand_rows, dtype=np.int64)
    table = pd.DataFrame({"func_id": np.arange(len(demand)), "day": 2, "group": "rare", "split": "train",
                          "weight": 1.0, "mem_mb": 1024.0})
    return table, demand


def two_days(level):
    """One function with a full previous day and a test day, both at a constant demand."""
    demand = np.full((2, 8), level, dtype=np.int64)
    table = pd.DataFrame({"func_id": 0, "day": [1, 2], "group": "rare", "split": "train", "weight": 1.0, "mem_mb": 1024.0})
    return table, demand


def pools_of(policy, table, demand, initial_pool=0, rows=None):
    rows = list(range(len(demand))) if rows is None else rows
    constants = pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=sorted(set(table["func_id"])))
    env = VecWarmPoolEnv(SIM_CONFIG)
    env.reset(**build_batch(table, demand, rows, constants, window=5), initial_pool=initial_pool)
    policy.start(env, rows)
    chosen = []
    for _ in range(demand.shape[1]):
        pool = policy(env)
        env.step_pool(pool)
        chosen.append(pool.copy())
    return np.array(chosen).T


def hpa(table, demand, **settings):
    base = {"target_pct": 100, "tolerance_pct": 10, "down_window": 5, "min_replicas": 0}
    return HPA(table, demand, **{**base, **settings})


# ---- the formula, with whole-number arithmetic ----
def test_scales_up_to_demand_over_target():
    table, demand = world([[6, 6, 6]])
    # minute 0 sees nothing (no history). Minute 1 sees demand 6: desired = ceil(6 / 0.5) = 12
    pools = pools_of(hpa(table, demand, target_pct=50), table, demand)
    assert pools.tolist() == [[0, 12, 12]]


def test_exact_boundary_is_not_rounded_up_by_float_error():
    table, demand = world([[7, 7, 7]])
    pools = pools_of(hpa(table, demand, target_pct=70), table, demand)
    assert pools[0, 1] == 10                       # 7 / 0.7 is exactly 10 (a float ceil would give 11)


def test_inside_the_tolerance_band_nothing_changes():
    # pool 10, demand 9, target 85%: utilisation 0.9, ratio 1.059, within 10% of 1 -> keep 10
    table, demand = two_days(9)
    pools = pools_of(hpa(table, demand, target_pct=85), table, demand, initial_pool=10, rows=[1])
    assert pools.tolist() == [[10] * 8]


def test_outside_the_tolerance_band_it_acts():
    # pool 10, demand 11, target 90%: utilisation 1.1, ratio 1.22 -> desired ceil(11 / 0.9) = 13
    table, demand = two_days(11)
    pools = pools_of(hpa(table, demand, target_pct=90), table, demand, initial_pool=10, rows=[1])
    assert pools[0].tolist() == [13] * 8


# ---- the scale-down stabilisation window ----
def test_scale_down_waits_for_the_stabilisation_window():
    table, demand = world([[10, 10, 0, 0, 0, 0, 0, 0, 0]])
    pools = pools_of(hpa(table, demand, down_window=5), table, demand)
    # scales up at once (minute 1), then keeps the highest recommendation of the last 5 minutes
    assert pools.tolist() == [[0, 10, 10, 10, 10, 10, 10, 0, 0]]


def test_scale_up_is_not_delayed():
    table, demand = world([[0, 0, 0, 8, 8, 8]])
    pools = pools_of(hpa(table, demand), table, demand)
    assert pools[0, 4] == 8                        # demand appears at minute 3, the pool follows at minute 4


# ---- minimum replicas: the two variants the plan asks for ----
def test_with_min_replicas_one_the_pool_never_reaches_zero():
    table, demand = world([[0] * 12])
    pools = pools_of(hpa(table, demand, min_replicas=1), table, demand)
    assert (pools == 1).all()


def test_with_min_replicas_zero_it_scales_to_zero():
    table, demand = world([[0] * 12])
    pools = pools_of(hpa(table, demand, min_replicas=0), table, demand)
    assert (pools == 0).all()


# ---- uses yesterday at the start of the day ----
def test_first_minute_uses_the_last_minute_of_yesterday():
    demand = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 5],       # day 1: busy in its last minute
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])      # day 2
    table = pd.DataFrame({"func_id": 0, "day": [1, 2], "group": "rare", "split": "train", "weight": 1.0, "mem_mb": 1.0})
    constants = pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=[0])
    env = VecWarmPoolEnv(SIM_CONFIG)
    env.reset(**build_batch(table, demand, [1], constants, window=5))
    policy = hpa(table, demand)
    policy.start(env, [1])
    assert policy(env)[0] == 5                             # sees yesterday's 5 at minute 0 of day 2


# ---- no lookahead ----
def test_changing_demand_from_minute_t_on_does_not_change_pools_up_to_t():
    rng = np.random.default_rng(0)
    base = (rng.random((6, 300)) < 0.1) * rng.integers(1, 6, size=(6, 300))
    table, demand = world(base)
    changed = demand.copy()
    changed[:, 150:] += 4
    a = pools_of(hpa(table, demand, target_pct=70, min_replicas=1), table, demand)
    b = pools_of(hpa(table, changed, target_pct=70, min_replicas=1), table, changed)
    np.testing.assert_array_equal(a[:, :151], b[:, :151])
    assert not np.array_equal(a[:, 151:], b[:, 151:])


# ---- sanity of the target ----
def test_a_lower_target_wastes_more_and_has_fewer_cold_starts():
    rng = np.random.default_rng(1)
    demand = (rng.random((20, 1440)) < 0.05) * rng.integers(1, 8, size=(20, 1440))
    table, _ = world(demand)
    counts = demand.astype(np.uint32)
    constants = pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=range(20))
    results = {}
    for target in (40, 90):
        r = evaluate_policy(hpa(table, demand, target_pct=target), table, demand, counts, list(range(20)), constants, SIM_CONFIG)
        results[target] = (r["cold_inv"].sum(), r["idle_gbmin"].sum())
    assert results[40][0] <= results[90][0] and results[40][1] >= results[90][1]
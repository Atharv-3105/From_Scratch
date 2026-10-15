import numpy as np
import pandas as pd
import pytest

from baselines.fixed import KeepAlive, window_max, oracle, never_scale
from sim.batch import build_batch, previous_day
from sim.evaluate import evaluate_policy
from sim.vec_env import VecWarmPoolEnv

SIM_CONFIG = {
    "reference_window": 5,
    "rho": 10,
    "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}


def make_table(days, func_id=0):
    return pd.DataFrame({"func_id": func_id, "day": days, "group": "rare", "split": "train",
                         "weight": 1.0, "mem_mb": 1024.0})


def constants_for(table):
    ids = sorted(set(table["func_id"]))
    return pd.DataFrame({"max_pool": 10 ** 6, "scale": 1.0}, index=ids)


def pools_of(policy, table, demand, rows):
    """Play the policy minute by minute and return the pool it chose at every minute (rows x minutes)."""
    env = VecWarmPoolEnv(SIM_CONFIG)
    env.reset(**build_batch(table, demand, rows, constants_for(table), window=5))
    if hasattr(policy, "start"):
        policy.start(env, rows)
    chosen = []
    for _ in range(demand.shape[1]):
        pool = policy(env)
        env.step_pool(pool)
        chosen.append(pool.copy())
    return np.array(chosen).T


# ---- the sliding maximum helper ----
@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 17, 40])
def test_window_max_matches_brute_force(k):
    rng = np.random.default_rng(k)
    values = rng.integers(0, 50, size=(4, 40))
    got = window_max(values, k)
    want = np.array([[values[r, i:i + k].max() for i in range(40 - k + 1)] for r in range(4)])
    np.testing.assert_array_equal(got, want)


# ---- hand-computed keep-alive ----
def test_keep_alive_three_minutes_by_hand():
    demand = np.array([[0, 5, 0, 0, 0, 0, 0, 2, 0]])
    table = make_table([1])
    pools = pools_of(KeepAlive(3, table, demand), table, demand, [0])
    # the 5 is used at minute 1, so it stays warm for minutes 2, 3, 4. The 2 (minute 7) stays for minute 8.
    assert pools.tolist() == [[0, 0, 5, 5, 5, 0, 0, 0, 2]]


def test_keep_alive_one_minute_is_the_previous_minutes_demand():
    demand = np.array([[0, 5, 0, 0, 0, 0, 0, 2, 0]])
    table = make_table([1])
    pools = pools_of(KeepAlive(1, table, demand), table, demand, [0])
    assert pools.tolist() == [[0, 0, 5, 0, 0, 0, 0, 0, 2]]


def test_keep_alive_reaches_back_into_yesterday():
    demand = np.array([[0, 0, 0, 0, 0, 0, 0, 4, 0],        # day 1: demand 4 at its minute 7, quiet at the end
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])       # day 2: nothing
    table = make_table([1, 2])
    pools = pools_of(KeepAlive(3, table, demand), table, demand, [1])
    assert pools.tolist() == [[4, 4, 0, 0, 0, 0, 0, 0, 0]]  # yesterday's last 3 minutes are [0, 4, 0]


def test_without_a_previous_day_the_history_is_zeros():
    demand = np.array([[0, 0, 0, 0, 0, 0, 0, 4, 0]])
    table = make_table([2])
    assert previous_day(table, demand, [0]).tolist() == [[0] * 9]
    assert pools_of(KeepAlive(3, table, demand), table, demand, [0]).tolist()[0][:3] == [0, 0, 0]


def test_never_unload_remembers_the_biggest_demand_ever_seen():
    demand = np.array([[0, 0, 0, 0, 0, 0, 0, 4, 0],
                       [0, 2, 0, 0, 0, 0, 0, 0, 0]])
    table = make_table([1, 2])
    pools = pools_of(KeepAlive(None, table, demand), table, demand, [1])
    assert pools.tolist() == [[4] * 9]                       # yesterday's 4 is never forgotten


def test_a_window_longer_than_the_day_is_rejected():
    demand = np.zeros((1, 9), dtype=np.int64)
    table = make_table([1])
    with pytest.raises(ValueError):
        pools_of(KeepAlive(10, table, demand), table, demand, [0])


# ---- the sweep behaves the way keep-alive should ----
def random_world(seed=0, rows=24, minutes=1440):
    rng = np.random.default_rng(seed)
    demand = (rng.random((rows, minutes)) < 0.03) * rng.integers(1, 6, size=(rows, minutes))
    func_ids = np.repeat(np.arange(rows // 2), 2)                    # two consecutive days per function
    table = pd.DataFrame({"func_id": func_ids, "day": np.tile([1, 2], rows // 2), "group": "rare",
                          "split": "train", "weight": 1.0, "mem_mb": 100.0})
    return table, demand.astype(np.int64)


def test_larger_k_never_gives_a_smaller_pool():
    table, demand = random_world()
    rows = list(range(len(table)))
    previous = None
    for k in [1, 5, 10, 20, 60, None]:
        pools = pools_of(KeepAlive(k, table, demand), table, demand, rows)
        if previous is not None:
            assert (pools >= previous).all(), f"pool shrank when K grew to {k}"
        previous = pools


def test_larger_k_means_fewer_cold_starts_and_more_waste():
    table, demand = random_world(seed=1)
    constants = constants_for(table)
    cold, waste = [], []
    for k in [1, 5, 10, 20, 60, None]:
        result = evaluate_policy(KeepAlive(k, table, demand), table, demand, demand.astype(np.uint32),
                                 list(range(len(table))), constants, SIM_CONFIG)
        cold.append(result["cold_inv"].sum())
        waste.append(result["idle_gbmin"].sum())
    assert cold == sorted(cold, reverse=True) and waste == sorted(waste)
    assert cold[0] > cold[-1] and waste[0] < waste[-1]            # and the sweep really moves


# ---- no lookahead ----
def test_changing_demand_from_minute_t_on_does_not_change_the_pool_up_to_t():
    table, demand = random_world(seed=2, rows=6)         # rows come in pairs: day 1, then day 2 of the same function
    rows = list(range(6))
    t0 = 800
    changed = demand.copy()
    changed[1::2, t0:] += 7                              # change only the day-2 rows. A day-1 row is the "yesterday" of
                                                         # its day-2 row, and a baseline is allowed to see yesterday.
    for k in (5, 60, None):
        a = pools_of(KeepAlive(k, table, demand), table, demand, rows)
        b = pools_of(KeepAlive(k, table, changed), table, changed, rows)
        np.testing.assert_array_equal(a[:, :t0 + 1], b[:, :t0 + 1])                # nothing up to and including minute t0 moved
        assert not np.array_equal(a[1::2, t0 + 1:], b[1::2, t0 + 1:])              # and the test is sensitive to the change


# ---- the bounds ----
def test_oracle_and_never_scale():
    demand = np.array([[0, 5, 0, 0, 0, 0, 0, 2, 0]])
    table = make_table([1])
    assert pools_of(oracle, table, demand, [0]).tolist() == demand.tolist()
    assert pools_of(never_scale, table, demand, [0]).tolist() == [[0] * 9]


def test_five_minute_keep_alive_is_the_environments_reference_demand():
    table, demand = random_world(seed=3, rows=6)
    rows = list(range(6))
    env_reference = lambda env: env.ref[:, env.minute]
    a = pools_of(KeepAlive(5, table, demand), table, demand, rows)
    b = pools_of(env_reference, table, demand, rows)
    np.testing.assert_array_equal(a[:, :], b[:, :])
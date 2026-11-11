import numpy as np
import pandas as pd
import pytest

from baselines.seasonal import SeasonalPredictive
from sim.batch import build_batch
from sim.evaluate import evaluate_policy
from sim.vec_env import VecWarmPoolEnv

SIM_CONFIG = {
    "reference_window": 5, "rho": 10, "kappa": 0.1,
    "actions": [["absolute", 0], ["absolute", 1], ["relative", "1/2"],
                ["relative", "1"], ["relative", "3/2"], ["relative", "2"]],
}


def world(day_rows):
    """One function, consecutive days 1, 2, ... one row of demand per day."""
    demand = np.array(day_rows, dtype=np.int64)
    table = pd.DataFrame({"func_id": 0, "day": np.arange(1, len(demand) + 1), "group": "rare", "split": "train",
                          "weight": 1.0, "mem_mb": 1024.0})
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


def seasonal(table, demand, half_window=2, margin_pct=100):
    return SeasonalPredictive(table, demand, half_window=half_window, margin_pct=margin_pct)


# ---- the forecast: the same minutes yesterday ----
def test_pool_follows_yesterdays_window_around_the_same_minute():
    day = [0] * 20
    day[10] = 3                                              # demand 3 at minute 10, every day
    table, demand = world([day, day])
    pools = pools_of(seasonal(table, demand, half_window=2), table, demand, [1])[0]
    # warm from minute 8 (= 10 - 2) to minute 12 (= 10 + 2): the window around yesterday's spike
    assert pools.tolist() == [0] * 8 + [3] * 5 + [0] * 7


def test_a_perfectly_periodic_function_has_no_cold_starts_and_little_waste():
    day = [0] * 1440
    day[100] = 1
    table, demand = world([day] * 3)
    result = evaluate_policy(seasonal(table, demand, half_window=2), table, demand, demand.astype(np.uint32),
                             [2], constants_for(table), SIM_CONFIG).iloc[0]
    assert result["cold_inv"] == 0
    assert result["idle_gbmin"] == 4                        # 5 warm minutes, 1 of them used


def test_window_is_cut_off_at_the_start_of_the_day():
    day = [0] * 20
    day[1] = 4
    table, demand = world([day, [0] * 20])
    pools = pools_of(seasonal(table, demand, half_window=3), table, demand, [1])[0]
    assert pools[0] == 4                                     # minute 0 looks at yesterday's minutes 0..3


def test_window_may_run_into_the_early_minutes_of_today_but_never_the_future():
    # at the last minutes of the day the window reaches past yesterday's end into today's FIRST minutes (all earlier than now)
    today = [0] * 20
    today[0] = 5
    table, demand = world([[0] * 20, today])
    pools = pools_of(seasonal(table, demand, half_window=3), table, demand, [1])[0]
    assert pools[17] == 5 and pools[16] == 0                 # minute 17 sees today's minute 0 (17 - 3 .. 17 + 3 reaches minute 20)


# ---- the reactive term and the margin ----
def test_first_day_is_purely_reactive():
    table, demand = world([[0, 3, 0, 0, 7, 0, 0, 0]])
    pools = pools_of(seasonal(table, demand, half_window=1), table, demand, [0])[0]
    assert pools.tolist() == [0, 0, 3, 0, 0, 7, 0, 0]       # pool = last minute's demand


def test_margin_scales_the_pool_and_rounds_up_exactly():
    table, demand = world([[0, 10, 0, 0], [0, 0, 0, 0]])
    assert pools_of(seasonal(table, demand, half_window=0, margin_pct=130), table, demand, [0])[0][2] == 13
    table, demand = world([[0, 3, 0, 0]])
    assert pools_of(seasonal(table, demand, half_window=0, margin_pct=150), table, demand, [0])[0][2] == 5   # ceil(4.5)


def test_a_busier_day_than_yesterday_is_followed_by_the_reactive_term():
    table, demand = world([[0] * 10, [0, 0, 6, 6, 6, 6, 0, 0, 0, 0]])
    pools = pools_of(seasonal(table, demand, half_window=1), table, demand, [1])[0]
    assert pools[3] == 6 and pools[4] == 6                   # no forecast, but it reacts to what it just saw


def test_a_quieter_day_than_yesterday_still_pays_for_the_forecast():
    day = [0] * 10
    day[5] = 4
    table, demand = world([day, [0] * 10])
    pools = pools_of(seasonal(table, demand, half_window=1), table, demand, [1])[0]
    assert pools[4] == 4 and pools[5] == 4 and pools[6] == 4  # warm for a call that never comes


# ---- monotonic knobs ----
def test_bigger_margin_and_wider_window_never_shrink_the_pool():
    rng = np.random.default_rng(0)
    rows = (rng.random((3, 300)) < 0.05) * rng.integers(1, 6, size=(3, 300))
    table, demand = world(rows)
    for rows_to_test in ([2],):
        base = pools_of(seasonal(table, demand, 5, 100), table, demand, rows_to_test)
        assert (pools_of(seasonal(table, demand, 5, 150), table, demand, rows_to_test) >= base).all()
        assert (pools_of(seasonal(table, demand, 15, 100), table, demand, rows_to_test) >= base).all()


# ---- no lookahead ----
def test_changing_todays_later_minutes_does_not_change_earlier_pools():
    rng = np.random.default_rng(1)
    rows = (rng.random((2, 400)) < 0.05) * rng.integers(1, 6, size=(2, 400))
    table, demand = world(rows)
    changed = demand.copy()
    changed[1, 200:] += 5
    a = pools_of(seasonal(table, demand, 10, 120), table, demand, [1])
    b = pools_of(seasonal(table, changed, 10, 120), table, changed, [1])
    np.testing.assert_array_equal(a[:, :201], b[:, :201])
    assert not np.array_equal(a[:, 201:], b[:, 201:])


def test_half_window_must_be_shorter_than_the_day():
    table, demand = world([[0] * 10, [0] * 10])
    with pytest.raises(ValueError):
        pools_of(seasonal(table, demand, half_window=10), table, demand, [1])
import numpy as np
import pandas as pd
import pytest

from features.builder import (
    N_MIN, TRIGGERS, FEATURE_NAMES, HISTORY_FEATURE_NAMES,
    history_features, precompute_history_features, pool_features, full_observation,
    OnlineFeatureBuilder, function_avg_duration, episode_inputs,
)

STATICS = {"mem_mb": 1024.0, "avg_dur_ms": 100.0, "scale": 2.0, "trigger": "timer"}


def value_of(features, name):
    return features[HISTORY_FEATURE_NAMES.index(name)]


def make_inputs():
    # Yesterday is quiet except 6 at minute 12 and 2 at minute 13. Today, up to minute 10, is
    # quiet except 4 at minute 8 and 2 at minute 9. Scale = 2, so demand features are divided by 2.
    yesterday = np.zeros(N_MIN)
    yesterday[12], yesterday[13] = 6, 2
    today = np.zeros(10)
    today[8], today[9] = 4, 2
    return today, yesterday


# ---- the layout ----
def test_feature_layout():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 28
    assert FEATURE_NAMES[-2:] == ["pool", "pool_over_ref"]      # the two agent-dependent features come last
    assert len(HISTORY_FEATURE_NAMES) == 26


# ---- every feature, computed by hand ----
def test_history_features_by_hand():
    today, yesterday = make_inputs()
    f = history_features(today, yesterday, has_yesterday=True, day=3, statics=STATICS)
    assert f.shape == (26,)

    assert value_of(f, "last_minute") == pytest.approx(2 / 2)              # demand at minute 9, divided by scale
    assert value_of(f, "mean_5") == pytest.approx((0 + 0 + 0 + 4 + 2) / 5 / 2)
    assert value_of(f, "ref_demand") == pytest.approx(4 / 2)                # max of the last 5 minutes
    assert value_of(f, "mean_15") == pytest.approx(6 / 15 / 2)              # 5 quiet minutes from yesterday + today's 10
    assert value_of(f, "max_15") == pytest.approx(4 / 2)
    assert value_of(f, "mean_60") == pytest.approx(6 / 60 / 2)
    assert value_of(f, "max_60") == pytest.approx(4 / 2)
    assert value_of(f, "trend") == pytest.approx((6 / 5 - 6 / 60) / 2)       # short mean minus long mean
    assert value_of(f, "log_minutes_since_demand") == pytest.approx(np.log1p(1))   # last demand was the minute before
    assert value_of(f, "time_sin") == pytest.approx(np.sin(2 * np.pi * 10 / 1440))
    assert value_of(f, "time_cos") == pytest.approx(np.cos(2 * np.pi * 10 / 1440))
    assert value_of(f, "weekday_sin") == pytest.approx(np.sin(2 * np.pi * 2 / 7))   # day 3 -> day index 2
    assert value_of(f, "weekday_cos") == pytest.approx(np.cos(2 * np.pi * 2 / 7))
    assert value_of(f, "yesterday_mean") == pytest.approx((6 + 2) / 15 / 2)          # yesterday's minutes 10..24
    assert value_of(f, "yesterday_max") == pytest.approx(6 / 2)
    assert value_of(f, "has_yesterday") == 1.0
    assert value_of(f, "log_mem") == pytest.approx(np.log1p(1024))
    assert value_of(f, "log_avg_duration") == pytest.approx(np.log1p(100))
    assert value_of(f, "log_scale") == pytest.approx(np.log(2))
    one_hot = [value_of(f, f"trigger_{name}") for name in TRIGGERS]
    assert one_hot == [1.0 if name == "timer" else 0.0 for name in TRIGGERS]


def test_unknown_trigger_counts_as_others():
    today, yesterday = make_inputs()
    f = history_features(today, yesterday, True, 3, {**STATICS, "trigger": "carrier-pigeon"})
    assert value_of(f, "trigger_others") == 1.0


def test_minutes_since_demand_is_capped_at_one_day():
    quiet_yesterday = np.zeros(N_MIN)
    f = history_features(np.zeros(100), quiet_yesterday, False, 1, STATICS)
    assert value_of(f, "log_minutes_since_demand") == pytest.approx(np.log1p(1440))
    yesterday = np.zeros(N_MIN)
    yesterday[100] = 1                                                      # last demand 1400 minutes before minute 60
    f = history_features(np.zeros(60), yesterday, True, 1, STATICS)
    assert value_of(f, "log_minutes_since_demand") == pytest.approx(np.log1p(1400))


def test_missing_yesterday_is_flagged():
    f = history_features(np.zeros(5), np.zeros(N_MIN), has_yesterday=False, day=1, statics=STATICS)
    assert value_of(f, "has_yesterday") == 0.0 and value_of(f, "yesterday_mean") == 0.0


# ---- the agent-dependent features ----
def test_pool_features():
    out = pool_features(pool=np.array([3, 4]), ref_demand=np.array([0, 2]), scale=np.array([2.0, 4.0]))
    assert out.tolist() == [[1.5, 3.0], [1.0, 2.0]]       # pool / scale, and pool / max(R, 1)


def test_full_observation_appends_the_pool_features():
    row = np.arange(26.0)[None, :]
    full = full_observation(row, pool=np.array([2]), ref_demand=np.array([1]), scale=np.array([2.0]))
    assert full.shape == (1, 28) and full[0, -2:].tolist() == [1.0, 2.0]


# ---- one builder, two ways to run it: they must agree ----
def random_episode(rng, n=5, minutes=N_MIN):
    demand = rng.integers(0, 5, size=(n, minutes)) * (rng.random((n, minutes)) < 0.2)
    yesterday = rng.integers(0, 5, size=(n, N_MIN)) * (rng.random((n, N_MIN)) < 0.2)
    has = np.array([True, True, False, True, False])[:n]
    yesterday[~has] = 0
    statics = {"mem_mb": rng.uniform(50, 500, n), "avg_dur_ms": rng.uniform(1, 5000, n),
               "scale": rng.uniform(1, 6, n), "trigger": rng.integers(0, len(TRIGGERS), n)}
    return demand, yesterday, has, np.array([1, 2, 6, 11, 14])[:n], statics


def test_fast_and_reference_builders_agree():
    rng = np.random.default_rng(0)
    demand, yesterday, has, day, statics = random_episode(rng)
    fast = precompute_history_features(demand, yesterday, has, day, statics)
    assert fast.shape == (5, N_MIN, 26)
    for i in range(5):
        one = {"mem_mb": statics["mem_mb"][i], "avg_dur_ms": statics["avg_dur_ms"][i],
               "scale": statics["scale"][i], "trigger": TRIGGERS[statics["trigger"][i]]}
        for t in range(0, N_MIN, 7):                          # every 7th minute, plus the last ones below
            slow = history_features(demand[i, :t], yesterday[i], has[i], day[i], one)
            np.testing.assert_allclose(fast[i, t], slow, rtol=1e-9, atol=1e-12, err_msg=f"episode {i} minute {t}")
        slow = history_features(demand[i, :N_MIN - 1], yesterday[i], has[i], day[i], one)
        np.testing.assert_allclose(fast[i, N_MIN - 1], slow, rtol=1e-9, atol=1e-12)


def test_online_builder_matches_the_function():
    rng = np.random.default_rng(1)
    demand, yesterday, has, day, statics = random_episode(rng, n=1)
    one = {"mem_mb": 200.0, "avg_dur_ms": 50.0, "scale": 3.0, "trigger": "queue"}
    builder = OnlineFeatureBuilder()
    builder.start_day(yesterday[0], bool(has[0]), int(day[0]), one)
    for t in range(60):
        np.testing.assert_allclose(builder.features(), history_features(demand[0, :t], yesterday[0], has[0], day[0], one))
        builder.observe(demand[0, t])


# ---- no lookahead ----
def test_changing_demand_at_or_after_minute_t_does_not_change_features_up_to_t():
    rng = np.random.default_rng(2)
    demand, yesterday, has, day, statics = random_episode(rng)
    t0 = 700
    changed = demand.copy()
    changed[:, t0:] += 9
    a = precompute_history_features(demand, yesterday, has, day, statics)
    b = precompute_history_features(changed, yesterday, has, day, statics)
    np.testing.assert_array_equal(a[:, :t0 + 1], b[:, :t0 + 1])        # features AT minute t0 only use minutes before t0
    assert not np.array_equal(a[:, t0 + 1:], b[:, t0 + 1:])            # and the test really is sensitive to the change


# ---- per-function constants and building inputs from the table ----
def make_table():
    table = pd.DataFrame({
        "func_id": [0, 0, 0, 1],
        "day": [1, 2, 11, 5],
        "dur_ms": [100.0, 300.0, 99999.0, 50.0],
        "total_invocations": [1, 3, 5, 0],
        "mem_mb": [128.0] * 4,
        "Trigger": ["http", "http", "http", "weird"],
    })
    return table


def test_average_duration_uses_train_days_weighted_by_invocations():
    avg = function_avg_duration(make_table(), train_days=range(1, 11))
    assert avg.loc[0] == pytest.approx((100 * 1 + 300 * 3) / 4)       # the day-11 row is ignored
    assert avg.loc[1] == pytest.approx(np.median([(100 * 1 + 300 * 3) / 4]))   # no invocations -> overall median


def test_episode_inputs():
    table = make_table()
    demand = np.vstack([np.full(N_MIN, 1), np.full(N_MIN, 2), np.full(N_MIN, 3), np.full(N_MIN, 4)]).astype(np.int32)
    constants = pd.DataFrame({"max_pool": [5, 6], "scale": [1.5, 2.5]}, index=[0, 1])
    avg = function_avg_duration(table, range(1, 11))
    out = episode_inputs(table, demand, [1, 3], constants, avg)
    assert out["yesterday"][0].tolist() == [1] * N_MIN and out["has_yesterday"].tolist() == [True, False]
    assert out["day"].tolist() == [2, 5] and out["demand"][1].tolist() == [4] * N_MIN
    assert out["statics"]["scale"].tolist() == [1.5, 2.5]
    assert out["statics"]["trigger"].tolist() == [TRIGGERS.index("http"), TRIGGERS.index("others")]
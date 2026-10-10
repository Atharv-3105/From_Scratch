import numpy as np
import pandas as pd
import pytest

from traces.stats import (
    N_MIN, rate_per_hour, avg_day_correlation, hourly_variance_and_mean,
    compute_function_stats, label_functions, sample_by_group,
)


def one_spike_day(minute, size=1):
    # A day with all zeros except one minute that has `size` invocations
    day = np.zeros(N_MIN)
    day[minute] = size
    return day


# ---- rate over the days a function is alive ----
def test_rate_per_hour():
    total = np.array([240, 240, 240])
    first_day = np.array([1, 9, 10])
    # 10 alive days -> 1/h, 2 alive days -> 5/h, 1 alive day -> 10/h
    np.testing.assert_allclose(rate_per_hour(total, first_day, 10), [1.0, 5.0, 10.0])


# ---- day-over-day correlation ----
def test_correlation_of_identical_days_is_one():
    days = np.vstack([one_spike_day(10), one_spike_day(10)])
    assert avg_day_correlation(days) == pytest.approx(1.0)


def test_correlation_of_two_different_spikes():
    days = np.vstack([one_spike_day(10), one_spike_day(11)])
    assert avg_day_correlation(days) == pytest.approx(-1.0 / (N_MIN - 1))


def test_correlation_averages_all_pairs():
    day_a, day_b = one_spike_day(10), one_spike_day(500)
    different = -1.0 / (N_MIN - 1)
    # pairs: (a,a)=1, (a,b)=different, (a,b)=different
    result = avg_day_correlation(np.vstack([day_a, day_a, day_b]))
    assert result == pytest.approx((1 + 2 * different) / 3)


def test_correlation_is_nan_when_it_cannot_be_computed():
    assert np.isnan(avg_day_correlation(one_spike_day(10)[None, :]))                        # only one day
    assert np.isnan(avg_day_correlation(np.vstack([one_spike_day(10), np.ones(N_MIN)])))    # other day is flat


# ---- burstiness against the "steady random traffic" baseline ----
def test_flat_traffic_has_zero_variance():
    variance_sum, mean_sum = hourly_variance_and_mean(np.full((1, N_MIN), 2.0))
    assert variance_sum[0] == pytest.approx(0.0)
    assert mean_sum[0] == pytest.approx(48.0)   # 24 hourly blocks, each with mean 2


def test_single_spike_scores_its_own_size():
    day = one_spike_day(30, size=60)[None, :]
    variance_sum, mean_sum = hourly_variance_and_mean(day)
    # one hour has mean 1 and variance 60, all other hours are empty
    assert variance_sum[0] / mean_sum[0] == pytest.approx(60.0)


def test_steady_random_traffic_scores_about_one():
    rng = np.random.default_rng(0)
    days = rng.poisson(5.0, size=(200, N_MIN))
    variance_sum, mean_sum = hourly_variance_and_mean(days)
    assert variance_sum.sum() / mean_sum.sum() == pytest.approx(1.0, abs=0.03)


# ---- per-function stats: train days only ----
def tiny_data():
    rows, days = [], []
    def add(func, day, counts):
        rows.append(("o", "a", func, day))
        days.append(counts)
    add("f1", 1, one_spike_day(10, 6))
    add("f1", 2, one_spike_day(10, 6))
    add("f2", 9, one_spike_day(50, 24))
    add("f3", 11, one_spike_day(5, 10_000))      # appears only after the train days
    add("f1", 11, one_spike_day(777, 10_000))    # test-day row of f1: must be ignored
    day_table = pd.DataFrame(rows, columns=["HashOwner", "HashApp", "HashFunction", "day"])
    minutes = np.vstack(days).astype(np.uint32)
    day_table["total_invocations"] = minutes.sum(axis=1)
    return day_table, minutes


def test_function_stats_use_train_days_only():
    day_table, minutes = tiny_data()
    stats, minute_of_day_totals = compute_function_stats(day_table, minutes, first_train_day=1, last_train_day=10)
    by_func = stats.set_index("HashFunction")

    assert len(by_func) == 3
    assert by_func.loc["f1", "n_days"] == 2
    assert by_func.loc["f1", "rate_per_hour"] == pytest.approx(12 / (10 * 24))
    assert by_func.loc["f1", "day_corr"] == pytest.approx(1.0)
    assert by_func.loc["f2", "first_day"] == 9
    assert by_func.loc["f2", "rate_per_hour"] == pytest.approx(24 / (2 * 24))
    assert not by_func.loc["f3", "has_train"]
    assert np.isnan(by_func.loc["f3", "rate_per_hour"])
    assert minute_of_day_totals.sum() == 12 + 24   # the 10,000-invocation day-11 rows are not counted


# ---- groups and the sample ----
CONFIG = {
    "thresholds": {"rare_rate_per_hour": 1.0, "hot_rate_per_hour": 50.0,
                   "periodic_corr": 0.9, "bursty_score": 3.0},
    "group_priority": ["periodic", "bursty", "hot", "rare"],
}


def make_stats(rows):
    return pd.DataFrame(rows, columns=["rate_per_hour", "day_corr", "burst_score", "has_train"])


def test_flags_and_group_priority():
    stats = make_stats([
        (0.5, 0.95, 1.0, True),        # rare AND periodic -> group periodic, but rare flag stays
        (100., 0.1, 10.0, True),       # hot AND bursty    -> group bursty
        (100., 0.1, 1.0, True),        # hot only
        (5.0, 0.1, 1.0, True),         # nothing           -> normal
        (np.nan, np.nan, np.nan, False),
    ])
    labelled = label_functions(stats, CONFIG)
    assert labelled.loc[0, ["rare", "periodic"]].all() and labelled.loc[0, "group"] == "periodic"
    assert labelled.loc[1, ["hot", "bursty"]].all() and labelled.loc[1, "group"] == "bursty"
    assert labelled.loc[2, "group"] == "hot"
    assert labelled.loc[3, "group"] == "normal"
    assert labelled.loc[4, "group"] == "unlabeled"
    assert not labelled.loc[4, ["rare", "hot", "periodic", "bursty"]].any()


def test_sample_by_group():
    groups = ["rare"] * 900 + ["hot"] * 90 + ["bursty"] * 10 + ["unlabeled"] * 50
    stats = pd.DataFrame({"group": groups})
    first = sample_by_group(stats, size=100, min_per_group=20, seed=0)
    second = sample_by_group(stats, size=100, min_per_group=20, seed=0)

    assert first.index.tolist() == second.index.tolist()          # same seed, same sample
    assert "unlabeled" not in set(first["group"])
    counts = first["group"].value_counts()
    assert counts["bursty"] == 10                                  # floor of 20 is capped by the group size
    assert counts["hot"] >= 20
    for name, part in first.groupby("group"):                      # weights add back up to the real group size
        assert part["weight"].sum() == pytest.approx((stats["group"] == name).sum())
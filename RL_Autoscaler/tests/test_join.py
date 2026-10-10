import pandas as pd
import pytest

from traces.join import build_function_day_table


def _keys():
    rows = [
        ("o1", "A", "f1", 1), ("o1", "A", "f1", 2), ("o1", "A", "f1", 3),
        ("o1", "B", "f2", 1), ("o1", "C", "f3", 11),
    ]
    return pd.DataFrame(rows, columns=["HashOwner", "HashApp", "HashFunction", "day"])


def _dur():
    rows = [
        ("o1", "A", "f1", 1, 100, 1), ("o1", "A", "f1", 1, 300, 3),   # two partial records, same day
        ("o1", "A", "f1", 2, -5, 2),                                    # invalid (negative) duration
        ("o1", "C", "f3", 11, 40, 5),
        ("o9", "Z", "f9", 1, 1000, 1),                                  # not in keys; only feeds the global median
    ]
    return pd.DataFrame(rows, columns=["HashOwner", "HashApp", "HashFunction", "day", "Average", "Count"])


def _mem():
    rows = [
        ("o1", "A", 1, 100), ("o1", "A", 2, 200),
        ("o1", "C", 11, 500),                                           # test-day only: must not be used
        ("o9", "Z", 1, 300),
    ]
    return pd.DataFrame(rows, columns=["HashOwner", "HashApp", "day", "AverageAllocatedMb"])


def test_rows_preserved_and_in_order():
    keys = _keys()
    out = build_function_day_table(keys, _dur(), _mem(), train_days=range(1, 11))
    assert len(out) == len(keys)
    assert out[["HashFunction", "day"]].values.tolist() == keys[["HashFunction", "day"]].values.tolist()


def test_duration_resolution_order():
    out = build_function_day_table(_keys(), _dur(), _mem(), train_days=range(1, 11))
    assert out.loc[0, "dur_ms"] == 250 and out.loc[0, "dur_source"] == "same_day"
    assert out.loc[1, "dur_ms"] == 250 and out.loc[1, "dur_source"] == "function_avg"
    assert out.loc[2, "dur_source"] == "function_avg"
    assert out.loc[3, "dur_ms"] == 625 and out.loc[3, "dur_source"] == "global"
    assert out.loc[4, "dur_ms"] == 40 and out.loc[4, "dur_source"] == "same_day"


def test_memory_resolution_order():
    out = build_function_day_table(_keys(), _dur(), _mem(), train_days=range(1, 11))
    assert out.loc[0, "mem_mb"] == 150 and out.loc[0, "mem_source"] == "app_median"
    assert out.loc[3, "mem_mb"] == 200 and out.loc[3, "mem_source"] == "global"
    assert out.loc[4, "mem_mb"] == 200 and out.loc[4, "mem_source"] == "global"


def test_no_missing_values_left():
    out = build_function_day_table(_keys(), _dur(), _mem(), train_days=range(1, 11))
    assert out[["dur_ms", "mem_mb"]].notna().all().all()
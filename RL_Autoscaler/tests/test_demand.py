import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from traces.demand import build_demand, subset_demand


def demand_of(n, dur_ms, concurrency=1):
    # Demand for ONE minute, using the real function on a 1-row, 1-minute-wide matrix
    counts = np.zeros((1, 1440), dtype=np.uint32)
    counts[0, 0] = n
    return int(build_demand(counts, np.array([dur_ms], dtype=float), concurrency)[0, 0])


def exact_demand(n, dur_ms, concurrency):
    # Slow but exact reference: whole fractions, no floats, no numpy
    if n == 0:
        return 0
    return max(1, math.ceil(Fraction(n * dur_ms, 60_000 * concurrency)))


# ---- hand-computed cases ----
def test_no_invocations_means_no_demand():
    assert demand_of(0, 1_800_000) == 0


def test_tiny_work_still_needs_one_instance():
    assert demand_of(1, 50) == 1          # 1 * 0.05s / 60 = 0.0008 -> rounds up, never below 1


def test_exact_boundary_does_not_round_up():
    assert demand_of(30, 2000) == 1       # 30 * 2s = 60s of work = exactly one instance-minute
    assert demand_of(60, 1000) == 1
    assert demand_of(120, 2000) == 4


def test_just_over_the_boundary_rounds_up():
    assert demand_of(31, 2000) == 2       # 62s of work needs a second instance


def test_zero_duration_with_traffic_needs_one_instance():
    assert demand_of(5, 0) == 1
    assert demand_of(0, 0) == 0


def test_concurrency_four_cuts_demand():
    assert demand_of(120, 2000, concurrency=1) == 4
    assert demand_of(120, 2000, concurrency=4) == 1


def test_fractional_duration_is_rounded_to_whole_ms():
    assert demand_of(30, 2000.4) == 1     # rounds to 2000 ms -> exactly one instance
    assert demand_of(30, 2000.6) == 2     # rounds to 2001 ms -> just over


# ---- the traps ----
def test_big_numbers_do_not_overflow():
    # 311,837 invocations x 1,800,409 ms is about 5.6e11, far above uint32's 4.29e9
    assert demand_of(311_837, 1_800_409) == 9_357_236


def test_matches_exact_reference_on_random_and_boundary_cases():
    rng = np.random.default_rng(0)
    n = rng.integers(0, 300_000, size=3000)
    dur = rng.integers(0, 1_800_000, size=3000)
    n[:200] = 60 * (np.arange(200) + 1)      # force exact boundaries: 60 * k invocations ...
    dur[:200] = 1000                         # ... of 1 second each = exactly k instances
    for concurrency in (1, 4):
        counts = np.zeros((3000, 1440), dtype=np.uint32)
        counts[:, 7] = n
        got = build_demand(counts, dur.astype(float), concurrency)[:, 7]
        want = [exact_demand(int(a), int(b), concurrency) for a, b in zip(n, dur)]
        assert got.tolist() == want


def test_concurrency_four_is_demand_over_four_rounded_up():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 500, size=(300, 1440)).astype(np.uint32)
    dur = rng.integers(0, 20_000, size=300).astype(float)
    one = build_demand(counts, dur, 1)
    four = build_demand(counts, dur, 4)
    np.testing.assert_array_equal(four, (one + 3) // 4)


# ---- output format and inputs ----
def test_output_is_signed_and_same_shape():
    counts = np.ones((3, 1440), dtype=np.uint32)
    out = build_demand(counts, np.array([10.0, 20.0, 30.0]))
    assert out.shape == counts.shape and out.dtype == np.int32    # signed: later "pool - demand" must not wrap


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 100, size=(11, 1440)).astype(np.uint32)
    dur = rng.integers(0, 5000, size=11).astype(float)
    np.testing.assert_array_equal(build_demand(counts, dur, 1, chunk_size=2), build_demand(counts, dur, 1, chunk_size=1000))


def test_bad_inputs_are_rejected():
    counts = np.ones((1, 1440), dtype=np.uint32)
    with pytest.raises(ValueError):
        build_demand(counts, np.array([np.nan]))
    with pytest.raises(ValueError):
        build_demand(counts, np.array([-5.0]))
    with pytest.raises(ValueError):
        build_demand(counts, np.array([5.0]), concurrency=0)


def test_subset_demand_reads_the_subset_files(tmp_path):
    counts = np.zeros((2, 1440), dtype=np.uint32)
    counts[0, 0], counts[1, 5] = 30, 31
    np.save(tmp_path / "subset_minutes.npy", counts)
    pd.DataFrame({"dur_ms": [2000.0, 2000.0]}).to_parquet(tmp_path / "subset_function_days.parquet")
    table, demand = subset_demand(tmp_path)
    assert len(table) == 2 and demand[0, 0] == 1 and demand[1, 5] == 2
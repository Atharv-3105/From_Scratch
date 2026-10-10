import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from traces.ingest import stream_invocations, load_invocations, N_MIN


def _write_feather(path, counts, days, batch_rows=3):
    n = counts.shape[0]
    cols = {
        "HashOwner": [f"o{i}" for i in range(n)],
        "HashApp": [f"a{i}" for i in range(n)],
        "HashFunction": [f"f{i}" for i in range(n)],
        "Trigger": ["http"] * n,
    }
    for m in range(N_MIN):
        cols[str(m + 1)] = counts[:, m].astype(np.int64)
    cols["day"] = np.asarray(days, dtype=np.int64)
    table = pa.table(cols)
    with ipc.new_file(str(path), table.schema) as w:
        for b in table.to_batches(max_chunksize=batch_rows):
            w.write_batch(b)


@pytest.fixture
def small(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(8, N_MIN))
    counts[2, 10] = 311_837          # real max in the Azure data; must survive
    days = [1, 1, 1, 2, 2, 2, 3, 3]
    src = tmp_path / "inv.feather"
    _write_feather(src, counts, days)
    return src, counts, days, tmp_path / "out"


def test_minutes_match_and_are_uint32(small):
    src, counts, days, out = small
    stream_invocations(src, out)
    keys, minutes = load_invocations(out)
    assert minutes.dtype == np.uint32
    assert minutes.shape == counts.shape
    np.testing.assert_array_equal(minutes, counts)


def test_keys_aligned_with_rows(small):
    src, counts, days, out = small
    stream_invocations(src, out)
    keys, minutes = load_invocations(out)
    assert list(keys["HashFunction"]) == [f"f{i}" for i in range(8)]
    assert list(keys["day"]) == days
    np.testing.assert_array_equal(keys["total_invocations"].to_numpy(), counts.sum(axis=1))


def test_overflow_is_an_error_not_silent_wraparound(tmp_path):
    counts = np.zeros((4, N_MIN), dtype=np.int64)
    counts[1, 5] = 2**32 + 1
    src = tmp_path / "inv.feather"
    _write_feather(src, counts, [1, 1, 1, 1])
    with pytest.raises(ValueError):
        stream_invocations(src, tmp_path / "out")
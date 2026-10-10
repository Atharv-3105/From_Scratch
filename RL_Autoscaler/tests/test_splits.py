import itertools

import numpy as np
import pandas as pd
import pytest

from traces.splits import (
    split_of, assign_splits, fingerprint, check_or_freeze,
    validate_unit_split, validate_day_split, cell_mask, group_shares,
)

CONFIG = {
    "seed": 7,
    "split_unit": "owner",
    "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
    "days": {"train": list(range(1, 11)), "val": [11, 13], "test": [12, 14]},
    "weekend_days": [6, 7, 13, 14],
    "cells": {
        "train": {"functions": "train", "days": "train"},
        "val": {"functions": "val", "days": "val"},
        "test_days": {"functions": "train", "days": "test"},
        "test_functions": {"functions": "test", "days": "train"},
        "test_both": {"functions": "test", "days": "test"},
    },
}


def make_stats(n_owners=5000, apps_per_owner=3, funcs_per_app=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for o in range(n_owners):
        for a in range(apps_per_owner):
            for f in range(funcs_per_app):
                rows.append((f"o{o}", f"o{o}-a{a}", f"o{o}-a{a}-f{f}",
                             rng.choice(["rare", "hot", "bursty", "periodic", "normal"], p=[.5, .1, .1, .2, .1]),
                             True))
    return pd.DataFrame(rows, columns=["HashOwner", "HashApp", "HashFunction", "group", "has_train"])


# ---- the assignment rule ----
def test_same_inputs_give_same_split():
    assert split_of("owner-1", 7, CONFIG["fractions"]) == split_of("owner-1", 7, CONFIG["fractions"])


def test_a_units_split_does_not_depend_on_which_other_units_exist():
    small = make_stats(n_owners=50)
    big = make_stats(n_owners=500)             # contains the same first 50 owners, plus many more
    a = assign_splits(small, CONFIG).set_index("HashFunction")["split"]
    b = assign_splits(big, CONFIG).set_index("HashFunction")["split"]
    assert (a == b.loc[a.index]).all()


def test_changing_the_seed_changes_the_split():
    stats = make_stats(n_owners=500)
    a = assign_splits(stats, CONFIG)["split"]
    b = assign_splits(stats, {**CONFIG, "seed": 8})["split"]
    assert (a != b).mean() > 0.2


def test_fractions_are_close_to_70_15_15():
    out = assign_splits(make_stats(n_owners=20000, apps_per_owner=1, funcs_per_app=1), CONFIG)
    shares = out["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.015)
    assert shares["val"] == pytest.approx(0.15, abs=0.015)
    assert shares["test"] == pytest.approx(0.15, abs=0.015)


def test_group_mix_is_similar_across_splits():
    out = assign_splits(make_stats(n_owners=8000), CONFIG)
    shares = group_shares(out)                       # rows = group, columns = split, values = share inside the split
    spread = shares.max(axis=1) - shares.min(axis=1)
    assert spread.max() < 0.03


# ---- no leakage: units are never split ----
def test_all_functions_of_an_owner_share_a_split():
    out = assign_splits(make_stats(n_owners=300), CONFIG)
    assert out.groupby("HashOwner")["split"].nunique().max() == 1
    validate_unit_split(out, CONFIG)


def test_app_unit_keeps_apps_together_but_can_separate_apps_of_one_owner():
    config = {**CONFIG, "split_unit": "app"}
    out = assign_splits(make_stats(n_owners=300, apps_per_owner=10), config)
    assert out.groupby("HashApp")["split"].nunique().max() == 1
    assert out.groupby("HashOwner")["split"].nunique().max() > 1
    validate_unit_split(out, config)


def test_functions_without_train_days_still_get_their_owners_split():
    stats = make_stats(n_owners=50)
    stats.loc[stats.index[::4], "has_train"] = False
    out = assign_splits(stats, CONFIG)
    assert out["split"].notna().all()
    assert out.groupby("HashOwner")["split"].nunique().max() == 1


def test_validate_unit_split_catches_a_leak():
    out = assign_splits(make_stats(n_owners=20), CONFIG)
    first_split = out.loc[0, "split"]
    out.loc[0, "split"] = "test" if first_split != "test" else "train"   # rows 0 and 1 share an owner
    with pytest.raises(ValueError):
        validate_unit_split(out, CONFIG)


# ---- the day split ----
def test_day_split_is_valid():
    validate_day_split(CONFIG)


@pytest.mark.parametrize("days", [
    {"train": list(range(1, 11)), "val": [10, 13], "test": [12, 14]},     # day 10 in train AND val
    {"train": list(range(2, 11)), "val": [1, 13], "test": [12, 14]},      # val day before the train days
    {"train": list(range(1, 11)), "val": [11, 12], "test": [13, 14]},     # val has no weekend day
    {"train": list(range(1, 11)), "val": [13, 14], "test": [11, 12]},     # val has no weekday
    {"train": list(range(1, 11)), "val": [11, 13], "test": [12]},         # day 14 is unused
])
def test_bad_day_splits_are_rejected(days):
    with pytest.raises(ValueError):
        validate_day_split({**CONFIG, "days": days})


# ---- the five cells ----
def grid_table():
    # every (split, day) pair appears exactly once
    return pd.DataFrame(list(itertools.product(["train", "val", "test"], range(1, 15))), columns=["split", "day"])


def test_each_cell_selects_the_right_rows():
    table = grid_table()
    both = table[cell_mask(table, "test_both", CONFIG)]
    assert set(both["split"]) == {"test"} and set(both["day"]) == {12, 14}
    days_only = table[cell_mask(table, "test_days", CONFIG)]
    assert set(days_only["split"]) == {"train"} and set(days_only["day"]) == {12, 14}
    funcs_only = table[cell_mask(table, "test_functions", CONFIG)]
    assert set(funcs_only["split"]) == {"test"} and set(funcs_only["day"]) == set(range(1, 11))


def test_cells_do_not_overlap_and_training_sees_nothing_held_out():
    table = grid_table()
    masks = {name: cell_mask(table, name, CONFIG) for name in CONFIG["cells"]}
    for a, b in itertools.combinations(masks, 2):
        assert not (masks[a] & masks[b]).any(), f"cells {a} and {b} overlap"
    train = table[masks["train"]]
    assert set(train["split"]) == {"train"}
    assert set(train["day"]).isdisjoint(CONFIG["days"]["val"] + CONFIG["days"]["test"])


# ---- freezing ----
def test_fingerprint_is_stable_and_sensitive():
    out = assign_splits(make_stats(n_owners=100), CONFIG)
    assert fingerprint(out) == fingerprint(out.copy())
    changed = out.copy()
    changed.loc[0, "split"] = "test" if changed.loc[0, "split"] != "test" else "train"
    assert fingerprint(changed) != fingerprint(out)


def test_first_run_freezes_and_later_runs_verify_or_refuse(tmp_path):
    path = tmp_path / "fingerprint.txt"
    out = assign_splits(make_stats(n_owners=100), CONFIG)
    assert check_or_freeze(out, path) == "frozen"
    assert check_or_freeze(out, path) == "verified"
    saved = path.read_text()
    other = assign_splits(make_stats(n_owners=100), {**CONFIG, "seed": 99})
    with pytest.raises(RuntimeError):
        check_or_freeze(other, path)
    assert path.read_text() == saved                  # a refused run never overwrites the frozen file
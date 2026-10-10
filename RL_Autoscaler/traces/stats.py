import numpy as np
import pandas as pd

N_MIN = 1440
FUNC_KEYS = ["HashOwner", "HashApp", "HashFunction"]


def rate_per_hour(total, first_day, last_train_day):
    # A function can only fire while it exists. We count from the first day we see it
    # (not from day 1), so a late-appearing function is not unfairly called "rare".
    alive_days = last_train_day - first_day + 1
    return total / (alive_days * 24)


def avg_day_correlation(day_counts):
    """How similar are a function's days? Average correlation over every pair of days.
    day_counts has one row per day and 1440 columns (invocations per minute).
    Returns NaN if fewer than 2 days can be compared."""
    if len(day_counts) < 2:
        return np.nan

    # Centre each day and measure its length. A day with length 0 is a flat line
    # (same count every minute), which has no correlation with anything.
    centred = day_counts - day_counts.mean(axis=1, keepdims=True)
    length = np.sqrt((centred ** 2).sum(axis=1))
    has_variation = length > 0
    if has_variation.sum() < 2:
        return np.nan

    # After dividing by its length, each row has length 1, so the dot product of two
    # rows IS their correlation. One matrix product gives every pair at once.
    unit_rows = centred[has_variation] / length[has_variation, None]
    corr_matrix = unit_rows @ unit_rows.T

    # Keep each pair once: the part above the diagonal (the diagonal is a day with itself).
    upper = np.triu_indices(len(unit_rows), k=1)
    return float(corr_matrix[upper].mean())


def hourly_variance_and_mean(day_counts):
    """For each day (row), cut the 1440 minutes into 24 one-hour blocks and return:
       - the sum over blocks of the variance of the 60 per-minute counts
       - the sum over blocks of the mean of the 60 per-minute counts
    variance_sum / mean_sum is our burst score. For steady random traffic (Poisson) the
    variance equals the mean, so the score is about 1. Spiky traffic gives a score much
    above 1. Using one-hour blocks stops a slow day/night cycle from looking like bursts."""
    hours = np.asarray(day_counts, dtype=np.float64).reshape(day_counts.shape[0], 24, 60)
    variance_sum = hours.var(axis=2, ddof=1).sum(axis=1)  # ddof=1 -> unbiased, so Poisson gives exactly 1
    mean_sum = hours.mean(axis=2).sum(axis=1)
    return variance_sum, mean_sum


def add_function_id(day_table):
    # Give every (owner, app, function) one number. sort=False keeps the order of first appearance.
    day_table = day_table.reset_index(drop=True).copy()
    day_table["func_id"] = day_table.groupby(FUNC_KEYS, sort=False).ngroup()
    return day_table


def compute_function_stats(day_table, minutes, first_train_day=1, last_train_day=10, chunk_size=20000):
    """One row per function, using ONLY the train days (no peeking at validation/test days).
    day_table: one row per function-day. minutes: matrix with the same row order."""
    day_table = add_function_id(day_table[FUNC_KEYS + ["day", "total_invocations"]])
    all_functions = day_table.drop_duplicates("func_id")[FUNC_KEYS + ["func_id"]].set_index("func_id").sort_index()

    train_rows = day_table[(day_table["day"] >= first_train_day) & (day_table["day"] <= last_train_day)]
    row_numbers = train_rows.index.to_numpy()

    active_minutes = np.zeros(len(row_numbers))
    hour_var_sum = np.zeros(len(row_numbers))
    hour_mean_sum = np.zeros(len(row_numbers))
    minute_of_day_totals = np.zeros(N_MIN)

    # Read the big matrix a chunk at a time so it never has to fit in memory.
    for start in range(0, len(row_numbers), chunk_size):
        end = start + chunk_size
        day_counts = np.asarray(minutes[row_numbers[start:end]])
        active_minutes[start:end] = (day_counts > 0).sum(axis=1)
        hour_var_sum[start:end], hour_mean_sum[start:end] = hourly_variance_and_mean(day_counts)
        minute_of_day_totals += day_counts.sum(axis=0, dtype=np.float64)

    train_rows = train_rows.assign(
        active_minutes=active_minutes, hour_var_sum=hour_var_sum, hour_mean_sum=hour_mean_sum
    )
    per_func = train_rows.groupby("func_id").agg(
        n_days=("day", "size"),
        first_day=("day", "min"),
        total=("total_invocations", "sum"),
        active_minutes=("active_minutes", "sum"),
        hour_var_sum=("hour_var_sum", "sum"),
        hour_mean_sum=("hour_mean_sum", "sum"),
    )
    alive_days = last_train_day - per_func["first_day"] + 1
    per_func["rate_per_hour"] = rate_per_hour(per_func["total"], per_func["first_day"], last_train_day)
    per_func["active_frac"] = per_func["active_minutes"] / (alive_days * N_MIN)
    # Pool variance and mean over all of a function's days, then divide. A function with
    # mean 0 would divide by zero, so those become NaN.
    per_func["burst_score"] = per_func["hour_var_sum"] / per_func["hour_mean_sum"].where(per_func["hour_mean_sum"] > 0)

    # Day-over-day similarity needs all of a function's train days together, so loop per function.
    day_corr = pd.Series(np.nan, index=per_func.index)
    for func_id, rows_of_func in train_rows.groupby("func_id").groups.items():
        if len(rows_of_func) >= 2:
            sorted_rows = np.sort(np.asarray(rows_of_func))  # sorted = faster read from the matrix
            day_corr.loc[func_id] = avg_day_correlation(np.asarray(minutes[sorted_rows], dtype=np.float64))
    per_func["day_corr"] = day_corr

    keep = ["n_days", "first_day", "total", "rate_per_hour", "active_frac", "burst_score", "day_corr"]
    stats = all_functions.join(per_func[keep])  # functions with no train days get NaN
    stats["has_train"] = stats["n_days"].notna()
    return stats.reset_index(), minute_of_day_totals


def label_functions(stats, config):
    """Add yes/no flags (a function can have several) and one 'group' per function
    that is used only for sampling."""
    limits = config["thresholds"]
    out = stats.copy()
    has_train = out["has_train"].astype(bool)

    # Comparing NaN gives False, so functions without enough data never get a flag.
    out["rare"] = has_train & (out["rate_per_hour"] < limits["rare_rate_per_hour"])
    out["hot"] = has_train & (out["rate_per_hour"] >= limits["hot_rate_per_hour"])
    out["periodic"] = has_train & (out["day_corr"] >= limits["periodic_corr"])
    out["bursty"] = has_train & (out["burst_score"] >= limits["bursty_score"])

    # Apply the priority list backwards so the FIRST name in the list is written last and wins.
    out["group"] = "normal"
    for name in reversed(config["group_priority"]):
        out.loc[out[name], "group"] = name
    out.loc[~has_train, "group"] = "unlabeled"
    return out


def sample_by_group(stats, size, min_per_group, seed, skip_groups=("unlabeled",)):
    """Pick about `size` functions, taking at least `min_per_group` from every group so small
    groups are still represented. Each picked function gets a weight = how many functions it
    stands for, so weighted totals match the real population."""
    pool = stats[~stats["group"].isin(skip_groups)]
    group_sizes = pool["group"].value_counts().sort_index()
    rng = np.random.default_rng(seed)

    picked_parts = []
    for name, group_size in group_sizes.items():
        share = round(size * group_size / group_sizes.sum())  # proportional share
        n_pick = int(min(group_size, max(min_per_group, share)))  # but not below the floor, not above the group
        chosen = rng.choice(pool.index[pool["group"] == name].to_numpy(), size=n_pick, replace=False)
        part = pool.loc[np.sort(chosen)].copy()
        part["weight"] = group_size / n_pick
        picked_parts.append(part)
    return pd.concat(picked_parts).sort_index()
import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"


def paper_facts(stats, config):
    """Text lines that compare our numbers with the Azure paper."""
    labelled = stats[stats["has_train"]]
    limits = config["thresholds"]

    # Sort functions from most to least active, then see what share the top 19% make up.
    totals = labelled["total"].sort_values(ascending=False).to_numpy()
    running_share = totals.cumsum() / totals.sum()
    top19_share = running_share[int(0.19 * len(labelled)) - 1]

    corr = labelled["day_corr"].dropna()
    lines = [
        f"functions with train-day data: {len(labelled):,} of {len(stats):,}",
        f"rare (< {limits['rare_rate_per_hour']}/hour): {100 * labelled['rare'].mean():.1f}%   [paper: ~45% of APPS at <= 1/hour; ours is per function]",
        f"top 19% of functions produce {100 * top19_share:.1f}% of invocations   [paper: top ~19% of apps -> ~99.6%]",
        f"periodic (corr >= {limits['periodic_corr']}): {100 * labelled['periodic'].mean():.1f}% of functions, "
        f"{100 * (corr >= limits['periodic_corr']).mean():.1f}% of those with a defined correlation   [paper: ~20% of apps highly regular]",
        f"hot: {100 * labelled['hot'].mean():.1f}%   bursty: {100 * labelled['bursty'].mean():.1f}%",
        "note: paper numbers are per app and use true inter-arrival times; ours are per function and minute-binned.",
    ]
    return "\n".join(lines)


def plot_eda(stats, minute_of_day_totals, config, out_png):
    labelled = stats[stats["has_train"]]
    limits = config["thresholds"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # 1. How many invocations per hour does each function get? (log scale on both axes)
    rates = labelled["rate_per_hour"][labelled["rate_per_hour"] > 0]
    bins = np.logspace(np.log10(rates.min()), np.log10(rates.max()), 50)
    axes[0, 0].hist(rates, bins=bins, color=BLUE)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="mean invocations / hour (alive days)", ylabel="functions",
                   title="Rate per function (log-log)")
    axes[0, 0].axvline(limits["rare_rate_per_hour"], color=ORANGE, ls="--", label="rare cut")
    axes[0, 0].axvline(limits["hot_rate_per_hour"], color=GREEN, ls="--", label="hot cut")
    axes[0, 0].legend()

    # 2. How concentrated is the traffic? Share of all invocations made by the busiest functions.
    totals = np.sort(labelled["total"].to_numpy())[::-1]
    running_share = totals.cumsum() / totals.sum()
    fraction_of_functions = np.arange(1, len(totals) + 1) / len(totals)
    axes[0, 1].plot(fraction_of_functions, running_share, color=BLUE)
    axes[0, 1].set(xscale="log", xlabel="fraction of functions (most active first)", ylabel="share of invocations",
                   title="Invocation concentration")
    axes[0, 1].axvline(0.19, color=GREY, ls=":")

    # 3. Traffic by hour of day (sum the 60 minutes of each hour)
    per_hour = minute_of_day_totals.reshape(24, 60).sum(axis=1)
    axes[0, 2].bar(range(24), per_hour / per_hour.sum(), color=BLUE)
    axes[0, 2].set(xlabel="hour of day", ylabel="share of invocations", title="Diurnal pattern (train days)")

    # 4. Do functions repeat the same pattern every day?
    corr = labelled["day_corr"].dropna()
    axes[1, 0].hist(corr, bins=40, color=BLUE)
    axes[1, 0].axvline(limits["periodic_corr"], color=ORANGE, ls="--", label="periodic cut")
    axes[1, 0].set(yscale="log", xlabel="average day-over-day correlation (per-minute counts)", ylabel="functions",
                   title=f"Predictability (defined for {len(corr):,} functions)")
    axes[1, 0].legend()

    # 5. How spiky is the traffic compared with steady random traffic (score 1)?
    scores = labelled["burst_score"].dropna()
    scores = scores[scores > 0]
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 50)
    axes[1, 1].hist(scores, bins=bins, color=BLUE)
    axes[1, 1].axvline(1.0, color=GREY, ls=":", label="steady random traffic = 1")
    axes[1, 1].axvline(limits["bursty_score"], color=ORANGE, ls="--", label="bursty cut")
    axes[1, 1].set(xscale="log", yscale="log", xlabel="within-hour variance / mean", ylabel="functions",
                   title="Burstiness (replaces inter-arrival CV)")
    axes[1, 1].legend()

    # 6. How many functions per flag (overlapping) and per group (each function in exactly one)
    flag_counts = [(name, int(labelled[name].sum())) for name in ["rare", "hot", "periodic", "bursty"]]
    group_counts = stats["group"].value_counts()
    labels = [f"flag: {name}" for name, _ in flag_counts] + [f"group: {name}" for name in group_counts.index]
    values = [count for _, count in flag_counts] + group_counts.tolist()
    colors = [BLUE] * len(flag_counts) + [ORANGE] * len(group_counts)
    axes[1, 2].barh(labels[::-1], values[::-1], color=colors[::-1])
    axes[1, 2].set(xlabel="functions", title="Flags (can overlap) and groups (exclusive)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
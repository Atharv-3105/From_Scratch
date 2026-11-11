import matplotlib 
import matplotlib.pyplot as plt 
matplotlib.use("Agg")
from sim.metrics import pareto_front

COLORS = {"fixed": "#0072B2", "hybrid": "#D55E00", "hpa": "#009E73", "seasonal": "#E69F00"}
NAMES = {"fixed": "fixed keep-alive", "hybrid": "hybrid histogram", "hpa": "HPA-style", "seasonal": "seasonal"}

def plot_pareto(sweep, cell, out_png, chosen = None):
    rows = sweep[sweep["cell"] == cell]
    fig, axes = plt.subplots(1, 2, figsize = (13, 5))
    for axis, column, label in zip(axes, ["cold_pct_mean", "cold_pct_p75"], ["cold starts", "mean over fractions(%)", "cold starts, 75th percentile over functions(%)"]):
        for method, part in rows.groupby("method"):
            axis.scatter(part["waste_vs_k10"], part[column], s = 18, color = COLORS[method], alpha = 0.5)
            front = pareto_front(part["waste_vs_k10"], part[column])
            ordered = part.iloc[front]
            axis.plot(ordered["waste_vs_k10"], ordered[column], color = COLORS[method], label = NAMES[method], lw = 1.8)
        
        if chosen is not None:
            picked = chosen[chosen["cell"] == cell]
            axis.scatter(picked["waste_vs_k10"], picked[column], s = 90, facecolors = "none", edgecolors = "black", lw = 1.5, label = "chosen(best reward on validation)")
            axis.axvline(1.0, color = "#999999", ls = ":")
            axis.set(xscale = "log", xlabel = "wasted memory(x the 10-minute keep-alive)")
        
        axes[0].legend(fontsize = 8)
        axes[0].set_title(f"Baselines on the {cell} cell")
        fig.tight_layout()
        fig.savefig(out_png, dpi = 130)
        plt.close(fig)
        
        
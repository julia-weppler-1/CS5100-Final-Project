import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from stage2_reconstruction import load_stage1

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "figure.dpi"      : 150,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})

C_GA_15   = "#2166ac"   # dark blue, GA 15km
C_RAND_15 = "#92c5de"   # light blue, random 15km
C_GA_25   = "#d6604d"   # dark red, GA 25km
C_RAND_25 = "#f4a582"   # light red, random 25km
C_SENSOR  = "#e31a1c"   # red = selected sensors
C_NONSENS = "#bdbdbd"   # grey = non-selected cells

BUDGET_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BUDGET_LABELS    = ["30%", "40%", "50%", "60%", "70%", "80%"]

# values input from stage 2
RANDOM_BASELINES = {
    (15, 0.3): 1.6524,
    (15, 0.4): 1.6946,
    (15, 0.5): 1.6716,
    (15, 0.6): 1.6966,
    (15, 0.7): 1.5739,
    (15, 0.8): 1.5774,
    (25, 0.3): 1.4847,
    (25, 0.4): 1.5215,
    (25, 0.5): 1.6366,
    (25, 0.6): 1.5714,
    (25, 0.7): 1.5707,
    (25, 0.8): 1.5746,
}

def load_ga_results(path: str = "ga_results_multiseed.npz") -> list:
    raw = np.load(path, allow_pickle=True)
    import re
    pattern = re.compile(r"^cfg(\d+)_run0_(.+)$")  # take run 0 only
    store = {}
    for k in raw.files:
        m = pattern.match(k)
        if not m:
            continue
        cfg, field = int(m.group(1)), m.group(2)
        val = raw[k]
        if val.ndim == 0:
            val = val.item()
        store.setdefault(cfg, {})[field] = val
    return [store[i] for i in sorted(store)]

def plot_rmse_vs_budget(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "Reconstruction RMSE vs Sensor Budget; GA-Optimized vs Random Placement",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, km in zip(axes, [15, 25]):
        runs = [r for r in results if r["cell_size_km"] == km]
        runs.sort(key=lambda r: r["budget_frac"])

        budgets   = [r["budget_frac"] * 100 for r in runs]
        ga_rmse   = [r["best_rmse"]          for r in runs]
        rand_rmse = [RANDOM_BASELINES[(km, r["budget_frac"])] for r in runs]
        improvement = [(rand - ga) / rand * 100
                       for rand, ga in zip(rand_rmse, ga_rmse)]

        c_ga   = C_GA_15   if km == 15 else C_GA_25
        c_rand = C_RAND_15 if km == 15 else C_RAND_25

        ax.plot(budgets, rand_rmse, "o--", color=c_rand, lw=2,
                ms=7, label="Random placement (mean)")
        ax.plot(budgets, ga_rmse,   "o-",  color=c_ga,   lw=2.5,
                ms=8, label="GA-optimised placement")

        ax.fill_between(budgets, ga_rmse, rand_rmse,
                        alpha=0.12, color=c_ga)

        for x, ga, imp in zip(budgets, ga_rmse, improvement):
            ax.annotate(f"−{imp:.0f}%",
                        xy=(x, ga),
                        xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=8.5, color=c_ga, fontweight="bold")

        N = runs[0]["N"]
        ax.set_title(f"{km}km cells  (N = {N} locations)")
        ax.set_xlabel("Sensor budget (% of all cells)")
        ax.set_ylabel("Test RMSE (mg/L)")
        ax.set_xticks(budgets)
        ax.set_xticklabels(BUDGET_LABELS)
        ax.legend(loc="lower right")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = "fig1_rmse_vs_budget.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results: list):
    fig, axes = plt.subplots(2, 6, figsize=(18, 7), sharey="row")
    fig.suptitle(
        "Best and Mean RMSE per Generation",
        fontsize=14, fontweight="bold"
    )

    row_labels = {15: "15km cells", 25: "25km cells"}

    for row_idx, km in enumerate([15, 25]):
        runs = [r for r in results if r["cell_size_km"] == km]
        runs.sort(key=lambda r: r["budget_frac"])
        c_ga = C_GA_15 if km == 15 else C_GA_25

        for col_idx, r in enumerate(runs):
            ax  = axes[row_idx][col_idx]
            gen = np.arange(len(r["history_best"]))

            ax.plot(gen, r["history_mean"], color="#aaaaaa", lw=1.2,
                    label="Mean RMSE", zorder=1)
            ax.plot(gen, r["history_best"], color=c_ga,     lw=2,
                    label="Best RMSE", zorder=2)

            # mark the final best with a star
            ax.scatter([gen[-1]], [r["history_best"][-1]],
                       marker="*", s=120, color=c_ga, zorder=3)

            frac  = r["budget_frac"]
            p     = r["p"]
            N     = r["N"]
            title = f"{frac:.0%} budget  (p={p}/{N})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Generation", fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[km]}\nRMSE (mg/L)", fontsize=9)

            ax.annotate(f"{r['history_best'][-1]:.3f}",
                        xy=(gen[-1], r["history_best"][-1]),
                        xytext=(-5, 8), textcoords="offset points",
                        fontsize=8, color=c_ga, fontweight="bold")

            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = "fig2_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_maps(results: list):
    SHOW_BUDGETS = [0.3, 0.7]  

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "GA-Selected Sensor Locations at 30% and 70% Budget\n",
        fontsize=14, fontweight="bold"
    )

    for row_idx, km in enumerate([15, 25]):
        data = load_stage1(km)
        cell_ids = data["cell_ids"]

        cx_arr = np.array([int(float(cid.split("_")[0])) for cid in cell_ids])
        cy_arr = np.array([int(float(cid.split("_")[1])) for cid in cell_ids])

        mean_do = np.nanmean(data["X_train_full"], axis=0)  

        vmin, vmax = np.nanpercentile(mean_do, [5, 95])
        norm  = Normalize(vmin=vmin, vmax=vmax)
        cmap  = plt.cm.YlOrRd

        for col_idx, frac in enumerate(SHOW_BUDGETS):
            ax = axes[row_idx][col_idx]

            run = next((r for r in results
                        if r["cell_size_km"] == km
                        and abs(r["budget_frac"] - frac) < 0.01), None)
            if run is None:
                ax.set_visible(False)
                continue

            sensor_idx = run["best_indices"]
            sensor_set = set(sensor_idx.tolist())

            for i, (cx, cy) in enumerate(zip(cx_arr, cy_arr)):
                color = cmap(norm(mean_do[i]))
                ax.add_patch(plt.Rectangle(
                    (cx - 0.45, cy - 0.45), 0.9, 0.9,
                    facecolor=color, edgecolor="white", linewidth=0.3
                ))

            for idx in sensor_idx:
                cx, cy = cx_arr[idx], cy_arr[idx]
                ax.add_patch(plt.Rectangle(
                    (cx - 0.45, cy - 0.45), 0.9, 0.9,
                    facecolor="none",
                    edgecolor=C_SENSOR,
                    linewidth=2.5
                ))
                ax.plot(cx, cy, "o", color=C_SENSOR,
                        ms=4, zorder=5)

            ax.set_xlim(cx_arr.min() - 1, cx_arr.max() + 1)
            ax.set_ylim(cy_arr.min() - 1, cy_arr.max() + 1)
            ax.set_aspect("equal")
            ax.set_title(
                f"{km}km cells — {frac:.0%} budget  "
                f"(p={run['p']}/{run['N']} sensors)\n"
                f"RMSE = {run['best_rmse']:.3f} mg/L",
                fontsize=10
            )
            ax.set_aspect("equal")
            ax.tick_params(axis="both", labelsize=7)
            ax.set_xlabel("Grid column (cx)", fontsize=9)
            ax.set_ylabel("Grid row (cy)", fontsize=9)

            sensor_patch = mpatches.Patch(
                facecolor="none", edgecolor=C_SENSOR,
                linewidth=2, label=f"Selected sensors (n={run['p']})"
            )
            ax.legend(handles=[sensor_patch], loc="lower left",
                      fontsize=8, framealpha=0.8)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[row_idx, :], shrink=0.6,
                            pad=0.02, aspect=20)
        cbar.set_label("Mean DO (mg/L)", fontsize=9)

    plt.tight_layout()
    path = "fig3_sensor_maps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_rmse(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Monthly Test RMSE Distribution by Sensor Budget",
        fontsize=14, fontweight="bold"
    )

    for ax, km in zip(axes, [15, 25]):
        runs  = [r for r in results if r["cell_size_km"] == km]
        runs.sort(key=lambda r: r["budget_frac"])

        c_ga    = C_GA_15   if km == 15 else C_GA_25
        c_light = C_RAND_15 if km == 15 else C_RAND_25

        monthly_data = []
        tick_labels  = []
        for r in runs:
            pm = r["per_month_rmse"]
            if len(pm) > 0:
                monthly_data.append(pm)
            else:
                monthly_data.append([np.nan])
            tick_labels.append(
                f"{r['budget_frac']:.0%}\n(p={r['p']})"
            )

        bp = ax.boxplot(
            monthly_data,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=c_ga, linewidth=1.5),
            capprops=dict(color=c_ga, linewidth=1.5),
            flierprops=dict(marker="o", color=c_ga, alpha=0.5,
                            markersize=5),
            boxprops=dict(facecolor=c_light, edgecolor=c_ga,
                          linewidth=1.5),
        )

        rng = np.random.default_rng(0)
        for i, pm in enumerate(monthly_data):
            pm_clean = [v for v in pm if np.isfinite(v)]
            if pm_clean:
                jitter = rng.uniform(-0.15, 0.15, size=len(pm_clean))
                ax.scatter(np.full(len(pm_clean), i + 1) + jitter,
                           pm_clean, color=c_ga, alpha=0.7,
                           s=30, zorder=3)

        for i, pm in enumerate(monthly_data):
            pm_clean = [v for v in pm if np.isfinite(v)]
            if pm_clean:
                med = np.median(pm_clean)
                ax.text(i + 1, med + 0.01, f"{med:.3f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color=c_ga, fontweight="bold")

        ax.set_title(f"{km}km cells  (N = {runs[0]['N']} locations)")
        ax.set_xlabel("Sensor budget")
        ax.set_ylabel("Monthly test RMSE (mg/L)")
        ax.set_xticks(range(1, len(runs) + 1))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = "fig4_monthly_rmse.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    results = load_ga_results("ga_results_multiseed.npz")
    print(results[0].keys())    
    plot_rmse_vs_budget(results)
    plot_convergence(results)
    plot_sensor_maps(results)
    plot_monthly_rmse(results)

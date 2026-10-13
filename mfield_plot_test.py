import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data:
#df = pd.read_csv("shots_only_raw_plotting.csv")
df = pd.read_csv("shot_study_raw_plotting_2.csv")

# Reference energy (Hartree):
E_ref = -3.669832005739848

df["deviation"] = df["energy"] - E_ref

# Statistics:
stats = df.groupby("shots").agg(
    mean = ("energy", "mean"),
    MD = ("deviation", "mean"),
    MAD = ("deviation", lambda x: x.abs().mean()),
    SD = ("deviation", "std"),
).reset_index()
stats.to_csv("shot_study_stats.csv", index=False)
print(stats)


# Plot:
plt.rcParams["font.size"] = 14

fig, ax = plt.subplots(figsize=(12, 7))
width = 0.17
rng = np.random.default_rng(0)  # only used to spread out the points horizontally

for i, shots in enumerate(stats["shots"]):
    row = stats.iloc[i]
    first = (i == 0)  # only add legend labels once


    ax.bar(i - 0.32, row["MD"], width, color="#1f22cc", edgecolor="black",
        label="Mean deviation (MD)" if first else None)
    ax.bar(i - 0.13, row["MAD"], width, color="#b3adff", edgecolor="black", hatch="//",
        label="Mean absolute deviation (MAD)" if first else None)
    ax.bar(i + 0.06, row["SD"], width, color="#5ce8f0", edgecolor="black", hatch="..",
        label="Standard deviation (SD)" if first else None)

    # Print the std value above the SD bar
    ax.annotate(f"{row['SD']:.5f}", xy=(i + 0.06, row["SD"]), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", rotation=90)

    deviations = df[df["shots"] == shots]["deviation"]
    ax.scatter(i + 0.32 + rng.uniform(-0.09, 0.09, len(deviations)), deviations,
            s=32, facecolors="#a9c4e8", edgecolors="#0d2a5c", linewidths=1.0, alpha=0.8, zorder=3,
            label="Individual deviations" if first else None)

    # Separator between shot counts
    ax.axvline(i + 0.5, color="#8aa8c4", linestyle="-.", linewidth=1.2)

ax.axhline(0, color="#0d2a5c", linestyle="--", linewidth=1.5)
# Leave some space above and below the data (and always show 0)
low = min(0, df["deviation"].min(), stats["MD"].min())
high = max(0, df["deviation"].max(), stats["MAD"].max(), stats["SD"].max())
pad = 0.05 * (high - low)
ax.set_ylim(low - pad, high + pad)

ax.set_xticks(np.arange(len(stats["shots"])))
ax.set_xticklabels(stats["shots"], rotation=25, ha="right")
ax.set_xlim(-0.5, len(stats["shots"]) - 0.5)
ax.set_xlabel("Number of shots")
ax.set_ylabel("Deviation from classically calculated energy (Eh)")

ax.grid(axis="y", color="#dddddd")
ax.minorticks_on()
ax.tick_params(axis="x", which="minor", bottom=False)
ax.grid(axis="y", which="minor", color="#eeeeee")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

n_runs = df.groupby("shots").size().max()
ax.text(1.0, -0.13, f"N = {n_runs} runs per shot count", transform=ax.transAxes,
        ha="right", va="top", color="#444444")

handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 3, 0]  # bars first, points last
ax.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center",
        bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.9, edgecolor="none",
            ncol=2, columnspacing=1.2, handlelength=1.8, labelspacing=0.6,
            borderpad=0.7)

fig.tight_layout()
#fig.savefig("shots_only.png", dpi=300)
fig.savefig("shots_device.png", dpi=300)

plt.show()

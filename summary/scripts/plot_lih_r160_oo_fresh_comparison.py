"""Plot finite-shot OO optimizer trajectories and paired statistics."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


LABELS = {"seq": "OO(seq)-UCCD(seq)", "lbfgs": "OO(L-BFGS)-UCCD(seq)"}
COLORS = {"seq": "#2ca02c", "lbfgs": "#e377c2"}
MARKERS = {"seq": "^", "lbfgs": "v"}


def trajectory_matrix(runs):
    maximum = max(len(run["energies_ha"]) for run in runs)
    matrix = np.full((len(runs), maximum), np.nan)
    for row, run in enumerate(runs):
        matrix[row, :len(run["energies_ha"])] = run["energies_ha"]
    return matrix


parser = argparse.ArgumentParser(description="Plot LiH R=1.6 finite-shot OO comparison")
parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_r160_oo_sampling_study.json"))
parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "images", "lih_r160_oo_sampling_study.png"))
args = parser.parse_args()

with open(args.input, encoding="utf-8") as handle:
    data = json.load(handle)
finite = data["finite_shot"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.30})

# Panel 1: every raw trajectory, plus mean and 95% CI only when multiple seeds exist.
for key in ("seq", "lbfgs"):
    runs = finite.get(key, [])
    if not runs:
        continue
    matrix = trajectory_matrix(runs)
    x = np.arange(1, matrix.shape[1] + 1)
    for row in matrix:
        axes[0].plot(x, row, color=COLORS[key], alpha=0.22, linewidth=1.0)
    mean = np.nanmean(matrix, axis=0)
    label = LABELS[key] if len(runs) == 1 else f"{LABELS[key]} mean"
    axes[0].plot(x, mean, color=COLORS[key], marker=MARKERS[key], linewidth=2.2, label=label)
    if len(runs) > 1:
        count = np.sum(~np.isnan(matrix), axis=0)
        standard_error = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(count)
        axes[0].fill_between(x, mean - 1.96 * standard_error, mean + 1.96 * standard_error, color=COLORS[key], alpha=0.16)
axes[0].set_title("Raw finite-shot trajectories")
axes[0].set_xlabel("Macro iteration")
axes[0].set_ylabel("Measured energy (Ha)")
axes[0].legend()

# Panel 2: best-so-far trajectories. This separates discovery from noisy remeasurement.
for key in ("seq", "lbfgs"):
    runs = finite.get(key, [])
    if not runs:
        continue
    matrix = trajectory_matrix(runs)
    best = np.minimum.accumulate(matrix, axis=1)
    x = np.arange(1, matrix.shape[1] + 1)
    for row in best:
        axes[1].plot(x, row, color=COLORS[key], alpha=0.22, linewidth=1.0)
    mean = np.nanmean(best, axis=0)
    axes[1].plot(x, mean, color=COLORS[key], marker=MARKERS[key], linewidth=2.2, label=LABELS[key])
    if len(runs) > 1:
        count = np.sum(~np.isnan(best), axis=0)
        standard_error = np.nanstd(best, axis=0, ddof=1) / np.sqrt(count)
        axes[1].fill_between(x, mean - 1.96 * standard_error, mean + 1.96 * standard_error, color=COLORS[key], alpha=0.16)
axes[1].set_title("Best measured energy so far")
axes[1].set_xlabel("Macro iteration")
axes[1].set_ylabel("Best-so-far energy (Ha)")
axes[1].legend()

# Panel 3: paired final difference by seed.
seq_by_seed = {run["seed"]: run["final_energy_ha"] for run in finite.get("seq", [])}
lbfgs_by_seed = {run["seed"]: run["final_energy_ha"] for run in finite.get("lbfgs", [])}
paired_seeds = sorted(set(seq_by_seed) & set(lbfgs_by_seed))
if paired_seeds:
    differences = np.array([(seq_by_seed[seed] - lbfgs_by_seed[seed]) * 1000 for seed in paired_seeds])
    positions = np.arange(len(paired_seeds))
    axes[2].scatter(positions, differences, color="#1f77b4", s=55, zorder=3)
    axes[2].axhline(0.0, color="black", linewidth=1.2)
    axes[2].set_xticks(positions, [str(seed) for seed in paired_seeds], rotation=30)
    if len(differences) > 1:
        mean_difference = differences.mean()
        ci = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
        axes[2].axhline(mean_difference, color="#d62728", linewidth=2, label=f"mean = {mean_difference:.2f} mHa")
        axes[2].fill_between([-0.5, len(differences) - 0.5], mean_difference - ci, mean_difference + ci, color="#d62728", alpha=0.15, label="95% CI")
        axes[2].legend()
    else:
        axes[2].set_title("Paired final difference (1 seed only)")
else:
    axes[2].text(0.5, 0.5, "No completed seed pairs", ha="center", va="center")
axes[2].set_title("Paired final-energy difference")
axes[2].set_xlabel("Paired seed")
axes[2].set_ylabel(r"$E_\mathrm{seq}-E_\mathrm{L-BFGS}$ (mHa)")

for axis in axes:
    axis.grid(True, linestyle="--", alpha=0.4)

completed_pairs = len(paired_seeds)
fig.suptitle(
    rf"LiH $R=1.6$ Å, finite-shot RDM: {data['shots']:,} shots, {completed_pairs} completed seed pair(s)",
    fontsize=14,
)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
fig.savefig(args.output, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {os.path.abspath(args.output)}")
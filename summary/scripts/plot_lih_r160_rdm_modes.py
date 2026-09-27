"""Visualize statevector vs finite-shot RDM energy estimates at LiH R=1.6 Å."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Plot LiH R=1.6 RDM-mode comparison")
parser.add_argument(
    "--input",
    default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_r160_rdm_mode_comparison.json"),
)
parser.add_argument(
    "--output",
    default=os.path.join(os.path.dirname(__file__), "..", "images", "lih_r160_rdm_mode_comparison.png"),
)
args = parser.parse_args()

with open(args.input, encoding="utf-8") as handle:
    data = json.load(handle)

trials = data["finite_shot_trials"]
energies = np.array([trial["energy_ha"] for trial in trials])
errors_mha = 1000.0 * np.array([trial["error_from_statevector_ha"] for trial in trials])
exact = data["statevector_energy_ha"]
mean = data["finite_shot_mean_energy_ha"]
std = data["finite_shot_standard_deviation_ha"]
shots = data["shots_per_pauli_word"]
x = np.arange(1, len(trials) + 1)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.32})

ax_top.axhline(exact, color="black", linewidth=2, label=f"Statevector: {exact:.8f} Ha")
ax_top.errorbar(
    x,
    energies,
    yerr=np.full_like(energies, std),
    fmt="o",
    capsize=4,
    color="#1f77b4",
    label=f"Finite-shot RDM ({shots:,} shots / Pauli word)",
)
ax_top.axhline(mean, color="#d62728", linestyle="--", label=f"Finite-shot mean: {mean:.8f} Ha")
ax_top.fill_between([0.5, len(trials) + 0.5], mean - std, mean + std, color="#d62728", alpha=0.12)
ax_top.set_xlim(0.5, len(trials) + 0.5)
ax_top.set_xticks(x)
ax_top.set_ylabel("Energy (Ha)")
ax_top.set_title(r"LiH $R=1.6$ Å: Statevector vs Finite-Shot RDM")
ax_top.legend(loc="best")

colors = ["#d62728" if error > 0 else "#1f77b4" for error in errors_mha]
ax_bottom.bar(x, errors_mha, color=colors, alpha=0.85)
ax_bottom.axhline(0.0, color="black", linewidth=1.5)
ax_bottom.axhline(1000.0 * std, color="#666666", linestyle=":", label=r"$+1\sigma$")
ax_bottom.axhline(-1000.0 * std, color="#666666", linestyle=":", label=r"$-1\sigma$")
ax_bottom.set_xticks(x, [f"seed {trial['seed']}" for trial in trials], rotation=25, ha="right")
ax_bottom.set_ylabel("Error from statevector (mHa)")
ax_bottom.set_xlabel("Independent finite-shot trial")
ax_bottom.legend(loc="best")

fig.text(
    0.5,
    0.01,
    f"Mean error = {(mean - exact) * 1000:.3f} mHa, sample standard deviation = {std * 1000:.3f} mHa",
    ha="center",
)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
fig.savefig(args.output, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {os.path.abspath(args.output)}")
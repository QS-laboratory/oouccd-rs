"""Compare OO(seq)-UCCD(seq) and OO(lbfgs)-UCCD(seq) at LiH R=1.6 Å."""

import argparse
import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


METHODS = {
    "OO(seq)-UCCD(seq)": {
        "default_db": os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes", "task_31.db"),
        "color": "#2ca02c",
        "marker": "^",
    },
    "OO(lbfgs)-UCCD(seq)": {
        "default_db": os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes", "task_45.db"),
        "color": "#e377c2",
        "marker": "v",
    },
}


def load_history(db_path, expected_method, bond_length=1.6):
    connection = sqlite3.connect(db_path)
    runs = connection.execute(
        "SELECT id, conditions_json, final_energy FROM runs WHERE is_completed = 1"
    ).fetchall()
    for run_id, conditions_json, final_energy in runs:
        conditions = json.loads(conditions_json)
        if conditions.get("method") != expected_method:
            continue
        geometry = conditions.get("geometry", "")
        if f"{bond_length:.5f}" not in geometry:
            continue
        rows = connection.execute(
            "SELECT macro_idx, energy FROM macro_iterations "
            "WHERE run_id = ? ORDER BY macro_idx",
            (run_id,),
        ).fetchall()
        connection.close()
        # Some runs write the final macro iteration twice. Keep one energy per index.
        history = {}
        for macro_idx, energy in rows:
            history[macro_idx] = energy
        return np.array(sorted(history)), np.array([history[index] for index in sorted(history)]), final_energy
    connection.close()
    raise RuntimeError(f"No completed R={bond_length} run for {expected_method} in {db_path}")


parser = argparse.ArgumentParser(description="Compare LiH R=1.6 OO optimizers")
parser.add_argument("--seq-db", default=METHODS["OO(seq)-UCCD(seq)"]["default_db"])
parser.add_argument("--lbfgs-db", default=METHODS["OO(lbfgs)-UCCD(seq)"]["default_db"])
parser.add_argument(
    "--output",
    default=os.path.join(
        os.path.dirname(__file__), "..", "images", "lih_r160_oo_seq_vs_lbfgs.png"
    ),
)
args = parser.parse_args()

paths = {
    "OO(seq)-UCCD(seq)": args.seq_db,
    "OO(lbfgs)-UCCD(seq)": args.lbfgs_db,
}
data = {
    method: load_history(paths[method], method)
    for method in METHODS
}
reference = min(final_energy for _, _, final_energy in data.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw={"hspace": 0.10})

for method, style in METHODS.items():
    iterations, energies, final_energy = data[method]
    plot_style = dict(
        color=style["color"], marker=style["marker"], markersize=7,
        linewidth=1.8, label=method,
    )
    axes[0].plot(iterations, energies, **plot_style)

    error = np.abs(energies - reference)
    positive = error > 0
    axes[1].plot(iterations[positive], error[positive], **plot_style)

    delta = np.abs(np.diff(energies))
    positive_delta = delta > 0
    axes[2].plot(iterations[1:][positive_delta], delta[positive_delta], **plot_style)

axes[0].set_ylabel("Energy (Ha)")
axes[0].set_title(r"LiH $R=1.6$ Å: Orbital Optimizer Comparison")
axes[0].legend(loc="best")

axes[1].set_yscale("log")
axes[1].set_ylabel(r"$|E-E_\mathrm{best}|$ (Ha)")
axes[1].axhline(1e-8, color="black", linestyle=":", linewidth=1.2, label="1e-8")

axes[2].set_yscale("log")
axes[2].set_ylabel(r"$|E_k-E_{k-1}|$ (Ha)")
axes[2].set_xlabel("Macro iteration")
axes[2].axhline(1e-8, color="black", linestyle=":", linewidth=1.2)
axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

seq_final = data["OO(seq)-UCCD(seq)"][2]
lbfgs_final = data["OO(lbfgs)-UCCD(seq)"][2]
fig.text(
    0.5,
    0.01,
    f"Final energy difference (seq − L-BFGS) = {(seq_final - lbfgs_final) * 1e9:.3f} nHa",
    ha="center",
)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
fig.savefig(args.output, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {os.path.abspath(args.output)}")
print(f"OO(seq) final:   {seq_final:.12f} Ha")
print(f"OO(L-BFGS) final:{lbfgs_final:.12f} Ha")
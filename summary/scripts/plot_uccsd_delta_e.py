"""
LiH Delta E Decay Plots

Generates one figure per method, with Delta E decay curves across all bond
lengths.  The historical UCCSD output path is kept as the default for
backward compatibility.
"""

import argparse
import sqlite3
import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.linewidth":      1.0,
    "axes.labelsize":      12,
    "axes.titlesize":      12,
    "legend.fontsize":     10,
    "legend.framealpha":   0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.alpha":          0.5,
    "grid.color":          "#aaaaaa",
    "figure.dpi":          150,
})

METHOD_ORDER = [
    "OO(seq)-UCCD(seq)",
    "OO(lbfgs)-UCCD(seq)",
    "UCCD(seq)",
    "UCCSD(seq)",
]

METHOD_LABELS = {
    "OO(seq)-UCCD(seq)": "OO(seq)-UCCD(seq)",
    "OO(lbfgs)-UCCD(seq)": "OO(lbfgs)-UCCD(seq)",
    "UCCD(seq)": "UCCD(seq)",
    "UCCSD(seq)": "UCCSD(seq)",
}


def method_slug(method):
    """Return the filename component used for a method's combined plot."""
    return method.lower().replace("(", "_").replace(")", "").replace("-", "_")


def plot_method(method, method_data, output_path):
    """Save the all-bond-length Delta E decay figure for one method."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_rs = sorted(method_data)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_rs)))

    for idx, r_key in enumerate(sorted_rs):
        raw_energies = method_data[r_key]

        delta_e_x = []
        delta_e_y = []
        for i in range(1, len(raw_energies)):
            delta_e = abs(raw_energies[i - 1] - raw_energies[i])
            if delta_e > 0:
                delta_e_x.append(i + 1)
                delta_e_y.append(delta_e)

        if delta_e_x:
            ax.plot(
                delta_e_x, delta_e_y,
                marker=".", markersize=5,
                color=colors[idx],
                label=f"R={r_key:.1f}",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta E$ (Ha)")
    ax.set_title(f"{METHOD_LABELS.get(method, method)} $\Delta E$ Decay by Bond Length")
    ax.axhline(1e-8, color="red", linestyle=":", label="Threshold (1e-8)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(output_path)}")


parser = argparse.ArgumentParser(description="LiH Delta E Plotter")
parser.add_argument("--results-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes"),
                    help="Directory containing task_*.db files")
parser.add_argument("--output", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "images", "lih_uccsd_delta_e_all.png"),
                    help="Output image path when --method is specified")
parser.add_argument("--method", type=str,
                    help="Generate only this method's plot")
args = parser.parse_args()

db_files = sorted(glob.glob(os.path.join(args.results_dir, "task_*.db")))

# Dictionary to hold energies: data[method][R_key] = [list of energies]
data = {}

for db_path in db_files:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        runs = cur.execute("SELECT id, conditions_json FROM runs").fetchall()
        for run_id, cond_json in runs:
            cond = json.loads(cond_json)
            method = cond.get("method", "unknown")
            
            geom = cond.get("geometry", "")
            m = re.search(r"H\s+0\s+0\s+([\d.]+)", geom)
            if not m:
                continue
                
            R_val = float(m.group(1))
            R_key = round(R_val, 5)
            
            iters = cur.execute("SELECT macro_idx, energy FROM macro_iterations WHERE run_id = ? ORDER BY macro_idx", (run_id,)).fetchall()
            
            if iters:
                if method not in data:
                    data[method] = {}
                data[method][R_key] = [e for idx, e in iters]
                
        conn.close()
    except Exception as e:
        print(f"[warn] {db_path}: {e}")

if args.method:
    if args.method not in data:
        raise SystemExit(f"No macro-iteration data found for method: {args.method}")
    plot_method(args.method, data[args.method], args.output)
else:
    output_dir = os.path.dirname(args.output)
    for method in METHOD_ORDER:
        if method not in data:
            print(f"[warn] No macro-iteration data found for method: {method}")
            continue

        if method == "UCCSD(seq)":
            output_path = args.output
        else:
            output_path = os.path.join(
                output_dir,
                f"lih_{method_slug(method)}_delta_e_all.png",
            )
        plot_method(method, data[method], output_path)

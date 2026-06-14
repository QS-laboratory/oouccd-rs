"""
UCCSD Delta E Decay Plot
Generates a single figure containing the Delta E decay curves for UCCSD(seq) across all R values.
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

parser = argparse.ArgumentParser(description="UCCSD Delta E Plotter")
parser.add_argument("--results-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes"),
                    help="Directory containing task_*.db files")
parser.add_argument("--output", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "images", "lih_uccsd_delta_e_all.png"),
                    help="Output image path")
args = parser.parse_args()

db_files = sorted(glob.glob(os.path.join(args.results_dir, "task_*.db")))

# Dictionary to hold energies: data[R_key] = [list of energies]
data = {}

for db_path in db_files:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        runs = cur.execute("SELECT id, conditions_json FROM runs").fetchall()
        for run_id, cond_json in runs:
            cond = json.loads(cond_json)
            method = cond.get("method", "unknown")
            
            # We only care about UCCSD(seq) for this combined plot
            if method != "UCCSD(seq)":
                continue
                
            geom = cond.get("geometry", "")
            m = re.search(r"H\s+0\s+0\s+([\d.]+)", geom)
            if not m:
                continue
                
            R_val = float(m.group(1))
            R_key = round(R_val, 5)
            
            iters = cur.execute("SELECT macro_idx, energy FROM macro_iterations WHERE run_id = ? ORDER BY macro_idx", (run_id,)).fetchall()
            
            if iters:
                data[R_key] = [e for idx, e in iters]
                
        conn.close()
    except Exception as e:
        print(f"[warn] {db_path}: {e}")

fig, ax = plt.subplots(figsize=(8, 6))

sorted_Rs = sorted(data.keys())
colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_Rs)))

for idx, R_key in enumerate(sorted_Rs):
    raw_energies = data[R_key]
    
    delta_e_x = []
    delta_e_y = []
    for i in range(1, len(raw_energies)):
        e_curr = raw_energies[i]
        e_prev = raw_energies[i-1]
        delta_e = abs(e_prev - e_curr)
        if delta_e > 0:
            delta_e_x.append(i + 1)
            delta_e_y.append(delta_e)
            
    if delta_e_x:
        ax.plot(
            delta_e_x, delta_e_y,
            marker='.', markersize=5,
            color=colors[idx],
            label=f"R={R_key:.1f}"
        )

ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\Delta E$ (Ha)")
ax.set_title(r"UCCSD(seq) $\Delta E$ Decay by Bond Length")
ax.axhline(1e-8, color='red', linestyle=':', label='Threshold (1e-8)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Put legend outside the plot if there are many lines
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
fig.savefig(args.output, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {os.path.abspath(args.output)}")

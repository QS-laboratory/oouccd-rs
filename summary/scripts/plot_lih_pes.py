"""
LiH Potential Energy Surface Plot
Loads results from cloud_results/lih_pes/task_*.db and plots energy vs. bond length.
Design follows lih_pes_partial.png reference with:
  - FCI reference line (black, computed via PySCF)
  - Upper panel: raw energies
  - Lower panel: E - E_FCI on log scale (Ha)
"""

import argparse
import sqlite3
import json
import glob
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyscf import gto, scf, fci as pyscf_fci

# ── Style (matches reference image) ────────────────────────────────────────
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

# ── Method config (colours/markers matching reference image) ────────────────
METHOD_ORDER = [
    "OO(seq)-UCCD(seq)",
    "UCCD(seq)",
    "UCCSD(seq)",
]

LABELS = {
    "UCCD(seq)":            "UCCD",
    "UCCSD(seq)":           "UCCSD",
    "OO(seq)-UCCD(seq)":    "OO-UCCD",
}

PLOTSTYLES = {
    "OO(seq)-UCCD(seq)":     dict(linestyle="solid", marker="^", markersize=8, color="#2ca02c"),
    "OO(lbfgs)-UCCD(seq)":   dict(linestyle="solid", marker="v", markersize=8, color="#e377c2"),
    "UCCD(seq)":            dict(linestyle="dashed", marker="o", markersize=4, color="#1f77b4"),
    "UCCSD(seq)":            dict(linestyle="dashed", marker="o", markersize=4, color="#ff7f0e"),
}

# ── Bond lengths (Angstrom) ─────────────────────────────────────────────────
R_VALUES = np.arange(1.0, 3.7, 0.2)   # 1.0, 1.2, ..., 3.6

# ── Compute FCI energies via PySCF ──────────────────────────────────────────
print("Computing FCI energies with PySCF ...")
fci_energies: dict[float, float] = {}

for R in R_VALUES:
    mol = gto.Mole()
    mol.atom = f"Li 0 0 0; H 0 0 {R:.5f}"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    norb  = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1    = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri   = mol.ao2mo(mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)

    cisolver = pyscf_fci.direct_spin0.FCI()
    cisolver.max_cycle = 200
    e_fci, _ = cisolver.kernel(h1, eri, norb, nelec)
    fci_energies[round(R, 5)] = e_fci + mol.energy_nuc()
    print(f"  R={R:.2f} Å  E_FCI={fci_energies[round(R,5)]:.8f} Ha")

print("FCI done.\n")

# ── Parse arguments and Load simulation results ───────────────────────────────
parser = argparse.ArgumentParser(description="LiH Potential Energy Surface Plot")
parser.add_argument("--results-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes"),
                    help="Directory containing task_*.db files")
parser.add_argument("--output", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "images", "lih_pes.png"),
                    help="Path to save the output PNG file")
args = parser.parse_args()

RESULTS_DIR = args.results_dir
db_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "task_*.db")))

data: dict[str, dict[float, float]] = {}

for db_path in db_files:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT conditions_json, final_energy, is_completed FROM runs"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[warn] {db_path}: {e}")
        continue

    for conditions_json, final_energy, is_completed in rows:
        if not is_completed or final_energy is None:
            continue
        cond   = json.loads(conditions_json)
        method = cond.get("method", "unknown")
        geom   = cond.get("geometry", "")

        m = re.search(r"H\s+0\s+0\s+([\d.]+)", geom)
        if m is None:
            print(f"[warn] Cannot parse geometry: {geom}")
            continue
        R = float(m.group(1))
        
        if method not in data:
            data[method] = {}
        data[method][round(R, 5)] = final_energy

plot_data: dict[str, list[tuple[float, float]]] = {}
for method in data:
    plot_data[method] = sorted([(r, e) for r, e in data[method].items()], key=lambda x: x[0])


# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(7, 9),
    gridspec_kw={"height_ratios": [1.8, 1], "hspace": 0.08},
    sharex=True,
)

# ── Upper panel: raw energies ───────────────────────────────────────────────

# FCI line
fci_Rs = sorted(fci_energies.keys())
fci_Es = [fci_energies[r] for r in fci_Rs]
ax_top.plot(fci_Rs, fci_Es, color="black", linestyle="-", linewidth=3,
            label="FCI")

for method in METHOD_ORDER:
    if method not in plot_data:
        continue
    Rs = [r for r, _ in plot_data[method]]
    Es = [e for _, e in plot_data[method]]
    ax_top.plot(
        Rs, Es,
        linewidth=1.5,
        label=LABELS[method],
        zorder=3,
        **PLOTSTYLES[method],
    )


ax_top.set_ylabel("Energy (Ha)")

ax_top.legend(loc="lower right")

# ── Lower panel: E - E_FCI  [log scale, Ha] ─────────────────────────────────
for method in METHOD_ORDER:
    if method not in plot_data:
        continue
    Rs = [r for r, _ in plot_data[method]]
    Es = [e for _, e in plot_data[method]]

    diff_Rs, diff_Es = [], []
    for r, e in zip(Rs, Es):
        key = round(r, 5)
        if key in fci_energies:
            delta = e - fci_energies[key]
            if delta > 0:          # log needs positive values
                diff_Rs.append(r)
                diff_Es.append(delta)

    if diff_Rs:
        ax_bot.plot(
            diff_Rs, diff_Es,
            linewidth=1.5,
            label=LABELS[method],
            zorder=3,
        **PLOTSTYLES[method],
        )

ax_bot.set_yscale("log")
ax_bot.set_xlabel("R (Å)")
ax_bot.set_ylabel(r"$E - E_\mathrm{FCI}$ (Ha)")
# ax_bot.legend(loc="upper left")

# Minor ticks on x-axis
for ax in (ax_top, ax_bot):
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
fig.savefig(args.output, bbox_inches="tight", dpi=200)
print(f"Saved: {os.path.abspath(args.output)}")
plt.show()

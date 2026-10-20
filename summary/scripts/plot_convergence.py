"""
LiH Convergence Plots
Loads results from summary/results/lih_pes/task_*.db and plots energy vs iteration for each bond length.
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
from pyscf import gto, scf, fci as pyscf_fci

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "grid.color": "#aaaaaa",
        "figure.dpi": 150,
    }
)

METHOD_ORDER = [
    "OO(seq)-UCCD(seq)",
    "OO(lbfgs)-UCCD(seq)",
    # "UCCD(seq)",
    # "UCCSD(seq)",
]

PLOTSTYLES = {
    "OO(seq)-UCCD(seq)": dict(
        linestyle="solid", marker="^", markersize=8, color="#2ca02c"
    ),
    "OO(lbfgs)-UCCD(seq)": dict(
        linestyle="dashed", marker="v", markersize=8, color="#e377c2"
    ),
    "UCCD(seq)": dict(linestyle="dashed", marker="o", markersize=4, color="#1f77b4"),
    "UCCSD(seq)": dict(linestyle="dashed", marker="o", markersize=4, color="#ff7f0e"),
}

# ── Parse arguments ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LiH Convergence Plotter")
parser.add_argument(
    "--results-dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "results", "lih_pes"),
    help="Directory containing task_*.db files",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__), "..", "images", "convergence_plots"
    ),
    help="Directory to save the convergence PNGs",
)
parser.add_argument(
    "--include-fixed-orbitals", action="store_true",
    help="Also plot UCCD and UCCSD, as in the manuscript figure",
)
parser.add_argument(
    "--pdf", action="store_true", help="Save PDF copies in addition to PNGs",
)
args = parser.parse_args()
if args.include_fixed_orbitals:
    METHOD_ORDER.extend(["UCCD(seq)", "UCCSD(seq)"])

os.makedirs(args.out_dir, exist_ok=True)
db_files = sorted(glob.glob(os.path.join(args.results_dir, "task_*.db")))

# data[R][method] = list of energies
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

            iters = cur.execute(
                "SELECT macro_idx, energy FROM macro_iterations WHERE run_id = ? ORDER BY macro_idx, rowid",
                (run_id,),
            ).fetchall()

            if iters:
                if R_key not in data:
                    data[R_key] = {}
                unique = {}
                for index, energy in iters:
                    if index in unique and abs(unique[index] - energy) > 1e-12:
                        raise ValueError(f"Conflicting histories at macro {index}: {db_path}")
                    unique[index] = energy
                data[R_key][method] = [unique[index] for index in sorted(unique)]

        conn.close()
    except Exception as e:
        print(f"[warn] {db_path}: {e}")

# ── Compute FCI energies via PySCF ──────────────────────────────────────────
print("Computing FCI and HF energies with PySCF ...")
fci_energies = {}
hf_energies = {}
for R_key in sorted(data.keys()):
    mol = gto.Mole()
    mol.atom = f"Li 0 0 0; H 0 0 {R_key:.5f}"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    hf_energies[R_key] = mf.e_tot

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = mol.ao2mo(mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)

    cisolver = pyscf_fci.direct_spin0.FCI()
    cisolver.max_cycle = 200
    e_fci, _ = cisolver.kernel(h1, eri, norb, nelec)
    fci_energies[R_key] = e_fci + mol.energy_nuc()
print("FCI done.\n")

# ── Plot ───────────────────────────────────────────────────────────────────
for R_key in sorted(data.keys()):
    # Figure 1: E - E_FCI
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # Figure 2: Delta E
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    e_fci = fci_energies[R_key]
    e_hf = hf_energies[R_key]

    for method in METHOD_ORDER:
        if method in data[R_key]:
            raw_energies = data[R_key][method]

            # Start with HF at iteration 0
            iters_to_plot = [0]
            diffs_to_plot = [e_hf - e_fci]

            for it, e in enumerate(raw_energies, start=1):
                delta = e - e_fci
                if delta > 0:
                    iters_to_plot.append(it)
                    diffs_to_plot.append(delta)
                else:
                    raise Exception("Energy is below FCI.")

            if diffs_to_plot:
                ax1.plot(
                    iters_to_plot,
                    diffs_to_plot,
                    label=method,
                    **PLOTSTYLES[method],
                )

            # Delta E calculation for second figure
            delta_e_x = []
            delta_e_y = []
            for i in range(1, len(raw_energies)):
                e_curr = raw_energies[i]
                e_prev = raw_energies[i - 1]
                delta_e = abs(e_prev - e_curr)
                if delta_e > 0:
                    delta_e_x.append(i + 1)  # 1-indexed macro_idx
                    delta_e_y.append(delta_e)

            if delta_e_x:
                ax2.plot(
                    delta_e_x,
                    delta_e_y,
                    label=method,
                    **PLOTSTYLES[method],
                )

    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$E - E_\mathrm{FCI}$ (Ha)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper right")

    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\Delta E$ (Ha)")
    ax2.axhline(1e-8, color="red", linestyle=":", label="Threshold (1e-8)")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right")

    # Format R=1.0 -> R010
    R_num = int(round(R_key * 10))
    R_str = f"R{R_num:03d}"

    out_path1 = os.path.join(args.out_dir, f"lih_{R_str}_convergence.png")
    fig1.savefig(out_path1, bbox_inches="tight", dpi=150)
    if args.pdf:
        fig1.savefig(os.path.splitext(out_path1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {os.path.abspath(out_path1)}")

    out_path2 = os.path.join(args.out_dir, f"lih_{R_str}_delta.png")
    fig2.savefig(out_path2, bbox_inches="tight", dpi=150)
    if args.pdf:
        fig2.savefig(os.path.splitext(out_path2)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {os.path.abspath(out_path2)}")

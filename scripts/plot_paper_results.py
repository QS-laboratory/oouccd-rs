#!/usr/bin/env python3
"""Plot the paper CSVs without PySCF, SQLite or cloud access.

The layouts and styles follow the original arXiv v1 figures. The input tables
come from the local rerun and are never modified by the plotter.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


METHODS = ['OO(seq)-UCCD(seq)', 'OO(lbfgs)-UCCD(seq)', 'UCCD(seq)', 'UCCSD(seq)']
PES_METHODS = ['OO(seq)-UCCD(seq)', 'UCCD(seq)', 'UCCSD(seq)']
PES_LABELS = {'OO(seq)-UCCD(seq)': 'OO-UCCD', 'UCCD(seq)': 'UCCD', 'UCCSD(seq)': 'UCCSD'}
STYLES = {
    'OO(seq)-UCCD(seq)': dict(linestyle='solid', marker='^', markersize=8, color='#2ca02c'),
    'OO(lbfgs)-UCCD(seq)': dict(linestyle='solid', marker='v', markersize=8, color='#e377c2'),
    'UCCD(seq)': dict(linestyle='dashed', marker='o', markersize=4, color='#1f77b4'),
    'UCCSD(seq)': dict(linestyle='dashed', marker='o', markersize=4, color='#ff7f0e'),
}
STYLE_REFERENCE = {
    'preprint': 'arXiv:2610.16032v1',
    'original_figure_sha256': {
        'lih_pes.pdf': 'bed3ad99e40805480af1efde8e6d011ac48f4c593eb9f4e659d08b79cf32a086',
        'lih_R016_convergence.pdf': 'b39aaa6047dc8089e5cec128e8873e5a520d0b26631560eeb6ff7d60e10bb93e',
    },
}

# Original manuscript settings, retained independently of the data source.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.linewidth': 1.0, 'axes.labelsize': 12,
                     'axes.titlesize': 12, 'legend.fontsize': 10,
                     'legend.framealpha': .9, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'axes.grid': True,
                     'grid.linestyle': '--', 'grid.alpha': .5,
                     'grid.color': '#aaaaaa', 'figure.dpi': 150,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def read_table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def energy_error(value, reference):
    error = value - reference
    if not math.isfinite(error) or error <= 0:
        raise ValueError('The original logarithmic plot requires positive finite E - E_FCI values')
    return error


def plot_pes(energies):
    fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1.8, 1], 'hspace': .08})
    reference = {float(row['R_angstrom']): float(row['fci_energy_hartree'])
                 for row in energies}
    distances = sorted(reference)
    axes[0].plot(distances, [reference[r] for r in distances], color='black',
                 linewidth=3, label='FCI')
    # The manuscript PES compares fixed UCCD/UCCSD with sequential OO-UCCD.
    for method in PES_METHODS:
        rows = sorted((row for row in energies if row['method'] == method),
                      key=lambda row: float(row['R_angstrom']))
        if len(rows) != 14:
            raise ValueError(f'Expected fourteen PES points for {method}')
        x = [float(row['R_angstrom']) for row in rows]
        y = [float(row['total_energy_hartree']) for row in rows]
        errors = [energy_error(energy, reference[r]) for r, energy in zip(x, y)]
        axes[0].plot(x, y, linewidth=1.5, label=PES_LABELS[method], zorder=3, **STYLES[method])
        axes[1].plot(x, errors, linewidth=1.5, label=PES_LABELS[method], zorder=3, **STYLES[method])
    axes[0].set(ylabel='Energy (Ha)')
    axes[0].legend(loc='lower right')
    axes[1].set(xlabel='R (Å)', ylabel=r'$E - E_\mathrm{FCI}$ (Ha)', yscale='log')
    for axis in axes:
        axis.xaxis.set_minor_locator(MultipleLocator(.1))
    return fig


def plot_convergence(energies, histories):
    reference = {row['method']: float(row['fci_energy_hartree']) for row in energies
                 if math.isclose(float(row['R_angstrom']), 1.6, abs_tol=1e-10)}
    fig, axis = plt.subplots(figsize=(6, 4))
    for method in METHODS:
        rows = sorted((row for row in histories if row['method'] == method
                       and math.isclose(float(row['R_angstrom']), 1.6, abs_tol=1e-10)),
                      key=lambda row: int(row['macro_idx']))
        if method not in reference or not rows:
            raise ValueError(f'Missing 1.6 Å convergence data for {method}')
        x = [int(row['macro_idx']) for row in rows]
        y = [float(row['energy_hartree']) for row in rows]
        axis.plot(x, [energy_error(energy, reference[method]) for energy in y],
                  label=method, **STYLES[method])
    axis.set(xlabel='Iteration', ylabel=r'$E - E_\mathrm{FCI}$ (Ha)', yscale='log')
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(loc='upper right')
    return fig


def write_provenance(dataset, output, figures):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        'inputs_sha256': {name: sha(dataset / name)
                          for name in ['energies.csv', 'macro_iterations.csv']},
        'plot_script': Path(__file__).name,
        'plot_script_sha256': sha(Path(__file__)),
        'software': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__},
        'style_reference': STYLE_REFERENCE,
        'display': {'energy_error': 'E - E_FCI', 'error_floor_hartree': None,
                    'pes_layout': 'two panels, height ratio 1.8:1',
                    'convergence_layout': 'one panel'},
        'outputs_sha256': {name: sha(output / name) for name in figures},
    }
    upstream = dataset / 'provenance.json'
    if upstream.exists():
        source = json.loads(upstream.read_text())
        record['dataset_provenance_sha256'] = sha(upstream)
        record['dataset_source_commit'] = source.get('source_commit')
        record['dataset_binary_sha256'] = source.get('binary_sha256')
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    energies = read_table(args.dataset / 'energies.csv')
    histories = read_table(args.dataset / 'macro_iterations.csv')
    if len(energies) != 56:
        parser.error('Expected the full 56-point paper dataset')
    if args.out_dir.resolve() == args.dataset.resolve():
        parser.error('--out-dir must differ from the input dataset directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for name, figure in [('lih_pes', plot_pes(energies)),
                         ('lih_R016_convergence', plot_convergence(energies, histories))]:
        for extension in ['png', 'pdf']:
            filename = f'{name}.{extension}'
            figure.savefig(args.out_dir / filename, bbox_inches='tight',
                           dpi=200 if name == 'lih_pes' else 150)
            figures.append(filename)
        plt.close(figure)
    write_provenance(args.dataset, args.out_dir, figures)


if __name__=='__main__':main()

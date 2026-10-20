#!/usr/bin/env python3
"""Replot the portable paper CSVs without PySCF, SQLite or cloud access."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path)
    p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    with(args.dataset/'energies.csv').open()as f:energies=list(csv.DictReader(f))
    with(args.dataset/'macro_iterations.csv').open()as f:histories=list(csv.DictReader(f))
    assert len(energies)==56
    methods=['UCCD(seq)','UCCSD(seq)','OO(seq)-UCCD(seq)','OO(lbfgs)-UCCD(seq)']
    colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    fig,axes=plt.subplots(2,1,figsize=(7.5,7),sharex=True)
    reference={float(r['R_angstrom']):float(r['fci_energy_hartree'])for r in energies}
    x=sorted(reference);axes[0].plot(x,[reference[v]for v in x],color='black',label='FCI')
    for method,color in zip(methods,colors):
        if method.startswith('OO(lbfgs)'):continue
        rows=sorted([r for r in energies if r['method']==method],key=lambda r:float(r['R_angstrom']))
        assert len(rows)==14
        x=[float(r['R_angstrom'])for r in rows]
        axes[0].plot(x,[float(r['total_energy_hartree'])for r in rows],'-o',ms=3,color=color,label=method)
        axes[1].plot(x,[max(1e-12,float(r['error_from_fci_hartree']))for r in rows],'-o',ms=3,color=color)
    axes[0].set(ylabel='Total energy (Ha)');axes[0].legend(fontsize=9)
    axes[1].set(xlabel='Li–H distance (Å)',ylabel='Energy above FCI (Ha)',yscale='log')
    axes[1].text(.02,.03,'Values below 1e−12 Ha drawn at the plotting floor.',transform=axes[1].transAxes,fontsize=8)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.out_dir/f'lih_pes.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7.5,7),sharex=True)
    for method,color in zip(methods,colors):
        rows=sorted([r for r in histories if r['method']==method and abs(float(r['R_angstrom'])-1.6)<1e-10],key=lambda r:int(r['macro_idx']))
        x=[int(r['macro_idx'])for r in rows];y=[float(r['energy_hartree'])for r in rows]
        axes[0].plot(x,y,'-o',ms=3,color=color,label=method)
        axes[1].plot(x[1:],[max(1e-15,abs(b-a))for a,b in zip(y,y[1:])],'-o',ms=3,color=color)
    axes[0].set(ylabel='Total energy (Ha)',title='LiH/STO-3G at 1.6 Å');axes[0].legend(fontsize=9)
    axes[1].axhline(1e-8,color='.4',linestyle=':')
    axes[1].set(xlabel='Macro iteration',ylabel='Consecutive-energy change (Ha)',yscale='log')
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.out_dir/f'lih_R016_convergence.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()

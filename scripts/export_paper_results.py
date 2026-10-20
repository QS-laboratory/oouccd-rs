#!/usr/bin/env python3
"""Export a validated full Slurm workspace to portable numerical tables.

Raw SQLite inputs are opened read-only. RHF/FCI references are recomputed;
no VQE calculation, optimizer continuation, or cloud access is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import tarfile

from check_paper_slurm import check


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def references(config):
    from pyscf import gto,scf,fci
    import numpy as np
    c=config['chemistry']
    assert c['freeze_core']==0 and c['spin']==0 and c['charge']==0
    mol=gto.M(atom=c['geometry'],basis=c['basis'],spin=0,charge=0,symmetry=False,verbose=0)
    mf=scf.RHF(mol).run();assert mf.converged
    n=mf.mo_coeff.shape[1];h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
    eri=mol.ao2mo(mf.mo_coeff,compact=False).reshape(n,n,n,n)
    solver=fci.direct_spin0.FCI();solver.max_cycle=200
    energy,_=solver.kernel(h1,eri,n,mol.nelectron);assert solver.converged
    assert np.isfinite(energy)
    return float(mf.e_tot),float(energy+mol.energy_nuc())


def sanitize(value):
    if isinstance(value,dict):
        return {k:sanitize(v)for k,v in value.items()if k not in ['archive_root','rerun_workspace','archive_db','argo_db']}
    if isinstance(value,list):return [sanitize(v)for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path);parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--historical-comparison',type=Path)
    parser.add_argument('--accounting-file',type=Path,help='sacct -P -n: JobID,State,ExitCode,ElapsedRaw,NodeList')
    args=parser.parse_args();work=args.workspace.resolve();output=args.out_dir
    manifest=json.loads((work/'manifest.json').read_text())
    assert manifest['scope']=='full'and manifest['task_indices']==list(range(56))
    assert sha(work/'bin/oouccd-rs')==manifest['binary_sha256']
    output.mkdir(parents=True,exist_ok=False)
    energies=[];histories=[];hashes={};reference={}
    for index in manifest['task_indices']:
        name=f'task_{index:03}';config_path=work/'configs'/f'{name}.json';db=work/'results'/f'{name}.db'
        cfg=json.loads(config_path.read_text());result=check(db,cfg)
        distance=round(float(cfg['chemistry']['geometry'].split()[-1]),8)
        if distance not in reference:reference[distance]=references(cfg)
        rhf,fci_energy=reference[distance]
        assert result['energy_hartree']>=fci_energy-1e-8
        energies.append({'task':name,'method':result['method'],'R_angstrom':distance,
            'total_energy_hartree':result['energy_hartree'],'rhf_energy_hartree':rhf,
            'fci_energy_hartree':fci_energy,'error_from_fci_hartree':result['energy_hartree']-fci_energy,
            'macro_iterations':result['iterations'],'converged':result['converged'],
            'final_delta_hartree':result['final_delta_hartree']})
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)as conn:
            rows=conn.execute('SELECT macro_idx,energy,converged FROM macro_iterations ORDER BY macro_idx,rowid').fetchall()
        history={i:(e,bool(flag))for i,e,flag in rows}
        history={0:(rhf,False),**history}
        for i,(energy,converged)in sorted(history.items()):
            histories.append({'task':name,'method':result['method'],'R_angstrom':distance,'macro_idx':i,
                'energy_hartree':energy,'converged':converged})
        hashes[f'configs/{name}.json']=sha(config_path);hashes[f'results/{name}.db']=sha(db)
    write_csv(output/'energies.csv',energies);write_csv(output/'macro_iterations.csv',histories)
    packages={}
    for line in (work/'provenance/python-freeze.txt').read_text().splitlines():
        if '=='in line:
            name,version=line.split('==',1)
            if name in ['numpy','scipy','pyscf','openfermion','matplotlib']:packages[name]=version
    numerical_sources={}
    with tarfile.open(work/'provenance/source.tar')as archive:
        for member in archive.getmembers():
            if member.isfile()and(member.name.startswith('src/')or member.name in ['Cargo.toml','Cargo.lock']):
                numerical_sources[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    provenance={'dataset':'LiH/STO-3G full statevector Slurm rerun','source_commit':manifest['source_commit'],
        'binary_sha256':manifest['binary_sha256'],'source_archive_sha256':sha(work/'provenance/source.tar'),
        'source_patch_sha256':sha(work/'provenance/source.patch'),'numerical_source_sha256':numerical_sources,
        'raw_input_and_database_sha256':hashes,'points':len(energies),'history_rows_including_rhf':len(histories),
        'converged_points':sum(r['converged']for r in energies),
        'iteration_capped_points':[{'method':r['method'],'R_angstrom':r['R_angstrom']}for r in energies if not r['converged']],
        'recorded_python_packages':packages,'reference_export_python':platform.python_version(),
        'cpus_per_task':manifest['cpus_per_task'],'array_concurrency':manifest['concurrency'],
        'notes':['Fresh rerun, not an export of every original manuscript database.',
                 'RHF/FCI references and iteration zero recomputed with the selected PySCF environment.',
                 'Duplicate final macro rows collapsed; full parameters remain in the raw SQLite files.']}
    if args.historical_comparison:
        comparison=sanitize(json.loads(args.historical_comparison.read_text()))
        (output/'historical_comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
        provenance['historical_comparison_input_sha256']=sha(args.historical_comparison)
    if args.accounting_file:
        scheduler=[]
        for line in args.accounting_file.read_text().splitlines():
            fields=line.split('|')
            if not re.fullmatch(r'\d+_\d+',fields[0]):continue
            scheduler.append({'job_id':fields[0],'task_index':int(fields[0].split('_')[-1]),
                'state':fields[1].strip(),'exit_code':fields[2],'elapsed_seconds':int(fields[3]),'node':fields[4]})
        scheduler.sort(key=lambda r:r['task_index'])
        assert [r['task_index']for r in scheduler]==list(range(56))
        assert all(r['state']=='COMPLETED'and r['exit_code']=='0:0'for r in scheduler)
        write_csv(output/'scheduler.csv',scheduler)
    provenance['export_files_sha256']={p.name:sha(p)for p in output.iterdir()if p.is_file()}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'points':len(energies),'converged':provenance['converged_points'],
        'history_rows':len(histories),'output':str(output)}))


if __name__=='__main__':main()

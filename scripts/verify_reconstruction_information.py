"""Verify that reconstruction fits depend on counts/constraints, not scoring truth."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from reconstruct_rdms import Reconstruction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True);args=p.parse_args();root=args.work.resolve()
    snapshot=json.loads((root/'verification/h2_snapshot.json').read_text())
    counts=None
    for line in (root/'verification/h2_reference_counts.jsonl').open():
        row=json.loads(line)
        if row['record_type']=='trial':counts=row['counts'];break
    if counts is None:raise RuntimeError('Missing matched measurement batch')
    changed=copy.deepcopy(snapshot)
    changed['exact_means']=[0.123]*len(snapshot['exact_means'])
    changed['exact_energy_ha']=123456.
    changed['nuclear_repulsion']=-123456.
    changed['h1']['data']=[-31*x for x in snapshot['h1']['data']]
    changed['h2']['data']=[17*x for x in snapshot['h2']['data']]
    checks=[]
    for method in ['spatial_psd','dqg','mle','sector_mle']:
        a=Reconstruction(snapshot,method,solver='CLARABEL').solve(counts)
        b=Reconstruction(changed,method,solver='CLARABEL').solve(counts)
        error=float(np.max(np.abs(np.asarray(a['means'])-np.asarray(b['means']))))
        if error>1e-9:raise RuntimeError(f'{method} depends on diagnostic truth or Hamiltonian: {error}')
        checks.append({'method':method,'maximum_fit_difference':error,'passed':True})
    output={'passed':True,'checks':checks,'information_used':'Original Pauli counts, reconstruction maps, particle-number/spin constraints only'}
    (root/'verification/reconstruction_information_check.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

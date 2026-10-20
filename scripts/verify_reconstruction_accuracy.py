"""Check numerical accuracy of the prototype MLE on two correlated LiH states.

Population-likelihood controls use fractional expected counts, not simulated
measurements. Independent fresh models fit the saved experimental counts;
oracle-control solutions are never used to initialize those fits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from reconstruct_rdms import Reconstruction,score


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):return float(value)if value is not None and math.isfinite(value)else None


def fit(snapshot,counts,tolerance):
    start=time.perf_counter();model=Reconstruction(snapshot,'mle','CLARABEL',tolerance=tolerance)
    setup=time.perf_counter()-start
    result=model.solve(counts);result['score']=score(snapshot,result['means'])
    result.update(model_build_seconds=setup,objective=finite(model.problem.value),tolerance=tolerance,completed=True)
    return result


def verify(root,output):
    if output.exists():raise RuntimeError('Refusing to overwrite numerical reference results')
    tasks=json.loads((root/'reconstruction_manifest.json').read_text());records=[];inputs={}
    for distance,suffix in [(1.6,'_seq_m3'),(3.,'_newton_stable_m3')]:
        task=next(t for t in tasks if t['distance']==distance and t['shots']==100000 and t['name'].endswith(suffix))
        snapshot=json.loads(Path(task['snapshot']).read_text());width=len(snapshot['exact_means'])
        for key in ['snapshot','data','output']:inputs[task[key]]=sha(task[key])
        raw=next(json.loads(line)for line in Path(task['data']).open()if json.loads(line)['record_type']=='trial')
        fitted=next(json.loads(line)for line in Path(task['output']).open()if json.loads(line)['record_type']=='trial')
        assert raw['index']==fitted['index']==0 and raw['seed']==fitted['seed']
        original=next(f for f in fitted['results']if f['method']=='mle')
        truth=np.asarray(snapshot['exact_means']);probability=np.clip((1+truth)/2,0,1)
        population={'shots':100000,'plus_counts':(100000*probability).tolist()}
        reference=Reconstruction(snapshot,'mle','CLARABEL')
        diagnostics=reference.validate(truth)
        assert diagnostics['max_constraint_residual']<1e-10 and diagnostics['max_pauli_bound_violation']<1e-10
        assert min(diagnostics['minimum_eigenvalues'].values())>-1e-10
        entropy=-sum(p*math.log(p)if p>0 else 0 for p in probability[reference.observed]) \
                -sum((1-p)*math.log(1-p)if p<1 else 0 for p in probability[reference.observed])
        del reference
        for tolerance in [1e-6,1e-9]:
            record={'snapshot':task['name'],'kind':'population_likelihood','tolerance':tolerance,
                'fractional_expected_counts':True,'true_point_validation':diagnostics,'known_optimum_objective':entropy}
            try:
                result=fit(snapshot,population,tolerance)
                result['maximum_pauli_error']=float(np.max(np.abs(np.asarray(result['means'])-truth)))
                record.update(result)
                record['objective_gap']=result['objective']-entropy if result['objective']is not None else None
            except Exception as error:record.update(completed=False,error=repr(error))
            records.append(record)
        record={'snapshot':task['name'],'kind':'saved_measurement_batch','tolerance':1e-9,
            'fractional_expected_counts':False,'original_default_fit':original,'batch_index':raw['index'],'seed':raw['seed']}
        try:
            # A fresh model is essential: the population fit is not a legitimate
            # warm start for the actual same-count estimator comparison.
            record.update(fit(snapshot,raw['counts'],1e-9))
            if original['completed']:
                record['maximum_change_from_default']=float(np.max(np.abs(np.asarray(record['means'])-np.asarray(original['means']))))
                record['energy_error_shift_mha']=1000*(record['score']['energy_error_ha']-original['score']['energy_error_ha'])
        except Exception as error:record.update(completed=False,error=repr(error))
        records.append(record)
    assert len(records)==6
    tight=[r for r in records if r['kind']=='population_likelihood'and r['tolerance']==1e-9]
    result={'records':records,'input_sha256':inputs,'source_sha256':sha(__file__),
        'reconstruction_source_sha256':sha(Path(__file__).with_name('reconstruct_rdms.py')),
        'reference_states':2,'additional_quantum_measurements':0,
        'high_accuracy_population_reference_passed':all(r['completed']and abs(r['score']['energy_error_ha']*1000)<=.01 for r in tight),
        'population_energy_error_target_mha':.01,
        'interpretation':'Selected numerical stress checks, not an estimator-performance experiment or evidence about a full MLE population. The original benchmark results and policies remain unchanged.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# MLE numerical accuracy checks','',
        'Two correlated LiH states are checked at default and tighter solver tolerances. Population controls use fractional expected counts with a known optimum; they are not quantum measurements.',
        'Separate fresh models fit the first saved measurement batch at tighter tolerance. Truth-based solutions never initialize those actual-data fits. These selected checks do not establish numerical accuracy for every benchmark fit.','',
        '| State | Data | Tolerance | Completed | Energy-estimator error (mHa) | Shift from original fit (mHa) |','|---|---|---:|---|---:|---:|']
    for r in records:
        error=1000*r['score']['energy_error_ha']if r['completed']else None
        lines.append(f"| {r['snapshot']} | {r['kind']} | {r['tolerance']} | {r['completed']} | {error} | {r.get('energy_error_shift_mha')} |")
    lines+=['',f"Tighter population-reference energy-error target: 0.01 mHa; met for both states: {result['high_accuracy_population_reference_passed']}.",
        'Pauli-vector errors, objective values and gaps, feasibility diagnostics, solver statuses, costs and original fits are retained in the JSON output.',
        'Differences between physical-reconstruction methods concern the recorded prototype and numerical settings. A feasible conic fit alone does not prove an exact maximum-likelihood optimum.']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=verify(args.work.resolve(),args.output)
    print(json.dumps({k:v for k,v in r.items()if k not in ['records','input_sha256']}))


if __name__=='__main__':main()

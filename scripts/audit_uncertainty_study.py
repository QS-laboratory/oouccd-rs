"""Independent scope, count, state and provenance checks for the complete study.

--allow-running records an incomplete progress audit; it can never certify
completion. Primary comparison statistics are not recomputed or inspected.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

EXPECTED={'export':21,'fixed':63,'pilot':320,'reconstruction':27,'bootstrap':18,
          'ucc_extension':40,'confirmation':7200,'controls':6,'decision_audit':76}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root,allow_running=False):
    study=json.loads((root/'study.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    freeze=json.loads((root/'policy_freeze.json').read_text())
    assert sha(root/'protocol.json')==freeze['protocol_sha256'],'Frozen protocol changed'
    assert sha(root/'confirmation_manifest.json')==freeze['manifest_sha256'],'Frozen test manifest changed'
    assert study['protocol']==protocol
    for name,binary in study['binaries'].items():assert sha(binary['path'])==binary['sha256'],name
    for name,want in freeze['analysis_source_sha256'].items():
        assert sha(root/'confirmation_source/scripts'/name)==want,name
    seeds=protocol['confirmation_seeds'];assert len(seeds)==len(set(seeds))==100
    assert all(b-a>=protocol['seed_window']for a,b in zip(sorted(seeds),sorted(seeds)[1:]))
    assert min(seeds)>max(protocol['development_seeds'])+3831
    for job in study['jobs']:
        if job['phase']=='confirmation':assert job['submitted_unix']>=freeze['frozen_unix']
    snapshots=json.loads((root/'export_manifest.json').read_text())
    initial={t['distance']:json.loads(Path(t['output']).read_text())for t in snapshots if t['iteration']==0}
    identity={distance:next(i for i,w in enumerate(s['plan']['pauli_words'])
        if not(w['x_mask']|w['y_mask']|w['z_mask']))for distance,s in initial.items()}
    for task in snapshots:assert sha(task['trajectory'])==task['origin_sha256'],'Snapshot origin changed'
    completed=True;phases={};native_points=0;counts_checked=0;maximum_orthogonality=0.
    manifests={}
    for phase,number in EXPECTED.items():
        manifest=root/f'{phase}_manifest.json'
        if not manifest.exists():
            assert allow_running,f'Missing {phase}';completed=False;phases[phase]={'missing_manifest':number};continue
        tasks=json.loads(manifest.read_text());manifests[phase]=tasks
        assert len(tasks)==number,(phase,'incorrect scope')
        assert [t['index']for t in tasks]==list(range(number)),(phase,'indices')
        assert len({t['output']for t in tasks})==number,(phase,'duplicate outputs')
        statuses=Counter()
        for task in tasks:
            output=Path(task['output']);execution=output.with_suffix(output.suffix+'.execution.json')
            if not execution.exists():completed=False;statuses['not_terminal']+=1;continue
            record=json.loads(execution.read_text());assert record['exit_code']==0,(phase,task['index'],'process failure')
            assert output.exists(),(phase,'missing successful output')
            if phase not in ('pilot','ucc_extension','confirmation','controls'):
                statuses['completed']+=1;continue
            native=json.loads(output.read_text());assert native['status']in('completed','failed')
            statuses[native['status']]+=1
            cfg=json.loads(Path(task['config']).read_text())
            if task.get('config_sha256'):assert sha(task['config'])==task['config_sha256']
            assert cfg['method']['max_macro_iterations']==10 and cfg['method']['energy_tolerance']<0
            assert native.get('ucc_mitigation','none')==('affine'if task.get('affine_ucc')else'none')
            assert native['ucc_measurement_mode']==('exact'if task.get('exact_ucc')else'configured')
            fci=study['references'][str(task['distance'])]['fci_energy_ha']
            points=native['iterations'];assert points and points[0]['macro_iteration']==0
            for index,point in enumerate(points):
                assert point['macro_iteration']==index
                rotation=np.asarray(point['orbital_rotation']);assert rotation.shape==(6,6)
                error=float(np.max(np.abs(rotation@rotation.T-np.eye(6))));maximum_orthogonality=max(maximum_orthogonality,error)
                assert error<1e-8 and abs(error-point['orthogonality_error'])<1e-11
                assert np.isfinite(point['exact_energy_ha'])and point['exact_energy_ha']>=fci-1e-7
                assert len(point['theta'])==76 and np.all(np.isfinite(point['theta']))
                native_points+=1
                if index==0:continue
                record=point['oo']['statistics'];assert abs(record['alpha']-.005)<1e-15
                if not task['shots']:continue
                counts=record['counts'];shot_count=task['shots'];plus=np.asarray(counts['plus_counts'])
                assert counts['shots']==shot_count and plus.shape==(2779,)
                assert np.all(plus>=0)and np.all(plus<=shot_count)
                assert plus[identity[task['distance']]]==shot_count
                stride=2 if task.get('exact_ucc')else 383
                assert counts['sampling_seed']==task['seed']+1+(index-1)*stride
                acceptance=task['acceptance'];training=int(np.floor(shot_count*.8))if acceptance!='nominal'else shot_count
                assert record['input_shots']==shot_count and record['training_shots']==training
                if acceptance=='heldout':
                    assert record['validation_shots']==shot_count-training
                    bound=record['validation_upper_bound_ha'];assert np.isfinite(bound)
                    assert record['accepted']==(bound<0.)
                    assert point['oo']['exact_energy_after_ha']==point['oo']['exact_energy_before_ha']if not record['accepted']else True
                else:
                    assert record['validation_shots']is None and record['validation_upper_bound_ha']is None
                if task['mitigation']=='affine':assert record['correction_residual_after']<1e-8
                counts_checked+=1
            if native['status']=='completed':
                assert len(points)==11 and native['rdm_evaluations']==3831
                assert native['diagnostic_rdm_evaluations']==11 and native['unique_measured_pauli_words']==2778
                measured=21 if task.get('exact_ucc')else 3831
                assert native['modeled_shots']==measured*2778*(task['shots']or 0)
            else:assert native['error']and native['failed_macro_iteration']is not None
        phases[phase]=dict(statuses)
    if 'confirmation'in manifests:
        keys={(t['distance'],t['shots'],t['seed'],t['name'])for t in manifests['confirmation']}
        assert len(keys)==7200
        for d in protocol['distances']:
            for shots in protocol['shots']:
                for seed in seeds:
                    assert all((d,shots,seed,v['name'])in keys for v in protocol['variants'])
    assert completed or allow_running,'Experiment scope has not completed'
    for name in ('verification/car_operator_check.json','verification/small_reference_completed.json'):
        assert(root/name).exists(),name
    if completed:
        replay=json.loads((root/'decision_replay_audit.json').read_text())
        assert replay['passed']and replay['trajectories']==7566
        for phase in ('fixed','pilot','bootstrap','reconstruction','ucc_extension','confirmation'):
            result=json.loads((root/f'{phase}_summary.json').read_text());assert result['validation_passed'],phase
        assert(root/'report.md').exists()and(root/'scheduler_accounting.json').exists()
        presentation=json.loads((root/'confirmation_render_provenance.json').read_text())
        assert presentation['summary_sha256']==sha(root/'confirmation_summary.json')
        assert presentation['frozen_analysis_sha256']==freeze['analysis_source_sha256']['analyze_uncertainty_study.py']
        assert presentation['primary_statistics_recomputed']is False
        assert presentation['additional_quantum_measurements']==0
        assert presentation['trials']==7200 and presentation['plots_rendered']
        for artifact in ['confirmation_supplement.md','confirmation_primary.csv','confirmation_runs.csv',
                         'confirmation_noiseless_controls.csv','figures/confirmation_primary_paired.png',
                         'figures/confirmation_supplement_convergence_noisy_ucc.pdf']:
            assert(root/artifact).exists(),artifact
        accounting=json.loads((root/'scheduler_accounting.json').read_text())
        assert all(r['state']=='COMPLETED'and r['exit_code']=='0:0'for r in accounting)
        expected=sum(j['tasks']for j in study['jobs']);assert len(accounting)==expected,(len(accounting),expected)
    workers_terminal=completed
    completed=completed and not allow_running
    evidence={'complete':completed,'workers_terminal':workers_terminal,'checked_unix':time.time(),'phases':phases,'native_points_checked':native_points,
        'raw_count_batches_checked':counts_checked,'maximum_orthogonality_error':maximum_orthogonality,
        'primary_protocol_sha256':sha(root/'protocol.json'),'source_sha256':sha(__file__),
        'note':'An allow-running audit cannot certify completion. Replay/statevector audits are separate evidence.'}
    path=root/('independent_completion_audit.json'if completed else'independent_progress_audit.json')
    path.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in evidence.items()if k not in('phases',)}))
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--allow-running',action='store_true');args=p.parse_args();audit(args.work.resolve(),args.allow_running)


if __name__=='__main__':main()

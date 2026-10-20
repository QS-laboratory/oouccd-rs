"""Regression checks for confirmation wiring and failure-aware paired inference."""
import json
import csv
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from analyze_uncertainty_study import paired
from run_uncertainty_study import freeze_confirmation,sha,submit_confirmation,write_json
from run_decision_replays import submit as submit_decision_replays
from render_uncertainty_results import render as render_confirmation,render_ucc
from summarize_uncertainty_auxiliary import reconstructed_updates,reconstruction_fit_summary


def fixture(root):
    binary=root/'bin/evaluator';binary.parent.mkdir();binary.write_bytes(b'fixture, not executed')
    protocol={'confirmation_seeds':[80261020+i*1000000 for i in range(100)]}
    write_json(root/'protocol.json',protocol)
    write_json(root/'study.json',{'protocol':protocol,'jobs':[],'python':'/unused/python',
        'binaries':{'evaluate_shot_noise':{'path':str(binary),'sha256':sha(binary)}}})
    write_json(root/'verification/verification.json',{'legacy_replay':{'exact':0.,'noisy':0.}})
    write_json(root/'fixed_summary.json',{'validation_passed':True,'tasks':63,'independent_batches':63000})
    write_json(root/'pilot_summary.json',{'validation_passed':True,'trials':320,'statuses':{'completed':320}})
    (root/'confirmation/results').mkdir(parents=True);(root/'logs').mkdir()


class PipelineTests(unittest.TestCase):
    def test_ucc_presentation_keeps_failures_and_pairs_the_same_subset(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);runs=[];old=[];cells=[];pairs=[]
            for distance in [1.6,3.]:
                for shots in [1000,100000]:
                    for solver in ['seq','newton_stable']:
                        name=solver+'_none_nominal_affine_ucc'
                        cells.append({'distance':distance,'shots':shots,'variant':name})
                        for seed in range(5):
                            index=len(runs);failed=distance==1.6 and shots==1000 and solver=='seq'and seed==4
                            base={'index':index,'distance':distance,'shots':shots,'seed':seed,'solver':solver,
                                'mitigation':'none','acceptance':'nominal','exact_ucc':False,'completed':True,
                                'error_mha':100. if failed else 1.}
                            old.append(base)
                            runs.append({**base,'name':name,'affine_ucc':True,'baseline_pilot_index':index,
                                'completed':not failed,'error_mha':None if failed else .5})
                            pairs.append({'index':index,'error_difference_mha':None if failed else -.5})
            write_json(root/'pilot_summary.json',{'runs':old})
            write_json(root/'ucc_extension_summary.json',{'validation_passed':True,'trials':40,
                'runs':runs,'cells':cells,'paired_against_pilot':pairs})
            with patch('render_uncertainty_results.plot_ucc_pairs'):evidence=render_ucc(root)
            selected=next(g for g in evidence['groups']if g['distance']==1.6 and g['shots']==1000 and g['solver']=='seq')
            self.assertEqual((selected['raw_completed'],selected['affine_completed'],selected['joint_completions']),(5,4,4))
            self.assertEqual(selected['raw_error_mha']['mean'],1.)
            with(root/'ucc_extension_paired.csv').open()as handle:
                failed=next(row for row in csv.DictReader(handle)if row['affine_completed']=='False')
            self.assertEqual((failed['affine_error_mha'],failed['difference_mha']),('',''))

    def test_reconstruction_baselines_use_the_identical_completed_subset(self):
        group=[]
        for raw,corrected,completed in [(1.,.5,True),(3.,1.5,True),(100.,None,False)]:
            fit={'completed':completed,'seconds':1.,'baseline':{'raw_energy_error_mha':raw,
                'affine_energy_error_mha':raw/2,'raw_rdm_squared_error':raw**2,'affine_rdm_squared_error':raw**2/4}}
            if completed:fit.update(score={'energy_error_ha':corrected/1000,'rdm_squared_error':corrected**2},
                validation={'minimum_eigenvalues':{'metric':0.},'max_constraint_residual':0.},status='optimal')
            else:fit['error']='numerical failure'
            group.append(fit)
        summary=reconstruction_fit_summary('fixture',group)
        self.assertEqual((summary['requested'],summary['completed']),(3,2))
        self.assertAlmostEqual(summary['matched_raw_energy_error_mha']['rmse'],5.**.5)
        self.assertAlmostEqual(summary['paired_energy_squared_error_minus_raw_mha2']['mean'],-3.75)
        self.assertEqual(summary['failures'],['numerical failure'])

    def test_reconstruction_cached_results_reject_changed_counts(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);binary=root/'evaluator';binary.write_bytes(b'not executed')
            write_json(root/'study.json',{'binaries':{'reconstructed_oo':{'path':str(binary),'sha256':sha(binary)}}})
            task={'output':str(root/'fitted.jsonl'),'snapshot':str(root/'snapshot.json'),'data':str(root/'counts.jsonl')}
            for key in ['output','snapshot','data']:Path(task[key]).write_text('{}\n')
            output=root/'fitted.oo.jsonl';output.write_text('saved output\n')
            write_json(output.with_suffix('.completed.json'),{'trials':25})
            write_json(output.with_suffix('.provenance.json'),{'binary_sha256':sha(binary),
                'input_sha256':sha(task['output']),'snapshot_sha256':sha(task['snapshot']),
                'data_sha256':sha(task['data']),'result_sha256':sha(output)})
            Path(task['data']).write_text('{"different_counts":true}\n')
            with patch('summarize_uncertainty_auxiliary.subprocess.run')as native:
                with self.assertRaisesRegex(RuntimeError,'inputs changed'):reconstructed_updates(root,task)
            native.assert_not_called()

    def test_renderer_preserves_frozen_statistics_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary)
            source=root/'confirmation_source/scripts/analyze_uncertainty_study.py'
            source.parent.mkdir(parents=True);source.write_text('# frozen fixture\n')
            write_json(root/'protocol.json',{'fixture':True})
            write_json(root/'policy_freeze.json',{'protocol_sha256':sha(root/'protocol.json'),
                'analysis_source_sha256':{'analyze_uncertainty_study.py':sha(source)}})
            write_json(root/'confirmation_analysis_provenance.json',{'sha256':sha(source)})
            comparisons=[{'distance':d,'shots':s,'a':'a','b':'b','wins':1,'losses':0,'ties':0,
                'p_two_sided':1.,'holm_p':1.,'paired_difference_mha':{'n':1,'mean':-123.},
                'paired_mean_difference_ci95_unadjusted':[-124.,-122.]}
                for d in [1.6,2.4,3.]for s in [1000,10000,100000]]
            run={'index':0,'distance':1.6,'shots':1000,'seed':1,'name':'a','status':'completed',
                 'error_mha':99.,'worker_seconds':1.}
            # Deliberately unrelated run data ensure the renderer copies the
            # saved inferential results instead of silently analyzing again.
            write_json(root/'confirmation_summary.json',{'validation_passed':True,'trials':1,'runs':[run],
                'cells':[],'primary_comparisons':comparisons,'controls':[
                    {'distance':d,'solver':s,'status':'completed','error_mha':.1,'worker_seconds':1.}
                    for d in [1.6,2.4,3.]for s in ['seq','newton_stable']]})
            before=sha(root/'confirmation_summary.json')
            with patch('render_uncertainty_results.plot_primary')as plot:
                evidence=render_confirmation(root,plots=False)
            plot.assert_not_called()
            self.assertEqual(sha(root/'confirmation_summary.json'),before)
            self.assertFalse(evidence['primary_statistics_recomputed'])
            with(root/'confirmation_primary.csv').open()as handle:
                exported=list(csv.DictReader(handle))
            self.assertEqual(len(exported),9)
            self.assertTrue(all(float(r['mean_difference_mha'])==-123. for r in exported))
            source.write_text('# changed after freezing\n')
            with self.assertRaises(AssertionError):render_confirmation(root,plots=False)

    def test_replay_depends_only_on_live_jobs_and_submits_once(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary)
            jobs=[{'phase':phase,'job_id':job}for phase,job in
                  [('pilot','11'),('confirmation','12'),('confirmation','13'),('controls','14')]]
            write_json(root/'study.json',{'jobs':jobs,'python':'/unused/python'})
            write_json(root/'decision_audit_manifest.json',[{'index':0}])
            queue=subprocess.CompletedProcess([],0,stdout='12|123\n13|13\n999|999\n',stderr='')
            with patch('run_decision_replays.subprocess.run',return_value=queue)as query, \
                 patch('run_decision_replays.subprocess.check_output',return_value='20000\n')as sbatch:
                submit_decision_replays(root)
                submit_decision_replays(root)
            self.assertEqual(query.call_count,1)
            self.assertEqual(sbatch.call_count,1)
            command=sbatch.call_args.args[0]
            self.assertIn('--dependency=afterok:12:13',command)
            self.assertIn('--array=0-0%16',command)
            saved=json.loads((root/'study.json').read_text())
            self.assertEqual(saved['jobs'][-1]['phase'],'decision_audit')
            self.assertEqual(saved['jobs'][-1]['tasks'],1)

    def test_fresh_confirmation_factorial_and_array_offsets(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);fixture(root);freeze_confirmation(root)
            tasks=json.loads((root/'confirmation_manifest.json').read_text())
            self.assertEqual(len(tasks),7200)
            self.assertEqual(len({t['seed']for t in tasks}),100)
            self.assertEqual(len({(t['distance'],t['shots'],t['seed'],t['name'])for t in tasks}),7200)
            for t in tasks:
                config=json.loads(Path(t['config']).read_text())
                self.assertEqual(sha(t['config']),t['config_sha256'])
                self.assertEqual(config['method']['max_macro_iterations'],10)
                self.assertEqual(config['method']['orbital_statistics']['training_fraction'],.8)
                self.assertFalse(Path(t['output']).exists())
            ids=iter(str(i)for i in range(20000,20010))
            with patch('run_uncertainty_study.subprocess.check_output',side_effect=lambda *a,**k:next(ids)+'\n'):
                submit_confirmation(root)
            jobs=json.loads((root/'study.json').read_text())['jobs']
            confirm=[j for j in jobs if j['phase']=='confirmation']
            self.assertEqual(len(confirm),9)
            for group,j in enumerate(confirm):
                self.assertIn('--array=0-799%8',j['command'])
                self.assertEqual(j['offset'],group*800)
                self.assertIn(f'SLURM_ARRAY_TASK_ID + {group*800}',(root/f'confirmation_{group}.sbatch').read_text())
            self.assertEqual(len([j for j in jobs if j['phase']=='controls']),1)

    def test_freeze_rejects_test_data_and_incomplete_development(self):
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);fixture(root)
            write_json(root/'pilot_summary.json',{'validation_passed':True,'trials':319,'statuses':{'completed':319}})
            with self.assertRaisesRegex(RuntimeError,'pilot'):freeze_confirmation(root)
            self.assertFalse((root/'policy_freeze.json').exists())
            fixture_pilot={'validation_passed':True,'trials':320,'statuses':{'completed':320}}
            write_json(root/'pilot_summary.json',fixture_pilot)
            (root/'confirmation/results/task_00000.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'data already exist'):freeze_confirmation(root)

    def test_failure_first_outcomes_do_not_impute_energy(self):
        rows=[]
        outcomes=[((True,2.),(False,None)),((False,None),(True,3.)),
                  ((False,None),(False,None)),((True,.5),(True,.5)),((True,.1),(True,.2))]
        for d in [1.6,2.4,3.0]:
            for shots in [1000,10000,100000]:
                for seed,(a,b)in enumerate(outcomes):
                    for name,(completed,error)in [('a',a),('b',b)]:
                        rows.append({'distance':d,'shots':shots,'seed':seed,'name':name,'completed':completed,'error_mha':error})
        comparisons=paired(rows,'a','b')
        self.assertEqual(len(comparisons),9)
        for row in comparisons:
            self.assertEqual((row['wins'],row['losses'],row['ties']),(2,1,2))
            self.assertEqual(row['paired_difference_mha']['n'],2)
            self.assertAlmostEqual(row['paired_difference_mha']['mean'],-.05)
            self.assertEqual(row['holm_p'],1.)


if __name__=='__main__':unittest.main()

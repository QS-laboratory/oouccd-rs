"""Render supplemental tables and figures from saved experiment summaries.

This command never reads native trajectories or recomputes primary comparisons.
The summary and frozen analysis sources must remain byte-for-byte unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from analyze_uncertainty_study import describe,plot_primary,plot_trajectories,write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path,columns,rows):
    with path.open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def render(root,plots=True):
    summary=root/'confirmation_summary.json';fingerprint=sha(summary)
    result=json.loads(summary.read_text());freeze=json.loads((root/'policy_freeze.json').read_text())
    provenance=json.loads((root/'confirmation_analysis_provenance.json').read_text())
    expected=freeze['analysis_source_sha256']['analyze_uncertainty_study.py']
    assert provenance['sha256']==expected,'Confirmation was not analyzed with the frozen source'
    assert sha(root/'confirmation_source/scripts/analyze_uncertainty_study.py')==expected
    assert sha(root/'protocol.json')==freeze['protocol_sha256']
    assert result['validation_passed']and len(result['runs'])==result['trials']
    assert len(result['primary_comparisons'])==9 and len(result['controls'])==6
    cells=result['cells'];rows=result['runs'];comparisons=result['primary_comparisons']
    columns=['distance','shots','exact_ucc','variant','trials','completed','mean_error_mha','sd_error_mha',
        'median_error_mha','maximum_error_mha','rmse_mha','chemical_accuracy_count','accepted_updates',
        'harmful_updates','rejected_useful_updates','mean_classical_oo_seconds_per_run','mean_worker_seconds']
    table(root/'confirmation_summary.csv',columns,[{
        **{k:c[k]for k in columns if k in c},
        **{f'{key}_error_mha':c['error_mha'][key]for key in ['mean','sd','median','maximum']},
        'rmse_mha':c['error_mha']['rmse'],'mean_classical_oo_seconds_per_run':c['oo_seconds_per_run']['mean'],
        'mean_worker_seconds':c['worker_seconds']['mean']}for c in cells])
    columns=['distance','shots','a','b','wins','losses','ties','joint_completions','mean_difference_mha',
        'ci95_low_mha','ci95_high_mha','p_two_sided','holm_p']
    table(root/'confirmation_primary.csv',columns,[{**{k:p[k]for k in columns if k in p},
        'joint_completions':p['paired_difference_mha']['n'],'mean_difference_mha':p['paired_difference_mha']['mean'],
        'ci95_low_mha':p['paired_mean_difference_ci95_unadjusted'][0],
        'ci95_high_mha':p['paired_mean_difference_ci95_unadjusted'][1]}for p in comparisons])
    columns=['index','distance','shots','seed','name','status','error_mha','worker_seconds']
    table(root/'confirmation_runs.csv',columns,[{k:r[k]for k in columns}for r in rows])
    columns=['distance','solver','status','error_mha','worker_seconds']
    table(root/'confirmation_noiseless_controls.csv',columns,[{k:r[k]for k in columns}for r in result['controls']])
    if plots:
        (root/'figures').mkdir(exist_ok=True)
        plot_primary(root,comparisons)
        plot_trajectories(root,'confirmation_supplement',rows,cells,result['controls'])
    def fmt(value):return '—'if value is None else f'{value:.5g}'
    lines=['# Supplemental presentation of frozen confirmation results','',
        'These tables and figures render the saved confirmation summary. Primary tests and bootstrap intervals are copied without recomputation.',
        'Endpoints are ideal circuit-state energies minus FCI after ten macro iterations. Exact energies are diagnostics and never select optimization updates.',
        'Scientific failures remain in completion-first comparisons and accuracy denominators; endpoint means and intervals condition on joint completions.','',
        '## Prespecified guarded-affine comparison','',
        '| R (A) | Shots | SOO wins / Newton wins / ties | Joint completions | Mean SOO−Newton (mHa) | Unadjusted 95% mean interval | Holm p |',
        '|---:|---:|---:|---:|---:|---|---:|']
    for p in comparisons:
        lo,hi=p['paired_mean_difference_ci95_unadjusted']
        lines.append(f"| {p['distance']} | {p['shots']:,} | {p['wins']} / {p['losses']} / {p['ties']} | {p['paired_difference_mha']['n']} | {fmt(p['paired_difference_mha']['mean'])} | [{fmt(lo)}, {fmt(hi)}] | {fmt(p['holm_p'])} |")
    lines+=['','The Holm-adjusted sign test concerns paired win probability, not equality of mean error. The mean intervals are descriptive and unadjusted. Nonsignificance does not establish equivalence.','',
        '![Paired comparisons](figures/confirmation_primary_paired.png)','',
        '## Noiseless controls at the same macro budget','',
        'These ten-macro endpoints are not assumed fully converged.','',
        '| R (A) | Solver | Status | FCI error (mHa) |','|---:|---|---|---:|']
    for c in result['controls']:lines.append(f"| {c['distance']} | {c['solver']} | {c['status']} | {fmt(c['error_mha'])} |")
    lines+=['','![Convergence with noiseless controls](figures/confirmation_supplement_convergence_noisy_ucc.png)','',
        '## Completion, update behavior and classical cost','',
        'Classical OO time includes correction, proposal and validation, excluding exact scoring diagnostics. Worker time includes those diagnostics and state preparation. Costs and update statistics below condition on completed trajectories.','',
        '| R (A) | Shots | Variant | Complete | Within 1.6 mHa / all trials | Accepted / OO updates | Harmful applied OO | Useful proposals rejected | Mean OO seconds/run | Mean worker seconds |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['distance']} | {c['shots']:,} | {c['variant']} | {c['completed']}/{c['trials']} | {c['chemical_accuracy_count']}/{c['trials']} | {c['accepted_updates']}/{c['oo_updates']} | {c['harmful_updates']} | {c['rejected_useful_updates']} | {fmt(c['oo_seconds_per_run']['mean'])} | {fmt(c['worker_seconds']['mean'])} |")
    lines+=['','A low harmful-update rate alone does not demonstrate useful optimization: a policy can reject every proposal. Noisy UCC updates can still increase error even when held-out OO updates are safe.','',
        'Machine-readable tables: confirmation_summary.csv, confirmation_primary.csv, confirmation_runs.csv and confirmation_noiseless_controls.csv.',
        'The primary frozen report, analysis source and summary remain unchanged.']
    (root/'confirmation_supplement.md').write_text('\n'.join(lines)+'\n')
    assert sha(summary)==fingerprint,'Renderer changed the frozen analysis summary'
    assert sha(root/'confirmation_source/scripts/analyze_uncertainty_study.py')==expected
    evidence={'summary_sha256':fingerprint,'frozen_analysis_sha256':expected,'source':str(Path(__file__).resolve()),
        'source_sha256':sha(__file__),'plot_source_sha256':sha(Path(__file__).with_name('analyze_uncertainty_study.py')),
        'python':sys.executable,'primary_statistics_recomputed':False,'additional_quantum_measurements':0,
        'trials':result['trials'],'plots_rendered':plots}
    write_json(root/'confirmation_render_provenance.json',evidence)
    return evidence


def render_ucc(root):
    inputs={name:sha(root/name)for name in ['ucc_extension_summary.json','pilot_summary.json']}
    result=json.loads((root/'ucc_extension_summary.json').read_text());pilot=json.loads((root/'pilot_summary.json').read_text())
    assert result['validation_passed']and result['trials']==40 and len(result['runs'])==40
    baseline={r['index']:r for r in pilot['runs']};paired={p['index']:p for p in result['paired_against_pilot']};groups=[]
    for cell in result['cells']:
        runs=[r for r in result['runs']if r['distance']==cell['distance']and r['shots']==cell['shots']and r['name']==cell['variant']]
        assert len(runs)==5
        pairs=[]
        for r in runs:
            old=baseline[r['baseline_pilot_index']]
            assert all(r[k]==old[k]for k in ['distance','shots','seed','solver','mitigation','acceptance'])
            assert r['mitigation']=='none'and r['acceptance']=='nominal'
            assert not r['exact_ucc']and not old['exact_ucc']and r['affine_ucc']
            difference=paired[r['index']]['error_difference_mha']
            if r['completed']and old['completed']:
                assert abs(difference-(r['error_mha']-old['error_mha']))<1e-10
            else:assert difference is None
            pairs.append({'seed':r['seed'],'raw_error_mha':old['error_mha'],'affine_error_mha':r['error_mha'],
                'difference_mha':difference,'raw_completed':old['completed'],'affine_completed':r['completed']})
        joint=[p for p in pairs if p['raw_completed']and p['affine_completed']]
        groups.append({'distance':cell['distance'],'shots':cell['shots'],'solver':runs[0]['solver'],'n':len(pairs),
            'raw_completed':sum(p['raw_completed']for p in pairs),'affine_completed':sum(p['affine_completed']for p in pairs),
            'joint_completions':len(joint),'raw_error_mha':describe([p['raw_error_mha']for p in joint]),
            'affine_error_mha':describe([p['affine_error_mha']for p in joint]),
            'paired_difference_mha':describe([p['difference_mha']for p in joint]),
            'improved':sum(p['difference_mha']<0 for p in joint),'pairs':pairs})
    lines=['# Development-only affine correction inside UCC','',
        'These forty trajectories are paired with their original raw-UCC pilot baselines. Both use raw, unguarded OO; only the UCC energy-evaluation correction changes.',
        'Each of eight cells has the same five development seeds and ten macro iterations. Endpoint errors are exact circuit-state energy minus FCI in mHa. Each pair has the same configured measurement budget; native results record actual consumption, including early failures.',
        'These exploratory results do not enter the frozen primary confirmation family. Counts of improvements across cells are descriptive: cells reuse the same five seed blocks.','',
        'Means and SDs condition on joint completions. Failed trajectories remain in completion denominators and are not assigned an endpoint error.','',
        '| R (A) | Shots/word | OO proposer | Complete raw / affine | Raw-UCC mean ± SD | Affine-UCC mean ± SD | Paired mean difference | Improved / joint |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '—'if x is None else f'{x:.3f}'
    for g in groups:
        a,b=g['raw_error_mha'],g['affine_error_mha']
        lines.append(f"| {g['distance']} | {g['shots']:,} | {g['solver']} | {g['raw_completed']}/{g['n']} / {g['affine_completed']}/{g['n']} | {fmt(a['mean'])} ± {fmt(a['sd'])} | {fmt(b['mean'])} ± {fmt(b['sd'])} | {fmt(g['paired_difference_mha']['mean'])} | {g['improved']}/{g['joint_completions']} |")
    improved_cells=sum(g['paired_difference_mha']['mean']is not None and g['paired_difference_mha']['mean']<0 for g in groups)
    lines+=['',f"Mean endpoint error decreases in {improved_cells}/{len(groups)} cells, and {sum(g['improved']for g in groups)}/{sum(g['joint_completions']for g in groups)} jointly completed paired outcomes improve. These are development observations about UCC sampling error as well as OO sampling error.",
        'Cross-solver rankings in this five-seed development set do not establish general SOO superiority.',
        'These settings differ from the guarded-affine-OO primary comparison: OO here is raw and unguarded. Directly pooling those policy families would confound the comparison.',
        'Estimated affine weights can introduce parameter-dependent finite-shot bias. The mean corrected UCC objective is not assumed to retain an exact degree-two Fourier profile.','',
        '![Paired raw and corrected UCC endpoints](figures/ucc_extension_paired.png)','',
        'Full run-level results, budgets and execution metadata remain in ucc_extension_summary.json and the native result files.']
    (root/'ucc_extension_paired_report.md').write_text('\n'.join(lines)+'\n')
    columns=['distance','shots','solver','seed','raw_completed','affine_completed','raw_error_mha','affine_error_mha','difference_mha']
    table(root/'ucc_extension_paired.csv',columns,[{**{k:g[k]for k in ['distance','shots','solver']},**p}for g in groups for p in g['pairs']])
    plot_ucc_pairs(root,groups)
    evidence={'input_sha256':inputs,'source_sha256':sha(__file__),'groups':groups,
        'development_only':True,'new_hypothesis_tests':0,'additional_quantum_measurements':0}
    assert all(sha(root/name)==digest for name,digest in inputs.items())
    write_json(root/'ucc_extension_render_provenance.json',evidence)
    return evidence


def plot_ucc_pairs(root,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),squeeze=False)
    for i,distance in enumerate([1.6,3.]):
        for j,shots in enumerate([1000,100000]):
            ax=axes[i,j]
            for k,method in enumerate(['seq','newton_stable']):
                group=next(g for g in groups if g['distance']==distance and g['shots']==shots and g['solver']==method)
                color=['#D55E00','#0072B2'][k]
                for index,pair in enumerate(group['pairs']):
                    if not(pair['raw_completed']and pair['affine_completed']):continue
                    jitter=(index-2)*.025;x=[2*k+jitter,2*k+1+jitter];y=[pair['raw_error_mha'],pair['affine_error_mha']]
                    ax.plot(x,y,color=color,alpha=.35,linewidth=.8)
                    ax.scatter(x,y,color=color,s=18,zorder=3)
                if group['joint_completions']:
                    ax.scatter([2*k,2*k+1],[group['raw_error_mha']['mean'],group['affine_error_mha']['mean']],
                        marker='_',s=220,color='black',zorder=4)
            ax.axhline(1.6,color='.4',linestyle=':')
            ax.set(title=f'R={distance:g} A, {shots:,} shots',ylabel='Final exact-state FCI error (mHa)',yscale='log',
                xticks=[0,1,2,3],xticklabels=['SOO raw UCC','SOO affine UCC','Newton raw UCC','Newton affine UCC'])
            ax.tick_params(axis='x',labelrotation=25,labelsize=8);ax.grid(alpha=.2)
    fig.suptitle('Development pilot: raw, unguarded OO; five paired seeds; black bars show means')
    fig.tight_layout();(root/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(root/'figures'/f'ucc_extension_paired.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--phase',choices=['confirmation','ucc_extension'],default='confirmation')
    args=parser.parse_args();root=args.work.resolve()
    result=render(root)if args.phase=='confirmation'else render_ucc(root)
    print(json.dumps({k:v for k,v in result.items()if k!='groups'}))


if __name__=='__main__':main()

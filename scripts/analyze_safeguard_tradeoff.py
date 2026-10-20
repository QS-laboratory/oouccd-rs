"""Describe the fixed-state safeguard's safety/utility tradeoff by shot count.

Only completed development summaries enter this analysis. Pooled rates across
different frozen states are descriptive; no new confirmatory tests are run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    source=root/'fixed_summary.json';data=json.loads(source.read_text())
    assert data['validation_passed']and data['independent_batches']==63000 and len(data['cells'])==63
    cells=[];maximum_shared_proposal_error=0.
    for cell in data['cells']:
        variants={v['variant']:v for v in cell['variants']}
        for method in ['seq','newton_stable']:
            split=variants[method+'_affine_splitnominal'];guard=variants[method+'_affine_heldout']
            assert split['trials']==guard['trials']==split['completed']==guard['completed']==1000
            # The control reserves the same validation shots and uses the same
            # training proposal. Its applied proposal summaries must match.
            for field in ['mean','sd','median','maximum','rmse']:
                error=abs(split['applied_delta_mha'][field]-guard['proposed_delta_mha'][field])
                maximum_shared_proposal_error=max(maximum_shared_proposal_error,error)
                assert error<1e-10,(cell['snapshot'],method,field)
            cells.append({'snapshot':cell['snapshot'],'distance':cell['distance'],'shots':cell['shots'],'method':method,
                'trials':guard['trials'],'accepted':guard['accepted'],'harmful_accepted':guard['harmful_updates'],
                'useful_rejected':guard['rejected_useful'],'split_harmful':split['harmful_updates'],
                'split_mean_delta_mha':split['applied_delta_mha']['mean'],
                'guarded_mean_delta_mha':guard['applied_delta_mha']['mean']})
    groups=[]
    for method in ['seq','newton_stable']:
        for shots in [1000,10000,100000]:
            selected=[c for c in cells if c['method']==method and c['shots']==shots];assert len(selected)==21
            totals={key:sum(c[key]for c in selected)for key in ['trials','accepted','harmful_accepted','useful_rejected','split_harmful']}
            n=totals['trials'];accepted=totals['accepted']
            groups.append({'method':method,'shots':shots,**totals,'acceptance_rate':accepted/n,
                'harmful_unconditional_rate':totals['harmful_accepted']/n,
                'harmful_given_accepted_rate':totals['harmful_accepted']/accepted if accepted else None,
                'useful_rejected_fraction_all_proposals':totals['useful_rejected']/n,
                **{key:sum(c[key]*c['trials']for c in selected)/n for key in ['split_mean_delta_mha','guarded_mean_delta_mha']}})
    result={'validation_passed':True,'development_only':True,'original_batches':63000,'cells':cells,'groups':groups,
        'maximum_shared_proposal_summary_error_mha':maximum_shared_proposal_error,
        'confirmation_outcomes_inspected':False,'new_hypothesis_tests':0,'additional_quantum_measurements':0,
        'interpretation':'The frequentist false-acceptance bound is unconditional on accepting; it is not a posterior or a conditional harmful-given-accepted bound.'}
    (root/'safeguard_tradeoff_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with(root/'safeguard_tradeoff_cells.csv').open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    lines=['# Fixed-state safeguard safety and utility','',
        'This secondary descriptive analysis uses completed fixed-state development data, without confirmation endpoints or new quantum measurements.',
        'Affine splitnominal and heldout use the same 80% training allocation and the same proposal; only heldout applies the validation decision.',
        'Each row pools 21 frozen states × 1,000 independent batches. These mixtures are not treated as a single homogeneous binomial experiment.','',
        '## Acceptance and harmful updates','',
        '| Proposer | Shots/word | Accepted / all | Harmful accepted | Harmful / all | Harmful / accepted | Useful proposals rejected |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        conditional='—'if g['harmful_given_accepted_rate']is None else f"{100*g['harmful_given_accepted_rate']:.2f}%"
        lines.append(f"| {g['method']} | {g['shots']:,} | {g['accepted']}/{g['trials']} | {g['harmful_accepted']} | {100*g['harmful_unconditional_rate']:.4f}% | {conditional} | {g['useful_rejected']} |")
    lines+=['','The nominal per-proposal false-acceptance budget is 0.5%. It bounds the joint event that an update is accepted and truly harmful, conditional on the prior history; it does not bound the fraction of accepted updates that are harmful.',
        'For example, at 1,000 shots the 14 accepted SOO updates are all harmful, while their frequency among all 21,000 proposals is only 0.0667%. This does not contradict the stated guarantee.',
        'Useful rejection uses a proposed true decrease exceeding 1e-10 Ha; harmful acceptance uses an applied true increase exceeding 1e-10 Ha. These are diagnostics unavailable to the optimizer.',
        'Observed small harmful frequencies are consistent with, but do not prove, the theoretical bound. The algebraic confidence argument is documented separately.','',
        '## Applied true energy changes','',
        'Negative is beneficial. Rejections contribute zero. The control below retains the same training-shot budget, isolating the acceptance decision.','',
        '| Proposer | Shots/word | Unguarded split mean (mHa) | Guarded mean (mHa) |','|---|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['method']} | {g['shots']:,} | {g['split_mean_delta_mha']:+.6f} | {g['guarded_mean_delta_mha']:+.6f} |")
    lines+=['','At 1,000 shots, both guarded averages still increase slightly because the rare accepted harmful steps outweigh the accepted benefits.',
        'At 10,000 shots, the guard turns the SOO average from positive to negative but reduces the mean decrease obtained by Newton. At 100,000 shots, it slightly improves SOO mean decrease and slightly reduces Newton mean decrease on this fixture collection.',
        'This relative benefit of safeguarding SOO is not an absolute SOO advantage: Newton still has the larger guarded mean decrease at 10,000 and 100,000 shots, and end-to-end trajectories must be compared separately.',
        'The proposed-update summary statistics match between splitnominal and heldout in every cell. This check covers the saved summaries; saved per-batch records remain the evidence for individual proposals.','',
        '![Safety and utility tradeoff](figures/safeguard_tradeoff.png)','',
        'All 126 snapshot/proposer/shot rows are retained in safeguard_tradeoff_cells.csv. No policy, allocation or primary comparison is changed by this analysis.']
    (root/'safeguard_tradeoff_report.md').write_text('\n'.join(lines)+'\n')
    plot(root,groups)
    (root/'safeguard_tradeoff_provenance.json').write_text(json.dumps({'source':str(Path(__file__).resolve()),
        'source_sha256':sha(__file__),'input_sha256':sha(source),'python':sys.executable,
        'primary_statistics_recomputed':False,'confirmation_outcomes_inspected':False},indent=2)+'\n')
    return result


def plot(root,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5));colors={'seq':'#D55E00','newton_stable':'#0072B2'}
    for method in ['seq','newton_stable']:
        rows=[g for g in groups if g['method']==method];x=[g['shots']for g in rows];color=colors[method]
        axes[0,0].plot(x,[100*g['acceptance_rate']for g in rows],'-o',color=color,label=method)
        axes[0,1].plot(x,[100*g['useful_rejected_fraction_all_proposals']for g in rows],'-o',color=color,label=method)
        ax=axes[1,0]if method=='seq'else axes[1,1]
        for key,label,style in [('split_mean_delta_mha','Split, unguarded','--o'),('guarded_mean_delta_mha','Heldout','-o')]:
            ax.plot(x,[g[key]for g in rows],style,label=label)
        ax.axhline(0,color='.4',linestyle=':');ax.set(title=method,ylabel='Mean applied true OO change (mHa)');ax.legend()
    axes[0,0].set(title='Accepted proposals',ylabel='Percent of all proposals');axes[0,0].legend()
    axes[0,1].set(title='Useful proposals rejected',ylabel='Percent of all proposals');axes[0,1].legend()
    for ax in axes.flat:ax.set(xscale='log',xlabel='Shots per Pauli word',xticks=[1000,10000,100000]);ax.grid(alpha=.2)
    fig.suptitle('Affine correction: fixed-state safety/utility tradeoff; 21,000 batches per shot level')
    fig.tight_layout();(root/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(root/'figures'/f'safeguard_tradeoff.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.work.resolve())
    print(json.dumps({k:v for k,v in result.items()if k not in ['cells','groups']}))


if __name__=='__main__':main()

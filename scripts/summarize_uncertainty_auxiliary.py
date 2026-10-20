"""Analyze matched physical reconstructions and joint synthetic uncertainty draws."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from analyze_uncertainty_study import describe,execution,write_json


def reconstructed_updates(root,task):
    study=json.loads((root/'study.json').read_text());binary=study['binaries'].get('reconstructed_oo',study['binaries'].get('frozen_oo'))
    if not binary:raise RuntimeError('Prepare the preserved reconstructed-OO evaluator before final reconstruction analysis')
    if hashlib.sha256(Path(binary['path']).read_bytes()).hexdigest()!=binary['sha256']:raise RuntimeError('Reconstruction-OO evaluator changed')
    output=Path(task['output']).with_suffix('.oo.jsonl');marker=output.with_suffix('.completed.json')
    provenance=output.with_suffix('.provenance.json')
    fingerprints={'binary_sha256':binary['sha256'],
        'input_sha256':hashlib.sha256(Path(task['output']).read_bytes()).hexdigest(),
        'snapshot_sha256':hashlib.sha256(Path(task['snapshot']).read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(Path(task['data']).read_bytes()).hexdigest()}
    if marker.exists():
        if not provenance.exists():raise RuntimeError('Cached reconstruction OO result has no provenance')
        saved=json.loads(provenance.read_text())
        if any(saved.get(k)!=v for k,v in fingerprints.items()):raise RuntimeError('Cached reconstruction OO inputs changed')
        if saved.get('result_sha256')!=hashlib.sha256(output.read_bytes()).hexdigest():
            raise RuntimeError('Cached reconstruction OO output changed')
    if not marker.exists():
        if output.exists():output.rename(output.with_name(output.name+f'.partial-{time.time_ns()}'))
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',
            LD_LIBRARY_PATH=study['library_path']+os.pathsep+env.get('LD_LIBRARY_PATH',''))
        command=[binary['path'],'reconstructed','--snapshot',task['snapshot'],'--reconstruction',task['output'],'--output',str(output)]
        with(root/'logs'/f"reconstructed-oo-{task['index']:03}.log").open('w')as log:
            subprocess.run(command,stdout=log,stderr=log,env=env,check=True)
        write_json(provenance,{'command':command,**fingerprints,
            'result_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    if json.loads(marker.read_text())['trials']!=task['replicates']:raise RuntimeError('Incomplete reconstruction OO postprocessing')
    originals={}
    for line in Path(task['data']).open():
        r=json.loads(line)
        if r['record_type']!='trial':continue
        if r['index']>=task['replicates']:break
        originals[r['index']]=r
    groups=defaultdict(list)
    for line in output.open():
        r=json.loads(line)
        if r['record_type']!='trial':continue
        original=originals[r['index']]
        if r['seed']!=original['seed']:raise RuntimeError('Reconstructed OO batch pairing changed')
        if len(r['outcomes'])!=2*len(task['methods']):raise RuntimeError('Missing reconstruction OO comparison')
        for outcome in r['outcomes']:
            if outcome['status']=='completed':
                baseline=next(o for o in original['outcomes']if o['variant']==outcome['solver']+'_none_nominal')
                affine=next(o for o in original['outcomes']if o['variant']==outcome['solver']+'_affine_nominal')
                outcome['minus_raw_delta_mha']=1000*(outcome['exact_delta_ha']-baseline['applied_exact_delta_ha'])
                outcome['minus_affine_delta_mha']=1000*(outcome['exact_delta_ha']-affine['applied_exact_delta_ha'])
            groups[outcome['method'],outcome['solver']].append(outcome)
    summaries=[]
    for(method,solver),group in groups.items():
        done=[x for x in group if x['status']=='completed']
        if len(group)!=task['replicates']:raise RuntimeError('Missing reconstructed-OO outcomes')
        summaries.append({'method':method,'solver':solver,'requested':len(group),'completed':len(done),
            'harmful_updates':sum(r['exact_delta_ha']>1e-10 for r in done),
            'exact_delta_mha':describe([1000*r['exact_delta_ha']for r in done]),
            'paired_minus_raw_delta_mha':describe([r['minus_raw_delta_mha']for r in done]),
            'paired_minus_affine_delta_mha':describe([r['minus_affine_delta_mha']for r in done]),
            'classical_oo_seconds':describe([r['classical_oo_seconds']for r in done])})
    return summaries


def reconstruction_fit_summary(method,group):
    done=[r for r in group if r['completed']]
    return {'method':method,'requested':len(group),'completed':len(done),
        'statuses':{state:sum(r.get('status','failed')==state for r in group)
            for state in sorted({r.get('status','failed')for r in group})},
        'energy_error_mha':describe([r['score']['energy_error_ha']*1000 for r in done]),
        'rdm_squared_error':describe([r['score']['rdm_squared_error']for r in done]),
        'matched_raw_energy_error_mha':describe([r['baseline']['raw_energy_error_mha']for r in done]),
        'matched_affine_energy_error_mha':describe([r['baseline']['affine_energy_error_mha']for r in done]),
        'matched_raw_rdm_squared_error':describe([r['baseline']['raw_rdm_squared_error']for r in done]),
        'matched_affine_rdm_squared_error':describe([r['baseline']['affine_rdm_squared_error']for r in done]),
        'paired_energy_squared_error_minus_raw_mha2':describe([
            (r['score']['energy_error_ha']*1000)**2-r['baseline']['raw_energy_error_mha']**2 for r in done]),
        'paired_energy_squared_error_minus_affine_mha2':describe([
            (r['score']['energy_error_ha']*1000)**2-r['baseline']['affine_energy_error_mha']**2 for r in done]),
        'seconds':describe([r['seconds']for r in group]),
        'minimum_metric_eigenvalue':min((min(r['validation']['minimum_eigenvalues'].values())for r in done),default=None),
        'maximum_constraint_residual':max((r['validation']['max_constraint_residual']for r in done),default=None),
        'failures':[r.get('error')for r in group if not r['completed']]}


def reconstruct(root):
    tasks=json.loads((root/"reconstruction_manifest.json").read_text());cells=[]
    for task in tasks:
        if execution(task)["exit_code"]:raise RuntimeError("Reconstruction worker failed")
        marker=Path(task["output"]).with_suffix(".completed.json")
        if not marker.exists():raise RuntimeError("Missing reconstruction completion marker")
        groups=defaultdict(list);raw=[];affine=[];raw_rdm=[];affine_rdm=[];count=0;header=None
        for line in Path(task["output"]).read_text().splitlines():
            row=json.loads(line)
            if row['record_type']=='metadata':header=row;continue
            if row["record_type"]!="trial":continue
            count+=1;raw.append(row["raw_energy_error_ha"]*1000);affine.append(row["affine_energy_error_ha"]*1000)
            raw_rdm.append(row["rdm_raw_squared_error"]);affine_rdm.append(row["rdm_affine_squared_error"])
            if len(row["results"])!=len(task["methods"]):raise RuntimeError("Missing reconstruction method")
            for r in row["results"]:
                groups[r['method']].append({**r,'baseline':{
                    'raw_energy_error_mha':row['raw_energy_error_ha']*1000,
                    'affine_energy_error_mha':row['affine_energy_error_ha']*1000,
                    'raw_rdm_squared_error':row['rdm_raw_squared_error'],
                    'affine_rdm_squared_error':row['rdm_affine_squared_error']}})
        if count!=task["replicates"]:raise RuntimeError("Incomplete reconstruction batches")
        methods=[]
        if header is None:raise RuntimeError('Missing reconstruction setup metadata')
        for method,group in groups.items():
            method_summary=reconstruction_fit_summary(method,group)
            method_summary['model_build_seconds']=header['build_seconds'][method]
            methods.append(method_summary)
        cells.append({"snapshot":task["name"],"distance":task["distance"],"shots":task["shots"],"matched_batches":count,
            "raw_energy_error_mha":describe(raw),"affine_energy_error_mha":describe(affine),
            "raw_rdm_squared_error":describe(raw_rdm),"affine_rdm_squared_error":describe(affine_rdm),"methods":methods,
            'oo_update_comparison':reconstructed_updates(root,task)})
    result={"validation_passed":True,"tasks":len(tasks),"cells":cells,
        "interpretation":"Reconstruction errors concern the prepared frozen state, not optimized circuit-state accuracy. DQG is a necessary-condition relaxation; failed fits are not assigned finite errors."}
    aggregates=[]
    for method in ['spatial_psd','dqg','mle']:
        for solver in ['seq','newton_stable']:
            group=[r for c in cells for r in c['oo_update_comparison']if r['method']==method and r['solver']==solver]
            n=sum(r['completed']for r in group)
            aggregates.append({'method':method,'solver':solver,'requested':sum(r['requested']for r in group),
                'completed':n,'harmful_updates':sum(r['harmful_updates']for r in group),
                **{field:sum(r[key]['mean']*r['completed']for r in group if r['completed'])/n if n else None
                   for field,key in [('mean_true_delta_mha','exact_delta_mha'),
                    ('mean_paired_minus_raw_delta_mha','paired_minus_raw_delta_mha'),
                    ('mean_paired_minus_affine_delta_mha','paired_minus_affine_delta_mha')]}})
    result['oo_update_aggregates']=aggregates
    reference=root/"verification/small_reference_completed.json"
    result["small_sector_reference"]=json.loads(reference.read_text())if reference.exists()else None
    write_json(root/"reconstruction_summary.json",result)
    lines=["# Physical RDM reconstruction benchmarks","",
        "Every estimator uses the same original Pauli counts. No additional quantum measurements enter reconstruction.",
        "Completed-only error statistics must be read with fit-completion counts. Numerical failures are preserved.",
        "Raw and affine baseline RMSEs in each method row use exactly the same successfully fitted batches; different methods can have different completion subsets.","",
        "| Snapshot | Shots | Method | Complete | Energy bias (mHa) | Energy RMSE | Matched raw RMSE | Matched affine RMSE | Mean RDM squared error | Mean seconds |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        for r in c["methods"]:
            def fmt(value):return "—"if value is None else f"{value:.5g}"
            lines.append(f"| {c['snapshot']} | {c['shots']:,} | {r['method']} | {r['completed']}/{r['requested']} | {fmt(r['energy_error_mha']['mean'])} | {fmt(r['energy_error_mha']['rmse'])} | {fmt(r['matched_raw_energy_error_mha']['rmse'])} | {fmt(r['matched_affine_energy_error_mha']['rmse'])} | {fmt(r['rdm_squared_error']['mean'])} | {fmt(r['seconds']['mean'])} |")
    lines += ["","The numerical acceptance tolerance is 5e-5 for constraint/Pauli-bound violations and negative metric eigenvalues.",
        "These feasibility checks do not prove exact N-representability for a DQG relaxation. Corrected estimator energy is not the circuit's exact energy.",
        "Mean fit time includes canonicalization when incurred and uses subsequent warm starts. One-time model setup costs are recorded separately as model_build_seconds in the JSON summary.",
        "The H2 fixed-sector density-matrix reference and solver-comparison smoke outputs are retained in verification/."]
    lines += ['','## Actual OO updates from the reconstructed RDMs','',
        'The JSON summaries include true energy changes for both SOO and Newton, paired against raw and affine nominal updates on the identical original measurement batch.',
        'These nominal postprocessing comparisons use all available counts; no held-out safety claim is made. Reconstruction/optimizer failures remain in completion denominators, and energy statistics are conditional on completed fits.',
        'No new quantum measurements or state preparations are required for these frozen-state comparisons.']
    lines+=['','The following pooled means describe this fixed fixture collection and condition on completed fits/updates. Completion subsets can differ across reconstruction methods. Negative changes are beneficial; paired baseline differences use the identical successful-fit batches.','',
        '| Estimator | OO proposer | Completed | Harmful | Mean true change (mHa) | Paired change minus raw | Paired change minus affine |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in aggregates:
        def fmt(value):return '—'if value is None else f'{value:+.5f}'
        lines.append(f"| {g['method']} | {g['solver']} | {g['completed']}/{g['requested']} | {g['harmful_updates']} | {fmt(g['mean_true_delta_mha'])} | {fmt(g['mean_paired_minus_raw_delta_mha'])} | {fmt(g['mean_paired_minus_affine_delta_mha'])} |")
    lines+=['','These update metrics must be assessed separately from single-energy estimator RMSE. Orbital updates depend on profile shapes and relative energies from the shared RDM; lower marginal-energy RMSE alone does not establish better rotations.']
    lines += ['','![Matched energy RMSE ratios](figures/reconstruction_matched_rmse_ratios.png)',
        'Each point compares a physical estimator with affine correction on the exact same completed-fit subset. A ratio below one favors the physical estimator. Completion counts remain necessary when interpreting these conditional comparisons.']
    (root/"reconstruction_report.md").write_text("\n".join(lines)+"\n")
    plot_reconstruction(root,cells)
    return result


def bootstrap(root):
    tasks=json.loads((root/"bootstrap_manifest.json").read_text());cells=[]
    for task in tasks:
        if execution(task)["exit_code"]:raise RuntimeError("Bootstrap worker failed")
        marker=Path(task["output"]).with_suffix(".completed.json")
        if not marker.exists():raise RuntimeError("Missing bootstrap completion marker")
        groups=defaultdict(list);proposals={};header=None
        for line in Path(task["output"]).read_text().splitlines():
            r=json.loads(line)
            if r["record_type"]=="metadata":header=r;continue
            key=(r["method"],r["affine"],r["proposal_kind"])
            if r["record_type"]=="proposal":proposals[key]=r
            else:
                # Rotations, rather than averaged angles, preserve periodic and
                # multimodal sweep outcomes. Every draw reruns the whole sweep.
                rotation=np.asarray(r["rotation"]["data"]).reshape(r["rotation"]["dim"])
                if np.max(np.abs(rotation @ rotation.T-np.eye(len(rotation))))>=1e-8:
                    raise RuntimeError("Synthetic sweep lost orbital orthogonality")
                groups[(*key,r["scheme"])].append(r)
        if header is None or header["modeled_additional_quantum_shots"]!=0:raise RuntimeError("Incorrect bootstrap measurement accounting")
        if len(proposals)!=12 or len(groups)!=24:raise RuntimeError("Missing matched uncertainty variants")
        summaries=[]
        for key,group in groups.items():
            if len(group)!=task["draws"]:raise RuntimeError("Incomplete synthetic draws")
            proposal=proposals[key[:3]]
            if proposal["heldout_accepted"]and proposal["validation_upper_bound_ha"]>=0:
                raise RuntimeError("Original proposal violated holdout decision")
            values=np.asarray([r["fixed_proposal_draw_delta_ha"]*1000 for r in group])
            summaries.append({"method":key[0],"affine":key[1],"proposal_kind":key[2],"scheme":key[3],
                "synthetic_draws":len(group),"original_proposal":proposal,
                "fixed_proposal_delta_mha":describe(values),"fixed_proposal_quantiles_025_975":np.quantile(values,[.025,.975]).tolist(),
                "diagnostic_probability_decrease":float(np.mean(values<0)),
                "replayed_oracle_delta_mha":describe([r["oracle_delta_ha"]*1000 for r in group]),
                "replay_seconds":describe([r["seconds"]for r in group])})
        cells.append({"snapshot":task["name"],"distance":task["distance"],"shots":task["shots"],
            "original_measurement_batches":1,"summaries":summaries})
    result={"validation_passed":True,"tasks":len(tasks),"original_measurement_batches":len(tasks),
        "additional_quantum_shots":0,"cells":cells,
        "interpretation":"Synthetic draw counts are Monte Carlo resolution, not independent experimental trials. Projected Beta is not a physically constrained posterior; intervals are diagnostics, not frequentist coverage guarantees."}
    write_json(root/"bootstrap_summary.json",result)
    draws=sorted({t['draws']for t in tasks})
    lines=["# Joint whole-sweep uncertainty diagnostics","",
        f"{len(tasks)} original measurement batches; synthetic draws per scheme/proposal variant: {draws}. Additional quantum shots: zero.",
        "All energies and coordinates in a draw share one reconstructed RDM; the complete adaptive sweep is rerun. Nonlinear estimated affine weights are recomputed for parametric bootstrap draws.",
        "Beta draws use a fixed fitted affine map when requested. This projected diagnostic model is not the exact posterior under physical constraints.",
        "Posterior-mean proposals optimize the corresponding mean RDM; risk proposals use beta=2 for both SOO profiles and a Newton-ray search. Each original proposal has the same independent held-out safety test.",
        "Angle averages are not used. Full rotations and all draw-level outputs are retained.","",
        "The summaries contain probabilities of energy decrease, selected-proposal distributions, oracle replay changes, and computational cost. These probabilities do not become more statistically certain by treating synthetic draws as new experiments."]
    (root/"bootstrap_report.md").write_text("\n".join(lines)+"\n")
    plot_bootstrap(root,cells)
    return result


def plot_reconstruction(root,cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=root/"figures";output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),squeeze=False)
    for ax,shots in zip(axes[0],[1000,10000,100000]):
        groups=defaultdict(list)
        for c in cells:
            if c["shots"]!=shots:continue
            groups["raw"].append(c["raw_energy_error_mha"]["rmse"]);groups["affine"].append(c["affine_energy_error_mha"]["rmse"])
            for r in c["methods"]:
                if r["energy_error_mha"]["rmse"]is not None:groups[r["method"]].append(r["energy_error_mha"]["rmse"])
        names=list(groups)
        for i,name in enumerate(names):ax.scatter([i]*len(groups[name]),groups[name],label=name,alpha=.6)
        ax.set(xticks=range(len(names)),xticklabels=names,yscale="log",ylabel="Energy RMSE (mHa)",title=f"{shots:,} shots")
        ax.tick_params(axis="x",labelrotation=40);ax.grid(alpha=.2)
    fig.suptitle("Matched reconstruction benchmarks; one point per fixed snapshot")
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"reconstruction_rmse.{ext}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),squeeze=False)
    for ax,shots in zip(axes[0],[1000,10000,100000]):
        groups=defaultdict(list)
        for c in cells:
            if c['shots']!=shots:continue
            for r in c['methods']:
                baseline=r['matched_affine_energy_error_mha']['rmse'];value=r['energy_error_mha']['rmse']
                if baseline is not None and baseline>0 and value is not None:groups[r['method']].append(value/baseline)
        names=list(groups)
        for i,name in enumerate(names):ax.scatter([i]*len(groups[name]),groups[name],alpha=.7)
        ax.axhline(1,color='.4',linestyle=':')
        ax.set(xticks=range(len(names)),xticklabels=names,yscale='log',ylabel='Physical / matched-affine energy RMSE',title=f'{shots:,} shots')
        ax.tick_params(axis='x',labelrotation=35);ax.grid(alpha=.2)
    fig.suptitle('Same-count comparisons conditional on completed fits; one point per snapshot')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'reconstruction_matched_rmse_ratios.{ext}',dpi=170)
    plt.close(fig)


def plot_bootstrap(root,cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=root/"figures";output.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5));labels=[];values=[]
    for kind in ("nominal","robust","posterior_mean"):
        for method in ("seq","newton_stable"):
            selected=[r["replayed_oracle_delta_mha"]["mean"]for c in cells for r in c["summaries"]
                if r["proposal_kind"]==kind and r["method"]==method and r["affine"]and r["scheme"]=="parametric_bootstrap"]
            labels.append(method+"\n"+kind);values.append(selected)
    ax.boxplot(values,tick_labels=labels,whis=(5,95));ax.axhline(0,color=".5",linestyle=":")
    ax.set(ylabel="Mean exact energy change of synthetic sweep replays (mHa)",title="Exploratory whole-sweep bootstrap: affine-corrected proposals")
    ax.grid(alpha=.2);fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"bootstrap_replay.{ext}",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--work",type=Path,required=True)
    p.add_argument("--phase",choices=["reconstruction","bootstrap"],required=True);args=p.parse_args();root=args.work.resolve()
    result=reconstruct(root)if args.phase=="reconstruction"else bootstrap(root)
    write_json(root/f"{args.phase}_analysis_provenance.json",{"source":str(Path(__file__).resolve()),
        "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"python":sys.executable,"numpy":np.__version__})
    print(json.dumps({"phase":args.phase,"tasks":result["tasks"],"validation_passed":result["validation_passed"]}))


if __name__=="__main__":main()

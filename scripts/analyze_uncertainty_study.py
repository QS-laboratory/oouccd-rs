"""Validate staged OO experiments and analyze independent trial-seed outcomes."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def describe(values):
    x=np.asarray(values,dtype=float)
    if len(x)==0:return {"n":0,"mean":None,"sd":None,"median":None,"maximum":None,"rmse":None}
    return {"n":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=1))if len(x)>1 else None,
        "median":float(np.median(x)),"maximum":float(x.max()),"rmse":float(np.sqrt(np.mean(x*x)))}


def interval(successes,trials):
    if not trials:return [None,None]
    ci=binomtest(successes,trials).proportion_ci(confidence_level=.95,method="exact")
    return [float(ci.low),float(ci.high)]


def execution(task):
    path=Path(task["output"]);record=path.with_suffix(path.suffix+".execution.json")
    if not record.exists():raise RuntimeError(f"No terminal execution record: {path}")
    return json.loads(record.read_text())


def analyze_fixed(root):
    tasks=json.loads((root/"fixed_manifest.json").read_text());cells=[];failures=[];validation=True
    for task in tasks:
        record=execution(task)
        if record["exit_code"]:raise RuntimeError(f"Fixed-state process failed: {task['index']}")
        completed=Path(task["output"]).with_suffix(".completed.json")
        if not completed.exists():raise RuntimeError("Missing fixed-state completion marker")
        groups=defaultdict(list);coefficients=[];estimated=[];raw_energy=[];affine_energy=[];raw_rdm=[];affine_rdm=[]
        with Path(task["output"]).open()as source:
            header=json.loads(next(source))
            for line in source:
                row=json.loads(line)
                if row["counts"]["shots"]!=task["shots"]:raise RuntimeError("Fixed-state shot budget mismatch")
                coefficients.append(row["profile_coefficients"]);estimated.append(row["profile_estimated_covariance"])
                raw_energy.append(row["raw_energy_error_ha"]*1000);affine_energy.append(row["affine_energy_error_ha"]*1000)
                raw_rdm.append(row["rdm_raw_squared_error"]);affine_rdm.append(row["rdm_affine_squared_error"])
                if len(row["outcomes"])!=12:raise RuntimeError("Missing fixed-state ablation")
                for outcome in row["outcomes"]:
                    groups[outcome["variant"]].append(outcome)
                    if outcome["status"]!="completed":
                        validation=False
                        if len(failures)<50:failures.append({"task":task["index"],"seed":row["seed"],**outcome})
        if len(coefficients)!=task["replicates"]:raise RuntimeError("Incomplete fixed-state measurement batches")
        true_cov=np.asarray(header["profile_true_covariance"]);empirical=np.cov(np.asarray(coefficients),rowvar=False,ddof=1)
        scale=np.linalg.norm(true_cov)
        relative=float(np.linalg.norm(empirical-true_cov)/scale)if scale>1e-20 else float(np.linalg.norm(empirical-true_cov))
        if relative>.5:validation=False
        summaries=[]
        for name,group in groups.items():
            done=[x for x in group if x["status"]=="completed"]
            harmful=sum(x["applied_exact_delta_ha"]>1e-10 for x in done)
            accepted=sum(x["statistics"]["accepted"]for x in done)
            rejected_useful=sum(not x["statistics"]["accepted"]and x["proposed_exact_delta_ha"] < -1e-10 for x in done)
            summaries.append({"variant":name,"trials":len(group),"completed":len(done),"harmful_updates":harmful,
                "harmful_rate_ci95":interval(harmful,len(done)),"accepted":accepted,"rejected_useful":rejected_useful,
                "applied_delta_mha":describe([x["applied_exact_delta_ha"]*1000 for x in done]),
                "proposed_delta_mha":describe([x["proposed_exact_delta_ha"]*1000 for x in done]),
                "classical_oo_seconds":describe([x["statistics"]["total_seconds"]for x in done])})
        cells.append({"snapshot":task["name"],"distance":task["distance"],"shots":task["shots"],
            "replicates":len(coefficients),"covariance_relative_frobenius_error":relative,
            "estimated_covariance_relative_error":float(np.linalg.norm(np.mean(estimated,axis=0)-true_cov)/max(scale,1e-30)),
            "raw_energy_error_mha":describe(raw_energy),"affine_energy_error_mha":describe(affine_energy),
            "raw_rdm_squared_error":describe(raw_rdm),"affine_rdm_squared_error":describe(affine_rdm),"variants":summaries})
    result={"validation_passed":validation,"tasks":len(tasks),"independent_batches":sum(c["replicates"]for c in cells),
        "cells":cells,"failures":failures,"interpretation":"Snapshot-specific noise experiments; no-update has zero energy change. Harm rate alone is not optimizer utility."}
    write_json(root/"fixed_summary.json",result)
    lines=["# Fixed-state RDM and orbital-update experiments","",f"{result['independent_batches']:,} independent batches across {len(cells)} fixed-state/shot cells.",
        f"Validation passed: {validation}. All twelve variants share each batch.","",
        "Confidence bounds use one complete proposal, alpha=0.005. Reported binomial intervals concern repeated batches at the specified fixed state.","",
        "| Snapshot | Shots | Covariance relative error | Raw energy RMSE (mHa) | Affine RMSE |","|---|---:|---:|---:|---:|"]
    for c in cells:lines.append(f"| {c['snapshot']} | {c['shots']:,} | {c['covariance_relative_frobenius_error']:.3f} | {c['raw_energy_error_mha']['rmse']:.3f} | {c['affine_energy_error_mha']['rmse']:.3f} |")
    lines += ["","All per-variant harmful-update, useful-rejection, energy-change and classical-cost statistics are in fixed_summary.json.",
        "A lower harmful-update rate with no energy decrease is not evidence of a useful optimizer improvement."]
    (root/"fixed_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"phase":"fixed","validation_passed":validation,"batches":result["independent_batches"]}))


def trajectory_rows(root,phase):
    study=json.loads((root/"study.json").read_text());tasks=json.loads((root/f"{phase}_manifest.json").read_text());rows=[]
    for task in tasks:
        e=execution(task);path=Path(task["output"])
        if task.get("config_sha256")and hashlib.sha256(Path(task["config"]).read_bytes()).hexdigest()!=task["config_sha256"]:
            raise RuntimeError("Frozen configuration changed")
        native=json.loads(path.read_text())if path.exists()else None
        status="process_failure"if e["exit_code"]else native["status"]if native else"missing_result"
        if status in ("process_failure","missing_result"):
            raise RuntimeError(f"Infrastructure/process failure must be resolved before inference: {path}")
        fci=study["references"][str(task["distance"])]["fci_energy_ha"]
        completed=status=="completed";error=None;records=[]
        if native:
            if native.get("ucc_mitigation","none")!=("affine"if task.get("affine_ucc")else"none"):
                raise RuntimeError("UCC mitigation mode does not match the recorded experiment")
            for point in native["iterations"]:
                if point["orthogonality_error"]>=1e-8 or point["exact_energy_ha"]<fci-1e-7:
                    raise RuntimeError("Unphysical exact trajectory")
            if completed:
                if len(native["iterations"])!=11 or native["rdm_evaluations"]!=3831:raise RuntimeError("Incorrect fixed macro/evaluation budget")
                expected=21 if task.get("exact_ucc") else 3831
                if native["modeled_shots"]!=expected*2778*(task["shots"]or 0):raise RuntimeError("Modeled measurement budget mismatch")
                error=(native["iterations"][-1]["exact_energy_ha"]-fci)*1000
            for point in native["iterations"][1:]:
                oo=point["oo"];stats=oo["statistics"]
                if task["shots"]:
                    if stats["counts"]["shots"]!=task["shots"]or len(stats["counts"]["plus_counts"])!=2779:
                        raise RuntimeError("Missing per-Pauli sufficient statistics")
                    macro=point["macro_iteration"];stride=2 if task.get("exact_ucc") else 383
                    if stats["counts"]["sampling_seed"]!=task["seed"]+1+(macro-1)*stride:raise RuntimeError("Measurement RNG stream changed")
                    training=int(np.floor(task["shots"]*.8))if task["acceptance"]!="nominal"else task["shots"]
                    if stats["training_shots"]!=training:raise RuntimeError("Training budget mismatch")
                if task["mitigation"]=="affine"and stats["correction_residual_after"]>1e-8:
                    raise RuntimeError("Affine correction failed")
                if task["acceptance"]=="heldout"and stats["accepted"]and stats["validation_upper_bound_ha"]>=0:
                    raise RuntimeError("Acceptance rule violated")
                applied=oo["exact_energy_after_ha"]-oo["exact_energy_before_ha"]
                proposed=oo["proposed_exact_energy_after_ha"]-oo["exact_energy_before_ha"]
                if not stats["accepted"]and abs(applied)>1e-10:raise RuntimeError("Rejected proposal changed physical energy")
                records.append({"macro":point["macro_iteration"],"accepted":stats["accepted"],
                    "applied_delta_mha":applied*1000,"proposed_delta_mha":proposed*1000,
                    "seconds":stats["total_seconds"],"proposal_seconds":stats["proposal_seconds"],
                    "correction_seconds":stats["correction_seconds"],"validation_seconds":stats["validation_seconds"],
                    "newton_shift":oo["hessian_shift"],"newton_step_norm":oo["step_norm"]})
        rows.append({**task,"status":status,"completed":completed,"error_mha":error,"worker_seconds":e["seconds"],
            "oo_records":records,"trajectory":[{"macro":p["macro_iteration"],"error_mha":1000*(p["exact_energy_ha"]-fci)}for p in native["iterations"]]if native else[]})
    return rows


def summarize_groups(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["distance"],r["shots"],r.get("exact_ucc",False),r["name"]].append(r)
    cells=[]
    for (distance,shots,exact_ucc,name),group in sorted(groups.items()):
        completed=[x for x in group if x["completed"]];oo=[o for x in completed for o in x["oo_records"]]
        cells.append({"distance":distance,"shots":shots,"exact_ucc":exact_ucc,"variant":name,"trials":len(group),
            "completed":len(completed),"error_mha":describe([x["error_mha"]for x in completed]),
            "chemical_accuracy_count":sum(x["error_mha"]<=1.6 for x in completed),
            "worker_seconds":describe([x["worker_seconds"]for x in completed]),
            "oo_seconds_per_run":describe([sum(o["seconds"]for o in x["oo_records"])for x in completed]),
            "accepted_updates":sum(o["accepted"]for o in oo),"oo_updates":len(oo),
            "harmful_updates":sum(o["applied_delta_mha"]>1e-7 for o in oo),
            "rejected_useful_updates":sum(not o["accepted"]and o["proposed_delta_mha"] < -1e-7 for o in oo),
            "applied_oo_delta_mha_per_run":describe([sum(o["applied_delta_mha"]for o in x["oo_records"])for x in completed])})
    return cells


def holm(comparisons):
    previous=0.
    for rank,index in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]["p_two_sided"])):
        previous=max(previous,min(1.,(len(comparisons)-rank)*comparisons[index]["p_two_sided"]))
        comparisons[index]["holm_p"]=previous


def paired(rows,a,b,tolerance=1e-10):
    output=[];lookup={(r["distance"],r["shots"],r["seed"],r["name"]):r for r in rows}
    for distance in [1.6,2.4,3.0]:
        for shots in [1000,10000,100000]:
            seeds=sorted({r["seed"]for r in rows if r["distance"]==distance and r["shots"]==shots})
            wins=losses=ties=0;differences=[];difference_seeds=[]
            for seed in seeds:
                left=lookup[distance,shots,seed,a];right=lookup[distance,shots,seed,b]
                if left["completed"]and not right["completed"]:wins+=1
                elif right["completed"]and not left["completed"]:losses+=1
                elif not left["completed"]:ties+=1
                else:
                    delta=left["error_mha"]-right["error_mha"]
                    differences.append(delta);difference_seeds.append(seed)
                    if abs(delta)<=tolerance*1000:ties+=1
                    elif delta<0:wins+=1
                    else:losses+=1
            p=float(binomtest(wins,wins+losses,.5).pvalue)if wins+losses else 1.
            output.append({"distance":distance,"shots":shots,"a":a,"b":b,"wins":wins,"losses":losses,"ties":ties,
                "p_two_sided":p,"paired_difference_mha":describe(differences),
                "paired_differences_by_seed":dict(zip(map(str,difference_seeds),differences))})
    holm(output)
    # One common draw of seed indices across cells preserves cross-cell pairing.
    seeds=sorted({int(s)for r in output for s in r["paired_differences_by_seed"]});rng=np.random.default_rng(2026102001)
    indices=rng.integers(0,len(seeds),(10000,len(seeds)))if seeds else np.empty((0,0),int)
    for row in output:
        values=np.asarray([row["paired_differences_by_seed"].get(str(s),np.nan)for s in seeds])
        if np.any(np.isfinite(values)):
            sampled=values[indices];counts=np.sum(np.isfinite(sampled),axis=1)
            means=np.nansum(sampled,axis=1)[counts>0]/counts[counts>0]
            row["paired_mean_difference_ci95_unadjusted"]=np.quantile(means,[.025,.975]).tolist()
        else:row["paired_mean_difference_ci95_unadjusted"]=[None,None]
    return output


def analyze_trajectories(root,phase):
    rows=trajectory_rows(root,phase);cells=summarize_groups(rows);statuses=dict(Counter(r["status"]for r in rows))
    result={"validation_passed":all(r["completed"]for r in rows),"statuses":statuses,"trials":len(rows),"cells":cells,"runs":rows}
    if phase=="ucc_extension":
        # Numerical scientific failures remain outcomes; infrastructure failures
        # have already raised in trajectory_rows.
        result["validation_passed"]=True
        baseline={r['index']:r for r in json.loads((root/'pilot_summary.json').read_text())['runs']}
        result['paired_against_pilot']=[{'index':r['index'],'baseline_index':r['baseline_pilot_index'],
            'distance':r['distance'],'shots':r['shots'],'solver':r['solver'],
            'completed':r['completed'],'baseline_completed':baseline[r['baseline_pilot_index']]['completed'],
            'error_difference_mha':r['error_mha']-baseline[r['baseline_pilot_index']]['error_mha']
                if r['completed']and baseline[r['baseline_pilot_index']]['completed']else None}for r in rows]
    if phase=="confirmation":
        # Scientific optimizer failures are outcomes, not missing infrastructure
        # data; they remain in the completion-first comparison and denominators.
        result["validation_passed"]=True
        result["controls"]=trajectory_rows(root,"controls")
        if not all(r["completed"]for r in result["controls"]):result["validation_passed"]=False
        freeze=json.loads((root/"policy_freeze.json").read_text())
        if hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()!=freeze["protocol_sha256"]:raise RuntimeError("Protocol changed after freezing")
        result["primary_comparisons"]=paired(rows,"seq_affine_heldout","newton_stable_affine_heldout")
        result["secondary_comparisons"]={family:paired(rows,a,b)for family,a,b in [
            ("raw_guarded","seq_none_heldout","newton_stable_none_heldout"),
            ("raw_nominal","seq_none_nominal","newton_stable_none_nominal"),
            ("affine_nominal","seq_affine_nominal","newton_stable_affine_nominal")]}
    write_json(root/f"{phase}_summary.json",result)
    columns=['distance','shots','exact_ucc','variant','trials','completed','mean_error_mha','sd_error_mha',
        'median_error_mha','maximum_error_mha','rmse_mha','chemical_accuracy_count','accepted_updates',
        'harmful_updates','rejected_useful_updates','mean_classical_oo_seconds_per_run','mean_worker_seconds']
    with(root/f'{phase}_summary.csv').open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for c in cells:
            writer.writerow({**{k:c[k]for k in columns if k in c},
                **{f'{key}_error_mha':c['error_mha'][key]for key in ['mean','sd','median','maximum']},
                'rmse_mha':c['error_mha']['rmse'],'mean_classical_oo_seconds_per_run':c['oo_seconds_per_run']['mean'],
                'mean_worker_seconds':c['worker_seconds']['mean']})
    lines=[f"# {phase.capitalize()} uncertainty-aware OO study","",f"Statuses: {statuses}.",
        "Final exact circuit-state errors are measured after ten macro iterations. Exact diagnostics never select updates.","",
        "| R (A) | Shots | Exact UCC | Variant | Complete | Error mean ± SD (mHa) | Within 1.6 mHa | Accepted OO |",
        "|---:|---:|---|---|---:|---:|---:|---:|"]
    for c in cells:
        d=c["error_mha"];value=f"{d['mean']:.4f} ± {d['sd']:.4f}"if d["mean"]is not None and d["sd"]is not None else str(d["mean"])
        lines.append(f"| {c['distance']} | {c['shots']:,} | {c['exact_ucc']} | {c['variant']} | {c['completed']}/{c['trials']} | {value} | {c['chemical_accuracy_count']}/{c['trials']} | {c['accepted_updates']}/{c['oo_updates']} |")
    if phase=="confirmation":
        lines += ['','## Noiseless controls at the same macro budget','',
            'These endpoints are not assumed fully converged.','',
            '| R (A) | Solver | Exact-state FCI error (mHa) |','|---:|---|---:|']
        for control in result['controls']:
            lines.append(f"| {control['distance']} | {control['solver']} | {control['error_mha']} |")
        lines += ["","## Prespecified primary comparisons","",
            "Completion-first paired sign tests concern win probability, not equality of mean error. Holm adjustment covers these nine comparisons.",
            "Mean-error differences use joint completions; bootstrap intervals are unadjusted descriptive intervals. Failures remain in completion-first outcomes and accuracy denominators.","",
            "| R (A) | Shots | SOO wins | Newton wins | Ties | Mean SOO−Newton (mHa) | Holm p |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
        for p in result["primary_comparisons"]:
            value=p["paired_difference_mha"]["mean"]
            lines.append(f"| {p['distance']} | {p['shots']:,} | {p['wins']} | {p['losses']} | {p['ties']} | {value} | {p['holm_p']:.5f} |")
        lines += ["","Non-rejection does not establish equivalence. A safeguard that rejects all useful updates is not an optimizer advantage."]
    elif phase=="ucc_extension":
        lines += ['','This development-only extension applies estimated-covariance affine correction to UCC energy evaluations, paired with the matching raw-UCC pilot cases.',
            'Total quantum measurement budgets are unchanged. Paired endpoint differences are in ucc_extension_summary.json.',
            'Estimated correction weights can introduce parameter-dependent finite-shot bias. Therefore the mean corrected UCC objective is not assumed to retain an exact degree-two Fourier representation.',
            'These development cases do not enter the nine primary fresh-seed comparisons.']
    (root/f"{phase}_report.md").write_text("\n".join(lines)+"\n")
    plot_trajectories(root,phase,rows,cells,result.get('controls',[]))
    if phase=='confirmation':plot_primary(root,result['primary_comparisons'])
    write_json(root/f"{phase}_analysis_provenance.json",{"source":str(Path(__file__).resolve()),
        "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"python":sys.executable,
        "numpy":np.__version__})
    print(json.dumps({"phase":phase,"validation_passed":result["validation_passed"],"trials":len(rows),"statuses":statuses}))


def plot_trajectories(root,phase,rows,cells,controls=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=root/"figures";output.mkdir(exist_ok=True)
    modes=[True,False]if phase=="pilot"else[False]
    for exact_ucc in modes:
        subset=[r for r in rows if r.get("exact_ucc",False)==exact_ucc]
        distances=sorted({r["distance"]for r in subset});shots=sorted({r["shots"]for r in subset});names=sorted({r["name"]for r in subset})
        fig,axes=plt.subplots(len(distances),len(shots),figsize=(5*len(shots),3.6*len(distances)),squeeze=False)
        for i,d in enumerate(distances):
            for j,s in enumerate(shots):
                ax=axes[i,j]
                for k,name in enumerate(names):
                    group=[r for r in subset if r["distance"]==d and r["shots"]==s and r["name"]==name and r["completed"]]
                    if not group:continue
                    values=np.asarray([[p["error_mha"]for p in r["trajectory"]]for r in group])
                    q=np.quantile(values,[.25,.5,.75],axis=0);x=np.arange(values.shape[1])
                    ax.plot(x,q[1],label=name,linewidth=1.2)
                    ax.fill_between(x,q[0],q[2],alpha=.07)
                for control in controls:
                    if control['distance']!=d or not control['completed']:continue
                    points=control['trajectory'];style='--'if control['solver']=='seq'else'-.'
                    ax.plot([p['macro']for p in points],[max(1e-8,p['error_mha'])for p in points],
                        color='.35',linestyle=style,linewidth=.9,label='Noiseless '+control['solver'])
                ax.set(title=f"R={d:g} A, {s:,} shots",xlabel="Macro iteration",ylabel="Exact FCI error (mHa)",yscale="log")
                ax.axhline(1.6,color=".5",linestyle=":");ax.grid(alpha=.2)
        axes[0,-1].legend(fontsize=6)
        fig.suptitle(f"{phase.capitalize()}: noisy median/IQR; UCC {'exact' if exact_ucc else 'finite-shot'}")
        fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(output/f"{phase}_convergence_{'exact'if exact_ucc else'noisy'}_ucc.{ext}",dpi=170)
        plt.close(fig)

        fig,axes=plt.subplots(len(distances),len(shots),figsize=(5*len(shots),3.9*len(distances)),squeeze=False)
        short=[name.replace("newton_stable","NR").replace("seq","S").replace("nominal","N").replace("heldout","H").replace("affine","A").replace("none","R")for name in names]
        for i,d in enumerate(distances):
            for j,s in enumerate(shots):
                ax=axes[i,j]
                groups=[[r["error_mha"]for r in subset if r["distance"]==d and r["shots"]==s and r["name"]==name and r["completed"]]for name in names]
                for k,values in enumerate(groups):
                    if values:ax.boxplot(values,positions=[k+1],widths=.55,whis=(5,95),showfliers=True)
                ax.set(xticks=np.arange(1,len(names)+1),xticklabels=short,yscale="log",ylabel="Final exact FCI error (mHa)",title=f"R={d:g} A, {s:,} shots")
                ax.tick_params(axis="x",labelrotation=60,labelsize=7);ax.axhline(1.6,color=".5",linestyle=":");ax.grid(alpha=.2)
        fig.suptitle("Endpoint distributions: S/NR proposer; R/A raw/affine; N/H nominal/heldout")
        fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(output/f"{phase}_endpoints_{'exact'if exact_ucc else'noisy'}_ucc.{ext}",dpi=170)
        plt.close(fig)


def plot_primary(root,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,shots in zip(axes,[1000,10000,100000]):
        for i,distance in enumerate([1.6,2.4,3.0]):
            r=next(r for r in comparisons if r['distance']==distance and r['shots']==shots)
            mean=r['paired_difference_mha']['mean'];low,high=r['paired_mean_difference_ci95_unadjusted']
            if mean is not None:
                ax.plot([low,high],[i,i],color='#0072B2');ax.scatter([mean],[i],color='#0072B2')
            ax.annotate(f"{r['wins']} W / {r['losses']} L / {r['ties']} T; pH={r['holm_p']:.3g}",
                xy=(.02,.92-i*.32),xycoords='axes fraction',fontsize=8)
        ax.axvline(0,color='.4',linestyle=':');ax.set(title=f'{shots:,} shots',xlabel='Mean SOO − Newton final FCI error (mHa)',
            yticks=[0,1,2],yticklabels=['1.6 A','2.4 A','3.0 A'],ylim=(2.5,-.5));ax.grid(alpha=.2)
    fig.suptitle('Guarded affine methods: mean differences and unadjusted bootstrap CIs; pH tests paired win probability')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(root/'figures'/f'confirmation_primary_paired.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True);parser.add_argument("--phase",choices=["fixed","pilot","confirmation","ucc_extension"],required=True)
    args=parser.parse_args();root=args.work.resolve()
    if args.phase=="fixed":analyze_fixed(root)
    else:analyze_trajectories(root,args.phase)


if __name__=="__main__":main()

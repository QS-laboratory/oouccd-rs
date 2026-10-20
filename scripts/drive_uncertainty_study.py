"""Continue staged Slurm experiments and frozen analysis without interactive prompts.

Run in the submission environment, where sacct is available. This process does
not reserve compute CPUs while waiting. Terminal worker records are checked
against live scheduler state; missing observations never cause job restarts.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import socket
import subprocess
import sys
import time

from run_uncertainty_study import (freeze_confirmation,submit_confirmation,write_json,sha,submit,
    prepare_reconstruction,prepare_bootstrap,prepare_ucc_extension)
from run_decision_replays import prepare as prepare_replays,submit as submit_replays,summarize as summarize_replays


def log(message):print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())+" UTC "+message,flush=True)


def phase_status(root,phase):
    path=root/f"{phase}_manifest.json"
    if not path.exists():return None
    counts=Counter()
    for t in json.loads(path.read_text()):
        out=Path(t["output"]);record=out.with_suffix(out.suffix+".execution.json")
        if record.exists():
            e=json.loads(record.read_text())
            if e["exit_code"]:counts["process_failure"]+=1
            else:counts["terminal"]+=1
        else:counts["active_or_pending"]+=1
    return counts


def scheduler_states(root,phase):
    study=json.loads((root/"study.json").read_text())
    jobs=[j["job_id"]for j in study["jobs"]if j["phase"]==phase]
    if not jobs:raise RuntimeError(f"No submitted job for {phase}")
    result=subprocess.run(["squeue","-h","-j",",".join(jobs),"-o","%i|%T|%R"],capture_output=True,text=True)
    if result.returncode:
        # Observation errors are not evidence that a job ended.
        raise RuntimeError("Scheduler observation failed: "+result.stderr.strip())
    return result.stdout.strip()


def wait_phase(root,phase,poll):
    previous=None
    while True:
        counts=phase_status(root,phase)
        if counts is None:raise RuntimeError(f"Missing {phase} manifest")
        if counts["process_failure"]:raise RuntimeError(f"{phase}: {counts['process_failure']} process failures require investigation")
        if not counts["active_or_pending"]:
            log(f"{phase}: all {counts['terminal']} workers terminal")
            return
        try:
            live=scheduler_states(root,phase)
        except RuntimeError as error:
            log(str(error)+"; observing the same submitted jobs again")
            time.sleep(poll);continue
        if not live:
            # Allow filesystem/accounting propagation without inventing a retry.
            time.sleep(min(10,poll))
            if phase_status(root,phase)["active_or_pending"]:
                raise RuntimeError(f"{phase}: scheduler has no live job but worker records are missing")
            continue
        value=dict(counts)
        if value!=previous:log(f"{phase}: {value}");previous=value
        time.sleep(poll)


def run_analysis(root,phase):
    study=json.loads((root/"study.json").read_text())
    if phase=="confirmation":script=root/"confirmation_source/scripts/analyze_uncertainty_study.py"
    else:
        script=root/"analysis_pipeline"/f"analyze_{phase}_frozen.py"
        script.parent.mkdir(exist_ok=True)
        if not script.exists():shutil.copy2(Path(__file__).with_name("analyze_uncertainty_study.py"),script)
    command=["srun","-p","defq","-N","1","-n","1","-c","1","--mem=7G","--time=00:30:00",
        "env","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1",study["python"],str(script),"--work",str(root),"--phase",phase]
    with(root/"logs"/f"{phase}-controller-analysis.out").open("w")as out,(root/"logs"/f"{phase}-controller-analysis.err").open("w")as err:
        subprocess.run(command,stdout=out,stderr=err,check=True)
    summary=json.loads((root/f"{phase}_summary.json").read_text())
    if not summary["validation_passed"]:raise RuntimeError(f"{phase} analysis did not pass validation")
    log(f"{phase} validated and analyzed")


def render_confirmation(root):
    study=json.loads((root/'study.json').read_text());sources=root/'presentation_source/scripts'
    sources.mkdir(parents=True,exist_ok=True)
    for name in ['render_uncertainty_results.py','analyze_uncertainty_study.py']:
        destination=sources/name
        if not destination.exists():shutil.copy2(Path(__file__).with_name(name),destination)
    command=['srun','-p','defq','-N','1','-n','1','-c','1','--mem=5G','--time=00:20:00',
        'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        study['python'],str(sources/'render_uncertainty_results.py'),'--work',str(root)]
    with(root/'logs/confirmation-render.out').open('w')as out,(root/'logs/confirmation-render.err').open('w')as err:
        subprocess.run(command,stdout=out,stderr=err,check=True)
    log('Supplemental confirmation tables and figures rendered from frozen results')


def ensure_development(root):
    """Make a prepared workspace runnable without undocumented fixture creation."""
    study=json.loads((root/'study.json').read_text())
    if not(root/'verification/verification.json').exists():
        previous=Path(study['previous_study']);old=json.loads((previous/'study.json').read_text())
        old_work=Path(next(w['work']for w in old['workspaces']if w['distance']==1.6))
        baseline=json.loads((old_work/'manifest.json').read_text())['binary']
        command=['srun','-p','defq','-N','1','-n','1','-c','1','--mem=5G','--time=00:15:00',
            'env','RAYON_NUM_THREADS=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'LD_LIBRARY_PATH='+study['library_path'],'PYTHONPATH='+study['python_site_packages'],
            study['python'],str(Path(__file__).with_name('verify_uncertainty_oo.py')),
            '--work',str(root/'verification'),'--baseline',baseline,'--previous-study',str(previous),
            '--binary',study['binaries']['evaluate_shot_noise']['path'],'--frozen-binary',study['binaries']['frozen_oo']['path']]
        with(root/'logs/controller-verification.out').open('w')as out,(root/'logs/controller-verification.err').open('w')as err:
            subprocess.run(command,stdout=out,stderr=err,check=True)
    for phase,limit in [('export',8),('fixed',24),('pilot',32)]:
        study=json.loads((root/'study.json').read_text())
        if any(j['phase']==phase for j in study['jobs']):continue
        dependency=next(j['job_id']for j in study['jobs']if j['phase']=='export')if phase=='fixed'else None
        submit(root,phase,limit,dependency)


def ensure_auxiliary(root):
    for phase,prepare,limit in [('reconstruction',prepare_reconstruction,8),('bootstrap',prepare_bootstrap,4),
                              ('ucc_extension',prepare_ucc_extension,4)]:
        if not(root/f'{phase}_manifest.json').exists():prepare(root)
        study=json.loads((root/'study.json').read_text())
        if not any(j['phase']==phase for j in study['jobs']):submit(root,phase,limit)
    study=json.loads((root/'study.json').read_text())
    pilot=next(j for j in study['jobs']if j['phase']=='pilot')
    if pilot.get('current_concurrency',pilot['concurrency'])<64:
        subprocess.run(['scontrol','update',f"JobId={pilot['job_id']}",'ArrayTaskThrottle=64'],capture_output=True,text=True)
        check=subprocess.run(['scontrol','show','job',pilot['job_id']],capture_output=True,text=True)
        if '64' in re.findall(r'ArrayTaskThrottle=(\d+)',check.stdout):
            pilot['current_concurrency']=64;write_json(root/'study.json',study)
    ensure_small_reference(root)


def ensure_small_reference(root):
    study=json.loads((root/'study.json').read_text());marker=root/'verification/small_reference_completed.json'
    if any(j['phase']=='small_reference'for j in study['jobs']):return
    if marker.exists():
        record=json.loads(marker.read_text());job=record.get('slurm_job_id')
        if not job:raise RuntimeError('Completed small reference has no scheduler provenance')
        command=None
    else:
        source=root/'small_reference_source/run_small_rdm_reference.py';source.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name('run_small_rdm_reference.py'),source)
        child=['srun','--cpu-bind=cores','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'RAYON_NUM_THREADS=1','LD_LIBRARY_PATH='+study['library_path'],'PYTHONPATH='+study['python_site_packages'],
            study['python'],str(source),'--work',str(root)]
        command=['sbatch','--parsable','--partition=defq','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=5G',
            '--time=00:20:00','--job-name=uoo-small-ref',f'--output={root}/logs/small-reference-%j.out',
            f'--error={root}/logs/small-reference-%j.err','--wrap',shlex.join(child)]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
    study['jobs'].append({'phase':'small_reference','job_id':job,'tasks':1,'single_job':True,'command':command})
    write_json(root/'study.json',study)


def accounting(root):
    study=json.loads((root/"study.json").read_text());records=[]
    for job in study["jobs"]:
        result=subprocess.run(["sacct","-j",job["job_id"],"--format=JobID,State%40,ExitCode,ElapsedRaw,NodeList","-P","-n"],text=True,capture_output=True,check=True)
        (root/"logs"/f"accounting-{job['job_id']}.txt").write_text(result.stdout)
        for line in result.stdout.splitlines():
            fields=line.split("|");match=re.fullmatch(re.escape(job["job_id"])+r"_(\d+)",fields[0])
            if not match and not(job.get('single_job')and fields[0]==job['job_id']):continue
            index=int(match[1])if match else 0
            records.append({"phase":job["phase"],"job_id":fields[0],"array_index":index,
                "global_index":index+job.get("offset",0),"state":fields[1].strip(),"exit_code":fields[2],
                "seconds":int(fields[3]),"node":fields[4]})
    write_json(root/"scheduler_accounting.json",records)
    return records


def final_report(root):
    confirmation=json.loads((root/"confirmation_summary.json").read_text())
    fixed=json.loads((root/"fixed_summary.json").read_text())
    pilot=json.loads((root/"pilot_summary.json").read_text())
    reconstruction=json.loads((root/"reconstruction_summary.json").read_text())
    bootstrap=json.loads((root/"bootstrap_summary.json").read_text())
    ucc=json.loads((root/"ucc_extension_summary.json").read_text())
    replay=json.loads((root/'decision_replay_audit.json').read_text())
    small_reference=json.loads((root/'verification/small_reference_completed.json').read_text())
    records=accounting(root)
    bad=[r for r in records if r["state"]!="COMPLETED"or r["exit_code"]!="0:0"]
    if bad:raise RuntimeError(f"Non-successful scheduler tasks remain: {bad[:5]}")
    requirements={"legacy_replay":json.loads((root/"verification/verification.json").read_text()),
        "fixed_validation":fixed["validation_passed"],"pilot_validation":pilot["validation_passed"],
        "confirmation_validation":confirmation["validation_passed"],"confirmation_tasks":confirmation["trials"],
        "noiseless_controls":len(confirmation["controls"]),"fixed_batches":fixed["independent_batches"],
        "reconstruction_tasks":reconstruction["tasks"],"bootstrap_tasks":bootstrap["tasks"],
        "ucc_extension_runs":ucc["trials"],"ucc_extension_validation":ucc["validation_passed"],
        "scheduler_tasks":len(records),"scheduler_failures":len(bad)}
    requirements['small_reference']=small_reference
    requirements['decision_replay']={k:v for k,v in replay.items()if k!='cases'}
    if requirements["confirmation_tasks"]!=7200 or requirements["noiseless_controls"]!=6:
        raise RuntimeError("Full confirmation scope is incomplete")
    if not all(requirements[k]for k in ("fixed_validation","pilot_validation","confirmation_validation")):
        raise RuntimeError("Scientific validation incomplete")
    if requirements['ucc_extension_runs']!=40 or not requirements['ucc_extension_validation']:
        raise RuntimeError('UCC correction extension is incomplete')
    if small_reference['matched_batches']!=25 or any(v['completed']!=25 for v in small_reference['methods'].values()):
        raise RuntimeError('Small fixed-sector reconstruction reference did not complete')
    if not replay['passed']or replay['trajectories']!=7566:raise RuntimeError('Full decision/physical-energy replay incomplete')
    write_json(root/"completion_evidence.json",requirements)
    significant=[p for p in confirmation["primary_comparisons"]if p["holm_p"]<.05]
    wins=[p for p in significant if p["wins"]>p["losses"]]
    losses=[p for p in significant if p["wins"]<p["losses"]]
    lines=["# RDM mitigation and uncertainty-aware orbital optimization","",
        "LiH/STO-3G, 1.6/2.4/3.0 A, 1,000/10,000/100,000 shots per independent Pauli word; ten fixed macro iterations.",
        "All eight confirmation variants use identical modeled measurement budgets and one-CPU allocations. Exact energies are diagnostic only.","",
        f"Completed scope: {fixed['independent_batches']:,} fixed-state batches; {pilot['trials']} development trajectories; {confirmation['trials']:,} fresh-seed confirmation trajectories and six noiseless controls.",
        f"Prespecified primary win-probability comparisons: {len(wins)} significant cells favor SOO; {len(losses)} favor Newton after Holm adjustment across nine cells.",
        "These tests do not test equality of mean error. Unadjusted paired mean-error intervals, tails, completion rates and classical cost must also be considered.","",
        "- [Confirmation results and primary comparisons](confirmation_report.md)",
        "- [Paired intervals, noiseless controls, costs and CSV tables](confirmation_supplement.md)",
        "- [Development pilot, including exact/noisy UCC](pilot_report.md)",
        "- [Matched fixed-state experiments](fixed_report.md)",
        "- [Physical reconstruction benchmarks](reconstruction_report.md)",
        "- [Joint whole-sweep bootstrap and posterior diagnostics](bootstrap_report.md)","",
        "- [Development-only affine correction inside UCC](ucc_extension_report.md)","",
        "![Confirmation convergence](figures/confirmation_convergence_noisy_ucc.png)","",
        "![Confirmation endpoints](figures/confirmation_endpoints_noisy_ucc.png)","",
        "The confidence safeguard controls harmful accepted OO proposals under the independent-shot model. It does not guarantee monotone noisy UCC updates or better endpoint accuracy.",
        "DQG reconstruction is a necessary-condition relaxation. Solver failures and feasibility tolerances remain in the reconstruction report.",
        "Projected Beta and robust training-score diagnostics are exploratory; they are not claimed to be physically constrained posteriors or certified confidence regions.",
        "Sources, binaries, counts, configuration hashes, seed schedules, policy freezing, runtime metadata and scheduler accounting are preserved in this workspace."]
    lines += ['',f"Independent replay checked {replay['trajectories']:,} trajectories, {replay['checked_decisions']:,} recorded OO decisions and {replay['statevector_diagnostic_calls']:,} physical-energy diagnostics without additional quantum measurements."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    log("All staged analyses and the full-scope report are written; root-agent completion audit remains required")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--poll-seconds",type=int,default=30);args=parser.parse_args();root=args.work.resolve()
    process={"pid":os.getpid(),"hostname":socket.gethostname(),"started_unix":time.time(),
        "source":str(Path(__file__).resolve()),"sha256":sha(Path(__file__)),"argv":sys.argv}
    attempts=json.loads((root/'controller_attempts.json').read_text())if(root/'controller_attempts.json').exists()else[]
    if(root/'controller_process.json').exists()and not attempts:attempts.append(json.loads((root/'controller_process.json').read_text()))
    attempts.append(process);write_json(root/'controller_attempts.json',attempts)
    write_json(root/"controller_process.json",process)
    ensure_development(root)
    wait_phase(root,"fixed",args.poll_seconds)
    if not(root/"fixed_summary.json").exists():run_analysis(root,"fixed")
    ensure_auxiliary(root)
    wait_phase(root,"pilot",args.poll_seconds)
    if not(root/"pilot_summary.json").exists():run_analysis(root,"pilot")
    if not(root/"policy_freeze.json").exists():freeze_confirmation(root)
    submit_confirmation(root)
    prepare_replays(root);submit_replays(root)
    wait_phase(root,"confirmation",args.poll_seconds);wait_phase(root,"controls",args.poll_seconds)
    if not(root/"confirmation_summary.json").exists():run_analysis(root,"confirmation")
    wait_phase(root,'decision_audit',args.poll_seconds)
    if not(root/'decision_replay_audit.json').exists():summarize_replays(root)
    for phase in ("ucc_extension","reconstruction","bootstrap"):
        wait_phase(root,phase,args.poll_seconds)
        if not(root/f"{phase}_summary.json").exists():
            if phase=='ucc_extension':run_analysis(root,phase)
            else:
                scientific_python=json.loads((root/'study.json').read_text())['python']
                command=[scientific_python,str(Path(__file__).with_name("summarize_uncertainty_auxiliary.py")),"--work",str(root),"--phase",phase]
                subprocess.run(command,check=True)
    marker=root/'verification/small_reference_completed.json'
    while not marker.exists():
        log('Waiting for the submitted small-sector reference')
        if not scheduler_states(root,'small_reference'):raise RuntimeError('Small reference ended without its completion marker')
        time.sleep(args.poll_seconds)
    render_confirmation(root)
    final_report(root)


if __name__=="__main__":main()

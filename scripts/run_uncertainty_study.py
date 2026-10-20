"""Prepare and run the staged, matched-budget RDM/OO uncertainty study.

The protocol is frozen before fresh confirmation seeds are submitted. Workers
use preserved binaries and sources and never overwrite existing results.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import sys
import sysconfig
import time

REPO=Path(__file__).resolve().parents[1]
DISTANCES=[1.6,2.4,3.0]
SHOTS=[1000,10000,100000]
VARIANTS=[{"name":f"{solver}_{mitigation}_{acceptance}","solver":solver,"mitigation":mitigation,"acceptance":acceptance}
          for solver in ("seq","newton_stable") for mitigation in ("none","affine")
          for acceptance in ("nominal","heldout")]


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(root):
    manifest=json.loads((root/"study.json").read_text());env=os.environ.copy()
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS"):env[name]="1"
    env["PYTHONPATH"]=manifest["python_site_packages"]
    env["LD_LIBRARY_PATH"]=manifest["library_path"]+os.pathsep+env.get("LD_LIBRARY_PATH","")
    return env


def config(distance,variant,seed,shots,iterations=10):
    solver={"type":"sequential"}if variant["solver"]=="seq"else{"type":"newton_raphson","stabilized":True}
    return {"task_name":f"lih_{distance:g}_{variant['name']}_n{shots}_s{seed}",
        "chemistry":{"geometry":f"Li 0 0 0; H 0 0 {distance:g}","basis":"sto-3g","spin":0,"charge":0,"freeze_core":0},
        "method":{"label":variant["name"],"max_macro_iterations":iterations,"energy_tolerance":-1.,
            "rdm_evaluation":{"type":"finite_shot","shots":shots,"seed":seed}if shots else{"type":"statevector"},
            "ucc_optimizer":{"ansatz_config":{"type":"UCCD"},"solver_config":{"type":"sequential"}},
            "orbital_optimizer":{"solver_config":solver},"orbital_statistics":{
                "mitigation":variant["mitigation"],"acceptance":variant["acceptance"],
                "training_fraction":.8,"trajectory_alpha":.05,"min_improvement_ha":0.,"record_counts":True}}}


def initialize(root,previous):
    root=root.resolve();previous=previous.resolve()
    if(root/"study.json").exists():raise RuntimeError("Study already initialized")
    for name in ("bin","source","logs","snapshots","fixed/results","pilot/configs","pilot/results",
                 "confirmation/configs","confirmation/results","reconstruction/results","bootstrap/results"):
        (root/name).mkdir(parents=True,exist_ok=True)
    binaries={}
    for name in ("evaluate_shot_noise","frozen_oo"):
        source=REPO/"target/release/examples"/name;destination=root/"bin"/name
        shutil.copy2(source,destination);binaries[name]={"path":str(destination),"sha256":sha(destination)}
    sources={}
    for relative in subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=REPO,text=True).splitlines():
        if relative.startswith(("src/","scripts/","examples/"))or relative in ("Cargo.toml","Cargo.lock"):
            p=REPO/relative
            if p.is_file():
                dest=root/"source"/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[relative]=sha(dest)
    (root/"source/diff-from-head.patch").write_bytes(subprocess.check_output(["git","diff","--binary","HEAD"],cwd=REPO))
    (root/"python_packages.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    old_study=json.loads((previous/"study.json").read_text());snapshot_tasks=[];fixed_tasks=[];references={}
    for entry in old_study["workspaces"]:
        work=Path(entry["work"]);old=json.loads((work/"manifest.json").read_text());distance=entry["distance"]
        references[str(distance)]={"fci_energy_ha":old["fci_energy_ha"],"rhf_energy_ha":old["rhf_energy_ha"]}
        for solver in ("seq","newton_stable"):
            origin=next(t for t in old["tasks"]if t["method"]==solver and t["shots"]is None)
            for iteration in ([0,1,3,9]if solver=="seq"else[1,3,9]):
                name=f"r{distance:g}_{solver}_m{iteration}";snapshot=root/"snapshots"/f"{name}.json"
                snapshot_tasks.append({"index":len(snapshot_tasks),"name":name,"distance":distance,
                    "solver":solver,"iteration":iteration,"config":origin["config"],"trajectory":origin["result"],
                    "origin_sha256":sha(origin["result"]),"output":str(snapshot)})
                for shots in SHOTS:
                    index=len(fixed_tasks)
                    fixed_tasks.append({"index":index,"snapshot":str(snapshot),"distance":distance,"name":name,
                        "shots":shots,"replicates":1000,"seed":501000000+index*100000,
                        "output":str(root/"fixed/results"/f"task_{index:03}.jsonl")})
    protocol={"version":1,"chemistry":"LiH/STO-3G, 4 electrons, 6 spatial orbitals, no frozen core",
        "distances":DISTANCES,"shots":SHOTS,"macro_iterations":10,"variants":VARIANTS,
        "measurement_model":"Independent Pauli words; no hardware noise; shared RDM across every OO angle and coordinate",
        "hypotheses":["RDM correction improves orbital-update quality","Held-out acceptance reduces harmful updates while preserving useful decrease",
                      "SOO outperforms equally corrected and safeguarded stabilized Newton in specified regimes"],
        "training_fraction":.8,"trajectory_alpha":.05,"validation_bound":"One-sided Hoeffding for shared-Pauli energy differences; one full-sweep decision",
        "primary_contrasts":"seq_affine_heldout vs newton_stable_affine_heldout in each of nine distance/shot cells",
        "primary_test":"Two-sided paired sign test on completion-first final exact-state errors; both failures or numerical equality are ties",
        "numerical_tie_tolerance_ha":1e-10,"multiplicity":"Holm across nine primary comparisons",
        "secondary_contrasts":"Raw-RDM guarded methods and nominal methods; separate multiplicity families; no equivalence claim from nonsignificance",
        "statistical_unit":"Independent trial seed; paired bootstrap resamples whole seed blocks, retaining cross-geometry/shot dependence",
        "confirmation_seeds":[80261020+i*1000000 for i in range(100)],
        "development_seeds":[20261020+i*1000000 for i in range(5)],
        "seed_window":1000000,"confirmation_runs":7200,
        "power_reference":{"non_tied_pairs":80,"alternative_win_probability":.7,"family_alpha":.05,"cells":9,
                           "conservative_exact_sign_test_power":.8046},
        "scoring":"Final exact circuit-state energy minus FCI, never minimum sampled energy; truth only in diagnostics",
        "runtime":"Same one-CPU allocation and thread limits; OO timing excludes exact diagnostics; total timing includes them",
        "freeze_rule":"Verification and pilot must finish; policies and sample size frozen before any confirmation result is generated",
        "strong_reconstruction":"Offline matched-count benchmarks only; no automatic promotion into confirmation without a separately frozen protocol amendment"}
    study={"root":str(root),"created_unix":time.time(),"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "previous_study":str(previous),"previous_summary_sha256":sha(previous/"study_summary.json"),
        "python":sys.executable,"python_site_packages":sysconfig.get_paths()["purelib"],"library_path":str(REPO/".paper-env/lib"),
        "binaries":binaries,"source_sha256":sources,"references":references,"protocol":protocol,"jobs":[],"phase":"prepared"}
    write_json(root/"study.json",study);write_json(root/"protocol.json",protocol)
    write_json(root/"export_manifest.json",snapshot_tasks);write_json(root/"fixed_manifest.json",fixed_tasks)
    pilot=[]
    for distance in (1.6,3.0):
        for shots in (1000,100000):
            for seed in protocol["development_seeds"]:
                for variant in VARIANTS:
                    for exact_ucc in (True,False):
                        index=len(pilot);cpath=root/"pilot/configs"/f"task_{index:04}.json"
                        write_json(cpath,config(distance,variant,seed,shots))
                        pilot.append({"index":index,"distance":distance,"shots":shots,"seed":seed,**variant,
                            "exact_ucc":exact_ucc,"config":str(cpath),"output":str(root/"pilot/results"/f"task_{index:04}.json")})
    write_json(root/"pilot_manifest.json",pilot)
    print(json.dumps({"study":str(root),"snapshots":len(snapshot_tasks),"fixed_tasks":len(fixed_tasks),"pilot_runs":len(pilot)},indent=2))


def worker(root,phase,index):
    study=json.loads((root/"study.json").read_text());task=json.loads((root/f"{phase}_manifest.json").read_text())[index]
    env=environment(root);output=Path(task["output"])
    if output.exists():raise RuntimeError(f"Refusing to overwrite {output}")
    if task.get("config_sha256")and sha(task["config"])!=task["config_sha256"]:
        raise RuntimeError("Frozen configuration hash mismatch")
    binary_name="frozen_oo"if phase in ("export","fixed")else"bootstrap_oo"if phase=="bootstrap"else"evaluate_affine_ucc"if phase=="ucc_extension"else"evaluate_shot_noise"
    binary=study["binaries"][binary_name]
    if sha(binary["path"])!=binary["sha256"]:raise RuntimeError("Evaluator hash mismatch")
    if phase=="bootstrap":
        command=[binary["path"],"--snapshot",task["snapshot"],"--data",task["data"],"--index",str(task["original_index"]),
            "--draws",str(task["draws"]),"--seed",str(task["seed"]),"--output",str(output)]
    elif phase=="reconstruction":
        command=[study["python"],str(root/"reconstruction_source/scripts/benchmark_rdm_reconstruction.py"),
            "--snapshot",task["snapshot"],"--data",task["data"],"--output",str(output),
            "--replicates",str(task["replicates"]),"--solver",task["solver"],"--methods",*task["methods"]]
    elif phase=="export":
        if sha(task["trajectory"])!=task["origin_sha256"]:raise RuntimeError("Snapshot origin changed")
        command=[binary["path"],"export","--config",task["config"],"--trajectory",task["trajectory"],
                 "--iteration",str(task["iteration"]),"--output",str(output)]
    elif phase=="fixed":
        command=[binary["path"],"study","--snapshot",task["snapshot"],"--shots",str(task["shots"]),
                 "--replicates",str(task["replicates"]),"--seed",str(task["seed"]),"--output",str(output)]
    else:
        command=[binary["path"],"--config",task["config"],"--output",str(output)]
        if task.get("exact_ucc"):command.append("--exact-ucc")
        if task.get("affine_ucc"):command.append("--affine-ucc")
    started=time.perf_counter()
    with(root/"logs"/f"{phase}-{index:04}.log").open("w")as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=log)
    write_json(output.with_suffix(output.suffix+".execution.json"),{"exit_code":result.returncode,
        "seconds":time.perf_counter()-started,"hostname":socket.gethostname(),"command":command,
        "slurm_job_id":os.getenv("SLURM_JOB_ID"),"slurm_array_task_id":os.getenv("SLURM_ARRAY_TASK_ID"),
        "affinity":sorted(os.sched_getaffinity(0)),"binary_sha256":None if phase=="reconstruction"else binary["sha256"],
        "script_sha256":sha(command[1])if phase=="reconstruction"else None})
    if result.returncode:raise SystemExit(result.returncode)


def submit(root,phase,concurrency,dependency=None):
    study=json.loads((root/"study.json").read_text());tasks=json.loads((root/f"{phase}_manifest.json").read_text())
    if any(j["phase"]==phase for j in study["jobs"]):raise RuntimeError("Phase already submitted")
    worker_path=root/("confirmation_source/scripts/run_uncertainty_study.py"if phase=="controls"else
        "reconstruction_source/scripts/run_uncertainty_study.py"if phase=="reconstruction"else
        "bootstrap_source/scripts/run_uncertainty_study.py"if phase=="bootstrap"else
        "ucc_extension_source/scripts/run_uncertainty_study.py"if phase=="ucc_extension"else"source/scripts/run_uncertainty_study.py")
    batch=root/f"{phase}.sbatch"
    batch.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+
        shlex.join(["srun","--cpu-bind=cores",study["python"],str(worker_path),"task","--work",str(root),
                    "--phase",phase,"--index"])+ ' "${SLURM_ARRAY_TASK_ID}"\n')
    command=["sbatch","--parsable","--partition=defq","--nodes=1","--ntasks=1","--cpus-per-task=1",
        "--mem=12G"if phase=="reconstruction"else"--mem=3G",
        "--time=02:00:00"if phase=="reconstruction"else"--time=01:00:00",f"--job-name=uoo-{phase}",f"--array=0-{len(tasks)-1}%{concurrency}",
        f"--output={root}/logs/{phase}-%A_%a.out",f"--error={root}/logs/{phase}-%A_%a.err"]
    if dependency:command.append(f"--dependency=afterok:{dependency}")
    command.append(str(batch));job=subprocess.check_output(command,text=True).strip().split(";")[0]
    study["jobs"].append({"phase":phase,"job_id":job,"tasks":len(tasks),"concurrency":concurrency,
                         "command":command,"submitted_unix":time.time()})
    write_json(root/"study.json",study);print(job)


def status(root):
    result={}
    for phase in ("export","fixed","pilot","reconstruction","bootstrap","ucc_extension","confirmation","controls"):
        p=root/f"{phase}_manifest.json"
        if not p.exists():continue
        count=Counter()
        for task in json.loads(p.read_text()):
            output=Path(task["output"]);execution=output.with_suffix(output.suffix+".execution.json")
            if execution.exists():
                e=json.loads(execution.read_text())
                if e["exit_code"]:state="process_failure"
                elif phase in ("pilot","confirmation"):state=json.loads(output.read_text())["status"]
                else:state="completed"
            elif output.exists():state="running"
            else:state="pending"
            count[state]+=1
        result[phase]=dict(count)
    print(json.dumps(result,indent=2))


def freeze_confirmation(root):
    """Freeze settings only after complete development evidence has been analyzed."""
    if(root/"policy_freeze.json").exists():raise RuntimeError("Confirmation protocol already frozen")
    if any((root/"confirmation/results").glob("task_*.json")):raise RuntimeError("Confirmation data already exist; cannot claim a prospective freeze")
    study=json.loads((root/"study.json").read_text());protocol=study["protocol"]
    if protocol!=json.loads((root/"protocol.json").read_text()):raise RuntimeError("Study and protocol definitions differ")
    for binary in study["binaries"].values():
        if sha(binary["path"])!=binary["sha256"]:raise RuntimeError("Preserved binary changed before freezing")
    verification=json.loads((root/"verification/verification.json").read_text())
    if any(value>2e-12 for value in verification["legacy_replay"].values()):raise RuntimeError("Legacy regression failed")
    for phase in ("fixed","pilot"):
        summary=root/f"{phase}_summary.json"
        if not summary.exists():raise RuntimeError(f"Analyze {phase} before freezing")
        analysis=json.loads(summary.read_text())
        if not analysis.get("validation_passed"):raise RuntimeError(f"{phase} validation has not passed")
        if phase=="fixed"and (analysis.get("tasks")!=63 or analysis.get("independent_batches")!=63000):
            raise RuntimeError("Fixed-state study scope is incomplete")
        if phase=="pilot"and (analysis.get("trials")!=320 or analysis.get("statuses")!={"completed":320}):
            raise RuntimeError("Full pilot has not completed successfully")
    for name in ("run_uncertainty_study.py","analyze_uncertainty_study.py"):
        destination=root/"confirmation_source/scripts"/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/"scripts"/name,destination)
    tasks=[]
    for distance in DISTANCES:
        for shots in SHOTS:
            for seed in protocol["confirmation_seeds"]:
                for variant in VARIANTS:
                    index=len(tasks);cpath=root/"confirmation/configs"/f"task_{index:05}.json"
                    write_json(cpath,config(distance,variant,seed,shots))
                    tasks.append({"index":index,"distance":distance,"shots":shots,"seed":seed,**variant,
                        "exact_ucc":False,"config":str(cpath),"config_sha256":sha(cpath),
                        "output":str(root/"confirmation/results"/f"task_{index:05}.json")})
    controls=[]
    for distance in DISTANCES:
        for solver in ("seq","newton_stable"):
            variant={"solver":solver,"mitigation":"none","acceptance":"nominal","name":solver+"_exact"}
            index=len(controls);cpath=root/"confirmation/configs"/f"control_{index:02}.json"
            write_json(cpath,config(distance,variant,None,None))
            controls.append({"index":index,"distance":distance,"shots":None,"seed":None,**variant,
                "exact_ucc":False,"config":str(cpath),"config_sha256":sha(cpath),
                "output":str(root/"confirmation/results"/f"control_{index:02}.json")})
    write_json(root/"confirmation_manifest.json",tasks);write_json(root/"controls_manifest.json",controls)
    freeze={"frozen_unix":time.time(),"protocol_sha256":sha(root/"protocol.json"),
        "pilot_summary_sha256":sha(root/"pilot_summary.json"),"fixed_summary_sha256":sha(root/"fixed_summary.json"),
        "manifest_sha256":sha(root/"confirmation_manifest.json"),"binary_sha256":study["binaries"],
        "confirmation_runs":len(tasks),"controls":len(controls),
        "policy_decision":"Keep the prespecified affine/none and nominal/heldout factorial; stronger reconstruction remains an offline reference",
        "analysis_source_sha256":{name:sha(REPO/"scripts"/name)for name in
            ("run_uncertainty_study.py","analyze_uncertainty_study.py")}}
    write_json(root/"policy_freeze.json",freeze)
    study["phase"]="confirmation_frozen";write_json(root/"study.json",study)
    print(json.dumps({"confirmation_runs":len(tasks),"controls":len(controls),"freeze":str(root/"policy_freeze.json")},indent=2))


def submit_confirmation(root):
    study=json.loads((root/"study.json").read_text());freeze=json.loads((root/"policy_freeze.json").read_text())
    if sha(root/"protocol.json")!=freeze["protocol_sha256"]or sha(root/"confirmation_manifest.json")!=freeze["manifest_sha256"]:
        raise RuntimeError("Frozen protocol/manifest changed")
    tasks=json.loads((root/"confirmation_manifest.json").read_text());block=100*len(VARIANTS)
    submitted={j.get("group")for j in study["jobs"]if j["phase"]=="confirmation"}
    for group in range(9):
        if group in submitted:continue
        batch=root/f"confirmation_{group}.sbatch";offset=group*block
        batch.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+
            f"TASK_INDEX=$((SLURM_ARRAY_TASK_ID + {offset}))\n"+
            shlex.join(["srun","--cpu-bind=cores",study["python"],str(root/"confirmation_source/scripts/run_uncertainty_study.py"),
                "task","--work",str(root),"--phase","confirmation","--index"])+ ' "$TASK_INDEX"\n')
        command=["sbatch","--parsable","--partition=defq","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=3G",
            "--time=00:40:00",f"--job-name=uoo-confirm-{group}",f"--array=0-{block-1}%8",
            f"--output={root}/logs/confirmation-{group}-%A_%a.out",f"--error={root}/logs/confirmation-{group}-%A_%a.err",str(batch)]
        job=subprocess.check_output(command,text=True).strip().split(";")[0]
        study["jobs"].append({"phase":"confirmation","group":group,"offset":offset,"job_id":job,
            "tasks":block,"concurrency":8,"command":command,"submitted_unix":time.time()})
        write_json(root/"study.json",study);print(job,flush=True)
    if not any(j["phase"]=="controls"for j in study["jobs"]):submit(root,"controls",6)


def terminal(root,phase):
    tasks=json.loads((root/f"{phase}_manifest.json").read_text())
    return all(Path(t["output"]).with_suffix(Path(t["output"]).suffix+".execution.json").exists()for t in tasks)


def prepare_reconstruction(root):
    if(root/"reconstruction_manifest.json").exists():raise RuntimeError("Reconstruction benchmark already prepared")
    for name in ("run_uncertainty_study.py","reconstruct_rdms.py","benchmark_rdm_reconstruction.py"):
        destination=root/"reconstruction_source/scripts"/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/"scripts"/name,destination)
    fixed=json.loads((root/"fixed_manifest.json").read_text());tasks=[]
    for t in fixed:
        if not (t["name"].endswith("seq_m0")or t["name"].endswith("_m3")):continue
        index=len(tasks)
        tasks.append({"index":index,"snapshot":t["snapshot"],"data":t["output"],"distance":t["distance"],
            "shots":t["shots"],"name":t["name"],"replicates":25,"solver":"CLARABEL",
            "methods":["spatial_psd","dqg","mle"],"output":str(root/"reconstruction/results"/f"task_{index:03}.jsonl")})
    write_json(root/"reconstruction_manifest.json",tasks)
    print(json.dumps({"tasks":len(tasks),"matched_batches_per_task":25,"methods":["spatial_psd","dqg","mle"]}))


def prepare_bootstrap(root):
    if(root/"bootstrap_manifest.json").exists():raise RuntimeError("Bootstrap study already prepared")
    study=json.loads((root/"study.json").read_text())
    source=REPO/"target/release/examples/bootstrap_oo";destination=root/"bin/bootstrap_oo"
    shutil.copy2(source,destination);study["binaries"]["bootstrap_oo"]={"path":str(destination),"sha256":sha(destination)}
    sources={}
    for relative in subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=REPO,text=True).splitlines():
        if relative.startswith(("src/","scripts/","examples/"))or relative in ("Cargo.toml","Cargo.lock"):
            p=REPO/relative
            if p.is_file():
                dest=root/"bootstrap_source"/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest);sources[relative]=sha(dest)
    study["bootstrap_source_sha256"]=sources;write_json(root/"study.json",study)
    tasks=[]
    for t in json.loads((root/"fixed_manifest.json").read_text()):
        if not t["name"].endswith("_m3"):continue
        index=len(tasks)
        tasks.append({"index":index,"snapshot":t["snapshot"],"data":t["output"],"distance":t["distance"],
            "shots":t["shots"],"name":t["name"],"original_index":0,"draws":256,
            "seed":190261020+index*1000000000,"output":str(root/"bootstrap/results"/f"task_{index:03}.jsonl")})
    write_json(root/"bootstrap_manifest.json",tasks)
    print(json.dumps({"tasks":len(tasks),"draws_per_scheme_and_proposal":256,"additional_quantum_shots":0}))


def prepare_ucc_extension(root):
    if(root/"ucc_extension_manifest.json").exists():raise RuntimeError("UCC extension already prepared")
    study=json.loads((root/"study.json").read_text());source=REPO/"target/release/examples/evaluate_shot_noise"
    destination=root/"bin/evaluate_affine_ucc";shutil.copy2(source,destination)
    study["binaries"]["evaluate_affine_ucc"]={"path":str(destination),"sha256":sha(destination)}
    sources={}
    for relative in subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=REPO,text=True).splitlines():
        if relative.startswith(("src/","scripts/","examples/"))or relative in ("Cargo.toml","Cargo.lock"):
            p=REPO/relative
            if p.is_file():
                dest=root/"ucc_extension_source"/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest);sources[relative]=sha(dest)
    study["ucc_extension_source_sha256"]=sources;write_json(root/"study.json",study)
    tasks=[]
    for t in json.loads((root/"pilot_manifest.json").read_text()):
        if t["exact_ucc"]or t["mitigation"]!="none"or t["acceptance"]!="nominal":continue
        index=len(tasks);cpath=root/"ucc_extension/configs"/f"task_{index:03}.json"
        variant=dict(t,name=t["name"]+"_affine_ucc")
        configuration=config(t["distance"],variant,t["seed"],t["shots"])
        write_json(cpath,configuration)
        tasks.append({**t,"index":index,"name":variant["name"],"affine_ucc":True,"config":str(cpath),
            "config_sha256":sha(cpath),"baseline_pilot_index":t["index"],
            "output":str(root/"ucc_extension/results"/f"task_{index:03}.json")})
    (root/"ucc_extension/results").mkdir(parents=True,exist_ok=True)
    write_json(root/"ucc_extension_manifest.json",tasks)
    print(json.dumps({"development_runs":len(tasks),"paired_pilot_baselines":True,"additional_shots_per_trajectory":0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare");p.add_argument("--work",type=Path,required=True);p.add_argument("--previous",type=Path,required=True)
    p=commands.add_parser("task");p.add_argument("--work",type=Path,required=True);p.add_argument("--phase",required=True);p.add_argument("--index",type=int,required=True)
    p=commands.add_parser("submit");p.add_argument("--work",type=Path,required=True);p.add_argument("--phase",required=True);p.add_argument("--concurrency",type=int,default=24);p.add_argument("--dependency")
    p=commands.add_parser("status");p.add_argument("--work",type=Path,required=True)
    p=commands.add_parser("freeze");p.add_argument("--work",type=Path,required=True)
    p=commands.add_parser("submit-confirmation");p.add_argument("--work",type=Path,required=True)
    p=commands.add_parser("prepare-reconstruction");p.add_argument("--work",type=Path,required=True)
    p=commands.add_parser("prepare-bootstrap");p.add_argument("--work",type=Path,required=True)
    p=commands.add_parser("prepare-ucc-extension");p.add_argument("--work",type=Path,required=True)
    args=parser.parse_args();root=args.work.resolve()
    if args.command=="prepare":initialize(root,args.previous)
    elif args.command=="task":worker(root,args.phase,args.index)
    elif args.command=="submit":submit(root,args.phase,args.concurrency,args.dependency)
    elif args.command=="freeze":freeze_confirmation(root)
    elif args.command=="submit-confirmation":submit_confirmation(root)
    elif args.command=="prepare-reconstruction":prepare_reconstruction(root)
    elif args.command=="prepare-bootstrap":prepare_bootstrap(root)
    elif args.command=="prepare-ucc-extension":prepare_ucc_extension(root)
    else:status(root)


if __name__=="__main__":main()

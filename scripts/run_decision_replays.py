"""Replay every recorded OO decision and independently recompute stored energies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def prepare(root):
    if(root/'decision_audit_manifest.json').exists():return
    study=json.loads((root/'study.json').read_text());repo=Path(__file__).resolve().parents[1]
    source=repo/'target/release/examples/audit_oo_decisions'
    destination=root/'bin/audit_oo_decisions';shutil.copy2(source,destination)
    study['binaries']['decision_audit']={'path':str(destination),'sha256':sha(destination)}
    sources={}
    for relative in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=repo,text=True).splitlines():
        if relative.startswith(('src/','scripts/','examples/'))or relative in ['Cargo.toml','Cargo.lock']:
            p=repo/relative
            if p.is_file():
                target=root/'decision_audit_source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[relative]=sha(target)
    study['decision_audit_source_sha256']=sources;write(root/'study.json',study)
    initial={t['distance']:t['output']for t in json.loads((root/'export_manifest.json').read_text())if t['iteration']==0}
    cases=[]
    for phase in ['pilot','ucc_extension','confirmation','controls']:
        for task in json.loads((root/f'{phase}_manifest.json').read_text()):
            cases.append({'phase':phase,'index':task['index'],'snapshot':initial[task['distance']],
                'config':task['config'],'trajectory':task['output'],
                'output':str(root/'decision_audit/cases'/f"{phase}_{task['index']:05}.json")})
    tasks=[{'index':index,'cases':cases[start:start+100],
        'output':str(root/'decision_audit/results'/f'task_{index:03}.json')}
        for index,start in enumerate(range(0,len(cases),100))]
    (root/'decision_audit/cases').mkdir(parents=True,exist_ok=True);(root/'decision_audit/results').mkdir(exist_ok=True)
    write(root/'decision_audit_manifest.json',tasks)
    print(json.dumps({'audit_tasks':len(tasks),'trajectories':len(cases),'additional_quantum_measurements':0}))


def submit(root):
    study=json.loads((root/'study.json').read_text())
    if any(j['phase']=='decision_audit'for j in study['jobs']):return
    tasks=json.loads((root/'decision_audit_manifest.json').read_text());dependencies=[]
    # Finished jobs can leave the live scheduler before this audit is submitted.
    # Query the live set once: asking squeue for an expired ID is an error.
    queue=subprocess.run(['squeue','-h','-o','%F|%A'],capture_output=True,text=True,check=True)
    live={job for line in queue.stdout.splitlines()for job in line.strip().split('|')}
    for j in study['jobs']:
        if j['phase']not in ['pilot','ucc_extension','confirmation','controls']:continue
        if j['job_id']in live:dependencies.append(j['job_id'])
    python=study['python'];worker=root/'decision_audit_source/scripts/run_decision_replays.py'
    script=root/'decision_audit.sbatch'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(
        ['srun','--cpu-bind=cores',python,str(worker),'task','--work',str(root),'--index'])+' "${SLURM_ARRAY_TASK_ID}"\n')
    command=['sbatch','--parsable','--partition=defq','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=4G',
        '--time=00:20:00','--job-name=uoo-decision-audit',f'--array=0-{len(tasks)-1}%16',
        f'--output={root}/logs/decision-audit-%A_%a.out',f'--error={root}/logs/decision-audit-%A_%a.err']
    if dependencies:command.append('--dependency=afterok:'+':'.join(dependencies))
    command.append(str(script));job=subprocess.check_output(command,text=True).strip().split(';')[0]
    study['jobs'].append({'phase':'decision_audit','job_id':job,'tasks':len(tasks),'concurrency':16,
        'command':command,'submitted_unix':time.time()});write(root/'study.json',study);print(job)


def worker(root,index):
    study=json.loads((root/'study.json').read_text());task=json.loads((root/'decision_audit_manifest.json').read_text())[index]
    binary=study['binaries']['decision_audit'];assert sha(binary['path'])==binary['sha256']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',
        LD_LIBRARY_PATH=study['library_path']+os.pathsep+env.get('LD_LIBRARY_PATH',''),PYTHONPATH=study['python_site_packages'])
    started=time.perf_counter();results=[]
    for case in task['cases']:
        output=Path(case['output']);provenance=output.with_suffix('.provenance.json')
        fingerprints={key:sha(case[key])for key in ['config','trajectory','snapshot']}
        if output.exists()and provenance.exists():
            prior=json.loads(provenance.read_text())
            assert prior['inputs']==fingerprints and prior['binary_sha256']==binary['sha256']
            assert sha(output)==prior['result_sha256']
            report=json.loads(output.read_text())
        else:
            if output.exists():output.rename(output.with_name(output.name+f'.partial-{time.time_ns()}'))
            native=json.loads(Path(case['trajectory']).read_text())
            command=[binary['path'],'--snapshot',case['snapshot'],'--config',case['config'],
                '--trajectory',case['trajectory'],'--output',str(output),'--verify-energies']
            if case['phase']=='controls'or len(native['iterations'])==1:command.append('--energies-only')
            log=output.with_suffix('.log')
            with log.open('w')as stream:result=subprocess.run(command,env=env,stdout=stream,stderr=stream)
            if result.returncode:
                results.append({**case,'passed':False,'exit_code':result.returncode,'log':str(log)});continue
            report=json.loads(output.read_text())
            write(provenance,{'inputs':fingerprints,'binary_sha256':binary['sha256'],'command':command,'result_sha256':sha(output)})
        results.append({'phase':case['phase'],'index':case['index'],**report})
    passed=all(r['passed']for r in results)
    write(Path(task['output']),{'passed':passed,'cases':results,'case_count':len(results)})
    write(Path(task['output']).with_suffix('.json.execution.json'),{'exit_code':0 if passed else 1,
        'seconds':time.perf_counter()-started,'slurm_job_id':os.getenv('SLURM_JOB_ID'),
        'slurm_array_task_id':os.getenv('SLURM_ARRAY_TASK_ID'),'binary_sha256':binary['sha256']})
    if not passed:raise SystemExit(1)


def summarize(root):
    tasks=json.loads((root/'decision_audit_manifest.json').read_text());cases=[]
    for task in tasks:
        output=Path(task['output']);execution=json.loads(output.with_suffix('.json.execution.json').read_text())
        assert execution['exit_code']==0
        report=json.loads(output.read_text());assert report['passed']and report['case_count']==len(task['cases'])
        cases.extend(report['cases'])
    assert len(cases)==7566 and len({(c['phase'],c['index'])for c in cases})==7566
    result={'passed':True,'trajectories':len(cases),'checked_decisions':sum(c['checked_decisions']for c in cases),
        'statevector_diagnostic_calls':sum(c['statevector_diagnostic_calls']for c in cases),
        'largest_rotation_error':max(c['largest_rotation_error']for c in cases),
        'largest_energy_error_ha':max(c['largest_energy_error_ha']for c in cases),
        'largest_physical_energy_error_ha':max(c['largest_physical_energy_error_ha']for c in cases),
        'additional_quantum_measurements':0,'cases':cases}
    write(root/'decision_replay_audit.json',result);print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','submit','summarize']:
        q=sub.add_parser(name);q.add_argument('--work',type=Path,required=True)
    q=sub.add_parser('task');q.add_argument('--work',type=Path,required=True);q.add_argument('--index',type=int,required=True)
    args=p.parse_args();root=args.work.resolve()
    if args.command=='prepare':prepare(root)
    elif args.command=='submit':submit(root)
    elif args.command=='summarize':summarize(root)
    else:worker(root,args.index)


if __name__=='__main__':main()

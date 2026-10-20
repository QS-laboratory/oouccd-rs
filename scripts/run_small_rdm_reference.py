"""Small exact-sector tomography reference using matched H2 measurement batches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--work",type=Path,required=True);args=p.parse_args();root=args.work.resolve()
    study=json.loads((root/"study.json").read_text());v=root/"verification"
    counts=v/"h2_reference_counts.jsonl";output=v/"h2_reference_reconstruction.jsonl"
    if counts.exists()or output.exists():raise RuntimeError("Small reference already started")
    start=time.perf_counter()
    subprocess.run([study["binaries"]["frozen_oo"]["path"],"study","--snapshot",str(v/"h2_snapshot.json"),
        "--shots","1000","--replicates","25","--seed","730261020","--output",str(counts)],check=True)
    subprocess.run([study["python"],str(root/"reconstruction_source/scripts/benchmark_rdm_reconstruction.py"),
        "--snapshot",str(v/"h2_snapshot.json"),"--data",str(counts),"--output",str(output),"--replicates","25",
        "--solver","CLARABEL","--methods","spatial_psd","dqg","mle","sector_mle"],check=True)
    statuses={}
    for line in output.read_text().splitlines():
        row=json.loads(line)
        if row["record_type"]!="trial":continue
        for result in row["results"]:
            item=statuses.setdefault(result["method"],{"completed":0,"failed":0})
            item["completed"if result["completed"]else"failed"]+=1
    (v/"small_reference_completed.json").write_text(json.dumps({"status":"completed","seconds":time.perf_counter()-start,
        "slurm_job_id":os.getenv("SLURM_JOB_ID"),"matched_batches":25,"methods":statuses,
        "input":str(counts),"output":str(output)},indent=2)+"\n")


if __name__=="__main__":main()

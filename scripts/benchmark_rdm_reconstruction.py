"""Benchmark physical reconstruction on matched, already-collected Pauli batches."""
import argparse
import json
from pathlib import Path
import resource
import time

from reconstruct_rdms import Reconstruction, score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replicates",type=int,default=25)
    parser.add_argument("--methods",nargs="+",choices=["spatial_psd","dqg","mle","sector_mle"],
                        default=["spatial_psd","dqg","mle"])
    parser.add_argument("--solver",choices=["SCS","CLARABEL"],default="SCS")
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError("Refusing to overwrite reconstruction benchmark")
    snapshot=json.loads(args.snapshot.read_text());models={};build_seconds={}
    for method in args.methods:
        started=time.perf_counter();models[method]=Reconstruction(snapshot,method,args.solver)
        build_seconds[method]=time.perf_counter()-started
    completed=0
    with args.data.open()as source,args.output.open("w")as output:
        output.write(json.dumps({"record_type":"metadata","snapshot":str(args.snapshot),
            "data":str(args.data),"methods":args.methods,"build_seconds":build_seconds,
            "requested_replicates":args.replicates})+"\n");output.flush()
        for line in source:
            row=json.loads(line)
            if row["record_type"]!="trial":continue
            if completed>=args.replicates:break
            results=[]
            for method,model in models.items():
                started=time.perf_counter()
                try:
                    result=model.solve(row["counts"])
                    result["score"]=score(snapshot,result["means"])
                    result["completed"]=True
                except Exception as error:
                    result={"method":method,"completed":False,"error":repr(error),
                            "seconds":time.perf_counter()-started}
                results.append(result)
            output.write(json.dumps({"record_type":"trial","index":row["index"],"seed":row["seed"],
                "raw_energy_error_ha":row["raw_energy_error_ha"],
                "affine_energy_error_ha":row["affine_energy_error_ha"],
                "rdm_raw_squared_error":row["rdm_raw_squared_error"],
                "rdm_affine_squared_error":row["rdm_affine_squared_error"],
                "results":results},allow_nan=False)+"\n");output.flush()
            completed+=1
    if completed!=args.replicates:raise RuntimeError("Input contained fewer batches than requested")
    args.output.with_suffix(".completed.json").write_text(json.dumps({"status":"completed", "replicates":completed,
        "maximum_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})+"\n")


if __name__=="__main__":main()

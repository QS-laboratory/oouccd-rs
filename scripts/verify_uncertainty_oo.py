"""Integration checks for legacy replay, honest held-out updates and spin metrics."""
import argparse
import itertools
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from reconstruct_rdms import array, plans_matrix, sector_map, spin_metric_maps


def validate_car_operators(snapshot):
    """Independent Fock-space operator check, not just an eigenvalue check."""
    if snapshot['n_spatial']!=2:raise ValueError('Small reference is defined for two spatial orbitals')
    maps=spin_metric_maps(snapshot);n=4;largest=0.0;pairs=[tuple(x)for x in snapshot['spin_plans']['pairs']]
    for electrons in range(5):
        fixture=dict(snapshot,n_elec=electrons);na,nb=(electrons+1)//2,electrons//2
        basis=[sum(1<<(2*i)for i in up)+sum(1<<(2*j+1)for j in down)
            for up in itertools.combinations(range(2),na)for down in itertools.combinations(range(2),nb)]
        lookup={mask:i for i,mask in enumerate(basis)};projection,dimension=sector_map(fixture)
        def matrix(ops):
            operator=np.zeros((dimension,dimension))
            for column,mask in enumerate(basis):
                current=mask;sign=1;nonzero=True
                for orbital,create in reversed(ops):
                    occupied=bool(current&(1<<orbital))
                    if occupied==create:nonzero=False;break
                    sign*=(-1)**((current&((1<<orbital)-1)).bit_count())
                    current^=1<<orbital
                if nonzero and current in lookup:operator[lookup[current],column]=sign
            return ((operator+operator.T)/2).ravel()
        definitions={
            'gamma':[[(i,True),(j,False)]for i in range(n)for j in range(n)],
            'D':[[(i,True),(j,True),(l,False),(k,False)]for i,j in pairs for k,l in pairs],
            'Q':[[(i,False),(j,False),(l,True),(k,True)]for i,j in pairs for k,l in pairs],
            'G':[[(i,True),(j,False),(l,True),(k,False)]for i in range(n)for j in range(n)for k in range(n)for l in range(n)]}
        for name,definitions_for_metric in definitions.items():
            direct=np.asarray([matrix(ops)for ops in definitions_for_metric])
            mapped=(maps[name]@projection).toarray()
            error=float(np.max(np.abs(direct-mapped)));largest=max(largest,error)
            assert error<1e-12,(electrons,name,error)
    return {'tested_particle_numbers':list(range(5)),'largest_operator_error':largest}


def execute(command, log):
    start=time.perf_counter()
    with log.open("w") as handle:
        subprocess.run([str(x) for x in command], check=True, stdout=handle, stderr=handle)
    return time.perf_counter()-start


def compare(current, baseline):
    assert current["status"] == baseline["status"] == "completed"
    maximum=0.0
    for a,b in zip(current["iterations"],baseline["iterations"],strict=True):
        for key in ("exact_energy_ha","measured_energy_ha","theta","orbital_rotation"):
            difference=float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
            maximum=max(maximum,difference)
        assert a["rdm_evaluations"] == b["rdm_evaluations"]
        assert a["modeled_shots"] == b["modeled_shots"]
    assert maximum < 2e-12, maximum
    return maximum


def validate_maps(snapshot):
    n=snapshot["n_spatial"]; nspin=2*n; mu=np.asarray(snapshot["exact_means"])
    maps=spin_metric_maps(snapshot); eigenvalues={}
    for name,matrix in maps.items():
        dimension=int(round(np.sqrt(matrix.shape[0])))
        value=np.asarray(matrix @ mu).reshape(dimension,dimension)
        assert np.max(np.abs(value-value.T))<1e-12,(name,"asymmetry")
        eigenvalues[name]=float(np.linalg.eigvalsh(value).min())
        assert eigenvalues[name]>-1e-10,(name,eigenvalues[name])
    spatial=plans_matrix(snapshot["plan"]["one_body"]+snapshot["plan"]["two_body"],len(mu))
    gamma=(maps["gamma"] @ mu).reshape(nspin,nspin)
    one=np.asarray(spatial[:n*n] @ mu).reshape(n,n)
    assert np.max(np.abs(one-gamma[::2,::2]-gamma[1::2,1::2]))<1e-12
    projection,dimension=sector_map(snapshot)
    # Uniform state in the fixed sector gives a separate feasible point and
    # checks CAR mappings without feeding a ground-state RDM into the solver.
    mixed=projection @ (np.eye(dimension)/dimension).ravel()
    for name,matrix in maps.items():
        d=int(round(np.sqrt(matrix.shape[0])));value=np.asarray(matrix @ mixed).reshape(d,d)
        assert np.linalg.eigvalsh(value).min()>-1e-10,(name,"mixed sector")
    assert np.max(np.abs(np.asarray(snapshot["constraints"]["rows"]) @ mixed
                         -snapshot["constraints"]["targets"]))<1e-12
    result={"minimum_eigenvalues":eigenvalues,"fixed_sector_dimension":dimension}
    if n==2:result['independent_car_operator_check']=validate_car_operators(snapshot)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--binary",type=Path,default=Path("target/release/examples/evaluate_shot_noise"))
    p.add_argument("--frozen-binary",type=Path,default=Path("target/release/examples/frozen_oo"))
    p.add_argument("--previous-study",type=Path,default=Path("local_runs/lih_shot_noise_20seeds_20261020"))
    args=p.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    binary=args.binary.resolve();frozen=args.frozen_binary.resolve()
    for mode in ("statevector","finite_shot"):
        for method in ("sequential","newton_raphson"):
            path=work/f"{mode}_{method}.json"
            if not path.exists():
                measurement={"type":mode}
                if mode=="finite_shot":measurement.update(shots=1000,seed=20261020)
                task={"task_name":f"verification_{mode}_{method}",
                    "chemistry":{"geometry":"H 0 0 0; H 0 0 0.735","basis":"sto-3g"},
                    "method":{"label":"verification","max_macro_iterations":2,"energy_tolerance":-1.,
                        "rdm_evaluation":measurement,
                        "ucc_optimizer":{"ansatz_config":{"type":"UCCD"},"solver_config":{"type":"sequential"}},
                        "orbital_optimizer":{"solver_config":{"type":method}}}}
                path.write_text(json.dumps(task,indent=2)+"\n")
    results={"legacy_replay":{},"guard_smoke":{},"spin_maps":{}}
    for mode in ("statevector","finite_shot"):
        for method in ("sequential","newton_raphson"):
            name=f"{mode}_{method}";config=work/f"{name}.json"
            current=work/f"{name}_current.json";baseline=work/f"{name}_baseline.json"
            execute([binary,"--config",config,"--output",current],work/f"{name}_current.log")
            execute([args.baseline,"--config",config,"--output",baseline],work/f"{name}_baseline.log")
            results["legacy_replay"][name]=compare(json.loads(current.read_text()),json.loads(baseline.read_text()))
    for method in ("sequential","newton_raphson"):
        config=json.loads((work/f"finite_shot_{method}.json").read_text())
        config["method"]["orbital_statistics"]={"mitigation":"affine","acceptance":"heldout","record_counts":True}
        cpath=work/f"guard_{method}.json";cpath.write_text(json.dumps(config)+"\n")
        for exact_ucc in (False,True):
            name=f"guard_{method}_{exact_ucc}";output=work/f"{name}.json"
            command=[binary,"--config",cpath,"--output",output]+(["--exact-ucc"]if exact_ucc else[])
            execute(command,work/f"{name}.log");report=json.loads(output.read_text())
            assert report["status"]=="completed",report.get("error")
            for point in report["iterations"][1:]:
                record=point["oo"]["statistics"]
                assert record["training_shots"]==800 and record["validation_shots"]==200
                assert record["correction_residual_after"]<1e-8
                if record["accepted"]:
                    assert record["validation_upper_bound_ha"]<0
                else:
                    assert abs(point["oo"]["exact_energy_after_ha"]-point["oo"]["exact_energy_before_ha"])<1e-12
            expected_batches=5 if exact_ucc else report["rdm_evaluations"]
            assert report["modeled_shots"]==expected_batches*1000*report["unique_measured_pauli_words"]
            results["guard_smoke"][name]={"modeled_shots":report["modeled_shots"],
                "accepted":[x["oo"]["statistics"]["accepted"]for x in report["iterations"][1:]]}
    sources=[("h2",work/"statevector_sequential.json",work/"statevector_sequential_current.json",1)]
    previous=json.loads((args.previous_study/"study.json").read_text())
    old=Path(next(w['work']for w in previous['workspaces']if w['distance']==1.6))
    manifest=json.loads((old/"manifest.json").read_text());task=manifest["tasks"][0]
    sources.append(("lih",Path(task["config"]),Path(task["result"]),3))
    for name,config,trajectory,iteration in sources:
        output=work/f"{name}_snapshot.json"
        execute([frozen,"export","--config",config,"--trajectory",trajectory,"--iteration",iteration,"--output",output],work/f"{name}_snapshot.log")
        results["spin_maps"][name]=validate_maps(json.loads(output.read_text()))
        trial=work/f"{name}_fixed.jsonl"
        execute([frozen,"study","--snapshot",output,"--shots",1000,"--replicates",3,"--seed",20269000,"--output",trial],work/f"{name}_fixed.log")
        rows=[json.loads(line)for line in trial.read_text().splitlines()]
        assert len(rows)==4
        assert all(x["status"]=="completed"for row in rows[1:]for x in row["outcomes"])
        assert all(len(row["outcomes"])==12 for row in rows[1:])
    (work/"verification.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()

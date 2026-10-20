#!/usr/bin/env bash
# Submit from the repository root with an existing prepared paper workspace:
#   sbatch scripts/benchmark_paper_cpus.sbatch /absolute/path/to/workspace
# One allocation runs all trials sequentially on the same node.
#SBATCH --job-name=lih-cpu-bench
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
REFERENCE="$(realpath "${1:?Pass a prepared paper workspace}")"
WORK="$(mktemp -d "$PWD/local_runs/paper-cpu-bench-${SLURM_JOB_ID}.XXXXXX")"
echo "Benchmark workspace: $WORK"
mkdir "$WORK/configs" "$WORK/results" "$WORK/logs" "$WORK/bin"
cp "$REFERENCE/bin/oouccd-rs" "$WORK/bin/oouccd-rs"
cp "$REFERENCE/manifest.json" "$WORK/paper-manifest.json"
cp "$0" "$WORK/benchmark.sbatch"
PY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$REFERENCE/manifest.json")"
unset DB_FILEPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export WORK REFERENCE
hostname | tee "$WORK/hostname.txt"
lscpu > "$WORK/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$WORK/slurm-job.txt"
"$PY" -m pip freeze > "$WORK/python-freeze.txt"
"$PY" - <<'PY'
import json, os
from pathlib import Path
work, reference = Path(os.environ['WORK']), Path(os.environ['REFERENCE'])
for index in (3, 17):
    task = json.loads((reference/'configs'/f'task_{index:03d}.json').read_text())
    task['task_name'] = 'CpuBenchmark_' + task['method']['label']
    task['method']['max_macro_iterations'] = 1
    task['method']['energy_tolerance'] = -1.0
    (work/'configs'/f'task_{index:03d}.json').write_text(json.dumps(task, indent=2)+'\n')
PY
# ABBA order: two trials at each CPU count, with a fresh process/DB per trial.
# UCCD covers the amplitude kernel used by both OO methods; UCCSD includes singles.
for index in 003 017; do
    repeat=0
    for cpus in 1 4 4 1; do
        name="task_${index}_trial_${repeat}_cpu_${cpus}"
        export RAYON_NUM_THREADS="$cpus"
        echo "Starting $name"
        srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task="$cpus" \
            --cpu-bind=cores --distribution=block:block \
            bash -c 'taskset -pc $$ >&2; exec "$@"' _ \
            /usr/bin/time \
            --format='{"elapsed_s":%e,"user_s":%U,"system_s":%S,"max_rss_kib":%M}' \
            --output="$WORK/results/$name.time.json" \
            "$WORK/bin/oouccd-rs" --config-filepath "$WORK/configs/task_$index.json" \
            --db-filepath "$WORK/results/$name.db" \
            >"$WORK/logs/$name.out" 2>"$WORK/logs/$name.err"
        echo "Finished $name: $(cat "$WORK/results/$name.time.json")"
        repeat=$((repeat + 1))
    done
done
"$PY" - <<'PY'
import json, os, re, sqlite3, statistics
from pathlib import Path
work = Path(os.environ['WORK'])
report = {'node': (work/'hostname.txt').read_text().strip(),
          'protocol': 'One complete first macro-iteration at R=1.60 A, fresh zero initialization; ABBA 1,4,4,1 CPUs for each method; BLAS/OpenMP=1; Rayon=allocated CPUs',
          'trials': [], 'summary': []}
for index, method in ((3,'UCCD(seq)'),(17,'UCCSD(seq)')):
    energies = []
    for timing in sorted((work/'results').glob(f'task_{index:03d}_*.time.json')):
        name = timing.name.removesuffix('.time.json')
        cpus = int(name.rsplit('_',1)[1])
        row = json.loads(timing.read_text())
        with sqlite3.connect((work/'results'/f'{name}.db').as_uri()+'?mode=ro',uri=True) as conn:
            runs = conn.execute('SELECT final_energy,is_completed FROM runs').fetchall()
            assert len(runs)==1 and runs[0][1], f'Incomplete {name}'
            energies.append(runs[0][0])
            indices = conn.execute('SELECT DISTINCT macro_idx FROM macro_iterations').fetchall()
            assert indices==[(1,)], f'Wrong workload in {name}'
        log = (work/'logs'/f'{name}.err').read_text()
        sweep_s = float(re.search(r'UCC done \(([\d.]+)s\)',log).group(1))
        affinity = re.search(r'current affinity list: ([\d,-]+)',log).group(1)
        report['trials'].append(dict(row, method=method, cpus=cpus,
                                     allocated_cpu_seconds=cpus*row['elapsed_s'],
                                     measured_cpu_seconds=row['user_s']+row['system_s'],
                                     energy_hartree=runs[0][0], amplitude_sweep_s=sweep_s,
                                     affinity=affinity, name=name))
    assert max(energies)-min(energies)<1e-12, f'CPU counts changed energy for {method}'
    rows = [row for row in report['trials'] if row['method']==method]
    times = {n: statistics.median(row['elapsed_s'] for row in rows if row['cpus']==n) for n in (1,4)}
    speedup = times[1]/times[4]
    summary = {'method':method, 'one_cpu_median_s':times[1], 'four_cpu_median_s':times[4],
               'speedup':speedup, 'parallel_efficiency':speedup/4,
               'allocated_cpu_time_ratio_4_to_1':4/speedup,
               'energy_spread_hartree':max(energies)-min(energies)}
    report['summary'].append(summary)
    print(json.dumps(summary,indent=2))
(work/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
PY

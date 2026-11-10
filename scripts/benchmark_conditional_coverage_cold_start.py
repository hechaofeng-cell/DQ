"""Measure per-pool cold input preparation in isolated serial processes."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

import conditional_coverage_common as cc
from run_conditional_coverage_selection import run_method


def worker(out,method):
    thresholds=json.loads((out/'thresholds.json').read_text())
    start=time.perf_counter()
    # Only exchange methods need the directed same-cell pair index.
    contexts=cc.load_contexts() if method in ['CS','AS'] else cc.old.load_contexts()
    loaded=time.perf_counter()-start
    selection_seconds=0.
    sets={}
    seed=2026110500 if method in ['R','CS','AS'] else -1
    for ci,(c,ctx) in enumerate(contexts.items()):
        selected,initial,record,trace=run_method(ctx,method,200,20,thresholds[c],seed,ci)
        selection_seconds+=record['seconds']
        sets[c]=selected
    elapsed=time.perf_counter()-start
    with np.load(out/'selected_sets.npz',allow_pickle=False) as z:
        for c,selected in sets.items():
            assert np.array_equal(selected,z[cc.selection_key(c,method,seed)])
    return dict(method=method,seed=seed,context_loading_seconds=loaded,
        algorithm_seconds=selection_seconds,cold_pool_pipeline_seconds=elapsed,
        peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        frozen_primary_outputs_reproduced=True,
        import_startup_and_output_verification_included=False,
        includes_witness_calibration=False)


def run(out):
    assert not (out/'cold_start_runs.csv').exists(),'Cold-start outputs already exist.'
    rng=np.random.default_rng(2026110597)
    rows=[]
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for repeat in range(3):
        for method in rng.permutation(cc.METHODS):
            result=subprocess.run([sys.executable,__file__,'--out',str(out),'--method',str(method)],
                env=env,text=True,capture_output=True,check=True)
            r=json.loads(result.stdout)
            rows.append(dict(replay=repeat,**r))
    cc.write(out/'cold_start_runs.csv',rows)
    cc.dump(out/'cold_start_audit.json',dict(status='pass',runs=len(rows),
        one_isolated_process_per_method_replay=True,serial_execution=True,
        classes=cc.old.CLASSES,selection_seed=2026110500,
        pair_index_preparation_charged_only_to_exchange_methods=True,
        startup_imports_and_validation_excluded=True,threshold_calibration_cost_separate=True,
        code_sha256=cc.sha(__file__)))
    cc.log(out,'cold_start_benchmark_complete',runs=len(rows))


if __name__=='__main__':
    parser=cc.parsed_args(__doc__)
    parser.add_argument('--method',choices=cc.METHODS)
    args=parser.parse_args()
    if args.method:
        print(json.dumps(worker(args.out,args.method)))
    else:
        run(args.out)

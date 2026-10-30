#!/usr/bin/env python3
"""Re-infer a fixed visual diagnostic suite and compare with saved predictions."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from automatic_strengthening_common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=BASE/'regression_suite.csv')
    parser.add_argument('--output',type=Path,default=BASE/'regression_replay');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();rows=read_csv(args.manifest);groups=defaultdict(list)
    for row in rows:groups[row['model'],int(row['seed'])].append(row)
    if args.smoke:groups=dict(list(sorted(groups.items()))[:1])
    import torch
    torch.set_num_threads(8)
    import sixclass_car_support as support
    support.configure();b=support.benchmark
    b.CHECKPOINT_ROOT=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'
    results=[]
    for (model,seed),members in sorted(groups.items()):
        for row in members:
            if sha256(row['crop_path'])!=row['crop_sha256']:raise ValueError('Regression input hash changed')
            if sha256(row['checkpoint_path'])!=row['checkpoint_sha256']:raise ValueError('Regression checkpoint hash changed')
        identities={r['sample_id']:r for r in members};targets=list(identities.values())
        context=dict(rows=targets,train_rows=read_csv(A/'manifests/train.csv'))
        folder=args.output/'runs'/model/str(seed);folder.mkdir(parents=True,exist_ok=True)
        original=ROOT/'artifacts/sixclass_car_extension_20261004/benchmark/runs'/model/str(seed)/'train_features.npz'
        if not (folder/'train_features.npz').exists():shutil.copy2(original,folder/'train_features.npz')
        predicted=b.infer_and_extract(args.output,context,model,seed,0)['candidate_logits'].argmax(1)
        lookup=dict(zip(identities,predicted))
        for row in members:results.append(dict(sample_id=row['sample_id'],model=model,seed=seed,
            expected_predicted_label=int(row['expected_predicted_label']),actual_predicted_label=int(lookup[row['sample_id']]),
            matches=bool(int(lookup[row['sample_id']])==int(row['expected_predicted_label']))))
        print(f'[replay] {model}/{seed} targets={len(targets)} matches={sum(r["matches"] for r in results)}/{len(results)}',flush=True)
    write_csv(args.output/'replay_results.csv',results)
    valid=bool(results) and all(r['matches'] for r in results)
    write_json(args.output/'audit.json',dict(status='pass' if valid else 'failed',scope='smoke' if args.smoke else 'full',
        comparisons=len(results),weights_retrained=False,manifest_sha256=sha256(args.manifest)))
    if not valid:raise ValueError('Frozen regression replay differs; inspect input/weight/software execution')


if __name__=='__main__':main()

"""Constants and helpers for the new, independently sampled six-class pool."""
from pathlib import Path
import json
import os
from dual_vlm_common import CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/home/hcf/project/QE')
BASE=ROOT/'artifacts/sixclass_independent_confirmation_20261004'
A=ROOT/'artifacts/sixclass_dual_vlm_replaced_20261004'
WEIGHTS=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'
PYTHON=PROJECT/'.venv/bin/python'
PROTOCOL=ROOT/'docs/sixclass_independent_confirmation_protocol_20261004.md'
LABELS=dict(zip(CLASSES,range(6)))
COCO=dict(zip(CLASSES,(16,15,17,18,0,2)))
STAGES=('prepare','annotate','freeze_selection','infer','confirm','compare','transfer','robustness','report')

def note(stage,**kwargs):
    item=dict(status='running',stage=stage,pid=os.getpid(),heartbeat=now(),**kwargs)
    write_json(BASE/'progress.json',item)
    print(f'[{now()}] {stage} {json.dumps(kwargs,ensure_ascii=False)}',flush=True)

def context():
    import analyze_dual_vlm_study as analysis
    analysis.STUDY=BASE
    analysis.job_path=lambda backend,domain:BASE/'annotations'/backend/domain
    analysis.schema_path=lambda domain:BASE/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')
    labels,_=analysis.native_labels(False)
    contexts,universe=analysis.contexts(labels,False)
    train=read_csv(BASE/'manifests/train.csv')
    for item in contexts.values():item['train_rows']=train
    return labels,contexts,universe

def freeze(path,paths):
    value={str(p):sha256(p) for p in paths}
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError(f'Changed frozen inputs: {path}')
    write_json(path,value)

def verify(path):
    mapping=json.loads(path.read_text())
    bad=[p for p,digest in mapping.items() if not Path(p).is_file() or sha256(p)!=digest]
    if bad:raise ValueError(f'Freeze changed: {bad[:3]}')
    return len(mapping)

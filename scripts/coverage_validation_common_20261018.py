"""Frozen inputs and pure NumPy utilities for the coverage validation study."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/coverage_diagnostic_validation_20261018'
SOURCE = ROOT / 'artifacts/other_classes_specific_experiment_20261018'
BASE = ROOT / 'artifacts/coco_five_dimension_fine_selection_20261015'
EXTERNAL = ROOT / 'artifacts/objects365_independent_confirmation_20261005'
CLASSES = ['horse', 'sheep', 'person', 'car']
ALL_CLASSES = ['dog', 'cat', *CLASSES]
MODELS = ['yolo26s_cls', 'pe_core_b16', 'siglip2_base', 'mobileclip2_s0', 'aimv2_large']
SEEDS = [20260916, 20260917, 20260918]
CN = dict(horse='马', sheep='羊', person='人', car='汽车')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen object changed: {path}'
    else:
        dump(path, value)


def log(stage, **values):
    message = dict(time=now(), stage=stage, **values)
    print(json.dumps(message, ensure_ascii=False), flush=True)
    with (OUT / 'research_log.md').open('a') as f:
        f.write('\n' + json.dumps(message, ensure_ascii=False) + '\n')


def manifest():
    return read(BASE / 'manifest.csv')


def load_contexts():
    rows = manifest(); risk = {x['sample_id']: x for x in read(SOURCE/'risk_scores_and_tiers.csv')}
    embpath = ROOT/'artifacts/sixclass_independent_confirmation_20261004/embeddings/vit_b16_candidates.npz'
    with np.load(embpath, allow_pickle=False) as z:
        lookup = {s: i for i, s in enumerate(z['sample_ids'].astype(str))}
        assert len(lookup) == 2400 and set(lookup) == {r['sample_id'] for r in rows}
        embeddings = z['embeddings'].astype(float)
    out = {}
    for c in CLASSES:
        with np.load(SOURCE/f'{c}_state_matrix.npz', allow_pickle=False) as z:
            ids = z['sample_ids'].astype(str); matrix = z['matrix'].astype(bool)
            tiers = z['tiers'].copy(); original = z['selected'].copy(); random = z['random'].copy()
        states = json.loads((SOURCE/f'{c}_state_dictionary.json').read_text())['states']
        crows = [r for r in rows if r['class_name'] == c]
        assert ids.tolist() == [r['sample_id'] for r in crows]
        keys = [(int(tiers[i]), round(float(risk[s]['score_midpoint'])*1e8),
                 round((float(risk[s]['score_upper'])-float(risk[s]['score_lower']))*1e8)) for i,s in enumerate(ids)]
        unique = sorted(set(keys)); cell = np.array([unique.index(k) for k in keys])
        fields = list(dict.fromkeys(tuple(s[:2]) for s in states))
        field_indices = [np.array([j for j,s in enumerate(states) if tuple(s[:2]) == field]) for field in fields]
        field_weights = np.zeros(len(states))
        for ix in field_indices: field_weights[ix] = 1 / len(ix)
        E = embeddings[[lookup[s] for s in ids]]
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        out[c] = dict(ids=ids, rows=crows, matrix=matrix, tiers=tiers, states=states,
                      keys=unique, cell=cell, field_indices=field_indices, field_weights=field_weights,
                      embeddings=E, original=original, original_random=random)
    return out


def quotas(ctx, budget):
    assert budget % 5 == 0
    counts = np.bincount(ctx['cell'], minlength=len(ctx['keys']))
    result = np.zeros(len(counts), dtype=int)
    for tier in range(1, 6):
        ix = [j for j,k in enumerate(ctx['keys']) if k[0] == tier]
        total = int(counts[ix].sum()); numer = {j: int(counts[j])*(budget//5) for j in ix}
        for j in ix: result[j] = numer[j] // total
        rem = budget//5-int(result[ix].sum())
        for j in sorted(ix, key=lambda j: (-(numer[j] % total), ctx['keys'][j]))[:rem]: result[j] += 1
    assert result.sum() == budget and np.all(result <= counts)
    return result


def validate_selection(ctx, selected, budget):
    selected = np.asarray(selected)
    assert len(selected) == len(set(selected.tolist())) == budget
    assert np.array_equal(np.bincount(ctx['cell'][selected], minlength=len(ctx['keys'])), quotas(ctx,budget))
    assert np.bincount(ctx['tiers'][selected], minlength=6)[1:].tolist() == [budget//5]*5


def choose(ctx, budget, method, cap=20, weighting='state', matrix=None):
    X = ctx['matrix'] if matrix is None else matrix
    q = quotas(ctx,budget); used = np.zeros(len(q),int); available = np.ones(len(X),bool)
    count = np.zeros(X.shape[1],int); selected=[]
    weights = ctx['field_weights'] if weighting=='field' else np.ones(X.shape[1])
    if method=='shared_greedy': weights = weights * (np.arange(X.shape[1]) < 71)
    static = (X / np.maximum(X.sum(0),1)).sum(1) if method=='static_rarity' else None
    E = ctx['embeddings']
    centroid = E.mean(0); centroid /= max(np.linalg.norm(centroid),1e-12)
    distance = None
    for _ in range(budget):
        feasible=np.flatnonzero(available & (used[ctx['cell']] < q[ctx['cell']]))
        assert len(feasible)
        if method in ('expanded_greedy','shared_greedy'):
            marginal=np.where(count<cap,np.log(count+2)-np.log(count+1),0)*weights
            gain=np.sum(X[feasible]*marginal,axis=1)
        elif method=='static_rarity': gain=static[feasible]
        elif method=='vit_kcenter':
            gain=E[feasible]@centroid if distance is None else distance[feasible]
        else: raise ValueError(method)
        ties=feasible[np.isclose(gain,gain.max(),atol=1e-12,rtol=0)]
        i=int(min(ties,key=lambda i:ctx['ids'][i]))
        selected.append(i);available[i]=False;used[ctx['cell'][i]]+=1;count+=X[i]
        if method=='vit_kcenter':
            d=np.maximum(0,1-E@E[i]); distance=d if distance is None else np.minimum(distance,d)
    result=np.array(selected,dtype=int); validate_selection(ctx,result,budget); return result


def random_choices(ctx,budget,repeats=1000,seed=20261018):
    # Preserve all existing primary draws exactly; other budgets are newly sampled.
    if budget==200 and repeats==1000 and seed==20261018: return ctx['original_random'].copy()
    rng=np.random.default_rng(seed); q=quotas(ctx,budget); parts=[]
    for j,n in enumerate(q):
        if n:
            members=np.flatnonzero(ctx['cell']==j)
            ix=np.argsort(rng.random((repeats,len(members))),axis=1,kind='stable')[:,:n]
            parts.append(members[ix])
    result=np.concatenate(parts,axis=1)
    for selected in result: validate_selection(ctx,selected,budget)
    return result


def coverage(ctx, selections, cap=20):
    a=np.asarray(selections)
    if a.ndim==1:a=a[None]
    counts=ctx['matrix'][a].sum(1)
    results={}
    for ns,mask in [('all',np.ones(counts.shape[1],bool)),('shared',np.arange(counts.shape[1])<71),('specific',np.arange(counts.shape[1])>=71)]:
        n=counts[:,mask]
        results[ns]=dict(presence_pct=100*(n>0).mean(1),support_completion_pct=100*np.minimum(n/cap,1).mean(1),threshold_pct=100*(n>=cap).mean(1))
    return results


def predictions(split, rows):
    ids=[r['sample_id'] for r in rows]; truth=np.array([int(r['label']) for r in rows]); result={}; hashes={}
    for m in MODELS:
        pp=[]
        for seed in SEEDS:
            path=ROOT/f'artifacts/recent_five_models_20261009/runs/{m}/{seed}/{split}_outputs.npz'
            with np.load(path,allow_pickle=False) as z:
                names=z['sample_ids'].astype(str).tolist(); lookup={s:i for i,s in enumerate(names)}
                assert len(lookup)==len(names) and set(ids)<=set(lookup)
                ix=[lookup[s] for s in ids]
                assert np.array_equal(z['labels'][ix],truth)
                a=z['logits'][ix];assert a.shape==(len(ids),6) and np.isfinite(a).all()
                pp.append(a.argmax(1))
            hashes[str(path)]=sha(path)
        result[m]=np.asarray(pp)
    return result,hashes

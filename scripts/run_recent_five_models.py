#!/usr/bin/env python3
"""Independent recent-five comparison; no historical classifier predictions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/recent_five_models_20261009'
sys.path.insert(0, str(OUT / 'vendor/python'))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

CONFIG_PATH = ROOT / 'configs/recent_five_models_20261009.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
CLASSES = CONFIG['classes']
DISPLAY = {'yolo26s_cls': 'YOLO26-S-cls', 'pe_core_b16': 'PE-Core-B16-224',
           'siglip2_base': 'SigLIP 2 Base', 'mobileclip2_s0': 'MobileCLIP2-S0',
           'aimv2_large': 'AIMv2-Large'}


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message, flush=True)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    def normalize(value):
        if isinstance(value, set): return sorted(value)
        raise TypeError(f'Cannot serialize {type(value)}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=normalize) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save_pt(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frame(path):
    return pd.read_csv(ROOT / path, dtype={'sample_id': str, 'image_id': str})


def splits():
    ev = frame(CONFIG['eval_manifest'])
    return {'train': frame(CONFIG['train_manifest']),
            'val': ev[ev.split == 'val'].reset_index(drop=True),
            'test': ev[ev.split == 'test'].reset_index(drop=True),
            **{p: frame(path) for p, path in CONFIG['pools'].items()}}


def prepare():
    rows = splits(); hashes = {}; manifests = [CONFIG['train_manifest'], CONFIG['eval_manifest'],
        *CONFIG['pools'].values(), str(CONFIG_PATH.relative_to(ROOT)),
        'docs/recent_five_model_comparison_protocol_20261009.md', str(Path(__file__).relative_to(ROOT)),
        'scripts/plot_recent_five_models.py']
    for p in manifests: hashes[p] = sha(ROOT / p)
    crop_records = []; unique = {}
    for split, df in rows.items():
        assert df.sample_id.is_unique
        assert set(df.class_name) == set(CLASSES)
        assert all(int(r.label) == CLASSES.index(r.class_name) for r in df.itertuples())
        assert df.groupby('class_name').size().nunique() == 1
        log(f'[prepare] {split}: {len(df)} targets')
        for r in df.itertuples():
            p = Path(r.crop_path)
            if not p.is_file(): raise FileNotFoundError(p)
            if str(p) not in unique: unique[str(p)] = sha(p)
            crop_records.append({'split': split, 'sample_id': r.sample_id, 'crop_path': str(p), 'sha256': unique[str(p)]})
    crop_df = pd.DataFrame(crop_records)
    for left in ('train', 'val'):
        for right in rows:
            if right == left or (left == 'val' and right == 'train'): continue
            overlap = set(crop_df[crop_df.split == left].sha256) & set(crop_df[crop_df.split == right].sha256)
            if overlap: raise RuntimeError(f'Exact crop leakage: {left} / {right}, {len(overlap)} hashes')
    for pool in CONFIG['pools']:
        portable = ROOT / CONFIG['portable_states']
        for suffix in ('targets.csv', 'states.json', 'predictions.npz'):
            p = portable / f'{pool}_{suffix}'; hashes[str(p.relative_to(ROOT))] = sha(p)
        with np.load(portable / f'{pool}_predictions.npz') as z:
            assert z['sample_ids'].tolist() == rows[pool].sample_id.tolist()
            for source in ('local', 'online'):
                order = z[source + '_full_orders']
                assert sorted(order.reshape(-1).tolist()) == list(range(len(rows[pool])))
                for c in range(6): assert (rows[pool].label.to_numpy()[order[c]] == c).all()
    freeze = OUT / 'input_freeze.json'
    if freeze.exists():
        previous = json.loads(freeze.read_text())
        if previous['files'] != hashes: raise RuntimeError('Frozen configuration or inputs changed; use a new experiment version')
    else:
        dump(freeze, {'files': hashes, 'crop_hash_count': len(unique), 'config': CONFIG})
        crop_df.to_csv(OUT / 'crop_content_audit.csv', index=False)
    dump(OUT/'environment.json', dict(python=sys.version, executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ('torch','torchvision','transformers','ultralytics','open_clip_torch','timm','numpy','pandas','scikit-learn')},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    log('[prepare] all paths, label mappings, class balance, target order and exact crop separation passed')


class Images(Dataset):
    def __init__(self, df, transform): self.df = df.to_dict('records'); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df[i]
        with Image.open(r['crop_path']) as im: x = self.transform(im.convert('RGB'))
        return x, int(r['label'])


def transform(meta, train=False):
    size = meta['input_size']
    ops = [T.Resize(round(size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC)]
    ops += ([T.RandomCrop(size), T.RandomHorizontalFlip(), T.ColorJitter(.15, .15, .1)]
            if train else [T.CenterCrop(size)])
    return T.Compose(ops + [T.ToTensor(), T.Normalize(meta['mean'], meta['std'])])


class Classifier(nn.Module):
    def __init__(self, vision, kind, dim):
        super().__init__(); self.vision = vision; self.kind = kind
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6))
    def forward(self, x):
        if self.kind == 'hf':
            out = self.vision(pixel_values=x)
            feat = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(1)
        else:
            feat = self.vision(x)
            if isinstance(feat, tuple): feat = feat[-1]
        return self.head(feat)


def load_model(name):
    meta = {'name': name, 'display_name': DISPLAY[name], 'input_size': CONFIG['input_sizes'][name]}
    if name == 'yolo26s_cls':
        from ultralytics import YOLO
        from ultralytics.data.augment import DEFAULT_MEAN, DEFAULT_STD
        path = OUT / 'weights/yolo26s-cls.pt'
        vision = YOLO(str(path)).model
        dim = vision.model[-1].linear.in_features
        vision.model[-1].linear = nn.Identity()
        # Ultralytics inference loading freezes parameters; restore full fine-tuning.
        for p in vision.parameters(): p.requires_grad_(True)
        meta.update(source='Ultralytics/assets/yolo26s-cls.pt', mean=list(DEFAULT_MEAN), std=list(DEFAULT_STD), weight_path=str(path))
        kind = 'plain'
    elif name == 'pe_core_b16':
        sys.path.insert(0, '/home/hcf/.cache/qe_external/perception_models')
        from core.vision_encoder import pe
        from core.vision_encoder.config import fetch_pe_checkpoint
        path = Path(fetch_pe_checkpoint('PE-Core-B16-224'))
        vision = pe.VisionTransformer.from_config('PE-Core-B16-224')
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in sd: sd = sd['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        if any(k.startswith('visual.') for k in sd): sd = {k[7:]: v for k, v in sd.items() if k.startswith('visual.')}
        vision.load_state_dict(sd, strict=True)
        dim = vision.output_dim
        meta.update(source='facebook/PE-Core-B16-224', mean=[.5]*3, std=[.5]*3, weight_path=str(path), weight_load='strict vision keys')
        kind = 'plain'
    elif name == 'mobileclip2_s0':
        import open_clip
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download('timm/MobileCLIP2-S0-OpenCLIP', 'open_clip_pytorch_model.bin'))
        clip = open_clip.create_model('MobileCLIP2-S0', pretrained=str(path))
        vision = clip.visual; del clip
        dim = open_clip.get_model_config('MobileCLIP2-S0')['embed_dim']
        meta.update(source='timm/MobileCLIP2-S0-OpenCLIP:dfndr2b', mean=[0.]*3, std=[1.]*3, weight_path=str(path), weight_load='OpenCLIP strict checkpoint')
        kind = 'plain'
    else:
        from transformers import SiglipVisionModel, Aimv2VisionModel, AutoImageProcessor
        from huggingface_hub import hf_hub_download
        source = 'google/siglip2-base-patch16-224' if name == 'siglip2_base' else 'apple/aimv2-large-patch14-224'
        cls = SiglipVisionModel if name == 'siglip2_base' else Aimv2VisionModel
        vision, loading = cls.from_pretrained(source, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'Pretrained vision loading failed: {loading}')
        unexpected = loading.get('unexpected_keys', [])
        if any(not (k.startswith('text_model.') or k in ('logit_scale', 'logit_bias')) for k in unexpected):
            raise RuntimeError(f'Unexpected vision weights: {loading}')
        proc = AutoImageProcessor.from_pretrained(source)
        path = Path(hf_hub_download(source, 'model.safetensors'))
        dim = vision.config.hidden_size
        meta.update(source=source, mean=list(proc.image_mean), std=list(proc.image_std), weight_path=str(path), loading_audit=loading)
        kind = 'hf'
    model = Classifier(vision, kind, dim)
    meta.update(feature_dim=dim, parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), weight_sha256=sha(path))
    if meta['parameters'] != meta['trainable_parameters']: raise RuntimeError('Unexpected frozen parameters')
    return model, meta


def metrics(y, logits):
    p = np.asarray(logits).argmax(-1)
    pr, re, f, support = precision_recall_fscore_support(y, p, labels=range(6), zero_division=0)
    result = dict(accuracy=float((p == y).mean()), macro_precision=float(pr.mean()), macro_recall=float(re.mean()), macro_f1=float(f.mean()))
    for c, name in enumerate(CLASSES):
        result.update({name + '_precision': float(pr[c]), name + '_recall': float(re[c]), name + '_f1': float(f[c])})
    return result


@torch.no_grad()
def predict(model, df, meta):
    model.eval(); result = []
    loader = DataLoader(Images(df, transform(meta)), batch_size=CONFIG['micro_batch'], shuffle=False,
                        num_workers=CONFIG['workers'], pin_memory=True)
    for x, y in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(x.cuda(non_blocking=True))
        if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite inference logits')
        result.append(logits.float().cpu().numpy())
    return np.concatenate(result)


def smoke(name):
    seed_all(20261009); model, meta = load_model(name); model.cuda().train()
    data = Images(splits()['train'], transform(meta, True))
    x = torch.stack([data[i][0] for i in range(CONFIG['micro_batch'])]).cuda()
    y = torch.tensor([data[i][1] for i in range(CONFIG['micro_batch'])], device='cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    torch.cuda.reset_peak_memory_stats()
    before = model.head[-1].weight.detach().clone()
    with torch.autocast('cuda', dtype=torch.bfloat16): out = model(x); loss = nn.functional.cross_entropy(out, y)
    assert out.shape == (len(y), 6) and torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
    assert not torch.equal(before, model.head[-1].weight)
    val = predict(model, splits()['val'].iloc[:8], meta)
    assert val.shape == (8, 6)
    meta.update(smoke_loss=float(loss.detach()), peak_gpu_bytes=torch.cuda.max_memory_allocated(), smoke='forward_backward_optimizer_eval_passed')
    dump(OUT / 'smoke' / f'{name}.json', meta)
    log(f'[smoke passed] {name}: params={meta["parameters"]:,}, peak={meta["peak_gpu_bytes"]/1e9:.2f} GB')


def worker(name, seed):
    run = OUT / 'runs' / name / str(seed); run.mkdir(parents=True, exist_ok=True)
    if (run / 'complete.json').exists(): log(f'[reuse] {name}/{seed} completed'); return
    seed_all(seed); torch.set_num_threads(6)
    model, meta = load_model(name); model.cuda()
    dump(run / 'model.json', meta); data = splits(); train = data['train']
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CONFIG['epochs'])
    history = []; best = -1.; start = 0
    if (run / 'last.pt').exists():
        state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        start = state['epoch']; history = state['history']; best = state['best']
        log(f'[resume] {name}/{seed} next_epoch={start+1}')
    counts = train.label.value_counts()
    weights = torch.tensor([1. / counts[y] for y in train.label], dtype=torch.double)
    for epoch in range(start, CONFIG['epochs']):
        seed_all(seed + 1000 * epoch)
        generator = torch.Generator().manual_seed(seed + 1000 * epoch)
        sampler = WeightedRandomSampler(weights, CONFIG['samples_per_epoch'], replacement=True, generator=generator)
        loader = DataLoader(Images(train, transform(meta, True)), sampler=sampler, batch_size=CONFIG['micro_batch'],
                            num_workers=CONFIG['workers'], pin_memory=True, generator=torch.Generator().manual_seed(seed + epoch))
        model.train(); opt.zero_grad(set_to_none=True); total_loss = 0.; seen = 0; accumulated = 0; t0 = time.time()
        for step, (x, y) in enumerate(loader):
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x); loss = nn.functional.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); accumulated += len(y); total_loss += float(loss.detach()); seen += len(y)
            if accumulated >= CONFIG['effective_batch'] or step + 1 == len(loader):
                for p in model.parameters():
                    if p.grad is not None: p.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True); accumulated = 0
            if (step + 1) % 100 == 0: log(f'[batch] {name}/{seed} epoch={epoch+1} targets={seen}/{CONFIG["samples_per_epoch"]} loss={total_loss/seen:.4f}')
        val = predict(model, data['val'], meta); score = metrics(data['val'].label.to_numpy(), val)['macro_f1']
        sched.step(); history.append(dict(epoch=epoch+1, train_loss=total_loss/seen, val_macro_f1=score, seconds=time.time()-t0))
        if score > best:
            best = score; save_pt(run / 'best.pt', {'model': model.state_dict(), 'epoch': epoch+1, 'val_macro_f1': best})
        save_pt(run / 'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch+1, history=history, best=best))
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        dump(run / 'progress.json', dict(status='training', model=name, seed=seed, epoch=epoch+1, epochs=CONFIG['epochs'], best_val_macro_f1=best))
        log(f'[epoch] {name}/{seed} {epoch+1}/{CONFIG["epochs"]} val_f1={score:.4f} best={best:.4f} seconds={history[-1]["seconds"]:.1f}')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model']); del checkpoint
    outputs = []
    for split, df in data.items():
        if split in ('train', 'val'): continue
        p = run / f'{split}_outputs.npz'
        if p.exists():
            with np.load(p) as z:
                assert z['sample_ids'].tolist() == df.sample_id.tolist(); logits = z['logits']
        else:
            log(f'[predict] {name}/{seed} {split} targets={len(df)}')
            logits = predict(model, df, meta)
            tmp = p.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, sample_ids=df.sample_id.to_numpy(dtype=str), labels=df.label.to_numpy(), logits=logits)
            tmp.replace(p)
        outputs.append(dict(model=name, seed=seed, pool=split, **metrics(df.label.to_numpy(), logits)))
    pd.DataFrame(outputs).to_csv(run / 'overall.csv', index=False)
    dump(run / 'complete.json', dict(status='complete', model=name, seed=seed, best_val_macro_f1=best, epochs=CONFIG['epochs']))
    log(f'[complete] {name}/{seed}')


def cm_metrics(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1); true = cm.sum(-1); predicted = cm.sum(-2)
    pr = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted>0)
    re = np.divide(tp, true, out=np.zeros_like(tp, dtype=float), where=true>0)
    f = np.divide(2*tp, true+predicted, out=np.zeros_like(tp, dtype=float), where=true+predicted>0)
    return dict(accuracy=tp.sum(-1)/true.sum(-1), macro_precision=pr.mean(-1), macro_recall=re.mean(-1), macro_f1=f.mean(-1))


def analyze():
    completed = [(m,s) for m in CONFIG['models'] for s in CONFIG['seeds'] if (OUT/'runs'/m/str(s)/'complete.json').exists()]
    if len(completed) != 15: raise RuntimeError(f'Only {len(completed)}/15 completed; formal comparison withheld')
    tables = OUT / 'tables'; tables.mkdir(exist_ok=True)
    data = splits(); overall=[]; layers=[]; gaps=[]; curves=[]; confusions=[]; breadth=[]
    parameters=[]; portable = ROOT / CONFIG['portable_states']; audits=0
    for m in CONFIG['models']:
        parameters.append(json.loads((OUT/'runs'/m/str(CONFIG['seeds'][0])/'model.json').read_text()))
    for pool, df in data.items():
        if pool in ('train', 'val'): continue
        pred=[]
        for m,s in completed:
            with np.load(OUT/'runs'/m/str(s)/f'{pool}_outputs.npz') as z:
                assert z['sample_ids'].tolist() == df.sample_id.tolist()
                assert np.array_equal(z['labels'],df.label.to_numpy()); logits=z['logits']; pred.append(logits.argmax(-1))
            overall.append(dict(model=m,seed=s,pool=pool,**metrics(df.label.to_numpy(),logits)))
            cm=confusion_matrix(df.label,logits.argmax(-1),labels=range(6))
            confusions.extend(dict(pool=pool,model=m,seed=s,true_class=CLASSES[a],predicted_class=CLASSES[b],count=int(cm[a,b])) for a in range(6) for b in range(6))
            audits+=1
        if pool=='test':continue
        pred=np.array(pred)
        with np.load(portable/f'{pool}_predictions.npz') as z:
            # Deliberately read only metadata, never z['predictions'] or z['logits'].
            assert z['sample_ids'].tolist()==df.sample_id.tolist()
            orders={source:z[source+'_full_orders'].copy() for source in ('local','online')}
        states=json.loads((portable/f'{pool}_states.json').read_text())
        for source,order in orders.items():
            for ri,(m,s) in enumerate(completed):
                endpoint=[]
                for t in range(5):
                    idx=order[:,t].reshape(-1); mm=metrics(df.label.to_numpy()[idx],np.eye(6)[pred[ri,idx]])
                    layers.append(dict(pool=pool,source=source,model=m,seed=s,tier=t+1,support=len(idx),**mm));endpoint.append(mm['macro_f1'])
                gaps.append(dict(pool=pool,source=source,model=m,seed=s,q1_macro_f1=endpoint[0],q5_macro_f1=endpoint[-1],q1_minus_q5=endpoint[0]-endpoint[-1]))
            # Freeze an eligible state universe per pool/source; descriptive breadth only.
            universe={}
            for i,r in enumerate(df.itertuples()):
                for field,value in states[source][r.sample_id].items():
                    for v in value if isinstance(value,list) else [value]:
                        if v in ('unknown','未知',None):continue
                        key=(r.class_name,field,str(v));universe.setdefault(key,[]).append(i)
            universe={k:np.array(v) for k,v in universe.items() if len(v)>=5}
            for ri,(m,s) in enumerate(completed):
                wrong=pred[ri]!=df.label.to_numpy()
                covered=sum(bool(wrong[idx].any()) for idx in universe.values())
                breadth.append(dict(pool=pool,source=source,model=m,seed=s,errors=int(wrong.sum()),eligible_states=len(universe),failure_states=covered,failure_state_fraction=covered/len(universe)))
            rng=np.random.default_rng(2026100910+ord(pool)+(source=='online'))
            repeats=CONFIG['mixture_repeats'];per_class=CONFIG['mixture_total']//6
            low=np.array([[rng.permutation(order[c,0])[:per_class] for c in range(6)] for _ in range(repeats)])
            high=np.array([[rng.permutation(order[c,4])[:per_class] for c in range(6)] for _ in range(repeats)])
            for n_high in CONFIG['mixture_high_per_class']:
                selection=np.concatenate([low[:,:,:per_class-n_high], high[:,:,:n_high]],-1)
                # One selection is shared by every architecture and seed.
                for ri,(m,s) in enumerate(completed):
                    cm=np.zeros((repeats,6,6),int)
                    for c in range(6):
                        chosen=pred[ri,selection[:,c]]
                        for p in range(6):cm[:,c,p]=(chosen==p).sum(-1)
                    mm=cm_metrics(cm)
                    # Independently check one replicate against sklearn at each point.
                    idx=selection[0].reshape(-1)
                    ref=metrics(df.label.to_numpy()[idx],np.eye(6)[pred[ri,idx]])
                    assert all(np.isclose(mm[k][0],ref[k],atol=1e-12) for k in mm);audits+=1
                    for metric,v in mm.items():
                        curves.append(dict(pool=pool,source=source,model=m,seed=s,high_risk_fraction=n_high/per_class,metric=metric,
                            mean=float(v.mean()),sampling_ci_low=float(np.quantile(v,.025)),sampling_ci_high=float(np.quantile(v,.975))))
    for name,records in [('overall_runs',overall),('risk_layers_runs',layers),('risk_gaps_runs',gaps),('mixture_curves_runs',curves),('confusion_runs',confusions),('failure_state_breadth_runs',breadth)]:
        pd.DataFrame(records).to_csv(tables/f'{name}.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('loading_audit',)} for r in parameters]).to_csv(tables/'model_parameters.csv',index=False)
    summ=pd.DataFrame(overall).groupby(['pool','model']).agg({k:['mean','std'] for k in ('accuracy','macro_precision','macro_recall','macro_f1')})
    summ.columns=['_'.join(c) for c in summ.columns];summ=summ.reset_index()
    summ['rank_in_pool']=summ.groupby('pool').macro_f1_mean.rank(method='min',ascending=False).astype(int)
    summ.to_csv(tables/'overall_summary.csv',index=False)
    text=['# 五组近期模型独立比较结果','', '三种子全量微调；仅本五组之间比较。Macro-F1等为0—1数值。',
          'MobileCLIP2输入256，其余224；不同预训练来源和规模意味着这是统一适配预算下的实际比较。','',
          '## 整体结果','',summ.to_markdown(index=False),'', '## 风险分层','',
          pd.DataFrame(gaps).groupby(['pool','source','model'])[['q1_macro_f1','q5_macro_f1','q1_minus_q5']].mean().reset_index().to_markdown(index=False),'',
          'Q1−Q5正值表示高风险层测得F1较低；不能由整体最高F1推出所有条件均最好。',
          '失败状态广度为自动标签条件下的描述性指标，受错误数和支持量影响。混合曲线区间反映重复抽样，不代表新来源的总体置信区间。',
          '各类指标、混淆矩阵、逐种子结果及21点风险混合曲线见tables。']
    (OUT/'results.md').write_text('\n'.join(text)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_recent_five_models.py')],check=True,cwd=ROOT)
    dump(OUT/'analysis_audit.json',dict(status='passed',completed_runs=len(completed),metric_cross_checks=audits,old_classifier_predictions_used=False))
    log('[analysis complete] CSV tables and results.md written')


def launch():
    path=OUT/'background.pid'
    if path.exists():
        pid=int(path.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'Background process {pid} is already running')
    with (OUT/'background.log').open('a') as f:
        p=subprocess.Popen([sys.executable,str(Path(__file__)),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    path.write_text(str(p.pid)+'\n');log(f'[launched] pid={p.pid} log={OUT}/background.log')


def run():
    prepare(); logs=OUT/'logs';logs.mkdir(exist_ok=True); failures=[]
    dump(OUT/'progress.json',dict(status='preflight',pid=os.getpid()))
    # Each smoke runs in a subprocess to release all GPU and import state.
    for name in CONFIG['models']:
        log(f'[preflight] {name}')
        with (logs/f'smoke_{name}.log').open('a') as f:
            result=subprocess.run([sys.executable,str(Path(__file__)),'smoke','--model',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(dict(stage='smoke',model=name,returncode=result.returncode))
            log(f'[preflight FAILED] {name}; see {logs}/smoke_{name}.log')
        else:log(f'[preflight PASSED] {name}')
    if failures:
        dump(OUT/'progress.json',dict(status='preflight_failed',failures=failures,pid=os.getpid()))
        raise RuntimeError('Preflight failed; formal sweep withheld until resolved')
    for i,(name,seed) in enumerate((m,s) for m in CONFIG['models'] for s in CONFIG['seeds']):
        dump(OUT/'progress.json',dict(status='running',pid=os.getpid(),model=name,seed=seed,task=i+1,total_tasks=15,failures=failures))
        log(f'[task {i+1}/15] {name}/{seed}')
        with (logs/f'{name}_{seed}.log').open('a') as f:
            proc=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--model',name,'--seed',str(seed)],cwd=ROOT,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:
                f.write(line);f.flush();print(line,end='',flush=True)
            returncode=proc.wait()
        if returncode:failures.append(dict(stage='worker',model=name,seed=seed,returncode=returncode))
        log(f'[task exit] {name}/{seed} returncode={returncode}')
    if failures:
        dump(OUT/'progress.json',dict(status='incomplete',failures=failures,pid=os.getpid()));return
    analyze();dump(OUT/'progress.json',dict(status='complete',completed_runs=15,pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','smoke','worker','launch','run','analyze'])
    parser.add_argument('--model',choices=CONFIG['models']);parser.add_argument('--seed',type=int,default=CONFIG['seeds'][0])
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    try:
        if args.mode=='smoke':smoke(args.model)
        elif args.mode=='worker':worker(args.model,args.seed)
        else:globals()[args.mode]()
    except Exception:
        if args.mode=='run':
            previous=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
            if previous.get('status')!='preflight_failed':dump(OUT/'progress.json',dict(status='failed',previous=previous,pid=os.getpid()))
        traceback.print_exc();sys.exit(1)

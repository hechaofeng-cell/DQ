#!/usr/bin/env python3
"""Run the frozen latest-five classifiers on the 20 cat pilot targets.

This produces the requested Q1--Q5 Cat-F1 line chart. It uses the already
trained checkpoints and does not retrain any model.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cat_five_major_equal_20261017'; OUT.mkdir(exist_ok=True)
RUNS=ROOT/'artifacts/recent_five_models_20261009/runs'
MODELS=['yolo26s_cls','pe_core_b16','siglip2_base','mobileclip2_s0','aimv2_large']
SEEDS=[20260916,20260917,20260918]
DISPLAY={'yolo26s_cls':'YOLO26s','pe_core_b16':'PE-Core-B16','siglip2_base':'SigLIP2-Base','mobileclip2_s0':'MobileCLIP2-S0','aimv2_large':'AIMv2-Large'}

spec=importlib.util.spec_from_file_location('recent_five_runtime',ROOT/'scripts/run_recent_five_models.py')
runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)

def main():
    manifest=pd.read_csv(ROOT/'artifacts/cat_human_vlm_v5_pilot_20261017/manifest.csv',encoding='utf-8-sig')
    scores=pd.read_csv(OUT/'scores.csv',encoding='utf-8-sig')
    qmap=dict(zip(scores.sample_id.astype(str),scores.Q.astype(int)))
    data=manifest.assign(label=1)
    rows=[]; prediction_rows=[]
    for name in MODELS:
        for seed in SEEDS:
            run=RUNS/name/str(seed); ckpt=run/'best.pt'
            model,meta=runtime.load_model(name); model.cuda()
            state=torch.load(ckpt,map_location='cpu',weights_only=False); model.load_state_dict(state['model']); del state
            logits=runtime.predict(model,data,meta); pred=logits.argmax(-1)
            prediction_rows.extend({'model':name,'display_name':DISPLAY[name],'seed':seed,'sample_id':sid,'predicted_class':int(cls)} for sid,cls in zip(manifest.sample_id.astype(str),pred))
            pred_map=dict(zip(manifest.sample_id.astype(str),pred))
            for q in range(1,6):
                ids=[s for s,t in qmap.items() if t==q]
                arr=np.array([pred_map[s]==1 for s in ids],dtype=bool); tp=int(arr.sum()); fp=int(len(arr)-tp); fn=0
                precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
                rows.append({'model':name,'display_name':DISPLAY[name],'seed':seed,'Q':q,'cat_precision':precision,'cat_recall':recall,'cat_f1':f1,'n':len(ids)})
            del model; torch.cuda.empty_cache()
    runs=pd.DataFrame(rows); summary=(runs.groupby(['model','display_name','Q'])[['cat_precision','cat_recall','cat_f1']].agg(['mean','std']).reset_index())
    summary.columns=['model','display_name','Q']+[f'{a}_{b}' for a,b in summary.columns.tolist()[3:]]
    runs.to_csv(OUT/'latest_five_cat_qtier_f1_runs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(prediction_rows).to_csv(OUT/'latest_five_cat_predictions.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT/'latest_five_cat_qtier_f1_summary.csv',index=False,encoding='utf-8-sig')
    font=ROOT/'scripts/assets/NotoSansSC-Regular.ttf'
    if font.exists(): font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    fig,ax=plt.subplots(figsize=(11,6),dpi=180)
    colors=['#1976D2','#D32F2F','#388E3C','#7B1FA2','#F57C00']
    for color,(model,g) in zip(colors,summary.groupby('model',sort=False)):
        g=g.set_index('Q').reindex(range(1,6)); label=g.display_name.iloc[0]
        ax.plot(range(1,6),100*g.cat_f1_mean,marker='o',lw=2.5,color=color,label=label)
        sd=g.cat_f1_std.fillna(0); ax.fill_between(range(1,6),100*(g.cat_f1_mean-sd),100*(g.cat_f1_mean+sd),color=color,alpha=.10)
    ax.set_xticks(range(1,6),['Q1\n易','Q2','Q3','Q4','Q5\n难']); ax.set_ylim(0,105); ax.set_xlabel('五大类等权总分风险层'); ax.set_ylabel('Cat-F1（%）'); ax.set_title('最新五模型在猫类 Q1–Q5 分层上的 Cat-F1',fontweight='bold'); ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'最新五模型猫类Q1-Q5_F1折线图.png',bbox_inches='tight'); fig.savefig(OUT/'最新五模型猫类Q1-Q5_F1折线图.svg',bbox_inches='tight'); plt.close(fig)
    print(summary.to_string(index=False))

if __name__=='__main__': main()

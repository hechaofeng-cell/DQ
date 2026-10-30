#!/usr/bin/env python3
"""Compute dog Q1--Q5 five-class Macro-F1 on the frozen dog-style strata."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/cross_category_balanced_difficulty_20260928'
OUT=BASE/'figures'
DATA=ROOT/'data/openimages_v7_dog_gap_v1'
PRED=ROOT/'artifacts/fixed_train_visual_test_adequacy_20260920/predictions'
MODELS=('resnet18','resnet50','mobilenet_v3_small','mobilenet_v3_large','densenet121','efficientnet_b0','convnext_tiny','swin_t')
DISPLAY={'resnet18':'ResNet-18','resnet50':'ResNet-50','mobilenet_v3_small':'MobileNetV3-S','mobilenet_v3_large':'MobileNetV3-L','densenet121':'DenseNet-121','efficientnet_b0':'EfficientNet-B0','convnext_tiny':'ConvNeXt-T','swin_t':'Swin-T'}
SEEDS=(20260916,20260917,20260918)

def main():
 pool=pd.concat([pd.read_csv(DATA/'manifests/test.csv',dtype={'sample_id':str}),pd.read_csv(DATA/'manifests/dog_difficulty_test.csv',dtype={'sample_id':str})],ignore_index=True)
 risk=pd.read_csv(BASE/'tables/dog_difficulty_scores.csv',dtype={'sample_id':str})
 assert risk.stratum.value_counts().sort_index().to_dict()=={'Q1':40,'Q2':40,'Q3':40,'Q4':40,'Q5':40}
 neg=pd.concat([pool[pool.class_name==c].sort_values('sample_id').iloc[:40] for c in ('cat','horse','sheep','person')],ignore_index=True)
 qframes={}
 for q in range(1,6):
  ids=set(risk.loc[risk.stratum==f'Q{q}','sample_id']); cats=pool[pool.sample_id.isin(ids)]
  qframes[q]=pd.concat([cats,neg],ignore_index=True)
 rows=[]
 for m in MODELS:
  for s in SEEDS:
   d=pd.read_csv(PRED/m/str(s)/'candidate_predictions.csv',dtype={'sample_id':str}).set_index('sample_id')
   pred=d.predicted_label.astype(int).to_dict()
   for q,f in qframes.items():
    y=f.label.to_numpy(int); p=np.array([pred[x] for x in f.sample_id],int)
    pr,re,ff,_=precision_recall_fscore_support(y,p,labels=range(5),zero_division=0)
    rows.append({'model':m,'display_model':DISPLAY[m],'seed':s,'Q':q,'macro_f1':ff.mean(),'cat_f1':ff[1],'cat_recall':re[1]})
 run=pd.DataFrame(rows);run.to_csv(BASE/'tables/dog_q1_q5_macro_f1_runs.csv',index=False)
 summary=run.groupby(['model','display_model','Q'],as_index=False).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),cat_f1_mean=('cat_f1','mean'),cat_recall_mean=('cat_recall','mean'))
 summary.to_csv(BASE/'tables/dog_q1_q5_macro_f1_summary.csv',index=False)
 eff=[]
 for m in MODELS:
  f=summary[summary.model==m].sort_values('Q');y=f.macro_f1_mean.to_numpy();eff.append({'model':m,'display_model':DISPLAY[m],'Q1_macro_f1':y[0],'Q5_macro_f1':y[-1],'Q5_minus_Q1':y[-1]-y[0]})
 pd.DataFrame(eff).to_csv(BASE/'tables/dog_q1_q5_macro_f1_effect.csv',index=False)
 fp='/home/hcf/.local/share/fonts/NotoSansCJKsc-Regular.otf';font_manager.fontManager.addfont(fp);plt.rcParams['font.sans-serif']=['Noto Sans CJK SC','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(figsize=(12,7)); colors=plt.cm.tab10(np.linspace(0,.875,len(MODELS)))
 for c,m in zip(colors,MODELS):
  f=summary[summary.model==m].sort_values('Q');ax.errorbar(f.Q,100*f.macro_f1_mean,yerr=100*f.macro_f1_std.fillna(0),marker='o',linewidth=1.9,capsize=3,color=c,label=DISPLAY[m])
 ax.set_xticks(range(1,6),['Q1\n易','Q2','Q3','Q4','Q5\n难']);ax.set_xlabel('狗类有序风险层（五维风险分数排序）');ax.set_ylabel('五分类 Macro-F1 (%)');ax.set_title('狗类风险分层：八个模型 Q1–Q5 Macro-F1',fontsize=15,weight='bold');ax.set_ylim(0,100);ax.grid(alpha=.22);ax.legend(frameon=False,fontsize=9,ncol=2);fig.tight_layout();fig.savefig(OUT/'dog_q1_q5_macro_f1.png',dpi=220);fig.savefig(OUT/'dog_q1_q5_macro_f1.svg');plt.close(fig)
 print(pd.DataFrame(eff).to_string(index=False))
if __name__=='__main__': main()

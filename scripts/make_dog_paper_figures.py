#!/usr/bin/env python3
import csv, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/dog_v3_1_figures';OUT.mkdir(parents=True,exist_ok=True)
FONT_PATH=ROOT/'scripts/assets/NotoSansSC-Regular.ttf'
if FONT_PATH.exists():
    fontManager.addfont(str(FONT_PATH))
FONT=FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
if FONT is not None:
    plt.rcParams['font.family']=FONT.get_name()
plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':180})

def rows(path): return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
def save(fig,name): fig.tight_layout();fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    base=rows(ROOT/'artifacts/dog500_features_v3_1_20260913/distribution/feature_state_distribution.csv')
    final=rows(ROOT/'artifacts/dog_v3_1_combined_supplement_20260913_final/feature_state_distribution_596.csv')
    fig,ax=plt.subplots(figsize=(7,4.2));labels=['初始500张','补图后596张'];coverage=[78.64,84.17];attainment=[63.41,74.80]
    x=np.arange(2);w=.34;b1=ax.bar(x-w/2,coverage,w,label='状态覆盖指数',color='#167c80');b2=ax.bar(x+w/2,attainment,w,label='目标达成率',color='#e87516')
    ax.set_ylim(0,100);ax.set_ylabel('百分比（%）');ax.set_xticks(x,labels);ax.set_title('补图前后覆盖指标对比');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25)
    for bars in (b1,b2):
      for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.2f}%',ha='center',fontsize=9)
    save(fig,'fig1_coverage_before_after.png')
    fig,ax=plt.subplots(figsize=(6.8,4));x=np.arange(3);w=.34
    b1=ax.bar(x-w/2,[9,10,26],w,label='初始500张',color='#1a7f86');b2=ax.bar(x+w/2,[7,10,16],w,label='补图后596张',color='#e87516')
    ax.set_xticks(x,['完全缺失','稀疏（<10）','较弱（10–29）']);ax.set_ylabel('状态数');ax.set_title('低频状态构成变化');ax.legend(frameon=False);ax.grid(axis='y',alpha=.25);save(fig,'fig2_gap_composition.png')
    rr=rows(ROOT/'artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv');c=Counter(r['scene'] for r in rr);names=['indoor_home','outdoor_grass','outdoor_other','indoor_other','outdoor_street','outdoor_water','vehicle','outdoor_beach','outdoor_snow','outdoor_park','unknown'];zh=['室内-家居','室外-草地','室外-其他','室内-其他','室外-街道','室外-水域','车内','室外-沙滩','室外-雪地','室外-公园','未知'];fig,ax=plt.subplots(figsize=(7.2,4.4));v=[c[n] for n in names];order=np.argsort(v);ax.barh(np.array(zh)[order],np.array(v)[order],color='#167c80');ax.set_xlabel('图片数量');ax.set_title('596张清洁集场景分布');ax.grid(axis='x',alpha=.25);save(fig,'fig3_scene_distribution.png')
    unknown=rows(ROOT/'artifacts/dog_v3_1_combined_supplement_20260913_final/unknown_summary_596.csv');unknown.sort(key=lambda r: float(r.get('unknown_rate',0)), reverse=True);unknown=unknown[:10];fig,ax=plt.subplots(figsize=(7.2,4.8));labels=[r['feature_name_zh'] for r in unknown][::-1];vals=[float(r['unknown_rate'])*100 for r in unknown][::-1];ax.barh(labels,vals,color='#d95f02');ax.set_xlabel('unknown率（%）');ax.set_title('unknown率最高的特征（清洁596张）');ax.set_xlim(0,100);ax.grid(axis='x',alpha=.25);save(fig,'fig4_unknown_rate.png')
    fig=plt.figure(figsize=(10,3.0));fig.text(.5,.76,r'$C(D)=\frac{1}{|\mathcal{S}|}\sum_{(j,s)\in\mathcal{S}}\min\left(\frac{n_{j,s}}{30},1\right)$',ha='center',va='center',fontsize=22);fig.text(.5,.47,r'$A(D)=\frac{1}{|\mathcal{S}|}\sum_{(j,s)\in\mathcal{S}}\mathbf{1}[n_{j,s}\geq t_{j,s}]$',ha='center',va='center',fontsize=22);fig.text(.5,.14,'C(D)为覆盖指数，A(D)为达到预设数量目标的状态比例',ha='center',fontsize=11);save(fig,'fig5_coverage_formula.png')
    print(OUT)
if __name__=='__main__':main()

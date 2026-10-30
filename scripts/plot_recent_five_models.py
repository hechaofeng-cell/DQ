#!/usr/bin/env python3
"""Real recent-five result tables only; editable SVG and PNG previews."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/recent_five_models_20261009'
CFG=json.loads((ROOT/'configs/recent_five_models_20261009.json').read_text())
NAMES={'yolo26s_cls':'YOLO26-S-cls','pe_core_b16':'PE-B16','siglip2_base':'SigLIP2-B',
       'mobileclip2_s0':'MobileCLIP2-S0','aimv2_large':'AIMv2-L'}
# Category colors plus styles; no red/green identity dependence.
COLORS=['#0F4D92','#42949E','#9A4D8E','#767676','#B98340']
STYLES=['-','--','-.',':','-']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,'axes.linewidth':.8,
    'xtick.direction':'out','ytick.direction':'out','axes.spines.top':False,'axes.spines.right':False})
FIG=OUT/'figures'


def save(fig,name):
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    validator=Path('/home/hcf/.codex/skills/research-figure-composer/scripts/validate_svg.py')
    result=subprocess.run([sys.executable,str(validator),str(FIG/f'{name}.svg')],capture_output=True,text=True,check=True)
    (FIG/f'{name}_svg_audit.txt').write_text(result.stdout)


def main():
    FIG.mkdir(exist_ok=True)
    overall=pd.read_csv(OUT/'tables/overall_summary.csv')
    fig,axes=plt.subplots(1,4,figsize=(13,4),constrained_layout=True)
    for ax,pool in zip(axes,['test','B','C','D']):
        data=overall[overall.pool==pool].set_index('model').loc[CFG['models']]
        x=np.arange(5);mean=100*data.macro_f1_mean.to_numpy();sd=100*data.macro_f1_std.to_numpy()
        ax.errorbar(x,mean,yerr=sd,fmt='o',color='#484878',capsize=3)
        for i,v in enumerate(mean):ax.annotate(f'{v:.1f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,[NAMES[m] for m in CFG['models']],rotation=45,ha='right')
        ax.set_title('Base test' if pool=='test' else f'Pool {pool}')
        ax.set_ylabel('Macro-F1 (%)' if ax is axes[0] else '')
    fig.suptitle('Same adaptation budget; mean ± SD across three seeds',fontsize=11)
    save(fig,'overall_macro_f1')

    layers=pd.read_csv(OUT/'tables/risk_layers_runs.csv')
    fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    for row,source in enumerate(['local','online']):
        for col,pool in enumerate(['B','C','D']):
            ax=axes[row,col]
            for mi,model in enumerate(CFG['models']):
                df=layers[(layers.source==source)&(layers.pool==pool)&(layers.model==model)]
                agg=df.groupby('tier').macro_f1.agg(['mean','std']);x=agg.index.to_numpy()
                y=100*agg['mean'].to_numpy();sd=100*agg['std'].to_numpy()
                ax.plot(x,y,label=NAMES[model],color=COLORS[mi],ls=STYLES[mi],marker='o',ms=3)
                ax.fill_between(x,y-sd,y+sd,color=COLORS[mi],alpha=.08)
            ax.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5'])
            ax.set_title(f'{pool} / {source} labels');ax.set_ylabel('Macro-F1 (%)' if col==0 else '')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('Frozen class-balanced risk layers; shaded bands: seed SD',fontsize=11)
    save(fig,'risk_layer_f1')

    curves=pd.read_csv(OUT/'tables/mixture_curves_runs.csv')
    for metric in ['macro_f1','accuracy','macro_recall','macro_precision']:
        fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
        for row,source in enumerate(['local','online']):
            for col,pool in enumerate(['B','C','D']):
                ax=axes[row,col]
                for mi,model in enumerate(CFG['models']):
                    df=curves[(curves.metric==metric)&(curves.source==source)&(curves.pool==pool)&(curves.model==model)]
                    agg=df.groupby('high_risk_fraction')['mean'].agg(['mean','min','max'])
                    x=100*agg.index.to_numpy();y=100*agg['mean'].to_numpy()
                    ax.plot(x,y,label=NAMES[model],color=COLORS[mi],ls=STYLES[mi],marker='o',markevery=5,ms=3)
                    ax.fill_between(x,100*agg['min'].to_numpy(),100*agg['max'].to_numpy(),color=COLORS[mi],alpha=.08)
                ax.set_title(f'{pool} / {source} labels');ax.set_xlabel('High-risk fraction (%)')
                ax.set_ylabel(metric.replace('macro_','Macro-').replace('_',' ').title()+' (%)' if col==0 else '')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=8)
        fig.suptitle('Fixed 120 targets, class-balanced; 21 measured points; bands: seed range',fontsize=11)
        save(fig,'risk_mixture_'+metric)
    (FIG/'figure_notes.md').write_text('# 图表口径\n\n整体图为三种子均值±标准差；风险层图带为种子标准差。混合图每条线包含21个真实计算点，每点1000次共享抽样的均值，带为三个种子点均值的范围；逐种子抽样区间见CSV。\n\n横轴是高风险样本占比或冻结风险层，不把它改称覆盖率。纵轴显示百分比并自动缩放，点图明确数值；未修改、平滑或放大实验差值。SVG文本可编辑，PNG仅预览。\n')


if __name__=='__main__':main()

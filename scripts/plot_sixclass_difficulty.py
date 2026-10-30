#!/usr/bin/env python3
"""Editable figures from frozen six-class difficulty supplement tables."""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/sixclass_difficulty_supplement_20261004'
OUT=BASE/'figures'
NAMES={'resnet18':'ResNet-18','resnet50':'ResNet-50',
       'mobilenet_v3_small':'MobileNetV3-Small','mobilenet_v3_large':'MobileNetV3-Large',
       'densenet121':'DenseNet-121','efficientnet_b0':'EfficientNet-B0',
       'convnext_tiny':'ConvNeXt-Tiny','swin_t':'Swin-T'}
COLORS=['#767676','#484878','#B4C0E4','#7884B4','#9A4D8E','#42949E','#0F4D92','#B64342']
BACK={'local':'Local VLM','online':'Online VLM'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.linewidth':.8,'axes.spines.top':False,
                     'axes.spines.right':False,'legend.frameon':False})

def curve(ax,x,y,color,label=None,style='-'):
    # Interpolation is only a guide; exactly five measured points remain visible.
    xx=np.linspace(min(x),max(x),101)
    ax.plot(xx,PchipInterpolator(x,y)(xx),color=color,label=label,linestyle=style,lw=1.4)
    ax.scatter(x,y,color=color,s=16,zorder=3)

def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=210,bbox_inches='tight')
    plt.close(fig)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(BASE/'tables/full_stratum_metrics.csv')
    full=full[full.unknown_mode=='midpoint']
    model=full.groupby(['source','model','tier'],as_index=False).macro_f1.mean()
    model.to_csv(OUT/'model_curve_input.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),sharey=True)
    for ax,source,letter in zip(axes,['local','online'],['a','b']):
        for (name,color) in zip(NAMES,COLORS):
            d=model[(model.source==source)&(model.model==name)].sort_values('tier')
            curve(ax,d.tier,d.macro_f1*100,color,NAMES[name])
        ax.set(title=f'{letter}  {BACK[source]} risk strata',xlabel='Within-class risk stratum',
               xticks=range(1,6),xticklabels=['Q1','Q2','Q3','Q4','Q5'],ylim=(72,101))
    axes[0].set_ylabel('Macro-F1 (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.subplots_adjust(bottom=.25,wspace=.10)
    save(fig,'full_stratum_model_f1')

    batch=pd.read_csv(BASE/'tables/selected_batch_metrics.csv').groupby(['source','tier'],as_index=False)[['macro_f1','accuracy']].mean()
    cum=pd.read_csv(BASE/'tables/cumulative_metrics.csv').groupby(['source','tier'],as_index=False)[['macro_f1','macro_frequency_coverage','accuracy','macro_precision','macro_recall']].mean()
    rnd=pd.read_csv(BASE/'tables/matched_random_summary.csv').groupby(['source','tier'],as_index=False)[['macro_f1_mean','macro_frequency_coverage_mean']].mean()
    cum.merge(rnd,on=['source','tier']).to_csv(OUT/'cumulative_curve_input.csv',index=False)
    batch.to_csv(OUT/'batch_curve_input.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.6))
    for source,color in [('local','#0F4D92'),('online','#9A4D8E')]:
        d=batch[batch.source==source];c=cum[cum.source==source];r=rnd[rnd.source==source]
        curve(axes[0],d.tier,d.macro_f1*100,color,BACK[source])
        curve(axes[1],c.tier*120,c.macro_f1*100,color,f'{BACK[source]} Coverage')
        curve(axes[1],r.tier*120,r.macro_f1_mean*100,color,f'{BACK[source]} risk-matched random','--')
        curve(axes[2],c.tier*120,c.macro_frequency_coverage*100,color)
        curve(axes[2],r.tier*120,r.macro_frequency_coverage_mean*100,color,style='--')
    axes[0].set(title='a  New, disjoint batches',xlabel='Risk stratum',ylabel='Macro-F1 (%)',xticks=range(1,6),ylim=(65,99))
    axes[0].legend(fontsize=8)
    axes[1].set(title='b  Cumulative test sets',xlabel='Selected targets (six classes)',ylabel='Macro-F1 (%)',xticks=[120,240,360,480,600],ylim=(83,96))
    axes[2].set(title='c  Cumulative state evidence',xlabel='Selected targets (six classes)',ylabel='Frequency coverage (%)',xticks=[120,240,360,480,600],ylim=(15,73))
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.subplots_adjust(bottom=.29,wspace=.32)
    save(fig,'batch_cumulative_and_coverage')

    recalls=[f'{c}_recall' for c in ['dog','cat','horse','sheep','person','car']]
    grouped=full.groupby(['source','tier'])[recalls].mean()
    grouped.to_csv(OUT/'class_risk_recall_input.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),sharey=True)
    for ax,source,letter in zip(axes,['local','online'],['a','b']):
        matrix=grouped.loc[source].to_numpy().T*100
        im=ax.pcolormesh(np.arange(6)-.5,np.arange(7)-.5,matrix,
                         vmin=60,vmax=100,cmap='Blues',shading='flat',rasterized=False)
        ax.set_ylim(5.5,-.5)
        for i in range(6):
            for j in range(5):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>87 else '#272727')
        ax.set(title=f'{letter}  {BACK[source]} risk strata',xticks=range(5),xticklabels=['Q1','Q2','Q3','Q4','Q5'],
               yticks=range(6),yticklabels=['dog','cat','horse','sheep','person','car'],xlabel='Within-class risk stratum')
    fig.subplots_adjust(right=.88,wspace=.14)
    cax=fig.add_axes([.91,.16,.014,.68]);bar=fig.colorbar(im,cax=cax,label='Class recall (%)')
    bar.solids.set_rasterized(False)
    save(fig,'class_risk_recall')

    comp=pd.read_csv(BASE/'tables/matched_equal_error_comparisons.csv')
    comp.to_csv(OUT/'risk_matched_diagnostic_input.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.3,3.3))
    labels=[]
    for i,row in comp.iterrows():
        value=float(row.mean_delta_auc);lo=float(row.ci95_low);hi=float(row.ci95_high)
        color='#0F4D92' if row.source=='local' else '#9A4D8E'
        ax.errorbar(value*100,3-i,xerr=np.array([[value-lo],[hi-value]])*100,fmt='o',color=color,capsize=3,lw=1.3)
        labels.append(f"{BACK[row.source]} / {'Confusion breadth' if row.metric=='confusion' else 'Error-cluster breadth'}")
        ax.text(2.1,3-i,f'Holm p = {row.holm_four_p:.4f}',va='center',fontsize=8)
    ax.axvline(0,color='#767676',lw=.8,linestyle='--')
    ax.set(yticks=range(4),yticklabels=labels[::-1],xlabel='Coverage minus exact-risk random: normalized AUC (percentage points)',
           title='Equal-error diagnostic breadth after exact risk matching',xlim=(-1.5,3.4),ylim=(-.5,3.5))
    save(fig,'risk_matched_independent_diagnostics')
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    print(f'Created four editable figures and their input tables in {OUT}',flush=True)

if __name__=='__main__':run()

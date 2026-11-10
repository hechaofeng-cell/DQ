"""Editable, data-bound research figures; can run standalone with --out RUN_DIR."""
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS=['R','G','SR','KC','CS','AS']
COLORS={'R':'#8B929B','G':'#3D527D','SR':'#9483AF','KC':'#BCA572','CS':'#278D92','AS':'#D68B62'}
NAMES={'R':'R: random','G':'G: greedy','SR':'SR: rarity','KC':'KC: k-center',
       'CS':'CS: conditional','AS':'AS: always'}


def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans'],
        'svg.fonttype':'none','pdf.fonttype':42,'font.size':9,'axes.titlesize':11,
        'axes.labelsize':9,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.direction':'out','ytick.direction':'out','legend.frameon':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})


def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.14,1.07,letter,transform=ax.transAxes,fontsize=12,fontweight='bold')


def save(fig,folder,name):
    for extension in ['svg','png','pdf']:
        fig.savefig(folder/f'{name}.{extension}',dpi=220,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    root=ET.parse(folder/f'{name}.svg').getroot()
    assert root.attrib.get('viewBox')
    assert root.findall('.//{http://www.w3.org/2000/svg}text')
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')


def overview(out,folder):
    summary=pd.read_csv(out/'method_summary.csv').set_index('method')
    complete=pd.read_csv(out/'complete_selection_runs.csv')
    diagnostic=pd.read_csv(out/'external_diagnostic_summary.csv')
    diagnostic=diagnostic[diagnostic.scope=='all'].set_index('method')
    trace=pd.read_csv(out/'swap_trace.csv')
    fig,axes=plt.subplots(2,2,figsize=(10.8,8.2),gridspec_kw={'wspace':.43,'hspace':.62})
    ax=axes[0,0]
    order=['R','SR','G','KC','CS','AS']
    for y,method in enumerate(order):
        part=1000*complete[complete.method==method].warm_seconds.to_numpy()
        q=np.quantile(part,[.025,.975]);median=np.median(part)
        ax.plot(q,[y,y],color=COLORS[method],linewidth=2)
        ax.scatter(median,y,color=COLORS[method],s=46,zorder=3)
        ax.annotate(f'{median:.1f}',(median,y),xytext=(6,6),textcoords='offset points',fontsize=8)
    ax.set_yticks(range(6),[NAMES[m] for m in order]);ax.invert_yaxis()
    ax.set_xscale('log');ax.set_xlim(.9,1000)
    ax.set_xlabel('Four-class selection time (ms; log scale)')
    panel(ax,'a','Selection cost at the primary setting')
    ax=axes[0,1];x=np.arange(6);width=.36
    ax.bar(x-width/2,[summary.loc[m,'mean_occurrence_pct'] for m in METHODS],width,
           color='#829AB5',label='Occurrence')
    ax.bar(x+width/2,[summary.loc[m,'mean_support_pct'] for m in METHODS],width,
           color='#8BBDBB',label='Support completion')
    ax.set_xticks(x,METHODS);ax.set_ylabel('Coverage (%)');ax.set_ylim(0,105)
    for i,m in enumerate(METHODS):
        ax.text(i,98,f"{summary.loc[m,'joint_pass_fraction']:.0%}",ha='center',fontsize=8,color='#4D4D4D')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,ncol=2)
    panel(ax,'b','Coverage; labels show joint pass rate')
    ax=axes[1,0];x=np.arange(6);bottom=np.zeros(6)
    for metric,color,label in [('confirmed','#599D9B','Confirmed'),('not_confirmed','#BBC2CA','Not confirmed'),
                                ('insufficient','#E5E8EB','Insufficient support')]:
        values=np.array([diagnostic.loc[m,metric] for m in METHODS])
        ax.bar(x,values,bottom=bottom,width=.65,color=color,label=label)
        bottom+=values
    for i,m in enumerate(METHODS):
        ax.text(i,diagnostic.loc[m,'confirmed']/2,f"{diagnostic.loc[m,'confirmed']:.1f}",
                ha='center',va='center',color='white',fontweight='bold',fontsize=8)
    ax.set_xticks(x,METHODS);ax.set_ylim(0,65);ax.set_ylabel('Condition instances (mean per run)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,ncol=1)
    panel(ax,'c','Historical reference: diagnostic outcomes')
    ax=axes[1,1]
    for method,style in [('AS','-'),('CS','-')]:
        part=trace[(trace.class_name=='sheep')&(trace.seed==2026110500)&(trace.method==method)].sort_values('step')
        steps=np.r_[0,part.step.to_numpy()]
        values=100*np.r_[part.support_before.iloc[0],part.support_after.to_numpy()]
        ax.plot(steps,values,color=COLORS[method],label=NAMES[method],lw=2 if method=='CS' else 1.6)
        if method=='CS':
            ax.scatter(steps[-1],values[-1],color=COLORS[method],s=45,zorder=4)
    threshold=pd.read_csv(out/'thresholds.csv').set_index('class_name').loc['sheep','support_completion']*100
    ax.axhline(threshold,color='#7F8790',ls='--',lw=.9,label='Support threshold')
    ax.set_xlabel('Accepted swaps');ax.set_ylabel('Support completion (%)')
    ax.legend(fontsize=8,loc='lower right')
    panel(ax,'d','One paired run: sheep, first fixed seed')
    fig.text(.08,-.015,'R/CS/AS: 50 paired selection seeds; G/SR/KC: one deterministic set. '
             'All final sets contain 1,200 targets across six classes.\n'
             'Timing excludes common input loading and calibration. Diagnostics use a previously analyzed reference pool.',
             fontsize=8,color='#59616B',va='top')
    save(fig,folder,'coverage_cost_diagnostics')
    summary.reset_index().to_csv(folder/'coverage_cost_source.csv',index=False)
    diagnostic.reset_index().to_csv(folder/'diagnostic_source.csv',index=False)
    complete.to_csv(folder/'timing_source.csv',index=False)
    trace[(trace.class_name=='sheep')&(trace.seed==2026110500)].to_csv(folder/'trajectory_source.csv',index=False)


def metrics_grid(out,folder):
    source=pd.read_csv(out/'model_metrics_selection_seed_summary.csv')
    source=source[source.class_name.isin(['horse','sheep','person','car'])&(source.Q>0)]
    metrics=['precision','recall','f1','one_vs_rest_accuracy']
    seed_mean=source.groupby(['method','selection_seed','Q'],as_index=False)[metrics].mean()
    seed_mean.to_csv(folder/'four_metrics_source.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(8.4,8.0),gridspec_kw={'wspace':.27,'hspace':.30})
    styles={'R':':','G':'--','SR':'-.','KC':(0,(4,2,1,2)),'CS':'-','AS':'--'}
    for ax,metric,title,letter in zip(axes.flat,metrics,['Precision','Recall','F1','One-vs-rest accuracy'],'abcd'):
        for method in METHODS:
            p=seed_mean[seed_mean.method==method].groupby('Q')[metric].agg(['mean','std']).reindex(range(1,6))
            x=np.arange(1,6);y=100*p['mean'].to_numpy();sd=100*p['std'].fillna(0).to_numpy()
            ax.plot(x,y,color=COLORS[method],ls=styles[method],lw=2 if method=='CS' else 1.25,
                    marker='o' if method=='CS' else None,markersize=3.5,label=NAMES[method])
            if method in ['R','CS','AS']:
                ax.fill_between(x,y-sd,y+sd,color=COLORS[method],alpha=.08,linewidth=0)
        ax.set_xticks(range(1,6),[f'Q{i}' for i in range(1,6)])
        ax.set_ylabel(f'{title} (%)');ax.set_xlim(.8,5.2)
        panel(ax,letter,title)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,1.03),ncol=3,fontsize=8.5)
    fig.text(.07,-.015,'Equal mean over 4 optimized classes and 5 fixed models; prediction seeds averaged first.\n'
             'Bands: SD across selection seeds, conditional on this pool. Accuracy is OvR, not six-class top-1.',
             fontsize=8,color='#59616B',va='top')
    save(fig,folder,'four_metrics_grid')


def sensitivity(out,folder):
    s=pd.read_csv(out/'sensitivity_summary.csv')
    main=pd.read_csv(out/'method_summary.csv')
    missing=pd.read_csv(out/'missingness_summary.csv')
    s.to_csv(folder/'sensitivity_source.csv',index=False)
    missing.to_csv(folder/'missingness_source.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(9.7,7.4),gridspec_kw={'wspace':.35,'hspace':.48})
    ax=axes[0,0]
    for method in METHODS:
        p=s[(s.axis=='budget')&(s.method==method)].copy()
        xx=p.budget.tolist()+[200]
        yy=(100*p.joint_pass_fraction).tolist()+[100*main.set_index('method').loc[method,'joint_pass_fraction']]
        order=np.argsort(xx)
        ax.plot(np.array(xx)[order],np.array(yy)[order],marker='o',ms=3,color=COLORS[method],
                lw=2 if method=='CS' else 1.1,ls='-' if method in ['CS','AS'] else '--',label=NAMES[method])
    ax.set_xticks([50,100,200,300]);ax.set_xlabel('Targets per optimized class')
    ax.set_ylabel('All-four-class pass rate (%)');ax.set_ylim(-4,104)
    panel(ax,'a','Same B=200 coverage requirement')
    ax=axes[0,1]
    for method in ['CS','AS']:
        p=s[(s.axis=='threshold')&(s.method==method)]
        x=p.fraction.tolist()+[.95];y=p.mean_swaps.tolist()+[main.set_index('method').loc[method,'mean_swaps_per_class']]
        order=np.argsort(x)
        ax.plot(np.array(x)[order],np.array(y)[order],marker='o',color=COLORS[method],label=NAMES[method])
    ax.set_xticks([.9,.95,1.0],['90%','95%','100%']);ax.set_xlabel('Fraction of frozen witness coverage')
    ax.set_ylabel('Accepted swaps per class (mean)');panel(ax,'b','Threshold controls search work')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    for method in ['CS','AS']:
        p=s[(s.axis=='support_cap')&(s.method==method)]
        x=p.support_cap.tolist()+[20];y=p.mean_swaps.tolist()+[main.set_index('method').loc[method,'mean_swaps_per_class']]
        order=np.argsort(x)
        ax.plot(np.array(x)[order],np.array(y)[order],marker='o',color=COLORS[method],label=NAMES[method])
    ax.set_xticks([10,20,30]);ax.set_xlabel('Support target K');ax.set_ylabel('Accepted swaps per class (mean)')
    panel(ax,'c','Same witness; cap-specific thresholds')
    ax=axes[1,1]
    classcolors={'horse':'#3D527D','sheep':'#9483AF','person':'#278D92','car':'#D68B62'}
    for c,p in missing.groupby('class_name'):
        ax.plot(100*p.missing_rate,100*p.clean_support,color=classcolors[c],marker='o',label=c)
    ax.set_xticks([5,10,20]);ax.set_xlabel('Specific-field hiding probability (%)')
    ax.set_ylabel('Clean support completion (%)');panel(ax,'d','Synthetic missingness, not label accuracy')
    ax.legend(fontsize=8,ncol=2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,1.04),ncol=3,fontsize=8)
    fig.text(.08,-.015,'One factor at a time. R/CS/AS: 50 fixed seeds; deterministic methods: one set.\n'
             'Missingness: 20 fixed repeats per class/rate; assessed using the unmodified state matrix.',
             fontsize=8,color='#59616B',va='top')
    save(fig,folder,'sensitivity')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    folder=args.out/'figures';folder.mkdir(exist_ok=True)
    setup()
    overview(args.out,folder);metrics_grid(args.out,folder);sensitivity(args.out,folder)
    destination=folder/'plot_conditional_coverage_experiment.py'
    if Path(__file__).resolve()!=destination.resolve():
        shutil.copyfile(__file__,destination)
    print('Saved 3 editable SVG figures, PNG/PDF previews, source CSVs and plot script.',flush=True)

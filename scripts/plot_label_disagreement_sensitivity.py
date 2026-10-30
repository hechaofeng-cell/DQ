#!/usr/bin/env python3
"""Plot cached retention and support-gated risk contrasts without changing analysis."""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/label_disagreement_sensitivity_20261005'


def main():
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,
        'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    table=pd.read_csv(OUT/'tables/risk_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(9.5,5.3),layout='constrained')
    groups=['all','risk_agree','shared_agree'];names=['All','Risk agree','14-field agree']
    colors=['#767676','#3775BA','#9A4D8E']
    for column,(pool,title) in enumerate([('B','COCO'),('C','VOC'),('D','Objects365')]):
        top=axes[0,column];bottom=axes[1,column]
        part=table[(table.pool==pool)&(table.source=='local')].set_index('group').loc[groups]
        bars=top.bar(range(3),100*part.retention,color=colors,width=.6)
        for bar,value in zip(bars,100*part.retention):
            top.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.2f}%',ha='center',fontsize=7)
        top.set(title=f'{pool}: {title}',ylabel='Targets retained (%)',ylim=(0,115),xticks=range(3),xticklabels=names,xlim=(-.4,2.4))
        bottom.axhline(0,color='.8',lw=.7)
        for source,shift,marker in [('local',-.10,'o'),('online',.10,'s')]:
            for j,group in enumerate(groups):
                r=table[(table.pool==pool)&(table.source==source)&(table.group==group)].iloc[0]
                if r.evaluable:
                    center=100*r.mean_delta
                    bottom.errorbar(j+shift,center,yerr=[[max(0,center-100*r.ci95_low)],[max(0,100*r.ci95_high-center)]],
                        fmt=marker,color=colors[j],ms=4,capsize=2,label=source if j==0 else None)
                else:
                    bottom.text(j,-2 if source=='local' else -6,'NA',ha='center',fontsize=7)
        bottom.set(ylabel='Q5 minus Q1 Macro-F1 (pp)',ylim=(-55,5),xticks=range(3),xticklabels=names,xlim=(-.4,2.4))
        bottom.legend(fontsize=7,loc='lower right')
        top.text(-.16,1.03,chr(ord('a')+column),transform=top.transAxes,fontweight='bold')
        bottom.text(-.16,1.03,chr(ord('d')+column),transform=bottom.transAxes,fontweight='bold')
        for ax in (top,bottom):ax.tick_params(axis='x',rotation=15)
    folder=OUT/'figures'
    fig.savefig(folder/'label_sensitivity.svg',bbox_inches='tight')
    fig.savefig(folder/'label_sensitivity.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(Path(__file__),folder/Path(__file__).name)


if __name__=='__main__':main()

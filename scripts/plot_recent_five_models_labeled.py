#!/usr/bin/env python3
"""Annotated real Q1-Q5 results; independent of the frozen training scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['yolo26s_cls', 'aimv2_large', 'mobileclip2_s0', 'siglip2_base', 'pe_core_b16']
NAMES = dict(zip(MODELS, ['YOLO26-S-cls', 'AIMv2-Large', 'MobileCLIP2-S0', 'SigLIP2-B', 'PE-B16-224']))
SOURCES = {'local': '本地VLM状态分层', 'online': '线上VLM状态分层'}
POOLS = {'B': 'B池', 'C': 'C池 · VOC', 'D': 'D池 · Objects365'}
METRICS = {'macro_f1': 'Macro-F1', 'accuracy': '准确率 Accuracy',
           'macro_recall': '宏平均召回率 Macro-Recall', 'macro_precision': '宏平均精确率 Macro-Precision'}
LABELS = ['Q1\n低风险', 'Q2\n较低风险', 'Q3\n中等风险', 'Q4\n较高风险', 'Q5\n高风险']
SOURCE_COLORS = {'local': '#0F4D92', 'online': '#A66B28'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/recent_five_models_20261009/tables/risk_layers_runs.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/recent_five_models_20261009/figures_labeled_20261010')
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    candidates = [args.font, ROOT/'scripts/assets/NotoSansSC-Regular.ttf', Path('/mnt/c/Windows/Fonts/msyh.ttc')]
    font = next((p for p in candidates if p and p.is_file()), None)
    if font is None: raise FileNotFoundError('Provide a Chinese font using --font')
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [family, 'DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False,
        'font.size': 10, 'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False})
    rows = pd.read_csv(args.input)
    assert len(rows) == 450 and rows.groupby(['pool','source','model','tier']).size().eq(3).all()
    group = ['pool','source','model','tier']
    values = rows.groupby(group)[list(METRICS)].agg(['mean','std'])
    support = rows.groupby(group).support.first()
    records = []
    for key, row in values.iterrows():
        record = dict(zip(group, key)); record.update(support=int(support.loc[key]), seeds=3)
        for metric in METRICS:
            record[metric+'_mean_percent'] = float(row[(metric,'mean')]*100)
            record[metric+'_seed_sd_pp'] = float(row[(metric,'std')]*100)
        records.append(record)
    table = pd.DataFrame(records); table.to_csv(out/'q1_q5_labeled_values.csv',index=False)
    audit = {'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
             'input_rows':len(rows), 'aggregated_rows':len(table), 'seeds_per_point':3,
             'labels_checked':0, 'files':[], 'font':str(font)}
    validator = Path('/home/hcf/.codex/skills/research-figure-composer/scripts/validate_svg.py')

    def save(fig, name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
        check = subprocess.run([sys.executable,str(validator),str(out/(name+'.svg'))],capture_output=True,text=True,check=True)
        (out/(name+'_svg_audit.txt')).write_text(check.stdout)
        audit['files'].append(name)

    cmap = LinearSegmentedColormap.from_list('state_blue', ['#F0F4F8','#7AA6CA','#0F4D92'])
    for metric,title in METRICS.items():
        fig, axes = plt.subplots(2,3,figsize=(15,9))
        fig.subplots_adjust(left=.12,right=.94,bottom=.12,top=.86,wspace=.48,hspace=.65)
        for ri,source in enumerate(SOURCES):
            for ci,pool in enumerate(POOLS):
                ax=axes[ri,ci]
                a=values.loc[(pool,source)][(metric,'mean')].unstack('tier').loc[MODELS].to_numpy()*100
                im=ax.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,a,cmap=cmap,vmin=30,vmax=100,edgecolors='white',linewidth=1)
                ax.invert_yaxis(); ax.set_aspect('auto')
                for i in range(5):
                    for j in range(5):
                        label=f'{a[i,j]:.2f}'
                        assert label==f'{100*values.loc[(pool,source,MODELS[i],j+1),(metric,"mean")]:.2f}'
                        ax.text(j,i,label,ha='center',va='center',color='white' if a[i,j]>76 else '#142E47',fontsize=10)
                        audit['labels_checked']+=1
                for j in range(5):
                    for i in np.flatnonzero(np.isclose(a[:,j],a[:,j].max(),atol=1e-10,rtol=0)):
                        ax.add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor='#202020',lw=1.4))
                ax.set_xticks(range(5),LABELS,fontsize=9)
                ax.set_yticks(range(5),[NAMES[m] for m in MODELS],fontsize=9)
                ax.tick_params(length=0,pad=6)
                ax.set_title(f'{POOLS[pool]}｜{SOURCES[source]}\n每层{len(rows[(rows.pool==pool)&(rows.source==source)&(rows.tier==1)])//15*int(support.loc[(pool,source,MODELS[0],1)])}个目标',fontsize=10,pad=12)
        # The same percentage color scale is used in every panel.
        bar=fig.colorbar(im,ax=axes,location='right',fraction=.02,pad=.02,shrink=.85)
        bar.set_label(title+'（%）');bar.outline.set_visible(False)
        fig.suptitle(f'Q1—Q5风险分层：{title}',fontsize=17,y=.97)
        fig.text(.5,.91,'单元格：三个种子的平均值（%）  ·  黑框：该层最高均值，未表示统计显著性',ha='center',fontsize=10)
        fig.text(.5,.035,'每个类别内按冻结风险评分排序，五层各占20%；分别评测，不累加，不表示覆盖率梯度。',ha='center',fontsize=10,color='#4D4D4D')
        save(fig,'q1_q5_'+metric+'_heatmap_labeled')

    for pool in POOLS:
        fig,axes=plt.subplots(2,3,figsize=(14,8.5))
        fig.subplots_adjust(left=.065,right=.97,bottom=.10,top=.85,wspace=.25,hspace=.44)
        for ai,model in enumerate(MODELS):
            ax=axes.flat[ai]
            y={s:values.loc[(pool,s,model)][('macro_f1','mean')].to_numpy()*100 for s in SOURCES}
            for source in SOURCES:
                vals=y[source]
                ax.plot(range(1,6),vals,color=SOURCE_COLORS[source],marker='o' if source=='local' else 's',
                        ls='-' if source=='local' else '--',lw=1.8,ms=4,label=SOURCES[source])
                other=y['online' if source=='local' else 'local']
                for j,v in enumerate(vals):
                    above=(v>=other[j]) if source=='local' else (v>other[j])
                    dy=10 if above else -14
                    ax.annotate(f'{v:.2f}',(j+1,v),xytext=(0,dy),textcoords='offset points',ha='center',
                                va='bottom' if above else 'top',fontsize=8,color=SOURCE_COLORS[source])
                    audit['labels_checked']+=1
            ax.set_title(NAMES[model],fontsize=12,pad=12)
            ax.set_xticks(range(1,6),LABELS,fontsize=8);ax.set_xlim(.6,5.4);ax.set_ylim(30,104)
            ax.set_yticks([30,40,50,60,70,80,90,100]);ax.set_ylabel('Macro-F1（%）')
        info=axes.flat[-1];info.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        info.legend(handles,labels,loc='upper left',frameon=False,fontsize=11)
        info.text(.02,.62,'读图说明\n\n每个点标注真实F1数值\n每条线为三个种子的均值\n各模型使用相同纵轴范围\n五个点为独立风险层，非累加\n种子标准差见配套CSV及原图',transform=info.transAxes,fontsize=11,va='top',linespacing=1.7)
        fig.suptitle(f'{POOLS[pool]}：五模型Q1—Q5表现（带数值标签）',fontsize=17,y=.97)
        fig.text(.5,.90,'蓝色实线＝本地VLM分层  ·  棕色虚线＝线上VLM分层  ·  同一模型权重，仅分层成员不同',ha='center',fontsize=10)
        save(fig,'q1_q5_'+pool+'_f1_curves_labeled')

    assert audit['labels_checked']==750
    (out/'annotation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    note='''# Q1—Q5带标签图表

四项指标分别提供总览热图；三个池分别提供五模型折线图。所有数值均为真实三种子均值，保留两位小数。数据来自risk_layers_runs.csv，未重新训练、修改结果或平滑曲线。

Q1—Q5是类别内风险排序五等分，每层六类等量；B、D每层480目标，C每层120目标。风险由预定义视觉状态评分决定，不保证每张Q5图片都比每张Q1更难。横轴不是覆盖率，也不是累加样本数。

热图各面板使用同一30—100百分比色标。黑框仅标记同一层五模型最高均值，未表示显著差异。折线图全部使用同一30—104显示范围，为标签留白；只显示均值，不把两条来源曲线的差别当成模型更换。线上／本地指VLM状态标注来源，分类器预测和权重不变。

CSV包含每点均值、种子标准差（单位：百分点）、样本量。种子波动请参考原有risk_layer_f1图；例如PE有明显训练种子波动，不能凭均值将其架构定性为固有最差。

SVG保持文字可编辑，PDF便于排版，PNG用于预览。750个数值标签已与聚合数据核对，SVG通过结构检查。

复现命令：

```bash
.venv/bin/python scripts/plot_recent_five_models_labeled.py
```

离开项目目录时可使用脚本的--input、--output和--font参数指定CSV、输出目录和中文字体。
'''
    (out/'README.md').write_text(note)
    # Generated bundle retains raw inputs, script and editable outputs.
    bundle=out.parent/'q1_q5_labeled_charts_20261010.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file(): z.write(f,f.name)
        z.write(Path(__file__),'source/plot_recent_five_models_labeled.py')
        z.write(args.input,'source/risk_layers_runs.csv')
    print(json.dumps({'output':str(out),'bundle':str(bundle),'labels_checked':750,'figures':len(audit['files'])},ensure_ascii=False))


if __name__=='__main__':main()

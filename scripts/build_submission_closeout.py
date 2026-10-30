#!/usr/bin/env python3
"""Versioned manuscript closeout, visual evidence and portable reanalysis pack."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from torchvision import transforms

import build_journal_revision as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submission_closeout_20261007'
EXP=ROOT/'artifacts/scale_only_confirmation_20261007'
OLD=ROOT/'artifacts/journal_revision_20261006'
DOCS=ROOT/'docs'
EN=DOCS/'visual_state_testing_journal_en_20261007.md'
ZH=DOCS/'visual_state_testing_journal_zh_20261007.md'
SI=DOCS/'visual_state_testing_journal_supplement_20261007.md'
COLORS={'full':'#2166AC','area':'#777777','accent':'#C96D2D'}
CLASSES=['dog','cat','horse','sheep','person','car']


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    for lang,dest in [('en',EN),('zh',ZH)]:
        if dest.exists():continue
        text=(DOCS/f'visual_state_testing_journal_{lang}_20261006.md').read_text()
        if lang=='en':
            text=re.sub(r'\b(Tables? )(\d+)(?:–(\d+))?',lambda m:m[1]+str(int(m[2])+(2 if int(m[2])>=5 else 0))+(('–'+str(int(m[3])+(2 if int(m[3])>=5 else 0))) if m[3] else ''),text)
            text=re.sub(r'\b(Figures? )(\d+)',lambda m:m[1]+str(int(m[2])+(1 if int(m[2])>=3 else 0)),text)
        else:
            text=re.sub(r'表(\d+)',lambda m:'表'+str(int(m[1])+(2 if int(m[1])>=5 else 0)),text)
            text=re.sub(r'图(\d+)',lambda m:'图'+str(int(m[1])+(1 if int(m[1])>=3 else 0)),text)
        text=text.replace('journal_revision_20261006/figures','submission_closeout_20261007/figures')
        dest.write_text(text)
    if not SI.exists():shutil.copy2(DOCS/'visual_state_testing_journal_supplement_20261006.md',SI)


def scale_tables():
    df=pd.read_csv(EXP/'tables/summary.csv')
    cond=pd.read_csv(EXP/'tables/area_condition_summary.csv')
    main=[];diagnosis=[]
    for r in df.itertuples():
        main.append([r.pool,r.source.capitalize(),f'{100*r.full_gap:.2f}',f'{100*r.area_gap:.2f}',
            f'{100*r.increment_mean:+.2f}',f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]',f'{r.holm_six_p:.5f}'])
        d=cond[(cond.pool==r.pool)&(cond.source==r.source)];valid=d[d.evaluable]
        diagnosis.append([r.pool,r.source.capitalize(),str(len(d)),str(len(valid)),
                          str((valid.area_adjusted_recall_gap<0).sum()),f'{100*valid.area_adjusted_recall_gap.mean():+.2f}'])
    t1=base.md_table(pd.DataFrame(main,columns=['Pool','Annotator','Full G (pp)','Area-only G (pp)',
                      'Full − area (pp)','Model/seed 95% CI','Six-test Holm p']))
    t2=base.md_table(pd.DataFrame(diagnosis,columns=['Pool','Annotator','Frozen non-scale hypotheses',
                      'Evaluable','Negative gap','Mean adjusted recall gap (pp)']))
    translations={'Pool':'池','Annotator':'标注来源','Local':'本地','Online':'线上','Full G (pp)':'完整G(pp)',
        'Area-only G (pp)':'仅面积G(pp)','Full − area (pp)':'完整−面积(pp)',
        'Model/seed 95% CI':'模型/种子95%区间','Six-test Holm p':'六项Holm p',
        'Frozen non-scale hypotheses':'冻结非尺度假设','Evaluable':'可评价','Negative gap':'负差',
        'Mean adjusted recall gap (pp)':'平均尺度调整Recall差(pp)'}
    for lang,path in [('en',EN),('zh',ZH)]:
        text=path.read_text()
        def translate(block):
            if lang=='en':return block
            for a,b in translations.items():block=block.replace('| '+a+' |','| '+b+' |')
            return block
        text=text.replace('<!-- SCALE_TABLE -->',translate(t1)).replace('<!-- AREA_DIAG_TABLE -->',translate(t2))
        path.write_text(text)
    df.to_csv(OUT/'scale_table_source.csv',index=False)
    return df,cond,t1,t2


def save(fig,name):
    for ext in ['svg','pdf','eps','png']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white',dpi=300)
    plt.close(fig)


def figures(df):
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    for p in (OLD/'figures').iterdir():shutil.copy2(p,OUT/'figures'/p.name)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(6.2,4.1),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
    for i,r in enumerate(df.itertuples()):
        axes[0].plot([100*r.area_gap,100*r.full_gap],[i,i],color='#BBBBBB',lw=1)
        axes[0].scatter(100*r.area_gap,i,color=COLORS['area'],marker='s',s=30,label='Area only' if i==0 else None)
        axes[0].scatter(100*r.full_gap,i,color=COLORS['full'],marker='D',s=30,label='Full risk' if i==0 else None)
        axes[0].annotate(f'{100*r.full_gap:.2f}',(100*r.full_gap,i),xytext=(-3,9),textcoords='offset points',ha='right',color=COLORS['full'],fontsize=9)
        axes[0].annotate(f'{100*r.area_gap:.2f}',(100*r.area_gap,i),xytext=(3,-15),textcoords='offset points',ha='left',color=COLORS['area'],fontsize=9)
        axes[1].errorbar(100*r.increment_mean,i,xerr=[[100*(r.increment_mean-r.ci_low)],[100*(r.ci_high-r.increment_mean)]],
                          fmt='o',color=COLORS['full'],capsize=3,ms=4)
        axes[1].annotate(f'{100*r.increment_mean:+.2f}',(100*r.increment_mean,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    labels=[f'{r.pool} · {r.source}' for r in df.itertuples()]
    axes[0].set(yticks=range(6),yticklabels=labels,ylim=(5.65,-.8),xlim=(5,53),xlabel='Q1 − Q5 Macro-F1 (pp)',title='a  Endpoint separation')
    axes[1].set(yticks=range(6),yticklabels=[],ylim=(5.65,-.8),xlim=(-5.5,10),xlabel='Full − area-only gap (pp)',title='b  Paired difference')
    axes[1].axvline(0,color='#777777',ls='--',lw=.6)
    axes[0].legend(ncol=2,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.22))
    save(fig,'05_scale_only_control')


def real_cases():
    manifest=pd.read_csv(base.BASES['B']/'manifests/candidate_pool.csv')
    states=json.loads((EXP/'portable_inputs/B_states.json').read_text())
    cache=np.load(EXP/'portable_inputs/B_predictions.npz')
    pred=cache['predictions'];models=cache['models'];seeds=cache['seeds']
    fig,axes=plt.subplots(4,3,figsize=(6.2,9),layout='constrained',gridspec_kw={'width_ratios':[1.1,1,1.1]})
    specs=[('local','resnet50','dog','body_orientation','back',False),
           ('local','resnet50','cat','target_scale','small',False),
           ('online','resnet50','sheep','blur_level','severe',False),
           ('local','densenet121','dog','target_scale','medium',True)]
    rows=[];(OUT/'case_inputs').mkdir(exist_ok=True)
    for rownum,(source,model,category,field,value,want_correct) in enumerate(specs):
        selected=[];runidx=np.flatnonzero(models==model)
        for i,r in manifest.iterrows():
            if r['class_name']!=category:continue
            if field=='target_scale':
                a=r.bbox_area_ratio;state='large' if a>=.1 else 'medium' if a>=.03 else 'small' if a>=.01 else 'tiny'
            else:state=states[source][r.sample_id][field]
            if state==value and np.all((pred[runidx,i]==int(r.label))==want_correct):selected.append(i)
        assert selected,(field,value)
        i=min(selected,key=lambda i:manifest.sample_id.iloc[i]);r=manifest.iloc[i]
        original=Image.open(r.image_path).convert('RGB');crop=Image.open(r.crop_path).convert('RGB')
        actual=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])(crop)
        actual.save(OUT/'case_inputs'/f'{rownum+1}_classifier_input.png')
        axes[rownum,0].imshow(original)
        box=json.loads(r.bbox_norm_xyxy);w,h=original.size
        axes[rownum,0].add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w,(box[3]-box[1])*h,
                                            fill=False,color='#C96D2D',lw=1.3))
        axes[rownum,1].imshow(actual)
        labels=[CLASSES[int(pred[j,i])] for j in runidx]
        text=f"{category} · {model}\n{source}: {field}\n= {value}\nArea ratio: {r.bbox_area_ratio:.4f}\n\nPredictions (3 seeds):\n"+' / '.join(labels)+f"\nGT: {category}\n"+('Correct example' if want_correct else 'Failure example')
        axes[rownum,2].text(0,.95,text,transform=axes[rownum,2].transAxes,va='top',fontsize=9,linespacing=1.35)
        for ax in axes[rownum]:ax.axis('off')
        axes[rownum,0].set_title('abcd'[rownum],loc='left',fontweight='bold',fontsize=10)
        rows.append(dict(sample_id=r.sample_id,source=source,model=model,class_name=category,field=field,value=value,
            area_ratio=float(r.bbox_area_ratio),seeds=seeds[runidx].tolist(),predictions=labels,
            image_path=r.image_path,crop_path=r.crop_path,image_sha256=r.image_sha256,crop_sha256=r.crop_sha256,
            selection='lexicographically first target satisfying the prelisted state and all-three-seed outcome',
            semantic_human_review=False))
    for ax,title in zip(axes[0],['Original + official box','Actual classifier input','Automatic state + outcome']):
        ax.set_title(title,fontsize=10)
    save(fig,'06_real_target_cases');dump(OUT/'case_manifest.json',rows)


def supplement(df,cond):
    text=(DOCS/'visual_state_testing_journal_supplement_20261006.md').read_text()
    text=text.replace('image assets and full prediction caches remain in the local experiment directories and are not falsely described as publicly deposited.',
        'the new portable bundle includes target identifiers, states and all classification logits/predictions for B/C/D. It is a local review package, not a public deposit. Full source images, training checkpoints and original feature caches are not bundled.')
    text=text.replace('Shared-field normalization is given in the included source code.',
        'Shared fields retain identical names and enumerations across the two native protocols; protocol-specific fields are excluded. The bundled shared schema specifies the 14 retained fields.')
    text=text.replace('Table 7 to each pool comparison table; Table 8 to frozen transfer tables; Tables 9–11',
        'Table 9 to each pool comparison table; Table 10 to frozen transfer tables; Tables 11–13')
    text=text.replace('Tables 5–6 to distribution/panel tables','Tables 7–8 to distribution/panel tables')
    text=text.replace('These indices identify retained local evidence rather than a currently public repository.',
        'The new scale tables 5–6 map to summary.csv and area_condition_summary.csv. These indices identify local review evidence, not a public repository.')
    text += '\n\n## S11 Continuous-area control and residual condition diagnosis\n\n'
    text += 'The supplementary cached analysis was specified on 2026-10-07 after earlier outcomes were known. Area-only ranking uses negative continuous official box-area ratio; full risk and area are sorted within class with sample-ID ties and equal quintiles. G is Q1−Q5 macro-F1. The primary paired contrast is G_full−G_area, with eight architecture units, three seeds per unit, 5,000 hierarchical bootstrap draws and a separate six-test Holm family. Random tie priorities are shared between schemes, changed only within exact-score ties, and repeated 1,000 times; their central ranges measure grouping variation, not model-population uncertainty.\n\n'
    text += '**Table S6. Complete paired scale-control outputs, in F1 units.**\n\n'+base.md_table(df)+'\n\n'
    text += 'Non-scale diagnostic hypotheses retain the discovery-A freeze. Within each class and continuous-area quintile, a state and the same-field known non-state reference must each have at least two images. A hypothesis is evaluable only with at least 20 state images overall, 20 matched state and 20 matched reference images, and at least 50% state retention. Recall differences are weighted by matched state counts. Matching controls quintile membership, not exact continuous area or all confounders. Overlapping hypotheses are not independent defects. All non-scale hypotheses, including nonnegative and insufficient-support rows, appear in area_condition_summary.csv.\n\n'
    u=pd.read_csv(EXP/'tables/unknown_rates.csv')
    u=u.groupby(['pool','source','field'])[['targets','unknown']].sum().reset_index()
    u['unknown_fraction']=u.unknown/u.targets
    text += '## S12 Unknown rates and stratum support\n\n'
    text += 'unknown_rates.csv provides every class-specific native field denominator, including protocol-specific fields. The table below aggregates native fields by pool and source; a field absent from the car protocol does not receive a false zero count. stratum_profiles.csv and memberships.csv retain every class-wise layer score/area range and support, for both full-risk and continuous-area schemes.\n\n'
    text += '**Table S7. Unknown fractions by native field; absent fields are excluded from that field denominator.**\n\n'+base.md_table(u)+'\n\n'
    text += '## S13 Reanalysis entry point, prompts and implementation scope\n\n'
    text += 'Extract the portable ZIP and run `python scripts/run_scale_only_confirmation.py --input portable_inputs --output reproduced_scale`. Requires Python with NumPy, Pandas and scikit-learn; pinned observed versions are in environment.json. No network, GPU or API credentials are required. The bundle provides the exact native old/car prompts, three annotation schemas, two-source automatic states, all 24-classifier predictions/logits on B/C/D, programmatic score parts, frozen historical full-risk memberships, target hashes and fixed weak-state hypotheses. It reproduces the new scale contrast, all historical full-risk layer metrics, area-conditioned diagnostics and unknown/support tables. Older random grouping and mixture outputs, selection analysis tables and provenance are included with the available caches. The entry point does not re-create original image sampling, classifier training or historical VLM calls.\n\n'
    text += 'For the old selection breadth comparisons, fixed/equal-error/random metric tables are retained and budget AUC can be reaggregated without relabeling. Original embedding feature extraction and clustering are not regenerated by the portable scale script. Model/version strings and recorded request settings are provided without credentials. API-version aliases do not establish provider-side weight hashes.\n\n'
    text += 'The second standalone command is `python scripts/reaggregate_selection_evidence.py --input historical_selection --output reproduced_selection`. It recomputes all 60 historical mean-AUC contrasts, exact architecture-level sign-flip p values and four correction families, and checks them against retained comparison tables. It does not reconstruct the original bootstrap intervals. Archived implementation files support inspection of historical definitions but are not standalone end-to-end runners.\n\n'
    text += '**Table S8. Historical common-quota baseline implementation details.**\n\n'
    text += '| Component | Recorded implementation |\n|---|---|\n'
    text += '| Common feasibility | Within class and each full-risk quintile, allocate by the joint cell of score and unknown-dimension count using proportional largest remainder. B/D add 20 per class per layer; C adds 10. Quotas and orders are retained. |\n'
    text += '| Coverage | Greedy sum of log(n+2)−log(n+1) over active field–state indicators whose previous count is below 20; counts persist across layers; ascending sample-ID ties. |\n'
    text += '| Random | 1,000 without-replacement within-cell sequences; the retained random_orders.npz fixes the actual selections. |\n'
    text += '| DeepGini | 1−sum(p_k squared), with probabilities from stabilized softmax of frozen logits; choose highest score within the common feasible set. |\n'
    text += '| DSA | Raw classifier feature distances. Nearest training example has label equal to the candidate predicted class; divide candidate-to-anchor distance by anchor-to-nearest different-label training distance, with denominator floor 1e-12. Choose highest score. |\n'
    text += '| k-center | L2-normalized ViT-B/16 vectors; cosine distance 1−dot product. First choose feasible target closest to the current layer centroid; later maximize nearest-selected distance across layers, with ascending ID ties. |\n'
    text += '| Coverage without H | Remove target_scale, body_visibility, occlusion_level, truncation_level, body_orientation and target_recognizability; retain the other nine shared fields under unchanged quotas. |\n'
    text += '| Error clusters | Per true category and frozen model/seed, L2-normalized wrong-target features; k=max(2,min(8,round(sqrt(number of errors)))) for at least two errors. Zero/one error has zero/one cluster. KMeans n_init=20, random_state=20260928+training_seed+class_offset. |\n\n'
    text += '## S14 Real-image cases and practical effect sizes\n\n'
    text += 'The four displayed targets are illustrative, chosen by the explicit first-ID rule in case_manifest.json within the four pre-existing manuscript conditions. Predictions are drawn from frozen caches; labels are automatic and are not certified by manual ground truth. The displayed classifier input is the stored margin crop after Resize(256) and CenterCrop(224), before tensor normalization. Original-image context available to the VLM is not the classifier input. These selected cases cannot estimate field-label accuracy or population failure rates.\n\n'
    text += 'practical_breadth.csv reports equal-error expected normalized confusion/cluster breadth at the largest budget, in addition to the budget AUC. Fractional discoveries arise from exact error-count rarefaction; they are not integer independently verified bugs. Universe sizes vary by architecture/representation, so a normalized panel mean cannot be multiplied by a single arbitrary denominator.\n'
    practical=pd.read_csv(OUT/'practical_breadth.csv')
    view=practical[['pool','source','budget_per_class','confusion_left_mean','confusion_right_mean','confusion_delta','cluster_left_mean','cluster_right_mean','cluster_delta']].copy()
    text += '\n**Table S9. Largest-budget equal-error breadth: Coverage (left) versus matched Random (right); normalized fractions, not percentage points or new significance tests.**\n\n'+base.md_table(view)+'\n'
    text += '\nRevision v2 (7 October 2026) synchronizes the manuscript and companion strengths/limitations reports without new experiments. Independent extraction replay matches all eight new scale/condition tables and all 60 historical selection effect estimates, exact tests and correction families. This verifies cached computational analysis, not human semantic truth, original feature/cluster reconstruction or from-scratch training.\n'
    SI.write_text(text)


def practical_breadth():
    rows=[]
    for pool,p in base.BASES.items():
        d=pd.read_csv(p/'comparison/tables/equal_error_metrics.csv')
        maximum=int(d.budget.max())
        d=d[(d.baseline=='random')&(d.budget==maximum)&(d.representation=='classifier')]
        assert len(d)==48
        for source,sub in d.groupby('source'):
            rows.append(dict(pool=pool,source=source,budget_per_class=maximum,
                confusion_left_mean=float(sub.left_confusion.mean()),confusion_right_mean=float(sub.right_confusion.mean()),
                cluster_left_mean=float(sub.left_clusters.mean()),cluster_right_mean=float(sub.right_clusters.mean()),
                confusion_delta=float(sub.delta_confusion.mean()),cluster_delta=float(sub.delta_clusters.mean()),
                scope='equal-error expected normalized breadth; not an integer independent bug count'))
    pd.DataFrame(rows).to_csv(OUT/'practical_breadth.csv',index=False)


REFERENCE_LABELS=['Aghababaeyan et al.(2023)','Aghababaeyan et al.(2024)','Aich et al.(2026)',
    'Dola et al.(2023)','Eyuboglu et al.(2022)','Feng et al.(2020)','Harel-Canada et al.(2020)',
    'Kim et al.(2019)','Lin et al.(2014)','Ma et al.(2018)','Objects365 Consortium(n.d.)',
    'Open Images(n.d.)','PASCAL VOC(2007)','Pei et al.(2017)','Sener and Savarese(2018)','Slyman et al.(2023)']


def render_documents():
    base.OUT=OUT;base.EN=EN;base.ZH=ZH;base.SI=SI
    names={}
    for i,label in enumerate(REFERENCE_LABELS,1):
        author,year=label[:-1].split('(')
        names[author+', '+year]=['ref'+str(i)]
    names['Aghababaeyan et al., 2023, 2024']=['ref1','ref2']
    def prose(text):
        macros=[]
        def token(m):
            key=f'@@CITE{len(macros)}@@';macros.append((key,m));return key
        def cite(m):
            parts=[p.strip() for p in m[1].split(';')]
            if not all(p in names for p in parts):return m[0]
            return token(r'\citep{'+','.join(k for p in parts for k in names[p])+'}')
        text=re.sub(r'\(([^()]+)\)',cite,text)
        for label,keys in names.items():
            if len(keys)==1:
                author,year=label.rsplit(', ',1)
                text=text.replace(author+' ('+year+')',token(r'\citet{'+keys[0]+'}'))
        result=base.tex_inline(text)
        for k,v in macros:result=result.replace(k,v)
        return result
    base.tex_prose=prose
    (OUT/'vendor').mkdir(exist_ok=True)
    shutil.copy2(OLD/'vendor/springer_template.zip',OUT/'vendor/springer_template.zip')
    base.latex_main(EN.read_text(),REFERENCE_LABELS)
    records=[base.word(EN),base.word(ZH),base.word(SI,False)]
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df,cond,t1,t2=scale_tables();figures(df);real_cases();practical_breadth();supplement(df,cond)
    # Ensure the Chinese review copy uses the same verified, alphabetic references.
    ZH.write_text(ZH.read_text().split('## 参考文献\n')[0]+'## 参考文献\n\n'+EN.read_text().split('## References\n')[1])
    print(json.dumps(render_documents(),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');args=p.parse_args()
    if args.initialize:initialize()
    else:main()

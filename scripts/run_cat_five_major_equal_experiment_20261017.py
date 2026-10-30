#!/usr/bin/env python3
"""Run the cat pilot with the latest five-major-class equal-weight protocol.

This is a versioned exploratory re-score of the frozen 20-target cat pilot. It
does not overwrite the historical five-risk-dimension pilot.
"""
from __future__ import annotations
import csv, json, math, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/cat_human_vlm_v5_pilot_20261017"
OUT = ROOT / "artifacts/cat_five_major_equal_20261017"
REF = ROOT / "artifacts/QE_FullFine_v5_20261013/annotations"
RUNS = ROOT / "artifacts/sixclass_dual_vlm_replaced_20261004/benchmark/runs"
OUT.mkdir(parents=True, exist_ok=True)

MAJOR = ["空间几何", "可见性与完整性", "外观与形态", "朝向与姿态", "场景与图像质量"]
COLORS = {"空间几何":"#4C78A8", "可见性与完整性":"#59A14F", "外观与形态":"#F28E2B", "朝向与姿态":"#B279A2", "场景与图像质量":"#E15759"}

def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def flat(v):
    if isinstance(v, list): return "+".join(map(str, v))
    return str(v) if v is not None else "unknown"

def is_known(v):
    return v not in (None, "unknown", "not_visible", "not_observable", "not_in_view")

def parse_num(v):
    try: return float(v)
    except Exception: return float("nan")

def read_annotations():
    rows=[]
    for p in sorted((SRC/"annotations").glob("*.json")):
        o=json.loads(p.read_text(encoding="utf-8"))
        rows.append({"sample_id":o["image_id"], "features":o["features"], "annotation_path":str(p)})
    manifest=pd.read_csv(SRC/"manifest.csv", encoding="utf-8-sig")
    m=manifest.set_index("sample_id").to_dict("index")
    for r in rows: r.update(m[r["sample_id"]])
    return rows

def reference_counts():
    fields = ["coat_primary_color","coat_pattern","coat_texture","ear_posture","body_shape","eye_state","mouth_state",
              "tail_posture","posture_action_state","scene","group_context","interaction_context"]
    counts={f:Counter() for f in fields}
    n=0
    for p in REF.glob("*-cat.json"):
        try: x=json.loads(p.read_text(encoding="utf-8"))["features"]
        except Exception: continue
        n+=1
        for f in fields:
            v=flat(x.get(f,"unknown"))
            if v not in ("unknown","not_visible","not_observable","not_in_view"): counts[f][v]+=1
    return counts,n

REF_COUNTS, REF_N = reference_counts()

def rarity(field, value):
    """Reference-distribution deviation in [0,1]; unknown is an interval."""
    if not is_known(value): return None
    c=REF_COUNTS[field]
    if not c: return 0.0
    # rare states receive larger exploratory deviation; +1 smoothing avoids 1.0 for unseen.
    return float(1.0 - (c.get(flat(value),0)+1)/(max(c.values())+1))

def fixed(v, mapping):
    if not is_known(v): return None
    return mapping.get(flat(v), None)

def geom_score(r):
    a=parse_num(r.get("bbox_area_ratio"))
    if not np.isfinite(a): return None
    if a >= .10: return 0.0
    if a >= .03: return 1/3
    if a >= .01: return 2/3
    return 1.0

def compute_group_values(r):
    f=r["features"]
    g=[geom_score(r)]
    vis_maps={"head_visibility":{"clear":0,"partial":.5},"torso_visibility":{"clear":0,"partial":.5},
              "limb_visibility":{"clear":0,"partial":.5},"occlusion_level":{"none":0,"partial":.5,"heavy":1},
              "truncation_level":{"none":0,"partial":.5,"heavy":1}}
    v=[fixed(f.get(k),mp) for k,mp in vis_maps.items()]
    a=[rarity(k,f.get(k)) for k in ["coat_primary_color","coat_pattern","coat_texture","ear_posture","body_shape","eye_state","mouth_state"]]
    # Orientation: front/side are direct; back is partial difficulty; upward/downward are hard to inspect.
    hm={"front":0,"left":0,"right":0,"back":.5,"upward":1,"downward":1}
    bm={"front":0,"left_side":0,"right_side":0,"left_front_three_quarter":0,"right_front_three_quarter":0,
        "back":.5,"left_back_three_quarter":.5,"right_back_three_quarter":.5}
    p=[fixed(f.get("head_orientation"),hm), fixed(f.get("body_orientation"),bm),
       rarity("tail_posture",f.get("tail_posture")), rarity("posture_action_state",f.get("posture_action_state"))]
    e=[rarity("scene",f.get("scene")), fixed(f.get("background_complexity"),{"simple":0,"moderate":.5,"complex":1}),
       fixed(f.get("blur_level"),{"clear":0,"mild":.5,"severe":1}),
       fixed(f.get("illumination"),{"normal":0,"low_light":1,"backlit":1,"strong_shadow":1,"overexposed":1,"mixed_or_colored":1}),
       fixed(f.get("target_background_contrast"),{"high":0,"medium":.5,"low":1}),
       rarity("group_context",f.get("group_context")), rarity("interaction_context",f.get("interaction_context")),
       fixed(f.get("target_recognizability"),{"high":0,"medium":.5,"low":1,"unrecognizable":1})]
    return {"空间几何":g,"可见性与完整性":v,"外观与形态":a,"朝向与姿态":p,"场景与图像质量":e}

COMPONENT_NAMES = {
    "空间几何":["bbox_area_ratio"],
    "可见性与完整性":["head_visibility","torso_visibility","limb_visibility","occlusion_level","truncation_level"],
    "外观与形态":["coat_primary_color","coat_pattern","coat_texture","ear_posture","body_shape","eye_state","mouth_state"],
    "朝向与姿态":["head_orientation","body_orientation","tail_posture","posture_action_state"],
    "场景与图像质量":["scene","background_complexity","blur_level","illumination","target_background_contrast","group_context","interaction_context","target_recognizability"],
}

def summarize(vals):
    known=[x for x in vals if x is not None]
    lo=float(sum(known)/len(vals)) if vals else 0.0
    hi=float((sum(known)+(len(vals)-len(known)))/len(vals)) if vals else 0.0
    mid=(lo+hi)/2
    return lo,mid,hi,len(known),len(vals)

def encode_states(r):
    f=r["features"]
    out={}
    # Geometry state bins are explicit, while fine-grained field states are grouped by major class.
    a=parse_num(r.get("bbox_area_ratio"));
    out["空间几何/bbox_scale"] = "large" if np.isfinite(a) and a>=.10 else ("medium" if np.isfinite(a) and a>=.03 else ("small" if np.isfinite(a) and a>=.01 else "tiny"))
    groups={
      "可见性与完整性":["head_visibility","torso_visibility","limb_visibility","occlusion_level","truncation_level"],
      "外观与形态":["coat_primary_color","coat_pattern","coat_texture","ear_posture","body_shape","eye_state","mouth_state"],
      "朝向与姿态":["head_orientation","body_orientation","tail_posture","posture_action_state"],
      "场景与图像质量":["scene","background_complexity","blur_level","illumination","target_background_contrast","group_context","interaction_context","target_recognizability"],
    }
    for g,fs in groups.items():
        for f in fs:
            v=flat(f and f in r["features"] and r["features"].get(f,"unknown"))
            if v not in ("unknown","not_visible","not_observable","not_in_view"):
                out[f"{g}/{f}={v}"]=1
    return out

def assign_tiers(df, score_col="total_mid"):
    df=df.sort_values([score_col,"sample_id"], ascending=[True,True]).reset_index(drop=True)
    n=len(df); q=np.minimum(5, (np.arange(n)*5//n)+1)
    df["Q"] = q
    return df

def coverage(keys_list, k=2):
    c=Counter()
    for keys in keys_list: c.update(keys)
    occur=sum(v>0 for v in c.values()); support=sum(v>=k for v in c.values())
    return {"state_count":len(c),"occurrence_covered":occur,"support_covered":support,"support_rate":support/max(1,len(c))}

def greedy_select(df, budget=10, k=2):
    remaining=set(df.index); chosen=[]; counts=Counter()
    allstates=set().union(*(set(x) for x in df.state_keys))
    # Q quotas, two per tier for the 20-target pilot.
    quota={q:budget//5 for q in range(1,6)}
    while remaining and len(chosen)<budget:
        candidates=[i for i in remaining if sum(df.loc[j,"Q"]==df.loc[i,"Q"] for j in chosen)<quota[int(df.loc[i,"Q"])]]
        if not candidates: candidates=list(remaining)
        best=None; best_gain=-1
        for i in candidates:
            gain=sum(math.log1p(min(k,counts[s]+1))-math.log1p(min(k,counts[s])) for s in df.loc[i,"state_keys"])
            # small deterministic preference for higher total within ties
            gain += 1e-9*float(df.loc[i,"total_mid"])
            if gain>best_gain: best_gain,best=gain,i
        chosen.append(best); remaining.remove(best); counts.update(df.loc[best,"state_keys"])
    return chosen

def model_metrics(ids_by_method, tier_map):
    rows=[]; cat_index=1
    run_files=sorted(RUNS.glob("*/20*/candidate_outputs.npz")) + sorted(RUNS.glob("*/202609*/candidate_outputs.npz"))
    # de-duplicate paths and keep all frozen seeds/architectures.
    run_files=sorted(set(run_files))
    idset=set().union(*[set(v) for v in ids_by_method.values()])
    for rp in run_files:
        z=np.load(rp, allow_pickle=True); sid=z["sample_ids"].astype(str); pred=np.argmax(z["logits"],axis=1)
        pos={s:i for i,s in enumerate(sid)}
        for method, ids in ids_by_method.items():
            for q in range(1,6):
                qids=[s for s in ids if tier_map.get(s)==q]
                if not qids: continue
                hit=sum(int(pred[pos[s]]==cat_index) for s in qids if s in pos)
                rows.append({"method":method,"Q":q,"model_seed":rp.parent.parent.name+"/"+rp.parent.name,"recall":hit/len(qids),"n":len(qids)})
    return pd.DataFrame(rows)

def recent_five_model_f1():
    """Read the frozen latest-five-model metrics; no model is retrained here."""
    base=ROOT/"artifacts/recent_five_models_20261009/runs"
    files=sorted(base.glob("*/20*/overall.csv"))
    frames=[]
    for p in files:
        try: frames.append(pd.read_csv(p))
        except Exception: pass
    if not frames: return pd.DataFrame(), pd.DataFrame()
    runs=pd.concat(frames,ignore_index=True)
    order=["yolo26s_cls","pe_core_b16","siglip2_base","mobileclip2_s0","aimv2_large"]
    runs["model"] = pd.Categorical(runs["model"], categories=order, ordered=True)
    summary=(runs.groupby(["model","pool"],observed=True)[["macro_f1","macro_precision","macro_recall","cat_f1","cat_precision","cat_recall"]]
             .agg(["mean","std"]).reset_index())
    summary.columns=["model","pool"]+[f"{a}_{b}" for a,b in summary.columns.tolist()[2:]]
    return runs,summary

def plot_recent_five_f1(summary):
    if summary.empty: return
    test=summary[summary.pool=="test"].copy().set_index("model").reindex(["yolo26s_cls","pe_core_b16","siglip2_base","mobileclip2_s0","aimv2_large"])
    font=ROOT/"scripts/assets/NotoSansSC-Regular.ttf"
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    fig,ax=plt.subplots(figsize=(10,5.5),dpi=180); x=np.arange(len(test)); w=.34
    ax.bar(x-w/2,100*test.macro_f1_mean,w,yerr=100*test.macro_f1_std.fillna(0),capsize=4,label="Macro-F1",color="#1976D2")
    ax.bar(x+w/2,100*test.cat_f1_mean,w,yerr=100*test.cat_f1_std.fillna(0),capsize=4,label="Cat-F1",color="#F57C00")
    ax.set_xticks(x,["YOLO26s","PE-Core\nB16","SigLIP2\nBase","MobileCLIP2\nS0","AIMv2\nLarge"]); ax.set_ylim(0,100); ax.set_ylabel("F1（%）"); ax.set_title("最新五模型：测试池 Macro-F1 与猫类 F1（3 seeds均值±标准差）",fontweight="bold"); ax.grid(axis="y",alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"最新五模型F1.png",bbox_inches="tight"); fig.savefig(OUT/"最新五模型F1.svg",bbox_inches="tight"); plt.close(fig)

def cat_qtier_f1(df):
    """Compute one-vs-rest Cat-F1 by the current Q tier from ID-aligned frozen runs."""
    qmap=dict(zip(df.sample_id.astype(str),df.Q.astype(int)))
    out=[]
    for rp in sorted(RUNS.glob("*/20*/candidate_outputs.npz")):
        z=np.load(rp,allow_pickle=True); sid=z["sample_ids"].astype(str); pred=np.argmax(z["logits"],axis=1); pos={s:i for i,s in enumerate(sid)}
        if not all(s in pos for s in qmap): continue
        for q in range(1,6):
            qids=[s for s,t in qmap.items() if t==q]; pred_cat=np.array([pred[pos[s]]==1 for s in qids]); tp=int(pred_cat.sum()); fp=int(pred_cat.size-tp); fn=0
            precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0; f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
            out.append({"model":rp.parent.parent.name,"seed":rp.parent.name,"Q":q,"cat_precision":precision,"cat_recall":recall,"cat_f1":f1,"n":len(qids)})
    runs=pd.DataFrame(out)
    if runs.empty: return runs,pd.DataFrame()
    summary=(runs.groupby(["model","Q"])[["cat_precision","cat_recall","cat_f1"]].agg(["mean","std"]).reset_index())
    summary.columns=["model","Q"]+[f"{a}_{b}" for a,b in summary.columns.tolist()[2:]]
    return runs,summary

def plot_cat_qtier_f1(summary):
    if summary.empty: return
    font=ROOT/"scripts/assets/NotoSansSC-Regular.ttf"
    if font.exists():
        font_manager.fontManager.addfont(str(font)); plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    fig,ax=plt.subplots(figsize=(11,6),dpi=180)
    palette=["#1976D2","#D32F2F","#388E3C","#7B1FA2","#F57C00","#0097A7","#5D4037","#455A64"]
    for color,(model,g) in zip(palette,summary.groupby("model",sort=True)):
        g=g.set_index("Q").reindex(range(1,6)); ax.plot(range(1,6),100*g.cat_f1_mean,marker="o",lw=2,color=color,label=model)
        if "cat_f1_std" in g: ax.fill_between(range(1,6),100*(g.cat_f1_mean-g.cat_f1_std.fillna(0)),100*(g.cat_f1_mean+g.cat_f1_std.fillna(0)),color=color,alpha=.08)
    ax.set_xticks(range(1,6),["Q1\n易","Q2","Q3","Q4","Q5\n难"]); ax.set_ylim(0,105); ax.set_xlabel("五大类等权总分风险层"); ax.set_ylabel("Cat-F1（%）"); ax.set_title("当前猫类 Q1–Q5 分层：冻结模型 Cat-F1",fontweight="bold"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
    fig.tight_layout(); fig.savefig(OUT/"猫类Q1-Q5模型F1折线图.png",bbox_inches="tight"); fig.savefig(OUT/"猫类Q1-Q5模型F1折线图.svg",bbox_inches="tight"); plt.close(fig)

def main():
    rows=read_annotations()
    out=[]; field_rows=[]
    for r in rows:
        vals=compute_group_values(r); rec={"sample_id":r["sample_id"],"bbox_area_ratio":r.get("bbox_area_ratio"),"state_keys":list(encode_states(r).keys())}
        for g in MAJOR:
            lo,mid,hi,nk,nt=summarize(vals[g]); rec[f"{g}_lower"]=20*lo; rec[f"{g}_mid"]=20*mid; rec[f"{g}_upper"]=20*hi; rec[f"{g}_known"]=nk; rec[f"{g}_components"]=nt
            for j,v in enumerate(vals[g]):
                field_rows.append({"sample_id":r["sample_id"],"major_class":g,"field_id":COMPONENT_NAMES[g][j],"raw_value":flat(r["features"].get(COMPONENT_NAMES[g][j],r.get("bbox_area_ratio") if g=="空间几何" else "unknown")),"component_index":j+1,"component_count":len(vals[g]),"normalized_lower":0.0 if v is None else float(v),"normalized_mid":0.5 if v is None else float(v),"normalized_upper":1.0 if v is None else float(v),"contribution_lower":20*(0.0 if v is None else float(v))/len(vals[g]),"contribution_mid":20*(0.5 if v is None else float(v))/len(vals[g]),"contribution_upper":20*(1.0 if v is None else float(v))/len(vals[g])})
        rec["total_lower"]=sum(rec[f"{g}_lower"] for g in MAJOR); rec["total_mid"]=sum(rec[f"{g}_mid"] for g in MAJOR); rec["total_upper"]=sum(rec[f"{g}_upper"] for g in MAJOR)
        rec["all_components_known"]=all(rec[f"{g}_known"]==rec[f"{g}_components"] for g in MAJOR)
        out.append(rec)
    df=assign_tiers(pd.DataFrame(out))
    df.to_csv(OUT/"scores.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(field_rows).to_csv(OUT/"field_scores.csv",index=False,encoding="utf-8-sig")
    # Frozen, quota-balanced greedy selection and random baselines.
    chosen=greedy_select(df,10,2); chosen_ids=df.loc[chosen,"sample_id"].tolist()
    rng=np.random.default_rng(20261017); random_rows=[]
    for rep in range(200):
        ids=[]
        for q in range(1,6): ids.extend(rng.choice(df.loc[df.Q==q].index,2,replace=False).tolist())
        random_rows.append(ids)
    cov_rows=[]
    cov_rows.append({"method":"candidate_pool",**coverage(df.state_keys.tolist())})
    cov_rows.append({"method":"greedy_Q_quota",**coverage(df.loc[chosen,"state_keys"].tolist())})
    rc=[coverage(df.loc[ix,"state_keys"].tolist()) for ix in random_rows]
    cov_rows.append({"method":"random_Q_quota_mean", **{k:float(np.mean([x[k] for x in rc])) for k in ["state_count","occurrence_covered","support_covered","support_rate"]}})
    pd.DataFrame(cov_rows).to_csv(OUT/"coverage_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"rank":i+1,"sample_id":sid,"Q":int(df.loc[df.sample_id==sid,"Q"].iloc[0])} for i,sid in enumerate(chosen_ids)]).to_csv(OUT/"provisional_test_set.csv",index=False,encoding="utf-8-sig")
    # Model predictions are read only after the new selection is frozen.
    tier_map=dict(zip(df.sample_id,df.Q)); ids_by={"candidate_pool":df.sample_id.tolist(),"greedy_Q_quota":chosen_ids}
    ids_by.update({f"random_rep_{i+1:03d}":df.loc[ix,"sample_id"].tolist() for i,ix in enumerate(random_rows)})
    model_df=model_metrics(ids_by,tier_map)
    if not model_df.empty: model_df.to_csv(OUT/"model_tier_recall.csv",index=False,encoding="utf-8-sig")
    recent_runs,recent_summary=recent_five_model_f1()
    if not recent_runs.empty:
        recent_runs.to_csv(OUT/"latest_five_model_f1_runs.csv",index=False,encoding="utf-8-sig")
        recent_summary.to_csv(OUT/"latest_five_model_f1_summary.csv",index=False,encoding="utf-8-sig")
    cat_q_runs,cat_q_summary=cat_qtier_f1(df)
    if not cat_q_runs.empty:
        cat_q_runs.to_csv(OUT/"cat_qtier_f1_runs.csv",index=False,encoding="utf-8-sig")
        cat_q_summary.to_csv(OUT/"cat_qtier_f1_summary.csv",index=False,encoding="utf-8-sig")

    protocol={"protocol_version":"cat-five-major-equal-v1-20261017","scope":"20-target frozen cat pilot, exploratory re-score","major_classes":MAJOR,"class_weight":{g:0.2 for g in MAJOR},"reference_annotation_count":REF_N,
      "human_review_scope":"human review is reserved for fine-grained feature definitions/labels; this run reuses the frozen Qwen3-VL annotations and performs scoring/selection/evaluation automatically.",
      "unknown_handling":"unknown components use lower=0 contribution and upper=1 contribution with fixed component denominators; midpoint is used for Q tiers.",
      "nominal_feature_note":"For nominal fine-grained fields, reference-distribution deviation u=1-(count+1)/(max_count+1) is an exploratory difficulty proxy, not a human-validated perceptual difficulty scale.",
      "mapping":{
       "空间几何":["bbox_area_ratio -> large>=0.10, medium>=0.03, small>=0.01, tiny<0.01"],
       "可见性与完整性":["head_visibility","torso_visibility","limb_visibility","occlusion_level","truncation_level"],
       "外观与形态":["coat_primary_color","coat_pattern","coat_texture","ear_posture","body_shape","eye_state","mouth_state"],
       "朝向与姿态":["head_orientation","body_orientation","tail_posture","posture_action_state"],
       "场景与图像质量":["scene","background_complexity","blur_level","illumination","target_background_contrast","group_context","interaction_context","target_recognizability"]}}
    dump(protocol,OUT/"scoring_protocol.json")
    dump({"selected_ids":chosen_ids,"selection_frozen_before_predictions":True,"random_repetitions":200,"seed":20261017},OUT/"selection_freeze.json")

    # Visualization
    font=ROOT/"scripts/assets/NotoSansSC-Regular.ttf"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        font_name=font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams["font.family"]=font_name
    plt.rcParams["axes.unicode_minus"]=False
    fig=plt.figure(figsize=(16,10),dpi=180); gs=fig.add_gridspec(2,2,wspace=.25,hspace=.32)
    ax=fig.add_subplot(gs[0,0]); plotdf=df.sort_values("total_mid").reset_index(drop=True); x=np.arange(len(plotdf)); bottom=np.zeros(len(plotdf))
    for g in MAJOR:
        y=plotdf[f"{g}_mid"].to_numpy(); ax.bar(x,y,bottom=bottom,color=COLORS[g],label=g,width=.78); bottom+=y
    ax.set_title("20个猫目标：五大类等权细粒度评分",fontsize=13,fontweight="bold"); ax.set_ylabel("总分（0–100，越高越难）"); ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in x],fontsize=8); ax.set_ylim(0,100); ax.grid(axis="y",alpha=.25); ax.legend(fontsize=8,ncol=2,loc="upper left")
    ax=fig.add_subplot(gs[0,1]); qmeans=df.groupby("Q").total_mid.mean().reindex(range(1,6)); qlo=df.groupby("Q").total_lower.mean().reindex(range(1,6)); qhi=df.groupby("Q").total_upper.mean().reindex(range(1,6)); ax.errorbar(range(1,6),qmeans,yerr=[qmeans-qlo,qhi-qmeans],fmt="o-",color="#173F5F",lw=2,capsize=4); ax.set_title("Q1–Q5分层（按总分五等分）",fontsize=13,fontweight="bold"); ax.set_xlabel("风险层级"); ax.set_ylabel("平均总分"); ax.set_xticks(range(1,6),["Q1\n易","Q2","Q3","Q4","Q5\n难"]); ax.set_ylim(0,105); ax.grid(alpha=.25)
    ax=fig.add_subplot(gs[1,0]); cov=pd.DataFrame(cov_rows).set_index("method"); labels=["候选池20","贪心选样10","随机配额10\n均值"]; occ=[cov.loc[k,"occurrence_covered"] for k in ["candidate_pool","greedy_Q_quota","random_Q_quota_mean"]]; sup=[cov.loc[k,"support_covered"] for k in ["candidate_pool","greedy_Q_quota","random_Q_quota_mean"]]; xx=np.arange(3); ax.bar(xx-.18,occ,.36,label="出现覆盖",color="#56B4E9"); ax.bar(xx+.18,sup,.36,label="支持覆盖 (≥2)",color="#009E73"); ax.set_xticks(xx,labels); ax.set_title("细粒度状态覆盖",fontsize=13,fontweight="bold"); ax.set_ylabel("状态数"); ax.legend(); ax.grid(axis="y",alpha=.25)
    ax=fig.add_subplot(gs[1,1]);
    if not model_df.empty:
        mm=model_df.groupby(["method","Q"]).recall.mean().unstack(0); mm=mm.reindex(index=range(1,6));
        for method in [m for m in ["candidate_pool","greedy_Q_quota"] if m in mm.columns]: ax.plot(mm.index,mm[method],marker="o",lw=2,label=method)
        random_cols=[m for m in mm.columns if m.startswith("random_rep_")]
        if random_cols: ax.plot(mm.index,mm[random_cols].mean(axis=1),marker="o",lw=2,ls="--",label="random_Q_quota_mean")
        ax.set_ylim(0,1.05); ax.set_xticks(range(1,6)); ax.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"]); ax.set_ylabel("猫类召回率"); ax.legend(fontsize=8)
    ax.set_title("冻结分类器的分层召回（描述性）",fontsize=13,fontweight="bold"); ax.grid(alpha=.25)
    fig.suptitle("猫类：最新五大类等权细粒度评分与覆盖选样实验（探索性）",fontsize=17,fontweight="bold",color="#173F5F")
    fig.savefig(OUT/"猫类五大类等权细粒度实验.png",bbox_inches="tight"); fig.savefig(OUT/"猫类五大类等权细粒度实验.svg",bbox_inches="tight")
    plt.close(fig)
    plot_recent_five_f1(recent_summary)
    plot_cat_qtier_f1(cat_q_summary)

    cand_recall=float(model_df.loc[model_df.method=="candidate_pool","recall"].mean()) if not model_df.empty else float("nan")
    greedy_recall=float(model_df.loc[model_df.method=="greedy_Q_quota","recall"].mean()) if not model_df.empty else float("nan")
    rand_recall=float(model_df.loc[model_df.method.str.startswith("random_rep_"),"recall"].mean()) if not model_df.empty else float("nan")
    test_f1_lines=[]
    if not recent_summary.empty:
        t=recent_summary[recent_summary.pool=="test"]
        for _,r0 in t.sort_values("model").iterrows(): test_f1_lines.append(f"| {r0['model']} | {100*r0['macro_f1_mean']:.2f} ± {100*r0['macro_f1_std']:.2f} | {100*r0['cat_f1_mean']:.2f} ± {100*r0['cat_f1_std']:.2f} |")
    test_f1_table="\n".join(test_f1_lines) if test_f1_lines else "（未找到最新五模型结果）"
    report=f"""# 猫类五大类等权细粒度实验（2026-10-17）\n\n本目录是对历史猫类 pilot 的**独立重评分版本**，不覆盖旧的“五维风险”结果。按最新执行口径，五大类各占 20 分：{'、'.join(MAJOR)}。\n\n## 数据与执行\n\n- 输入：20 个已冻结猫目标、现有 Qwen3-VL 细粒度标注；没有重新调用 VLM。\n- 人工协同范围：人工只负责细粒度字段定义/标签审查；本次分数、五等分、配额贪心选样和模型评价自动完成。\n- 五等分：按总分中点从低到高排序，20 个目标每层 4 个；贪心测试集每层配额 2 个。\n- 未知值：保留下界/上界；图中使用中点。\n- 名义特征：外观、场景、姿态中的名义字段使用 1053 个猫标注的参考频率偏离作为探索性代理，因此不能直接解释为已被人工验证的“客观难度”。\n\n## 五大类结果\n\n- 候选池平均总分：{df.total_mid.mean():.2f}；范围 {df.total_mid.min():.2f}–{df.total_mid.max():.2f}。\n- Q1–Q5平均总分：{'；'.join(f'Q{q}={df.loc[df.Q==q,"total_mid"].mean():.2f}' for q in range(1,6))}。\n- 贪心选样：{'、'.join(chosen_ids)}。\n- 冻结分类器平均猫类召回：候选池 {cand_recall:.3f}，贪心配额 {greedy_recall:.3f}，随机配额均值 {rand_recall:.3f}。\n\n## Q1–Q5 模型 F1 折线图\n\n当前猫类 20 个目标的 sample ID 可以与 `sixclass_dual_vlm_replaced_20261004` 的 24 个冻结模型运行直接对齐，因此这里按每个 Q 层计算 Cat-F1（一对其余类别），每条线代表一个模型架构，阴影表示 3 个 seed 的标准差。\n\n- 数据：`cat_qtier_f1_summary.csv`。\n- 图：`猫类Q1-Q5模型F1折线图.png` / `.svg`。\n- 该图报告的是 Cat-F1；20 个样本全为猫，不能把这个单类别子集当作六类别 Macro-F1。\n\n## 最新五模型 F1\n\n以下数值来自 `recent_five_models_20261009` 已冻结的 5 个模型、3 个 seed，在其独立测试池（每类 50 个目标）上的结果。它们不能直接替代本次 20 个 OpenImages 猫目标的 Q 层 F1，因为两者 sample ID 不重合。\n\n| 模型 | 测试池 Macro-F1（%） | 测试池 Cat-F1（%） |\n|---|---:|---:|\n{test_f1_table}\n\nB/C/D 外部池的完整指标见 `latest_five_model_f1_summary.csv`；可视化见 `最新五模型F1.png` / `.svg`。\n\n## 覆盖与限制\n\n覆盖结果见 `coverage_summary.csv`；逐字段贡献见 `field_scores.csv`；分层分类器召回见 `model_tier_recall.csv`。由于本次每个 Q 层只有 4 个候选、每层只选 2 个，分层结果仅作流程联调，不能作为稳定性能结论。\n\n图：`猫类五大类等权细粒度实验.png` / `.svg`。\n"""
    latest_q=OUT/"latest_five_cat_qtier_f1_summary.csv"
    if latest_q.exists():
        report += "\n## 补充：最新五模型直接推理的 Q1–Q5 F1\n\n已使用 `recent_five_models_20261009` 的 5 个模型、3 个 seed 在当前 20 个猫目标上直接推理，生成了真正对应当前 Q1–Q5 的折线图。结果数据见 `latest_five_cat_qtier_f1_summary.csv`，图见 `最新五模型猫类Q1-Q5_F1折线图.png` / `.svg`。由于每层只有 4 个目标，该结果仍是小样本流程验证。\n"
    report += "\n风险校准诊断见 `risk_calibration_diagnostic.md`。\n"
    (OUT/"report.md").write_text(report,encoding="utf-8")
    dump({"status":"completed","input_targets":len(df),"major_classes":MAJOR,"outputs":["scores.csv","field_scores.csv","coverage_summary.csv","provisional_test_set.csv","model_tier_recall.csv","cat_qtier_f1_runs.csv","cat_qtier_f1_summary.csv","latest_five_model_f1_runs.csv","latest_five_model_f1_summary.csv","latest_five_cat_qtier_f1_runs.csv","latest_five_cat_qtier_f1_summary.csv","猫类五大类等权细粒度实验.png","猫类五大类等权细粒度实验.svg","猫类Q1-Q5模型F1折线图.png","猫类Q1-Q5模型F1折线图.svg","最新五模型F1.png","最新五模型F1.svg","最新五模型猫类Q1-Q5_F1折线图.png","最新五模型猫类Q1-Q5_F1折线图.svg"],"reference_annotations":REF_N},OUT/"verification.json")
    print(json.dumps({"out":str(OUT),"n":len(df),"chosen":chosen_ids,"coverage":cov_rows,"model_rows":len(model_df)},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()

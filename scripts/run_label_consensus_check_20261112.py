"""Fixed-selection diagnostic sensitivity on two-VLM state-membership agreement."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import run_label_factorial_20261112 as factorial

study, cc, old = factorial.study, factorial.cc, factorial.old
OUT = study.ROOT / 'artifacts/label_consensus_check_20261112'


def dump(name, value):
    cc.dump(OUT / name, value)


def write(name, rows):
    cc.write(OUT / name, rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'completion.json').exists(), 'Completed output is immutable'
    cc.verify_freeze(factorial.OUT / 'final_outputs_freeze.json')
    cc.verify_freeze(factorial.OUT / 'prediction_freeze.json')
    cc.freeze(OUT / 'protocol.json', dict(
        hypothesis='P20-vs-R and P20-vs-Greedy gains persist after excluding disputed state membership from both diagnostic arms',
        scope='Exploratory follow-up to completed 2x2; historical caches, no semantic ground truth',
        selections='All 9600 frozen factorial selections unchanged; duplicate AA threshold regimes retained and identified',
        positive='state presence in both A and B',
        negative='known same-field complement in both A and B; unknown never counts as negative',
        excluded='Targets with conflicting or unknown membership for a given state',
        distinction='Agreement on each state membership; not necessarily exact agreement of entire multi-valued field',
        diagnostic='At least20 positive and20 negative distinct source images; three-seed mean recall gap <=-10pp in B and historical D',
        outcomes='Confirmed conditions per start; P20-R and P20-Greedy paired differences; reference and selected eligibility attrition',
        success='Report direction and paired-start signs in all8 scenarios without tuning thresholds or dropping low-support cases',
        stop='Complete all frozen selections and report unfavorable results; no new VLM calls, prompt tuning, or reselection',
        caveats='Agreement can share errors; exclusion changes population, risk mixture and support; no accuracy or causal claims'))
    cc.freeze(OUT / 'input_freeze.json', dict(files={str(p): study.sha(p) for p in
        [Path(__file__), Path(factorial.__file__), Path(factorial.prior.__file__), factorial.OUT / 'selections.npz',
         factorial.OUT / 'selection_runs.csv', factorial.OUT / 'summary.csv', factorial.OUT / 'completion.json',
         factorial.OUT / 'final_outputs_freeze.json', factorial.OUT / 'prediction_freeze.json',
         *list(factorial.PREV.glob('*_matrix.npz'))]}))
    (OUT / 'research_log.md').write_text('# 共识状态检查\n\n在已知2×2结果后提出的探索性假设。先冻结规则，再计算所有场景；不挑起点，不重新选样，不把一致性当准确率。\n')
    (OUT / 'paper_brief.md').write_text('保持原论文主线，仅检查自动标签分歧排除后的历史诊断稳定性。\n')
    rows, risks, matrices = factorial.contexts()
    raw, inventory, support, hashes = {}, [], [], {}
    for phase in rows:
        pred, hh = cc.old.predictions(phase, rows[phase]); hashes.update(hh)
        offset = {r['sample_id']: i for i, r in enumerate(rows[phase])}
        for ci, c in enumerate(factorial.CLASSES):
            states = risks[phase, 'A', c]['states']
            a, b = [matrices[phase, source, c] for source in ['A','B']]
            ac, bc = [factorial.prior.value_known_mask(x, states) for x in [a,b]]
            x, control = a & b, ac & bc
            assert not (x & control).any()
            assert np.all(x <= a) and np.all(x <= b) and np.all(control <= ac) and np.all(control <= bc)
            np.savez_compressed(OUT / f'{phase}_{c}_consensus.npz', positive=x, negative=control,
                                sample_ids=risks[phase, 'A', c]['ids'])
            pos = [offset[s] for s in risks[phase,'A',c]['ids']]
            stats = {}
            for model in old.MODELS:
                correct = (pred[model][:,pos] == ci).mean(0)
                stats[model] = correct, factorial.prior.reference_stats(x,control,correct)
                for source, xx, yy in [('A',a,ac),('B',b,bc),('consensus',x,control)]:
                    n1,n0,gap,eligible,weak = factorial.prior.reference_stats(xx,yy,correct)
                    for j,(namespace,field,value) in enumerate(states):
                        inventory.append(dict(phase=phase,class_name=c,model=model,source=source,
                            state_index=j,namespace=namespace,field=field,value=value,
                            positive_n=int(n1[j]),negative_n=int(n0[j]),gap_pp=float(gap[j]),
                            eligible=bool(eligible[j]),weak=bool(weak[j])))
            support.append(dict(phase=phase,class_name=c,states=len(states),
                A_positive_memberships=int(a.sum()),B_positive_memberships=int(b.sum()),consensus_positive_memberships=int(x.sum()),
                A_negative_memberships=int(ac.sum()),B_negative_memberships=int(bc.sum()),consensus_negative_memberships=int(control.sum()),
                excluded_memberships=int((~(x|control)).sum()),total_target_state_pairs=x.size))
            raw[phase,c] = x,control,stats
    cc.freeze(OUT / 'prediction_freeze.json', dict(files=hashes))
    z = np.load(factorial.OUT / 'selections.npz',allow_pickle=False)
    runs = study.read(factorial.OUT / 'selection_runs.csv')
    diagnostic,coverage = [],[]
    independent=0
    for r in runs:
        risk,cover,mode,c,method,seed = [r[f] for f in ['risk_source','coverage_source','threshold_mode','class_name','method','selection_seed']]
        selected = z[factorial.key(risk,cover,mode,c,method,seed)]
        base = dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,class_name=c,method=method,selection_seed=int(seed))
        x,control,stats = raw['B',c]
        a,b = x[selected],control[selected]
        counts=a.sum(0)
        coverage.append(dict(**base,consensus_support_completion_pct=100*float(np.minimum(counts,20).sum()/(20*len(counts)))))
        for model in old.MODELS:
            correct = stats[model][0][selected]
            n1,n0,gap,eligible,weak = factorial.prior.reference_stats(a,b,correct)
            d = raw['D',c][2][model][1]
            confirmed=weak&d[4]
            values = [int(eligible.sum()),int(weak.sum()),int((weak&d[3]).sum()),int(confirmed.sum())]
            diagnostic.append(dict(**base,model=model,eligible=values[0],proposed=values[1],
                reference_evaluable=values[2],confirmed=values[3],error_pct=100*float(1-correct.mean())))
            if int(seed)==factorial.SEEDS[0]:
                independent_values=np.zeros(4,int)
                dx,dc,_=raw['D',c]
                dcorrect=raw['D',c][2][model][0]
                for j in range(a.shape[1]):
                    def scalar(left,right,acc):
                        eligible=int(left.sum())>=20 and int(right.sum())>=20
                        weak=eligible and 100*(float(acc[left].mean())-float(acc[right].mean())) <= -10+1e-12
                        return eligible,weak
                    e,w=scalar(a[:,j],b[:,j],correct)
                    de,dw=scalar(dx[:,j],dc[:,j],dcorrect)
                    independent_values += [int(e),int(w),int(w and de),int(w and dw)]
                assert list(independent_values)==values
                independent+=1
    write('diagnostic.csv',diagnostic);write('coverage.csv',coverage)
    write('condition_inventory.csv',inventory);write('membership_support.csv',support)
    groups,paired=defaultdict(list),defaultdict(lambda:defaultdict(int))
    for r in diagnostic:
        k=tuple(r[f] for f in ['risk_source','coverage_source','threshold_mode','method'])
        groups[k].append(r);paired[k[:-1]][r['selection_seed'],r['method']] += r['confirmed']
    prior_summary={tuple(r[f] for f in ['risk_source','coverage_source','threshold_mode','evaluator','reference','method']):r
                   for r in study.read(factorial.OUT/'summary.csv')}
    summary=[]
    for k,rr in groups.items():
        risk,cover,mode,method=k
        assert len(rr)==1500
        baseline=prior_summary[risk,cover,mode,'A','A',method]
        assert abs(float(baseline['error_pct'])-np.mean([r['error_pct'] for r in rr]))<1e-9
        summary.append(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,method=method,
            eligible_per_start=sum(r['eligible'] for r in rr)/50,
            proposed_per_start=sum(r['proposed'] for r in rr)/50,
            confirmed_per_start=sum(r['confirmed'] for r in rr)/50,
            A_labels_confirmed_per_start=float(baseline['confirmed_per_start']),
            B_labels_confirmed_per_start=float(prior_summary[risk,cover,mode,'B','B',method]['confirmed_per_start']),
            error_pct=float(baseline['error_pct'])))
    diffs=[]
    for k,counts in paired.items():
        for method,comparator in [('P20','R'),('P20','Greedy20'),('Greedy20','R')]:
            diff=np.array([counts[seed,method]-counts[seed,comparator] for seed in factorial.SEEDS])
            diffs.append(dict(zip(['risk_source','coverage_source','threshold_mode'],k),method=method,comparator=comparator,
                mean_confirmed_difference=float(diff.mean()),positive_starts=int((diff>0).sum()),
                ties=int((diff==0).sum()),negative_starts=int((diff<0).sum())))
    write('summary.csv',summary);write('paired_differences.csv',diffs)
    validation=dict(status='pass',selected_sets=len(runs),diagnostic_rows=len(diagnostic),
        scalar_diagnostic_rows_independently_verified=independent,no_reselection=True,
        unknowns_never_negative=True,consensus_memberships_are_subsets=True,original_errors_reproduced=True)
    dump('validation.json',validation)
    report(summary,diffs,inventory,support)
    cc.verify_freeze(OUT/'input_freeze.json');cc.verify_freeze(OUT/'prediction_freeze.json')
    cc.log(OUT,'completed',diagnostic_rows=len(diagnostic),scalar_audit_rows=independent)
    outputs={str(p):study.sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['completion.json','final_outputs_freeze.json']}
    doc=study.ROOT/'docs/实验方案与结果/双VLM共识状态诊断复验_20261112.md'
    outputs[str(doc)]=study.sha(doc)
    cc.freeze(OUT/'final_outputs_freeze.json',dict(files=outputs))
    dump('completion.json',dict(status='complete',time=cc.old.now(),new_VLM_calls=0,
        final_outputs_freeze_sha256=study.sha(OUT/'final_outputs_freeze.json')))


def report(summary,diffs,inventory,support):
    idx={tuple(r[f] for f in ['risk_source','coverage_source','threshold_mode','method']):r for r in summary}
    p_r=[r for r in diffs if r['method']=='P20' and r['comparator']=='R']
    p_g=[r for r in diffs if r['method']=='P20' and r['comparator']=='Greedy20']
    counts=defaultdict(lambda:[0,0])
    for r in inventory:
        counts[r['phase'],r['source']][0]+=int(r['eligible']);counts[r['phase'],r['source']][1]+=int(r['weak'])
    lines=['# 双VLM共识状态的固定选集复验','',
        '本轮是看到2×2结果后增加的探索性检查。所有选集固定，状态阳性取A∩B，阴性取两者已知同字段补集的交集；分歧及未知目标在该状态下被排除。每侧≥20目标、三预测种子平均召回差≤−10pp的门槛保持不变。共识只代表状态成员关系一致，不证明语义正确；排除操作改变目标群体、风险构成和支持量。','',
        '共9600组选集，48000行诊断；首个预定起点的960行另以逐状态均值核算，原实际错误率完全复现。没有重选、训练、VLM新调用或人工核验。','',
        '| 风险/覆盖/门槛 | P20共识确认数 | Greedy20共识确认数 | R共识确认数 | P−G | P−R | P−G正/平/负起点 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for pg in p_g:
        risk,cover,mode=[pg[f] for f in ['risk_source','coverage_source','threshold_mode']]
        p,g,r=[idx[risk,cover,mode,m] for m in ['P20','Greedy20','R']]
        lines.append(f'| {risk}/{cover}/{mode} | {p["confirmed_per_start"]:.2f} | {g["confirmed_per_start"]:.2f} | {r["confirmed_per_start"]:.2f} | {p["confirmed_per_start"]-g["confirmed_per_start"]:+.2f} | {p["confirmed_per_start"]-r["confirmed_per_start"]:+.2f} | {pg["positive_starts"]}/{pg["ties"]}/{pg["negative_starts"]} |')
    positive_r=sum(r['mean_confirmed_difference']>0 for r in p_r)
    positive_g=sum(r['mean_confirmed_difference']>0 for r in p_g)
    lines+=['',f'8个场景中，P20相对R的均值为正：{positive_r}/8；相对Greedy20：{positive_g}/8。AA两个门槛重复、其余场景高度相关，不能当作独立重复或显著性证据。','',
            '| 数据池 | 标签 | 可评价条件（6类×5分类器汇总） | 弱条件 |','|---|---|---:|---:|']
    for phase in ['B','D']:
        for source in ['A','B','consensus']:
            n,w=counts[phase,source];lines.append(f'| {phase} | {source} | {n} | {w} |')
    lines+=['','完整候选池的可评价条件不等于选后可评价条件；逐状态支持量见condition_inventory.csv，选后数量见summary.csv。确认条件数先跨6类×5分类器求和，再对50起点平均。不同VLM及共识有不同的可评价条件集合，绝对确认数不能直接当作方法质量或准确率。','',
            '原协议给出的每个选集在两套标签下的结果均保留；本轮不替代不利结果。即便共识检查中的方向一致，也只能支持此共识子群上的历史收益，不能恢复跨VLM稳定优于Greedy20或语义正确的强结论。','',
            '复现：.venv/bin/python scripts/run_label_consensus_check_20261112.py；完成目录不可覆盖。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text)
    (study.ROOT/'docs/实验方案与结果/双VLM共识状态诊断复验_20261112.md').write_text(text)
    dump('decision.json',dict(P20_above_R_scenarios=positive_r,P20_above_Greedy20_scenarios=positive_g,total_scenarios=8,
        stable_P20_superiority_supported=False,semantic_accuracy_established=False))
    (OUT/'claim_evidence.md').write_text('共识子群历史收益方向见paired_differences.csv；支持量损失见condition_inventory.csv和membership_support.csv。不能推断标签准确率、因果根因或新数据表现。\n')
    (OUT/'review.md').write_text('引用完整性：无新增外部引用。结果追溯：所有选集复用，960行诊断独立标量重算，原错误率复现。主张边界：仅共识成员关系筛选后的历史条件数，排除造成的群体/支持变化明确列出；不宣称真值。\n')


if __name__=='__main__':
    run()

"""Review and report the frozen support mechanism experiment without retuning."""
from pathlib import Path
import numpy as np
import run_label_support_diagnosis_20261112 as study_run

OUT, cc, study = study_run.OUT, study_run.cc, study_run.study


def run():
    assert not (OUT / 'completion.json').exists(), 'Completed experiment cannot be overwritten'
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'prediction_freeze.json')
    rows = study.read(OUT / 'summary.csv')
    idx = {tuple(r[f] for f in study_run.GROUP + ['method', 'comparator']): r for r in rows}
    methods = {tuple(r[f] for f in study_run.GROUP + ['method']): r for r in study.read(OUT / 'method_summary.csv')}
    validation = __import__('json').loads((OUT / 'validation.json').read_text())
    assert validation['status'] == 'pass'
    def at(risk, cover, mode, ev, ref, method='P20', comparator='R'):
        return idx[risk, cover, mode, ev, ref, method, comparator]
    def number(r, f):
        return float(r[f])
    contrasts = {}
    for method, comparator in study_run.COMPARISONS:
        rr = [r for r in rows if (r['method'], r['comparator']) == (method, comparator)]
        assert len(rr) == 40
        contrasts[method + '_minus_' + comparator] = {f + '_positive_scenarios': sum(number(r, f) > 1e-10 for r in rr)
            for f in ['total_difference', 'common_gap_net', 'same_label_common_difference', 'all_label_common_difference']}
        contrasts[method + '_minus_' + comparator]['scenarios'] = len(rr)
    a = at('A', 'A', 'original', 'A', 'A')
    b = at('B', 'B', 'recalibrated', 'B', 'B')
    c = at('B', 'B', 'recalibrated', 'C', 'C')
    lines = ['# 诊断收益的支持量分解与共同可评价条件复验', '',
        '本轮验证：确认条件增益来自更多状态达到每侧20样本的可评价门槛，还是在双方原本都可评价的状态上仍有增益？这是看到跨VLM与共识结果后追加的探索性分析，原A标签已有类似分解，本轮复现它并扩展到所有标签口径及P20与Greedy20的比较。', '',
        '## 方法与固定因素', '',
        '复用8个风险/覆盖/门槛场景、50起点、6类的R/Greedy20/P20，共7200个冻结选集；原AA两种门槛相同，重复列示。标签A=Qwen、B=SenseNova、C=两者状态成员关系共识。保留A/A、A/B、B/A、B/B、C/C五种发现/参考口径。没有重选、训练、新VLM调用或人工核验。', '',
        '每状态阳性与已知同字段阴性均≥20；召回差使用3预测种子的平均正确率，≤−10pp判为弱条件，再与历史D参考弱条件相交。固定原阈值，不按结果调参。', '',
        '将方法M相对比较方法B的确认条件净差精确分成四项：新增且B不可评价 + 新增且B可评价 − 丢失且M不可评价 − 丢失且M可评价。前后对应项合并得到“支持量净贡献”和“共同可评价上的差异”；后者只表示不同选中样本导致的弱点阈值判定差异，不能称为真实诊断能力提升或因果证据。', '',
        '额外比较两种集合：①当前标签口径下三方法和D参考都可评价；②所有A/B/C标签下三方法和D参考都可评价。这些集合只依赖支持量，不按召回差或确认结果挑状态。由于C的阳性/阴性都是A/B子集，②等价于C下三方法及D都可评价，程序逐项验证。共同状态ID并不意味着相同目标成员、相同每侧样本数或相同实际错误率。', '',
        '## 关键结果', '',
        f'原AA选集、原数值门槛、A/A评价：P20相对R总差 {number(a,"total_difference"):+.2f} = 支持量净贡献 {number(a,"eligibility_net"):+.2f} + 双方共同可评价上的差异 {number(a,"common_gap_net"):+.2f}。', '',
        f'BB选集、重标定门槛：B/B评价下总差 {number(b,"total_difference"):+.2f} = {number(b,"eligibility_net"):+.2f} + {number(b,"common_gap_net"):+.2f}；共识C/C评价下总差 {number(c,"total_difference"):+.2f} = {number(c,"eligibility_net"):+.2f} + {number(c,"common_gap_net"):+.2f}。', '',
        '| 比较 | 原总差为正的口径 | 双方共同可评价差为正 | 当前标签三方法共同差为正 | 所有标签三方法共同差为正 |',
        '|---|---:|---:|---:|---:|']
    for name, counts in contrasts.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{counts[f + "_positive_scenarios"]}/40' for f in
            ['total_difference', 'common_gap_net', 'same_label_common_difference', 'all_label_common_difference']) + ' |')
    lines += ['', '这里的40种口径高度相关，包含重复AA门槛、共享样本与模型，不能当作40个独立实验、胜率估计或显著性检验。', '',
        '## P20相对R：完整分解（同源评价与共识）', '',
        '| 风险/覆盖/门槛 | 评价 | 总差 | 支持量净贡献 | 双方共同差 | 当前标签三方法共同差 | 所有标签三方法共同差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for risk in ['A', 'B']:
        for cover in ['A', 'B']:
            for mode in study_run.previous.MODES:
                for ev in ['A', 'B', 'C']:
                    r = at(risk, cover, mode, ev, ev)
                    lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | ' + ' | '.join(f'{number(r,f):+.2f}' for f in
                        ['total_difference','eligibility_net','common_gap_net','same_label_common_difference','all_label_common_difference']) + ' |')
    lines += ['', '## P20相对Greedy20：共同可评价条件上的差异', '',
        '| 风险/覆盖/门槛 | 评价 | 总差 | 支持量净贡献 | 双方共同差 | 当前标签三方法共同差 | 所有标签三方法共同差 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for risk in ['A','B']:
        for cover in ['A','B']:
            for mode in study_run.previous.MODES:
                for ev in ['A','B','C']:
                    r=at(risk,cover,mode,ev,ev,'P20','Greedy20')
                    lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | ' + ' | '.join(f'{number(r,f):+.2f}' for f in
                        ['total_difference','eligibility_net','common_gap_net','same_label_common_difference','all_label_common_difference']) + ' |')
    lines += ['', '## 支持量保留与配对起点', '',
        '| 风险/覆盖/门槛 | 评价 | R可评价条件 | 三方法当前标签共同条件 | 三方法所有标签共同条件 | P−R双方共同差：正/平/负起点 |',
        '|---|---|---:|---:|---:|---|']
    for risk in ['A','B']:
        for cover in ['A','B']:
            for mode in study_run.previous.MODES:
                for ev in ['A','B','C']:
                    r=at(risk,cover,mode,ev,ev)
                    m=methods[risk,cover,mode,ev,ev,'R']
                    signs='/'.join(r['common_gap_net_'+f] for f in ['positive_starts','ties','negative_starts'])
                    lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | {float(m["eligible_per_start"]):.2f} | {float(m["same_label_common_n_per_start"]):.2f} | {float(m["all_label_common_n_per_start"]):.2f} | {signs} |')
    lines += ['', '上述条件数先跨6类×5分类器求和，再对50起点平均，不是独立根因数；可评价条件的定义没有使用模型输出，同一标签状态的支持量因此在分类器间重复。R可评价数未额外要求D可评价，共同集合则要求D可评价。全部交叉标签、配对起点与按字段四项分解保存在CSV，零支持与负结果均保留。', '',
        '## 自动核验与结论边界', '',
        f'复现旧诊断 {validation["original_diagnostic_rows_reproduced"]} 行；四项分解恒等式检查 {validation["decomposition_identities"]} 行；首个预定起点 {validation["independent_set_partition_rows"]} 行另以集合运算核算；原A历史逐起点机制记录 {validation["original_A_mechanism_rows_reproduced"]} 行全部复现；字段贡献加总严格守恒。', '',
        '若净收益主要来自支持量，应表述为提高可评价性，而不能直接把确认条件增多解释为更多真实根因。在共同可评价集合上仍为正的部分，也受不同样本、实际错误率、标签语义与历史D重复使用的影响。这一分解没有消除这些因素。', '',
        '当前结果继续只支持按具体标签、风险构造、门槛和共同集合限定的历史观察。稳定优于Greedy20、标签准确率或新数据泛化的主张不能由本轮恢复。未因结果更换支持/弱点阈值、去掉不利字段或改动论文中心主张。', '',
        '复现：依次运行 .venv/bin/python scripts/run_label_support_diagnosis_20261112.py prepare、同脚本 run，再运行 .venv/bin/python scripts/report_label_support_diagnosis_20261112.py。完成目录不可覆盖。']
    text='\n'.join(lines)+'\n'
    (OUT/'report_zh.md').write_text(text)
    doc=study_run.ROOT/'docs/实验方案与结果/诊断收益支持量分解与共同可评价复验_20261112.md'
    doc.write_text(text)
    study_run.dump('decision.json', dict(contrasts=contrasts, stable_P20_superiority_supported=False,
        scope='Historical support/gap accounting; no semantic truth, causal mechanism or fresh-data validation'))
    (OUT/'claim_evidence.md').write_text(
        '# 主张与证据\n\n四项净收益分解：mechanism.csv、mechanism_per_start.csv、summary.csv。\n\n'
        '共同条件集合的计数及均值：diagnostic.csv、method_summary.csv、summary.csv；按字段分解：by_field.csv。\n\n'
        '所有数值仅算术分解与历史条件计数。共同ID不等于相同目标成员或固定错误率；支持量收益不能解释成真实根因增多。\n')
    (OUT/'review.md').write_text(
        '# 自动核验\n\n引用完整性：本轮没有新增外部引文事实。\n\n结果可追溯：通过。原诊断完整复现，原A逐起点机制复现；全部四项恒等式、首起点独立集合算法及字段守恒通过。\n\n'
        '主张边界：通过。探索性历史分析、相关口径、支持集合/目标成员变化、非准确率/非根因/非因果/非新数据均明确。\n')
    cc.log(OUT,'report_complete',comparisons=len(rows))
    outputs={str(p):study.sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['final_outputs_freeze.json','completion.json']}
    outputs[str(doc)]=study.sha(doc)
    cc.freeze(OUT/'final_outputs_freeze.json',dict(files=outputs))
    study_run.dump('completion.json',dict(status='complete',time=cc.old.now(),new_VLM_calls=0,new_classifier_inference=False,
        output_freeze_sha256=study.sha(OUT/'final_outputs_freeze.json'),report=str(doc)))


if __name__=='__main__':
    run()

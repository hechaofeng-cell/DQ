"""Export the actual automatic-only follow-up results with their limitations."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_validation_common_20261018 import *

DEST=OUT/'automatic_robustness'
METHODS=['expanded_greedy','shared_greedy','static_rarity','vit_kcenter','same_cell_random']
NAMES=dict(zip(METHODS,['扩展贪心','共享贪心','静态稀缺度','ViT k-center','同单元随机']))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    assert json.loads((DEST/'independent_audit.json').read_text())['status']=='pass'
    summaries=read(DEST/'summary.csv');unique=read(DEST/'deduplicated_conditions.csv');checks=read(DEST/'condition_checks.csv')
    def get(m,scope='all'):return next(r for r in summaries if r['method']==m and r['scope']==scope)
    def number(r,k):return f'{float(r[k]):.2f}' if r['method']=='same_cell_random' else f'{float(r[k]):.0f}'
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    labels=['Expanded greedy','Shared greedy','Static rarity','ViT k-center','Matched random'];colors=['#0072B2','#999999','#D55E00','#009E73','#CC79A7']
    for ax,scope,title in zip(axes,['all','specific'],['All-state discovery','Specific-state discovery']):
        for m,label,color in zip(METHODS,labels,colors):
            ax.plot([0,5,10,20],[float(get(m,scope)[f'confirmed_{t}pp']) for t in [0,5,10,20]],marker='o',lw=1.5,label=label,color=color)
        ax.set_title(title);ax.set_xticks([0,5,10,20]);ax.set_xlabel('Required external recall deficit (pp)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Replicated conditions (support >= 20)');axes[0].set_ylim(0,52)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.07));fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(DEST/'severity_thresholds.png',dpi=180,bbox_inches='tight');fig.savefig(DEST/'severity_thresholds.svg',bbox_inches='tight');plt.close(fig)
    parts=[]
    def p(s):parts.append(s)
    p('# 全自动诊断稳健性实验结果\n\n生成时间：'+now())
    p('按照用户最新要求，取消等待人工标注的环节，直接完成本轮自动实验。人工盲审材料保留为历史产物，不是后续自动运行的前置条件。本报告不声称测得人工语义准确率。')
    p('本轮在原有结果基础上新增四项分析：更严格的外部效应阈值、三个训练种子的方向一致性、风险分层内的可比性及调整分析、跨模型重复条件去重。沿用已冻结的 800 个 Objects365 目标、状态词典、候选弱项和 5 模型 × 3 种子预测；本轮没有重新训练模型，也没有新生成标签。')
    p('这是**看到前轮结果后的探索性补充分析**，不是前瞻验证。新协议先于本轮计算保存，原主结果不被覆盖，所有预设阈值和失败的可比性检查均保留。')
    p('**主要结论：**全状态条件要求外部 Recall 至少低 10 pp 时，扩展贪心复现 39 条，k-center 42 条；三个种子分别都达到 10 pp 时，两者均为 34 条。扩展贪心没有稳定领先。仅专属状态的次分析存在一些增量，但去重后也不是最多；覆盖收益与诊断收益仍需分别陈述。')
    p('## 1. 更严格的外部复现标准')
    p('保持发现条件不变：发现池支持 ≥20、相对该选集整类 Recall 低至少 10 pp、最多取 3 个。只改变外部判定：外部支持仍需 ≥20，三个种子的平均条件 Recall 相对外部整类的下降分别要求 >0、≥5、≥10、≥20 pp。阈值 0 表示严格负方向，其余阈值包含等号。')
    for scope,title in [('all','全部状态（沿用主分析范围）'),('specific','仅专属状态（次分析范围）')]:
        p('### '+title)
        p(table(['方法','提出条件数','方向复现','至少低5pp','至少低10pp','至少低20pp'],[[NAMES[m]]+[number(get(m,scope),k) for k in ['proposed','confirmed_0pp','confirmed_5pp','confirmed_10pp','confirmed_20pp']] for m in METHODS]))
    p('全状态范围下，10 pp 阈值时 k-center 为 42 条、扩展贪心为 39 条；20 pp 阈值时共享贪心 25 条、扩展贪心 24 条、k-center 21 条。排名随阈值变化，不能只选对本方法有利的阈值作为结论。仅专属状态时，10 pp 阈值下扩展贪心 31 条，高于 k-center 的 29 条和稀缺度的 26 条，但这是次分析。')
    p(f'![复现标准收紧后的条件数量]({DEST/"severity_thresholds.png"})')
    p('所有计数是“类别 × 模型 × 条件”实例，存在重复与相关性。随机列为 100 次抽样的计数均值；这些重复不是 100 个新数据集。没有对方法间计数差异作显著性声明。')
    p('## 2. 三个训练种子是否分别复现？')
    p('上一节使用三个种子的平均表现。本节要求每个种子分别满足标准，防止均值掩盖方向不一致；样本、状态、类别基准均与上一节相同。')
    for scope,title in [('all','全部状态'),('specific','仅专属状态')]:
        p('### '+title)
        p(table(['方法','均值方向复现','三个种子均为负','均值至少低10pp','三个种子均至少低10pp'],[[NAMES[m]]+[number(get(m,scope),k) for k in ['confirmed_0pp','all_seeds_negative','confirmed_10pp','all_seeds_at_least_10pp']] for m in METHODS]))
    p('全状态范围，扩展贪心的 44 条均值方向复现条件中，有 40 条在三个种子上均为负；至少低 10 pp 的 39 条中，有 34 条在三个种子上均达到该阈值。仅专属范围的对应数量为 37→35 和 31→25。种子一致性不是标注准确率，也不能把三个种子视为三个独立外部数据集。')
    p('## 3. 风险分层内能否获得足够可比的对照？')
    p('重新使用原 `equal_parts` 五维等权风险规则，先在 1600 个发现池目标上复现原分数，再计算 800 个外部目标分数。尺度、可见性、遮挡、截断、朝向均进入原规则；没有调节风险权重。分层使用风险分数每 10 分一个固定区间，并按是否存在风险区间不确定性进一步分开。这是对聚合代理风险的粗分层，不代表逐项风险因素完全相同。')
    p('每个层中至少需要 5 个标出该状态的目标和 5 个未标出该状态的目标；保留层合计两组均至少 20 个目标，并保留至少 80% 的原状态目标，才允许计算风险调整后差值。调整值按保留状态目标在各层的比例，加权层内“标出组 Recall − 未标出组 Recall”。这一比较对象不同于前文的整类基准，不能直接将两种效应量相减当作混杂效应。')
    p(table(['方法','全部状态：提出','全部状态：可调整','仅专属：提出','仅专属：可调整'],[[NAMES[m],number(get(m),'proposed'),number(get(m),'adjustment_evaluable'),number(get(m,'specific'),'proposed'),number(get(m,'specific'),'adjustment_evaluable')] for m in METHODS]))
    retained=max(float(r['retained_fraction']) for r in checks if r['method']=='expanded_greedy' and r['scope']=='all')
    p(f'**扩展贪心的全状态 59 条和专属 56 条条件均未达到预设可比性要求。** 全状态条件中，满足单层样本要求后能保留的原状态目标比例最高仅为 {retained*100:.2f}%，低于 80%。因此本轮无法报告扩展贪心风险调整后的效果；“可调整数为 0”不是“调整后效果等于 0”，也不是“已证明没有弱点”。没有在看到结果后放宽分层或保留比例。')
    p('少数其他方法的条件可以计算，逐条结果保留在 CSV 中，但各方法可评估子集不同且数量极少，不据此作调整后优劣排名。未标出组包括自动字段未知的情形，不一定等于语义上确实不存在该状态；这也是解释边界。对可评估条件执行了分层、组内的目标 bootstrap 1000 次，区间条件于已保留分层与固定权重，未作多重比较校正。')
    p('## 4. 去掉跨模型重复后，有多少不同状态类型？')
    p('按 `(类别、共享/专属、字段、取值)` 去重；同一状态在多个模型上出现只算一个类型。“至少一个提出该条件的模型复现”是较宽松定义，“所有曾提出该条件的模型均复现”不等于在全部 5 个模型上都复现。仍使用外部支持 ≥20 和负方向标准。')
    for scope,title in [('all','全部状态'),('specific','仅专属状态')]:
        rows=[]
        for m in METHODS:
            rr=[r for r in unique if r['scope']==scope and r['method']==m]
            rows.append([NAMES[m]]+[f'{np.mean([float(r[k]) for r in rr]):.2f}' if m=='same_cell_random' else str(int(rr[0][k])) for k in ['proposed_types','any_model_confirmed_types','all_proposed_models_confirmed_types']])
        p('### '+title);p(table(['方法','提出类型数','至少一个模型复现','所有曾提出模型均复现'],rows))
    p('全状态去重后，扩展贪心复现 18 种，稀缺度 22 种，k-center 18 种；仅专属去重后，至少一个模型复现的类型数分别为 14、15、13。因此“复现条件实例更多”不能直接写成“发现的不同弱点类型更多”。')
    p('## 5. 本轮完成状态与论文表述')
    p('全部计划内自动分析已运行并审计，人工标注不再作为本轮任务的前置条件。结论限定在冻结的自动状态标签、四类、固定外部池和五个已有模型上。此前覆盖实验仍支持相对随机的支持完成度收益，但与稀缺度的差距很小；本轮进一步显示诊断排名依赖判定阈值、是否要求种子一致和是否去重。')
    p('建议论文表述为：**在固定代理风险构成下，视觉状态引导选样改善了测试集合的覆盖与支持；它能产生部分跨数据集复现的弱项条件，但不保证优于所有简单基线。现有外部样本尚不足以支持大多数条件的风险分层内调整推断。** 不声称真实状态准确率已验证、不声称覆盖提升导致诊断提升，也不声称调整后效果已成立。')
    p('若继续自动扩展，优先补足预设风险分层内的对照目标，并在新样本预测前固定条件、分层和停止标准；仅重复当前 800 个目标或更换有利阈值不能解决可比性不足。本轮不追加无预设范围的大规模数据下载或训练。')
    p('## 6. 证据与复现')
    for filename,label in [('protocol.json','本轮协议'),('summary.csv','汇总结果'),('condition_checks.csv','逐条件结果'),('group_checks.csv','类别×模型分组结果'),('deduplicated_conditions.csv','去重结果'),('external_risk_scores.csv','外部风险分数'),('audit.json','运行审计'),('independent_audit.json','独立重算审计')]:p(f'- [{label}]({DEST/filename})')
    p('原分数复现 1600 个目标；440 条不同“类别 × 模型 × 状态”的可比性与加权差值独立核对，10 行方法汇总重新计算。前轮负方向计数、预测哈希及冻结条件均复现通过。首次运行的汇总键错误及修复记录已保存，未改变协议或阈值。')
    p('```bash\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_automatic_diagnostic_robustness_20261018.py\nOPENBLAS_NUM_THREADS=1 python3 scripts/audit_automatic_diagnostic_robustness_20261018.py\nOPENBLAS_NUM_THREADS=1 python3 scripts/report_automatic_diagnostic_robustness_20261018.py\n```')
    path=ROOT/'docs/全自动诊断稳健性实验结果_20261018.md';path.write_text('\n\n'.join(parts)+'\n');(DEST/'结果报告.md').write_text(path.read_text())
    dump(DEST/'report_manifest.json',dict(report=str(path),sha256=sha(path),sources={n:sha(DEST/n) for n in ['summary.csv','deduplicated_conditions.csv','condition_checks.csv','independent_audit.json']}))
    print(path)


if __name__=='__main__':main()

"""Build a Chinese, raw-result-linked report and exportable scientific figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_validation_common_20261018 import *

METHODS=['expanded_greedy','shared_greedy','static_rarity','vit_kcenter','same_cell_random']
LABEL=dict(zip(METHODS,['扩展贪心','共享贪心','静态稀缺度','ViT k-center','同单元随机']))
EN=dict(zip(METHODS,['Expanded greedy','Shared greedy','Static rarity','ViT k-center','Matched random']))
COLORS=dict(zip(METHODS,['#0072B2','#999999','#D55E00','#009E73','#CC79A7']))
DOC=ROOT/'docs/覆盖与诊断补充实验结果_20261018.md'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def link(name,label=None):
    return f'[{label or name}]({OUT/name})'


def main():
    cv=read(OUT/'coverage_metrics.csv');ext=read(OUT/'external_diagnostic_summary.csv')
    rep=read(OUT/'annotation_repeat_summary.csv');sensitivity=read(OUT/'synthetic_missingness.csv')
    details=read(OUT/'external_condition_results.csv')
    assert json.loads((OUT/'final_numerical_audit.json').read_text())['status']=='pass'
    def get(c,m='expanded_greedy',b=200,ns='all',metric='support_completion_pct',cap=20,weight='state',evalcap=20):
        rr=[r for r in cv if r['class_name']==c and r['method']==m and int(r['budget'])==b and r['namespace']==ns
            and r['metric']==metric and int(r['selection_cap'])==cap and r['weighting']==weight and int(r['evaluation_cap'])==evalcap]
        assert len(rr)==1
        return rr[0]
    def value(*args,**kwargs):return float(get(*args,**kwargs)['mean_pct'])
    def em(method,scope='all'):return next(r for r in ext if r['method']==method and r['scope']==scope)
    random_delta=[value(c)-value(c,'same_cell_random') for c in CLASSES]
    rarity_delta=[value(c)-value(c,'static_rarity') for c in CLASSES]
    source=load_contexts()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for ax,c in zip(axes.flat,CLASSES):
        for m in METHODS:
            xs=[50,100,200,300];ys=[value(c,m,b) for b in xs]
            ax.plot(xs,ys,marker='o',markersize=3,lw=1.5,label=EN[m],color=COLORS[m])
            if m=='same_cell_random':
                ax.fill_between(xs,[float(get(c,m,b)['p025_pct']) for b in xs],[float(get(c,m,b)['p975_pct']) for b in xs],color=COLORS[m],alpha=.12)
        ax.set_title(c.capitalize());ax.set_xticks([50,100,200,300]);ax.set_ylim(32,83);ax.grid(alpha=.15)
    axes[1,0].set_xlabel('Selection budget per class');axes[1,1].set_xlabel('Selection budget per class')
    axes[0,0].set_ylabel('Support completion at k=20 (%)');axes[1,0].set_ylabel('Support completion at k=20 (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(figs/'budget_support.svg',bbox_inches='tight');fig.savefig(figs/'budget_support.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax,scope,title in zip(axes,['all','specific'],['All-state discovery (primary)','Specific-state discovery (secondary)']):
        confirmed=np.array([float(em(m,scope)['confirmed']) for m in METHODS]);failed=np.array([float(em(m,scope)['not_confirmed']) for m in METHODS]);unsupported=np.array([float(em(m,scope)['insufficient']) for m in METHODS]);x=np.arange(5)
        ax.bar(x,confirmed,label='Direction confirmed',color='#0072B2');ax.bar(x,failed,bottom=confirmed,label='Not confirmed',color='#D55E00');ax.bar(x,unsupported,bottom=confirmed+failed,label='Insufficient support',color='#BBBBBB')
        ax.set_xticks(x,['Expanded','Shared','Rarity','k-center','Random'],rotation=20);ax.set_title(title);ax.set_ylim(0,65)
    axes[0].set_ylabel('Conditions across 4 classes and 5 models')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.04));fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(figs/'external_confirmation.svg',bbox_inches='tight');fig.savefig(figs/'external_confirmation.png',dpi=180,bbox_inches='tight');plt.close(fig)
    sections=[]
    def p(s):sections.append(s)
    p('# 风险构成受控的视觉状态覆盖与模型诊断：补充实验结果\n\n生成时间：'+now())
    p('本报告记录实际运行的补充实验。研究对象是马、羊、人、汽车四类；分类器仍在原来的六分类任务上评估，未重新训练。新增的工作包括强基线、独立预算选样、阈值与权重敏感性、外部专属状态标注与弱项复核、自动标注重复性以及人工盲审材料。')
    p('为避免与原方法文档的 RQ1–RQ4 混淆，本报告将新增实验记为补充问题 A–D。其中 A 扩展原覆盖问题，C 对应原 RQ4 的覆盖—诊断关系；原文关于不同 Q 层模型表现的实验仍保留，不在本轮重新编号。')
    p(f'**先说结论：**主预算下，扩展贪心的支持完成度比同单元随机均值高 {min(random_delta):.2f}–{max(random_delta):.2f} 个百分点，比静态稀缺度仅高 {min(rarity_delta):.2f}–{max(rarity_delta):.2f} 个百分点。覆盖收益可以报告，算法相对简单强基线的优势较小。外部诊断结果见后文；人工标注准确率仍未测得，不能把自动重复一致性写成人工验证。')
    p(f'**诊断结果有取舍：**主分析中扩展贪心的 {float(em("expanded_greedy")["proposed"]):.0f} 条条件有 {float(em("expanded_greedy")["confirmed"]):.0f} 条方向复现，{float(em("expanded_greedy")["insufficient"]):.0f} 条支持不足；k-center 方向复现 {float(em("vit_kcenter")["confirmed"]):.0f} 条。当前结果不支持“覆盖最高的方法诊断也最好”。仅专属状态的次分析中，扩展贪心复现数为 {float(em("expanded_greedy","specific")["confirmed"]):.0f}，其他方法略低，不能用次分析替换主分析。')
    p('## 1. 做了什么，数据从哪里来')
    p(table(['项目','本次实际设置'],[
        ['发现池','COCO B：400 个目标/类，共 1600；已有共享与专属状态标签'],
        ['覆盖选样','预算 50、100、200、300 分别独立重跑；不是从预算 200 的序列截取'],
        ['基线','扩展贪心、共享贪心、静态稀缺度、预训练 ViT-B/16 特征 k-center、同单元随机'],
        ['风险约束','每种方法精确匹配 (Q、冻结风险分数中点、区间宽度) 单元配额；每个 Q 占 20%'],
        ['随机对照','每类每预算 1000 次；主预算复用已冻结的 1000 次抽样，其他预算新增抽样'],
        ['外部数据','Objects365 历史 D 池按固定哈希抽取 200/类，共 800 个目标；全部新增当前专属字段标注'],
        ['分类器','5 个模型 × 3 个已有训练种子；读取缓存预测，未重新训练或宣称新模型能力提升'],
        ['标注重复性','80 个目标，使用原证据图、原词典和原提示重新调用 Qwen3-VL；每类 20'],
        ['缺失敏感性','专属字段随机隐藏 5%、10%、20%；每比例每类 20 次，共 240 次重新选样'],
        ['人工盲审','320 个目标、4480 个字段判断/人；两位独立标注者材料已生成，真实人工标签尚缺']]))
    p('冻结模型标识为 `yolo26s_cls`、`pe_core_b16`、`siglip2_base`、`mobileclip2_s0`、`aimv2_large`，训练种子为 20260916、20260917、20260918。新自动标注使用 `qwen3-vl:30b-a3b-instruct`；推理原始响应和证据图全部保存。')
    p('ViT 基线使用既有 ImageNet 预训练 ViT-B/16 的 768 维特征，L2 归一化后按余弦距离最远点选样；首点取可行样本中最接近类别中心者。静态稀缺度对样本包含的各状态累加池内频数的倒数。它们使用相同候选池和风险配额；两种状态方法与稀缺度基线使用相同自动状态标签，ViT 方法不使用这些标签，其视觉表征计算成本不同，本次未比较成本优劣。')
    p('风险匹配只控制冻结代理分数的构成，不等于证明真实风险、语义难度或所有混杂因素完全相同。共享词典固定 71 个状态，四类扩展词典分别为 144、133、144、128 个；覆盖率以各自冻结词典为分母，包括当前池内没有出现的合法状态。')
    p('## 2. 指标怎么理解')
    p('设测试集为 $T$，冻结词典为 $S$，$n_s(T)$ 是状态 $s$ 在测试集中出现的目标数，$k$ 是希望每个状态至少获得的支持数。全部指标越高越好，但含义不同。')
    p('$$C_{\\mathrm{occ}}(T)=\\frac{1}{|S|}\\sum_{s\\in S}\\mathbf{1}[n_s(T)>0],\\qquad C_k(T)=\\frac{1}{|S|}\\sum_{s\\in S}\\min\\{n_s(T)/k,1\\}.$$\n\n$$C_{\\mathrm{thr},k}(T)=\\frac{1}{|S|}\\sum_{s\\in S}\\mathbf{1}[n_s(T)\\geq k].$$')
    p('出现覆盖率 $C_{occ}$ 回答“见过多少种状态”；支持完成度 $C_k$ 回答“离每种状态都积累 k 个目标还有多远”；达阈值覆盖率回答“多少种状态已达到 k 个目标”。例如某状态只出现 1 次，出现覆盖已计满，但 k=20 时支持完成度只计 1/20，达阈值覆盖仍计 0。正文百分数均乘以 100。支持完成度不是状态出现覆盖率，也不是分类准确率。')
    p('选样目标仍为 $\\sum_s w_s\\log(1+\\min\\{n_s(T),k\\})$，不是直接最大化上述线性支持完成度。主分析 $w_s=1,k=20$；字段等权实验使每个字段下的状态权重之和为 1。因此少数轻微反转可以出现，不应通过更换主指标掩盖。')
    p('## 3. 补充问题 A：相同风险构成和预算下，覆盖率是否更高？')
    p('主预算为每类 200，评价支持阈值 k=20。下表为**全部状态的支持完成度（%）**；随机列是 1000 次均值。')
    p(table(['类别']+[LABEL[m] for m in METHODS]+['扩展−稀缺度（pp）'],[[CN[c]]+[f'{value(c,m):.2f}' for m in METHODS]+[f'{value(c)-value(c,"static_rarity"):+.2f}'] for c in CLASSES]))
    p('下面单独列出**状态出现覆盖率（%）**，避免把“支持增加”误写成“覆盖了更多种状态”。')
    p(table(['类别']+[LABEL[m] for m in METHODS],[[CN[c]]+[f'{value(c,m,metric="presence_pct"):.2f}' for m in METHODS] for c in CLASSES]))
    p('四类的出现覆盖率与静态稀缺度全部相同。因此，相对该强基线不能声称“覆盖了更多种状态”；主表的小幅收益来自部分状态的支持数变化。')
    p('达阈值覆盖率（%，至少 20 个目标支持）如下。它与平滑支持完成度的排序不必一致，例如人类的共享贪心高于扩展贪心。')
    p(table(['类别']+[LABEL[m] for m in METHODS],[[CN[c]]+[f'{value(c,m,metric="threshold_pct"):.2f}' for m in METHODS] for c in CLASSES]))
    p('扩展贪心相对于共享贪心的共享/专属权衡如下。数字是扩展−共享的百分点变化，两个状态空间分别使用自己的固定分母。')
    p(table(['类别','共享出现覆盖 Δ','专属出现覆盖 Δ','共享支持完成度 Δ','专属支持完成度 Δ'],[[CN[c]]+[f'{value(c,ns=ns,metric=metric)-value(c,"shared_greedy",ns=ns,metric=metric):+.2f}' for metric in ['presence_pct','support_completion_pct'] for ns in ['shared','specific']] for c in CLASSES]))
    p('四类的共享出现覆盖没有变化，但共享支持完成度均有下降；专属支持完成度均有上升。应表述为共享与专属支持的分配权衡，不能声称专属状态增益没有成本。')
    p('候选池本身的可达范围也需要交代。即使取满 400 个目标，池内从未出现的状态也无法通过重新选样获得。下表是全池统计，作为描述性上限，不是预算 200 且受风险约束时一定能达到的最优值。')
    p(table(['类别','冻结状态数','全池未出现状态数','全池出现覆盖（%）','全池支持完成度（%）'],[[CN[c],len(source[c]['states']),int(np.sum(source[c]['matrix'].sum(0)==0)),f'{100*np.mean(source[c]["matrix"].sum(0)>0):.2f}',f'{100*np.minimum(source[c]["matrix"].sum(0)/20,1).mean():.2f}'] for c in CLASSES]))
    p('同单元随机的支持完成度 2.5%–97.5% 分位区间如下。这是**抽样分布区间**，不是总体效果的置信区间，也不能当作算法间显著性检验。')
    p(table(['类别','随机均值','随机分位区间','扩展贪心'],[[CN[c],f'{value(c,"same_cell_random"):.2f}',f'[{float(get(c,"same_cell_random")["p025_pct"]):.2f}, {float(get(c,"same_cell_random")["p975_pct"]):.2f}]',f'{value(c):.2f}'] for c in CLASSES]))
    p('数据：'+link('coverage_metrics.csv')+'；每次随机抽样的实际结果：'+link('coverage_runs.csv')+'。')
    p('## 4. 补充问题 B：预算、阈值和权重变化后，结论是否稳定？')
    p(f'![不同预算下的支持完成度]({figs/"budget_support.png"})')
    p('所有曲线统一在 k=20 下评价。阴影是同单元随机的 95% 抽样分位区间。下表逐项给出扩展贪心相对静态稀缺度的支持完成度差值（pp）。')
    p(table(['类别','预算50','预算100','预算200','预算300'],[[CN[c]]+[f'{value(c,b=b)-value(c,"static_rarity",b=b):+.3f}' for b in [50,100,200,300]] for c in CLASSES]))
    wins=sum(value(c,b=b)>value(c,'static_rarity',b=b)+1e-10 for c in CLASSES for b in [50,100,200,300])
    p(f'16 个设置中有 {wins} 个支持完成度高于静态稀缺度。羊类预算 300 时存在轻微反转，不能表述为“所有预算均优于强基线”。随着预算接近候选池规模，方法间差距总体缩小。')
    p('阈值/权重敏感性：下表各列均以同一个 k=20 评价，只改变选样时的阈值或权重，避免因改变指标本身造成虚假的收益。原始 CSV 同时保留共同评价阈值 10、20、30 的全量结果。')
    p(table(['类别','选样k=10','选样k=20（主）','选样k=30','字段等权，k=20'],[[CN[c],f'{value(c,cap=10):.2f}',f'{value(c):.2f}',f'{value(c,cap=30):.2f}',f'{value(c,weight="field"):.2f}'] for c in CLASSES]))
    p('字段等权没有破坏总体模式，部分数值略高于原设置；不据此事后更换主设置。本实验只覆盖这些离散预算、阈值和权重，不能外推为任意参数鲁棒。')
    p('## 5. 补充问题 C（原 RQ4）：发现的模型弱项能否在外部数据中重现？')
    p('发现阶段对每个“类别 × 模型 × 选样方法”，先将 3 个种子的正确性逐目标平均，再计算状态内 Recall 与该选集的类别 Recall 之差。只保留支持数 ≥20 且差值 ≤−10 pp 的状态，按差值从低到高取最多 3 个，词典键打破平局。条件及文件哈希在读取本次外部预测前冻结；不足 3 个不补齐。主分析允许全部状态，次分析限定专属状态，两者不能混合计算。')
    p('外部复核使用固定的每类 200 个目标；支持数 ≥20 且状态内 Recall 低于外部整类 Recall，记为“方向复现”。外部未继续要求下降 ≥10 pp，所以方向复现是较宽松标准，不等于显著、严重或因果弱点。每模型每类对目标做 1000 次配对 bootstrap，条件与整类基准使用同一次重采样；未进行多重比较校正。')
    p('外部状态样本与外部整类之间没有再次逐风险单元匹配；状态可能与尺度、遮挡等共现。因此这里检验的是关联方向能否迁移，不识别该状态对模型错误的独立因果作用。')
    for scope,title in [('all','主分析：全部状态'),('specific','次分析：仅专属状态')]:
        p('### '+title)
        p(table(['方法','提出数','支持充分','方向复现','未复现','支持不足','复现/支持充分','CI上界<0'],[[LABEL[m]]+[f'{float(em(m,scope)[key]):.2f}' for key in ['proposed','evaluable','confirmed','not_confirmed','insufficient']]+[f'{float(em(m,scope)["confirmation_pct"]):.2f}%',f'{float(em(m,scope)["confirmed_ci_below_zero"]):.2f}'] for m in METHODS]))
        proposed=em('expanded_greedy',scope);rarity=em('static_rarity',scope);shared=em('shared_greedy',scope)
        p(f'扩展贪心提出 {float(proposed["proposed"]):.0f} 个条件，其中 {float(proposed["confirmed"]):.0f} 个方向复现；静态稀缺度为 {float(rarity["confirmed"]):.0f} 个，共享贪心为 {float(shared["confirmed"]):.0f} 个。应同时比较提出数、支持不足数与复现率，不能只比较复现率。')
    p('计数汇总 4 类 × 5 模型，确定性方法理论最多 60 个条件。随机结果是 100 次选集的计数均值，因而可为小数；不能把随机的 100 次重复当作新增独立数据集。同一状态可在多个模型出现，条件之间相关，两个范围也可能重叠，以上数量不是互不重复弱点总数。CI 上界低于 0 仅为未校正的逐条件描述，不能称为全部条件通过显著性检验。')
    p('主分析的方向复现数量由 k-center 取得最高值 47，扩展贪心为 44；扩展贪心 100% 的条件复现率只针对支持充分的 44 条，另外 15 条不能算作已复现。仅专属状态时扩展贪心为 37，k-center 为 36、稀缺度为 35、共享贪心为 33，差距有限。可报告扩展表示在专属诊断上的潜在增量，同时明确它未在主分析中全面领先。')
    p(f'![外部条件复核结果]({figs/"external_confirmation.png"})')
    p('这是一项**历史数据上的回顾性外部复核**：D 数据和这些模型此前已用于项目，虽然本次新增条件与专属词典应用已冻结，不能称为从未接触过外部结果的前瞻验证。原图精确哈希检查没有发现发现池/外部池重合；新检查未独立证明不存在感知近重复。新专属标签由同一个 VLM 生成，跨数据集复现仍可能受到共同标注偏差影响。')
    p('数据：'+link('external_diagnostic_summary.csv')+'、'+link('external_diagnostic_groups.csv')+'、'+link('external_condition_results.csv')+'；冻结记录：'+link('conditions_before_external_predictions.json')+'。')
    p('## 6. 补充问题 D：自动状态标签与选样是否稳定？')
    p('下面是实际重新调用标注模型的结果。枚举字段比较取值，多选字段比较集合；Jaccard 比较集合交并比。原证据图、词典、提示不变，温度为 0，只改变随机种子。高一致性可能来自低温度/确定性解码，**不证明取值正确**。')
    p(table(['类别','目标数','字段数','完全一致（%）','平均Jaccard','unknown判断一致（%）'],[[CN[r['class_name']],r['targets'],r['fields'],f'{float(r["exact_agreement_pct"]):.2f}',f'{float(r["mean_jaccard"]):.4f}',f'{float(r["unknown_agreement_pct"]):.2f}'] for r in rep]))
    p('随机字段缺失实验中，每次先隐藏部分专属字段的所有状态，重新选样，再用原始自动标签矩阵评价。下表为支持完成度相对无隐藏版本的平均变化（pp），每格 20 次。')
    p(table(['类别','隐藏5%','隐藏10%','隐藏20%'],[[CN[c]]+[f'{np.mean([float(r["support_completion_pct"]) for r in sensitivity if r["class_name"]==c and float(r["missingness_rate"])==rate])-value(c):+.3f}' for rate in [.05,.1,.2]] for c in CLASSES]))
    p('该结果只说明对所模拟随机缺失的敏感性，不覆盖系统性误标、相关错误或状态定义偏差；原自动标签并非人工真值，小幅增益也不代表删除信息有真实收益。数据：'+link('annotation_repeat_summary.csv')+'、'+link('annotation_repeat_pairs.csv')+'、'+link('synthetic_missingness.csv')+'。')
    p('### 人工可靠性：材料已准备，实验尚未完成')
    p('按类别与 Q 分层抽取 320 个目标，生成两份不同随机顺序的独立盲审网页；展示图像、字段定义、取值规则，隐藏自动标签、方法选择和模型结果。每位标注者需完成 4480 个字段判断。真实人工标注者、分歧仲裁和人工修正后重选样尚未执行，因此不能报告人工准确率或人工修正后的覆盖增益。')
    p('入口：'+link('human_review/blind_review_A.html','标注者 A 盲审')+'；'+link('human_review/blind_review_B.html','标注者 B 盲审')+'；'+link('human_review/使用说明.md','填写、导出与导入说明')+'。未达成共识的字段需仲裁；仅已达成共识子集上的一致率不能直接充当全样本准确率。')
    p('## 7. 论文结论可以写到什么程度？')
    p('可以把“风险构成受控的视觉状态覆盖与模型诊断实证研究”继续作为主线，并将问题写成以下四个：相同代理风险构成下覆盖是否提升；提升是否超过简单状态稀缺度及视觉多样性基线；新增覆盖能否转化为可复核的诊断条件；结论对预算、支持阈值、字段权重及标注稳定性有多敏感。')
    p('当前最可靠的结论是固定四类、固定词典、固定候选池上的覆盖和支持改善；不能宣称算法全面领先、覆盖增长必然带来诊断收益、风险构成相同就等于真实难度相同、或模型泛化能力得到提升。外部方向复现需逐方法报告，不把覆盖优势自动转换为诊断优势。')
    p('**对投稿成熟度的保守判断：**这些补充结果提高了实证完整性，但静态稀缺度很接近，算法贡献应克制；人工语义准确率仍是关键缺口。现在适合继续整理论文与补充材料，不能仅凭本次结果认定已达到某个会议/期刊的录用水平，也不提供缺少依据的分数。下一步优先完成盲审、仲裁及人工修正敏感性；如要主张更普遍诊断能力，再增加真正未用于项目选择的外部数据或新模型，预先冻结条件。')
    p('## 8. 复现、审计与文件')
    p('覆盖数值已从保存选集独立重算，核对每个风险单元配额、静态稀缺度的独立单元内排序和冻结输入哈希；外部标注逐项核对原始响应、解析结果与证据哈希，外部分类与汇总重新核对。'+link('final_numerical_audit.json','数值审计')+'。本报告由结果 CSV 自动生成，不手工改表格。')
    p('运行命令（在项目根目录；新增推理可断点续跑）：\n\n```bash\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py check\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py select\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py discover\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py sensitivity\nOPENBLAS_NUM_THREADS=1 python3 scripts/annotate_coverage_validation_20261018.py --workers 2\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py reliability\nOPENBLAS_NUM_THREADS=1 python3 scripts/run_coverage_validation_20261018.py confirm\nOPENBLAS_NUM_THREADS=1 python3 scripts/audit_coverage_validation_20261018.py --external\nOPENBLAS_NUM_THREADS=1 python3 scripts/report_coverage_validation_20261018.py\n```')
    p('协议：'+link('protocol.json')+'；执行日志：'+link('research_log.md')+'；标注原始结果位于 '+str(OUT/'annotations')+'；可编辑导出图：'+link('figures/budget_support.svg')+'、'+link('figures/external_confirmation.svg')+'。选集、阈值、条件、自动标签均保留；历史实验源文件未被本次覆盖。')
    DOC.write_text('\n\n'.join(sections)+'\n')
    (OUT/'结果报告.md').write_text(DOC.read_text())
    dump(OUT/'report_manifest.json',dict(generated_at=now(),report=str(DOC),report_sha256=sha(DOC),
        data_sha256={n:sha(OUT/n) for n in ['coverage_metrics.csv','external_diagnostic_summary.csv','annotation_repeat_summary.csv','synthetic_missingness.csv','final_numerical_audit.json']}))
    print(str(DOC))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Preserve raw local-model proposals; export explicitly assistant-revised enums."""
import copy
import json
from pathlib import Path
import build_specific_coverage_dictionary_20261018 as build


def revise(f, definition, values, kind='enum'):
    f.update(definition_zh=definition, value_type=kind,
             values=[dict(value_id=i,label_zh=n,criterion_zh=c) for i,n,c in values]+
                    [dict(value_id='unknown',label_zh='未知',criterion_zh='所需区域不可见、证据不足或无法可靠区分。')],
             unknown_rule_zh='不可见、出画、遮挡、模糊或无法可靠区分时记unknown；不能将未知当作没有。',
             applicability_zh='仅在定义所需区域和参照可辨时适用；否则unknown。',
             limitations_zh='只描述可见部分，不推断隐藏属性、品种、身份或视觉难度。',
             semantic_revision_by='assistant; not human-reviewed')


def main():
    proposals={}
    for category in build.CLASSES:
        old=json.loads((build.PREVIOUS/category/'proposed_for_review.json').read_text())
        raw=json.loads((build.OUT/category/'llm_additional_appearance_proposal.json').read_text())
        proposal=copy.deepcopy(raw)
        fields={f['feature_id']:f for f in proposal['features']}
        if category=='horse':
            revise(fields['mane_form'],'可见鬃毛按编束、竖立、悬垂、贴伏的优先次序单选。',[
                ('braided','编束','可见编织或束结，优先于其他类型。'),
                ('upright','竖立','无编束，主要毛束明显离开颈部竖立。'),
                ('hanging','悬垂','无以上形态，主要毛束沿颈侧下垂。'),
                ('flat','贴伏','无以上形态，主要毛束紧贴颈部。'),
                ('other','其他可辨形态','清楚可辨但不符合上述类型。')])
            revise(fields['forehead_marking'],'额部皮毛可检查时记录异色斑轮廓；阴影和反光不作异色斑。',[
                ('none_observed','未见明显异色斑','额部清晰可检查，未见明显异色；不代表全脸无斑。'),
                ('compact','紧凑块状','单个主要斑块，长宽比小于3。'),
                ('elongated','长条状','单个主要连续斑块，长宽比至少为3。'),
                ('multiple_or_complex','多块或复杂形','多块斑或复杂轮廓，无法稳定归入前两类。'),
                ('other','其他可辨形态','清楚可辨但不符合上述类型。')])
            revise(fields['leg_marking_distribution'],'只数可检查区域中有明显异色被毛的腿数；不推断隐藏腿，应与肢体可见性联合解读。',[
                ('zero_observed','可见腿未见异色','至少一条腿的皮毛可检查，已检查区域均未见异色。'),
                ('one_observed','可见1条腿有异色','可见异色分布在一条腿上。'),
                ('two_observed','可见2条腿有异色','可见异色分布在两条不同腿上。'),
                ('three_plus_observed','可见至少3条腿有异色','可见异色分布在至少三条不同腿上。')])
        elif category=='sheep':
            revise(fields['horn_shape'],'只描述可辨角的轮廓；看不见角不能推断无角；多个可辨角形态不同时记混合。',[
                ('straight','近直形','主体近直，无明显弯曲或旋绕。'),
                ('curved','弧曲形','明显弯曲但无清楚旋绕结构。'),
                ('spiral','旋绕形','可见清楚卷曲或旋绕轮廓。'),
                ('mixed','混合形态','不同可辨角分属不同类型。'),
                ('other','其他可辨形态','角轮廓清楚但不符合上述类型。')])
            revise(fields['ear_shape'],'描述耳廓末端轮廓，不用缺少参照的大小或厚度；多耳形态不同记混合。',[
                ('tapered','渐窄尖端','耳廓向末端渐窄，末端偏尖。'),
                ('rounded','圆钝末端','末端呈圆弧，无明显尖端。'),
                ('mixed','混合轮廓','不同可辨耳廓分别属于以上两类。'),
                ('other','其他可辨轮廓','轮廓清楚但不符合上述类型。')])
        elif category=='person':
            revise(fields['hair_shape'],'先判断可见束发结构，再按松散头发与肩线关系单选；不把长度和卷直混作单选。',[
                ('tied','可见束发','可见发髻、辫子或马尾等束发结构。'),
                ('loose_above_shoulder','松散且高于肩线','无可见束发，发梢和肩线均可辨，发梢高于肩线。'),
                ('loose_at_or_below_shoulder','松散且到达肩线或以下','无可见束发，发梢和肩线均可辨，发梢到达或低于肩线。'),
                ('other','其他可辨轮廓','轮廓清楚但不符合上述类型。')])
            original=fields['clothing_pattern']
            values=[(v['value_id'],v['label_zh'],v['criterion_zh']) for v in original['values'] if v['value_id']!='unknown']
            values=[(i,n,'至少一个可检查衣着面板为纯色、无明显图案。' if i=='solid_color' else c) for i,n,c in values]
            revise(original,'记录可见衣着面板图案，可多选；不同面板可以分别为纯色和有图案。unknown不能与其他值共存。',values,'set')
            revise(fields['footwear_appearance'],'按可见包覆几何形态单选：后跟开放优先，其次露趾，再分包踝或低帮；两脚不同记混合。',[
                ('open_back','后跟开放鞋','可辨鞋后部不包覆后跟。'),
                ('open_toe_supported','露趾且固定后跟','脚趾外露，后跟有可见固定结构。'),
                ('closed_high','包头且覆盖踝部','鞋头封闭，鞋面或鞋筒包覆踝部以上。'),
                ('closed_low','包头低帮','鞋头封闭，鞋面低于踝部。'),
                ('barefoot','可见赤足','脚部清楚，可直接确认没有穿鞋。'),
                ('mixed_or_other','混合或其他可辨外观','两脚鞋形不同或清楚但不属于以上类型。')])
        else:
            revise(fields['light_shape'],'记录可辨灯具外轮廓，可多选；前后灯可同时贡献状态，不推断未见灯具。',[
                ('round','圆形或椭圆形','外轮廓主要为连续圆弧。'),
                ('elongated','细长条形','长宽比至少3，无明显折角。'),
                ('rectangular','近矩形','近四边形且长宽比小于3。'),
                ('angular','折角形','轮廓呈L形等明显折角。'),
                ('other','其他可辨形态','灯具清楚但不属于上述类型。')],'set')
            revise(fields['grille_shape'],'只在格栅区域本身可辨时描述内部结构；不能因车头看似封闭就推断没有格栅。',[
                ('horizontal','横向条结构','主要为横向平行条，非网格。'),
                ('vertical','竖向条结构','主要为竖向平行条，非网格。'),
                ('mesh','网格结构','可辨相交或重复网孔结构。'),
                ('mixed','混合结构','同一可见格栅有不同结构分区。'),
                ('other','其他可辨结构','格栅清楚但不属于上述类型。')])
            revise(fields['wheel_design'],'描述可见轮辋或轮罩结构，不将颜色与结构混作单选；先判断交叉网状，再数可辨独立辐条。',[
                ('mesh','交叉网状','可见辐条形成明确交叉网络。'),
                ('few_spokes','少辐结构','非交叉网状，可辨独立辐条不超过5根。'),
                ('many_spokes','多辐结构','非交叉网状，可辨独立辐条至少6根。'),
                ('disc','盘面结构','以连续盘面为主，无开放辐条。'),
                ('mixed_or_other','混合或其他可辨结构','多个轮辋结构不同或清楚但不属于上述类型。')])
        proposal['revision_note']='LLM候选经助手修订单选互斥、缺失值与可观察判据；不等于人工确认。'
        build.dump(build.OUT/category/'revised_additional_appearance_proposal.json',proposal)
        proposals[category]=(old,proposal)
    build.export(proposals)
    for category in build.CLASSES:
        path=build.OUT/category/'coverage_features.json'
        data=json.loads(path.read_text())
        for f in data['features']:
            if f.get('semantic_revision_by'):
                f['provenance']='本地Qwen3-VL候选，经助手语义修订；尚未人工审查'
        build.dump(path,data)
    overview=build.OUT/'开始这里.md'
    overview.write_text(overview.read_text()+'\n## 来源与修订\n\n四类新增字段由本地Qwen3-VL实际生成，原始请求和响应逐类保留。助手修订了单选重叠、未观察到与未知混用、无参照大小及轮辋颜色/结构混用；详见各类revised_additional_appearance_proposal.json。尚未人工确认具体定义。\n')
    audit_path=build.OUT/'audit.json'
    audit=json.loads(audit_path.read_text())
    paths=[Path(__file__),overview,build.OUT/'专属特征审查.csv',build.OUT/'risk_lock_audit.json']
    paths += [build.OUT/c/n for c in build.CLASSES for n in ['coverage_features.json','revised_additional_appearance_proposal.json']]
    audit['source_sha256'].update({str(p):build.sha(p) for p in paths})
    audit['assistant_semantic_revision']=True
    audit['human_definition_review_completed']=False
    build.dump(audit_path,audit)


if __name__=='__main__':
    main()

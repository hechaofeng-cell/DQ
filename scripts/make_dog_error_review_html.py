#!/usr/bin/env python3
"""Create a local visual review page for dog-v2 classification errors."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/dog_feature_pipeline_v2_20260911"
RESULTS = OUT / "results"
FEATURE_ZH = {
    "coat_primary_color": ("毛发主颜色", {"black":"黑色","white":"白色","brown":"棕色","gray":"灰色","tan":"黄褐色","golden":"金色","cream":"奶油色","red":"红色","mixed":"混合","other":"其他","unknown":"未知"}),
    "coat_pattern": ("毛发图案", {"solid":"纯色","bicolor":"双色","tricolor":"三色","spotted":"斑点","striped":"条纹","patchy":"斑驳","unknown":"未知"}),
    "coat_length": ("毛发长度", {"short":"短毛","medium":"中等","long":"长毛","hairless_or_not_visible":"无毛或不可见","unknown":"未知"}),
    "coat_texture": ("毛发质地", {"smooth":"顺滑","wavy":"波浪","curly":"卷曲","wiry":"硬直","fluffy":"蓬松","unknown":"未知"}),
    "white_marking_presence": ("白色斑纹", {"none":"无","small":"少量","moderate":"中等","large":"较大面积","unknown":"未知"}),
    "head_visibility": ("头部可见性", {"clear":"清晰","partial":"部分可见","not_visible":"不可见","unknown":"未知"}),
    "ear_shape": ("耳朵形状", {"upright":"竖立","floppy":"垂耳","semi_upright":"半竖立","folded":"折叠","not_visible":"不可见","unknown":"未知"}),
    "ear_position": ("耳朵位置", {"forward":"向前","sideways":"向侧面","backward":"向后","mixed":"混合","not_visible":"不可见","unknown":"未知"}),
    "muzzle_visibility": ("口鼻可见性", {"clear":"清晰","partial":"部分可见","not_visible":"不可见","unknown":"未知"}),
    "eye_visibility": ("眼睛可见性", {"both":"双眼","one":"单眼","none":"不可见","unknown":"未知"}),
    "tail_visibility": ("尾巴可见性", {"clear":"清晰","partial":"部分可见","not_visible":"不可见","unknown":"未知"}),
    "tail_position": ("尾巴位置", {"up":"向上","down":"向下","curved":"弯曲","extended":"伸展","unknown":"未知"}),
    "leg_visibility": ("腿部可见性", {"four_or_more":"四条或以上","two_or_three":"两到三条","one":"一条","none":"不可见","unknown":"未知"}),
    "body_shape": ("身体形状", {"slender":"纤细","average":"平均","stocky":"粗壮","puppy_like":"幼犬样","unknown":"未知"}),
    "orientation": ("朝向", {"front":"正面","back":"背面","left_side":"左侧面","right_side":"右侧面","three_quarter":"斜侧面","unknown":"未知"}),
    "pose": ("姿势", {"standing":"站立","sitting":"坐姿","lying":"躺卧","walking":"行走","running":"奔跑","jumping":"跳跃","unknown":"未知"}),
    "action_state": ("动作状态", {"stationary":"静止","walking":"行走","running":"奔跑","playing":"玩耍","eating":"进食","drinking":"饮水","interacting":"互动","unknown":"未知"}),
    "occlusion_level": ("遮挡程度", {"none":"无遮挡","partial":"部分遮挡","heavy":"严重遮挡","unknown":"未知"}),
    "truncation_level": ("边界截断", {"none":"无截断","partial":"部分截断","heavy":"严重截断","unknown":"未知"}),
    "outline_visibility": ("轮廓清晰度", {"clear":"清晰","partial":"部分清晰","unclear":"不清晰","unknown":"未知"})
}


def esc(value):
    return html.escape(str(value if value not in (None, "") else "-"))


def img(path):
    path = Path(path).resolve()
    return "/" + str(path.relative_to(ROOT)).replace("\\", "/")


def feature_block(image_id):
    path = RESULTS / "parsed/features" / f"{image_id}.json"
    if not path.exists():
        return "<p class='missing'>特征 VLM 未产生有效结果。</p>"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, item in value.get("features", {}).items():
        rows.append(f"<tr><td>{esc(key)}</td><td>{esc(item)}</td><td>{esc(value.get('evidence', {}).get(key, ''))}</td></tr>")
    return "<table class='features'><thead><tr><th>feature_id</th><th>value</th><th>evidence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def chinese_feature_block(image_id):
    path = RESULTS / "parsed/features" / f"{image_id}.json"
    if not path.exists():
        return "<p class='missing'>特征 VLM 未产生有效结果。</p>"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, item in value.get("features", {}).items():
        name, values = FEATURE_ZH.get(key, (key, {}))
        rows.append(f"<tr><td>{esc(name)}</td><td>{esc(values.get(item, item))}</td><td>{esc(value.get('evidence', {}).get(key, ''))}</td></tr>")
    return "<table class='features'><thead><tr><th>中文特征</th><th>中文取值</th><th>模型证据（原文）</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def card(row, index, errors):
    image_id = row["image_id"]
    error = next((e["error"] for e in errors if e["image_id"] == image_id and e["stage"] == "features"), "无特征错误")
    return f"""<article class='card'><h2>{index}. {esc(image_id)} <span class='bad'>预测: {esc(row['predicted_class'])}</span></h2>
      <p><b>真实类别：</b>{esc(row['target_class'])}　<b>分类正确：</b>{esc(row['classification_correct'])}　<b>特征状态：</b>{esc(row['feature_status'])}　<b>特征错误：</b>{esc(error)}</p>
      <div class='images'><figure><img src='{img(row["image_path"])}'><figcaption>原图</figcaption></figure><figure><img src='{img(row["marked_image_path"])}'><figcaption>A框标记图</figcaption></figure><figure><img src='{img(row["crop_image_path"])}'><figcaption>修复后放大裁剪图（v2.1输入）</figcaption></figure></div>
      <details><summary>程序字段</summary><pre>{esc(json.dumps({k: row[k] for k in ('target_bbox','target_area_ratio','target_center_x','target_center_y','image_width','image_height','visible_dog_count_in_scene','truncation_level_from_bbox','feature_unknown_count','candidate_difficulty_factors')}, ensure_ascii=False, indent=2))}</pre></details>
      <details><summary>20项特征与证据</summary>{feature_block(image_id)}</details></article>"""


def main():
    rows = list(csv.DictReader((RESULTS / "results.csv").open(encoding="utf-8-sig", newline="")))
    manifest = {r["image_id"]: r for r in csv.DictReader((ROOT / "artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv").open(encoding="utf-8-sig", newline=""))}
    for row in rows:
        row.update(manifest.get(row["image_id"], {}))
    bad = [r for r in rows if r["classification_correct"] == "False"]
    errors = json.loads((RESULTS / "errors.json").read_text(encoding="utf-8"))
    example = next(r for r in rows if r["feature_status"] == "ok" and r["classification_correct"] == "True")
    example_card = card(example, "正确样例", errors).replace("<span class='bad'>预测: dog</span>", "<span class='good'>预测: dog</span>")
    example_card += "<h3>正确样例的中文特征</h3>" + chinese_feature_block(example["image_id"])
    page = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Dog v2 误分类复核</title><style>
body{font-family:Arial,'Noto Sans CJK SC',sans-serif;background:#f4f6f8;color:#17202a;margin:0}.wrap{max-width:1500px;margin:0 auto;padding:24px}.card{background:white;border:1px solid #d9dee5;border-radius:8px;padding:18px;margin:18px 0;box-shadow:0 1px 2px #0000000b}h1{margin:0 0 8px}h2{font-size:18px;margin:0 0 8px}.bad{color:#b42318}.good{color:#067647}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.images figure{margin:0}.images img{width:100%;height:300px;object-fit:contain;background:#eef1f4;border:1px solid #d9dee5}.images figcaption{text-align:center;font-size:13px;color:#667085;padding:5px}details{margin-top:12px}summary{cursor:pointer;font-weight:600}pre{background:#101828;color:#e6edf3;padding:12px;overflow:auto;border-radius:5px}.features{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}.features th,.features td{border:1px solid #d9dee5;padding:6px;vertical-align:top;text-align:left}.features th{background:#f2f4f7}.missing{color:#b42318}@media(max-width:800px){.images{grid-template-columns:1fr}.images img{height:240px}}</style></head><body><main class='wrap'>"""
    page += f"<h1>犬类 v2 独立分类误判复核</h1><p>共 {len(bad)} 张误分类图片。真实类别均为 dog；页面同时展示原图、固定 A 框、裁剪图和 VLM 特征。</p><h2>完整正确特征样例</h2>{example_card}<h2>误分类图片（{len(bad)} 张）</h2>"
    page += "".join(card(row, i, errors) for i, row in enumerate(bad, 1))
    page += "</main></body></html>"
    path = OUT / "dog_v2_misclassification_review.html"
    path.write_text(page, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

# 犬类视觉特征标注与独立识别方案 v2

## 核心调整

每张图片固定一个 COCO dog A 目标，因此删除 `object_count`。目标框、面积比例、中心位置和边界截断由程序从 COCO 标注计算，不由 VLM 判断。删除模型自由输出的 `difficulty_factors`，困难因素由程序推导。

特征 VLM 知道目标是 dog，只填写犬类视觉特征；独立分类 VLM 不知道真实类别、COCO 标签或特征结果。两种任务分别统计。

## 20 个 VLM 特征

```text
coat_primary_color
coat_pattern
coat_length
coat_texture
white_marking_presence
head_visibility
ear_shape
ear_position
muzzle_visibility
eye_visibility
tail_visibility
tail_position
leg_visibility
body_shape
orientation
pose
action_state
occlusion_level
truncation_level
outline_visibility
```

所有字段使用有限枚举；无法判断时使用 `unknown`。颜色允许 black、white、brown、gray、tan、golden、cream、red、mixed、other、unknown。姿势允许 standing、sitting、lying、walking、running、jumping、unknown。动作允许 stationary、walking、running、playing、eating、drinking、interacting、unknown。

禁止推断品种、身份、血统、健康、性格等敏感或不可见属性。

## 程序字段

```text
target_bbox
target_area_ratio
target_center_x
target_center_y
image_width
image_height
visible_dog_count_in_scene
truncation_level_from_bbox
feature_unknown_count
visibility_score
candidate_difficulty_factors
```

候选困难因素固定为 `small_target`、`occlusion`、`truncation`、`low_contrast`、`blur`、`complex_background`、`unusual_pose`、`missing_key_parts`。可见性评分为 `100 * (1 - occlusion_ratio - truncation_ratio)`，无法估计时为 null。

## 特征 VLM 提示词

```text
目标类别：dog。只评价 A 框指定的 target_1。
输入为原图、A 框标记图和未标记原图裁剪放大图；不得更换目标或选择框外对象。
严格按照 dog_feature_schema_v2.json 输出 20 个字段，不得添加、删除、改名或合并字段。
只能使用枚举值，无法判断时使用 unknown。边界外属于 truncation，不属于其他物体遮挡。
pose 描述身体姿势，action_state 描述可观察动作。不得推断品种、身份、健康或性格。
每个字段必须有 evidence。证据必须描述直接可见的视觉事实，不能只重复枚举值。
unknown 也必须说明无法判断的原因；证据不得包含 URL、文件名、COCO 标签或虚构来源。
只输出：{"schema_version":"dog_v2","image_id":"...","target_instance_id":"target_1","features":{},"evidence":{}}。
```

程序检查 JSON、字段完整性、枚举值、ID、evidence 一一对应、证据长度和证据是否仅为枚举值。格式错误不自动转换为 unknown。

## 独立分类 VLM

只输入原图、A 框标记图、裁剪图和固定候选类别列表。不得输入 `target_class=dog`、COCO 标签、特征结果或人工结果。程序计算 `classification_correct = predicted_class == target_class`，模型不得自行报告准确率。

## 划分、审核和报告

固定划分为 calibration 100、development 200、validation 200。validation 规则冻结后只验证。人工审核覆盖随机 30 张、不同目标大小、遮挡/轮廓状态、多犬、边界、unknown、分类错误和模型分歧样本。

报告包含每个特征的状态分布、覆盖率、unknown 率、evidence 缺失/重复率、人工一致率、独立分类准确率、各状态错误率、可见性分组错误率及接口/格式错误。

## 验收

500 个唯一 ID，每张恰好一个固定 A 目标，三种输入图齐全，原始响应完整保存，失败/unknown/人工未审核分开记录，协议可重算且不覆盖历史结果。v1 结果保留为基线，v2 使用新的协议版本和输出目录。

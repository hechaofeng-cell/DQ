#!/usr/bin/env node

const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');
const { imageSize } = require('image-size');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'QE Project';
pptx.company = 'QE';
pptx.subject = '视觉测试数据充分性组会汇报';
pptx.title = '面向视觉测试数据的可解释状态覆盖与缺口驱动补样';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};
pptx.defineSlideMaster({
  title: 'QE_MASTER',
  background: { color: 'F7F9FB' },
  objects: [],
  slideNumber: { x: 12.55, y: 7.08, w: 0.35, h: 0.18, fontFace: 'Aptos', fontSize: 9, color: '77838F', align: 'right' },
});

const OUT = path.resolve(__dirname, '../docs/QE_group_meeting_visual_test_adequacy_20260915.pptx');
const IMG_WORKFLOW = path.resolve(__dirname, '../output/imagegen/dog_v3_1_workflow_zh_polished.png');
const IMG_COVERAGE = path.resolve(__dirname, '../docs/dog_v3_1_figures/fig1_coverage_before_after.png');
const IMG_GAPS = path.resolve(__dirname, '../docs/dog_v3_1_figures/fig2_gap_composition.png');
const IMG_UNKNOWN = path.resolve(__dirname, '../docs/dog_v3_1_figures/fig4_unknown_rate.png');
const DOG_EXAMPLES = [
  {
    path: path.resolve(__dirname, '../artifacts/dog_candidate400_v2_1_inputs_20260912/marked/000000247068.jpg'),
    title: '多犬与人物共现',
    states: 'multiple_dogs · interacting',
  },
  {
    path: path.resolve(__dirname, '../artifacts/dog_candidate400_v2_1_inputs_20260912/marked/000000438744.jpg'),
    title: '背向与雪地场景',
    states: 'head_back · outdoor_snow',
  },
  {
    path: path.resolve(__dirname, '../artifacts/dog_candidate400_v2_1_inputs_20260912/marked/000000495454.jpg'),
    title: '极小目标',
    states: 'tiny_target · back_view',
  },
  {
    path: path.resolve(__dirname, '../artifacts/dog_candidate400_v2_1_inputs_20260912/marked/000000280709.jpg'),
    title: '跳跃与交互',
    states: 'jumping · person_cooccurrence',
  },
];

const C = {
  bg: 'F7F9FB',
  white: 'FFFFFF',
  ink: '17212B',
  muted: '5F6B76',
  line: 'D8E0E6',
  navy: '153B5B',
  blue: '2E6F9E',
  blueSoft: 'E7F0F7',
  teal: '187D7B',
  tealSoft: 'E5F3F1',
  orange: 'E87524',
  orangeSoft: 'FCECDD',
  green: '2F7D5A',
  greenSoft: 'E7F2EC',
  red: 'B84A3A',
  redSoft: 'F8E7E4',
  purple: '7C527C',
  purpleSoft: 'F1EAF2',
};

const S = pptx.ShapeType;

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || 'Microsoft YaHei',
    fontSize: opts.fontSize || 20,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    align: opts.align || 'left',
    valign: opts.valign || 'mid',
    margin: opts.margin === undefined ? 0 : opts.margin,
    breakLine: false,
    fit: 'shrink',
    paraSpaceAfterPt: 0,
    lineSpacingMultiple: 1.0,
    ...opts,
  });
}

function addBox(slide, x, y, w, h, fill, line = C.line, radius = 0.08) {
  slide.addShape(radius ? S.roundRect : S.rect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill },
    line: { color: line, width: 1 },
    radius,
  });
}

function addRule(slide, x, y, w, color = C.line, width = 1) {
  slide.addShape(S.line, { x, y, w, h: 0, line: { color, width } });
}

function addTitle(slide, section, title, subtitle) {
  addText(slide, section.toUpperCase(), 0.62, 0.28, 3.1, 0.25, {
    fontFace: 'Aptos', fontSize: 10, bold: true, color: C.teal, charSpacing: 1.4,
  });
  addText(slide, title, 0.62, 0.62, 12.0, 0.58, { fontSize: 29, bold: true, color: C.navy });
  if (subtitle) addText(slide, subtitle, 0.64, 1.20, 11.9, 0.34, { fontSize: 13, color: C.muted });
  addRule(slide, 0.62, subtitle ? 1.64 : 1.38, 12.05, C.line, 1);
}

function addFooter(slide, label = 'MAIN') {
  addText(slide, `QE · ${label}`, 0.62, 7.05, 1.4, 0.18, {
    fontFace: 'Aptos', fontSize: 8.5, bold: true, color: C.muted,
  });
}

function addPill(slide, text, x, y, w, fill, color = C.ink) {
  slide.addShape(S.roundRect, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color: fill }, line: { color: fill } });
  addText(slide, text, x + 0.08, y + 0.02, w - 0.16, 0.3, { fontSize: 11, bold: true, color, align: 'center' });
}

function addArrow(slide, x1, y1, x2, y2, color = C.blue, width = 2) {
  slide.addShape(S.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: 'none', endArrowType: 'triangle' },
  });
}

function addNumber(slide, n, x, y, fill = C.navy) {
  slide.addShape(S.ellipse, { x, y, w: 0.38, h: 0.38, fill: { color: fill }, line: { color: fill } });
  addText(slide, String(n), x, y + 0.01, 0.38, 0.34, { fontFace: 'Aptos', fontSize: 13, bold: true, color: C.white, align: 'center' });
}

function addSource(slide, text, url) {
  addText(slide, text, 0.64, 6.75, 11.9, 0.2, {
    fontSize: 8.5, color: '6E7A84', italic: true,
    hyperlink: url ? { url } : undefined,
  });
}

function addMetric(slide, value, label, x, y, w, color, detail) {
  addBox(slide, x, y, w, 1.12, C.white, C.line);
  slide.addShape(S.rect, { x, y, w: 0.08, h: 1.12, fill: { color }, line: { color } });
  addText(slide, value, x + 0.24, y + 0.13, w - 0.36, 0.43, { fontFace: 'Aptos', fontSize: 25, bold: true, color });
  addText(slide, label, x + 0.24, y + 0.57, w - 0.36, 0.25, { fontSize: 12.5, bold: true, color: C.ink });
  if (detail) addText(slide, detail, x + 0.24, y + 0.84, w - 0.36, 0.17, { fontSize: 9.5, color: C.muted });
}

function addImageContain(slide, imagePath, x, y, w, h) {
  const size = imageSize(fs.readFileSync(imagePath));
  const imageRatio = size.width / size.height;
  const boxRatio = w / h;
  let drawX = x;
  let drawY = y;
  let drawW = w;
  let drawH = h;
  if (imageRatio > boxRatio) {
    drawH = w / imageRatio;
    drawY += (h - drawH) / 2;
  } else {
    drawW = h * imageRatio;
    drawX += (w - drawW) / 2;
  }
  slide.addImage({ path: imagePath, x: drawX, y: drawY, w: drawW, h: drawH });
}

// 1. Title
{
  const slide = pptx.addSlide('QE_MASTER');
  slide.background = { color: C.bg };
  slide.addShape(S.rect, { x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: C.teal }, line: { color: C.teal } });
  addPill(slide, 'QE · 组会汇报', 0.72, 0.62, 1.75, C.tealSoft, C.teal);
  addText(slide, '面向视觉测试数据的\n可解释状态覆盖与缺口驱动补样', 0.72, 1.35, 11.6, 1.55, {
    fontSize: 34, bold: true, color: C.navy, breakLine: true, valign: 'top',
  });
  addText(slide, '从深度学习测试充分性研究到犬类图像初步验证', 0.75, 3.05, 10.4, 0.44, {
    fontSize: 18, color: C.muted,
  });
  const xs = [0.78, 3.7, 6.62, 9.54];
  addText(slide, '研究闭环概览｜可解释测量与缺口诊断已跑通；等预算对照和独立验证待完成', 0.78, 4.10, 11.04, 0.30, {
    fontSize: 13, bold: true, color: C.muted,
  });
  const labels = ['可解释特征', '覆盖缺口', '预算化补样', '独立验证'];
  const fills = [C.blueSoft, C.tealSoft, C.orangeSoft, C.greenSoft];
  const colors = [C.blue, C.teal, C.orange, C.green];
  for (let i = 0; i < 4; i++) {
    addBox(slide, xs[i], 4.65, 2.28, 0.82, fills[i], colors[i]);
    addNumber(slide, i + 1, xs[i] + 0.18, 4.87, colors[i]);
    addText(slide, labels[i], xs[i] + 0.68, 4.81, 1.36, 0.37, { fontSize: 15, bold: true, color: colors[i] });
    if (i < 3) addArrow(slide, xs[i] + 2.36, 5.06, xs[i + 1] - 0.12, 5.06, C.line, 2);
  }
  addRule(slide, 0.75, 6.42, 11.86, C.line, 1);
  addText(slide, '组会汇报 · 2026.09.15', 0.75, 6.58, 4.0, 0.28, { fontFace: 'Aptos', fontSize: 12, color: C.muted });
  addText(slide, '研究对象：测试数据集，而非单张图片或单一模型分数', 6.3, 6.52, 6.3, 0.36, { fontSize: 12, color: C.red, bold: true, align: 'right' });
  slide.addNotes('先提出问题：图片很多是否就代表测试充分？再指向页面下方的研究闭环。第一，用可解释特征把图片转换成视觉状态；第二，统计缺失或低频状态；第三，在固定预算下进行定向补样并与基线比较；第四，用独立测试集验证模型收益。当前已经跑通状态提取、缺口诊断和定向补图，等预算对照与独立模型验证尚未完成。');
}

// 2. Motivation
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '01 · MOTIVATION', '同一类别内部，测试条件差异很大', '真实补图候选显示：dog 不是一个单一状态，而是一组可解释的视觉变化');
  const cardXs = [0.66, 3.83, 7.00, 10.17];
  const cardColors = [C.purple, C.teal, C.orange, C.green];
  DOG_EXAMPLES.forEach((example, i) => {
    const x = cardXs[i];
    addBox(slide, x, 1.92, 2.53, 3.78, C.white, C.line);
    addBox(slide, x + 0.16, 2.10, 2.21, 1.78, 'EEF2F4', 'EEF2F4');
    addImageContain(slide, example.path, x + 0.16, 2.10, 2.21, 1.78);
    addText(slide, example.title, x + 0.18, 4.16, 2.17, 0.34, { fontSize: 15, bold: true, color: cardColors[i], align: 'center' });
    addPill(slide, example.states, x + 0.22, 4.70, 2.09, i === 2 ? C.orangeSoft : (i === 3 ? C.greenSoft : (i === 1 ? C.tealSoft : C.purpleSoft)), cardColors[i]);
    addText(slide, `候选排名 #${[1, 2, 3, 6][i]}`, x + 0.18, 5.26, 2.17, 0.22, { fontFace: 'Aptos', fontSize: 9.5, color: C.muted, align: 'center' });
  });
  addBox(slide, 1.12, 5.99, 11.09, 0.56, C.navy, C.navy);
  addText(slide, '只知道“有多少张 dog”还不够：这些状态是否出现、是否有足够证据？', 1.35, 6.13, 10.63, 0.29, { fontSize: 18, bold: true, color: C.white, align: 'center' });
  addSource(slide, '真实补图候选（COCO 2017）；红框为程序固定的 A 目标，状态来自 dog_v3.1 结构化输出。');
  addFooter(slide);
  slide.addNotes('这四张图来自实际补图候选。它们分别体现多犬和人物共现、背向雪地、极小目标、跳跃交互。重点不是逐张解释所有标签，而是让听众直观看到：同一个dog类别内部有完全不同的测试条件。只统计500张dog不能告诉我们这些状态是否出现，也不能告诉我们是否有足够证据评价模型。');
}

// 3. Concepts
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '01 · MOTIVATION', '三个概念必须分开', '覆盖是“触及了什么”，充分性是“证据是否够用”');
  const items = [
    { x: 0.86, n: '01', title: '模型表现', en: 'Performance', q: '已有图片上\n模型表现怎样？', ex: '准确率、错误率、校准', fill: C.blueSoft, color: C.blue },
    { x: 4.49, n: '02', title: '条件覆盖', en: 'Coverage', q: '测试集包含了\n哪些视觉条件？', ex: '视角、尺度、遮挡、场景', fill: C.tealSoft, color: C.teal },
    { x: 8.12, n: '03', title: '测试充分性', en: 'Adequacy', q: '这些证据是否足以\n支持当前评价目的？', ex: '需要目标、样本量与效用', fill: C.orangeSoft, color: C.orange },
  ];
  items.forEach((it, idx) => {
    addBox(slide, it.x, 1.98, 3.34, 3.64, C.white, C.line);
    slide.addShape(S.rect, { x: it.x, y: 1.98, w: 3.34, h: 0.1, fill: { color: it.color }, line: { color: it.color } });
    addText(slide, it.n, it.x + 0.25, 2.28, 0.62, 0.36, { fontFace: 'Aptos', fontSize: 20, bold: true, color: it.color });
    addText(slide, it.title, it.x + 0.25, 2.76, 2.45, 0.42, { fontSize: 21, bold: true, color: C.navy });
    addText(slide, it.en, it.x + 0.25, 3.2, 2.45, 0.25, { fontFace: 'Aptos', fontSize: 11, color: C.muted });
    addBox(slide, it.x + 0.25, 3.69, 2.84, 0.94, it.fill, it.fill);
    addText(slide, it.q, it.x + 0.43, 3.86, 2.48, 0.58, { fontSize: 15, bold: true, color: it.color, align: 'center', breakLine: true });
    addText(slide, it.ex, it.x + 0.25, 4.92, 2.84, 0.3, { fontSize: 12.5, color: C.muted, align: 'center' });
    if (idx < 2) addArrow(slide, it.x + 3.43, 3.82, it.x + 3.60, 3.82, C.line, 2);
  });
  addBox(slide, 1.42, 5.98, 10.5, 0.58, C.redSoft, C.red);
  addText(slide, '出现一次只表示“触及”；充分性要求相对于测试目标具有足够证据。', 1.65, 6.12, 10.04, 0.3, { fontSize: 17, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('强调三个概念不能互相替代。嘴部开合程度是单图属性；图片难度是图片相对于模型的表现；覆盖和充分性是整批数据相对于目标的性质。一个状态出现一次只说明被触及，不代表已经有足够样本稳定估计该条件下的模型表现。');
}

// 4. Literature roadmap
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '02 · RELATED WORK', '相关研究怎样一步步接近这个问题', '从网络内部反馈，走向输入条件、可解释诊断与测试收益');
  const cols = [
    { x: 0.66, year: '2017–2018', name: 'DeepXplore\nDeepGauge', role: '网络行为覆盖', desc: '用内部活动指导测试', fill: C.blueSoft, color: C.blue },
    { x: 3.14, year: '2019–2023', name: 'IDC', role: '输入分布覆盖', desc: '潜空间中的特征组合', fill: C.tealSoft, color: C.teal },
    { x: 5.62, year: '2021', name: 'DeepHyperion', role: '可解释特征地图', desc: '将行为放入可读条件空间', fill: C.purpleSoft, color: C.purple },
    { x: 8.10, year: '2019', name: 'Asudeh et al.', role: '缺口与补足', desc: '属性组合欠覆盖与新增数据', fill: C.orangeSoft, color: C.orange },
    { x: 10.58, year: '2024', name: 'TEASMA\nSELECT', role: '效用与公平基线', desc: '检错能力、统一预算比较', fill: C.greenSoft, color: C.green },
  ];
  addRule(slide, 1.02, 3.0, 11.15, '9EB0BD', 2.5);
  cols.forEach((it, i) => {
    slide.addShape(S.ellipse, { x: it.x + 0.86, y: 2.78, w: 0.42, h: 0.42, fill: { color: it.color }, line: { color: C.white, width: 2 } });
    addText(slide, it.year, it.x + 0.2, 2.13, 1.75, 0.3, { fontFace: 'Aptos', fontSize: 11, bold: true, color: it.color, align: 'center' });
    addBox(slide, it.x, 3.38, 2.17, 2.08, C.white, C.line);
    slide.addShape(S.rect, { x: it.x, y: 3.38, w: 2.17, h: 0.08, fill: { color: it.color }, line: { color: it.color } });
    addText(slide, it.name, it.x + 0.17, 3.64, 1.83, 0.55, { fontFace: 'Aptos', fontSize: 15, bold: true, color: C.navy, align: 'center', breakLine: true });
    addText(slide, it.role, it.x + 0.17, 4.29, 1.83, 0.31, { fontSize: 13, bold: true, color: it.color, align: 'center' });
    addText(slide, it.desc, it.x + 0.2, 4.76, 1.77, 0.44, { fontSize: 10.5, color: C.muted, align: 'center' });
  });
  addText(slide, '不是简单替代：覆盖对象不同；共同趋势是走向可解释、可行动、可验证。', 1.15, 5.98, 11.0, 0.45, { fontSize: 19, bold: true, color: C.navy, align: 'center' });
  addFooter(slide);
  slide.addNotes('这页只交代发展脉络，不逐篇展开结果。路线不是简单的优劣替代：内部覆盖、输入覆盖、语义覆盖分别回答不同问题。我要强调的是，研究正在从抽象内部反馈向可解释、可行动、可验证的测试证据推进。');
}

// 5. DeepXplore
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '02 · RELATED WORK', 'DeepXplore：让深度学习测试拥有可计算反馈', '白盒覆盖 + 差分行为 + 梯度搜索');
  const y = 2.18;
  const nodes = [
    { x: 0.86, t: '测试输入', sub: 'image x', fill: C.blueSoft, color: C.blue },
    { x: 3.25, t: '神经元覆盖', sub: '内部活动', fill: C.blueSoft, color: C.blue },
    { x: 5.64, t: '模型差分', sub: '交叉参考', fill: C.orangeSoft, color: C.orange },
    { x: 8.03, t: '联合优化', sub: '梯度搜索', fill: C.purpleSoft, color: C.purple },
    { x: 10.42, t: '边角输入', sub: '触发异常行为', fill: C.greenSoft, color: C.green },
  ];
  nodes.forEach((n, i) => {
    addBox(slide, n.x, y, 1.86, 1.05, n.fill, n.color);
    addText(slide, n.t, n.x + 0.1, y + 0.18, 1.66, 0.32, { fontSize: 15, bold: true, color: n.color, align: 'center' });
    addText(slide, n.sub, n.x + 0.1, y + 0.59, 1.66, 0.23, { fontSize: 10.5, color: C.muted, align: 'center' });
    if (i < nodes.length - 1) addArrow(slide, n.x + 1.92, y + 0.53, nodes[i + 1].x - 0.06, y + 0.53, '9BAAB5', 1.5);
  });
  const deepXploreCards = [
    { x: 0.86, label: '解决什么问题', text: '传统代码覆盖难以直接用于 DNN；新输入还经常缺少人工测试预言。', fill: C.blueSoft, color: C.blue },
    { x: 4.46, label: '核心思路', text: '用神经元覆盖提供反馈，以多模型差分暴露潜在错误，再用梯度搜索生成测试。', fill: C.greenSoft, color: C.green },
    { x: 8.06, label: '与我的关系', text: '它证明覆盖可以量化；但内部激活难直接回答测试数据具体缺少哪类图片。', fill: C.orangeSoft, color: C.orange },
  ];
  deepXploreCards.forEach((c) => {
    addBox(slide, c.x, 3.72, 3.40, 1.74, C.white, C.line);
    addPill(slide, c.label, c.x + 0.22, 3.96, 1.34, c.fill, c.color);
    addText(slide, c.text, c.x + 0.22, 4.43, 2.96, 0.76, { fontSize: 14.5, bold: true, color: C.ink, valign: 'top' });
  });
  addBox(slide, 1.16, 5.82, 11.0, 0.63, C.purpleSoft, C.purple);
  addText(slide, '补充：DeepGauge 将内部覆盖细化为激活区间和边界覆盖，但仍属于模型内部状态。', 1.42, 5.97, 10.48, 0.31, { fontSize: 15.5, bold: true, color: C.purple, align: 'center' });
  addSource(slide, 'Pei et al., DeepXplore, SOSP 2017；Ma et al., DeepGauge, ASE 2018', 'https://arxiv.org/abs/1705.06640');
  addFooter(slide);
  slide.addNotes('讲法：DeepXplore的重要性在于提供了测试反馈，而不是说神经元覆盖等于测试充分性。它通过神经元覆盖和多个同功能模型之间的差分行为来搜索异常输入。对我的工作而言，它留下的关键问题是：内部活动变化很难直接翻译成应该补充哪一种图片。');
}

// 6. IDC
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '02 · RELATED WORK', 'IDC：把测试充分性转向输入分布', '在缺少显式输入规格时，以训练数据学习覆盖域');
  const y = 2.25;
  addBox(slide, 0.86, y, 2.05, 1.18, C.white, C.line);
  addText(slide, '训练数据', 1.06, y + 0.18, 1.65, 0.32, { fontSize: 17, bold: true, color: C.navy, align: 'center' });
  addText(slide, '隐式输入规格', 1.06, y + 0.67, 1.65, 0.25, { fontSize: 11, color: C.muted, align: 'center' });
  addArrow(slide, 2.98, y + 0.59, 3.54, y + 0.59, C.teal, 2);
  addBox(slide, 3.6, y, 2.05, 1.18, C.tealSoft, C.teal);
  addText(slide, 'VAE', 3.8, y + 0.18, 1.65, 0.32, { fontFace: 'Aptos', fontSize: 19, bold: true, color: C.teal, align: 'center' });
  addText(slide, '低维潜空间', 3.8, y + 0.67, 1.65, 0.25, { fontSize: 11, color: C.muted, align: 'center' });
  addArrow(slide, 5.72, y + 0.59, 6.28, y + 0.59, C.teal, 2);
  addBox(slide, 6.34, y, 2.05, 1.18, C.blueSoft, C.blue);
  addText(slide, '可行区域', 6.54, y + 0.18, 1.65, 0.32, { fontSize: 17, bold: true, color: C.blue, align: 'center' });
  addText(slide, '特征组合', 6.54, y + 0.67, 1.65, 0.25, { fontSize: 11, color: C.muted, align: 'center' });
  addArrow(slide, 8.46, y + 0.59, 9.02, y + 0.59, C.teal, 2);
  addBox(slide, 9.08, y, 3.22, 1.18, C.orangeSoft, C.orange);
  addText(slide, 'Input Distribution Coverage', 9.25, y + 0.17, 2.88, 0.36, { fontFace: 'Aptos', fontSize: 14, bold: true, color: C.orange, align: 'center' });
  addText(slide, '测试输入覆盖了哪些组合', 9.25, y + 0.67, 2.88, 0.25, { fontSize: 11, color: C.muted, align: 'center' });
  const idcCards = [
    { x: 0.86, label: '解决什么问题', text: '神经网络缺少显式输入规格，因此难以判断测试集覆盖了哪些有效输入区域。', fill: C.blueSoft, color: C.blue },
    { x: 4.46, label: '核心思路', text: '把训练数据当作隐式规格，用 VAE 学习潜空间，再统计测试输入覆盖的可行特征组合。', fill: C.tealSoft, color: C.teal },
    { x: 8.06, label: '与我的关系', text: '它把覆盖转向黑盒输入域；我进一步使用显式视觉状态，让缺口能直接变成补图条件。', fill: C.orangeSoft, color: C.orange },
  ];
  idcCards.forEach((c) => {
    addBox(slide, c.x, 3.88, 3.40, 1.72, C.white, C.line);
    addPill(slide, c.label, c.x + 0.22, 4.11, 1.34, c.fill, c.color);
    addText(slide, c.text, c.x + 0.22, 4.57, 2.96, 0.76, { fontSize: 14.5, bold: true, color: C.ink, valign: 'top' });
  });
  addBox(slide, 1.16, 5.91, 11.0, 0.55, C.redSoft, C.red);
  addText(slide, '边界：显式状态更便于解释和行动，但不能据此声称它的表达能力一定优于潜变量。', 1.42, 6.04, 10.48, 0.29, { fontSize: 14.5, bold: true, color: C.red, align: 'center' });
  addSource(slide, 'Dola, Dwyer & Soffa, Input Distribution Coverage, ACM TOSEM 2023 · DOI:10.1145/3576040', 'https://doi.org/10.1145/3576040');
  addFooter(slide);
  slide.addNotes('IDC是与本方向最直接的输入覆盖工作。它用VAE从训练数据学习低维表示，再把潜空间中的特征组合当作覆盖域。这里要中性地讲：IDC解决了没有显式输入规格时怎样构造黑盒覆盖，不要贬低它。我的问题是，它学到的维度未必能够直接告诉数据人员应该补充侧面、遮挡还是小目标图片。');
}

// 7. Interpretable + remedy
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '02 · RELATED WORK', '可解释条件与缺口补足：离当前工作更近一步', '一类工作定义可读的覆盖空间，另一类工作把缺口转成数据行动');
  addBox(slide, 0.74, 1.92, 5.83, 3.88, C.white, C.line);
  slide.addShape(S.rect, { x: 0.74, y: 1.92, w: 5.83, h: 0.1, fill: { color: C.purple }, line: { color: C.purple } });
  addPill(slide, 'DeepHyperion · ISSTA 2021', 1.05, 2.22, 2.48, C.purpleSoft, C.purple);
  addText(slide, '可解释特征地图', 1.05, 2.73, 4.9, 0.38, { fontSize: 21, bold: true, color: C.navy });
  addText(slide, '解决问题', 1.05, 3.28, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.purple });
  addText(slide, '生成了很多测试输入后，仍难说明系统究竟在哪些输入条件下容易失效。', 2.03, 3.23, 4.08, 0.48, { fontSize: 13.5, color: C.ink });
  addText(slide, '核心思路', 1.05, 4.04, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.purple });
  addText(slide, '用领域可理解的特征作为地图坐标，通过 illumination search 探索不同区域并记录行为。', 2.03, 3.96, 4.08, 0.58, { fontSize: 13.5, color: C.ink });
  addText(slide, '与我关系', 1.05, 4.88, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.purple });
  addText(slide, '说明视觉状态可以成为测试空间坐标，使覆盖缺口能够对应具体、可读的变化。', 2.03, 4.80, 4.08, 0.58, { fontSize: 13.5, bold: true, color: C.ink });

  addBox(slide, 6.78, 1.92, 5.83, 3.88, C.white, C.line);
  slide.addShape(S.rect, { x: 6.78, y: 1.92, w: 5.83, h: 0.1, fill: { color: C.orange }, line: { color: C.orange } });
  addPill(slide, 'Asudeh et al. · ICDE 2019', 7.09, 2.22, 2.46, C.orangeSoft, C.orange);
  addText(slide, '属性组合欠覆盖与补足', 7.09, 2.73, 4.9, 0.38, { fontSize: 21, bold: true, color: C.navy });
  addText(slide, '解决问题', 7.09, 3.28, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '多维离散属性组合中，哪些群体覆盖不足，以及还需要补充多少数据？', 8.07, 3.23, 4.08, 0.48, { fontSize: 13.5, color: C.ink });
  addText(slide, '核心思路', 7.09, 4.04, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '先定位 under-covered groups，再把数据补充转化为面向属性组合缺口的选择问题。', 8.07, 3.96, 4.08, 0.58, { fontSize: 13.5, color: C.ink });
  addText(slide, '与我关系', 7.09, 4.88, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '直接支持“发现低频视觉状态，再从候选池定向补图”的研究思路。', 8.07, 4.80, 4.08, 0.58, { fontSize: 13.5, bold: true, color: C.ink });
  addBox(slide, 1.34, 6.06, 10.65, 0.5, C.redSoft, C.red);
  addText(slide, '共同前提尚未自动成立：特征必须与任务相关，而且标签本身必须可靠。', 1.56, 6.18, 10.21, 0.28, { fontSize: 17, bold: true, color: C.red, align: 'center' });
  addSource(slide, 'Zohdinasab et al., DeepHyperion, ISSTA 2021；Asudeh et al., ICDE 2019', 'https://arxiv.org/abs/2107.06997');
  addFooter(slide);
  slide.addNotes('DeepHyperion说明可解释特征可以成为测试地图的坐标。Asudeh等人的工作说明，离散属性组合中的欠覆盖不仅可以被发现，还可以进一步计算需要补多少数据。这两篇共同构成我方法的桥梁。但它们也暴露一个前提：视觉图像中的属性从哪里来，是否可信，哪些组合才有任务意义。');
}

// 8. Utility and baselines
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '02 · RELATED WORK', '覆盖分数不是终点：还要回答“有没有测试收益”', 'TEASMA约束效用结论，SELECT约束策略比较');
  addBox(slide, 0.76, 1.95, 5.8, 3.88, C.white, C.line);
  addPill(slide, 'TEASMA · IEEE TSE 2024', 1.07, 2.23, 2.27, C.greenSoft, C.green);
  addText(slide, '把覆盖连接到故障检出', 1.07, 2.73, 4.95, 0.38, { fontSize: 21, bold: true, color: C.navy });
  addText(slide, '解决问题', 1.07, 3.28, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.green });
  addText(slide, '一个测试集的充分性分数，是否真的对应更强的故障发现能力？', 2.05, 3.23, 4.05, 0.48, { fontSize: 13.5, color: C.ink });
  addText(slide, '核心思路', 1.07, 4.04, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.green });
  addText(slide, '利用已有充分性指标，为具体 DNN 建立测试集故障检出率预测模型。', 2.05, 3.96, 4.05, 0.52, { fontSize: 13.5, color: C.ink });
  addText(slide, '对我启示', 1.07, 4.86, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.green });
  addText(slide, '覆盖不是终点，必须再用新增错误或独立故障检出进行验证。', 2.05, 4.78, 4.05, 0.58, { fontSize: 13.5, bold: true, color: C.ink });

  addBox(slide, 6.78, 1.95, 5.8, 3.88, C.white, C.line);
  addPill(slide, 'SELECT · NeurIPS 2024', 7.09, 2.23, 2.12, C.orangeSoft, C.orange);
  addText(slide, '统一预算下比较数据选择', 7.09, 2.73, 4.95, 0.38, { fontSize: 21, bold: true, color: C.navy });
  addText(slide, '研究范围', 7.09, 3.28, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '它研究训练数据策展，而不是测试充分性；这里不能把两类任务混为一谈。', 8.07, 3.23, 4.05, 0.48, { fontSize: 13.5, color: C.ink });
  addText(slide, '核心原则', 7.09, 4.04, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '不同选择策略使用同一数据预算、同一任务和同一评价条件，再比较下游表现。', 8.07, 3.96, 4.05, 0.52, { fontSize: 13.5, color: C.ink });
  addText(slide, '对我启示', 7.09, 4.86, 0.92, 0.26, { fontSize: 12.5, bold: true, color: C.orange });
  addText(slide, '定向补 100 张必须与随机补 100 张公平比较，不能只报告覆盖率上升。', 8.07, 4.78, 4.05, 0.58, { fontSize: 13.5, bold: true, color: C.ink });
  addBox(slide, 1.42, 6.1, 10.5, 0.48, C.orangeSoft, C.orange);
  addText(slide, '注意：SELECT研究训练数据策展；这里借鉴的是“统一预算与基线”的评价原则。', 1.63, 6.21, 10.08, 0.28, { fontSize: 14, bold: true, color: C.orange, align: 'center' });
  addSource(slide, 'Abbasishahkoo et al., TEASMA, IEEE TSE 2024；Feuer et al., SELECT, NeurIPS 2024', 'https://arxiv.org/abs/2308.01311');
  addFooter(slide);
  slide.addNotes('这一页决定后面怎样诚实解释自己的结果。TEASMA提醒我们，充分性指标要和故障检出能力建立联系。SELECT虽然研究的是训练数据策展，不是测试充分性，但它强调同一预算和统一基线。由此可知，仅仅让自己定义的覆盖指数升高，不能证明补样策略更好。');
}

// 9. Gaps
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '03 · RESEARCH GAP', '文献不是“没做”，而是链路仍然断开', '充分性评价至少包含测量、规格和效用三层证据');
  const x0 = 0.68, y0 = 2.28;
  const blocks = [
    { x: x0, title: '图像', sub: '原始测试数据', fill: C.blueSoft, color: C.blue },
    { x: x0 + 3.12, title: '可解释状态', sub: '视角 / 遮挡 / 场景', fill: C.tealSoft, color: C.teal },
    { x: x0 + 6.24, title: '覆盖缺口', sub: '缺失 / 稀疏 / 组合', fill: C.orangeSoft, color: C.orange },
    { x: x0 + 9.36, title: '测试收益', sub: '新增证据 / 独立错误', fill: C.greenSoft, color: C.green },
  ];
  blocks.forEach((b) => {
    addBox(slide, b.x, y0, 2.45, 1.2, b.fill, b.color);
    addText(slide, b.title, b.x + 0.14, y0 + 0.22, 2.17, 0.36, { fontSize: 19, bold: true, color: b.color, align: 'center' });
    addText(slide, b.sub, b.x + 0.14, y0 + 0.72, 2.17, 0.24, { fontSize: 10.5, color: C.muted, align: 'center' });
  });
  const gaps = [
    { x: 3.14, n: '断点 1', t: '属性测量是否可靠？', c: C.red },
    { x: 6.26, n: '断点 2', t: '缺口相对什么目标？', c: C.red },
    { x: 9.38, n: '断点 3', t: '补样是否真的有用？', c: C.red },
  ];
  gaps.forEach((g, i) => {
    addArrow(slide, g.x - 0.35, y0 + 0.6, g.x + 0.31, y0 + 0.6, '9EADB8', 1.5);
    slide.addShape(S.ellipse, { x: g.x - 0.03, y: y0 + 0.4, w: 0.4, h: 0.4, fill: { color: C.red }, line: { color: C.white, width: 1.5 } });
    addText(slide, '!', g.x - 0.03, y0 + 0.4, 0.4, 0.36, { fontFace: 'Aptos', fontSize: 17, bold: true, color: C.white, align: 'center' });
    addPill(slide, g.n, g.x - 0.73, 3.88, 1.8, C.redSoft, C.red);
    addText(slide, g.t, g.x - 1.0, 4.44, 2.35, 0.45, { fontSize: 14.5, bold: true, color: C.red, align: 'center' });
  });
  addBox(slide, 1.25, 5.38, 10.83, 0.89, C.white, C.navy);
  addText(slide, '研究机会', 1.54, 5.65, 1.12, 0.33, { fontSize: 16, bold: true, color: C.navy });
  addText(slide, '把可靠属性测量、任务约束、预算化补样和独立验证组织成一个可审计闭环。', 2.7, 5.57, 9.05, 0.49, { fontSize: 19, bold: true, color: C.navy, align: 'center' });
  addFooter(slide);
  slide.addNotes('转折句可以直接说：这些工作不是没有解决问题，而是分别解决了不同环节。真正困难的是把环节连接起来。第一，VLM输出确定标签并不等于标签正确；第二，没有任务要求时只能描述分布，不能断言缺失就是不足；第三，按覆盖目标选图后覆盖上升具有一定必然性，必须再用独立收益检验。');
}

// 10. Research questions
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '04 · MY DIRECTION', '我的研究方向：从“测量表”走向“决策闭环”', '黑盒条件下，诊断视觉测试数据缺少哪些可解释证据，并指导有限预算行动');
  addBox(slide, 0.8, 1.92, 11.73, 0.79, C.navy, C.navy);
  addText(slide, '给定测试目标与候选数据，如何识别缺少证据的视觉状态，并给出可验证的复核与补样决策？', 1.08, 2.11, 11.17, 0.37, { fontSize: 20, bold: true, color: C.white, align: 'center' });
  const rqs = [
    { x: 0.82, id: 'RQ1', title: '可靠测量', body: '如何将图片稳定转换为\n有限、可核验的视觉状态？', tags: 'schema · unknown · 人工复核', fill: C.blueSoft, color: C.blue },
    { x: 4.48, id: 'RQ2', title: '缺口定义', body: '如何依据测试目标判断\n哪些状态缺失或证据不足？', tags: '需求 · 参照 · 组合约束', fill: C.tealSoft, color: C.teal },
    { x: 8.14, id: 'RQ3', title: '行动与收益', body: '固定预算下，如何分配\n复核与补样并验证收益？', tags: '随机基线 · 故障 · 成本', fill: C.orangeSoft, color: C.orange },
  ];
  rqs.forEach((r) => {
    addBox(slide, r.x, 3.08, 3.38, 2.53, C.white, C.line);
    addPill(slide, r.id, r.x + 0.25, 3.35, 0.76, r.color, C.white);
    addText(slide, r.title, r.x + 1.16, 3.34, 1.8, 0.33, { fontSize: 19, bold: true, color: r.color });
    addText(slide, r.body, r.x + 0.25, 4.0, 2.88, 0.72, { fontSize: 16, bold: true, color: C.ink, align: 'center', breakLine: true });
    addPill(slide, r.tags, r.x + 0.33, 5.02, 2.72, r.fill, r.color);
  });
  addText(slide, '当前犬类实验覆盖 RQ1、RQ2 与初步补样；RQ3 的独立效用验证尚未完成。', 1.08, 6.12, 11.16, 0.39, { fontSize: 18, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('这一页给出我的正式研究问题。定位不是再做一个编码器，也不是声称提出通用充分性理论，而是做可解释、可追溯的测量和决策流程。当前dog实验已经把图片转为状态、统计缺口并做了候选补图，但尚未完成随机基线和独立模型故障收益验证。');
}

// 11. Dog workflow
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '05 · CURRENT WORK', '当前验证载体：dog_v3.1 犬类视觉状态覆盖流程', '固定目标、三视图输入、23项枚举特征、严格校验与缺口驱动选图');
  slide.addImage({ path: IMG_WORKFLOW, x: 0.65, y: 1.78, w: 12.03, h: 4.01 });
  addBox(slide, 0.86, 6.04, 11.61, 0.5, C.orangeSoft, C.orange);
  addText(slide, '为什么选 dog：COCO 有目标框、类内状态丰富；VLM只负责属性测量，不是被测识别模型。', 1.08, 6.16, 11.17, 0.28, { fontSize: 14.5, bold: true, color: C.orange, align: 'center' });
  addSource(slide, '项目流程与证据：docs/dog_v3_1_overall_workflow.md；docs/dog_v3_1_claim_evidence.md');
  addFooter(slide);
  slide.addNotes('按图从左到右讲，不逐项念文字。第一，初始500张和候选400张保持独立。第二，每张图由程序根据COCO框固定一个A目标，并提供原图、标记图和裁剪图。第三，VLM输出23项有限枚举特征。第四，程序严格校验JSON、枚举、依赖关系并保留原始响应。第五，根据低频状态从候选池选择100张，最后冻结清洁数据集。');
}

// 12. Results
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '05 · CURRENT WORK', '初步结果：覆盖能够被测量，也能够被定向改善', '但当前证据只支持“状态覆盖改善”，不支持“识别性能改善”');
  addBox(slide, 0.68, 1.86, 7.18, 4.46, C.white, C.line);
  slide.addImage({ path: IMG_COVERAGE, x: 0.94, y: 2.15, w: 6.65, h: 3.83 });
  addMetric(slide, '900 / 900', '严格特征响应通过校验', 8.16, 1.88, 2.02, C.blue, '500初始 + 400候选');
  addMetric(slide, '596', '最终清洁图片', 10.42, 1.88, 2.02, C.teal, '600合并 - 4非真实犬');
  addMetric(slide, '78.64% → 84.17%', '状态覆盖指数', 8.16, 3.25, 4.28, C.orange, '提升5.53个百分点');
  addMetric(slide, '45 → 33', '最终低频状态数量', 8.16, 4.62, 4.28, C.green, '清洁596张口径');
  addBox(slide, 0.88, 6.30, 3.72, 0.55, C.greenSoft, C.green);
  addText(slide, '观察｜覆盖 +5.53 pp，低频状态 45 → 33', 1.02, 6.42, 3.44, 0.28, { fontSize: 11.5, bold: true, color: C.green, align: 'center' });
  addBox(slide, 4.80, 6.30, 3.72, 0.55, C.orangeSoft, C.orange);
  addText(slide, '解释｜当前协议下，状态缺口得到改善', 4.94, 6.42, 3.44, 0.28, { fontSize: 11.5, bold: true, color: C.orange, align: 'center' });
  addBox(slide, 8.72, 6.30, 3.72, 0.55, C.redSoft, C.red);
  addText(slide, '边界｜不是识别性能；尚无随机基线', 8.86, 6.42, 3.44, 0.28, { fontSize: 11.5, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('先讲可支持结论：在固定dog_v3.1协议和当前候选池内，定向补图后覆盖指数由78.64%提高到84.17%，最终低频状态数从45降到33。然后主动说边界：900/900只表示输出通过严格格式和规则校验，不是23项特征全部识别正确；补图策略正是优化覆盖，所以还不能据此证明策略优于随机，更不能解释成识别准确率提高。');
}

// 13. Research method: total dataset -> dog subset -> supplement -> merged dataset
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '06 · RESEARCH METHOD', '第一阶段验证主线：D0 → dog 训练补样 → D1', '检验覆盖诊断的训练数据效用；测试集故障检出能力需要另行验证');
  const cards = [
    { x: 0.78, n: '1', title: '固定总集 D0', body: '按类别整理，固定\n训练 / 验证 / 测试划分', foot: '计划：统一模型得到\n基线指标 M_before', fill: C.blueSoft, color: C.blue },
    { x: 3.80, n: '2', title: '分析 dog 训练子集', body: '固定 COCO 目标\n输出23项视觉特征', foot: '当前已测：补图前覆盖率\nC_before = 78.64%', fill: C.tealSoft, color: C.teal },
    { x: 6.82, n: '3', title: '得到 Ddog+', body: '只在训练候选中\n选择补充图片', foot: '合并 dog 子集后已测\nC_after = 84.17%', fill: C.orangeSoft, color: C.orange },
    { x: 9.84, n: '4', title: '计划构建 D1', body: 'Ddog+ 合并进 D0_train\n保持相同实验配置', foot: '重训得到 M_after；\n固定 D_test 比较', fill: C.greenSoft, color: C.green },
  ];
  cards.forEach((c, i) => {
    addBox(slide, c.x, 1.93, 2.58, 3.02, C.white, C.line);
    slide.addShape(S.rect, { x: c.x, y: 1.93, w: 2.58, h: 0.1, fill: { color: c.color }, line: { color: c.color } });
    addNumber(slide, c.n, c.x + 0.22, 2.20, c.color);
    addText(slide, c.title, c.x + 0.70, 2.18, 1.63, 0.45, { fontSize: 17, bold: true, color: c.color, align: 'center' });
    addBox(slide, c.x + 0.20, 2.92, 2.18, 0.86, c.fill, c.fill);
    addText(slide, c.body, c.x + 0.30, 3.10, 1.98, 0.52, { fontSize: 14, bold: true, color: C.ink, align: 'center', breakLine: true });
    addText(slide, c.foot, c.x + 0.22, 4.12, 2.14, 0.58, { fontSize: 11.5, color: C.muted, align: 'center', breakLine: true });
    if (i < cards.length - 1) addArrow(slide, c.x + 2.66, 3.44, c.x + 2.95, 3.44, '9DAEB9', 1.8);
  });
  addBox(slide, 0.78, 5.38, 11.64, 0.78, C.navy, C.navy);
  addText(slide, '两条结果线分别报告', 1.06, 5.55, 2.05, 0.25, { fontSize: 14, bold: true, color: '8ED1CC' });
  addText(slide, '覆盖变化：ΔC = C_after − C_before', 3.20, 5.51, 4.15, 0.33, { fontFace: 'Aptos', fontSize: 18, bold: true, color: C.white, align: 'center' });
  addText(slide, '模型变化：ΔM = M_after − M_before', 7.55, 5.51, 4.25, 0.33, { fontFace: 'Aptos', fontSize: 18, bold: true, color: C.white, align: 'center' });
  addText(slide, '当前仅完成 dog 覆盖对比；D0 / D1 模型训练待完成，且不能替代固定模型下的测试故障检出验证。', 1.06, 6.44, 11.14, 0.30, { fontSize: 14, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('先明确这是第一阶段的训练数据效用验证，不是对当前数据流程的重复。D0固定训练、验证和测试划分并得到M_before；随后只分析dog训练子集，根据低频状态得到Ddog+，测试集不参与选图。Ddog+与原dog子集合并后，当前已测覆盖由78.64%到84.17%；把Ddog+并入D0_train形成D1、重训M_after并在固定D_test评价仍待完成。该结果即使完成，也不能替代固定被测模型下的测试集新增故障检出验证。');
}

// 14. Analysis: interpreting delta C and delta M
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '06 · ANALYSIS', '真正要分析的是：覆盖变化能否解释模型变化？', 'ΔC 是过程结果，ΔM 才能回答补图是否改善了任务表现；两者必须联合解释');
  addBox(slide, 0.76, 1.90, 4.00, 4.45, C.white, C.line);
  addText(slide, '当前已知', 1.06, 2.18, 1.2, 0.30, { fontSize: 15, bold: true, color: C.teal });
  addMetric(slide, '+5.53 pp', '覆盖指数变化 ΔC', 1.04, 2.72, 3.42, C.orange, '78.64% → 84.17%');
  addMetric(slide, '待测', '模型性能变化 ΔM', 1.04, 4.08, 3.42, C.red, '同一固定测试集');
  addText(slide, '因此当前只能说：\n缺口驱动补图改善了定义下的状态覆盖。', 1.04, 5.52, 3.35, 0.55, { fontSize: 15, bold: true, color: C.navy, align: 'center', breakLine: true });

  addBox(slide, 5.08, 1.90, 7.50, 4.45, C.white, C.line);
  addText(slide, 'ΔC / ΔM 联合判读', 5.40, 2.18, 2.35, 0.30, { fontSize: 15, bold: true, color: C.navy });
  const matrix = [
    { y: 2.72, a: 'ΔC ↑', b: 'ΔM ↑', c: '支持“补充状态”可能带来任务收益', fill: C.greenSoft, color: C.green },
    { y: 3.47, a: 'ΔC ↑', b: 'ΔM ≈ 0', c: '覆盖指标变好，但新增状态未转化为模型收益', fill: C.orangeSoft, color: C.orange },
    { y: 4.22, a: 'ΔC ≈ 0', b: 'ΔM ↑', c: '性能变化可能来自训练随机性或其他数据因素', fill: C.blueSoft, color: C.blue },
    { y: 4.97, a: 'ΔC ↓', b: 'ΔM ↓', c: '检查清洁、标签、重复与数据分布变化', fill: C.redSoft, color: C.red },
  ];
  matrix.forEach((m) => {
    addBox(slide, 5.40, m.y, 1.05, 0.52, m.fill, m.color);
    addText(slide, m.a, 5.48, m.y + 0.13, 0.89, 0.22, { fontFace: 'Aptos', fontSize: 14, bold: true, color: m.color, align: 'center' });
    addBox(slide, 6.55, m.y, 1.05, 0.52, m.fill, m.color);
    addText(slide, m.b, 6.63, m.y + 0.13, 0.89, 0.22, { fontFace: 'Aptos', fontSize: 14, bold: true, color: m.color, align: 'center' });
    addText(slide, m.c, 7.88, m.y + 0.11, 4.22, 0.28, { fontSize: 13.5, bold: true, color: C.ink });
  });
  addBox(slide, 5.40, 5.83, 6.55, 0.32, C.navy, C.navy);
  addText(slide, '需要控制：模型、训练轮数、随机种子、总预算、固定测试集、去重规则', 5.58, 5.87, 6.18, 0.22, { fontSize: 11.5, bold: true, color: C.white, align: 'center' });
  addText(slide, '分析结论的强弱取决于：ΔM 是否显著、是否跨模型复现、是否优于等预算基线。', 1.02, 6.58, 11.30, 0.28, { fontSize: 14.5, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('这页补上真正的分析。第一，当前已有的数据只有ΔC，所以只能说明定义下的状态覆盖增加。第二，只有把补图加入训练集，并在完全固定的测试集上比较M_before和M_after，才能知道模型性能是否变化。第三，不同ΔC/ΔM组合有不同解释：最理想是两者都上升；如果覆盖上升但性能不变，说明覆盖指标可能没有预测任务收益；如果性能变化但覆盖不变，要检查训练随机性和其他数据因素。最后要做等预算基线、多个随机种子和跨模型复现。');
}

// 15. Next steps
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, '07 · NEXT STEPS', '下一步：让覆盖诊断经得起独立检验', '从可运行流程，推进到可比较、可迁移的研究证据');
  const steps = [
    { x: 0.78, n: '1', title: '标签有效性', body: '双人独立标注\n一致性与混淆分析', fill: C.blueSoft, color: C.blue },
    { x: 3.78, n: '2', title: '公平基线', body: '随机 / 分层随机\nCore-set / 简单贪心', fill: C.tealSoft, color: C.teal },
    { x: 6.78, n: '3', title: '独立效用', body: '受控删减与恢复\n新故障 / 成本 / 区间', fill: C.orangeSoft, color: C.orange },
    { x: 9.78, n: '4', title: '跨类迁移', body: 'dog → person / 新类别\n开发与验证分离', fill: C.greenSoft, color: C.green },
  ];
  steps.forEach((s, i) => {
    addBox(slide, s.x, 2.0, 2.55, 2.42, C.white, C.line);
    addNumber(slide, s.n, s.x + 0.22, 2.25, s.color);
    addText(slide, s.title, s.x + 0.75, 2.25, 1.48, 0.35, { fontSize: 18, bold: true, color: s.color });
    addBox(slide, s.x + 0.22, 2.91, 2.1, 0.99, s.fill, s.fill);
    addText(slide, s.body, s.x + 0.33, 3.08, 1.88, 0.63, { fontSize: 14.5, bold: true, color: s.color, align: 'center', breakLine: true });
    if (i < 3) addArrow(slide, s.x + 2.58, 3.21, s.x + 2.96, 3.21, 'A6B4BE', 1.5);
  });
  addBox(slide, 0.9, 4.91, 11.53, 1.17, C.navy, C.navy);
  addText(slide, 'Take-home message', 1.22, 5.14, 2.15, 0.28, { fontFace: 'Aptos', fontSize: 12, bold: true, color: '8ED1CC' });
  addText(slide, '覆盖不是模型安全的替代品；它的价值在于指出“哪些测试证据还没有”。', 3.23, 5.1, 8.82, 0.38, { fontSize: 20, bold: true, color: C.white, align: 'center' });
  addText(slide, '当前已完成可审计的犬类试验链路，下一阶段的关键是独立验证，而不是继续堆叠特征数量。', 1.38, 6.42, 10.58, 0.39, { fontSize: 16.5, bold: true, color: C.red, align: 'center' });
  addFooter(slide);
  slide.addNotes('结束时收束成一句话：我的目标不是用一个覆盖分数替代准确率，而是让我们知道测试数据还缺少哪些条件证据，并在有限预算下采取可验证的行动。当前工程链路已经跑通，后续最关键的工作是标签有效性、公平基线、独立故障收益和跨类别验证。');
}

// 14. Backup: nearest work
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, 'BACKUP A', '最近邻工作与本项目的技术边界', '这些工作必须知道，但不需要在主讲部分逐篇展开');
  const rows = [
    ['Visual Genome', '对象、属性、关系的结构化标注', '不是测试充分性指标'],
    ['S3C', '空间语义场景覆盖', '面向自动驾驶关系场景'],
    ['RBT4DNN', '自然语言需求驱动测试', '需求到生成与后置条件检查'],
    ['HiBug2', '任务属性与可解释错误切片', '依赖模型错误信号，目标是调试'],
    ['TALISMAN', '稀有类别/切片的目标化采样', '主动学习与检测数据获取'],
    ['POPE', 'VLM对象幻觉评价', '说明自动视觉测量需要核验'],
  ];
  const xs = [0.78, 3.45, 8.03];
  const ws = [2.45, 4.36, 4.5];
  ['工作', '它已经解决什么', '与当前研究的边界'].forEach((h, i) => {
    addBox(slide, xs[i], 1.88, ws[i], 0.52, C.navy, C.navy, 0);
    addText(slide, h, xs[i] + 0.12, 2.01, ws[i] - 0.24, 0.27, { fontSize: 14, bold: true, color: C.white, align: i === 0 ? 'left' : 'center' });
  });
  rows.forEach((r, idx) => {
    const y = 2.42 + idx * 0.63;
    const fill = idx % 2 === 0 ? C.white : 'F0F4F7';
    r.forEach((v, i) => {
      addBox(slide, xs[i], y, ws[i], 0.61, fill, C.line, 0);
      addText(slide, v, xs[i] + 0.14, y + 0.09, ws[i] - 0.28, 0.4, { fontFace: i === 0 ? 'Aptos' : 'Microsoft YaHei', fontSize: i === 0 ? 12.5 : 12, bold: i === 0, color: i === 0 ? C.navy : C.ink });
    });
  });
  addBox(slide, 1.25, 6.35, 10.85, 0.42, C.redSoft, C.red);
  addText(slide, '定位原则：不声称单个组件“首次提出”，把贡献放在受噪标签下的证据治理与可验证决策闭环。', 1.48, 6.44, 10.39, 0.26, { fontSize: 14, bold: true, color: C.red, align: 'center' });
  addFooter(slide, 'BACKUP');
  slide.addNotes('如果老师问最近工作，先承认这些方向已经存在，再说明问题设定的差别。尤其HiBug2已经使用任务相关属性做错误切片，因此不能把自动属性本身当作创新。项目更可能的贡献需要落在标签不确定性、任务覆盖要求、复核和补样预算，以及独立验证上。');
}

// 15. Backup: Q&A
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, 'BACKUP B', '高概率追问：建议用这些边界回答', '先承认当前证据，再说明下一步怎样使结论成立');
  const qs = [
    ['为什么阈值是30？', '当前是预先冻结的阶段性证据目标，用于验证流程；不是理论上的普适充分性阈值。'],
    ['覆盖上升是否具有必然性？', '是。当前证明优化目标被实现；策略优越性需要等预算随机、分层和表示空间基线。'],
    ['VLM标签是否可靠？', '100%是格式成功率，不是语义准确率；需要双人标注、一致性、混淆和标签扰动分析。'],
    ['为什么先做dog？', 'COCO提供目标框，类内又有视角、姿态、遮挡与场景变化，适合受控概念验证。'],
    ['创新点到底是什么？', '不是“VLM+覆盖率”，而是候选的：受噪测量、任务约束、预算行动与独立效用闭环。'],
  ];
  qs.forEach((q, i) => {
    const y = 1.86 + i * 0.95;
    addBox(slide, 0.78, y, 2.8, 0.74, i % 2 === 0 ? C.blueSoft : C.tealSoft, i % 2 === 0 ? C.blue : C.teal);
    addText(slide, q[0], 0.98, y + 0.16, 2.4, 0.4, { fontSize: 14.5, bold: true, color: i % 2 === 0 ? C.blue : C.teal, align: 'center' });
    addBox(slide, 3.78, y, 8.8, 0.74, C.white, C.line);
    addText(slide, q[1], 4.04, y + 0.12, 8.28, 0.48, { fontSize: 13.5, color: C.ink });
  });
  addBox(slide, 1.45, 6.68, 10.42, 0.3, C.orangeSoft, C.orange);
  addText(slide, '回答顺序：当前证据 → 不能支持什么 → 需要什么实验才能支持。', 1.62, 6.72, 10.08, 0.2, { fontSize: 12.5, bold: true, color: C.orange, align: 'center' });
  addFooter(slide, 'BACKUP');
  slide.addNotes('这些回答的核心是不要防御性回避局限。先明确当前结果支持的范围，再指出计划中的验证设计。老师如果质疑覆盖增加是自我实现目标，可以直接承认，这是为什么下一步必须做等预算基线和独立故障收益。');
}

// 16. Backup: evidence details
{
  const slide = pptx.addSlide('QE_MASTER');
  addTitle(slide, 'BACKUP C', '结果口径与证据位置', '避免把清洁前600张、清洁后596张和900张标注运行混为同一个统计口径');
  addBox(slide, 0.76, 1.86, 7.0, 4.78, C.white, C.line);
  slide.addImage({ path: IMG_GAPS, x: 1.02, y: 2.13, w: 6.48, h: 3.78 });
  addBox(slide, 8.05, 1.86, 4.56, 1.07, C.blueSoft, C.blue);
  addText(slide, '900张运行', 8.31, 2.08, 1.5, 0.32, { fontSize: 17, bold: true, color: C.blue });
  addText(slide, '500初始 + 400候选；全部严格格式通过', 9.6, 2.02, 2.72, 0.49, { fontSize: 12.5, color: C.ink });
  addBox(slide, 8.05, 3.18, 4.56, 1.07, C.orangeSoft, C.orange);
  addText(slide, '600张合并', 8.31, 3.40, 1.5, 0.32, { fontSize: 17, bold: true, color: C.orange });
  addText(slide, '500初始 + 100补图；选择阶段剩31个缺口', 9.6, 3.34, 2.72, 0.49, { fontSize: 12.5, color: C.ink });
  addBox(slide, 8.05, 4.50, 4.56, 1.07, C.greenSoft, C.green);
  addText(slide, '596张清洁', 8.31, 4.72, 1.5, 0.32, { fontSize: 17, bold: true, color: C.green });
  addText(slide, '排除4张非真实犬；最终剩33个低频状态', 9.6, 4.66, 2.72, 0.49, { fontSize: 12.5, color: C.ink });
  addBox(slide, 8.05, 5.82, 4.56, 0.82, C.redSoft, C.red);
  addText(slide, '汇报主口径：最终清洁596张，45 → 33。', 8.3, 6.03, 4.06, 0.37, { fontSize: 15.5, bold: true, color: C.red, align: 'center' });
  addFooter(slide, 'BACKUP');
  slide.addNotes('如果被问到为什么有31和33两个数字：600张选择结果中解决14个缺口，剩31个；清洁时从原始500张中排除了4张非真实犬表现形式，重新统计后最终596张有33个低频状态。主讲统一使用最终清洁口径：45降到33。');
}

async function main() {
  await pptx.writeFile({ fileName: OUT, compression: true });
  console.log(`Wrote ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

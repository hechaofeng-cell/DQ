# QE

Project workspace for QE.

## DTA 第一阶段

`qe_quality.dta` 在 Imagenette 100 张上运行本地 Qwen3-VL、SAM vit_h 和在线
VLM 对照。每张结果独立落盘，使用相同协议与 `--resume` 可断点续跑。
当前配置使用 [主体选择协议 v8](docs/dta_subject_selection_v8.md)：前景中被展示或被操作的对象优先于人物，旧 v7 结果不复用。

先运行三张真实 smoke：

```bash
.venv/bin/python -u -m qe_quality.dta phase1 \
  --config configs/dta_phase1.json \
  --output artifacts/dta_phase1_imagenette_v8_foreground_20260910 \
  --resume --smoke-only
```

检查 `mask_overlays/` 和 `smoke_report.json` 后，在同一输出目录去掉
`--smoke-only` 运行全量。后台推荐使用 systemd 用户服务：

```bash
mkdir -p /home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910
systemd-run --user --unit=dta-phase1-v8 --collect \
  --property=WorkingDirectory=/home/hcf/project/QE \
  --property=StandardOutput=append:/home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910/background.log \
  --property=StandardError=append:/home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910/background.log \
  /home/hcf/project/QE/.venv/bin/python -u -m qe_quality.dta phase1 \
  --config /home/hcf/project/QE/configs/dta_phase1.json \
  --output /home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910 --resume
```

实时日志与结构化状态：

```bash
tail -F /home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910/background.log
journalctl --user -u dta-phase1-v8 -f
watch -n 5 '/home/hcf/project/QE/.venv/bin/python -m json.tool /home/hcf/project/QE/artifacts/dta_phase1_imagenette_v8_foreground_20260910/pipeline_status.json'
```

自动阶段结束后填写 `human_review.csv`，再用同一全量命令 `--resume` 重算验收；
已完成的本地、SAM 和在线结果不会重复请求。

## 原图难度评分

新增 CPU 评分与验证模块复用已有 GPU 输出，只处理基准原图。工程完成与评分有效性分别验收；不划分易中难、不调整分布、不运行蜕变测试。

协议和证据边界见 [评分协议](docs/difficulty_protocol.md)。仅需 Python 3.12 与 Pillow；现有 `.venv` 可直接使用。独立安装依赖可运行 `python -m pip install -r requirements-difficulty.txt`。

```bash
.venv/bin/python -m qe_quality.difficulty prepare --config configs/difficulty_baseline_v1.json --output artifacts/my_run/prepared
.venv/bin/python -m qe_quality.difficulty review-sheets --prepared artifacts/my_run/prepared --output artifacts/my_run/review_sheets
.venv/bin/python -m qe_quality.difficulty evaluate --prepared artifacts/my_run/prepared --output artifacts/my_run/diagnostic
.venv/bin/python -m unittest discover -s tests -v
```

每个输出目录必须不存在。输入路径相对于配置文件解析；配置中 archive 为 null 时沿用历史 GPU 协议的归档路径。

`prepared` 中包含原图表、完整预测、配对关系、审核队列、审核/划分模板、问题清单、模型配置与文件哈希。审核记录通过 `--reviews /path/to/reviews.csv` 接入，明确数据用途后再传 `--splits /path/to/splits.csv`。
未审核标签只有单独的继承标签诊断列，存疑样本不评分；校准条件不足时 B 留空。结果目录输出逐图 `scores.csv`、`report.md` / `report.json`、失败样例、校准参数、自动检查和运行记录。

规则冻结及验证使用：

```bash
.venv/bin/python -m qe_quality.difficulty evaluate --prepared artifacts/my_run/prepared --reviews reviews.csv --splits splits.csv --output artifacts/my_run/development
.venv/bin/python -m qe_quality.difficulty freeze --development artifacts/my_run/development --output artifacts/my_run/frozen
.venv/bin/python -m qe_quality.difficulty evaluate --prepared artifacts/my_run/prepared --frozen artifacts/my_run/frozen/frozen.json --output artifacts/my_run/validation
.venv/bin/python -m qe_quality.difficulty verify --run artifacts/my_run/validation
```

未分配开发/验证数据或缺少审核有效样本时，冻结会拒绝执行。历史结果已经被查看过，事后冻结不能使其成为盲测数据。

本轮实际产物与结论见 [运行记录](docs/difficulty_research_log.md) 和 [证据表](docs/difficulty_claim_evidence.md)。历史 `data/` 文件保留不覆盖。

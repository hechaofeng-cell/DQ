# ImageNetV2 Matched Frequency full evaluation

Requested on 2026-09-07: run the existing seven-model panel on the selected
10,000-image Matched Frequency dataset and report overall recognition accuracy.
The user selected Matched Frequency explicitly. No difficulty grades, dataset
distribution changes, model tuning, calibration or metamorphic tests are in scope.

## Fixed evaluation

- Input: `data/imagenetv2/imagenetv2-matched-frequency.tar.gz`. Despite its suffix,
  the existing file is already an uncompressed POSIX TAR. Images are read directly
  from it, without another download or extraction.
- Archive inventory: exactly 10,000 image files, class indices 0 through 999,
  exactly 10 images per class. Numeric directories provide inherited labels.
- Model panel: ResNet-50, Swin-T, CLIP ViT-B/32, ConvNeXt-Tiny,
  EfficientNet-B0, MobileNet-V3-Large, RegNet-Y-8GF; same cached weights as the
  preceding seven-model run. No selection based on this run's outcomes.
- Task: full 1,000-class classification. Prior ten-class restricted accuracy
  cannot be directly compared as a measure of dataset difficulty.
- CLIP: one fixed `a photo of a {category}.` prompt per class, using the
  torchvision class vocabulary, including separate names for crane bird and
  crane machinery. No prompt tuning or prompt ensemble.
- Primary metric: exact top-1 accuracy per model. Secondary: top-5, per-class
  accuracy, macro class accuracy and counts of models correct on each image.
  A numerical average across model accuracies is not ensemble accuracy.
- Missing/undecodable images are reported explicitly, with both total-image
  and valid-image top-1 denominators. Nonfinite outputs or class mapping errors
  stop the run. Source hashes are checked before and after evaluation.
- GPU: NVIDIA GeForce RTX 5090 D v2, CUDA float32, TF32 disabled. Sandbox CUDA
  discovery failed; approved host execution confirmed the GPU was available.
- Raw evidence: image IDs and SHA-256, full float32 logits, per-image predictions,
  per-class counts, checkpoint hashes, preprocessing, code hashes and runtime
  metadata. Top-1 and top-5 are independently recomputed from saved logits.
- Stop criteria: full aligned outputs for all requested models, or a recorded
  input/runtime failure. Accuracy being lower than previous results is not a
  reason to tune or rerun with different settings.

## Commands and checkpoints

Smoke test: all seven models on 32 explicitly selected rows, full 1,000-class
candidate space. Passed without decode failures; all saved-logit checks passed.
This subset is only an implementation check, not the reported dataset result.

```bash
.venv/bin/python data/evaluate_imagenetv2_full.py --archive data/imagenetv2/imagenetv2-matched-frequency.tar.gz --output artifacts/recognition/imagenetv2_matched_10000_smoke_20260907 --limit 32 --batch-size 32 --workers 4
.venv/bin/python data/evaluate_imagenetv2_full.py --archive data/imagenetv2/imagenetv2-matched-frequency.tar.gz --output artifacts/recognition/imagenetv2_matched_10000_20260907 --batch-size 64 --workers 4
```

All output directories are new and the runner refuses overwrite. Final numerical
claims must be taken from the complete run's `summary.json`, not the smoke test.
Model training membership remains unresolved; this run does not establish a
universal image difficulty scale or a formal easy/medium/hard partition.

## Related sample-count question

The historical original-image manifest contains 13,394 original paths in ten
classes: nine classes have 1,350 each and chainsaw has 1,244. These are historical
Imagenette/ImageNet-derived images with source-overlap limitations, not newly
independent samples. Matched Frequency contains only 10 per class, so restricting
it to the original ten classes gives 100 images, as recorded in the prior pilot.

## Completed results and evidence

The full run completed all seven models and all 10,000 images with zero decode
failures. Results are recorded in
`artifacts/recognition/imagenetv2_matched_10000_20260907/summary.json` and the
same directory's `report.md`.

| Model | Top-1 correct / 10,000 | Top-1 accuracy | Top-5 accuracy |
| --- | ---: | ---: | ---: |
| ResNet-50 | 6,989 | 69.89% | 88.86% |
| Swin-T | 6,939 | 69.39% | 89.10% |
| CLIP ViT-B/32 | 5,274 | 52.74% | 78.95% |
| ConvNeXt-Tiny | 7,144 | 71.44% | 89.89% |
| EfficientNet-B0 | 6,605 | 66.05% | 86.08% |
| MobileNet-V3-Large | 6,210 | 62.10% | 84.23% |
| RegNet-Y-8GF | 7,240 | 72.40% | 90.48% |

The arithmetic mean of seven top-1 accuracies is 66.28714285714287%; this is
not the accuracy of an ensemble. All seven models are correct on 3,859 images;
all seven are wrong on 1,562 images. These are raw panel-support counts, not
easy/hard labels.

An independent readback verified all artifact hashes, the 10,000 unique row IDs,
the 1,000 classes with 10 images each, all seven finite (10,000, 1,000) logit
arrays, all 70,000 prediction alignments, top-1 and top-5 totals reconstructed
from those arrays, and the per-class totals. The input archive hash matched
before and after inference. No inference rerun was needed after the smoke test.

Evidence scope: the table is supported by the saved arrays, prediction CSVs and
per-model run metadata. It describes the named weights under this full-class
protocol. A claim that these numbers directly measure a difficulty change from
the previous ten-class task is unsupported, because the candidate class space
also changed. CLIP uses a single fixed prompt per class, not an optimized prompt
ensemble. There was no scientific acceptance of a difficulty scoring method.

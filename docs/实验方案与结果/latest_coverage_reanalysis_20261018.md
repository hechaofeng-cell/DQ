# Latest Coverage and Diagnostic Reanalysis

This package recomputes the frozen 2026-10-18 coverage, selection, external
diagnostic and automatic robustness analyses. It includes target-level automatic
labels, risk inputs, state matrices, cached ViT embeddings, and classifier
logits for five architectures and three training seeds. It does not run training,
classifier inference, VLM annotation, or network requests.

## Run

Download `docs/archive/coverage_diagnostic_reanalysis_20261018.zip` from the GitHub
repository and extract it into an empty directory. In the extracted directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --output reproduced
```

Python 3.12 is the tested interpreter. The only analysis dependency is NumPy;
its tested version is pinned in `requirements.txt`. Installing that dependency
may require network access, but replay itself uses only bundled files. CPU is
sufficient. Use a new output directory for every run. The commands above use
Linux/macOS virtual-environment paths; on Windows use `.venv\Scripts\python.exe`.

The replay verifies package hashes, copies inputs and historical code into a new
workspace, and runs seven stages: implementation checks, selection, condition
discovery, annotation repeatability, synthetic missingness, external confirmation,
and automatic robustness. It regenerates 17 CSV tables and five NPZ files, then
compares them with the historical results. CSV numeric tolerance is 1e-10
(absolute and relative); string values, row order and array contents must match.

Success ends with `PASS` and writes `reproduced/verification.json`. Individual
stage logs are in `reproduced/logs/`. Regenerated tables are under
`reproduced/workspace/artifacts/coverage_diagnostic_validation_20261018/`.
Original bundle files remain unchanged. A failing comparison stops the run.

## What Is Recomputed

- Four independent budgets and five matched selection methods, including 1000
  random draws, support-cap and weighting sensitivity, and exact risk quotas.
- Weak-condition discovery from frozen discovery-pool predictions and external
  confirmation from frozen predictions for 800 targets, including target bootstrap.
- Repeat agreement from the existing 80 paired automatic annotations and
  synthetic missingness sensitivity from 240 reruns. No new labels are generated.
- Severity thresholds, seed consistency, risk-overlap eligibility, adjusted
  contrasts where eligible, and deduplicated condition counts.

The four target-category analyses evaluate predictions from the same six-class
classification task. This package covers these 2026-10-18 analyses, not every
historical project experiment or the unimplemented 2026-10-27 replacement design.
Paper formatting and figure rendering are not part of the verified replay.

## Files and Boundaries

`inputs/` holds only the selected frozen dependencies. `snapshot/` contains the
historical scripts and risk configuration. `expected/` contains comparison
outputs, which are never used as analysis inputs. `provenance/` records source
hashes, the source commit, historical freezes, and environment information.
`reports/` retains the original result reports and dataset-license inventory.

Historical manifests contain original absolute image paths. They are retained
as provenance metadata; replay does not open those paths or need those images.
Historical freezes are retained separately: replay generates new path-specific
freezes and does not claim to recreate historical execution times. Reports may
contain links to local figures and older publication-status text.

Raw images, trained weights, optimizer states, API credentials, raw service
responses, and unrelated downloaded dataset archives are excluded. Cached state
labels and model outputs ARE included. Matching calculations does not validate
the semantic accuracy of automatic labels, prove a causal effect, or establish
that coverage selection consistently beats strong diagnostic baselines.

To rebuild this bundle from the complete local project, run
`python scripts/package_coverage_diagnostic_reanalysis.py --output /path/to/new.zip`.
That packaging command needs local frozen artifacts; users of the uploaded ZIP
need only `reproduce.py` and the bundled files.

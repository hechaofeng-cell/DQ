# Portable Reanalysis Validation

Validated on 2026-11-02 for the frozen 2026-10-18 coverage and diagnostic study.

- Archive: `coverage_diagnostic_reanalysis_20261018.zip`
- SHA-256: `f17cdb458fa116646e417a7ba35ca9e74f4dd2caabee1b508191bc4bae4887f9`
- Archive size: 15,715,377 bytes (14.99 MiB).
- Payload size: 34,150,207 bytes (32.57 MiB), excluding the package manifest.
- Verified payload: 1854 files; unchanged after replay.
- Environment: Python 3.12.3, NumPy 2.5.2, CPU, two BLAS/OpenMP threads.
- All seven replay stages passed in approximately 7 seconds on the validation host.
- All 17 CSV tables matched with numeric tolerance 1e-10.
- All five NPZ files matched array-for-array, including 92 selection arrays.

The actual ZIP was extracted into a new directory outside the project. A separate
virtual environment was created without system site packages, and only the tested
NumPy installation and its binary libraries were copied into that environment.
The extracted `reproduce.py` ran in a fresh output workspace. Dependency download
from a package registry was not tested. Replay used no original project artifacts,
images, model weights, GPU, VLM service, or API credentials. It does not invoke
training, inference, or annotation. The code reads only the frozen inputs bundled
with the archive; the original project was left intact.

The full generated check record is available in
[coverage_diagnostic_reanalysis_verification_20261102.json](coverage_diagnostic_reanalysis_verification_20261102.json).
Every replay creates its own equivalent `verification.json`, including table row
counts, the actual environment, elapsed time and scope.

This validates numerical reanalysis of frozen automatic labels and predictions.
It does not validate the accuracy of those labels, reproduce historical online
model responses, rerun training, or implement the later replacement proposal.

Instructions: [latest_coverage_reanalysis_20261018.md](latest_coverage_reanalysis_20261018.md).

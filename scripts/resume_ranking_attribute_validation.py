#!/usr/bin/env python3
"""Apply the existing Objects365 pixel rounding tolerance in scale-only auditing."""
import json
from pathlib import Path
import numpy as np
import analyze_ranking_attribute_validation as analysis
from dual_vlm_common import read_csv, write_json, sha256, now


def scale_check():
    from run_multiclass_object_selection_benchmark import scale_state
    details, summary = [], []
    for pool, path in {'A':analysis.A, **analysis.POOLS}.items():
        rows = read_csv(path / 'manifests/candidate_pool.csv')
        pool_records = []
        for row in rows:
            x1, y1, x2, y2 = json.loads(row['bbox_norm_xyxy'])
            assert x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0
            excess = 0.
            if pool == 'D':
                excess = max(0., (x2 - 1) * float(row['image_width']), (y2 - 1) * float(row['image_height']))
                assert excess <= .001 + 1e-9, 'Outside original Objects365 qualification tolerance'
            else:
                assert x2 <= 1 and y2 <= 1
            area = (x2 - x1) * (y2 - y1)
            clipped_area = (min(x2, 1) - x1) * (min(y2, 1) - y1)
            saved = float(row['bbox_area_ratio'])
            record = dict(pool=pool, sample_id=row['sample_id'], recomputed_area=area,
                manifest_area=saved, clipped_area=clipped_area, abs_difference=abs(area - saved),
                scale_recomputed=scale_state(area), scale_manifest=scale_state(saved),
                scale_clipped=scale_state(clipped_area),
                scale_changed=scale_state(area) != scale_state(saved),
                clipping_scale_changed=scale_state(clipped_area) != scale_state(saved),
                original_coordinates_over_boundary=excess > 0, boundary_excess_pixels=excess)
            details.append(record); pool_records.append(record)
        summary.append(dict(pool=pool, targets=len(rows),
            max_abs_difference=max(r['abs_difference'] for r in pool_records),
            scale_changes=sum(r['scale_changed'] for r in pool_records),
            clipping_scale_changes=sum(r['clipping_scale_changed'] for r in pool_records),
            original_coordinates_over_boundary=sum(r['original_coordinates_over_boundary'] for r in pool_records),
            max_boundary_excess_pixels=max(r['boundary_excess_pixels'] for r in pool_records),
            area_difference_over_1e6=sum(r['abs_difference'] > 1e-6 for r in pool_records)))
    assert all(r['scale_changes'] == 0 and r['clipping_scale_changes'] == 0
               and r['area_difference_over_1e6'] == 0 for r in summary)
    return details, summary


def main():
    paths = [Path(__file__).resolve(), analysis.ROOT / 'docs/ranking_attribute_validation_technical_amendment_20261005.md']
    mapping = {str(p):sha256(p) for p in paths}
    destination = analysis.OUT / 'technical_amendment.sha256.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == mapping
    else:
        write_json(destination, mapping)
    analysis.scale_check = scale_check
    analysis.note('resume_scale_audit', qualification_rule='Objects365 original <=0.001 pixel rounding tolerance')
    analysis.main()
    assert all(sha256(p) == digest for p, digest in mapping.items())


if __name__ == '__main__':
    main()

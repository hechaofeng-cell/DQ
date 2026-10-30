#!/usr/bin/env python3
"""Inventory existing fine labels without inventing or combining score protocols."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/QE_VisualState_CoverRisk_v1_20261013'
FRESH = ROOT / 'artifacts/fresh_vlm_feature_scoring_20261012'
EARLY = ROOT / 'artifacts/dog_v3_1_combined_supplement_20260913_final/clean_results_596.csv'
OUT = ROOT / 'artifacts/QE_FeatureRisk_Diagnostics_20261013'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(data)
    return hasher.hexdigest()


def serial(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def known(value):
    return value not in ('unknown', 'not_observable', '') if not isinstance(value, list) else bool(value) and not any(v in ('unknown', 'not_observable', '') for v in value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(DATA / 'manifest.csv'); lookup = {r['sample_id']: r for r in manifest}
    assert len(lookup) == len(manifest) == 7800
    assert Counter(r['class_name'] for r in manifest) == Counter(dict.fromkeys(CLASSES, 1300))
    schemas = {c: json.loads((FRESH / 'schemas' / f'{c}.json').read_text()) for c in CLASSES}
    fields = {c: {f['feature_id']: f for f in schemas[c]['universal_features'] + schemas[c]['category_specific_features']} for c in CLASSES}
    assert [len(fields[c]) for c in CLASSES] == [35, 35, 27, 27, 32, 26]
    role_rows = read_csv(FRESH / 'feature_dimension_assignment.csv')
    role = {(r['category'], r['field']): r for r in role_rows}
    catalog_rows = read_csv(ROOT / 'artifacts/category_feature_dimension_review_20261013/feature_dimension_catalog.csv')
    catalog = {(r['category'], r['field']): r for r in catalog_rows}
    fresh_scores = read_csv(FRESH / 'scores.csv'); score_ids = {r['sample_id'] for r in fresh_scores}
    assert len(score_ids) == len(fresh_scores) == 60
    assert score_ids <= set(lookup)
    counts = defaultdict(Counter)
    values = defaultdict(Counter)
    native_field_sets = defaultdict(set)
    fine_annotations = []
    freeze_paths = [Path(__file__).resolve(), DATA / 'manifest.csv', EARLY, FRESH / 'scores.csv', FRESH / 'scoring_protocol.json', FRESH / 'feature_dimension_assignment.csv', ROOT / 'artifacts/category_feature_dimension_review_20261013/feature_dimension_catalog.csv'] + [FRESH / 'schemas' / f'{c}.json' for c in CLASSES]
    fresh_seen = set(); native_records = 0
    for row in manifest:
        sid = row['sample_id']; c = row['class_name']; path = DATA / row['annotation_path']
        annotation = json.loads(path.read_text()); freeze_paths.append(path)
        assert annotation['sample_id'] == sid and annotation['class_name'] == c and int(annotation['official_label']) == int(row['label'])
        for source in ('local', 'online'):
            native = annotation[source]['native']
            feature_map = native['features']; evidence = native['evidence']
            assert len(feature_map) == (18 if c == 'car' else 19) and set(feature_map) == set(evidence)
            native_field_sets[c, source].add(tuple(sorted(feature_map)))
            for field, value in feature_map.items():
                key = (c, source + '_native', field)
                counts[key]['annotated'] += 1
                counts[key]['known'] += int(known(value))
                counts[key]['not_in_view'] += int(value == 'not_in_view')
                counts[key]['evidence'] += int(bool(evidence[field]))
                values[key][serial(value)] += 1
                native_records += 1
        if 'fine_feature_supplement' in annotation:
            fine = annotation['fine_feature_supplement']
            original = json.loads((FRESH / 'annotations' / f'{sid}.json').read_text())
            assert fine == original
            assert fine['features'].keys() == fields[c].keys()
            assert sid in score_ids
            fresh_seen.add(sid)
            freeze_paths.append(FRESH / 'annotations' / f'{sid}.json')
            for field, value in fine['features'].items():
                key = (c, 'fresh_local_fine', field)
                counts[key]['annotated'] += 1
                counts[key]['known'] += int(known(value))
                counts[key]['not_in_view'] += int(value == 'not_in_view')
                counts[key]['evidence'] += int(bool(fine['evidence'][field]))
                values[key][serial(value)] += 1
                r = role[c, field]
                fine_annotations.append(dict(sample_id=sid, class_name=c, annotation_source='fresh_local_fine',
                    schema_version=fine['schema_version'], field=field, name_zh=fields[c][field].get('feature_name_zh', catalog.get((c, field), {}).get('特征', field)),
                    dimension_or_context=r['dimension_or_context'], existing_fine_score_role=r['role'],
                    value_json=serial(value), evidence=fine['evidence'][field], annotation_path=str(path),
                    original_annotation_path=str(FRESH / 'annotations' / f'{sid}.json')))
    assert native_records == 293800
    assert fresh_seen == score_ids and len(fine_annotations) == 1820
    assert all(len(v) == 1 for v in native_field_sets.values())
    main_scored = {'body_visibility', 'body_orientation', 'occlusion_level', 'truncation_level'}
    inventory = []
    fresh_protocol = json.loads((FRESH / 'scoring_protocol.json').read_text())
    for (c, source, field), counter in sorted(counts.items()):
        meta = fields[c].get(field, {})
        r = role.get((c, field), {})
        cat = catalog.get((c, field), {})
        possible = meta.get('possible_values', [])
        existing_role = 'global_scored' if source.endswith('_native') and field in main_scored else r.get('role', 'description') if source == 'fresh_local_fine' else 'global_description_only'
        current_fine_scored = source == 'fresh_local_fine' and existing_role == 'scored'
        if source.endswith('_native'):
            schema_path = ROOT / 'artifacts/sixclass_dual_vlm_replaced_20261004/configs' / ('car_annotation_schema.json' if c == 'car' else 'old_annotation_schema.json')
            schema = json.loads(schema_path.read_text())
            feature_list = schema['universal_features'] + schema.get('category_specific_features', [])
            native_meta = next(f for f in feature_list if f['feature_id'] == field)
            possible = native_meta['possible_values']
            name = native_meta.get('feature_name_zh', cat.get('特征', field))
            schema_version = schema['schema_version']
        else:
            name = meta.get('feature_name_zh', cat.get('特征', field))
            schema_version = schemas[c]['schema_version']
        if existing_role in ('global_scored', 'scored'):
            eligibility = 'used_in_existing_score_protocol'
        elif field in ('leg_visibility', 'leg_visibility_count', 'wheel_visibility') or {'clear', 'partial', 'not_visible'} <= set(possible):
            eligibility = 'ordered_visibility_candidate_not_used_in_this_score'
        elif field in ('head_orientation', 'body_visibility'):
            eligibility = 'existing_ordinal_mapping_but_not_used_in_this_score'
        elif 'no_risk_order' in r.get('role', '') or field in ('car_color', 'coat_primary_color', 'coat_pattern', 'coat_texture', 'hair_color', 'hair_length', 'hair_texture', 'hair_presence', 'headwear', 'eyewear'):
            eligibility = 'nominal_description_no_risk_order'
        else:
            eligibility = 'description_no_validated_risk_mapping'
        inventory.append(dict(class_name=c, annotation_source=source, field=field, name_zh=name,
            dimension_or_context=cat.get('维度', r.get('dimension_or_context', '关联条件')),
            subgroup=cat.get('子组', ''), schema_version=schema_version,
            pool_targets=1300, annotated_targets=counter['annotated'], missing_annotations=1300-counter['annotated'],
            known_value_targets=counter['known'], unknown_value_targets=counter['annotated']-counter['known'],
            normal_view_not_in_view_targets=counter['not_in_view'], evidence_targets=counter['evidence'],
            annotation_fraction=counter['annotated']/1300, existing_score_role=existing_role,
            risk_scoring_eligibility=eligibility, possible_values_json=serial(possible), observed_values_json=serial(dict(values[c, source, field]))))
    missing_detail = []
    for c in CLASSES:
        native_fields = set(next(iter(native_field_sets[c, 'local'])))
        for field, meta in fields[c].items():
            if field not in native_fields:
                missing_detail.append(dict(class_name=c, field=field, name_zh=meta.get('feature_name_zh', catalog.get((c, field), {}).get('特征', field)),
                    native_targets_with_field=0, fine_targets_with_field=counts[c, 'fresh_local_fine', field]['annotated'],
                    pool_targets_without_field=1300-counts[c, 'fresh_local_fine', field]['annotated'],
                    policy='not inferred from unrelated fields or targets'))
    fine_summary = []
    for c in CLASSES:
        rows = [r for r in fresh_scores if r['class_name'] == c]
        assert len(rows) == 10
        output = dict(class_name=c, annotation_source='fresh_local_fine', targets=10, fine_fields_per_target=len(fields[c]),
            pool_targets=1300, fresh_scored_fraction=10/1300,
            fully_known_targets=sum(r['fully_known']=='True' for r in rows), evaluable_targets=sum(r['evaluable']=='True' for r in rows),
            score_protocol=fresh_protocol['version'], original_score_path=str(FRESH/'scores.csv'))
        for key in ('score_lower', 'score_midpoint', 'score_upper'):
            output['mean_'+key] = float(np.mean([float(r[key]) for r in rows]))
            output['min_'+key] = float(min(float(r[key]) for r in rows))
            output['max_'+key] = float(max(float(r[key]) for r in rows))
        for dimension in ('scale', 'visibility', 'occlusion', 'truncation', 'orientation'):
            output[dimension+'_mean_contribution_points'] = 20*float(np.mean([float(r[dimension+'_midpoint']) for r in rows]))
        fine_summary.append(output)
    early = read_csv(EARLY)
    assert len(early) == 596 and all(r['target_class']=='dog' for r in early)
    early_ids = {r['image_id'] for r in early}
    candidate_image_matches = [r for r in manifest if r['class_name']=='dog' and r['image_id'] in early_ids]
    assert not candidate_image_matches
    class_summary = [dict(class_name=c, pool_targets=1300, native_fields_per_target=18 if c=='car' else 19,
        local_native_annotated_targets=1300, online_native_annotated_targets=1300,
        fresh_fine_annotated_targets=10, fresh_fields_per_target=len(fields[c]),
        targets_without_fresh_schema=1290,
        new_fine_fields_not_present_in_native_schema=len([r for r in missing_detail if r['class_name']==c])) for c in CLASSES]
    for name, rows in [('fine_feature_inventory.csv',inventory),('fine_feature_annotations.csv',fine_annotations),
                       ('fine_score_summary.csv',fine_summary),('fine_feature_missing_fields.csv',missing_detail),
                       ('annotation_schema_availability_by_class.csv',class_summary),('fine_scores_preserved.csv',fresh_scores)]:
        write_csv(OUT/name, rows)
    audit = dict(status='pass', pool_targets=7800, native_annotations=15600, native_feature_records=293800,
        fresh_fine_targets=60, fresh_fine_records=1820, fresh_fully_known_targets=59, fresh_evaluable_targets=59,
        early_dog23_rows=596, early_has_sample_id_column='sample_id' in early[0], early_pool_same_image_matches=len(candidate_image_matches),
        early_verified_target_matches=0, early_annotations_merged_into_pool=False,
        matching_policy='Early CSV has only COCO original image_id and target bbox; no dog image_id overlap with pool, so no target-level association is possible.',
        existing_fine_scores_preserved=True, fresh_scores_not_mixed_into_global_main_protocol=True,
        main_score_scope='official bbox scale + body_visibility + occlusion_level + truncation_level + body_orientation',
        fine_score_scope='60 targets, category-specific part visibility and body/head orientation, original fixed five-dimension protocol',
        nominal_fields_assigned_risk_order=False, new_vlm_inference=False, semantic_labels_fully_human_verified=False,
        outputs=['fine_feature_inventory.csv','fine_feature_annotations.csv','fine_score_summary.csv','fine_scores_preserved.csv','fine_feature_missing_fields.csv','annotation_schema_availability_by_class.csv'])
    (OUT/'inventory_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    freeze = {str(p):digest(p) for p in freeze_paths}
    for name in audit['outputs']+['inventory_audit.json']:
        path=OUT/name;freeze[str(path)]=digest(path)
    (OUT/'inventory_freeze.sha256.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

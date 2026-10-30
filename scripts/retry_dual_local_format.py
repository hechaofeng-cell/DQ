#!/usr/bin/env python3
"""Enforce existing exclusive set enums in local JSON decoding on resume.

The native validator already rejects an exclusive token mixed with other items.
This only communicates the same rule to the constrained decoder. No label is
coerced, removed, or accepted under a weaker validation rule.
"""
from __future__ import annotations

import json

import annotate_dual_vlm as annotation
from dual_vlm_common import STUDY, now, schema_path, sha256, write_json

original_schema = annotation.local.feature_schema


def exclusive_schema(path):
    output = original_schema(path)
    native = json.loads(path.read_text())
    properties = output['properties']['features']['properties']
    for feature in native['universal_features'] + native.get('category_specific_features', []):
        exclusive = feature.get('exclusive_values', [])
        if feature['value_type'] != 'set' or not exclusive:
            continue
        alternatives = [{'type': 'array', 'items': {'type': 'string', 'enum': [value]},
                         'minItems': 1, 'maxItems': 1} for value in exclusive]
        ordinary = [value for value in feature['possible_values'] if value not in exclusive]
        if ordinary:
            alternatives.append({'type': 'array', 'items': {'type': 'string', 'enum': ordinary},
                                 'minItems': 1, 'uniqueItems': True})
        properties[feature['feature_id']] = {'anyOf': alternatives}
    return output


if __name__ == '__main__':
    annotation.local.feature_schema = exclusive_schema
    write_json(STUDY / 'local_format_recovery.json', {
        'started_at': now(), 'script_sha256': sha256(__file__),
        'native_schema_sha256': sha256(schema_path('car')),
        'change': 'constrained JSON grammar enforces the existing exclusive_values rule',
        'unchanged': ['model', 'seed', 'temperature', 'semantic prompt', 'native enums',
                      'images', 'target', 'strict validator'],
        'resume': 'only absent valid parsed labels are called again'})
    annotation.main()

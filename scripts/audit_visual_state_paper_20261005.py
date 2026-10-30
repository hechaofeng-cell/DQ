#!/usr/bin/env python3
"""Read-only audit of the nine result tables and local links in the full draft."""
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/visual_state_stratified_testing_empirical_paper_20261005.md'
POOLS = {'B': 'sixclass_independent_confirmation_20261004',
         'C': 'sixclass_voc_external_confirmation_20261005'}
SOURCES = {'本地': 'local', '线上': 'online'}
METRICS = {'混淆类型': 'confusion', '错误簇': 'clusters'}
MODELS = dict(zip(('ResNet-18', 'ResNet-50', 'MobileNetV3-Small', 'MobileNetV3-Large',
                   'DenseNet-121', 'EfficientNet-B0', 'ConvNeXt-Tiny', 'Swin-T'),
                  ('resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large',
                   'densenet121', 'efficientnet_b0', 'convnext_tiny', 'swin_t')))
checked = 0


def records(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load(pool, relative):
    return records(ROOT / 'artifacts' / POOLS[pool] / relative)


def check(got, expected, digits=2):
    global checked
    value = float(str(got).replace('−', '-').replace(',', ''))
    assert abs(value - round(float(expected), digits)) < 1e-9, (got, expected)
    checked += 1


def run():
    text = PAPER.read_text(encoding='utf-8')
    tables = {}
    for number in range(1, 10):
        part = text.split(f'**表{number} ')[1].split('\n\n', 1)[1]
        lines = []
        for line in part.splitlines():
            if not line.startswith('|'):
                break
            lines.append([cell.strip() for cell in line.strip('|').split('|')])
        tables[number] = lines[2:]
    assert [len(tables[n]) for n in range(1, 10)] == [5, 4, 10, 8, 8, 8, 4, 6, 4]
    for row in tables[1]:
        column = 1
        for pool in POOLS:
            values = load(pool, 'confirmation/tables/full_stratum_metrics.csv')
            for source in SOURCES.values():
                selected = [r for r in values if r['source'] == source
                            and r['unknown_mode'] == 'midpoint'
                            and r['tier'] == row[0][1:]]
                assert len(selected) == 24
                check(row[column], 100 * statistics.mean(float(r['macro_f1']) for r in selected))
                column += 1
    for row in tables[2]:
        result = next(r for r in load(row[0], 'confirmation/tables/difficulty_contrasts.csv')
                      if r['source'] == SOURCES[row[1]] and r['unknown_mode'] == 'midpoint')
        check(row[2], 100 * float(result['mean_delta_auc']))
        for value, key in zip(re.findall(r'[−-]?\d+\.\d+', row[3]), ('ci95_low', 'ci95_high')):
            check(value, 100 * float(result[key]))
        assert row[4] == '24/24' and int(result['negative_runs']) == 24
        check(row[5], result['holm_six_p'], 5)
    for row in tables[3]:
        column = 3
        values = load(row[0], 'confirmation/tables/cumulative_metrics.csv')
        for source in SOURCES.values():
            selected = [r for r in values if r['source'] == source and r['tier'] == row[1][1:]]
            assert len(selected) == 24 and all(r['n'] == row[2] for r in selected)
            for key in ('macro_frequency_coverage', 'macro_f1'):
                check(row[column], 100 * statistics.mean(float(r[key]) for r in selected))
                column += 1
    for row in tables[4]:
        result = next(r for r in load(row[0], 'comparison/tables/comparisons.csv')
                      if r['source'] == SOURCES[row[1]] and r['baseline'] == 'random'
                      and r['metric'] == METRICS[row[2]])
        check(row[3], result['mean_delta_auc'], 4)
        for value, key in zip(re.findall(r'[−+-]?\d+\.\d+', row[4]), ('ci95_low', 'ci95_high')):
            check(value, result[key], 4)
        for column, key in ((5, 'holm_primary4_p'), (6, 'holm_all20_p')):
            check(row[column], result[key], 5)
    for row in tables[5]:
        result = next(r for r in load(row[0], 'confirmation/tables/matched_equal_error_comparisons.csv')
                      if r['source'] == SOURCES[row[1]] and r['metric'] == METRICS[row[2]])
        check(row[3], result['mean_delta_auc'], 4)
        check(row[4], result['holm_four_p'], 5)
    for row in tables[6]:
        column = 1
        for pool in POOLS:
            values = load(pool, 'confirmation/tables/model_robustness.csv')
            for source in SOURCES.values():
                result = next(r for r in values if r['source'] == source and r['model'] == MODELS[row[0]])
                check(row[column], 100 * float(result['high_risk_f1']))
                column += 1
    for row in tables[7]:
        data = json.loads((ROOT / 'artifacts' / POOLS[row[0]] / 'transfer/summary.json').read_text())
        summary = next(r for r in data['summary'] if r['source'] == SOURCES[row[1]])
        result = next(r for r in data['comparisons'] if r['source'] == SOURCES[row[1]])
        check(row[2], summary['frozen_state_models'], 0)
        check(row[3], summary['evaluable_state_models'], 0)
        check(row[4], 100 * float(result['mean_delta_auc']))
        check(row[5], result['holm_two_p'], 5)
    evidence = ROOT / 'artifacts/empirical_evidence_strengthening_20261005'
    cases = records(evidence / 'selected_state_cases.csv')
    for row in tables[8]:
        result = next(r for r in cases if r['study'] == POOLS['B']
                      and r['model'] == MODELS[row[0]] and r['class_name'] == row[1].split('，')[0])
        for column, key, scale in ((3, 'support', 1), (4, 'class_recall', 100),
                                   (5, 'state_recall', 100), (6, 'gap', 100)):
            check(row[column], scale * float(result[key]), 0 if column == 3 else 2)
    costs = records(evidence / 'cost_summary.csv')
    for row in tables[9]:
        result = next(r for r in costs if r['study'] == POOLS[row[0]] and r['backend'] == SOURCES[row[1]])
        for column, key in ((2, 'successful_targets'), (3, 'raw_response_files'), (4, 'raw_prompt_tokens'),
                            (5, 'raw_completion_tokens'), (6, 'successful_latency_median_seconds'),
                            (7, 'successful_latency_p95_seconds')):
            check(row[column], result[key], 0 if column < 6 else 2)
    links = re.findall(r'\]\((/[^)]+)\)', text)
    assert all(Path(path).is_file() for path in links)
    assert text.count('$$') % 2 == 0
    print(json.dumps(dict(status='pass', numeric_cells_checked=checked, result_tables_checked=9,
                          local_links_checked=len(links), figures=text.count('!['),
                          characters=len(text), manuscript=str(PAPER)), ensure_ascii=False))


if __name__ == '__main__':
    run()

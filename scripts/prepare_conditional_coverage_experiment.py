"""Freeze historical inputs and calibrate thresholds without reading predictions."""
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info

import conditional_coverage_common as cc


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'protocol.json').exists(), 'Use a new directory; do not overwrite a frozen run.'
    plan = json.loads(cc.PLAN.read_text())
    plan['status'] = 'frozen_before_main_selection_retrospective_not_preregistered'
    plan['scope']['new_conditional_algorithm_implemented'] = True
    plan['implementation_entrypoints_status'] = 'implemented_local_execution'
    plan['execution'] = dict(start_time=cc.old.now(), phase='retrospective',
        diagnostic_reference='historical_Objects365_reanalysis_not_fresh_confirmation',
        timing_order_seed=2026110501, threads=1,
        memory_metric='process_high_water_RSS_not_method_incremental_memory',
        quality_repeats_deterministic=1,
        timing_replays='three_total_runs_for_first_ten_main_seeds_and_deterministic_baselines',
        diagnostic_known_rule='any_declared_valid_state_in_same_namespace_field_and_not_target_state',
        data_roles='COCO_B_historical_discovery_Objects365_D_historical_reference',
        input_loading_cost='separately_timed_not_in_warm_algorithm_time',
        sensitivity_selection_seeds=plan['phases']['retrospective']['selection_seeds'],
        bootstrap_reference_role='historical_only_no_fresh_noninferiority_claim')
    cc.freeze(out / 'protocol.json', plan)
    scripts = [cc.ROOT / 'scripts' / name for name in [
        'conditional_coverage_common.py', 'check_conditional_coverage_experiment.py',
        'prepare_conditional_coverage_experiment.py', 'run_conditional_coverage_selection.py',
        'coverage_validation_common_20261018.py']]
    inputs = [cc.PLAN, cc.old.BASE/'manifest.csv', cc.old.SOURCE/'risk_scores_and_tiers.csv',
              cc.old.BASE/'selection.npz',
              cc.ROOT/'artifacts/sixclass_independent_confirmation_20261004/embeddings/vit_b16_candidates.npz']
    inputs += [cc.old.SOURCE/f'{c}_{suffix}' for c in cc.old.CLASSES
               for suffix in ['state_matrix.npz', 'state_dictionary.json']]
    cc.freeze(out / 'input_hashes.json', dict(time=cc.old.now(),
        files={str(p):cc.sha(p) for p in [*inputs, *scripts]}, classifier_predictions_read=False))
    cc.dump(out / 'environment.json', dict(time=cc.old.now(), python=sys.version,
        executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
        cpu_count=os.cpu_count(), cpu_info=next((line.strip() for line in open('/proc/cpuinfo')
          if line.startswith('model name')), ''), threadpools=threadpool_info(),
        command=sys.argv, git_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        git_working_tree=subprocess.check_output(['git','status','--short'], text=True)))
    start = time.perf_counter()
    contexts = cc.load_contexts()
    loading = time.perf_counter() - start
    thresholds, witnesses, quota_rows, witness_times, data_rows = {}, {}, [], [], []
    for c, ctx in contexts.items():
        # Prewarm common NumPy paths outside the timed calibration.
        cc.baseline(ctx, 200, 'G')
        start = time.perf_counter()
        witness, work = cc.baseline(ctx, 200, 'G')
        elapsed = time.perf_counter() - start
        reference = cc.old.choose(ctx, 200, 'expanded_greedy')
        assert np.array_equal(witness, reference)
        assert np.array_equal(witness, ctx['original']), f'Old greedy reproduction failed for {c}'
        cv = cc.coverage_rows(ctx, witness)['all']
        thresholds[c] = dict(occurrence=cv['occurrence']*.95,
                              support_completion=cv['support_completion']*.95)
        witnesses[c] = witness
        witness_times.append(dict(class_name=c, seconds=elapsed, **work,
                                  occurrence=cv['occurrence'], support_completion=cv['support_completion']))
        for cell, n in enumerate(cc.old.quotas(ctx, 200)):
            key = ctx['keys'][cell]
            quota_rows.append(dict(class_name=c,cell=cell,Q=key[0],midpoint=key[1]/1e8,
                                   uncertainty_width=key[2]/1e8,quota=int(n)))
        for i, row in enumerate(ctx['rows']):
            data_rows.append(dict(**row, state_count=int(ctx['matrix'][i].sum()),
                                  risk_cell=int(ctx['cell'][i]), Q=int(ctx['tiers'][i]),
                                  data_role='historical_selection_pool'))
        cc.log(out, 'calibration_class_complete', class_name=c,
               thresholds=thresholds[c], seconds=elapsed)
    cc.dump(out/'thresholds.json', thresholds)
    cc.write(out/'thresholds.csv', [dict(class_name=c, **v) for c,v in thresholds.items()])
    cc.write(out/'calibration_times.csv', witness_times)
    cc.write(out/'quotas.csv',quota_rows)
    cc.write(out/'manifest.csv',data_rows)
    np.savez_compressed(out/'threshold_witnesses.npz', **witnesses)
    with np.load(cc.old.BASE/'selection.npz', allow_pickle=False) as z:
        all_ids = z['sample_ids'].astype(str)
        selected = z['E4'][0]
        assert all_ids.shape == (6,400) and selected.shape == (6,200)
        fixed_ids = {str(all_ids[ci,i]) for ci in [0,1] for i in selected[ci]}
    rows = cc.old.manifest()
    background = [r for r in rows if r['class_name'] in ['dog','cat'] and r['sample_id'] in fixed_ids]
    assert all(sum(r['class_name']==c for r in background)==200 for c in ['dog','cat'])
    cc.write(out/'fixed_background.csv', background)
    cc.freeze(out/'threshold_freeze.json', dict(time=cc.old.now(),
        files={str(out/n):cc.sha(out/n) for n in ['protocol.json','thresholds.json','threshold_witnesses.npz',
              'fixed_background.csv','input_hashes.json']}, classifier_predictions_read=False))
    cc.dump(out/'preparation_cost.json', dict(context_loading_seconds=loading,
        calibration_seconds=sum(r['seconds'] for r in witness_times),
        first_task_cost_rule='context_loading_plus_calibration_plus_actual_method_selection',
        process_high_water_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        common_VLM_annotation_cost='historical_cached_not_remeasured_no_annotation_savings_claim'))
    cc.dump(out/'fresh_data_availability.json', dict(status='not_available_for_this_protocol',
        required='two_unused_image_disjoint_pools_C_select_and_C_ref_2400_targets_each',
        existing_reference='historical_Objects365_800_targets_already_analyzed',
        may_claim_fresh_confirmation=False, automatic_new_data_calls_made=0))
    cc.log(out,'preparation_complete', classifier_predictions_read=False,
           context_loading_seconds=loading, fixed_background_targets=len(background))


if __name__ == '__main__':
    args = cc.parsed_args(__doc__).parse_args()
    prepare(args.out)

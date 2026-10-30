"""New source destinations, preserving the frozen historical algorithms."""
from pathlib import Path
import inspect
import textwrap
from dual_vlm_common import CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now
from audit_objects365_confirmation_source import BASE, ROOT, note

PROTOCOL = ROOT/'docs/objects365_independent_confirmation_protocol_20261005.md'
PROJECT = Path('/home/hcf/project/QE')
PYTHON = PROJECT/'.venv/bin/python'
WEIGHTS = ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'
A = ROOT/'artifacts/sixclass_dual_vlm_replaced_20261004'
OLD_B = ROOT/'artifacts/sixclass_independent_confirmation_20261004'
LABELS = dict(zip(CLASSES,range(6)))
STAGES = ('prepare','annotate','freeze_selection','support','infer','confirm','compare','transfer','report')


def activate():
    import independent_confirmation_common as common
    common.BASE=BASE;common.PROTOCOL=PROTOCOL;common.note=note
    import analyze_dual_vlm_study as a
    a.STUDY=BASE
    a.job_path=lambda backend,domain:BASE/'annotations'/backend/domain
    a.schema_path=lambda domain:BASE/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')
    import freeze_independent_sixclass_selection as selection
    selection.BASE=BASE;selection.PROTOCOL=PROTOCOL
    import analyze_independent_sixclass as analysis
    analysis.BASE=BASE;analysis.PROTOCOL=PROTOCOL;analysis.note=note
    # Change provenance labels only; sizes, scoring, support thresholds and tests stay intact.
    source=textwrap.dedent(inspect.getsource(analysis.confirm))
    before='new samples; same COCO dataset source'
    assert before in source
    source=source.replace(before,'unused Objects365 v2 train-release targets; separate dataset source, image deduplicated')
    exec(compile(source,str(Path(__file__)) + ':confirm_provenance','exec'),analysis.__dict__)
    return common,a,selection,analysis


def code_paths():
    names=('audit_objects365_confirmation_source.py','objects365_confirmation_common.py',
           'prepare_objects365_confirmation.py','run_objects365_confirmation.py','analyze_objects365_confirmation.py',
           'automatic_strengthening_common.py','prepare_automatic_strengthening.py','run_automatic_strengthening.py',
           'independent_confirmation_common.py','annotate_dual_vlm.py','annotate_multiclass_online_sensenova.py',
           'run_dog_local_pipeline.py','analyze_dual_vlm_study.py','analyze_sixclass_difficulty.py',
           'freeze_independent_sixclass_selection.py','analyze_independent_sixclass.py',
           'run_multiclass_object_selection_benchmark.py','run_class_stratified_error_controls.py',
           'run_coverage_incremental_value.py','sixclass_car_support.py','prepare_multiclass_object_candidate_pool.py',
           'audit_multiclass_near_duplicates.py')
    return [PROTOCOL,*[ROOT/'scripts'/name for name in names]]
